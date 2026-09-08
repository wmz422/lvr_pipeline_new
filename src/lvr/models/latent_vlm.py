"""Lightning model combining Qwen3-VL, LAM and a latent projector. Stage behavior is selected by configuration."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import nn

from lvr.checkpoint.io import load_init_checkpoint
from lvr.models import builder, freezing, generation, injection, losses
from lvr.tokens import LATENT_PAD_TOKEN


class LatentVLM(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        latent_projector_learning_rate: float | None = None,
        qwen_learning_rate: float | None = None,
        qwen_model_name_or_path: str | None = None,
        qwen_torch_dtype: str = "bfloat16",
        add_latent_special_tokens: bool = True,
        latent_pad_token: str = LATENT_PAD_TOKEN,
        lam_checkpoint_path: str | None = None,
        lam_vision_model_path: str | None = None,
        lam_model_dim: int = 1024,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 16,
        lam_dec_blocks: int = 16,
        lam_num_heads: int = 16,
        lam_num_latent: int = 4,
        latent_projector_type: str = "linear",
        latent_projector_mlp_hidden_dim: int | None = None,
        latent_head: bool = False,
        train_qwen_lm: bool = False,
        train_qwen_lm_head: bool = False,
        train_latent_projector: bool = False,
        train_latent_head: bool = False,
        train_lam: bool = False,
        init_checkpoint_path: str | None = None,
        qwen_gradient_checkpointing: bool = False,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
        save_trainable_only: bool = False,
        loss_lambda: float = 0.1,
        generation_mode: str = "sft",  # "sft"=自回归生成 latent | "align"=latent 已知注入(generate_align)
        generation_max_new_tokens: int = 64,
        generation_output_path: str = "runs/generation/test_predictions.jsonl",
        generation_output_append: bool = False,
        initialize_from_config: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # 路径不留代码内默认值（机器相关，单一真源在 configs/base.yaml model 段）。
        missing = [
            name for name, value in (
                ("qwen_model_name_or_path", qwen_model_name_or_path),
                ("lam_checkpoint_path", lam_checkpoint_path or (initialize_from_config and init_checkpoint_path)),
                ("lam_vision_model_path", lam_vision_model_path),
            ) if not value
        ]
        if missing:
            raise ValueError(f"LatentVLM 缺少必填参数 {missing}（应由 configs/base.yaml model 段提供）。")

        self.qwen = None
        self.lam = None
        self.latent_projector = None
        self.latent_head = None  # 视 train_latent_head 决定（暂未实现）
        self.latent_pad_token_id: int | None = None
        self.tokenizer = None

        self.qwen, self.tokenizer, self.latent_pad_token_id = builder.build_qwen(
            model_name_or_path=qwen_model_name_or_path,
            qwen_torch_dtype=qwen_torch_dtype,
            add_latent_special_tokens=add_latent_special_tokens,
            latent_pad_token=latent_pad_token,
            gradient_checkpointing=qwen_gradient_checkpointing,
            initialize_from_config=initialize_from_config,
        )
        self.latent_projector = builder.build_latent_projector(
            latent_projector_type,
            lam_latent_dim,
            builder.qwen_hidden_size(self.qwen),
            latent_projector_mlp_hidden_dim,
        )
        self.lam = builder.build_lam(
            checkpoint_path=lam_checkpoint_path,
            vision_model_path=lam_vision_model_path,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            num_latent=lam_num_latent,
            initialize_from_config=initialize_from_config,
        )
        # TODO(M-latent_head): self.latent_head = ...
        if init_checkpoint_path is not None:
            load_init_checkpoint(self, init_checkpoint_path)
        self._align_dtype()
        freezing.set_trainable(
            self,
            train_qwen_lm=train_qwen_lm,
            train_qwen_lm_head=train_qwen_lm_head,
            train_latent_projector=train_latent_projector,
            train_latent_head=train_latent_head,
            train_lam=train_lam,
        )

    def train(self, mode: bool = True) -> "LatentVLM":
        """Switch modes while keeping a frozen LAM deterministic.

        Lightning/DeepSpeed calls ``train()`` recursively on the whole module.
        Without this guard, a frozen VAE-style LAM re-enters training mode and
        samples ``z_rep`` even though none of its parameters are trainable.
        """
        super().train(mode)
        if self.lam is not None and not bool(self.hparams.train_lam):
            self.lam.eval()
        return self

    # ---------------- forward / loss ----------------

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # latent 生成 → projector 映射 → 注入 latent 占位位置 → Qwen forward。
        if self.qwen is None or self.lam is None or self.latent_projector is None:
            raise RuntimeError("Key modules missing to forward.")

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        lam_inputs = batch.get("lam_inputs")
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        mm_token_type_ids = batch.get("mm_token_type_ids")

        latent = injection.compute_latents(self.lam, lam_inputs, self.device, self._compute_dtype())
        mapped_latent = self.latent_projector(latent)

        latent_mask = input_ids.eq(self.latent_pad_token_id)
        inputs_embeds = injection.inject_latents(self.qwen, input_ids, mapped_latent, latent_mask)
        labels = injection.mask_latent_labels(labels, latent_mask)

        outputs = self._qwen_forward_from_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            latent_mask=latent_mask,
        )
        return {
            "loss": outputs["loss"],
            "ce_loss": outputs["ce_loss"],
            "mse_loss": outputs["mse_loss"],
            "logits": outputs["logits"],
        }

    def _qwen_forward_from_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        latent_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        model = self.qwen.model
        inputs_embeds, image_mask, deepstack_image_embeds = injection.inject_vision(
            model, input_ids, inputs_embeds, pixel_values, image_grid_thw,
        )
        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        outputs = model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_image_embeds,
        )
        hidden_states = outputs[0]

        ce_loss = torch.tensor(0.0, device=self.device)
        if labels is not None:
            # 显存高效 CE：只在被监督位跑 lm_head（不在全序列×词表 materialize logits）。
            # 数值等价于旧 text_ce_loss(全logits)。logits 在训练/验证不被消费（_step 只读 loss），
            # 故不再返回完整 logits（generate 走独立解码路径）。
            ce_loss = losses.text_ce_loss_from_hidden(self.qwen, hidden_states, labels)
        mse_loss = torch.tensor(0.0, device=self.device)
        if latent_mask is not None:
            mse_loss = losses.latent_mse_loss(latent_mask, inputs_embeds, hidden_states)
        loss = ce_loss + self.hparams.loss_lambda * mse_loss
        return {"loss": loss, "logits": None, "ce_loss": ce_loss, "mse_loss": mse_loss}

    # ---------------- train / val steps ----------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, prefix="val")

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs["loss"]
        batch_size = self._batch_size(batch)
        log_kwargs = dict(
            prog_bar=True, sync_dist=True, batch_size=batch_size, on_step=True, on_epoch=True,
        )
        self.log(f"{prefix}/loss", loss, **log_kwargs)
        self.log(f"{prefix}/ce_loss", outputs["ce_loss"], **log_kwargs)
        self.log(f"{prefix}/mse_loss", outputs["mse_loss"], **log_kwargs)
        return loss

    def _batch_size(self, batch: dict[str, Any]) -> int:
        if "input_ids" in batch:
            return int(batch["input_ids"].size(0))
        if "answer" in batch:
            return len(batch["answer"])
        return 1

    # ---------------- generation / test ----------------

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Any],
        max_new_tokens: int = 256,
        latent_len: int | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool = True,
        force_latent_end: bool = False,
    ) -> torch.Tensor:
        return generation.generate(
            self,
            batch,
            max_new_tokens=max_new_tokens,
            latent_len=latent_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            force_latent_end=force_latent_end,
        )

    @torch.no_grad()
    def generate_align(
        self,
        batch: dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool = False,
        latent_condition: str = "correct",
    ) -> torch.Tensor:
        # alignment 推理模式：latent 作为已知在 prefill 注入，纯文本自回归生成（需 lam_inputs）。
        return generation.generate_align(
            self,
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            latent_condition=latent_condition,
        )

    def on_test_start(self) -> None:
        # trainer 在所有 test_step 之前自动调用一次：准备输出文件。
        if self.global_rank == 0:
            output_path = Path(self.hparams.generation_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.hparams.generation_output_append:
                output_path.write_text("", encoding="utf-8")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor | None:
        # 1. 算 loss（如果 batch 有 labels 和 lam_inputs）
        loss = None
        if batch.get("labels") is not None and batch.get("lam_inputs") is not None:
            loss = self._step(batch, prefix="test")
        # 2. 生成并写 case（generation_mode 选 SFT 自回归生成 latent / align latent 已知注入）
        if self.qwen is not None:
            if self.hparams.generation_mode == "align":
                generated_ids = self.generate_align(
                    batch,
                    max_new_tokens=self.hparams.generation_max_new_tokens,
                    do_sample=False,
                )
            else:
                generated_ids = self.generate(
                    batch,
                    max_new_tokens=self.hparams.generation_max_new_tokens,
                    latent_len=self.hparams.lam_num_latent,
                )
            self._write_generation_case(batch, batch_idx, generated_ids, loss)
        return loss

    @rank_zero_only
    def _write_generation_case(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        generated_ids: torch.Tensor,
        loss: torch.Tensor | None,
    ) -> None:
        output_path = Path(self.hparams.generation_output_path)
        # skip_special_tokens=False，方便看是否学到生成 <abs_vis_token>
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        questions = batch.get("questions", [""] * len(generated_texts))
        # align 阶段目标是 observation（AlignmentCollator 不产 answer）；SFT 用 answer。
        answers = batch.get("answer") or batch.get("observations") or [""] * len(generated_texts)
        prompt_texts = batch.get("prompt_texts", [""] * len(generated_texts))
        loss_val = loss.item() if loss is not None else None

        with output_path.open("a", encoding="utf-8") as f:
            for i, gen_text in enumerate(generated_texts):
                record = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "question": questions[i] if i < len(questions) else "",
                    "prompt": prompt_texts[i] if i < len(prompt_texts) else "",
                    "generated": gen_text,
                    "ground_truth": answers[i] if i < len(answers) else "",
                    "loss": loss_val,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------------- optimizer / dtype / checkpoint ----------------

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create module-specific optimizer groups without duplicating tied weights."""
        assigned_parameter_ids: set[int] = set()
        parameter_groups: list[dict[str, Any]] = []

        def add_group(module: nn.Module | None, learning_rate: float) -> None:
            if module is None:
                return
            params = [
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad and id(parameter) not in assigned_parameter_ids
            ]
            if not params:
                return
            assigned_parameter_ids.update(id(parameter) for parameter in params)
            parameter_groups.append({"params": params, "lr": learning_rate})

        # Fresh projector weights need larger updates than pretrained Qwen.
        # Unspecified per-module values preserve the previous single-LR behavior.
        projector_lr = self.hparams.latent_projector_learning_rate or self.hparams.learning_rate
        qwen_lr = self.hparams.qwen_learning_rate or self.hparams.learning_rate
        add_group(self.latent_projector, projector_lr)
        if self.qwen is not None:
            add_group(self.qwen.model.language_model, qwen_lr)
            add_group(self.qwen.lm_head, qwen_lr)

        # Future trainable modules (LAM / latent head) use the base rate.
        add_group(self, self.hparams.learning_rate)
        if not parameter_groups:
            raise RuntimeError("No trainable parameters are enabled.")
        optimizer_name = self.hparams.optimizer_name.lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameter_groups, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
            )
        if optimizer_name == "deepspeed_cpu_adam":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            return DeepSpeedCPUAdam(
                parameter_groups, lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay, adamw_mode=True,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameter_groups, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer_name: {self.hparams.optimizer_name}")

    def _compute_dtype(self) -> torch.dtype:
        return builder.compute_dtype(self.hparams.qwen_torch_dtype)

    def _align_dtype(self) -> None:
        resolved = self._compute_dtype()
        self.lam.to(resolved)
        self.latent_projector.to(resolved)
        if self.latent_head is not None:
            self.latent_head.to(resolved)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if not self.hparams.save_trainable_only:
            return
        # 只保留可训练部分，避免每次保存 5B+ frozen 权重。
        trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            name: tensor
            for name, tensor in checkpoint["state_dict"].items()
            if name in trainable_names
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        if self.hparams.get("save_trainable_only", False):
            strict = False
        elif strict:
            # DeepSpeed ZeRO-3 with exclude_frozen_parameters=True omits frozen params
            # from checkpoints. Resume should still be strict for trainable weights.
            trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
            missing_trainable = [
                name for name in trainable_names
                if name in self.state_dict() and name not in state_dict
            ]
            if not missing_trainable:
                missing_keys = set(self.state_dict()) - set(state_dict)
                if missing_keys:
                    strict = False
        return super().load_state_dict(state_dict, strict=strict)
