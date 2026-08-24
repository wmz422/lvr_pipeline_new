"""PlainVLM：普通 Qwen3-VL SFT LightningModule。

用于 baseline + 本仓库数据的普通图文 SFT。它刻意不包含 LVR 组件：
不建 LAM、不建 latent_projector、不注册 latent special tokens、不计算 MSE。
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch

from lvr.models.builder import compute_dtype


class PlainVLM(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-6,
        qwen_model_name_or_path: str | None = None,
        qwen_torch_dtype: str = "bfloat16",
        train_qwen_lm: bool = True,
        train_qwen_lm_head: bool = True,
        qwen_gradient_checkpointing: bool = False,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if not qwen_model_name_or_path:
            raise ValueError("PlainVLM 缺少 qwen_model_name_or_path（应由 configs/base_plain.yaml model 段提供）。")

        self.qwen = self._build_qwen(
            qwen_model_name_or_path=qwen_model_name_or_path,
            qwen_torch_dtype=qwen_torch_dtype,
            gradient_checkpointing=qwen_gradient_checkpointing,
        )
        self._set_trainable(
            train_qwen_lm=train_qwen_lm,
            train_qwen_lm_head=train_qwen_lm_head,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "labels": batch["labels"],
            "pixel_values": batch.get("pixel_values"),
            "image_grid_thw": batch.get("image_grid_thw"),
            "mm_token_type_ids": batch.get("mm_token_type_ids"),
        }
        inputs = {k: v for k, v in inputs.items() if v is not None}
        outputs = self.qwen(**inputs)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs["loss"]
        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=self._batch_size(batch),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def _batch_size(self, batch: dict[str, Any]) -> int:
        if "input_ids" in batch:
            return int(batch["input_ids"].size(0))
        if "answer" in batch:
            return len(batch["answer"])
        return 1

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters. Check train_qwen_lm/train_qwen_lm_head.")
        name = self.hparams.optimizer_name.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(params, lr=self.hparams.learning_rate)
        raise ValueError(f"Unsupported optimizer_name={self.hparams.optimizer_name!r}")

    def _build_qwen(
        self,
        *,
        qwen_model_name_or_path: str,
        qwen_torch_dtype: str,
        gradient_checkpointing: bool,
    ):
        from transformers import Qwen3VLForConditionalGeneration

        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=compute_dtype(qwen_torch_dtype),
        )
        if gradient_checkpointing:
            if hasattr(qwen, "gradient_checkpointing_enable"):
                qwen.gradient_checkpointing_enable()
            elif hasattr(qwen.model.language_model, "gradient_checkpointing_enable"):
                qwen.model.language_model.gradient_checkpointing_enable()
            if hasattr(qwen.config, "use_cache"):
                qwen.config.use_cache = False
            if hasattr(qwen.config, "text_config") and hasattr(qwen.config.text_config, "use_cache"):
                qwen.config.text_config.use_cache = False
        return qwen

    def _set_trainable(self, *, train_qwen_lm: bool, train_qwen_lm_head: bool) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        if train_qwen_lm:
            self.qwen.model.language_model.train()
            for param in self.qwen.model.language_model.parameters():
                param.requires_grad = True
        if train_qwen_lm_head:
            self.qwen.lm_head.train()
            for param in self.qwen.lm_head.parameters():
                param.requires_grad = True
