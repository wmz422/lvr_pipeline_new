"""Qwen QA with a visual bottleneck and optional image-free region description supervision."""

from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
from torch import nn

from lvr.models.builder import build_latent_projector, load_qwen_vl_backbone, qwen_hidden_size
from lvr.models.chunked_loss import ChunkedLMHead
from lvr.models.injection import inject_latents, inject_vision
from lvr.tokens import IGNORE_INDEX
from lvr.training.diagnostics import trace_event, trace_span, traced


def _read_checkpoint(path):
    from torch.utils.serialization import config

    # Avoid one shared-storage seek per tensor when reading complete checkpoints.
    with config.patch("load.calculate_storage_offsets", True):
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def load_checkpoint_weights(model, state_dict, *, assign=True):
    """Preserve parameter aliases when assigning checkpoint tensors and dtypes.

    assign=True creates separate Parameters for tied state-dict names. They may
    initially share CPU storage, but a device/dtype move then splits the weights
    and an optimizer treats them independently. Restore constructor-defined ties
    before moving the model, and reject checkpoints whose tied values diverged.
    """
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    aliases = [names for names in groups.values() if len(names) > 1]
    for names in aliases:
        if any(name not in state_dict for name in names):
            continue  # strict loading reports missing names below.
        reference = state_dict[names[0]]
        for name in names[1:]:
            value = state_dict[name]
            if (reference.shape != value.shape or reference.dtype != value.dtype
                    or (reference.data_ptr() != value.data_ptr() and not torch.equal(reference, value))):
                raise ValueError(f"Checkpoint contains inconsistent tied weights: {names[0]} and {name}")
    model.load_state_dict(state_dict, strict=True, assign=assign)
    for names in aliases:
        shared = model.get_parameter(names[0])
        for name in names[1:]:
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent).register_parameter(leaf, shared)


class VisualBottleneck(pl.LightningModule):
    def __init__(self, encoder_model_name_or_path: str, qa_model_name_or_path: str,
                 num_latents: int = 8, latent_dim: int = 512, gaussian: bool = True,
                 learning_rate: float = 1e-5, head_learning_rate: float = 1e-4,
                 kl_weight: float = 1e-4, kl_warmup_ratio: float = 0.1,
                 lr_warmup_ratio: float = 0.03, weight_decay: float = 0.01,
                 gradient_checkpointing: bool = True, seed: int = 42, train_qa: bool = False,
                 encoder_uses_question: bool = False, encoder_type: str = "qwen",
                 encoder_includes_original: bool = True, qformer_hidden_size: int = 768,
                 qformer_num_layers: int = 6, qformer_num_heads: int = 12,
                 qformer_cross_attention_frequency: int = 2, qformer_use_geometry: bool = False,
                 description_weight: float = 0.0, qa_loss_weight: float = 1.0,
                 description_image_mode: str = "none", lr_schedule_steps: int | None = None,
                 reconstruction_weight: float = 0.0, reconstruction_hidden_size: int = 512,
                 reconstruction_num_layers: int = 2, reconstruction_num_heads: int = 8,
                 reconstruction_normalize: bool = False):
        super().__init__()
        self.save_hyperparameters()
        if lr_schedule_steps is not None and lr_schedule_steps < 1:
            raise ValueError("lr_schedule_steps must be positive when provided")
        if not math.isfinite(description_weight) or description_weight < 0 or (description_weight and not num_latents):
            raise ValueError("Description weight must be finite/nonnegative and requires latent slots")
        if not math.isfinite(qa_loss_weight) or qa_loss_weight < 0:
            raise ValueError("QA loss weight must be finite and nonnegative")
        if (not math.isfinite(reconstruction_weight) or reconstruction_weight < 0
                or (reconstruction_weight and not num_latents)):
            raise ValueError("Reconstruction weight must be finite/nonnegative and requires latent slots")
        if not qa_loss_weight and not description_weight:
            raise ValueError("At least one supervised text objective must be enabled")
        if description_image_mode not in ("none", "original"):
            raise ValueError("Description image mode must be 'none' or 'original'")
        if num_latents < 0 or latent_dim < 1 or kl_weight < 0:
            raise ValueError("Latent count/KL must be nonnegative and latent dimension positive")
        if not num_latents and (gaussian or not train_qa):
            raise ValueError("Zero-latent QA training requires gaussian=False and train_qa=True")
        if encoder_uses_question and not num_latents:
            raise ValueError("Question-conditioned encoding requires latent slots")
        if encoder_type not in ("qwen", "qformer") or (encoder_type == "qformer" and not num_latents):
            raise ValueError("Use a qwen encoder, or a qformer encoder with positive latent count")
        if qformer_use_geometry and encoder_type != "qformer":
            raise ValueError("Explicit geometry encoding requires a Q-Former")
        self.encoder = load_qwen_vl_backbone(
            encoder_model_name_or_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        ) if num_latents and encoder_type == "qwen" else None
        self.qa = load_qwen_vl_backbone(
            qa_model_name_or_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        self.qa.lm_head = ChunkedLMHead.from_linear(self.qa.lm_head)
        self.qa.requires_grad_(False)
        if train_qa:
            self.qa.model.language_model.requires_grad_(True)
            self.qa.lm_head.requires_grad_(True)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
            self.encoder.model.language_model.requires_grad_(True)
            self.encoder.model.language_model.embed_tokens.requires_grad_(False)
        if encoder_type == "qformer":
            from lvr.models.qformer import QueryFormerEncoder

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.encoder = QueryFormerEncoder(
                    qwen_hidden_size(self.qa), qformer_hidden_size, qformer_num_layers, qformer_num_heads,
                    qformer_cross_attention_frequency, encoder_uses_question, qformer_use_geometry,
                    gradient_checkpointing,
                )
        # DDP + BF16 autocast, with FP32 master parameters and AdamW states.
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.queries = self.mu_head = self.projector = self.logvar_head = None
        if self.encoder is not None:
            hidden_dim = qformer_hidden_size if encoder_type == "qformer" else qwen_hidden_size(self.encoder)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.queries = nn.Parameter(torch.empty(num_latents, hidden_dim))
                nn.init.normal_(self.queries, std=0.02)
                self.mu_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
                self.projector = build_latent_projector("mlp", latent_dim, qwen_hidden_size(self.qa), 1024)
                self.logvar_head = nn.Linear(hidden_dim, latent_dim) if gaussian else None
                if self.logvar_head is not None:
                    nn.init.zeros_(self.logvar_head.weight)
                    nn.init.constant_(self.logvar_head.bias, -4.0)
        self.reconstruction_decoder = None
        if reconstruction_weight:
            from lvr.models.feature_decoder import AuxiliaryFeatureDecoder

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.reconstruction_decoder = AuxiliaryFeatureDecoder(
                    qwen_hidden_size(self.qa), latent_dim, reconstruction_hidden_size,
                    reconstruction_num_layers, reconstruction_num_heads,
                )
        if gradient_checkpointing:
            for backbone in (self.encoder if encoder_type == "qwen" else None, self.qa):
                if backbone is not None:
                    backbone.model.language_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False},
                    )
        self.train()

    def initialize_from_checkpoint(self, path):
        """Copy complete model weights while preserving this phase's trainability and dtypes."""
        import inspect

        checkpoint = _read_checkpoint(path)
        defaults = inspect.signature(type(self)).parameters
        for key in ("num_latents", "latent_dim", "gaussian", "encoder_type",
                    "encoder_uses_question", "encoder_includes_original", "qformer_use_geometry"):
            saved = checkpoint["hyper_parameters"].get(key, defaults[key].default)
            if saved != self.hparams[key]:
                raise ValueError(f"Initial checkpoint {key}={saved!r}, configured {self.hparams[key]!r}")
        # assign=False keeps newly trainable B parameters in FP32, even when saved frozen in BF16.
        state_dict = checkpoint["state_dict"]
        initialized_modules = []
        if self.reconstruction_decoder is not None and not checkpoint["hyper_parameters"].get("reconstruction_weight", 0):
            # Adding this optional head to an existing teacher initializes only
            # that head. Missing or unexpected backbone tensors remain errors.
            if any(key.startswith("reconstruction_decoder.") for key in state_dict):
                raise ValueError("Checkpoint has reconstruction weights but declares the objective disabled")
            state_dict = dict(state_dict)
            state_dict.update({f"reconstruction_decoder.{key}": value
                               for key, value in self.reconstruction_decoder.state_dict().items()})
            initialized_modules.append("reconstruction_decoder")
        load_checkpoint_weights(self, state_dict, assign=False)
        self.initial_checkpoint = {
            "path": str(path), "source_global_step": checkpoint["global_step"],
            "source_epoch": checkpoint["epoch"], "model_tensors": len(checkpoint["state_dict"]),
            "optimizer_restored": False, "scheduler_restored": False,
            "initialized_modules": initialized_modules,
        }
        return self.initial_checkpoint

    @classmethod
    def from_pretrained(cls, path, device):
        """Evaluate untouched Qwen weights through the same zero-slot QA path."""
        model = cls(str(path), str(path), num_latents=0, gaussian=False,
                    train_qa=True, gradient_checkpointing=False, kl_weight=0.0)
        model.checkpoint_step = 0
        # The training constructor promotes trainable parameters to FP32. Restore
        # their inference dtype without rounding native FP32 MRoPE frequency
        # buffers, which a whole-module .to(dtype=...) would also convert.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
        return model.requires_grad_(False).eval().to(device=device)

    @classmethod
    def from_checkpoint(cls, path, device, *, base_model_path=None):
        """Load a complete stage-one checkpoint for generation, preserving saved dtypes."""
        import inspect

        checkpoint = _read_checkpoint(path)
        parameters = inspect.signature(cls).parameters
        kwargs = {key: value for key, value in checkpoint["hyper_parameters"].items() if key in parameters}
        if base_model_path is not None:
            kwargs.update(encoder_model_name_or_path=str(base_model_path), qa_model_name_or_path=str(base_model_path))
        model = cls(**{**kwargs, "gradient_checkpointing": False})
        load_checkpoint_weights(model, checkpoint["state_dict"])
        model.checkpoint_step = checkpoint["global_step"]
        return model.requires_grad_(False).eval().to(device)

    def train(self, mode=True):
        super().train(mode)
        # Language blocks use training mode for checkpointing; Qwen has zero dropout.
        for backbone in (self.encoder if self.hparams.encoder_type == "qwen" else None, self.qa):
            if backbone is not None:
                backbone.model.visual.eval()
        return self

    @staticmethod
    @traced("hidden")
    def _hidden(backbone, batch, slots=None):
        trace_event("hidden.embeddings.begin")
        if slots is None:
            if batch["slot_mask"].any():
                raise ValueError("QA without latents requires zero latent slots")
            embeddings = backbone.model.get_input_embeddings()(batch["input_ids"])
        else:
            embeddings = inject_latents(backbone, batch["input_ids"], slots, batch["slot_mask"])
        trace_event("hidden.embeddings.end")
        trace_event("hidden.vision.begin", has_images="pixel_values" in batch)
        embeddings, image_mask, deepstack = inject_vision(
            backbone.model, batch["input_ids"], embeddings, batch.get("pixel_values"), batch.get("image_grid_thw"),
        )
        trace_event("hidden.vision.end")
        trace_event("hidden.positions.begin")
        position_ids, _ = backbone.model.get_rope_index(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            mm_token_type_ids=batch["mm_token_type_ids"], image_grid_thw=batch.get("image_grid_thw"),
        )
        trace_event("hidden.positions.end")
        trace_event("hidden.language.begin")
        # Qwen2.5-VL has a single merged visual stream; Qwen3-VL also injects
        # DeepStack features at intermediate language layers.
        visual_kwargs = ({"visual_pos_masks": image_mask, "deepstack_visual_embeds": deepstack}
                         if deepstack is not None else {})
        hidden = backbone.model.language_model(
            inputs_embeds=embeddings, attention_mask=batch["attention_mask"], position_ids=position_ids,
            use_cache=False, **visual_kwargs,
        ).last_hidden_state
        trace_event("hidden.language.end")
        return hidden

    @traced("encoder")
    def encode(self, batch):
        if self.encoder is None:
            raise ValueError("This QA checkpoint has no latent encoder")
        if self.hparams.encoder_type == "qformer":
            # Reuse B's frozen native vision tower/merger; no second Qwen language model.
            with torch.no_grad():
                features = self.qa.model.get_image_features(
                    batch["pixel_values"], batch["image_grid_thw"], return_dict=True,
                ).pooler_output
                question = (self.qa.model.get_input_embeddings()(batch["question_input_ids"])
                            if self.hparams.encoder_uses_question else None)
            hidden = self.encoder(features, batch, self.queries,
                                  self.qa.model.visual.spatial_merge_size, question)
        else:
            size = batch["input_ids"].shape[0]
            hidden = self._hidden(self.encoder, batch, self.queries.unsqueeze(0).expand(size, -1, -1))
            hidden = hidden[batch["slot_mask"]].reshape(size, self.hparams.num_latents, -1)
        mu = self.mu_head(hidden).float()
        logvar = self.logvar_head(hidden).float().clamp(-12.0, 8.0) if self.logvar_head is not None else None
        return mu, logvar

    @traced("latent_projection")
    def project_latents(self, latent):
        # Latents stay FP32 for alignment/KL, but DeepSpeed BF16 weights do not
        # run under DDP's autocast context. This cast preserves the gradient path
        # through a frozen projector as well as a trainable one. Reading dtype
        # also works for ZeRO-3's partitioned parameter placeholders.
        return self.projector(latent.to(dtype=next(self.projector.parameters()).dtype))

    @traced("answer_loss")
    def answer_loss(self, batch, latent=None):
        hidden = self._hidden(self.qa, batch, self.project_latents(latent) if latent is not None else None)
        labels = batch["labels"][:, 1:]
        supervised = labels != IGNORE_INDEX
        targets = labels[supervised]
        if not targets.numel():
            raise ValueError("The text branch contains no supervised target tokens")
        # Avoid materializing vocabulary logits for image/prompt/latent positions.
        answer_hidden = hidden[:, :-1][supervised]
        with trace_span("answer_head"):
            return self.qa.lm_head(answer_hidden, targets)

    @torch.inference_mode()
    def generate_answers(self, batch, latent=None, *, max_new_tokens=256,
                         pad_token_id=None, eos_token_id=None):
        """Greedy QA generation with native Qwen vision, MRoPE and KV caching.

        The batch must contain only the prompt and latent slots. Passing None
        removes the slots entirely and evaluates the unchanged QA backbone.
        """
        from torch.nn.utils.rnn import pad_sequence
        from transformers import GenerationConfig

        if "labels" in batch:
            raise ValueError("Generation requires a prompt-only batch without answer labels")
        if self.training:
            raise ValueError("Call eval() before generation")
        pad_token_id = self.qa.generation_config.pad_token_id if pad_token_id is None else pad_token_id
        eos_token_id = self.qa.generation_config.eos_token_id if eos_token_id is None else eos_token_id
        keep = batch["attention_mask"].bool()
        if latent is None:
            keep = keep & ~batch["slot_mask"]
        prompt = {}
        for key in ("input_ids", "mm_token_type_ids", "slot_mask", "attention_mask"):
            prompt[key] = pad_sequence(
                [row[mask] for row, mask in zip(batch[key], keep)], batch_first=True,
                padding_value=pad_token_id if key == "input_ids" else 0, padding_side="left",
            )
        slots = prompt.pop("slot_mask")
        if latent is not None:
            prompt["inputs_embeds"] = inject_latents(
                self.qa, prompt["input_ids"], self.project_latents(latent), slots,
            )
        prompt.update({key: batch[key] for key in ("pixel_values", "image_grid_thw") if key in batch})
        config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, use_cache=True,
        )
        output = self.qa.generate(**prompt, generation_config=config)
        return output[:, prompt["input_ids"].shape[1]:]

    def reconstruction_losses(self, batch, latent):
        """Frozen image features supervise the decoder and differentiable teacher latent."""
        from lvr.models.feature_decoder import feature_reconstruction_loss

        if self.reconstruction_decoder is None:
            raise ValueError("Enable reconstruction_weight to use the auxiliary decoder")
        if "reconstruction" not in batch:
            raise ValueError("Feature reconstruction requires data.include_reconstruction=True")
        layout = batch["reconstruction"]
        indices = layout["target_indices"]
        with torch.no_grad():
            # Use one fixed feature space for both original inputs and targets.
            original = self.qa.model.get_image_features(
                batch["qa"]["pixel_values"], batch["qa"]["image_grid_thw"], return_dict=True,
            ).pooler_output
            encoded = self.qa.model.get_image_features(
                batch["encoder"]["pixel_values"], batch["encoder"]["image_grid_thw"], return_dict=True,
            ).pooler_output
        if indices.ndim != 1 or indices.numel() == 0 or indices.min() < 0 or indices.max() >= len(encoded):
            raise ValueError("Invalid auxiliary indices in encoder image sequence")
        targets = [encoded[index] for index in indices.tolist()]
        prediction, valid = self.reconstruction_decoder(
            latent, original, batch["qa"]["image_grid_thw"], layout["original_counts"],
            batch["encoder"]["image_grid_thw"][indices], layout["target_sample_ids"],
            layout["target_image_ids"], self.qa.model.visual.spatial_merge_size,
        )
        return feature_reconstruction_loss(
            prediction, targets, valid, layout["target_sample_ids"], len(latent),
            normalize=self.hparams.reconstruction_normalize,
        )

    def forward(self, batch, sample=None):
        if self.encoder is None:
            ce = self.answer_loss(batch["qa"]) if self.hparams.qa_loss_weight else next(self.parameters()).new_zeros(())
            return {"ce": ce, "kl": ce.new_zeros(()), "mu": None, "logvar": None}
        mu, logvar = self.encode(batch["encoder"])
        use_sample = self.hparams.gaussian if sample is None else sample and self.hparams.gaussian
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if use_sample else mu
        ce = self.answer_loss(batch["qa"], latent) if self.hparams.qa_loss_weight else mu.new_zeros(())
        description_ce = mu.new_zeros(())
        if self.hparams.description_weight:
            description_batch = batch["description"]
            has_description_images = "pixel_values" in description_batch or "image_grid_thw" in description_batch
            if self.hparams.description_image_mode == "none" and (
                has_description_images or description_batch["mm_token_type_ids"].any()
            ):
                raise ValueError("Image-free description supervision must not receive images or visual tokens")
            if self.hparams.description_image_mode == "original" and not has_description_images:
                raise ValueError("Original-image description mode requires image inputs")
            description_ce = self.answer_loss(description_batch, latent)
        kl = (0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
              if logvar is not None else mu.new_zeros(()))
        outputs = {"ce": ce, "description_ce": description_ce, "kl": kl, "mu": mu, "logvar": logvar}
        if self.hparams.reconstruction_weight:
            outputs.update(self.reconstruction_losses(batch, latent))
        return outputs

    @traced("training_step")
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        steps = self.trainer.estimated_stepping_batches
        warmup = max(1, int(steps * self.hparams.kl_warmup_ratio))
        beta = self.hparams.kl_weight * min(1.0, (self.global_step + 1) / warmup) if self.hparams.gaussian else 0.0
        loss = self.hparams.qa_loss_weight * outputs["ce"] + beta * outputs["kl"]
        if self.hparams.description_weight:
            loss = loss + self.hparams.description_weight * outputs["description_ce"]
        if self.hparams.reconstruction_weight:
            loss = loss + self.hparams.reconstruction_weight * outputs["reconstruction_mse"]
        metrics = {
            "train/loss": loss, "train/ce": outputs["ce"], "train/kl": outputs["kl"],
            "train/kl_per_dim": outputs["kl"] / self.hparams.latent_dim,
            "train/beta": beta,
        }
        if self.hparams.description_weight:
            metrics["train/description_ce"] = outputs["description_ce"]
        if self.hparams.reconstruction_weight:
            for key in ("reconstruction_mse", "reconstruction_raw_mse"):
                metrics[f"train/{key}"] = outputs[key]
        if outputs["mu"] is not None:
            metrics["train/mu_rms"] = outputs["mu"].square().mean().sqrt()
        if outputs["logvar"] is not None:
            metrics["train/sigma_mean"] = (0.5 * outputs["logvar"]).exp().mean()
        with trace_span("metrics_sync", mode="train"):
            self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True,
                          batch_size=len(batch["ids"]), prog_bar=False)
        return loss

    @traced("validation_step")
    def validation_step(self, batch, batch_idx):
        outputs = self(batch, sample=False)
        metrics = {"val/ce": outputs["ce"], "val/ce_mu": outputs["ce"], "val/kl": outputs["kl"]}
        if self.hparams.reconstruction_weight:
            for key in ("reconstruction_mse", "reconstruction_raw_mse"):
                metrics[f"val/{key}"] = outputs[key]
        if self.hparams.description_weight:
            metrics["val/description_ce"] = outputs["description_ce"]
            if len(batch["ids"]) > 1:
                # In-batch cyclic mismatch has no fixed points; diagnostics never affect training.
                wrong_ce = self.answer_loss(batch["description"], outputs["mu"].roll(1, dims=0))
                metrics["val/description_mismatched_ce"] = wrong_ce
                metrics["val/description_mismatch_gap"] = wrong_ce - outputs["description_ce"]
        if self.hparams.gaussian and self.hparams.qa_loss_weight:
            latent = outputs["mu"] + (0.5 * outputs["logvar"]).exp() * torch.randn_like(outputs["mu"])
            metrics["val/ce_sample"] = self.answer_loss(batch["qa"], latent)
        with trace_span("metrics_sync", mode="validation"):
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True,
                          batch_size=len(batch["ids"]), prog_bar=True)

    def configure_optimizers(self):
        groups = []
        for backbone in (self.encoder, self.qa):
            parameters = [p for p in backbone.parameters() if p.requires_grad] if backbone is not None else []
            if parameters:
                new_encoder = backbone is self.encoder and self.hparams.encoder_type == "qformer"
                groups.append({"params": parameters, "lr": (self.hparams.head_learning_rate if new_encoder
                                                             else self.hparams.learning_rate)})
        if self.encoder is not None:
            heads = [self.queries, *self.mu_head.parameters(), *self.projector.parameters()]
            if self.logvar_head is not None:
                heads.extend(self.logvar_head.parameters())
            if self.reconstruction_decoder is not None:
                heads.extend(self.reconstruction_decoder.parameters())
            groups.append({"params": heads, "lr": self.hparams.head_learning_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay, foreach=True)
        # A resumed run may extend its update budget without changing the
        # original warmup/decay horizon stored in its experiment recipe.
        total_steps = self.hparams.lr_schedule_steps or self.trainer.estimated_stepping_batches
        warmup = max(1, int(total_steps * self.hparams.lr_warmup_ratio))

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "interval": "step",
        }}
