"""Predict a frozen A's latent slots with the answer model's own language stack."""

from __future__ import annotations

import copy
import hashlib
import inspect
import math

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from lvr.models.visual_bottleneck import VisualBottleneck, _read_checkpoint, load_checkpoint_weights
from lvr.training.diagnostics import trace_span, traced


def latent_alignment_loss(prediction, target, cosine_weight=1.0, energy_floor=1e-6):
    """Relative per-slot MSE retains amplitude; cosine separately constrains direction."""
    prediction, target = prediction.float(), target.detach().float()
    error = (prediction - target).square().mean(dim=-1)
    energy = target.square().mean(dim=-1).clamp_min(energy_floor)
    relative_mse = (error / energy).mean()
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()
    return {
        "latent_loss": relative_mse + cosine_weight * (1 - cosine),
        "relative_mse": relative_mse, "raw_mse": error.mean(), "cosine": cosine,
        "pred_rms": prediction.square().mean().sqrt(),
        "teacher_rms": target.square().mean().sqrt(),
        "pred_slot_std": prediction.std(dim=1, unbiased=False).mean(),
        "teacher_slot_std": target.std(dim=1, unbiased=False).mean(),
    }


def predicted_probability(step):
    """Joint optimizer steps 0–199 / 200–499 / 500+ use 20% / 50% / 100%."""
    return 0.2 if step < 200 else 0.5 if step < 500 else 1.0


def predicted_mask(ids, step, seed, probability, device):
    """Per-sample draws are rank-independent and reproducible across checkpoint resume."""
    draws = [int.from_bytes(hashlib.blake2b(f"{seed}:{step}:{identity}".encode(), digest_size=8).digest(),
                            "big") / 2**64 for identity in ids]
    return torch.tensor([value < probability for value in draws], dtype=torch.bool, device=device)


def prediction_prompt(qa, pad_token_id):
    """Keep the original image, question and slots; physically remove all answer tokens."""
    slots = qa["slot_mask"]
    if not slots.any(dim=1).all():
        raise ValueError("Prediction requires latent query slots in every sample")
    positions = torch.arange(slots.shape[1], device=slots.device).expand_as(slots)
    ends = positions.masked_fill(~slots, -1).max(dim=1).values + 1
    width = int(ends.max())
    keep = positions[:, :width] < ends[:, None]
    result = {key: qa[key] for key in ("pixel_values", "image_grid_thw") if key in qa}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "slot_mask"):
        value = qa[key][:, :width]
        result[key] = value.masked_fill(~keep, pad_token_id if key == "input_ids" else 0)
    return result


class LatentPrediction(pl.LightningModule):
    def __init__(self, source_checkpoint: str, prediction_phase: str = "warmup",
                 learning_rate: float = 1e-5, head_learning_rate: float = 1e-4,
                 weight_decay: float = 0.01, lr_warmup_ratio: float = 0.03,
                 latent_weight: float = 1.0, cosine_weight: float = 1.0,
                 qa_loss_weight: float = 1.0, description_weight: float = 1.0,
                 energy_floor: float = 1e-6, gradient_checkpointing: bool = True,
                 seed: int = 42, base_model_path: str | None = None,
                 lr_schedule_steps: int | None = None, num_loops: int = 1,
                 loop_alignment: str = "all", allow_question_conditioned_teacher: bool = False):
        super().__init__()
        if isinstance(num_loops, bool) or not isinstance(num_loops, int) or num_loops < 1:
            raise ValueError("num_loops must be a positive integer")
        if loop_alignment not in ("all", "final"):
            raise ValueError("loop_alignment must be all or final")
        if lr_schedule_steps is not None and lr_schedule_steps < 1:
            raise ValueError("lr_schedule_steps must be positive when provided")
        if prediction_phase not in ("warmup", "joint"):
            raise ValueError("prediction_phase must be warmup or joint")
        if any(not math.isfinite(x) or x <= 0 for x in
               (latent_weight, cosine_weight, energy_floor, qa_loss_weight)):
            raise ValueError("Loss weights and energy floor must be finite and positive")
        if not math.isfinite(description_weight) or description_weight < 0:
            raise ValueError("Description weight must be finite and nonnegative")
        self.save_hyperparameters()
        self.network = VisualBottleneck.from_checkpoint(source_checkpoint, "cpu", base_model_path=base_model_path)
        source = self.network.hparams
        if (source.gaussian or source.encoder_type != "qwen" or source.num_latents != 8
                or source.latent_dim != 512 or not source.encoder_includes_original
                or (source.encoder_uses_question and not allow_question_conditioned_teacher)
                or source.description_image_mode != "none"):
            raise ValueError("Expected the deterministic original+all-crops 8x512 teacher checkpoint")
        # These fixed interface fields also allow the shared data/run audit to check this phase.
        for key in ("num_latents", "latent_dim", "encoder_uses_question", "encoder_type",
                    "encoder_includes_original", "qformer_use_geometry", "description_image_mode"):
            self.hparams[key] = source[key]
        self.hparams.model_kind = "latent_prediction"
        self.prediction_queries = nn.Parameter(self.network.queries.detach().clone().float())
        self.prediction_head = copy.deepcopy(self.network.mu_head).float().requires_grad_(True)
        # K=1 retains the original state_dict. Only loop checkpoints own feedback
        # parameters, so inference cannot silently add untrained recurrent weights.
        self.loop_feedback = None
        self.register_parameter("loop_gate_logit", None)
        if num_loops > 1:
            # Isolate new initialization from the existing training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.loop_feedback = nn.Sequential(
                    nn.LayerNorm(source.latent_dim),
                    nn.Linear(source.latent_dim, self.prediction_queries.shape[-1]))
                nn.init.normal_(self.loop_feedback[1].weight, std=0.02 / math.sqrt(source.latent_dim))
                nn.init.zeros_(self.loop_feedback[1].bias)
            self.loop_gate_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        if prediction_phase == "joint":
            self.qa.model.language_model.requires_grad_(True)
            self.qa.lm_head.requires_grad_(True)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if gradient_checkpointing:
            # Frozen RMSNorm weights can be repartitioned to ZeRO-3's empty
            # placeholders while non-reentrant checkpointing saves intermediates.
            # Reentrant replay brackets the complete frozen block with ZeRO hooks.
            # The injected prediction_queries keep its positional hidden states
            # differentiable; no backbone unfreezing/input-gradient hook is needed.
            # Joint's trainable language stack retains non-reentrant checkpointing.
            self.qa.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": prediction_phase == "warmup"})
        self.initial_checkpoint = {
            "path": source_checkpoint, "source_global_step": self.network.checkpoint_step,
            "optimizer_restored": False, "scheduler_restored": False,
            "prediction_initialization": "independent copies of A queries and mu head",
        }
        self.train()

    @property
    def qa(self):
        return self.network.qa

    @property
    def encoder(self):
        return self.network.encoder

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "network"):
            self.encoder.eval()
            self.network.mu_head.eval()
            self.network.projector.eval()
            self.qa.model.visual.eval()
        return self

    def initialize_from_checkpoint(self, path):
        checkpoint = _read_checkpoint(path)
        if checkpoint["hyper_parameters"].get("model_kind") != "latent_prediction":
            raise ValueError("Expected a latent prediction checkpoint")
        if checkpoint["hyper_parameters"]["source_checkpoint"] != self.hparams.source_checkpoint:
            raise ValueError("Warm-up and joint training must share the frozen target checkpoint")
        state = checkpoint["state_dict"]
        if self.loop_feedback is not None and checkpoint["hyper_parameters"].get("num_loops", 1) == 1:
            new_state = {key: value for key, value in self.state_dict().items()
                         if key.startswith("loop_feedback.") or key == "loop_gate_logit"}
            if new_state.keys() & state.keys():
                raise ValueError("Single-round checkpoint unexpectedly contains loop parameters")
            # Strict loading still checks every baseline parameter. Missing loop
            # weights are allowed only for this explicit K=1 -> loop initialization.
            state = {**state, **new_state}
        load_checkpoint_weights(self, state, assign=False)
        self.initial_checkpoint = {"path": path, "source_global_step": checkpoint["global_step"],
                                   "optimizer_restored": False, "scheduler_restored": False}
        return self.initial_checkpoint

    @classmethod
    def from_checkpoint(cls, path, device="cpu", *, source_checkpoint=None, base_model_path=None):
        checkpoint = _read_checkpoint(path)
        parameters = inspect.signature(cls).parameters
        args = {key: value for key, value in checkpoint["hyper_parameters"].items() if key in parameters}
        if source_checkpoint is not None:
            args["source_checkpoint"] = str(source_checkpoint)
        if base_model_path is not None:
            args["base_model_path"] = str(base_model_path)
        model = cls(**{**args, "gradient_checkpointing": False})
        load_checkpoint_weights(model, checkpoint["state_dict"])
        model.checkpoint_step = checkpoint["global_step"]
        return model.requires_grad_(False).eval().to(device)

    def prediction_loop_count(self, num_loops=None):
        count = self.hparams.num_loops if num_loops is None else num_loops
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("num_loops must be a positive integer")
        if count > 1 and self.loop_feedback is None:
            raise ValueError("Multiple rounds require a checkpoint trained with loop feedback")
        return count

    @traced("predict_latents")
    def predict_latents(self, qa_batch, *, num_loops=None, return_all=False):
        count = self.prediction_loop_count(num_loops)
        prompt = prediction_prompt(qa_batch, self.qa.generation_config.pad_token_id)
        size = prompt["input_ids"].shape[0]
        queries = self.prediction_queries.unsqueeze(0).expand(size, -1, -1)
        states = []
        for index in range(count):
            injected = queries
            if index:
                # No detach: final QA/description and alignment can train earlier
                # rounds through both the feedback path and the residual update.
                feedback = self.loop_feedback(states[-1].to(self.loop_feedback[0].weight.dtype))
                injected = queries + feedback
            hidden = self.network._hidden(self.qa, prompt, injected)
            slots = hidden[prompt["slot_mask"]].reshape(size, self.hparams.num_latents, -1)
            candidate = self.prediction_head(slots).float()
            state = candidate if not index else (
                states[-1] + self.loop_gate_logit.float().sigmoid() * (candidate - states[-1]))
            states.append(state)
        return states if return_all else states[-1]

    def forward(self, batch, *, step=0, validation=False):
        with torch.no_grad():
            teacher, _ = self.network.encode(batch["encoder"])
        states = self.predict_latents(batch["qa"], return_all=True)
        predicted = states[-1]
        alignment = []
        for index, state in enumerate(states):
            # Earlier alignment values are diagnostics in final-only mode. Only
            # detach this loss input; the recurrent states retain their graph.
            alignment_state = (state.detach() if self.hparams.loop_alignment == "final"
                               and index < len(states) - 1 else state)
            alignment.append(latent_alignment_loss(alignment_state, teacher, self.hparams.cosine_weight,
                                                  self.hparams.energy_floor))
        values = dict(alignment[-1])
        if len(states) > 1:
            values["final_latent_loss"] = values["latent_loss"]
            if self.hparams.loop_alignment == "all":
                values["latent_loss"] = torch.stack([item["latent_loss"] for item in alignment]).mean()
            values["loop_gate"] = self.loop_gate_logit.float().sigmoid()
            for index, item in enumerate(alignment, 1):
                for key in ("latent_loss", "relative_mse", "cosine"):
                    values[f"loop{index}_{key}"] = item[key]
            values["loop_update_rms"] = (states[-1] - states[-2]).square().mean().sqrt()
        loss = self.hparams.latent_weight * values["latent_loss"]
        if self.hparams.prediction_phase == "joint" or validation:
            probability = 1.0 if validation else predicted_probability(step)
            use_predicted = predicted_mask(batch["ids"], step, self.hparams.seed, probability, predicted.device)
            selected = torch.where(use_predicted[:, None, None], predicted, teacher)
            values["ce"] = self.network.answer_loss(batch["qa"], selected)
            values["predicted_fraction"] = use_predicted.float().mean()
            values["predicted_probability"] = predicted.new_tensor(probability)
            loss = loss + self.hparams.qa_loss_weight * values["ce"]
            if self.hparams.description_weight:
                values["description_ce"] = self.network.answer_loss(batch["description"], predicted)
                loss = loss + self.hparams.description_weight * values["description_ce"]
            if validation:
                values["teacher_ce"] = self.network.answer_loss(batch["qa"], teacher)
        values["loss"] = loss
        values["predicted"] = predicted
        values["teacher"] = teacher
        return values

    @traced("training_step")
    def training_step(self, batch, batch_idx):
        values = self(batch, step=self.global_step)
        if not torch.isfinite(values["loss"]):
            raise FloatingPointError("Nonfinite prediction training loss")
        with trace_span("metrics_sync", mode="train"):
            self.log_dict({f"train/{k}": v for k, v in values.items() if v.ndim == 0},
                          on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch["ids"]))
        return values["loss"]

    @traced("validation_step")
    def validation_step(self, batch, batch_idx):
        values = self(batch, validation=True)
        with trace_span("metrics_sync", mode="validation"):
            self.log_dict({f"val/{k}": v for k, v in values.items() if v.ndim == 0},
                          on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch["ids"]))

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # DeepSpeed performs globally correct clipping on partitioned gradients from
        # Trainer.gradient_clip_val. Calling torch's local implementation on ZeRO
        # shards would both duplicate the operation and compute the wrong norm.
        if getattr(self.trainer.strategy, "strategy_name", None) == "deepspeed":
            return
        parameters = [p for p in self.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_val or 1.0, error_if_nonfinite=True)

    def configure_optimizers(self):
        head_parameters = [self.prediction_queries, *self.prediction_head.parameters()]
        if self.loop_feedback is not None:
            head_parameters.extend([*self.loop_feedback.parameters(), self.loop_gate_logit])
        groups = [{"params": head_parameters,
                   "lr": self.hparams.head_learning_rate}]
        language = [p for p in self.qa.parameters() if p.requires_grad]
        if language:
            groups.insert(0, {"params": language, "lr": self.hparams.learning_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay, foreach=True)
        # Extending a resumed run must not stretch its original cosine schedule.
        total = int(self.hparams.lr_schedule_steps or self.trainer.estimated_stepping_batches)
        warmup = max(1, int(total * self.hparams.lr_warmup_ratio))

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "interval": "step"}}
