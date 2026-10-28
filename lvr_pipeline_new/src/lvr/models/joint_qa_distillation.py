"""Warmup -> joint online distillation through the same trainable Student QA."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from lvr.models.chunked_loss import ChunkedLMHead
from lvr.models.latent_prediction import (
    LatentPrediction, latent_alignment_loss, predicted_mask, predicted_probability,
)


class _JointFullVocabularyLoss(torch.autograd.Function):
    """Chunk/recompute exact-vocabulary CE + KL, including LM-head gradients.

    Teacher logits use the current SAME head but are detached as a target.
    All token chunks stay inside one module forward/backward for ZeRO-3.
    Only O(chunk_tokens * vocabulary) distributions are materialized.
    """

    @staticmethod
    def forward(ctx, student, teacher, weight, bias, targets, temperature, chunk_tokens):
        count = targets.numel()
        kd = student.new_zeros((), dtype=torch.float32)
        ce, teacher_ce = kd.clone(), kd.clone()
        for start in range(0, count, chunk_tokens):
            end = start + chunk_tokens
            s = F.linear(student[start:end], weight, bias)
            t = F.linear(teacher[start:end], weight, bias)
            logits_dtype = s.dtype
            slog = F.log_softmax(s.float() / temperature, -1)
            tlog = F.log_softmax(t.float() / temperature, -1)
            kd += (tlog.exp() * (tlog - slog)).sum()
            ce += F.cross_entropy(s.float(), targets[start:end], reduction="sum")
            teacher_ce += F.cross_entropy(t.float(), targets[start:end], reduction="sum")
        ctx.save_for_backward(student, teacher, weight, bias, targets)
        ctx.temperature, ctx.chunk_tokens, ctx.logits_dtype = temperature, chunk_tokens, logits_dtype
        ctx.set_materialize_grads(False)
        teacher_ce = teacher_ce / count
        ctx.mark_non_differentiable(teacher_ce)
        return kd * temperature**2 / count, ce / count, teacher_ce

    @staticmethod
    def backward(ctx, grad_kd, grad_ce, _grad_teacher_ce):
        student, teacher, weight, bias, targets = ctx.saved_tensors
        count, tau, dtype = targets.numel(), ctx.temperature, ctx.logits_dtype
        grad_student = torch.empty_like(student) if ctx.needs_input_grad[0] else None
        accum_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype) if ctx.needs_input_grad[2] else None
        grad_bias = (torch.zeros_like(bias, dtype=accum_dtype)
                     if bias is not None and ctx.needs_input_grad[3] else None)
        w, b = weight.to(dtype), bias.to(dtype) if bias is not None else None
        for start in range(0, count, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, count)
            states = student[start:end].to(dtype)
            s = F.linear(states, w, b).float()
            t = F.linear(teacher[start:end].to(dtype), w, b).float()
            grad = torch.zeros_like(s)
            if grad_kd is not None:
                grad += (F.softmax(s/tau, -1) - F.softmax(t/tau, -1)) * (tau/count) * grad_kd
            if grad_ce is not None:
                ce_grad = F.softmax(s, -1)
                ce_grad[torch.arange(end-start, device=s.device), targets[start:end]] -= 1
                grad += ce_grad * (grad_ce/count)
            linear_grad = grad.to(dtype)
            if grad_student is not None:
                grad_student[start:end] = (linear_grad @ w).to(student.dtype)
            if grad_weight is not None:
                grad_weight.add_((linear_grad.T @ states).to(accum_dtype))
            if grad_bias is not None:
                grad_bias.add_(linear_grad.sum(0).to(accum_dtype))
        return (grad_student, None, grad_weight.to(weight.dtype) if grad_weight is not None else None,
                grad_bias.to(bias.dtype) if grad_bias is not None else None, None, None, None)


class JointDistillationLMHead(ChunkedLMHead):
    @classmethod
    def from_head(cls, head):
        if isinstance(head, cls):
            return head
        if not isinstance(head, nn.Linear):
            raise TypeError("Distillation requires a Linear LM head")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = head.in_features, head.out_features
        result.register_parameter("weight", head.weight)
        result.register_parameter("bias", head.bias)
        result.train(head.training)
        return result

    def forward(self, input, targets=None, *, teacher_hidden=None, temperature=2.0, chunk_tokens=64):
        if teacher_hidden is None:
            return super().forward(input, targets)
        if (targets is None or input.ndim != 2 or teacher_hidden.shape != input.shape
                or targets.ndim != 1 or targets.numel() != input.shape[0] or not targets.numel()):
            raise ValueError("Distillation requires matching nonempty answer states and targets")
        if not math.isfinite(temperature) or temperature <= 0 or chunk_tokens < 1:
            raise ValueError("Temperature and chunk size must be positive")
        return _JointFullVocabularyLoss.apply(input, teacher_hidden.detach(), self.weight, self.bias,
                                              targets, float(temperature), chunk_tokens)


class JointQAOutputDistillation(LatentPrediction):
    def __init__(self, source_checkpoint: str, prediction_phase: str = "joint",
                 learning_rate: float = 1e-5, head_learning_rate: float = 1e-4,
                 weight_decay: float = 0.01, lr_warmup_ratio: float = 0.03,
                 latent_weight: float = 1.0, cosine_weight: float = 1.0,
                 qa_loss_weight: float = 1.0, description_weight: float = 1.0,
                 energy_floor: float = 1e-6, gradient_checkpointing: bool = True,
                 seed: int = 42, base_model_path: str | None = None,
                 lr_schedule_steps: int | None = None, num_loops: int = 1,
                 loop_alignment: str = "all", kd_weight: float = 0.1,
                 kd_temperature: float = 2.0, kd_chunk_tokens: int = 64):
        if prediction_phase != "joint" or num_loops != 1:
            raise ValueError("Expected the baseline single-loop joint stage")
        if not math.isfinite(kd_weight) or kd_weight < 0:
            raise ValueError("KD weight must be finite and nonnegative")
        if not math.isfinite(kd_temperature) or kd_temperature <= 0 or kd_chunk_tokens < 1:
            raise ValueError("KD temperature and chunk size must be positive")
        super().__init__(source_checkpoint, prediction_phase, learning_rate, head_learning_rate,
                         weight_decay, lr_warmup_ratio, latent_weight, cosine_weight,
                         qa_loss_weight, description_weight, energy_floor,
                         gradient_checkpointing, seed, base_model_path,
                         lr_schedule_steps, num_loops, loop_alignment)
        self.hparams.update(kd_weight=kd_weight, kd_temperature=kd_temperature,
                            kd_chunk_tokens=kd_chunk_tokens,
                            kd_target="stop_gradient_current_student_on_teacher_latent")
        self.qa.lm_head = JointDistillationLMHead.from_head(self.qa.lm_head)

    def answer_hidden(self, batch, latent):
        hidden = self.network._hidden(self.qa, batch, self.network.project_latents(latent))
        labels = batch["labels"][:, 1:]
        mask = (labels != -100) & batch["attention_mask"][:, 1:].bool() & batch["attention_mask"][:, :-1].bool()
        if not mask.any():
            raise ValueError("QA distillation contains no valid answer tokens")
        return hidden[:, :-1][mask], labels[mask]

    def forward(self, batch, *, step=0, validation=False):
        if not self.hparams.kd_weight:
            # Exactly the baseline joint objective, curriculum and trainability.
            values = super().forward(batch, step=step, validation=validation)
            values["kd"] = values["loss"].new_zeros(())
            return values
        with torch.no_grad():
            teacher, _ = self.network.encode(batch["encoder"])
        predicted = self.predict_latents(batch["qa"])
        values = latent_alignment_loss(predicted, teacher, self.hparams.cosine_weight, self.hparams.energy_floor)
        # Target path stops all parameter/latent gradients for this forward.
        # Its distribution changes with the SAME Student's future updates.
        with torch.no_grad():
            teacher_hidden, _ = self.answer_hidden(batch["qa"], teacher)
        student_hidden, targets = self.answer_hidden(batch["qa"], predicted)
        kd, predicted_ce, teacher_ce = self.qa.lm_head(
            student_hidden, targets, teacher_hidden=teacher_hidden,
            temperature=self.hparams.kd_temperature, chunk_tokens=self.hparams.kd_chunk_tokens)
        probability = 1.0 if validation else predicted_probability(step)
        mask = predicted_mask(batch["ids"], step, self.hparams.seed, probability, predicted.device)
        if probability == 1.0:
            ce = predicted_ce
        else:
            selected = torch.where(mask[:, None, None], predicted, teacher)
            ce = self.network.answer_loss(batch["qa"], selected)
        values.update(kd=kd, ce=ce, predicted_ce=predicted_ce, teacher_ce=teacher_ce,
                      predicted_probability=predicted.new_tensor(probability),
                      predicted_fraction=mask.float().mean())
        loss = (self.hparams.latent_weight * values["latent_loss"]
                + self.hparams.qa_loss_weight * ce + self.hparams.kd_weight * kd)
        if self.hparams.description_weight:
            values["description_ce"] = self.network.answer_loss(batch["description"], predicted)
            loss += self.hparams.description_weight * values["description_ce"]
        values.update(loss=loss, predicted=predicted, teacher=teacher)
        return values
