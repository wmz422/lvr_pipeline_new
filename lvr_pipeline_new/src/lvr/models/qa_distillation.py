"""Frozen final-student QA distillation; inference state_dict stays unchanged."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from lvr.models.chunked_loss import ChunkedLMHead
from lvr.models.latent_prediction import LatentPrediction, latent_alignment_loss


class _FrozenFullVocabularyLoss(torch.autograd.Function):
    """Recompute token chunks in backward instead of retaining N x vocabulary.

    The owning LM-head forward/backward brackets every chunk with a single
    ZeRO-3 gather. Only the student hidden states are differentiable. Both CE
    and KL use precisely the same causal answer positions and full vocabulary.
    """

    @staticmethod
    def forward(ctx, student, teacher, weight, bias, targets, temperature, chunk_tokens):
        count = targets.numel()
        kd = student.new_zeros((), dtype=torch.float32)
        ce = kd.clone()
        teacher_ce = kd.clone()
        for start in range(0, count, chunk_tokens):
            end = start + chunk_tokens
            s = F.linear(student[start:end], weight, bias)
            t = F.linear(teacher[start:end], weight, bias)
            logits_dtype = s.dtype
            slog = F.log_softmax(s.float() / temperature, dim=-1)
            tlog = F.log_softmax(t.float() / temperature, dim=-1)
            kd += (tlog.exp() * (tlog - slog)).sum()
            ce += F.cross_entropy(s.float(), targets[start:end], reduction="sum")
            teacher_ce += F.cross_entropy(t.float(), targets[start:end], reduction="sum")
        ctx.save_for_backward(student, teacher, weight, targets)
        ctx.bias = bias
        ctx.temperature, ctx.chunk_tokens = temperature, chunk_tokens
        ctx.logits_dtype = logits_dtype
        # This is a frozen-head operator: weight/teacher never get gradients.
        ctx.set_materialize_grads(False)
        teacher_ce = teacher_ce / count
        ctx.mark_non_differentiable(teacher_ce)
        return kd * temperature**2 / count, ce / count, teacher_ce

    @staticmethod
    def backward(ctx, grad_kd, grad_ce, _grad_teacher_ce):
        student, teacher, weight, targets = ctx.saved_tensors
        count, tau = targets.numel(), ctx.temperature
        dtype = ctx.logits_dtype
        grad_student = torch.empty_like(student)
        # Match the precision of the forward Linear, including BF16 autocast.
        w = weight.to(dtype)
        bias = ctx.bias.to(dtype) if ctx.bias is not None else None
        for start in range(0, count, ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, count)
            s = F.linear(student[start:end].to(dtype), w, bias).float()
            t = F.linear(teacher[start:end].to(dtype), w, bias).float()
            grad = torch.zeros_like(s)
            if grad_kd is not None:
                grad += (F.softmax(s / tau, -1) - F.softmax(t / tau, -1)) * (tau / count) * grad_kd
            if grad_ce is not None:
                ce_grad = F.softmax(s, -1)
                ce_grad[torch.arange(end-start, device=s.device), targets[start:end]] -= 1
                grad += ce_grad * (grad_ce / count)
            grad_student[start:end] = (grad.to(dtype) @ w).to(student.dtype)
        return grad_student, None, None, None, None, None, None


class FrozenDistillationLMHead(ChunkedLMHead):
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
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("The distillation scoring head must be frozen")
        if (targets is None or input.ndim != 2 or teacher_hidden.shape != input.shape
                or targets.ndim != 1 or input.shape[0] != targets.numel() or not targets.numel()):
            raise ValueError("Distillation expects matching nonempty answer hidden states and targets")
        if not math.isfinite(temperature) or temperature <= 0 or chunk_tokens < 1:
            raise ValueError("Temperature and chunk size must be positive")
        return _FrozenFullVocabularyLoss.apply(input, teacher_hidden.detach(), self.weight,
                                               self.bias, targets, float(temperature), chunk_tokens)


class FrozenQAOutputDistillation(LatentPrediction):
    def __init__(self, source_checkpoint: str, prediction_phase: str = "joint",
                 learning_rate: float = 1e-6, head_learning_rate: float = 1e-5,
                 weight_decay: float = 0.01, lr_warmup_ratio: float = 0.0,
                 latent_weight: float = 1.0, cosine_weight: float = 1.0,
                 qa_loss_weight: float = 1.0, description_weight: float = 1.0,
                 energy_floor: float = 1e-6, gradient_checkpointing: bool = True,
                 seed: int = 42, base_model_path: str | None = None,
                 lr_schedule_steps: int | None = None, num_loops: int = 1,
                 loop_alignment: str = "all", kd_weight: float = 0.1,
                 kd_temperature: float = 2.0, kd_chunk_tokens: int = 64,
                 baseline_step: int = 1394):
        if prediction_phase != "joint" or num_loops != 1:
            raise ValueError("This experiment continues the one-loop final joint checkpoint")
        if not math.isfinite(kd_weight) or kd_weight < 0:
            raise ValueError("KD weight must be finite and nonnegative")
        if not math.isfinite(kd_temperature) or kd_temperature <= 0 or kd_chunk_tokens < 1:
            raise ValueError("KD temperature and chunk size must be positive")
        # Warmup construction leaves the shared QA frozen and enables reentrant
        # checkpointing, which preserves frozen ZeRO-3 parameter boundaries.
        # The overridden forward below still computes QA/description joint CE.
        super().__init__(source_checkpoint, "warmup", learning_rate, head_learning_rate,
                         weight_decay, lr_warmup_ratio, latent_weight, cosine_weight,
                         qa_loss_weight, description_weight, energy_floor,
                         gradient_checkpointing, seed, base_model_path,
                         lr_schedule_steps, num_loops, loop_alignment)
        self.hparams.update(prediction_phase="joint", kd_weight=kd_weight,
                            kd_temperature=kd_temperature, kd_chunk_tokens=kd_chunk_tokens,
                            baseline_step=baseline_step, kd_target="fixed_final_student_on_teacher_latent")
        self.qa.requires_grad_(False)
        self.qa.lm_head = FrozenDistillationLMHead.from_head(self.qa.lm_head)

    def answer_hidden(self, batch, latent):
        hidden = self.network._hidden(self.qa, batch, self.network.project_latents(latent))
        labels = batch["labels"][:, 1:]
        mask = (labels != -100) & batch["attention_mask"][:, 1:].bool() & batch["attention_mask"][:, :-1].bool()
        if not mask.any():
            raise ValueError("QA distillation contains no valid answer tokens")
        return hidden[:, :-1][mask], labels[mask]

    def forward(self, batch, *, step=0, validation=False):
        with torch.no_grad():
            teacher, _ = self.network.encode(batch["encoder"])
        predicted = self.predict_latents(batch["qa"])
        values = latent_alignment_loss(predicted, teacher, self.hparams.cosine_weight, self.hparams.energy_floor)
        student_hidden, targets = self.answer_hidden(batch["qa"], predicted)
        if self.hparams.kd_weight:
            with torch.no_grad():
                teacher_hidden, _ = self.answer_hidden(batch["qa"], teacher)
            kd, ce, teacher_ce = self.qa.lm_head(
                student_hidden, targets, teacher_hidden=teacher_hidden,
                temperature=self.hparams.kd_temperature, chunk_tokens=self.hparams.kd_chunk_tokens)
            values.update(kd=kd, ce=ce, teacher_ce=teacher_ce)
        else:
            values["kd"] = predicted.new_zeros(())
            values["ce"] = self.qa.lm_head(student_hidden, targets)
            if validation:
                with torch.no_grad():
                    values["teacher_ce"] = self.network.answer_loss(batch["qa"], teacher)
        loss = (self.hparams.latent_weight * values["latent_loss"]
                + self.hparams.qa_loss_weight * values["ce"] + self.hparams.kd_weight * values["kd"])
        if self.hparams.description_weight:
            values["description_ce"] = self.network.answer_loss(batch["description"], predicted)
            loss += self.hparams.description_weight * values["description_ce"]
        values.update(loss=loss, predicted=predicted, teacher=teacher,
                      predicted_fraction=predicted.new_tensor(1.0),
                      predicted_probability=predicted.new_tensor(1.0),
                      effective_joint_step=predicted.new_tensor(self.hparams.baseline_step + step))
        return values

    def configure_optimizers(self):
        # A new optimizer is intentional because QA is now frozen. Continue at
        # the baseline's terminal head LR instead of restarting its warmup.
        parameters = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(parameters, lr=self.hparams.head_learning_rate,
                                 weight_decay=self.hparams.weight_decay, foreach=True)
