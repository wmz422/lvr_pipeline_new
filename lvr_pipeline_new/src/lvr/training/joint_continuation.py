"""Audit full-state Joint continuation and save exact evaluation milestones."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from lvr.models.latent_prediction import predicted_probability
from lvr.training.continuation import PlainContinuationAudit, reset_epoch_boundary_iterator


def restore_epoch_iterator(trainer, steps_per_epoch):
    """Recreate the restored epoch's sampler and skip already consumed batches."""
    offset = trainer.global_step % steps_per_epoch
    reset_epoch_boundary_iterator(trainer)
    fetcher = trainer.fit_loop._data_fetcher
    for _ in range(offset * trainer.accumulate_grad_batches):
        next(fetcher)
    # Lightning's epoch loop adds the restored ready-batch count in on_run_start.
    # Keep the iterator cursor, but avoid counting skipped batches twice.
    fetcher.fetched = 0
    fetcher.done = False
    return offset


class JointMilestoneModelCheckpoint(ModelCheckpoint):
    """Persist only the explicitly requested global steps, retaining every one."""

    def __init__(self, steps: list[int], dirpath: str | None = None):
        if not steps or any(step < 1 for step in steps) or len(set(steps)) != len(steps):
            raise ValueError("Milestone steps must be distinct positive integers")
        self.steps = tuple(sorted(steps))
        super().__init__(dirpath=dirpath, filename="milestone-{step:06d}",
                         every_n_train_steps=1, save_top_k=-1, save_last=False,
                         save_on_train_epoch_end=False, auto_insert_metric_name=False)

    @property
    def state_key(self):
        return self._generate_state_key(steps=self.steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class JointContinuationAudit(PlainContinuationAudit):
    def __init__(self, resume_checkpoint: str, baseline_run: str, start_step: int = 1394,
                 end_step: int = 4182, language_lr: float = 1e-6, head_lr: float = 1e-5):
        super().__init__(resume_checkpoint, baseline_run, start_step, end_step, language_lr)
        self.expected_rates = [language_lr, head_lr]

    def on_train_start(self, trainer, module):
        super().on_train_start(trainer, module)
        # Lightning skips on_train_epoch_start when resuming inside an epoch.
        if trainer.global_step % self.start_step:
            trainer.train_dataloader.sampler.set_epoch(trainer.current_epoch)
            self.on_train_epoch_start(trainer, module)

    def validate_model(self, module):
        if (module.hparams.lr_schedule_steps != self.start_step
                or module.hparams.num_latents != 8 or module.hparams.prediction_phase != "joint"):
            raise RuntimeError("Expected Joint continuation with the original learning-rate horizon")
        if any(parameter.requires_grad for parameter in module.encoder.parameters()):
            raise RuntimeError("The teacher encoder must remain frozen")

    def check_lr(self, trainer):
        rates = [float(group["lr"]) for group in trainer.optimizers[0].param_groups]
        if len(rates) != 2 or any(not math.isclose(a, b, rel_tol=1e-9)
                                  for a, b in zip(rates, self.expected_rates)):
            raise RuntimeError(f"Joint continuation learning rates changed: {rates}")
        self.record.setdefault("min_lrs", rates.copy())
        self.record.setdefault("max_lrs", rates.copy())
        self.record["min_lrs"] = [min(a, b) for a, b in zip(self.record["min_lrs"], rates)]
        self.record["max_lrs"] = [max(a, b) for a, b in zip(self.record["max_lrs"], rates)]
        self.record["min_lr"], self.record["max_lr"] = min(rates), max(rates)
        if predicted_probability(trainer.global_step) != 1.0:
            raise RuntimeError("Joint continuation reset the predicted-latent curriculum")

    def on_train_epoch_start(self, trainer, module):
        expected_epoch = trainer.global_step // self.start_step
        if trainer.current_epoch != expected_epoch or not 1 <= expected_epoch < 3:
            raise RuntimeError(f"Unexpected resumed epoch: {trainer.current_epoch}, step={trainer.global_step}")
        sampler = trainer.train_dataloader.sampler
        if getattr(sampler, "epoch", None) != expected_epoch:
            raise RuntimeError("The distributed sampler epoch does not match restored training progress")
        skipped = restore_epoch_iterator(trainer, self.start_step)
        self.epoch_indices = list(iter(sampler))
        self.record["epochs"].append({
            "epoch": trainer.current_epoch, "step": trainer.global_step,
            "sampler_epoch": sampler.epoch, "skipped_optimizer_steps": skipped,
            "sampler_indices_sha256": hashlib.sha256(json.dumps(self.epoch_indices).encode()).hexdigest()})
        self.write()

    def on_train_end(self, trainer, module):
        if trainer.global_step != self.end_step:
            raise RuntimeError(f"Joint continuation stopped at {trainer.global_step}, expected {self.end_step}")
        super().on_train_end(trainer, module)
