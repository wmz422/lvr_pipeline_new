"""Check full-state Plain SFT continuation before its first optimizer update."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from lightning.pytorch import Callback
import torch


def tensor_sample(tensor):
    flat = tensor.detach().reshape(-1)
    return torch.cat((flat[:32], flat[-32:])).float().cpu().tolist()


def reset_epoch_boundary_iterator(trainer):
    # Worker prefetch can materialize epoch 0 before Lightning restores the
    # completed epoch and advances its sampler. Rebuild only at that boundary.
    fetcher = trainer.fit_loop._data_fetcher
    if fetcher is None:
        raise RuntimeError("Training data fetcher is unavailable at the resume boundary")
    iter(fetcher)


class PlainContinuationAudit(Callback):
    def __init__(self, resume_checkpoint: str, baseline_run: str, start_step: int = 1394,
                 end_step: int = 2788, learning_rate: float = 1e-6):
        self.resume_checkpoint = Path(resume_checkpoint)
        self.baseline_run = Path(baseline_run)
        self.start_step = start_step
        self.end_step = end_step
        self.learning_rate = learning_rate

    def write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n")
        temporary.replace(self.path)

    def check_lr(self, trainer):
        rates = [float(group["lr"]) for group in trainer.optimizers[0].param_groups]
        if not rates or any(not math.isclose(rate, self.learning_rate, rel_tol=1e-9) for rate in rates):
            raise RuntimeError(f"Continuation learning rate changed: {rates}")
        self.record["min_lr"] = min(self.record["min_lr"], *rates)
        self.record["max_lr"] = max(self.record["max_lr"], *rates)

    def validate_model(self, module):
        if int(module.hparams.lr_schedule_steps) != self.start_step or module.hparams.num_latents != 0:
            raise RuntimeError("Continuation must preserve the original Plain SFT learning-rate horizon")

    def on_train_start(self, trainer, module):
        rank = trainer.global_rank
        tag = (self.resume_checkpoint / "latest").read_text().strip()
        directory = self.resume_checkpoint / tag
        metadata = torch.load(directory / "zero_pp_rank_0_mp_rank_00_model_states.pt",
                              map_location="cpu", weights_only=False, mmap=True)
        expected_step = int(metadata["global_step"])
        if not self.start_step <= expected_step < self.end_step or trainer.global_step != expected_step:
            raise RuntimeError(f"Wrong restored global step: {trainer.global_step}, expected {expected_step}")
        self.validate_model(module)
        payload = torch.load(directory / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                             map_location="cpu", weights_only=False, mmap=True)
        saved = payload["optimizer_state_dict"]
        saved_optimizer = saved["optimizer_state_dict"]
        expected_states = [saved_optimizer["state"][key] for group in saved_optimizer["param_groups"]
                           for key in group["params"]]
        optimizer = trainer.optimizers[0]
        actual_states = list(optimizer.state.values())
        if not actual_states or len(actual_states) != len(expected_states):
            raise RuntimeError("Adam optimizer states were not restored")
        steps, samples = [], []
        for expected, actual in zip(expected_states, actual_states):
            step = int(actual["step"])
            if step != expected_step or step != int(expected["step"]):
                raise RuntimeError("Restored Adam step does not match the source checkpoint")
            steps.append(step)
            for key in ("exp_avg", "exp_avg_sq"):
                sample = tensor_sample(actual[key])
                if list(actual[key].shape) != list(expected[key].shape) or sample != tensor_sample(expected[key]):
                    raise RuntimeError(f"Restored Adam {key} differs from its saved rank partition")
                samples.append({"field": key, "shape": list(actual[key].shape),
                                "sample_sha256": hashlib.sha256(json.dumps(sample).encode()).hexdigest()})
        masters = optimizer.fp32_partitioned_groups_flat
        if len(masters) != len(saved["fp32_flat_groups"]):
            raise RuntimeError("Wrong number of restored FP32 master partitions")
        for actual, expected in zip(masters, saved["fp32_flat_groups"]):
            if list(actual.shape) != list(expected.shape) or tensor_sample(actual) != tensor_sample(expected):
                raise RuntimeError("FP32 master weights differ from the source checkpoint")
        baseline = json.loads((self.baseline_run / f"runtime_rank{rank}.json").read_text())
        dataset_ids = [row["id"] for row in trainer.train_dataloader.dataset.items]
        ids_hash = hashlib.sha256(json.dumps(dataset_ids).encode()).hexdigest()
        if ids_hash != baseline["sample_ids_sha256"] or len(dataset_ids) != 178339:
            raise RuntimeError("Continuation changed the training dataset")
        self.path = Path(trainer.default_root_dir) / f"continuation_rank{rank}.json"
        self.record = {"status": "restored", "rank": rank, "at_unix": time.time(),
                       "resume_checkpoint": str(self.resume_checkpoint.resolve()),
                       "initial_global_step": trainer.global_step, "end_step": self.end_step,
                       "adam_steps": steps, "optimizer_moment_samples": samples,
                       "optimizer_moments_match": True, "fp32_master_samples_match": True,
                       "sample_scope": "first_and_last_32_values_per_partition",
                       "sample_ids_sha256": ids_hash, "train_records": len(dataset_ids),
                       "lr_schedule_steps": module.hparams.lr_schedule_steps,
                       "min_lr": self.learning_rate, "max_lr": self.learning_rate,
                       "epochs": [], "optimizer_steps": trainer.global_step}
        self.check_lr(trainer)
        self.write()
        print(f"[continuation] rank={rank} restored_step={trainer.global_step} "
              f"Adam/moments/FP32_samples=matched lr={self.learning_rate:g}", flush=True)

    def on_train_epoch_start(self, trainer, module):
        if trainer.current_epoch != 1:
            raise RuntimeError(f"Expected the second data epoch, got {trainer.current_epoch}")
        sampler = trainer.train_dataloader.sampler
        if getattr(sampler, "epoch", None) != 1:
            raise RuntimeError("The distributed sampler did not advance to epoch 1")
        if self.record["initial_global_step"] == self.start_step:
            reset_epoch_boundary_iterator(trainer)
            self.record["epoch_boundary_iterator_reset"] = True
        self.epoch_indices = list(iter(sampler))
        self.record["epochs"].append({"epoch": trainer.current_epoch, "step": trainer.global_step,
                                      "sampler_epoch": sampler.epoch,
                                      "sampler_indices_sha256": hashlib.sha256(
                                          json.dumps(self.epoch_indices).encode()).hexdigest()})
        self.write()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.check_lr(trainer)
        loader = trainer.train_dataloader
        begin = batch_idx * loader.batch_size
        expected_ids = [loader.dataset.items[index]["id"]
                        for index in self.epoch_indices[begin:begin + loader.batch_size]]
        if batch["ids"] != expected_ids:
            raise RuntimeError("The resumed dataloader does not follow the second epoch's sampler order")
        if "first_batch_idx" not in self.record:
            self.record["first_batch_idx"] = batch_idx
            self.record["first_batch_ids_sha256"] = hashlib.sha256(json.dumps(batch["ids"]).encode()).hexdigest()
            self.write()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.record["optimizer_steps"] = trainer.global_step
        if trainer.global_step % 10 == 0:
            self.write()

    def on_train_end(self, trainer, module):
        if trainer.global_step != self.end_step:
            raise RuntimeError("Continuation ended before the requested one extra epoch")
        self.record.update(status="training_complete", optimizer_steps=trainer.global_step,
                           added_optimizer_steps=trainer.global_step - self.start_step, finished_at_unix=time.time())
        self.write()
