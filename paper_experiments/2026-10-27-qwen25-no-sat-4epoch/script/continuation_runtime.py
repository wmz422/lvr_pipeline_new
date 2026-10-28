"""Full-state continuation, exact sample cursor, and epoch-boundary checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from lvr.data.crop_qa import CropQADataModule
from lvr.training.continuation import tensor_sample
from lvr.training.checkpoints import OptimizerStepModelCheckpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def native_state(checkpoint):
    checkpoint = Path(checkpoint)
    tag = (checkpoint / "latest").read_text().strip()
    return torch.load(checkpoint / tag / "zero_pp_rank_0_mp_rank_00_model_states.pt",
                      map_location="cpu", mmap=True, weights_only=False)


def original_lr(step, peak=1e-5, horizon=2784, warmup=83):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1., (step - warmup) / (horizon - warmup))
    return peak * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)


class AccumulationBatchSampler(Sampler):
    """Keep all examples and give DeepSpeed two microbatches per optimizer update.

    A rank has 43,056 examples: 1,345 groups of 32 plus a tail of 16.
    Split the tail into 8+8 instead of carrying an unstepped gradient across
    the epoch boundary. No samples are duplicated or dropped.
    Cursor skipping happens before image decoding/prefetch, not by replaying data.
    """
    def __init__(self, sampler, batch_size=16, accumulation=2):
        self.sampler = sampler
        self.batch_size = batch_size
        self.accumulation = accumulation
        self.resume_epoch = -1
        self.start_batch = 0
        self.drop_last = False

    def batches(self):
        indices = list(self.sampler)
        group_size = self.batch_size * self.accumulation
        for start in range(0, len(indices), group_size):
            group = indices[start:start + group_size]
            width, remainder = divmod(len(group), self.accumulation)
            if width < 1 or remainder:
                raise ValueError("The final accumulation group must split into equal nonempty microbatches")
            for offset in range(0, len(group), width):
                yield group[offset:offset + width]

    def __iter__(self):
        skip = self.start_batch if self.sampler.epoch == self.resume_epoch else 0
        for index, batch in enumerate(self.batches()):
            if index >= skip:
                yield batch

    def __len__(self):
        # Lightning must see the full epoch length even when restoring a cursor.
        return math.ceil(len(self.sampler) / (self.batch_size * self.accumulation)) * self.accumulation


class ContinuationDataModule(CropQADataModule):
    def _loader(self, dataset, training):
        original = super()._loader(dataset, training)
        kwargs = {"num_workers": original.num_workers, "collate_fn": original.collate_fn,
                  "pin_memory": original.pin_memory}
        if original.num_workers:
            kwargs.update(persistent_workers=original.persistent_workers,
                          prefetch_factor=original.prefetch_factor)
        if not training:
            # Automatic sampler replacement is disabled for the custom train
            # batch sampler; keep Lightning's usual distributed validation split.
            sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size,
                                         rank=self.trainer.global_rank, shuffle=False)
            return DataLoader(dataset, batch_size=original.batch_size, sampler=sampler, **kwargs)
        accumulation = int(self.trainer.accumulate_grad_batches)
        sampler = AccumulationBatchSampler(original.sampler, self.hparams.batch_size, accumulation)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)


def normalize_completed_epoch(checkpoint, steps_per_epoch):
    """Finish only the loop cursor saved during on_train_epoch_end, in memory."""
    step = int(checkpoint["global_step"])
    if step % steps_per_epoch:
        return False
    fit = checkpoint["loops"]["fit_loop"]
    batch = fit["epoch_loop.batch_progress"]
    if not batch["is_last_batch"]:
        raise ValueError("An integer-epoch checkpoint must have finished its last batch")
    for scope in ("total", "current"):
        progress = fit["epoch_progress"][scope]
        if not (progress["ready"] == progress["started"] == progress["processed"]
                and progress["completed"] in (progress["processed"], progress["processed"] - 1)):
            raise ValueError("Expected an on_train_epoch_end checkpoint")
        progress["completed"] = progress["processed"]

    def reset_current(value):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key == "current" and isinstance(item, dict):
                for name in item:
                    item[name] = 0
            else:
                reset_current(item)
    for key, value in fit.items():
        if key.startswith("epoch_loop."):
            reset_current(value)
    batch["is_last_batch"] = False
    fit["epoch_loop.val_loop.batch_progress"]["is_last_batch"] = False
    return True


class RestoreDataCursor(Callback):
    """Restore sampler epoch and batch offset; verify every consumed sample ID."""
    def __init__(self, steps_per_epoch: int = 1346):
        self.steps_per_epoch = steps_per_epoch
        self.configured_epoch = None

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.cursor_normalized = normalize_completed_epoch(checkpoint, self.steps_per_epoch)

    def restore_cursor(self, trainer):
        epoch, updates = divmod(trainer.global_step, self.steps_per_epoch)
        if trainer.current_epoch != epoch:
            raise RuntimeError(f"Restored epoch {trainer.current_epoch} differs from expected {epoch}")
        loader = trainer.train_dataloader
        sampler = loader.batch_sampler
        if not isinstance(sampler, AccumulationBatchSampler):
            raise RuntimeError("Continuation requires the accumulation-aligned batch sampler")
        sampler.sampler.set_epoch(epoch)
        sampler.resume_epoch = epoch
        sampler.start_batch = updates * trainer.accumulate_grad_batches
        self.epoch_batches = list(sampler.batches())
        if len(self.epoch_batches) != self.steps_per_epoch * trainer.accumulate_grad_batches:
            raise RuntimeError("Data loader length differs from the declared epoch budget")
        fetcher = trainer.fit_loop._data_fetcher
        if fetcher is None:
            raise RuntimeError("Missing Lightning data fetcher")
        iter(fetcher)
        # Lightning adds its restored ready count in epoch_loop.on_run_start.
        fetcher.fetched = 0
        fetcher.done = False
        self.configured_epoch = epoch
        self.first_expected_batch = sampler.start_batch

    def on_train_start(self, trainer, module):
        if trainer.global_step % self.steps_per_epoch:
            self.restore_cursor(trainer)

    def on_train_epoch_start(self, trainer, module):
        self.restore_cursor(trainer)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if self.configured_epoch != trainer.current_epoch:
            raise RuntimeError("The data iterator was not restored for the current epoch")
        if not 0 <= batch_idx < len(self.epoch_batches):
            raise RuntimeError(f"Unexpected resumed batch index: {batch_idx}")
        dataset = trainer.train_dataloader.dataset
        expected = [dataset.items[index]["id"] for index in self.epoch_batches[batch_idx]]
        if list(batch["ids"]) != expected:
            raise RuntimeError("The resumed batch differs from the expected sampler cursor")


class ContinuationAudit(RestoreDataCursor):
    def __init__(self, resume_checkpoint: str, baseline_run: str, end_step: int,
                 steps_per_epoch: int = 1346):
        super().__init__(steps_per_epoch)
        self.resume_checkpoint = Path(resume_checkpoint)
        self.baseline_run = Path(baseline_run)
        self.end_step = end_step

    def on_train_start(self, trainer, module):
        super().on_train_start(trainer, module)
        saved = native_state(self.resume_checkpoint)
        start = int(saved["global_step"])
        if (trainer.global_step != start or int(saved["global_steps"]) != start
                or not 696 <= start < self.end_step or trainer.world_size != 4):
            raise RuntimeError("Wrong restored step or ZeRO partition count")
        if module.hparams.num_latents != 0 or module.hparams.lr_schedule_steps != 2784:
            raise RuntimeError("Plain SFT and the original LR horizon must be retained")
        rank = trainer.global_rank
        tag = (self.resume_checkpoint / "latest").read_text().strip()
        payload = torch.load(self.resume_checkpoint / tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
                             map_location="cpu", weights_only=False, mmap=True)["optimizer_state_dict"]
        expected_optimizer = payload["optimizer_state_dict"]
        expected_states = [expected_optimizer["state"][key]
                           for group in expected_optimizer["param_groups"] for key in group["params"]]
        actual = trainer.optimizers[0]
        states = list(actual.state.values())
        if not states or len(states) != len(expected_states):
            raise RuntimeError("Adam states were not restored")
        for state, expected in zip(states, expected_states):
            if int(state["step"]) != start or int(expected["step"]) != start:
                raise RuntimeError("Adam step differs from the restored checkpoint")
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].shape != expected[key].shape or tensor_sample(state[key]) != tensor_sample(expected[key]):
                    raise RuntimeError(f"Adam {key} partition sample differs after restore")
        masters = actual.fp32_partitioned_groups_flat
        if len(masters) != len(payload["fp32_flat_groups"]):
            raise RuntimeError("FP32 master partitions were not restored")
        for state, expected in zip(masters, payload["fp32_flat_groups"]):
            if state.shape != expected.shape or tensor_sample(state) != tensor_sample(expected):
                raise RuntimeError("FP32 master partition sample differs after restore")
        schedulers = [config.scheduler.state_dict() for config in trainer.lr_scheduler_configs]
        if len(schedulers) != 1:
            raise RuntimeError("Expected one restored LR scheduler")
        for key in ("last_epoch", "_step_count", "base_lrs", "_last_lr"):
            if schedulers[0][key] != saved["lr_schedulers"][0][key]:
                raise RuntimeError(f"Scheduler state differs after restore: {key}")
        ids = [row["id"] for row in trainer.train_dataloader.dataset.items]
        ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        baseline = json.loads((self.baseline_run / f"runtime_rank{rank}.json").read_text())
        if len(ids) != 172224 or ids_hash != baseline["sample_ids_sha256"]:
            raise RuntimeError("The no-SAT training data changed")
        self.path = Path(trainer.default_root_dir) / f"continuation_rank{rank}.json"
        self.started = time.time()
        self.record = {"status": "training", "rank": rank, "initial_global_step": start,
                       "target_step": self.end_step, "optimizer_steps": start,
                       "resume_checkpoint": str(self.resume_checkpoint), "started_at_unix": self.started,
                       "optimizer_moments_match": True, "fp32_master_samples_match": True,
                       "partition_check_scope": "first_and_last_32_values", "scheduler_state_matches": True,
                       "sample_ids_sha256": ids_hash, "train_samples": len(ids), "num_workers_per_gpu":
                       trainer.train_dataloader.num_workers, "initial_learning_rate": actual.param_groups[0]["lr"]}
        self.check_lr(trainer)
        write_json(self.path, self.record)

    def check_lr(self, trainer):
        expected = original_lr(trainer.global_step)
        if any(not math.isclose(float(group["lr"]), expected, rel_tol=1e-9)
               for group in trainer.optimizers[0].param_groups):
            raise RuntimeError("Learning rate diverged from the original schedule")
        if int(trainer.strategy.deepspeed_engine.global_steps) != trainer.global_step:
            raise RuntimeError("DeepSpeed optimizer updates diverged from Lightning's global step")

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        super().on_train_batch_start(trainer, module, batch, batch_idx)
        self.check_lr(trainer)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.check_lr(trainer)
        if trainer.global_step % 10 == 0:
            self.record.update(optimizer_steps=trainer.global_step, epoch=trainer.current_epoch,
                               elapsed_seconds=time.time()-self.started, updated_at_unix=time.time(),
                               learning_rate=float(trainer.optimizers[0].param_groups[0]["lr"]),
                               loss=float(outputs["loss"].detach()),
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(self.path, self.record)
            if trainer.is_global_zero:
                print(f"[continuation] step={trainer.global_step}/{self.end_step} "
                      f"loss={self.record['loss']:.6f} lr={self.record['learning_rate']:.8g}", flush=True)

    def on_train_end(self, trainer, module):
        if trainer.global_step != self.end_step:
            raise RuntimeError(f"Training stopped at {trainer.global_step}, expected {self.end_step}")
        self.check_lr(trainer)
        self.record.update(status="training_complete", optimizer_steps=trainer.global_step,
                           finished_at_unix=time.time(), elapsed_seconds=time.time()-self.started)
        write_json(self.path, self.record)


class EpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, steps_per_epoch: int = 1346, dirpath: str | None = None):
        self.steps_per_epoch = steps_per_epoch
        super().__init__(dirpath=dirpath, filename="candidate-{step:06d}", save_top_k=-1,
                         save_last="link", every_n_epochs=1, save_on_train_epoch_end=True,
                         auto_insert_metric_name=False, enable_version_counter=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_step == (trainer.current_epoch + 1) * self.steps_per_epoch:
            super().on_train_epoch_end(trainer, pl_module)


class HalfEpochModelCheckpoint(OptimizerStepModelCheckpoint):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.fit_loop._should_accumulate():
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class RecoveryModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str | None = None):
        super().__init__(dirpath=dirpath, filename="recovery-{step:06d}", every_n_train_steps=400,
                         save_top_k=1, save_last=False, save_on_train_epoch_end=False,
                         auto_insert_metric_name=False, enable_version_counter=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.fit_loop._should_accumulate():
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
