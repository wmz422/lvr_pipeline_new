"""Resume the data cursor and save only after complete optimizer updates."""
import hashlib
import json
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lvr.training.audit import RunAudit as OriginalRunAudit
from lvr.training.checkpoints import OptimizerStepModelCheckpoint
from .data import AccumulationBatchSampler


class RestoreDataCursor(Callback):
    def __init__(self, steps_per_epoch: int = 1346):
        self.steps_per_epoch = steps_per_epoch
        self.configured_epoch = None

    def restore_cursor(self, trainer):
        epoch = trainer.current_epoch
        updates = trainer.global_step - epoch * self.steps_per_epoch
        if not 0 <= updates <= self.steps_per_epoch:
            raise RuntimeError("Checkpoint optimizer and epoch cursors disagree")
        sampler = trainer.train_dataloader.batch_sampler
        if not isinstance(sampler, AccumulationBatchSampler):
            raise RuntimeError("Missing accumulation-aligned sampler")
        sampler.sampler.set_epoch(epoch)
        sampler.resume_epoch = epoch
        sampler.start_batch = updates * trainer.accumulate_grad_batches
        self.epoch_batches = list(sampler.batches())
        if len(self.epoch_batches) != self.steps_per_epoch * trainer.accumulate_grad_batches:
            raise RuntimeError("Unexpected number of data batches")
        fetcher = trainer.fit_loop._data_fetcher
        if fetcher is None:
            raise RuntimeError("Missing Lightning data fetcher")
        iter(fetcher)
        fetcher.fetched, fetcher.done = 0, False
        self.configured_epoch = epoch

    def on_train_start(self, trainer, module):
        if trainer.global_step % self.steps_per_epoch:
            self.restore_cursor(trainer)

    def on_train_epoch_start(self, trainer, module):
        self.restore_cursor(trainer)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if self.configured_epoch != trainer.current_epoch or not 0 <= batch_idx < len(self.epoch_batches):
            raise RuntimeError("Unexpected data cursor")
        dataset = trainer.train_dataloader.dataset
        expected = [dataset.items[index]["id"] for index in self.epoch_batches[batch_idx]]
        if list(batch["ids"]) != expected:
            raise RuntimeError("Consumed examples differ from the expected sampler cursor")


class RunAudit(OriginalRunAudit):
    def on_train_start(self, trainer, module):
        super().on_train_start(trainer, module)
        sampler = trainer.train_dataloader.batch_sampler
        self.record.update(batch_size_per_gpu=sampler.batch_size,
                           loss_gradient_reference_accumulation=getattr(
                               trainer.strategy.precision_plugin, "reference_accumulation", None),
                           sampler_indices_sha256=hashlib.sha256(
                               json.dumps(list(sampler.sampler)).encode()).hexdigest(),
                           reconstruction_transfer_dtype="normalized_vae_float32" if hasattr(module, "flow_mapper") else None,
                           online_vae_loaded=False,
                           epoch_tail_global_batch=len(sampler.sampler) % (
                               sampler.batch_size * sampler.accumulation) * trainer.world_size)
        self.path.write_text(json.dumps(self.record, indent=2) + "\n")


class UpdateCheckpoint(OptimizerStepModelCheckpoint):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if not trainer.fit_loop._should_accumulate():
            super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)


class RecoveryCheckpoint(ModelCheckpoint):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if not trainer.fit_loop._should_accumulate():
            super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)
