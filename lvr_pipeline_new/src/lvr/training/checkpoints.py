"""Save complete training state at explicitly planned optimizer updates."""
from lightning.pytorch.callbacks import ModelCheckpoint


class OptimizerStepModelCheckpoint(ModelCheckpoint):
    """Use ModelCheckpoint's distributed/resume handling at fractional-epoch milestones."""

    def __init__(self, steps: list[int], **kwargs):
        if not steps or any(not isinstance(step, int) or step < 1 for step in steps):
            raise ValueError("Checkpoint steps must be positive integers")
        self.steps = tuple(sorted(set(steps)))
        super().__init__(every_n_train_steps=1, every_n_epochs=0,
                         save_on_train_epoch_end=False, **kwargs)

    @property
    def state_key(self):
        return f"{super().state_key}:steps={self.steps}"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_end(self, trainer, pl_module):
        # These are scheduled candidates, independent of validation cadence.
        pass
