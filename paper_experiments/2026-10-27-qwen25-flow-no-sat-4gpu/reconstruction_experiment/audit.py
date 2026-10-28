"""Fail on an accidentally enabled description/helper input in any phase."""
from lightning.pytorch import Callback


class NoDescriptionAudit(Callback):
    def on_fit_start(self, trainer, module):
        data = trainer.datamodule.hparams
        assert module.hparams.description_weight == 0
        assert not data.include_descriptions and data.caption_path is None
        assert not data.qa_crops and data.original_pixels == 2000000
        assert data.batch_size * trainer.world_size * trainer.accumulate_grad_batches == 128
        if hasattr(module, "flow_mapper"):
            assert data.flow_vae_model_path == module.hparams.flow_model_path
            from .precision import FlowDeepSpeedPrecision
            assert isinstance(trainer.precision_plugin, FlowDeepSpeedPrecision)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        assert "description" not in batch

    def on_validation_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
        assert "description" not in batch
