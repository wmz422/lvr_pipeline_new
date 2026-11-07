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


class DescriptionAudit(Callback):
    def on_fit_start(self, trainer, module):
        data = trainer.datamodule.hparams
        assert module.hparams.description_weight == 1.0
        assert data.include_descriptions and data.caption_path
        assert data.description_image_mode == "none"
        assert not data.qa_crops and data.original_pixels == 2000000
        assert data.batch_size * trainer.world_size * trainer.accumulate_grad_batches == 128
        assert data.flow_vae_model_path == module.hparams.flow_model_path
        from .precision import FlowDeepSpeedPrecision
        assert isinstance(trainer.precision_plugin, FlowDeepSpeedPrecision)

    def check_description(self, batch):
        description = batch["description"]
        assert "pixel_values" not in description and "image_grid_thw" not in description
        assert not description["mm_token_type_ids"].any()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.check_description(batch)

    def on_validation_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
        self.check_description(batch)


class AnswerOnlyAudit(NoDescriptionAudit):
    def on_fit_start(self, trainer, module):
        super().on_fit_start(trainer, module)
        assert module.hparams.latent_weight == 0 and module.hparams.qa_loss_weight == 1
        assert module.hparams.prediction_phase == "joint"
        assert not trainer.datamodule.hparams.include_encoder
        assert trainer.datamodule.hparams.predicted_latents_only

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        super().on_train_batch_start(trainer, module, batch, batch_idx)
        assert "encoder" not in batch and "flow" not in batch

    def on_validation_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_start(trainer, module, batch, batch_idx, dataloader_idx)
        assert "encoder" not in batch and "flow" not in batch
