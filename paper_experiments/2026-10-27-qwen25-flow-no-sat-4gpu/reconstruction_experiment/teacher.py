"""Use the established reconstruction branch, with no description execution/logging."""
import torch
from flow_reconstruction.teacher import FlowVisualBottleneck
from .cached_flow import CachedFrozenFlowRuntime
from .latent_cache import REPRESENTATION


class NoDescriptionFlowTeacher(FlowVisualBottleneck):
    def flow_runtime(self, device=None):
        if self.__dict__["_flow_runtime"] is None:
            self.__dict__["_flow_runtime"] = CachedFrozenFlowRuntime(
                self.hparams.flow_model_path, checkpointing=self.hparams.flow_gradient_checkpointing,
                device=self.device if device is None else device,
                generator_batch_size=self.hparams.flow_generator_batch_size)
        return self.__dict__["_flow_runtime"]

    def forward(self, batch, sample=None):
        if self.hparams.description_weight or "description" in batch:
            raise ValueError("This experiment forbids the description branch")
        flow = batch.get("flow", {})
        if (flow.get("representation") != REPRESENTATION
                or flow.get("gt_protocol_sha256") != self.hparams.flow_protocol_sha256):
            raise ValueError("Offline teacher requires the bound normalized FP32 VAE latents")
        return super().forward(batch, sample=sample)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["flow_metadata"]["input_representation"] = REPRESENTATION

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.hparams.qa_loss_weight * out["ce"] + self.hparams.flow_weight * out["flow_mse"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite teacher QA/reconstruction loss")
        self.log_dict({"train/loss": loss, "train/ce": out["ce"],
                       "train/flow_mse": out["flow_mse"],
                       "train/flow_weighted": self.hparams.flow_weight * out["flow_mse"],
                       "train/mu_rms": out["mu"].square().mean().sqrt()},
                      on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch["ids"]))
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch, sample=False)
        self.log_dict({"val/ce": out["ce"], "val/ce_mu": out["ce"], "val/flow_mse": out["flow_mse"]},
                      on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch["ids"]))
