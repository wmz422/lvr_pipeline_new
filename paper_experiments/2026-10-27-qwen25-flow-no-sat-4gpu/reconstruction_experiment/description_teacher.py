"""Teacher alignment/joint training with cached Flow targets and image-free descriptions."""
import torch

from flow_reconstruction.stage1 import FlowStageCLI
from flow_reconstruction.teacher import FlowVisualBottleneck
from lvr.training.diagnostics import diagnostic_session
from .data import FlowDataModule
from .latent_cache import REPRESENTATION
from .teacher import NoDescriptionFlowTeacher


class DescriptionFlowTeacher(NoDescriptionFlowTeacher):
    """Reuse the FP32 VAE cache while enabling the existing description objective."""

    def forward(self, batch, sample=None):
        flow = batch.get("flow", {})
        if (flow.get("representation") != REPRESENTATION
                or flow.get("gt_protocol_sha256") != self.hparams.flow_protocol_sha256):
            raise ValueError("Offline teacher requires the bound normalized FP32 VAE latents")
        return FlowVisualBottleneck.forward(self, batch, sample=sample)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = (self.hparams.qa_loss_weight * out["ce"]
                + self.hparams.description_weight * out["description_ce"]
                + self.hparams.flow_weight * out["flow_mse"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite teacher QA/description/reconstruction loss")
        self.log_dict({"train/loss": loss, "train/ce": out["ce"],
                       "train/description_ce": out["description_ce"],
                       "train/flow_mse": out["flow_mse"],
                       "train/flow_weighted": self.hparams.flow_weight * out["flow_mse"],
                       "train/mu_rms": out["mu"].square().mean().sqrt()},
                      on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch["ids"]))
        return loss

    def validation_step(self, batch, batch_idx):
        return FlowVisualBottleneck.validation_step(self, batch, batch_idx)


def main():
    with diagnostic_session():
        FlowStageCLI(DescriptionFlowTeacher, FlowDataModule, seed_everything_default=42,
                     save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
