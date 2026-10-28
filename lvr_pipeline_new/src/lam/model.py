"""AdaWorld-style Lightning wrapper for the feature-space LAM.

AdaWorld trains a latent-action VAE by reconstructing the second frame from
the first frame plus a latent action, with a KL regularizer.  Here the shared
LVR module reconstructs Qwen visual features instead of RGB pixels, so the
reconstruction term is feature-space MSE.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from lvr.modules.lam import LatentActionModel


class FeatureLAM(pl.LightningModule):
    """Train ``lvr.modules.lam.LatentActionModel`` with an AdaWorld objective."""

    def __init__(
        self,
        vision_model_path: str,
        *,
        model_dim: int = 1024,
        latent_dim: int = 32,
        patch_size: int = 16,
        enc_blocks: int = 16,
        dec_blocks: int = 16,
        num_heads: int = 16,
        dropout: float = 0.0,
        num_latent: int = 4,
        image_dim: int = 1024,
        beta: float = 2e-4,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 1e-2,
        feature_loss: str = "mse",
    ) -> None:
        super().__init__()
        if not vision_model_path:
            raise ValueError("vision_model_path is required")
        if feature_loss not in {"mse", "smooth_l1"}:
            raise ValueError("feature_loss must be 'mse' or 'smooth_l1'")
        self.save_hyperparameters()

        self.lam = LatentActionModel(
            in_dim=3,
            model_dim=model_dim,
            latent_dim=latent_dim,
            patch_size=patch_size,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            feature_space=True,
            image_dim=image_dim,
            num_latent=num_latent,
            save_dir=vision_model_path,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.lam(batch)

    @staticmethod
    def _kl_loss(z_mu: Tensor, z_log_var: Tensor) -> Tensor:
        """KL(q(z|x)||N(0,I)); lam_feature stores log variance in z_var."""
        per_latent = -0.5 * (
            1.0 + z_log_var - z_mu.square() - z_log_var.exp()
        )
        return per_latent.sum(dim=-1).mean()

    def _shared_step(self, batch: dict[str, Tensor], split: str) -> Tensor:
        outputs = self.lam(batch)
        prediction = outputs["feature"]
        target = outputs["gt_feature"]
        if prediction.shape != target.shape:
            raise RuntimeError(
                "Feature reconstruction shape mismatch: "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
            )

        if self.hparams.feature_loss == "mse":
            reconstruction_loss = F.mse_loss(prediction, target)
        else:
            reconstruction_loss = F.smooth_l1_loss(prediction, target)
        kl_loss = self._kl_loss(outputs["z_mu"], outputs["z_var"])
        loss = reconstruction_loss + self.hparams.beta * kl_loss

        self.log(
            f"{split}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{split}/feature_loss",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{split}/kl_loss",
            kl_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            trainable_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
