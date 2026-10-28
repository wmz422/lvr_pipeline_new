"""Exercise real checkpoint saves with accumulation, validation, and resumption."""
from pathlib import Path
import tempfile
import unittest

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader, TensorDataset

from lvr.training.checkpoints import OptimizerStepModelCheckpoint


class SmallTrainingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.layer(batch[0]).square().mean())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


class OptimizerStepCheckpointTests(unittest.TestCase):
    def test_accumulation_resume_keeps_candidates_and_last_points_to_final(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def callbacks():
                return [OptimizerStepModelCheckpoint(
                    steps=[3, 6, 9, 12], dirpath=root, filename="candidate-{step:06d}", save_top_k=-1,
                    save_last="link", auto_insert_metric_name=False), ModelCheckpoint(
                    dirpath=root, filename="recovery-{step:06d}", every_n_train_steps=4, save_top_k=1,
                    save_last=False, save_on_train_epoch_end=False, auto_insert_metric_name=False)]

            def trainer(steps):
                return pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, max_steps=steps,
                                  accumulate_grad_batches=2, logger=False, callbacks=callbacks(),
                                  enable_progress_bar=False, enable_model_summary=False,
                                  num_sanity_val_steps=0, val_check_interval=0.5)

            data = DataLoader(TensorDataset(torch.ones(11, 1)), batch_size=1)
            first = trainer(4)
            first.fit(SmallTrainingModel(), data, data)
            self.assertTrue((root / "candidate-000003.ckpt").is_file())
            self.assertTrue((root / "recovery-000004.ckpt").is_file())
            self.assertEqual((root / "last.ckpt").resolve().name, "candidate-000003.ckpt")
            continued = trainer(12)
            continued.fit(SmallTrainingModel(), data, data, ckpt_path=root / "recovery-000004.ckpt")
            self.assertEqual(continued.global_step, 12)
            self.assertEqual(sorted(path.name for path in root.glob("candidate-*.ckpt")), [
                "candidate-000003.ckpt", "candidate-000006.ckpt", "candidate-000009.ckpt", "candidate-000012.ckpt"])
            self.assertEqual((root / "last.ckpt").resolve().name, "candidate-000012.ckpt")
            checkpoint = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 12)
            self.assertTrue(checkpoint["optimizer_states"][0]["state"])


if __name__ == "__main__":
    unittest.main()
