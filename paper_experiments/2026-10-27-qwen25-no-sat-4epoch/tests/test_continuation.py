"""Real CPU resumes cover optimizer/LR state, workers, sample cursor and epoch tails."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
import workflow
from continuation_runtime import (
    AccumulationBatchSampler, RestoreDataCursor, EpochModelCheckpoint,
    HalfEpochModelCheckpoint, original_lr,
)


class Examples(Dataset):
    def __init__(self, count=18):
        self.items = [{"id": f"sample-{index}"} for index in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return {"ids": self.items[index]["id"], "x": torch.tensor([index / 20.]),
                "y": torch.tensor([index / 40. + .2])}


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return torch.nn.functional.mse_loss(self.layer(batch["x"]), batch["y"])

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", torch.nn.functional.mse_loss(self.layer(batch["x"]), batch["y"]), batch_size=len(batch["ids"]))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                     lambda step: original_lr(step, peak=1., horizon=12, warmup=1))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class TrackCursor(RestoreDataCursor):
    def __init__(self, source=None):
        super().__init__(steps_per_epoch=5)
        self.events = []
        self.validation_epochs = []
        self.source = source
        self.state_checked = False

    def on_train_start(self, trainer, module):
        super().on_train_start(trainer, module)
        if self.source:
            saved = torch.load(self.source, map_location="cpu", weights_only=False)
            assert trainer.global_step == saved["global_step"]
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved["state_dict"][key], rtol=0, atol=0)
            actual = trainer.optimizers[0].state_dict()
            expected = saved["optimizer_states"][0]
            assert actual["param_groups"] == expected["param_groups"]
            for key, state in expected["state"].items():
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(actual["state"][key][field], state[field], rtol=0, atol=0)
            assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == saved["lr_schedulers"][0]
            self.state_checked = True

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        super().on_train_batch_start(trainer, module, batch, batch_idx)
        self.events.append((trainer.current_epoch, batch_idx, list(batch["ids"])))

    def on_validation_epoch_end(self, trainer, module):
        self.validation_epochs.append(trainer.current_epoch)


class ContinuationTests(unittest.TestCase):
    def test_real_epoch_sampler_keeps_exact_coverage_and_original_prefix(self):
        # Match the actual rank count and 172,224-record split without decoding images.
        dataset = Examples(workflow.TRAIN_RECORDS)
        complete = []
        for rank in range(4):
            base = DistributedSampler(dataset, num_replicas=4, rank=rank, seed=42)
            sampler = AccumulationBatchSampler(base, 16, 2)
            expected = list(base)
            batches = list(sampler)
            self.assertEqual(len(sampler), 2692)
            self.assertEqual([item for batch in batches for item in batch], expected)
            self.assertEqual([len(batch) for batch in batches[-2:]], [8, 8])
            self.assertEqual([item for batch in batches[:1392] for item in batch], expected[:696*32])
            sampler.resume_epoch, sampler.start_batch = 0, 1392
            self.assertEqual(list(sampler), batches[1392:])
            base.set_epoch(1)
            self.assertEqual(len(list(sampler)), 2692)
            complete.extend(expected)
        self.assertEqual(len(set(complete)), workflow.TRAIN_RECORDS)
        self.assertEqual(len(complete), workflow.TRAIN_RECORDS)

    def test_recipe_preserves_original_lr_and_matches_requested_schedule(self):
        args = SimpleNamespace(num_workers=4, prefetch_factor=1)
        original = workflow.yaml.safe_load((workflow.SOURCE / "configs/plain_sft.yaml").read_text())
        self.assertEqual(workflow.EPOCH_STEPS, [1346, 2692, 4038, 5384])
        self.assertEqual(workflow.HALF_STEPS, [1346, 2019, 2692, 3365, 4038, 4711, 5384])
        for epoch in range(1, 5):
            recipe = workflow.build_recipe(epoch, workflow.SOURCE_CHECKPOINT, args)
            self.assertEqual(recipe["model"], original["model"])
            self.assertEqual(recipe["trainer"]["max_steps"], 1346*epoch)
            self.assertEqual(recipe["trainer"]["max_epochs"], epoch)
            self.assertEqual(recipe["trainer"]["accumulate_grad_batches"], 2)
            self.assertEqual(recipe["trainer"]["devices"], 4)
            self.assertEqual(recipe["data"]["num_workers"], 4)
            self.assertEqual(recipe["data"]["additional_train_path"], original["data"]["additional_train_path"])
            self.assertIn("ckpt_path", recipe)
            self.assertNotIn("init_checkpoint", recipe)
            command = workflow.benchmark_command(epoch)
            self.assertEqual(command[command.index("--suite")+1], str(workflow.SUITE))
            self.assertEqual(command[command.index("--expected-step")+1], str(epoch*1346))
        self.assertAlmostEqual(original_lr(696), 8.903831545224372e-6, places=15)
        for step in (2784, 4038, 5384):
            self.assertAlmostEqual(original_lr(step), 1e-6, places=15)

    def test_workers_full_state_mid_epoch_and_epoch_boundary_resume(self):
        torch.set_num_threads(1)
        dataset = Examples()
        def loader(original=False):
            sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=42)
            if original:
                return DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2,
                                  persistent_workers=True, prefetch_factor=1)
            return DataLoader(dataset, batch_sampler=AccumulationBatchSampler(sampler, 2, 2),
                              num_workers=2, persistent_workers=True, prefetch_factor=1)
        torch.manual_seed(42)
        initial = copy.deepcopy(TinyModel().state_dict())
        def model():
            value = TinyModel()
            value.load_state_dict(initial)
            return value
        def trainer(directory, epochs, max_steps, callbacks):
            return pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, max_steps=max_steps,
                              accumulate_grad_batches=2, num_sanity_val_steps=0,
                              logger=False, enable_progress_bar=False, enable_model_summary=False,
                              use_distributed_sampler=False, default_root_dir=str(directory), callbacks=callbacks)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = model()
            trace = TrackCursor()
            full = trainer(root / "full", 4, 20, [trace,
                EpochModelCheckpoint(5, str(root / "full")),
                HalfEpochModelCheckpoint([3,8,13,18], dirpath=str(root / "full"),
                    filename="half-{step:06d}", save_top_k=-1, auto_insert_metric_name=False)])
            full.fit(reference, train_dataloaders=loader(), val_dataloaders=DataLoader(dataset, batch_size=2))
            self.assertEqual(full.global_step, 20)
            self.assertEqual(len(trace.events), 40)
            self.assertEqual(trace.validation_epochs, [0, 1, 2, 3])
            for epoch in range(4):
                observed = [sample for e, _, ids in trace.events if e==epoch for sample in ids]
                self.assertEqual(len(observed), 18)
                self.assertEqual(set(observed), {r['id'] for r in dataset.items})

            # The original run used the standard loader; its first two updates
            # have the same microbatches as the continuation's aligned loader.
            first = model()
            cut = trainer(root / "source", 1, 2, [pl.callbacks.ModelCheckpoint(
                dirpath=str(root / "source"), filename="source", every_n_train_steps=2, save_top_k=-1)])
            cut.fit(first, train_dataloaders=loader(original=True))
            checkpoint = root / "source/source.ckpt"
            combined = []
            resumed = None
            for epoch in range(1,5):
                resumed = model()
                cursor = TrackCursor(checkpoint)
                target = root / f"stage{epoch}"
                continuation = trainer(target, epoch, epoch*5,
                                       [cursor, EpochModelCheckpoint(5, str(target))])
                continuation.fit(resumed, train_dataloaders=loader(), ckpt_path=checkpoint,
                                 val_dataloaders=DataLoader(dataset, batch_size=2))
                self.assertTrue(cursor.state_checked)
                self.assertEqual(cursor.validation_epochs, [epoch - 1])
                self.assertEqual(continuation.global_step, epoch*5)
                combined.extend(cursor.events)
                checkpoint = target / f"candidate-{epoch*5:06d}.ckpt"
                self.assertTrue(checkpoint.is_file())
            self.assertEqual(combined, trace.events[4:])
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)

            # Restart again from a half-epoch candidate, including prefetched workers.
            checkpoint = root / "full/half-000008.ckpt"
            cursor = TrackCursor(checkpoint)
            interrupted = model()
            recovery = trainer(root / "recovery", 4, 20, [cursor])
            recovery.fit(interrupted, train_dataloaders=loader(), ckpt_path=checkpoint,
                         val_dataloaders=DataLoader(dataset, batch_size=2))
            self.assertEqual(cursor.events, trace.events[16:])
            self.assertEqual(cursor.validation_epochs, [1, 2, 3])
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(interrupted.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
