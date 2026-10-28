import tempfile
from pathlib import Path
import unittest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from flow_reconstruction.sampling import DistributedLayoutSampler
from reconstruction_experiment.data import AccumulationBatchSampler
from reconstruction_experiment.runtime import RestoreDataCursor, UpdateCheckpoint


class Numbers(Dataset):
    def __init__(self, count):
        self.items = [{"id": str(i), "value": i / count} for i in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(rows):
    return {"ids": [row["id"] for row in rows], "x": torch.tensor([row["value"] for row in rows])}


class ToyData(LightningDataModule):
    def train_dataloader(self):
        dataset = Numbers(40)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=42)
        return DataLoader(dataset, batch_sampler=AccumulationBatchSampler(sampler, 4, 4), collate_fn=collate)


class ToyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.extend(batch["ids"])
        return ((self.weight * batch["x"] - 1) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=.01)


class RuntimeTests(unittest.TestCase):
    def test_all_four_ranks_cover_plain_data_without_drops_or_repeats(self):
        count = 172224
        dataset = list(range(count))
        layouts = [(16 * (i % 3 + 1), 32, ((16, 16),)) for i in dataset]
        for teacher, batch, accumulation in ((True, 2, 16), (False, 4, 8)):
            combined = []
            for rank in range(4):
                if teacher:
                    base = DistributedLayoutSampler(dataset, layouts, batch, num_replicas=4, rank=rank)
                else:
                    base = DistributedLengthGroupedSampler(batch_size=batch * 4, num_replicas=4,
                        rank=rank, lengths=[i % 41 + 1 for i in dataset], seed=42)
                sampler = AccumulationBatchSampler(base, batch, accumulation)
                batches = list(sampler)
                flattened = [i for group in batches for i in group]
                self.assertEqual(flattened, list(base))
                self.assertEqual(len(sampler), 1346 * accumulation)
                self.assertTrue(all(len(group) == batch for group in batches[:-accumulation]))
                self.assertTrue(all(len(group) == batch // 2 for group in batches[-accumulation:]))
                combined.extend(flattened)
                sampler.resume_epoch, sampler.start_batch = 0, 400 * accumulation
                self.assertEqual(list(sampler), batches[400 * accumulation:])
            self.assertEqual(sorted(combined), dataset)

    def test_lightning_resume_retains_examples_optimizer_and_short_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(name, steps, checkpoint=None):
                model = ToyModel()
                saver = UpdateCheckpoint(steps=[1, 3], dirpath=str(Path(directory) / name),
                    filename="step-{step}", save_top_k=-1, save_last="link")
                trainer = Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=1,
                    accumulate_grad_batches=4, use_distributed_sampler=False, logger=False,
                    enable_progress_bar=False, enable_model_summary=False,
                    callbacks=[RestoreDataCursor(steps_per_epoch=3), saver])
                trainer.fit(model, datamodule=ToyData(), ckpt_path=checkpoint)
                self.assertEqual(trainer.global_step, steps)
                return model, trainer, saver.last_model_path
            full, full_trainer, _ = run("full", 3)
            first, _, checkpoint = run("first", 1)
            rest, resumed_trainer, _ = run("resumed", 3, checkpoint)
            self.assertEqual(full.seen, first.seen + rest.seen)
            self.assertEqual(len(full.seen), 40)
            self.assertEqual(len(set(full.seen)), 40)
            self.assertTrue(torch.equal(full.weight, rest.weight))
            states = [next(iter(trainer.optimizers[0].state.values())) for trainer in (full_trainer, resumed_trainer)]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(states[0][key], states[1][key]))


if __name__ == "__main__":
    unittest.main()
