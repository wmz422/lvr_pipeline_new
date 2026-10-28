import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lvr.data.crop_qa import CropQADataModule
from lvr.data.length_manifest import (load_plain_qa_lengths, plain_qa_items_fingerprint,
                                    processor_fingerprint, select_sample_items)


class ItemsDataset:
    def __init__(self, items):
        self.items = copy.deepcopy(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PlainQALengthManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
            (self.root / name).write_text("{}")
        self.items = [{"id": str(i), "question": "Q", "answer": "A", "question_image": f"{i}.png",
                       "boxes": [[0, 0, 10, 10]], "description": "unused"} for i in range(16)]
        self.manifest = {"format": "lvr-plain-qa-lengths-v1", "records": len(self.items),
                         "original_pixels": 2000000,
                         "processor_files_sha256": processor_fingerprint(self.root),
                         "items_sha256": plain_qa_items_fingerprint(self.items),
                         "lengths": list(range(100, 116))}
        self.path = self.root / "lengths.json"
        self.path.write_text(json.dumps(self.manifest))

    def load(self, **kwargs):
        values = dict(path=self.path, items=self.items, processor_directory=self.root,
                      original_pixels=2000000, max_length=8192)
        return load_plain_qa_lengths(**(values | kwargs))

    def test_exact_binding_accepts_full_dataset(self):
        self.assertEqual(self.load(), list(range(100, 116)))

    def test_changed_text_order_resolution_and_processor_are_rejected(self):
        changed = copy.deepcopy(self.items)
        changed[0]["question"] = "changed question"
        for kwargs in ({"items": changed}, {"items": self.items[::-1]}, {"original_pixels": 1000000}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.load(**kwargs)
        (self.root / "preprocessor_config.json").write_text('{"min_pixels": 999}')
        with self.assertRaisesRegex(ValueError, "processor_files"):
            self.load()

    def test_partial_invalid_and_over_budget_lengths_are_rejected(self):
        for lengths in ([100], [True] * 16, [-1] * 16, [9000] * 16):
            with self.subTest(lengths=lengths[:2]), self.assertRaises(ValueError):
                self.path.write_text(json.dumps(self.manifest | {"lengths": lengths}))
                self.load()

    def test_stress_order_repeats_real_samples_and_rejects_unknown_ids(self):
        path = self.root / "stress.json"
        path.write_text(json.dumps({"sample_ids": ["1", "0", "1"]}))
        self.assertEqual([row["id"] for row in select_sample_items(path, self.items)], ["1", "0", "1"])
        for ids in ([], ["missing"], [False]):
            path.write_text(json.dumps({"sample_ids": ids}))
            with self.assertRaises(ValueError):
                select_sample_items(path, self.items)

    def test_training_branches_cannot_reuse_plain_lengths(self):
        for kwargs in ({"num_latents": 2, "include_encoder": True}, {"qa_crops": True},
                       {"train_sample_ids_path": "stress.json"}):
            values = dict(processor_name_or_path=str(self.root), image_root=str(self.root),
                          num_latents=0, include_encoder=False, train_length_manifest_path=str(self.path))
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CropQADataModule(**(values | kwargs))

    def module(self, **kwargs):
        return CropQADataModule(str(self.root), str(self.root), batch_size=2, num_workers=0,
                                num_latents=0, include_encoder=False, original_pixels=2000000, **kwargs)

    def setup_module(self, module):
        tokenizer = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: 1)
        processor = SimpleNamespace(tokenizer=tokenizer)
        with patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                patch("lvr.data.crop_qa.cap_qwen_pixels"), \
                patch("lvr.data.crop_qa.CropQADataset", side_effect=[ItemsDataset(self.items), ItemsDataset(self.items)]):
            module.setup("fit")

    def test_datamodule_uses_bound_lengths_and_keeps_complete_sample_coverage(self):
        module = self.module(train_length_manifest_path=str(self.path))
        self.setup_module(module)
        assigned = []
        for rank in range(8):
            module.trainer = SimpleNamespace(world_size=8, global_rank=rank)
            loader = module.train_dataloader()
            self.assertEqual(loader.sampler.lengths, list(range(100, 116)))
            assigned.extend(list(loader.sampler))
        self.assertEqual(sorted(assigned), list(range(16)))
        self.assertEqual(len(module.val_dataset), 16)

    def test_stress_datamodule_distributes_exact_rank_order_without_shuffle(self):
        path = self.root / "stress.json"
        ids = [str(i % 16) for i in range(128)]
        path.write_text(json.dumps({"sample_ids": ids}))
        module = self.module(train_sample_ids_path=str(path))
        self.setup_module(module)
        self.assertEqual([item["id"] for item in module.train_dataset.items], ids)
        for rank in range(8):
            module.trainer = SimpleNamespace(world_size=8, global_rank=rank)
            sampler = module.train_dataloader().sampler
            self.assertFalse(sampler.shuffle)
            self.assertEqual([module.train_dataset[i]["id"] for i in sampler], ids[rank::8])
            sampler.set_epoch(3)
            self.assertEqual([module.train_dataset[i]["id"] for i in sampler], ids[rank::8])
        self.assertEqual(len(module.val_dataset), 16)


if __name__ == "__main__":
    unittest.main()
