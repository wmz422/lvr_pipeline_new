import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from flow_reconstruction.gt import persist_image, digest_file
from reconstruction_experiment.latent_cache import (
    LatentStore, bind_plan, build_plan, encode_shard, finalize, fingerprint, shard_metadata)
from reconstruction_experiment.prepare import write_json
from reconstruction_experiment.data import LatentFlowCollator


def fixture(root):
    root = Path(root)
    rng = np.random.default_rng(42)
    images = [persist_image(Image.fromarray(rng.integers(0, 256, (32, width, 3), dtype=np.uint8)),
                            root, "images") for width in (32, 48, 32, 48)]
    protocol = {"fixture": "deterministic_aligned_rgb"}
    protocol["protocol_sha256"] = fingerprint(protocol)
    write_json(root / "protocol.json", protocol)
    status = {"status": "complete", "protocol_sha256": protocol["protocol_sha256"], "splits": {}}
    rows = {}
    for split, indices in (("train", [0, 1, 2]), ("val", [3])):
        rows[split] = [{"id": f"{split}-{i}", "target": images[i], "references": [images[0], images[1]]}
                       for i in indices]
        path = root / (split + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[split]))
        status["splits"][split] = {"records": len(indices), "manifest_sha256": digest_file(path)}
    write_json(root / "status.json", status)
    return rows


class ProbeEncoder:
    vae_batch_size = 2
    last_stats = {}
    def encode_many(self, images):
        return [torch.nn.functional.avg_pool2d(image.float().div(127.5).sub(1).mean(0, keepdim=True)
                    .unsqueeze(0), 16).repeat(1, 128, 1, 1) for image in images]


class LatentCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt, self.cache = self.root / "gt", self.root / "cache"
        self.rows = fixture(self.gt)
        self.identity = {"fake_encoder_for_storage_test": True}
        patcher = patch("reconstruction_experiment.latent_cache.vae_identity", return_value=self.identity)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.plan = build_plan(self.gt, self.root / "model", batch_size=2, shard_bytes=3000, device_type="cpu")
        bind_plan(self.cache, self.plan)

    def complete(self):
        for index in range(len(self.plan["shards"])):
            encode_shard(self.cache, self.plan, index, ProbeEncoder())
        finalize(self.cache)

    def test_dedup_four_worker_partition_resume_and_exact_disk_roundtrip(self):
        self.assertEqual(self.plan["unique_images"], 4)
        self.assertEqual(self.plan["image_occurrences"], 12)
        self.assertEqual(len(self.plan["shards"]), 4)
        partitions = [list(range(rank, len(self.plan["shards"]), 4)) for rank in range(4)]
        self.assertEqual(sorted(i for part in partitions for i in part), list(range(4)))
        encode_shard(self.cache, self.plan, 0, ProbeEncoder())
        first = self.cache / self.plan["shards"][0]["file"]
        mtime = first.stat().st_mtime_ns
        # Half-written, uncommitted output is rebuilt, while completed shards survive.
        partial = self.cache / self.plan["shards"][1]["file"]
        partial.with_suffix(".partial").write_bytes(b"interrupted")
        with self.assertRaisesRegex(ValueError, "Missing"):
            finalize(self.cache)
        self.complete()
        self.assertEqual(first.stat().st_mtime_ns, mtime)
        store = LatentStore(self.cache, expected_encoder=self.identity)
        for shard in self.plan["shards"]:
            for item in shard["items"]:
                from reconstruction_experiment.data import read_rgb8
                expected = ProbeEncoder().encode_many([read_rgb8(self.gt / item["path"])])[0]
                self.assertTrue(torch.equal(store.read(item["key"]), expected))
        restored = pickle.loads(pickle.dumps(store))
        self.assertEqual(len(restored._maps), 0)
        key = next(iter(store.tensors))
        self.assertTrue(torch.equal(store.read(key), restored.read(key)))

    def test_collator_reads_latents_without_flow_png_decode(self):
        self.complete()
        store = LatentStore(self.cache)
        index = {row["id"]: row for row in self.rows["train"]}
        def base(examples):
            return {"ids": [row["id"] for row in examples], "qa": {"input_ids": torch.tensor([[1, 2]])}}
        collator = LatentFlowCollator(base, self.gt, index, store)
        with patch("reconstruction_experiment.data.read_rgb8", side_effect=AssertionError("Online RGB read")):
            batch = collator([self.rows["train"][0]])
        self.assertEqual(batch["flow"]["targets"][0].dtype, torch.float32)
        self.assertEqual(len(batch["flow"]["references"][0]), 2)
        self.assertTrue(torch.equal(batch["qa"]["input_ids"], torch.tensor([[1, 2]])))

    def test_corruption_and_different_vae_are_rejected(self):
        self.complete()
        with self.assertRaisesRegex(ValueError, "different weights"):
            LatentStore(self.cache, expected_encoder={"different": True})
        store = LatentStore(self.cache)
        item = self.plan["shards"][0]["items"][0]
        path = self.cache / self.plan["shards"][0]["file"]
        with path.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "Corrupt cached latent"):
            store.read(item["key"])
        with self.assertRaisesRegex(ValueError, "Corrupt cache shard"):
            shard_metadata(self.cache, self.plan, 0)

    def test_reduced_precision_encoding_and_mixed_plans_are_rejected(self):
        class HalfEncoder(ProbeEncoder):
            def encode_many(self, images):
                return [value.half() for value in super().encode_many(images)]
        with self.assertRaisesRegex(ValueError, "FP32"):
            encode_shard(self.cache, self.plan, 0, HalfEncoder())
        other = build_plan(self.gt, self.root / "model", batch_size=1, shard_bytes=3000, device_type="cpu")
        with self.assertRaisesRegex(ValueError, "settings changed"):
            bind_plan(self.cache, other)

    def test_deepspeed_precision_preserves_cache_without_changing_qa_conversion(self):
        from lightning.pytorch.plugins.precision import DeepSpeedPrecision
        from reconstruction_experiment.precision import FlowDeepSpeedPrecision
        from reconstruction_experiment.latent_cache import REPRESENTATION
        latent = torch.linspace(-1.1, 1.1, 128 * 4).reshape(1, 128, 2, 2)
        batch = {"qa": {"pixel_values": torch.randn(2, 16), "input_ids": torch.tensor([[1, 2]])},
                 "flow": {"representation": REPRESENTATION, "targets": [latent], "references": [[latent]]}}
        original = DeepSpeedPrecision("bf16-mixed").convert_input(batch)
        converted = FlowDeepSpeedPrecision().convert_input(batch)
        self.assertEqual(original["flow"]["targets"][0].dtype, torch.bfloat16)
        self.assertFalse(torch.equal(original["flow"]["targets"][0].float(), latent))
        self.assertTrue(torch.equal(converted["flow"]["targets"][0], latent))
        self.assertEqual(converted["flow"]["targets"][0].dtype, torch.float32)
        for key in batch["qa"]:
            self.assertTrue(torch.equal(converted["qa"][key], original["qa"][key]))
        # Exercise Lightning's actual plugin/precision selection without GPU allocation.
        from lightning.pytorch import Trainer
        trainer = Trainer(accelerator="cpu", devices=1, precision=None,
                          plugins=[FlowDeepSpeedPrecision()], logger=False, enable_checkpointing=False)
        self.assertIsInstance(trainer.precision_plugin, FlowDeepSpeedPrecision)
        self.assertEqual(trainer.precision, "bf16-mixed")


if __name__ == "__main__":
    unittest.main()
