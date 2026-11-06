"""Metric correctness, native observer integrity, coverage, and cache provenance."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Use the inference Python for latent similarity tests")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import digest, write_json
from download_models import check_file, verify_asset
from monet_observer import observed_source
from similarity_cache import read_trace, save_trace, trace_path
from similarity_models import DIRECT_FOOTER, messages, teacher_text
from similarity_stats import cosine_curve, independent_groups, load_cohort


class SimilarityTests(unittest.TestCase):
    def test_linear_formula_matches_naive_pairs_for_all_scopes(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(11, 8, 17))
        groups = [0, 0, 1, 2, 3, 4, 4, 5, 6, 6, 7]
        tasks = ["a"] * 5 + ["b"] * 6
        unit = values / np.linalg.norm(values, axis=-1, keepdims=True)
        for scope in ("within-benchmark", "within-task", "cross-task"):
            expected = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if groups[i] == groups[j]:
                        continue
                    if scope == "within-task" and tasks[i] != tasks[j]:
                        continue
                    if scope == "cross-task" and tasks[i] == tasks[j]:
                        continue
                    expected.append(np.einsum("sd,sd->s", unit[i], unit[j]))
            curve, pairs = cosine_curve(values, groups, tasks, scope)
            self.assertEqual(pairs, len(expected))
            np.testing.assert_allclose(curve, np.mean(expected, axis=0), atol=1e-14)

    def test_identical_vectors_are_one_and_opposites_are_minus_one(self):
        values = np.ones((3, 8, 5))
        curve, pairs = cosine_curve(values, [0, 1, 2], ["a"] * 3)
        np.testing.assert_allclose(curve, 1)
        self.assertEqual(pairs, 3)
        curve, pairs = cosine_curve(np.stack((values[0], -values[0])), [0, 1], ["a"] * 2)
        np.testing.assert_allclose(curve, -1)
        self.assertEqual(pairs, 1)

    def test_no_pair_is_nan_and_zero_norm_is_rejected(self):
        curve, pairs = cosine_curve(np.ones((2, 8, 5)), [0, 0], ["a", "a"])
        self.assertEqual(pairs, 0)
        self.assertTrue(np.isnan(curve).all())
        with self.assertRaises(ValueError):
            cosine_curve(np.zeros((2, 8, 5)), [0, 1], ["a", "a"])

    def test_group_exclusion_is_transitive_across_problem_and_image(self):
        rows = [{"group_id": group, "images": [{"sha256": im}]} for group, im in
                [("a", "i1"), ("a", "i2"), ("b", "i2"), ("c", "i3")]]
        groups = independent_groups(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[-1])

    def test_cache_records_missing_native_positions_without_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.safetensors"
            for size in (0, 3, 8):
                save_trace(path, torch.ones(size, 13), "case", "binding", {"native_positions": size})
                value, record = read_trace(path, "case", "binding")
                self.assertEqual(value.shape, (size, 13))
                self.assertEqual(record["observed_positions"], size)
            with self.assertRaises(ValueError):
                read_trace(path, "different-case", "binding")
            with self.assertRaises(ValueError):
                read_trace(path, "case", "different-run")
            for bad in (torch.zeros(8, 13), torch.full((8, 13), float("nan"))):
                with self.assertRaises(ValueError):
                    save_trace(path, bad, "case", "binding", {"native_positions": 8})

    def test_corrupted_finite_tensor_is_rejected(self):
        from safetensors import safe_open
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.safetensors"
            save_trace(path, torch.ones(8, 3), "case", "binding", {"native_positions": 8})
            with safe_open(str(path), framework="pt") as stream:
                metadata = stream.metadata()
            save_file({"latent": torch.full((8, 3), 2.)}, str(path), metadata=metadata)
            with self.assertRaises(ValueError):
                read_trace(path, "case", "binding")

    def test_cohort_uses_all_six_branches_and_refuses_incomplete_or_smoke(self):
        rows = [{"id": f"case{i}", "benchmark": "MIRA"} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for method in ("monet", "lvr", "ours"):
                for kind in ("teacher", "predicted"):
                    config = {"schema": "cross-case-latent-run-v1", "method": method, "kind": kind,
                              "retained_positions": 8, "scope": "smoke", "suite_manifest_sha256": "suite",
                              "selected_ids": [r["id"] for r in rows]}
                    write_json(output / method / kind / "run_config.json", config)
                    for i, row in enumerate(rows):
                        size = 0 if method == "monet" and kind == "predicted" and i == 0 else 8
                        save_trace(trace_path(output, method, kind, row["id"]), torch.ones(size, 5),
                                   row["id"], digest(config), {"native_positions": size})
            with self.assertRaises(ValueError):
                load_cohort(output, rows)
            common, _, coverage, _ = load_cohort(output, rows, allow_smoke=True)
            self.assertEqual([r["id"] for r in common], ["case1", "case2"])
            self.assertEqual(coverage["monet/predicted"]["MIRA"]["positions"], [2] * 8)
            trace_path(output, "ours", "teacher", "case2").unlink()
            with self.assertRaises(ValueError):
                load_cohort(output, rows, allow_smoke=True)

    def test_assets_verify_git_blob_and_reject_modified_download_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            data = b'{"config": 1}'
            path.write_bytes(data)
            spec = {"files": [{"path": "config.json", "bytes": len(data),
                               "git_blob": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}]}
            verify_asset(root, spec)
            verify_asset(root, spec, hash_files=False)
            path.write_bytes(b'{"config": 2}')
            with self.assertRaises(ValueError):
                verify_asset(root, spec, hash_files=False)
            with self.assertRaises(ValueError):
                check_file(path, spec["files"][0])

    def test_native_monet_observer_is_observational_and_keeps_first_eight(self):
        source = '''class GPUModelRunner:
    def __init__(self):
        self.native_flag = True
    def forward(self, req_id, st):
        override_embeds = []
        override_embeds.append(st["pending"])
        return override_embeds
'''
        patched = observed_source(source)
        ast.parse(patched)
        self.assertEqual(patched.count('override_embeds.append(st["pending"])'), 1)
        with self.assertRaises(ValueError):
            observed_source(source + source)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, MONET_ABLATION_TRACE_DIR=directory):
            namespace = {"__file__": "observed_native_runner.py"}
            exec(patched, namespace)
            runner = namespace["GPUModelRunner"]()
            for i in range(10):
                value = torch.full((7,), i + 1., dtype=torch.float32)
                before = value.clone()
                self.assertIs(runner.forward("native_request", {"pending": value})[0], value)
                torch.testing.assert_close(value, before)
            path = Path(directory) / (hashlib.sha256(b"native_request").hexdigest() + ".pt")
            record = torch.load(path, weights_only=True)
            self.assertEqual(record["native_positions"], 10)
            torch.testing.assert_close(record["latent"][:, 0], torch.arange(1., 9.))
            self.assertEqual(len(list(Path(directory).glob("ready_*.json"))), 1)

    def test_teacher_messages_only_include_originals_question_and_helpers(self):
        row = {"question": "question", "images": [{"path": "original.jpg"}],
               "helper_images": [{"path": "helper.jpg"}], "answer": "SECRET", "metadata": {"state": "SECRET"}}
        before = copy.deepcopy(row)
        chat = messages(row, Path("/benchmark"), teacher=True, monet=True)
        self.assertNotIn("SECRET", json.dumps(chat))
        self.assertEqual(row, before)

        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "user original<|im_start|>assistant\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>"
        text = teacher_text(Processor(), chat)
        self.assertEqual(text.count("<abs_vis_token_pad>"), 8)
        self.assertEqual(text.count("<|image_pad|>"), 1)

    def test_baseline_native_prompt_removes_only_shared_answer_footer(self):
        row = {"question": "Choose color. A. red B. blue\nAnswer with only the option letter.\n" + DIRECT_FOOTER,
               "images": [{"path": "original.jpg"}]}
        lvr = messages(row, Path("/benchmark"))[-1]["content"][-1]["text"]
        monet = messages(row, Path("/benchmark"), monet=True)[-1]["content"][-1]["text"]
        self.assertNotIn("<answer>", lvr)
        self.assertTrue(lvr.endswith("Answer with only the option letter."))
        self.assertEqual(monet, "Choose color. A. red B. blue")


if __name__ == "__main__":
    unittest.main()
