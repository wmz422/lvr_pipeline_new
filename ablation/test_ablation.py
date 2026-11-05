"""CPU contract tests; do not claim GPU/model inference coverage."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, MODES, bind_json, digest, item_path, latent_source, model_item,
                    qa_identity, required_latent_rows, select_rows, validate_donors,
                    validate_family, write_json)
from run import read_result


def records():
    return [{"id": f"row-{i}", "group_id": f"group-{i}", "benchmark": "HRBench4K",
             "question": "Question with options", "answer": "SECRET", "scoring": {"solution": "SECRET"},
             "images": [{"path": f"original-{i}.png", "sha256": str(i), "width": 100 + i, "height": 100}],
             "helper_images": [{"path": f"helper-{i}.png"}]} for i in range(4)]


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.rows = records()
        self.donors = {row["id"]: {"id": self.rows[(i + 1) % 4]["id"], "scope": "global", "seed": 42}
                       for i, row in enumerate(self.rows)}

    def test_both_shuffles_use_same_global_donor_and_keep_latent_kind(self):
        validate_donors(self.rows, self.donors)
        target = self.rows[0]["id"]
        self.assertEqual(latent_source("teacher", target, self.donors), (target, "teacher"))
        self.assertEqual(latent_source("predicted", target, self.donors), (target, "predicted"))
        self.assertEqual(latent_source("teacher_shuffled", target, self.donors), ("row-1", "teacher"))
        self.assertEqual(latent_source("predicted_shuffled", target, self.donors), ("row-1", "predicted"))
        self.assertEqual(latent_source("zero", target, self.donors), (None, None))

    def test_bad_permutations_are_rejected(self):
        self.donors["row-0"]["id"] = "row-0"
        with self.assertRaises(ValueError):
            validate_donors(self.rows, self.donors)
        self.setUp()
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "same-problem-group"):
            validate_donors(self.rows, self.donors)

    def test_answers_states_and_helpers_do_not_enter_student(self):
        with tempfile.TemporaryDirectory() as directory:
            student = model_item(self.rows[0], directory)
            teacher = model_item(self.rows[0], directory, teacher=True)
        self.assertEqual(student["answer"], "")
        self.assertNotIn("SECRET", json.dumps(student))
        self.assertNotIn("SECRET", json.dumps(teacher))
        self.assertNotIn("auxiliary_images", student)
        self.assertEqual(len(teacher["auxiliary_images"]), 1)
        self.assertEqual(student["images"], teacher["images"])

    def test_cache_covers_target_and_donor_even_for_small_smoke(self):
        selected = [self.rows[0]]
        cache = required_latent_rows(self.rows, selected, self.donors)
        self.assertEqual([row["id"] for row in cache], ["row-0", "row-1"])

    def test_changing_gpu_count_preserves_artifact_ids_and_complete_coverage(self):
        wanted = {(mode, row["id"]) for row in self.rows for mode in MODES}
        completed = {("teacher", "row-0"), ("zero", "row-3")}
        for workers in (1, 2, 3, 8):
            jobs = [(mode, row["id"]) for rank in range(workers)
                    for row in self.rows[rank::workers] for mode in MODES if (mode, row["id"]) not in completed]
            self.assertEqual(len(jobs), len(set(jobs)))
            self.assertEqual(set(jobs) | completed, wanted)
            for mode, identity in jobs:
                self.assertNotIn(str(workers) + "-shard", str(item_path("run", "answer", identity, mode)))

    def test_foreign_and_torn_predictions_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.json"
            self.assertIsNone(read_result(path, "row-0", "teacher", "binding"))
            path.write_text('{"id":')
            with self.assertRaises(ValueError):
                read_result(path, "row-0", "teacher", "binding")
            write_json(path, {"id": "other", "mode": "teacher", "run_sha256": "binding", "prediction": "A"})
            with self.assertRaisesRegex(ValueError, "Foreign"):
                read_result(path, "row-0", "teacher", "binding")

    def test_run_binding_rejects_changed_weights_and_allows_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            config = {"student": {"sha256": "a"}, "modes": list(MODES)}
            bind_json(path, config)
            bind_json(path, config)
            config["student"]["sha256"] = "b"
            with self.assertRaises(ValueError):
                bind_json(path, config)

    def test_base_family_checks_config_rather_than_directory_name(self):
        with tempfile.TemporaryDirectory(prefix="unrelated-name-") as directory:
            root = Path(directory)
            for family, model_type, width in (("qwen3-4b", "qwen3_vl", 2560), ("qwen25-7b", "qwen2_5_vl", 3584)):
                write_json(root / "config.json", {"model_type": model_type, "text_config": {"hidden_size": width}})
                validate_family(root, family)
                other = "qwen25-7b" if family == "qwen3-4b" else "qwen3-4b"
                with self.assertRaises(ValueError):
                    validate_family(root, other)

    def test_recipient_identity_ignores_answers_and_helpers(self):
        before = qa_identity(self.rows[0])
        self.rows[0]["answer"] = "changed reference"
        self.rows[0]["helper_images"] = [{"path": "other-helper.png"}]
        self.assertEqual(qa_identity(self.rows[0]), before)
        self.rows[0]["question"] = "changed question"
        self.assertNotEqual(qa_identity(self.rows[0]), before)

    def test_standard_library_cli_does_not_require_torch_for_help(self):
        result = subprocess.run([sys.executable, "-S", str(HERE / "run.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qwen25-7b", result.stdout)
        self.assertIn("--gpus", result.stdout)


if __name__ == "__main__":
    unittest.main()
