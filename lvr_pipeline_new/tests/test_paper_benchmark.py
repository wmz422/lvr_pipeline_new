"""Coverage, SAT reversal/scoring, checkpoint dependencies and real image processing."""
import copy
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lvr.eval.paper_benchmark import (DEFAULT_SUITE, EXPERIMENT, ROOT, check_counts, file_sha,
                                      load_suite, model_item, score_record, summarize, write_json)


def script(name):
    spec = importlib.util.spec_from_file_location("paper_" + name, ROOT / f"scripts/paper_benchmarks/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_session_requires_success_and_same_live_process(self):
        module = script("run_after")
        identity = module.process_identity(os.getpid())
        job = {"session_file": str(self.root / "session.json"), "session_identity": {"started_at": "frozen", "launcher_pid": os.getpid()},
               "launcher_process": identity}
        session = {**job["session_identity"], "status": "running"}
        write_json(Path(job["session_file"]), session)
        self.assertEqual(module.dependency_state(job)[0], "waiting_for_experiment")
        with patch.object(module, "process_identity", return_value=None):
            self.assertEqual(module.dependency_state(job)[0], "failed")
        for status, exit_code, expected in (("failed", 1, "failed"), ("complete", 1, "failed"), ("complete", 0, "ready")):
            write_json(Path(job["session_file"]), {**session, "status": status, "exit_code": exit_code})
            self.assertEqual(module.dependency_state(job)[0], expected)
        write_json(Path(job["session_file"]), {**session, "status": "complete", "exit_code": 0, "started_at": "another-run"})
        self.assertEqual(module.dependency_state(job)[0], "failed")

    def test_torn_tail_is_recoverable_but_interior_damage_or_duplicates_are_not(self):
        module = script("evaluate")
        path = self.root / "generations.jsonl"
        path.write_bytes(b'{"id":"first"}\n{"id":')
        self.assertEqual(module.read_journal(path, ["first", "second"]), [{"id": "first"}])
        self.assertEqual(path.read_bytes(), b'{"id":"first"}\n')
        for raw in (b'{"id":"first"}\n{"id":"first"}\n', b'{"id":"foreign"}\n', b'bad\n{"id":"first"}\n'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                module.read_journal(path, ["first"])

    def test_resume_configuration_cannot_change(self):
        module = script("evaluate")
        path = self.root / "run.json"
        module.bind_json(path, {"checkpoint": "A", "suite": "full"})
        module.bind_json(path, {"checkpoint": "A", "suite": "full"})
        with self.assertRaises(ValueError):
            module.bind_json(path, {"checkpoint": "B", "suite": "full"})

    def test_complete_dev_selection_is_rechecked_and_bound_to_export(self):
        module = script("run_after")
        import sys
        sys.path.insert(0, str(ROOT / "scripts/reasoning_data"))
        import validate_plain_sft_candidates as validation
        policy = {"source_weights": {"source": 1.0}, "validation_source_counts": {"source": 1}}
        policy_path = self.root / "policy.json"
        write_json(policy_path, policy)
        steps = [696, 1392, 2088, 2784]
        summaries = []
        exports = {}
        for step, loss in zip(steps, [4.0, 2.0, 2.0, 3.0]):
            source, export = self.root / f"native-{step}.ckpt", self.root / f"export-{step}.ckpt"
            source.write_bytes(b"native")
            export.write_bytes(b"export")
            identity = validation.checkpoint_identity(source)
            write_json(export.with_suffix(".ckpt.source.json"), {"source": identity, "export": validation.checkpoint_identity(export)})
            plan = {"checkpoint": str(export), "checkpoint_step": step, "source_checkpoint_identity": identity,
                    "checkpoint_bytes": export.stat().st_size, "checkpoint_mtime_ns": export.stat().st_mtime_ns,
                    "config_sha256": "fixed", "samples_sha256": "fixed", "samples": 1,
                    "selection_policy_sha256": validation.digest(policy), "code_sha256": {"fixed": "code"}}
            metrics = validation.summarize([{"id": "x", "group": "group", "source": "source", "sample_ce": loss, "target_tokens": 1, "nll_sum": loss}])
            summary = {"status": "complete", "checkpoint_step": step, "plan_sha256": validation.digest(plan),
                       "metrics": metrics, "selection_policy": policy, "selection_policy_sha256": validation.digest(policy), "selection_score": loss}
            write_json(self.root / f"step_{step:06d}/plan.json", plan)
            write_json(self.root / f"step_{step:06d}/summary.json", summary)
            summaries.append(summary)
            exports[step] = export
        selection = validation.select_candidate_summaries(summaries, steps, policy)
        selection["checkpoint"] = str(self.root / "native-1392.ckpt")
        write_json(self.root / "candidate_selection.json", selection)
        job = {"validation_results": str(self.root), "selection_policy": str(policy_path),
               "selection_policy_sha256": file_sha(policy_path), "expected_steps": steps}
        export, step, _ = module.selected_checkpoint(job)
        self.assertEqual((export, step), (exports[1392], 1392))
        # No fallback to the final checkpoint or best available partial subset.
        incomplete = self.root / "step_002784/summary.json"
        saved = incomplete.read_text()
        incomplete.unlink()
        with self.assertRaises(FileNotFoundError):
            module.selected_checkpoint(job)
        incomplete.write_text(saved)
        exports[1392].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            module.selected_checkpoint(job)


class FullProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.protocol, cls.manifest = load_suite()

    def test_complete_official_scope_and_sat_option_orders(self):
        check_counts(self.rows, self.protocol)
        self.assertEqual(len(self.rows), 10107)
        self.assertEqual(len({r["base_id"] for r in self.rows}), 9957)
        self.assertEqual(len(self.protocol["blink_tasks"]), 14)
        sat = [r for r in self.rows if r["benchmark"] == "SAT"]
        self.assertEqual(len(sat), 300)
        self.assertEqual({r["split"] for r in sat}, {"test"})
        pair = sat[:2]
        self.assertNotEqual(pair[0]["answer"], pair[1]["answer"])
        for row in pair:
            self.assertTrue(score_record(row, f"<answer>{row['answer']}</answer>")["correct"])
            result = score_record(row, "<answer>A or B</answer>")
            self.assertFalse(result["correct"])
            self.assertTrue(result["reference_correct"])
        with self.assertRaises(ValueError):
            check_counts(self.rows[:-1], self.protocol)
        mutated = copy.deepcopy(self.rows)
        next(r for r in mutated if r["benchmark"] == "SAT" and r["option_order"] == 1)["answer"] = pair[0]["answer"]
        with self.assertRaisesRegex(ValueError, "same answer"):
            check_counts(mutated, self.protocol)

    def test_blink_uses_task_macro_and_missing_results_cannot_publish(self):
        results = [{"id": r["id"], "correct": r["task"] == "Visual_Correspondence", "parsed": True,
                    "reference_correct": False} for r in self.rows]
        summary = summarize(self.rows, results, self.protocol)
        blink = next(r for r in summary["metrics"] if r["benchmark"] == "BLINK")
        self.assertAlmostEqual(blink["accuracy"], 1 / 14)
        self.assertAlmostEqual(blink["sample_accuracy"], 172 / 1901)
        with self.assertRaises(ValueError):
            summarize(self.rows, results[:-1], self.protocol)

    def test_sat_reference_diagnostic_matches_the_pinned_author_code(self):
        source = DEFAULT_SUITE / "upstream/SAT/lmms_eval/tasks/sat_real/utils.py"
        parsed = ast.parse(source.read_text())
        functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ("sat_process_results", "sat_aggregation")]
        import re
        namespace = {"re": re}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        pair = [r for r in self.rows if r["benchmark"] == "SAT"][:2]
        with contextlib.redirect_stdout(io.StringIO()):
            for row in pair:
                doc = {"question_type": row["task"], "answers": row["options"],
                       "correct_answer": row["options"][ord(row["answer"]) - 65]}
                for prediction in ("A", "B", "a", "b", " A ", "<answer>A</answer>", "<answer>A or B</answer>", "The answer is B", "", "uncertain"):
                    reference = namespace["sat_process_results"](doc, [prediction])["sat_score_overall"]
                    expected = bool(namespace["sat_aggregation"]([reference]))
                    self.assertEqual(score_record(row, prediction)["reference_correct"], expected, prediction)

    def test_direct_prompts_and_generation_allowlist_do_not_include_labels(self):
        suffix = self.protocol["inference"]["answer_instruction"]
        self.assertTrue(all(r["question"].endswith(suffix) for r in self.rows))
        for name in self.protocol["benchmarks"]:
            row = next(r for r in self.rows if r["benchmark"] == name)
            item = model_item(row, DEFAULT_SUITE)
            changed = {**row, "answer": "SECRET_GOLD", "scoring": {"secret": "GOLD"}, "metadata": {"gold": "SECRET"}}
            self.assertEqual(item, model_item(changed, DEFAULT_SUITE))
            self.assertEqual(item["answer"], "")
            self.assertNotIn("helper_images", item)

    def test_qwen25_real_processor_preserves_more_than_training_2mp(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from lvr.data.crop_qa import CropQACollator
        torch.set_num_threads(2)
        model = ROOT.parent / "paper_experiments/models/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        native_size = dict(processor.image_processor.size)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (2800, 1400), "blue").save(image)
            collator = CropQACollator(processor, directory, num_latents=0, original_pixels=None,
                                      include_encoder=False, include_answers=False, max_length=131072)
            batch = collator([{"id": "highres", "question": "Give only the final answer.", "answer": "",
                               "images": [{"path": str(image), "boxes": []}]}])["qa"]
            self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 100, 200]])
            self.assertEqual(dict(processor.image_processor.size), native_size)
            self.assertNotIn("labels", batch)


if __name__ == "__main__":
    unittest.main()
