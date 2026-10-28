"""Regression checks for format repair without answer selection or completion."""
import copy
import importlib.util
from pathlib import Path
import unittest

from lvr.eval.answer_extraction import extract_structured_answer
from lvr.eval.reasoning_scoring import score

spec = importlib.util.spec_from_file_location("offline_rescore", Path(__file__).resolve().parents[1] /
                                             "scripts/reasoning_benchmarks/rescore.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class AnswerExtractionTests(unittest.TestCase):
    def scored(self, row, prediction, **kwargs):
        return score(row, prediction, answer_policy="structured_v1", **kwargs)

    def test_vsp_wrapper_changes_no_actions_and_default_is_frozen(self):
        row = {"answer": "D,R", "scoring": {"kind": "grid", "grid": [[1, 0], [0, 2]]}}
        for prediction in [r"<answer>\boxed{R,D}", "<answer>R,D", "<answer>R,D</answer>", r"\boxed{R,D}", "R,D"]:
            result = self.scored(row, prediction)
            self.assertTrue(result["correct"], prediction)
            self.assertEqual(result["actions"], 2)
        self.assertFalse(score(row, r"<answer>\boxed{R,D}")["parsed"])
        self.assertFalse(self.scored(row, "<answer>R")["correct"])

    def test_syntax_repair_does_not_fix_an_illegal_path(self):
        row = {"answer": "D,R", "scoring": {"kind": "grid", "grid": [[1, -1], [0, 2]]}}
        result = self.scored(row, r"<answer>\boxed{R,D}")
        self.assertTrue(result["parsed"])
        self.assertFalse(result["legal"])
        self.assertFalse(result["correct"])

    def test_hinge_missing_rotations_are_not_supplied(self):
        row = {"answer": "A 90, B 180", "scoring": {"kind": "hinge"}}
        self.assertTrue(self.scored(row, "<answer>B 180, A 90")["correct"])
        for prediction in ["<answer>A 90", "<answer>A 90, B 90"]:
            self.assertFalse(self.scored(row, prediction)["correct"])
        self.assertFalse(self.scored(row, "<answer>A 90, A 180")["parsed"])

    def test_finite_numbers_keep_units_and_values(self):
        row = {"answer": "8", "scoring": {"kind": "mathvista", "question_type": "free_form", "answer_type": "integer"}}
        for prediction in ["<answer>8", "<answer>8.0", r"<answer>\boxed{8}", "Answer: <answer>8"]:
            self.assertTrue(self.scored(row, prediction)["correct"])
        for prediction in ["<answer>9", "<answer>8 m", "<answer>8 or 9", "<answer>nan", "<answer>inf"]:
            self.assertFalse(self.scored(row, prediction)["correct"])

    def test_mathvista_never_fuzzy_matches_ambiguous_or_invalid_choices(self):
        row = {"answer": "blue", "options": ["blue", "red"], "scoring": {
            "kind": "mathvista", "question_type": "multi_choice", "answer_type": "text"}}
        for prediction in ["<answer>A", "<answer>(A) blue", "<answer>blue", "option A"]:
            self.assertTrue(self.scored(row, prediction)["correct"])
        for prediction in ["A or B", "<answer>Z", "<answer>(A) red", "<answer>Perhaps blue is right", "(A) blue (B) red"]:
            self.assertFalse(self.scored(row, prediction)["parsed"], prediction)

    def test_currency_amount_is_preserved_without_dropping_other_content(self):
        row = {"answer": "24", "scoring": {"kind": "mathvista", "question_type": "free_form", "answer_type": "integer"}}
        for prediction in ["$24", "<answer>$24", "$24$", "<answer>$24</answer>"]:
            self.assertTrue(self.scored(row, prediction)["correct"])
        for prediction in ["$25", "$24 or $25", "$24 plus tax"]:
            self.assertFalse(self.scored(row, prediction)["correct"])

    def test_incomplete_reasoning_boxes_and_multiple_answers_are_rejected(self):
        for prediction in ["<think>A", "</think>A", "<answer>A</answer><answer>B</answer>",
                           r"\boxed{A} \boxed{B}", r"<answer>\boxed{A", "<answer>A</answer> or B",
                           "<answer>A<answer>B", "```text\nA", "<answer>A or B", r"\boxed{A} explanation"]:
            self.assertIsNotNone(extract_structured_answer(prediction)["failure_reason"], prediction)
        self.assertEqual(extract_structured_answer(r"<think>B</think><answer>\boxed{A}")["text"], "A")

    def test_nested_boxes_preserve_complete_payload(self):
        self.assertEqual(extract_structured_answer(r"<answer>\boxed{[1, {2}]}")["text"], "[1, {2}]")
        self.assertTrue(extract_structured_answer(r"<answer>\boxed{[1, {2}]")["failure_reason"])

    def test_generation_limit_is_not_repaired(self):
        row = {"answer": "A", "options": ["one", "two"], "scoring": {"kind": "choice"}}
        for prediction in ["A", "<answer>A", "<answer>A</answer>"]:
            result = self.scored(row, prediction, truncated=True)
            self.assertFalse(result["correct"])
            self.assertEqual(result["failure_reason"], "generation_token_limit")

    def test_extraction_and_parse_decision_are_reference_independent(self):
        a = {"answer": "A 90, B 180", "scoring": {"kind": "hinge"}}
        b = copy.deepcopy(a); b["answer"] = "A 180, B 90"
        x, y = self.scored(a, "<answer>A 90, B 180"), self.scored(b, "<answer>A 90, B 180")
        for key in ["extracted", "parsed", "extraction_source"]:
            self.assertEqual(x[key], y[key])
        self.assertTrue(x["correct"]); self.assertFalse(y["correct"])

    def test_mira_policy_stays_explicit_and_incomplete_fields_stay_wrong(self):
        row = {"task": "rolling_dice_two", "answer": "black, 24", "scoring": {
            "kind": "mira_pair", "answer1": "black", "answer2": 24}}
        self.assertFalse(self.scored(row, "<answer>black, 24")["correct"])
        self.assertTrue(self.scored(row, "<answer>black, 24", mira_policy="task_normalized_v1")["correct"])
        self.assertFalse(self.scored(row, "<answer>black", mira_policy="task_normalized_v1")["parsed"])

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError):
            score({"answer": "A", "scoring": {"kind": "choice"}}, "A", answer_policy="unknown")

    def test_offline_rescoring_preserves_output_and_checks_frozen_scores(self):
        row = {"id": "choice/1", "benchmark": "VisuLogic", "task": "choice", "answer": "A",
               "options": ["one", "two"], "scoring": {"kind": "choice"}}
        prediction = {"id": row["id"], "benchmark": row["benchmark"], "mode": "plain",
                      "prediction": "<answer>A", **score(row, "<answer>A")}
        result = offline.rescore_rows([row], [prediction])[0]
        self.assertFalse(result["strict"]["correct"])
        self.assertTrue(result["rescored"]["correct"])
        self.assertEqual(result["prediction"], prediction["prediction"])
        changed = dict(prediction, correct=True)
        with self.assertRaisesRegex(ValueError, "Frozen score differs"):
            offline.rescore_rows([row], [changed])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            offline.rescore_rows([row], [prediction, prediction])
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            offline.rescore_rows([row], [])


if __name__ == "__main__":
    unittest.main()
