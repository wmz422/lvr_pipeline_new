"""Check that ablations preserve source, sample identity and scoring denominators."""

import unittest

import torch

from lvr.eval.bottleneck import sample_latents, score_prediction, shuffled_donors, source, summarize


class BottleneckEvaluationTests(unittest.TestCase):
    def test_relocation_preserves_recorded_ids_and_shuffle_donors(self):
        import json
        import tempfile
        from pathlib import Path
        from lvr.eval.latent_prediction import shuffled_sources
        from lvr.eval.vlmeval import build_benchmark_sample, load_annotations

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {str(i): {"benchmark": "example", "index": str(i), "sample_id": str(i),
                             "qa_prompt": "Question\nAnswer with only the option letter.",
                             "images": [{"path": f"images/example/{i}.png", "boxes": [[0, 0, 4, 4]]}],
                             "annotation_source": "test", "source_path": str(i)} for i in range(4)}
            manifest = root / "annotations.json"
            manifest.write_text(json.dumps({"samples": rows}))
            mappings = []
            for location in (root / "first", root / "second"):
                annotations, lookup = load_annotations(manifest, location)
                mappings.append(shuffled_sources(annotations))
                for identity, row in annotations.items():
                    message = [{"type": "image", "value": row["images"][0]["path"]},
                               {"type": "text", "value": "Question"}]
                    sample, _ = build_benchmark_sample(message, "example", lookup)
                    self.assertEqual(sample["id"], identity)
                    self.assertTrue(Path(sample["images"][0]["path"]).is_relative_to(location))
            self.assertEqual(mappings[0], mappings[1])

    def test_prediction_shuffling_is_bijective_and_stays_in_benchmark(self):
        from lvr.eval.latent_prediction import shuffled_sources

        rows = {f"{group}-{i}": {"benchmark": group} for group in ("a", "b") for i in range(8)}
        mapping = shuffled_sources(rows)
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertEqual(mapping, shuffled_sources(dict(reversed(list(rows.items())))))
        for identity, donor in mapping.items():
            self.assertNotEqual(identity, donor)
            self.assertEqual(rows[identity]["benchmark"], rows[donor]["benchmark"])

    def test_predicted_adapter_accepts_only_original_question_inputs(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from lvr.eval.latent_prediction import PredictedLatentModel

        adapter = object.__new__(PredictedLatentModel)
        adapter.mode = "predicted"
        latent = torch.randn(1, 8, 512)
        adapter.student = SimpleNamespace(predict_latents=Mock(return_value=latent))
        qa = {"sentinel": "original_and_question"}
        self.assertIs(adapter.get_latent({"qa": qa}, "id"), latent)
        adapter.student.predict_latents.assert_called_once_with(qa)
        with self.assertRaisesRegex(AssertionError, "must not construct"):
            adapter.get_latent({"qa": qa, "encoder": {}}, "id")

    def test_shuffling_uses_another_image_from_the_same_source(self):
        items = [{"id": f"{name}-{i}", "question_image": f"images/original/{name}/{i}.jpg"}
                 for name in ("TreeVGR-RL-37K", "Visual_CoT") for i in range(4)]
        donors = shuffled_donors(items, 42)
        by_id = {item["id"]: item for item in items}
        self.assertEqual(set(donors), set(donors.values()))
        for item in items:
            donor = by_id[donors[item["id"]]]
            self.assertNotEqual(item["question_image"], donor["question_image"])
            self.assertEqual(source(item), source(donor))

    def test_sampling_is_invariant_to_batch_order(self):
        mu, logvar = torch.randn(3, 2, 8), torch.randn(3, 2, 8)
        ids = ["a", "b", "c"]
        together = sample_latents(mu, logvar, ids, 42)
        for index in (2, 0, 1):
            alone = sample_latents(mu[index:index + 1], logvar[index:index + 1], [ids[index]], 42)
            self.assertTrue(torch.equal(together[index], alone[0]))

    def test_choice_accuracy_and_open_answer_metrics_remain_separate(self):
        choice = {"question": "What color? (A) red (B) blue", "answer": "B"}
        free = {"question": "What is visible?", "answer": "2 red cars"}
        rows = []
        for mode in ("none", "mu"):
            for sample_id, item, prediction in [("choice", choice, "(B)"), ("free", free, "Two red cars.")]:
                rows.append({"id": sample_id, "source": "source", "mode": mode,
                             "prediction": prediction, "truncated": False, **score_prediction(prediction, item)})
        result = summarize(rows)
        metrics = result["metrics"]["mu"]["all"]
        self.assertEqual(metrics["mcq_correct"], {"n": 1, "mean": 1.0})
        self.assertEqual(metrics["normalized_em"], {"n": 1, "mean": 1.0})
        self.assertEqual(metrics["token_f1"], {"n": 1, "mean": 1.0})
        self.assertEqual(result["deltas_vs_no_latent"]["mu"]["mcq_correct"]["mean"], 0.0)

    def test_reasoning_list_index_is_not_a_multiple_choice_answer(self):
        item = {"question": "How many cars? (A) 3 (B) 4 (C) 5 (D) 6", "answer": "A"}
        prediction = "Let me count the cars carefully.\n1. White car\n2. Red car\n3. Red"
        result = score_prediction(prediction, item)
        self.assertEqual(result["choice"], "")
        self.assertEqual(result["mcq_correct"], 0)


if __name__ == "__main__":
    unittest.main()
