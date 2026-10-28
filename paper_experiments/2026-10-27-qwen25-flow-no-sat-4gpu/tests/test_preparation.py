import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from reconstruction_experiment.prepare import prepare, protocol, validate_ready, write_json


class PreparationTests(unittest.TestCase):
    def test_complete_clean_manifests_resume_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32), "red").save(root / "original.png")
            Image.new("RGB", (16, 16), "blue").save(root / "helper.png")
            data = {"image_root": str(root), "original_pixels": 512, "crop_pixels": 256}
            for split, indices in (("train", [1, 2]), ("val", [3])):
                qa = [{"question_image": "original.png", "auxiliary_image": None,
                       "raw": {"dataset": "unit", "index": i}} for i in indices]
                boxes = [{**r, "boxes": [[0, 0, 8, 8]] + ([[16, 16, 24, 24]] if j else [])}
                         for j, r in enumerate(qa)]
                reasoning = [{"id": "reason-" + split, "source": "unit-reasoning",
                    "teacher_view_mode": "auxiliary_images", "problem_images": [{"path": "original.png"}],
                    "auxiliary_images": [{"path": "helper.png", "step": 1}]}]
                for key, rows in ((split + "_path", qa), (split + "_boxes_path", boxes),
                                  ("additional_" + split + "_path", reasoning)):
                    path = root / key
                    path.mkdir()
                    (path / "index.json").write_text(json.dumps(rows))
                    data[key] = str(path)
            # Match the actual no-SAT Plain input: additional train is a file.
            reasoning_file = root / "reasoning_train_without_sat.json"
            reasoning_file.write_bytes((Path(data["additional_train_path"]) / "index.json").read_bytes())
            data["additional_train_path"] = str(reasoning_file)
            settings = protocol(data, root)
            output, counts = root / "prepared", {"train": 3, "val": 2}
            prepare(output, settings, counts, workers=2)
            self.assertTrue(validate_ready(output, settings, counts))
            first = (output / "train.jsonl").read_bytes()
            rows = [json.loads(line) for line in first.splitlines()]
            self.assertEqual([r["id"] for r in rows], ["original.png:1", "original.png:2", "reason-train"])
            self.assertEqual([r["layout"]["kind"] for r in rows],
                             ["single_crop", "masked_original", "horizontal_auxiliary"])
            # Simulate an interrupted manifest write; cached complete rows are reused.
            write_json(output / "status.json", {"status": "failed"})
            partial = output / "train.jsonl.partial"
            partial.write_bytes(b"\n".join(first.splitlines()[:2]) + b'\n{"id":"interrupted')
            (output / "train.jsonl").unlink()
            prepare(output, settings, counts, workers=2)
            self.assertEqual(first, (output / "train.jsonl").read_bytes())
            self.assertTrue(validate_ready(output, settings, counts))
            (output / "train.jsonl").write_bytes(first + b"{}\n")
            with self.assertRaisesRegex(ValueError, "manifest/count"):
                prepare(output, settings, counts, workers=2)


if __name__ == "__main__":
    unittest.main()
