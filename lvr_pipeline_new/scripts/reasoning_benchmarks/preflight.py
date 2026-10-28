"""Replay real Qwen preprocessing for every benchmark QA and helper branch."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT, file_sha, model_item
from download import write_json

DEFAULT_MODEL = Path("/workspace/wmz/.cache/Qwen3-VL-4B-Instruct")
_COLLATORS = None
_ROOT = None
_SEEN = None


def initialize(model, root):
    global _COLLATORS, _ROOT, _SEEN
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoProcessor
    from lvr.data.crop_qa import CropQACollator
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    common = dict(processor=processor, image_root=root, original_pixels=None, crop_pixels=262144,
                  max_length=131072, include_answers=False)
    _COLLATORS = {"student": CropQACollator(**common, include_encoder=False),
                  "plain": CropQACollator(**common, include_encoder=False, num_latents=0),
                  "teacher": CropQACollator(**common, include_encoder=True)}
    _ROOT, _SEEN = Path(root), set()


def one(row):
    import torch
    item = model_item(row, _ROOT)
    batch = _COLLATORS["student"]([item])
    qa = batch["qa"]
    assert "encoder" not in batch and "labels" not in qa
    assert int(qa["slot_mask"].sum()) == 8
    assert int(qa["image_grid_thw"].shape[0]) == len(row["images"])
    assert torch.isfinite(qa["pixel_values"]).all()
    result = {"id": row["id"], "qa_tokens": int(qa["input_ids"].shape[-1]),
              "input_images": len(row["images"]), "helper_images": len(row["helper_images"]),
              "encoder_tokens": None, "isolation_checked": False}
    if row["helper_images"]:
        teacher = _COLLATORS["teacher"]([model_item(row, _ROOT, teacher=True)])
        for key, value in qa.items():
            assert torch.equal(value, teacher["qa"][key]), (row["id"], key)
        assert int(teacher["encoder"]["image_grid_thw"].shape[0]) == len(row["images"])+len(row["helper_images"])
        assert torch.isfinite(teacher["encoder"]["pixel_values"]).all()
        result["encoder_tokens"] = int(teacher["encoder"]["input_ids"].shape[-1])
        del teacher
    key = row["benchmark"], row["task"]
    if key not in _SEEN:
        changed = {**row, "answer": "DO_NOT_LEAK", "scoring": {"secret": "DO_NOT_LEAK"},
                   "helper_images": [{"path": "missing-helper-must-not-be-opened.png"}]}
        adversarial = _COLLATORS["student"]([model_item(changed, _ROOT)])["qa"]
        assert all(torch.equal(value, adversarial[k]) for k, value in qa.items())
        plain = _COLLATORS["plain"]([item])["qa"]
        assert torch.equal(plain["input_ids"], qa["input_ids"][:, :-8])
        assert torch.equal(plain["pixel_values"], qa["pixel_values"])
        result["isolation_checked"] = True
        _SEEN.add(key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.suite/"records.jsonl").read_text().splitlines()]
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.model), str(args.suite))) as pool:
        with (args.suite/"processor_records.jsonl").open("w") as output:
            for r in pool.map(one, records, chunksize=4):
                results.append(r); output.write(json.dumps(r)+"\n")
                if len(results) % 200 == 0:
                    output.flush(); print("PROCESSED", len(results), "/", len(records), flush=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    report = {"status": "passed", "records": len(results), "records_sha256": file_sha(args.suite/"records.jsonl"),
              "processor": processor.image_processor.to_dict(), "model_path": str(args.model),
              "processor_file_hashes": {p.name: file_sha(p) for p in args.model.iterdir()
                                         if p.is_file() and p.suffix in (".json", ".jinja")},
              "original_pixels": None, "helper_pixels": 262144, "max_length": 131072,
              "max_qa_tokens": max(r["qa_tokens"] for r in results),
              "max_encoder_tokens": max(r["encoder_tokens"] or 0 for r in results),
              "student_isolation_probes": sum(r["isolation_checked"] for r in results),
              "teacher_qa_tensor_equal_records": sum(r["encoder_tokens"] is not None for r in results),
              "no_truncation": True}
    write_json(args.suite/"validation_processor.json", report)
    print("COMPLETE", len(results), report["max_qa_tokens"], report["max_encoder_tokens"], flush=True)


if __name__ == "__main__":
    main()
