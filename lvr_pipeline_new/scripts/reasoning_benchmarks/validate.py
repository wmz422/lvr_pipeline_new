"""Independently verify images, reference answers and upstream scorer parity."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image
from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, EXPECTED, ROOT, file_sha, safe_path
from download import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-image-verification", action="store_true")
    args = parser.parse_args()
    root = args.suite
    rows = [json.loads(line) for line in (root/"records.jsonl").read_text().splitlines()]
    assert dict(Counter(r["benchmark"] for r in rows)) == EXPECTED
    assert len({r["id"] for r in rows}) == len(rows)
    unique = {im["sha256"]: im for row in rows for im in row["images"]+row["helper_images"]}
    def check_image(im):
        path = safe_path(root, im["path"])
        if file_sha(path) != im["sha256"]:
            raise ValueError(f"Changed image: {path}")
        with Image.open(path) as image:
            image.load(); rgb = image.convert("RGB")
            assert rgb.size == (im["width"], im["height"])
            pixel = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
            if pixel != im["pixel_sha256"]:
                raise ValueError(f"Pixel hash mismatch: {path}")
    stamp = root/"image_verification.json"
    fingerprint = {"records_sha256": file_sha(root/"records.jsonl"),
                   "assets": {im["path"]: [safe_path(root, im["path"]).stat().st_size, safe_path(root, im["path"]).stat().st_mtime_ns] for im in unique.values()}}
    if not (args.reuse_image_verification and stamp.exists() and json.loads(stamp.read_text()) == fingerprint):
        with ThreadPoolExecutor(args.workers) as pool:
            for i, _ in enumerate(pool.map(check_image, unique.values()), 1):
                if i % 1000 == 0:
                    print("DECODED", i, "/", len(unique), flush=True)
        write_json(stamp, fingerprint)
    # Snapshot and bind the exact upstream execution engine; never alter raw data.
    raw = ROOT/"data/raw/reasoning_expansion/mentis-oculi"
    tree = json.loads((raw/"tree.json").read_text())
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    vendor = root/"upstream/MentisOculi/rushhour"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "generator.py", "evaluate_responses.py"):
        relative = "datasets/rushhour/"+name
        payload = (raw/relative).read_bytes()
        assert hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest() == entries[relative]["sha"]
        (vendor/name).write_bytes(payload)
    gold_input, gold_output = root/"reference_scorer_input.jsonl", root/"reference_scores.jsonl"
    def reference_response(row):
        # Some MathVista option texts are themselves '(a)', '(b)', etc. The
        # official normalizer interprets those as option POSITIONS. Generate the
        # correct displayed position, preserving the original gold option text.
        if row["benchmark"] == "MathVista_testmini" and row["scoring"]["question_type"] == "multi_choice":
            if row["answer"] not in row["options"]:
                raise ValueError(f"Missing reference option: {row['id']}")
            return chr(65 + row["options"].index(row["answer"]))
        return row["answer"]
    gold_input.write_text("".join(json.dumps({"record": r, "prediction": "<answer>"+reference_response(r)+"</answer>"}, ensure_ascii=False)+"\n" for r in rows))
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"src"), MPLBACKEND="Agg", LVR_REASONING_SUITE=str(root.resolve()))
    subprocess.run([str(ROOT/".venv-mentis/bin/python"), "-m", "lvr.eval.reasoning_scoring", str(gold_input), str(gold_output)],
                   env=environment, check=True)
    scores = [json.loads(line) for line in gold_output.read_text().splitlines()]
    failures = [r for r in scores if not r["correct"]]
    # Execute only the relevant pure functions from the pinned official files.
    # This avoids importing API clients or invoking any external model judge.
    import Levenshtein
    functions = {"get_most_similar", "normalize_extracted_answer"}
    source = root/"upstream/MathVista/evaluation/calculate_score.py"
    tree = ast.parse(source.read_text())
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions], type_ignores=[])
    namespace = {"re": re, "distance": Levenshtein.distance}
    exec(compile(module, str(source), "exec"), namespace)
    from lvr.eval.reasoning_scoring import mathvista_normalize, score
    comparisons = 0
    for row in rows:
        if row["benchmark"] != "MathVista_testmini":
            continue
        spec = row["scoring"]
        for extraction in (row["answer"], "", "A", "(B)", "3.0", "1.234", "nonsense"):
            expected = namespace["normalize_extracted_answer"](extraction, row["options"], spec["question_type"],
                              spec["answer_type"], spec["precision"], ignore_empty_extractions=True)
            actual = mathvista_normalize(extraction, row)
            # The official list normalizer returns empty string for empty extraction;
            # our explicit failure marker is None, and both score it incorrect.
            if extraction == "" and expected == "":
                expected = None
            if expected != actual:
                raise ValueError(f"MathVista normalizer parity failed: {row['id']} {extraction!r}")
            comparisons += 1
    report = {"status": "passed" if not failures else "reference_failures", "records": len(rows),
              "images_decoded_and_hashed": len(unique), "reference_answers_correct": len(scores)-len(failures),
              "reference_failures": failures, "mathvista_official_normalizer_comparisons": comparisons,
              "records_sha256": file_sha(root/"records.jsonl"), "helper_records": sum(bool(r["helper_images"]) for r in rows),
              "max_input_images": max(len(r["images"]) for r in rows), "max_helpers": max(len(r["helper_images"]) for r in rows)}
    write_json(root/"validation_data.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit("Reference-answer failures require explicit review before acceptance")


if __name__ == "__main__":
    main()
