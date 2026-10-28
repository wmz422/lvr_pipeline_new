"""Freeze a fully validated benchmark package; refuse incomplete acceptance."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, EXPECTED, ROOT, SCHEMA, file_sha
from download import write_json


def main():
    root = DEFAULT_SUITE
    if (root/"manifest.json").exists():
        raise FileExistsError("Benchmark manifest already published")
    data = json.loads((root/"validation_data.json").read_text())
    processor = json.loads((root/"validation_processor.json").read_text())
    rows = [json.loads(line) for line in (root/"records.jsonl").read_text().splitlines()]
    record_sha = file_sha(root/"records.jsonl")
    for report in (data, processor):
        if report["status"] != "passed" or report["records"] != sum(EXPECTED.values()) or report["records_sha256"] != record_sha:
            raise ValueError("Stale or incomplete acceptance evidence")
    if dict(Counter(r["benchmark"] for r in rows)) != EXPECTED:
        raise ValueError("Dataset counts changed")
    selection = json.loads((root/"selection.json").read_text())
    excluded = set(selection["excluded_ids"])
    donors = json.loads((root/"donors.json").read_text())
    for name in ("full", "clean"):
        subset = {r["id"]: r for r in rows if name == "full" or r["id"] not in excluded}
        if set(subset) != set(donors[name]):
            raise ValueError("Donor coverage differs from selection")
        for sid, donor in donors[name].items():
            if donor["id"] == sid or donor["id"] not in subset or subset[sid]["task"] != subset[donor["id"]]["task"]:
                raise ValueError("Invalid shuffled donor")
    names = ["records.jsonl", "source_lock.json", "donors.json", "selection.json", "build_report.json",
             "validation_data.json", "validation_processor.json", "processor_records.jsonl", "reference_scores.jsonl",
             "overlap_report.json", "overlap_exact.jsonl", "overlap_perceptual_candidates.jsonl",
             "overlap_visual_review.json", "overlap_structures.json", "within_suite_shared_images.json"]
    names += [str(p.relative_to(root)) for p in sorted((root/"upstream").rglob("*.py"))]
    images = {im["path"]: im["sha256"] for r in rows for im in r["images"]+r["helper_images"]}
    scorer_sources = {}
    for name, repo in (("MathVista", "lupantech/MathVista"), ("MIRA", "aiming-lab/MIRA")):
        tree = json.loads((root/"upstream"/(name+"_tree.json")).read_text())
        entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        for path in (root/"upstream"/name).rglob("*.py"):
            payload = path.read_bytes()
            assert hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest() == entries[str(path.relative_to(root/"upstream"/name))]["sha"]
        scorer_sources[name] = {"repo": repo, "revision": tree["sha"]}
    manifest = {"schema": SCHEMA, "version": "reasoning_v1", "status": "accepted", "records": len(rows),
                "benchmarks": EXPECTED, "clean_counts": selection["clean_counts"], "images": len(images),
                "files": {name: file_sha(root/name) for name in names}, "asset_sha256": images, "scorer_sources": scorer_sources,
                "protocol": {"primary_modes": ["plain", "predicted", "zero", "shuffled"], "diagnostic_mode": "teacher",
                  "teacher_benchmarks": ["MentisOculi", "MIRA"], "default_selection": "clean",
                  "original_pixels": "processor_default", "helper_pixels": 262144,
                  "max_new_tokens_default": 512, "max_length": 131072, "num_latents": 8, "latent_dim": 512,
                  "answer_format": "<answer>...</answer>", "generation": "greedy",
                  "mathvista_scoring": "official normalization with local extraction, no external model judge",
                  "mira_scoring": "official EM including two-part dice task; no external judge"},
                "scope": "Data/scorer/full real-processor acceptance. Small GPU generation evidence is recorded separately; no full model benchmark scores are implied."}
    write_json(root/"manifest.json", manifest)
    print("PUBLISHED", len(rows), "full", sum(selection["clean_counts"].values()), "filtered", len(images), "images")


if __name__ == "__main__":
    main()
