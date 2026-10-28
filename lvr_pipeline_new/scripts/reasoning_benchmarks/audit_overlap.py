"""Audit the frozen suite against the actual old/new train and development splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from lvr.data.records import read_records
from lvr.eval.reasoning_benchmark import DEFAULT_SUITE, ROOT, file_sha, make_donors
from download import write_json


def normalized(text):
    text = re.sub(r"<image_start>.*?<image_end>|<image\d*>", "", text)
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()
    suite = args.suite
    rows = [json.loads(line) for line in (suite / "records.jsonl").read_text().splitlines()]
    pixels, maps, questions = defaultdict(list), defaultdict(list), Counter()
    # Distinct perceptual hashes are sufficient for nearest-distance screening;
    # retain representatives while exact matches retain all training identities.
    phashes = {}
    counts, bindings = Counter(), {}
    def add_image(im, record):
        pixels[im["pixel_sha256"]].append(record)
        phashes.setdefault(int(im["phash"], 16), record)
    for split in ("train", "val"):
        directory = ROOT / "data/processed/reasoning_v1" / split
        index = json.loads((directory / "index.json").read_text())
        bindings["reasoning_v1/" + split] = {"index_sha256": file_sha(directory/"index.json"),
            "shards": {e["file"]: file_sha(directory/e["file"]) for e in index["shards"]}}
        for row in read_records(directory):
            scope = "new_" + split
            record = {"scope": scope, "id": row["id"], "group_id": row["group_id"]}
            counts[scope] += 1
            questions[(scope, normalized(row["question"]))] += 1
            if row["group_id"].startswith("map:"):
                maps[row["group_id"]].append(record)
            for im in row["problem_images"] + row["auxiliary_images"]:
                add_image(im, {**record, "path": im["path"]})
    database = ROOT / "data/processed/reasoning_build/exclusions.sqlite"
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    cached = {(scope, identity): (pixel, phash) for scope, identity, pixel, phash in db.execute(
        "select scope,identity,pixel_sha,phash from images where scope in ('old_train','old_test')")}
    bindings["old_fingerprint_cache"] = {"path": str(database.relative_to(ROOT)), "sha256": file_sha(database)}
    for split in ("train", "test"):
        directory = ROOT / "data/sft" / split
        index = json.loads((directory / "index.json").read_text())
        bindings["old/" + split] = {"index_sha256": file_sha(directory/"index.json"),
            "shards": {e["file"]: file_sha(directory/e["file"]) for e in index["shards"]}}
        for row in read_records(directory):
            scope = "old_train" if split == "train" else "old_val"
            identity = row["question_image"]
            pixel, phash = cached[("old_"+split, identity)]
            record = {"scope": scope, "id": identity+":"+str(row["raw"]["index"]), "path": identity}
            counts[scope] += 1
            questions[(scope, normalized(row["question"]))] += 1
            add_image({"pixel_sha256": pixel, "phash": phash}, record)
    db.close()
    print("INDEXED", dict(counts), len(pixels), "pixel identities", len(phashes), "perceptual hashes", flush=True)
    # Five disjoint blocks guarantee a shared block for every <=4 bit neighbor.
    buckets = [defaultdict(set) for _ in range(5)]
    widths = [13, 13, 13, 13, 12]
    for value in phashes:
        offset = 0
        for i, width in enumerate(widths):
            buckets[i][(value >> offset) & ((1 << width)-1)].add(value); offset += width
    evidence, near, within = [], [], defaultdict(list)
    excluded = set()
    image_cache = {}
    def image_matches(im):
        key = im["sha256"]
        if key in image_cache:
            return image_cache[key]
        value = int(im["phash"], 16); candidates = set(); offset = 0
        for i, width in enumerate(widths):
            candidates.update(buckets[i].get((value >> offset) & ((1 << width)-1), ())); offset += width
        close = sorted((distance, other) for other in candidates if (distance := (value ^ other).bit_count()) <= 4)
        image_cache[key] = close
        return close
    for ordinal, row in enumerate(rows, 1):
        for role in ("images", "helper_images"):
            for im in row[role]:
                within[im["pixel_sha256"]].append({"id": row["id"], "role": role})
                matches = pixels.get(im["pixel_sha256"], [])
                if matches:
                    excluded.add(row["id"])
                    evidence.append({"id": row["id"], "role": role, "asset": im["path"], "kind": "decoded_pixel_identity",
                                     "matches": matches})
                if role == "images" and not matches:
                    close = image_matches(im)
                    if close:
                        near.append({"id": row["id"], "asset": im["path"], "matching_distinct_hashes": len(close),
                            "candidates": [{"hamming": d, **phashes[h]} for d, h in close[:3]],
                            "status": "perceptual_candidate_not_confirmed_overlap"})
        if row["group_id"] in maps:
            excluded.add(row["id"])
            evidence.append({"id": row["id"], "kind": "grid_map_identity", "matches": maps[row["group_id"]]})
        if ordinal % 500 == 0:
            print("AUDITED", ordinal, "/", len(rows), "exact_excluded", len(excluded), "near_candidates", len(near), flush=True)
    # Near matches are a review queue, never silently declared duplicates or clean.
    (suite/"overlap_exact.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in evidence))
    (suite/"overlap_perceptual_candidates.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in near))
    duplicate_groups = [items for items in within.values() if len({i["id"] for i in items}) > 1]
    write_json(suite/"within_suite_shared_images.json", duplicate_groups)
    clean = [r for r in rows if r["id"] not in excluded]
    write_json(suite/"selection.json", {"default": "clean", "excluded_ids": sorted(excluded),
        "policy": "Conservative exclusion for exact input/helper pixel identity or map identity with train or development data. Perceptual candidates require review; clean is not proof of semantic independence.",
        "full_counts": dict(Counter(r["benchmark"] for r in rows)), "clean_counts": dict(Counter(r["benchmark"] for r in clean))})
    write_json(suite/"donors.json", {"full": make_donors(rows), "clean": make_donors(clean)})
    write_json(suite/"overlap_report.json", {"training_and_development_records": dict(counts), "bindings": bindings,
        "exact_overlap_records": len(excluded), "exact_evidence": len(evidence), "perceptual_candidate_records": len({e["id"] for e in near}),
        "perceptual_threshold": 4, "within_suite_shared_pixel_groups": len(duplicate_groups),
        "scope": "All suite question and helper images vs old original-image fingerprints and new published problem/helper fingerprints; exact VSP map identity. Near-image screen covers suite question images. Old fingerprints reused from prior audit with actual current shard identity lookup. No claim of exhaustive semantic or scene independence.",
        "limitations": ["Question-only matches are not exclusions because templates repeat across distinct images.",
                        "Sliding train/val share 32 abstract 2x2 arrangements; this is not unseen-structure testing.",
                        "Perceptual candidates are not automatically confirmed; exact-clean counts remain subject to review."]})
    print("COMPLETE", len(rows), "full", len(clean), "exact-clean", len(near), "candidate images", flush=True)


if __name__ == "__main__":
    main()
