"""Count low-resolution original images in the active no-SAT training release."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parent.parent
PROJECT = WORKSPACE / "lvr_pipeline_new"
LEGACY = WORKSPACE / "paper_experiments/datasets/vstar_decontaminated_v1/sft/train"
IMAGE_ROOT = Path("/workspace/wmz/lvr_pipeline_new/data")
REASONING = EXPERIMENT.parent / "2026-10-26-qwen25-no-sat-ablation/data/reasoning_train_without_sat.json"
OUTPUT = EXPERIMENT / "results/hr_data_diagnostic/under_0_1mp_full"
THRESHOLD = 100000


def read_json(path, hashes):
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def image_size(path):
    try:
        with Image.open(path) as im:
            return path, im.width, im.height, None
    except Exception as error:
        return path, None, None, repr(error)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes, items, expected_sizes = {}, [], {}
    index = read_json(LEGACY / "index.json", hashes)
    for shard in index["shards"]:
        path = LEGACY / shard["file"]
        rows = read_json(path, hashes)
        assert hashes[str(path)] == shard["sha256"] and len(rows) == shard["records"]
        for row in rows:
            path = Path(row["question_image"])
            if not path.is_absolute():
                path = IMAGE_ROOT / path
            items.append({"id": f"{row['question_image']}:{row['raw']['index']}",
                          "source": row["raw"]["dataset"], "images": [str(path)]})
    assert len(items) == index["records"] == 122706
    reasoning = read_json(REASONING, hashes)
    plan = json.loads((REASONING.parent.parent / "configs/plan.json").read_text())
    assert hashes[str(REASONING)] in str(plan)
    assert len(reasoning) == 49518
    for row in reasoning:
        assert row["source"] != "sat_perspective"
        paths = []
        for image in row["problem_images"]:
            path = Path(image["path"])
            if not path.is_absolute():
                path = PROJECT / path
            paths.append(str(path))
            size = (int(image["width"]), int(image["height"]))
            assert str(path) not in expected_sizes or expected_sizes[str(path)] == size
            expected_sizes[str(path)] = size
        assert paths
        items.append({"id": row["id"], "source": row["source"], "images": paths})
    del reasoning
    assert len(items) == 172224
    assert len({i["id"] for i in items}) == len(items)
    paths = sorted({p for item in items for p in item["images"]})
    print(json.dumps({"phase": "reading_image_headers", "training_qa": len(items),
                      "unique_image_paths": len(paths)}), flush=True)
    sizes, errors = {}, []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (path, width, height, error) in enumerate(pool.map(image_size, paths), 1):
            if error:
                errors.append({"path": path, "error": error})
            else:
                sizes[path] = (width, height)
                if path in expected_sizes and expected_sizes[path] != (width, height):
                    errors.append({"path": path, "metadata_size": expected_sizes[path],
                                   "actual_size": [width, height]})
            if i % 10000 == 0 or i == len(paths):
                print(json.dumps({"checked": i, "total": len(paths), "errors": len(errors),
                                  "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    if errors:
        (OUTPUT / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise RuntimeError(f"Image verification failed: {len(errors)} errors")

    grouped = defaultdict(Counter)
    source_images = defaultdict(set)
    low_qa, image_occurrences, total_multi = [], 0, 0
    for item in items:
        pixels = [sizes[p][0] * sizes[p][1] for p in item["images"]]
        flags = [area < THRESHOLD for area in pixels]
        source = item["source"]
        source_images[source].update(item["images"])
        grouped[source]["train_qa"] += 1
        grouped[source]["qa_all_originals_under_threshold"] += int(all(flags))
        grouped[source]["qa_any_original_under_threshold"] += int(any(flags))
        grouped[source]["low_resolution_image_occurrences"] += sum(flags)
        grouped[source]["image_occurrences"] += len(flags)
        image_occurrences += len(flags)
        total_multi += int(len(flags) > 1)
        if any(flags):
            low_qa.append({"id": item["id"], "source": source, "original_images": len(flags),
                           "minimum_original_pixels": min(pixels), "maximum_original_pixels": max(pixels),
                           "all_originals_under_threshold": all(flags),
                           "image_paths": json.dumps(item["images"])})
    rows = []
    for source, counts in sorted(grouped.items()):
        rows.append({"source": source, **counts,
                     "qa_all_under_pct": counts["qa_all_originals_under_threshold"] / counts["train_qa"] * 100,
                     "unique_original_image_paths": len(source_images[source]),
                     "unique_low_resolution_image_paths": sum(sizes[p][0]*sizes[p][1] < THRESHOLD for p in source_images[source])})
    image_rows = [{"path": p, "width": w, "height": h, "pixels": w*h,
                   "under_100000_pixels": w*h < THRESHOLD} for p, (w, h) in sorted(sizes.items())]
    write_csv(OUTPUT / "by_source.csv", rows, list(rows[0]))
    write_csv(OUTPUT / "image_dimensions.csv", image_rows, list(image_rows[0]))
    write_csv(OUTPUT / "low_resolution_qa.csv", low_qa, list(low_qa[0]))
    summary = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "training_qa": len(items), "threshold_pixels": THRESHOLD, "comparison": "strictly less than",
               "qa_all_originals_under_threshold": sum(r["qa_all_originals_under_threshold"] for r in rows),
               "qa_any_original_under_threshold": len(low_qa), "multi_image_qa": total_multi,
               "unique_original_image_paths": len(sizes),
               "unique_low_resolution_image_paths": sum(r["under_100000_pixels"] for r in image_rows),
               "original_image_occurrences": image_occurrences,
               "low_resolution_image_occurrences": sum(r["low_resolution_image_occurrences"] for r in rows),
               "metadata_dimensions_verified": len(expected_sizes), "image_read_errors": 0,
               "scope": "Current no-SAT training data only. Original problem images before resizing; auxiliary images excluded. Unique images are counted by path, not by image content hash.",
               "sources_sha256": hashes, "by_source": rows,
               "elapsed_seconds": round(time.monotonic()-started, 1)}
    summary["qa_all_under_pct"] = summary["qa_all_originals_under_threshold"] / len(items) * 100
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["sources_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
