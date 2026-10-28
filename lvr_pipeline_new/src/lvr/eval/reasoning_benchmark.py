"""Versioned local reasoning benchmark inputs, without training-label coupling."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

SCHEMA = "lvr-reasoning-benchmark-v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = ROOT / "benchmark/reasoning_v1"
EXPECTED = {"MentisOculi": 1250, "BLINK_Spatial": 426, "VSP": 400,
            "VisuLogic": 1000, "MIRA": 546, "MathVista_testmini": 1000}
ANSWER_INSTRUCTION = "\nReturn only the final answer inside <answer>...</answer>, without an explanation."


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe suite path: {relative}")
    full = Path(root) / path
    if not full.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Suite path escapes root: {relative}")
    return full


def load_suite(root=DEFAULT_SUITE, *, benchmarks=None, selection="clean"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["status"] != "accepted":
        raise ValueError("Benchmark suite has not passed acceptance")
    for name in manifest["files"]:
        if file_sha(root / name) != manifest["files"][name]:
            raise ValueError(f"Benchmark protocol changed: {name}")
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if selection not in ("clean", "full"):
        raise ValueError("selection must be clean or full")
    wanted = set(benchmarks or EXPECTED)
    if not wanted <= set(EXPECTED):
        raise ValueError(f"Unknown benchmarks: {wanted - set(EXPECTED)}")
    excluded = set(json.loads((root / "selection.json").read_text())["excluded_ids"]) if selection == "clean" else set()
    return [r for r in records if r["benchmark"] in wanted and r["id"] not in excluded], manifest


@lru_cache(maxsize=20000)
def _verify_asset(path, expected, size, modified):
    if file_sha(path) != expected:
        raise ValueError(f"Benchmark image has changed: {path}")
    return str(path)


def asset_path(root, im):
    path = safe_path(root, im["path"])
    if "sha256" not in im:
        raise ValueError("Benchmark images must have a fixed SHA256")
    stat = path.stat()
    return _verify_asset(path, im["sha256"], stat.st_size, stat.st_mtime_ns)


def model_item(row, root=DEFAULT_SUITE, *, teacher=False):
    """Construct an explicit allowlist. Answers, maps, metadata never enter prompts."""
    images = [{"path": asset_path(root, im), "boxes": []} for im in row["images"]]
    item = {"id": row["id"], "question": row["question"], "images": images,
            "boxes": [], "answer": "", "teacher_view_mode": "crops"}
    if teacher:
        if not row["helper_images"]:
            raise ValueError(f"Teacher helper unavailable: {row['id']}")
        item["teacher_view_mode"] = "auxiliary_images"
        item["auxiliary_images"] = [{"path": asset_path(root, im)} for im in row["helper_images"]]
    return item


def make_donors(records):
    """Seedless deterministic derangements within task, difficulty and input count."""
    buckets = defaultdict(list)
    for row in records:
        buckets[(row["benchmark"], row["task"], str(row["difficulty"]), len(row["images"]))].append(row)
    result = {}
    for key, rows in sorted(buckets.items()):
        rows.sort(key=lambda r: hashlib.sha256(("reasoning-donor-v1:" + r["id"]).encode()).hexdigest())
        if len(rows) < 2:
            # Sparse buckets may use the same task and input count at another difficulty.
            pool = [r for r in records if r["benchmark"] == key[0] and r["task"] == key[1]
                    and len(r["images"]) == key[3] and r["id"] != rows[0]["id"]]
            if not pool:
                raise ValueError(f"No within-task donor: {key}")
            donor = min(pool, key=lambda r: r["id"])
            result[rows[0]["id"]] = {"id": donor["id"], "policy": "same_task_input_count_sparse_difficulty"}
        else:
            # Prefer a cyclic permutation avoiding the same underlying problem group.
            shifts = range(1, len(rows))
            shift = min(shifts, key=lambda k: sum(r["group_id"] == rows[(i+k) % len(rows)]["group_id"] for i, r in enumerate(rows)))
            for i, row in enumerate(rows):
                donor = rows[(i + shift) % len(rows)]
                result[row["id"]] = {"id": donor["id"], "policy": "same_task_difficulty_input_count",
                                     "same_group": row["group_id"] == donor["group_id"]}
    return result
