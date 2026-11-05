"""Portable input validation and immutable state for the five-mode ablation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT / "lvr_pipeline_new"
MODES = ("teacher", "teacher_shuffled", "predicted", "predicted_shuffled", "zero")
FAMILIES = {"qwen3-4b": ("qwen3_vl", 2560), "qwen25-7b": ("qwen2_5_vl", 3584)}
COUNTS = {"HRBench4K": 800, "VStarBench": 191, "MIRA": 546, "MentisOculi": 1250}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bind_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Inputs, weights or code differ from this run: {path}; use a new --output")
    write_json(path, value)


def safe_path(root, relative):
    path = Path(relative)
    root = Path(root).resolve()
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise ValueError(f"Unsafe benchmark path: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Benchmark path escapes its directory: {relative}")
    return target


def validate_donors(rows, donors):
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows) or set(donors) != set(lookup):
        raise ValueError("Donor mapping must cover every unique target")
    sources = [entry["id"] for entry in donors.values()]
    if len(set(sources)) != len(rows) or set(sources) != set(lookup):
        raise ValueError("Global donors must be a one-to-one permutation")
    for target, entry in donors.items():
        donor = entry["id"]
        if entry.get("scope") != "global" or entry.get("seed") != 42:
            raise ValueError("Expected the frozen global seed-42 permutation")
        if target == donor or lookup[target]["group_id"] == lookup[donor]["group_id"]:
            raise ValueError("Self and same-problem-group donors are forbidden")


def load_data(root, *, verify_images=True):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "paper-benchmark-v1" or manifest.get("status") != "accepted":
        raise ValueError("An accepted ablation benchmark is required")
    release = json.loads((HERE / "benchmark.json").read_text())
    if sha256(root / "manifest.json") != release["suite_manifest_sha256"]:
        raise ValueError("Benchmark does not match the frozen ablation release")
    for name, expected in manifest["files"].items():
        if sha256(safe_path(root, name)) != expected:
            raise ValueError(f"Benchmark metadata changed: {name}")
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if Counter(row["benchmark"] for row in rows) != COUNTS:
        raise ValueError("Expected all 2787 records of the four complete benchmarks")
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["inference"]["original_pixels"] is not None or protocol["inference"]["do_sample"]:
        raise ValueError("Use native processor defaults and greedy decoding")
    donors = json.loads((root / "donors.json").read_text())["full"]
    if sha256(root / "donors.json") != release["donors_sha256"]:
        raise ValueError("Donor permutation differs from the shared frozen mapping")
    validate_donors(rows, donors)
    assets = {}
    for row in rows:
        if not row["images"] or not row["helper_images"]:
            raise ValueError(f"Originals and teacher helpers are required: {row['id']}")
        for image in row["images"] + row["helper_images"]:
            name = image["path"]
            if name in assets and assets[name] != image["sha256"]:
                raise ValueError("Inconsistent asset checksum")
            assets[name] = image["sha256"]
    if len(assets) != release["images"]:
        raise ValueError("Image coverage changed")
    for name, expected in assets.items():
        path = safe_path(root, name)
        if not path.is_file() or (verify_images and sha256(path) != expected):
            raise ValueError(f"Missing or changed image: {name}")
    return rows, protocol, manifest, donors


def model_item(row, root, *, teacher=False):
    """Answers, scoring states and metadata never enter either model input."""
    item = {"id": row["id"], "question": row["question"], "answer": "", "boxes": [],
            "images": [{"path": str(safe_path(root, im["path"])), "boxes": []} for im in row["images"]],
            "teacher_view_mode": "auxiliary_images" if teacher else "crops"}
    if teacher:
        item["auxiliary_images"] = [{"path": str(safe_path(root, im["path"]))} for im in row["helper_images"]]
    return item


def item_path(output, kind, identity, mode=None):
    filename = hashlib.sha256(identity.encode()).hexdigest()
    if kind == "latent":
        return Path(output) / "latents" / (filename + ".safetensors")
    if kind == "answer" and mode in MODES:
        return Path(output) / "predictions" / mode / (filename + ".json")
    raise ValueError("Unknown ablation artifact kind/mode")


def latent_source(mode, identity, donors):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "zero":
        return None, None
    source = donors[identity]["id"] if mode.endswith("_shuffled") else identity
    return source, "teacher" if mode.startswith("teacher") else "predicted"


def qa_identity(row):
    return digest({"question": row["question"], "images": [im["sha256"] for im in row["images"]]})


def select_rows(rows, smoke_per_benchmark=0):
    if not smoke_per_benchmark:
        return rows
    chosen = []
    for name in COUNTS:
        pool = [row for row in rows if row["benchmark"] == name]
        # Exercise high-resolution inputs and records with many ordered helpers.
        pool.sort(key=lambda r: (len(r["helper_images"]), sum(im["width"] * im["height"] for im in r["images"])), reverse=True)
        chosen.extend(pool[:smoke_per_benchmark])
    return chosen


def required_latent_rows(rows, selected, donors):
    wanted = {row["id"] for row in selected}
    wanted.update(donors[identity]["id"] for identity in list(wanted))
    return [row for row in rows if row["id"] in wanted]


def validate_family(path, family):
    config = json.loads((Path(path) / "config.json").read_text())
    expected_type, expected_hidden = FAMILIES[family]
    hidden = config.get("text_config", config).get("hidden_size")
    if config.get("model_type") != expected_type or hidden != expected_hidden:
        raise ValueError(f"--model-family {family} does not match the local base config")
    return config
