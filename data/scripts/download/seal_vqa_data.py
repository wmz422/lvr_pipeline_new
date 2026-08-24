#!/usr/bin/env python3
"""Download SEAL VQA annotations and collect records that contain boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DATASET = "seal_vqa_data"
REPO_ID = "craigwu/seal_vqa_data"
BOX_SOURCE_FILES = (
    "GQA_data.json",
    "llava_focus_data.json",
    "spatial_relation_data.json",
    "vaw_attribute_data.json",
)
ALL_SOURCE_FILES = BOX_SOURCE_FILES + (
    "llava_instruct_data.json",
    "negative_data.json",
)
BOX_OUTPUT = "with_bbox_191k.json"
MANIFEST_OUTPUT = "with_bbox_191k_manifest.json"


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def has_bbox(item: dict[str, Any]) -> bool:
    target_instances = item.get("target_instances")
    if not isinstance(target_instances, list):
        return False

    for instance in target_instances:
        if not isinstance(instance, dict):
            continue
        bbox = instance.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return True
    return False


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError(f"expected a list of objects in {path}")
    return data


def download_raw(raw_dir: Path) -> None:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=[*ALL_SOURCE_FILES, ".gitattributes"],
    )


def write_box_subset(raw_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for filename in BOX_SOURCE_FILES:
        path = raw_dir / filename
        data = load_json_list(path)
        filtered = [item for item in data if has_bbox(item)]
        counts[filename] = {"total": len(data), "with_bbox": len(filtered)}

        for item in filtered:
            item = dict(item)
            item.setdefault("_source_file", filename)
            records.append(item)

    output_path = raw_dir / BOX_OUTPUT
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    manifest = {
        "repo_id": REPO_ID,
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "selection": "records whose target_instances contain at least one bbox list of length 4",
        "source_files": list(BOX_SOURCE_FILES),
        "counts": counts,
        "total_with_bbox": len(records),
    }
    with (raw_dir / MANIFEST_OUTPUT).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    root = args.data_root.resolve()
    raw_dir = root / "raw" / DATASET
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_raw(raw_dir)

    manifest = write_box_subset(raw_dir)
    print(f"raw dir: {raw_dir}")
    print(f"{BOX_OUTPUT}: {manifest['total_with_bbox']}")
    for filename, counts in manifest["counts"].items():
        print(f"{filename}: total={counts['total']} with_bbox={counts['with_bbox']}")


if __name__ == "__main__":
    main()
