#!/usr/bin/env python3
"""Validate LLM-stage train/test JSON files and referenced images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"question_image", "auxiliary_image", "question", "answer", "raw"}


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_split(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def validate_record(root: Path, record: dict[str, Any], split: str, index: int) -> tuple[str, str, int]:
    missing = REQUIRED_KEYS - set(record)
    if missing:
        raise ValueError(f"{split}[{index}] missing keys: {sorted(missing)}")

    for key in ("question_image", "auxiliary_image"):
        value = record[key]
        if not isinstance(value, str):
            raise TypeError(f"{split}[{index}].{key} must be a string")
        if Path(value).is_absolute():
            raise ValueError(f"{split}[{index}].{key} must be relative: {value}")
        if not (root / value).exists():
            raise FileNotFoundError(f"{split}[{index}].{key} does not exist: {root / value}")

    for key in ("question", "answer"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{split}[{index}].{key} must be a non-empty string")

    raw = record["raw"]
    if not isinstance(raw, dict):
        raise TypeError(f"{split}[{index}].raw must be an object")
    dataset = raw.get("dataset")
    raw_index = raw.get("index")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"{split}[{index}].raw.dataset must be a non-empty string")
    if not isinstance(raw_index, int):
        raise ValueError(f"{split}[{index}].raw.index must be an integer")
    raw_split = raw.get("split", "")
    if raw_split is not None and not isinstance(raw_split, str):
        raise ValueError(f"{split}[{index}].raw.split must be a string when present")

    return dataset, raw_split or "", raw_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--version", default="v1")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--ratio-tolerance", type=int, default=1)
    args = parser.parse_args()

    root = args.data_root.resolve()
    llm_root = root / "llm" / args.version
    splits = {
        "train": load_split(llm_root / "train.json"),
        "test": load_split(llm_root / "test.json"),
    }

    if not args.allow_empty and (not splits["train"] or not splits["test"]):
        raise ValueError("train/test JSON files must not be empty")

    seen_by_split: dict[str, set[tuple[str, str, int]]] = {}
    for split, records in splits.items():
        seen: set[tuple[str, str, int]] = set()
        for index, record in enumerate(records):
            key = validate_record(root, record, split, index)
            if key in seen:
                raise ValueError(f"duplicate raw id in {split}: {key}")
            seen.add(key)
        seen_by_split[split] = seen

    overlap = seen_by_split["train"] & seen_by_split["test"]
    if overlap:
        raise ValueError(f"train/test raw ids overlap, first overlap: {sorted(overlap)[0]}")

    total = len(splits["train"]) + len(splits["test"])
    expected_test = max(1, round(total * 0.01)) if total else 0
    if abs(len(splits["test"]) - expected_test) > args.ratio_tolerance:
        raise ValueError(
            f"test split size must be within {args.ratio_tolerance} of {expected_test}, got {len(splits['test'])}"
        )

    print(f"valid llm/{args.version} train records: {len(splits['train'])}")
    print(f"valid llm/{args.version} test records: {len(splits['test'])}")


if __name__ == "__main__":
    main()
