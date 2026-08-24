#!/usr/bin/env python3
"""Generate train/test JSON files for the LLM stage."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from dataset_utils import (
    clean_question,
    data_root,
    image_relative_name,
    load_json_list,
    load_metadata,
    raw_dataset_dirs,
    split_json_path,
    write_json,
)


DEFAULT_SFT_DATA_ROOT = Path("/private/wmz/my_projects/sft/data")


def build_records(root: Path, raw_dataset_dir: Path) -> list[dict[str, Any]]:
    dataset = raw_dataset_dir.name
    metadata = load_metadata(raw_dataset_dir)
    records: list[dict[str, Any]] = []

    for index, row in metadata.reset_index(drop=True).iterrows():
        relative_name = image_relative_name(row["images"])
        records.append(
            {
                "question_image": (Path("images") / "original" / dataset / relative_name).as_posix(),
                "auxiliary_image": (Path("images") / "auxiliary" / dataset / relative_name).as_posix(),
                "question": clean_question(str(row["problem"])),
                "answer": str(row["answer"]),
                "raw": {"dataset": dataset, "index": int(index)},
            }
        )

    return records


def split_records(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    test_count = max(1, round(len(shuffled) * 0.01))
    test_records = shuffled[:test_count]
    train_records = shuffled[test_count:]
    return train_records, test_records


def sft_image_path(value: Any, image_kind: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"SFT image path must be a non-empty string, got {value!r}")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"SFT image path must be relative, got {value!r}")
    if image_kind not in {"original", "auxiliary"}:
        raise ValueError(f"unsupported SFT image kind: {image_kind}")
    if path.parts and path.parts[0] == "images":
        return path.as_posix()
    if path.parts[:2] == ("Visual_CoT", "images"):
        return (Path("images") / image_kind / "Visual_CoT" / path.name).as_posix()
    return (Path("images") / image_kind / path).as_posix()


def load_sft_records(sft_data_root: Path, split: str) -> list[dict[str, Any]]:
    path = sft_data_root / f"{split}.json"
    records = load_json_list(path)
    normalized: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        item = dict(record)
        item["question_image"] = sft_image_path(item.get("question_image"), "original")
        item["auxiliary_image"] = sft_image_path(item.get("auxiliary_image"), "auxiliary")
        item["question"] = str(item.get("question", "")).strip()
        item["answer"] = str(item.get("answer", "")).strip()
        item["raw"] = {"dataset": "Visual_CoT", "split": split, "index": index, "source": "sft"}
        normalized.append(item)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--include-sft", action="store_true")
    parser.add_argument("--sft-data-root", type=Path, default=DEFAULT_SFT_DATA_ROOT)
    args = parser.parse_args()

    root = args.data_root.resolve()
    dataset_dirs = raw_dataset_dirs(root, args.dataset)
    if not dataset_dirs:
        raise FileNotFoundError("no supported raw dataset found")

    records: list[dict[str, Any]] = []
    for raw_dataset_dir in dataset_dirs:
        records.extend(build_records(root, raw_dataset_dir))

    train_records, test_records = split_records(records, args.seed)

    if args.include_sft:
        sft_data_root = args.sft_data_root.resolve()
        train_records.extend(load_sft_records(sft_data_root, "train"))
        test_records.extend(load_sft_records(sft_data_root, "test"))

    train_path = split_json_path(root, "llm", args.version, "train")
    test_path = split_json_path(root, "llm", args.version, "test")
    write_json(train_path, train_records)
    write_json(test_path, test_records)
    print(f"{train_path.relative_to(root)}: {len(train_records)}")
    print(f"{test_path.relative_to(root)}: {len(test_records)}")


if __name__ == "__main__":
    main()
