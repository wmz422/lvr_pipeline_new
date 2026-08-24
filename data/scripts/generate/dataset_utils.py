"""Shared helpers for data generation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SUPPORTED_DATASETS = {"TreeVGR-RL-37K"}
SPLITS = ("train", "test")


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_dataset_dirs(root: Path, dataset: str | None = None) -> list[Path]:
    raw_root = root / "raw"
    if dataset:
        candidates = [raw_root / dataset]
    else:
        candidates = sorted(path for path in raw_root.iterdir() if path.is_dir())

    return [path for path in candidates if path.exists() and path.name in SUPPORTED_DATASETS]


def normalize_version(version: str) -> str:
    if not version:
        raise ValueError("version must not be empty")
    if not version.startswith("v") or not version[1:].isdigit():
        raise ValueError(f"version must look like v0, v1, v2, got {version!r}")
    return version


def stage_dir(root: Path, stage: str, version: str) -> Path:
    return root / stage / normalize_version(version)


def split_json_path(root: Path, stage: str, version: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unsupported split {split!r}; expected one of {SPLITS}")
    return stage_dir(root, stage, version) / f"{split}.json"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def metadata_path(raw_dataset_dir: Path) -> Path:
    parquet_files = sorted(raw_dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet file found in {raw_dataset_dir}")
    if len(parquet_files) > 1:
        raise ValueError(f"multiple parquet files found in {raw_dataset_dir}: {parquet_files}")
    return parquet_files[0]


def load_metadata(raw_dataset_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(metadata_path(raw_dataset_dir))


def image_relative_name(images_value: Any) -> str:
    if isinstance(images_value, str):
        image_path = images_value
    else:
        values = list(images_value)
        if len(values) != 1:
            raise ValueError(f"expected one image, got {values}")
        image_path = str(values[0])

    path = Path(image_path)
    if path.parts and path.parts[0] == "images":
        path = Path(*path.parts[1:])
    return path.as_posix()


def boxes_from_target_instances(target_instances: Any) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for instance in list(target_instances):
        bbox = instance.get("bbox") if isinstance(instance, dict) else None
        if bbox is None:
            continue
        values = [float(value) for value in list(bbox)]
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = values
        boxes.append((x1, y1, x2, y2))
    return boxes


def clean_question(problem: str) -> str:
    return problem.replace("<image>", "", 1).strip()


def write_json(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
