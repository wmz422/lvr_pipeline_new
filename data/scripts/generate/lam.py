#!/usr/bin/env python3
"""Generate the two-image JSON files consumed by LAM training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from auxiliary_image import draw_auxiliary_image
from dataset_utils import data_root, image_relative_name, load_metadata, write_json


def rel(root: Path, path: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def record(question_image: Path, auxiliary_image: Path, root: Path) -> dict[str, str]:
    return {
        "question_image": rel(root, question_image),
        "auxiliary_image": rel(root, auxiliary_image),
    }


def bbox_image_relative(value: str) -> tuple[str, Path]:
    parts = Path(value).parts
    dataset = parts[0]
    relative = Path(*parts[1:])
    if dataset in {"coco2014", "coco2017"} and relative.parts and relative.parts[0].startswith("train"):
        relative = Path(*relative.parts[1:])
    elif dataset == "gqa" and relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    return dataset, relative


def treevgr_records(root: Path) -> list[dict[str, str]]:
    parquet = next((root / "raw" / "TreeVGR-RL-37K").glob("*.parquet"), None)
    if parquet is None:
        return []
    metadata = load_metadata(parquet.parent)
    output = []
    for _, row in metadata.iterrows():
        relative = Path(image_relative_name(row["images"]))
        output.append(
            record(
                root / "images" / "original" / "TreeVGR-RL-37K" / relative,
                root / "images" / "auxiliary" / "TreeVGR-RL-37K" / relative,
                root,
            )
        )
    return output


def bbox_records(root: Path) -> list[dict[str, str]]:
    path = root / "raw" / "seal_vqa_data" / "with_bbox_191k.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for index, item in enumerate(records):
        dataset, relative = bbox_image_relative(str(item["image"]))
        output.append(
            record(
                root / "images" / "original" / dataset / relative,
                root / "images" / "auxiliary" / dataset / "records" / f"{index:06d}.png",
                root,
            )
        )
    return output


def visual_cot_records(root: Path) -> list[dict[str, str]]:
    path = root / "raw" / "Visual-CoT" / "viscot_363k.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for index, item in enumerate(records):
        image_path = Path(str(item["image"][0]).split("###", 1)[0])
        relative = Path(*image_path.parts[1:])
        output.append(
            record(
                root / "images" / "original" / "visual_cot_400k" / relative,
                root / "images" / "auxiliary" / "visual_cot_400k" / "records" / f"{index:06d}.png",
                root,
            )
        )
    return output


def visdrone_records(root: Path) -> list[dict[str, str]]:
    output = []
    auxiliary_root = root / "images" / "auxiliary" / "VisDrone"
    for auxiliary in sorted(auxiliary_root.glob("VisDrone2019-DET-*/*.png")):
        split = auxiliary.parent.name
        original = root / "images" / "original" / "VisDrone" / split / auxiliary.with_suffix(".jpg").name
        if not original.is_file():
            continue
        try:
            with Image.open(original) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            continue
        output.append(record(original, auxiliary, root))
    return output


def file_key(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_records(root: Path, benchmark_root: Path) -> list[dict[str, str]]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from benchmark_auxiliary import BENCHMARKS, iter_samples

    output = []
    hr_seen: set[tuple[str, str]] = set()
    original_root = root / "images" / "original" / "benchmark"
    auxiliary_root = root / "images" / "auxiliary" / "benchmark"
    for benchmark in BENCHMARKS:
        for sample in tqdm(iter_samples(benchmark_root, benchmark), desc=f"LAM benchmark {benchmark}"):
            auxiliary = auxiliary_root / sample.output_relative
            if not auxiliary.is_file():
                continue

            if benchmark.startswith("hr_bench"):
                if sample.image_path is not None:
                    image_key = file_key(sample.image_path)
                else:
                    image_key = f"memory:{sample.sample_id}"
                question_key = " ".join(sample.question.split()).casefold()
                key = (image_key, question_key)
                if key in hr_seen:
                    continue
                hr_seen.add(key)

            if sample.image_path is not None:
                question_image = sample.image_path
            elif sample.image is not None:
                question_image = original_root / sample.output_relative.with_suffix(".jpg")
                question_image.parent.mkdir(parents=True, exist_ok=True)
                if not question_image.is_file():
                    sample.image.convert("RGB").save(question_image, format="JPEG", quality=95, subsampling=0)
            else:
                continue
            if question_image.is_file():
                output.append(record(question_image, auxiliary, root))
    return output


def validate_records(records: Iterable[dict[str, str]], root: Path) -> None:
    missing = []
    for index, item in enumerate(records):
        for key in ("question_image", "auxiliary_image"):
            path = root / item[key]
            if not path.is_file():
                missing.append((index, key, item[key]))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        raise FileNotFoundError(f"LAM records reference missing files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--version", default="v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root.resolve()
    benchmark_root = args.benchmark_root or root.parent / "benchmark" / "bench"
    records: list[dict[str, str]] = []
    records.extend(treevgr_records(root))
    records.extend(bbox_records(root))
    records.extend(visual_cot_records(root))
    records.extend(visdrone_records(root))
    records.extend(benchmark_records(root, benchmark_root.resolve()))
    validate_records(records, root)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if len(records) <= args.test_count:
        raise ValueError(f"need more than {args.test_count} records, got {len(records)}")
    test_records = records[: args.test_count]
    train_records = records[args.test_count :]

    output_dir = root / "lam" / args.version
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    write_json(train_path, train_records)
    write_json(test_path, test_records)
    print(f"{train_path.relative_to(root)}: {len(train_records)}")
    print(f"{test_path.relative_to(root)}: {len(test_records)}")


if __name__ == "__main__":
    main()
