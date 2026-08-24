#!/usr/bin/env python3
"""Generate auxiliary images for the datasets used by the LAM stage."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from auxiliary_image import draw_auxiliary_image


ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = Path("/data/private/wmz/my_projects/sft/data")


def render_task(task: tuple[Path, Path, list[tuple[float, float, float, float]]]) -> int:
    source, target, boxes = task
    draw_auxiliary_image(source, target, boxes)
    return 1


def render_tasks(
    tasks: list[tuple[Path, Path, list[tuple[float, float, float, float]]]],
    description: str,
    workers: int,
) -> int:
    if not tasks:
        return 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(render_task, tasks, chunksize=16)
        return sum(tqdm(results, total=len(tasks), desc=description))


def as_xyxy(bbox: Any, xywh: bool) -> tuple[float, float, float, float] | None:
    try:
        values = [float(value) for value in list(bbox)]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    x1, y1, a, b = values
    return (x1, y1, x1 + a, y1 + b) if xywh else (x1, y1, a, b)


def image_relative_from_annotation(value: str) -> tuple[str, Path]:
    parts = Path(value).parts
    if not parts:
        raise ValueError(f"empty image path: {value!r}")
    dataset = parts[0]
    relative = Path(*parts[1:])
    if dataset in {"coco2014", "coco2017"} and relative.parts and relative.parts[0].startswith("train"):
        relative = Path(*relative.parts[1:])
    elif dataset == "gqa" and relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    return dataset, relative


def target_instances_boxes(value: Any, xywh: bool) -> list[tuple[float, float, float, float]]:
    boxes = []
    if value is None:
        return boxes
    for instance in list(value):
        if isinstance(instance, dict):
            bbox = instance.get("bbox")
        else:
            bbox = None
        box = as_xyxy(bbox, xywh)
        if box is not None:
            boxes.append(box)
    return boxes


def generate_bbox_json(root: Path, overwrite: bool, workers: int) -> int:
    source_path = root / "raw" / "seal_vqa_data" / "with_bbox_191k.json"
    records = json.loads(source_path.read_text(encoding="utf-8"))
    tasks = []
    missing = 0
    for index, record in enumerate(records):
        dataset, relative = image_relative_from_annotation(str(record["image"]))
        source = root / "images" / "original" / dataset / relative
        target = root / "images" / "auxiliary" / dataset / "records" / f"{index:06d}.png"
        if target.exists() and not overwrite:
            continue
        if not source.is_file():
            missing += 1
            continue
        boxes = target_instances_boxes(record.get("target_instances"), xywh=True)
        tasks.append((source, target, boxes))
    if missing:
        raise FileNotFoundError(f"COCO/GQA missing original images: {missing}")
    return render_tasks(tasks, "COCO/GQA auxiliary", workers)


def parse_visual_box(value: str) -> list[tuple[float, float, float, float]]:
    if "###" not in value:
        return []
    raw = value.split("###", 1)[1].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
        parsed = [parsed]
    return [box for item in parsed or [] if (box := as_xyxy(item, xywh=False)) is not None]


def generate_visual_cot(root: Path, overwrite: bool, workers: int) -> int:
    source_path = root / "raw" / "Visual-CoT" / "viscot_363k.json"
    records = json.loads(source_path.read_text(encoding="utf-8"))
    tasks = []
    missing = 0
    for index, record in enumerate(records):
        image_value = record["image"]
        image_path = str(image_value[0]).split("###", 1)[0]
        parts = Path(image_path).parts
        if len(parts) < 3 or parts[0] != "cot":
            raise ValueError(f"unsupported Visual-CoT image path: {image_path}")
        relative = Path(*parts[1:])
        source = root / "images" / "original" / "visual_cot_400k" / relative
        target = root / "images" / "auxiliary" / "visual_cot_400k" / "records" / f"{index:06d}.png"
        if target.exists() and not overwrite:
            continue
        if not source.is_file():
            missing += 1
            continue
        boxes = parse_visual_box(str(image_value[1]))
        tasks.append((source, target, boxes))
    if missing:
        raise FileNotFoundError(f"Visual-CoT missing original images: {missing}")
    return render_tasks(tasks, "Visual-CoT auxiliary", workers)


def generate_visdrone(root: Path, overwrite: bool, workers: int) -> int:
    raw_root = root / "raw" / "VisDrone"
    tasks = []
    invalid = []
    for annotation_dir in sorted(raw_root.glob("VisDrone2019-DET-*/annotations")):
        split = annotation_dir.parent.name
        for annotation_path in sorted(annotation_dir.glob("*.txt")):
            source = root / "images" / "original" / "VisDrone" / split / f"{annotation_path.stem}.jpg"
            target = root / "images" / "auxiliary" / "VisDrone" / split / f"{annotation_path.stem}.png"
            if target.exists() and not overwrite:
                continue
            if not source.is_file():
                continue
            try:
                with Image.open(source) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                invalid.append(f"{source}: {type(exc).__name__}")
                continue
            boxes = []
            for line in annotation_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) < 5:
                    continue
                try:
                    score = int(float(fields[4]))
                except ValueError:
                    continue
                if score != 1:
                    continue
                box = as_xyxy(fields[:4], xywh=True)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                continue
            tasks.append((source, target, boxes))
    if invalid:
        print(f"VisDrone skipped unreadable originals: {len(invalid)}")
        for item in invalid[:20]:
            print(f"  {item}")
    return render_tasks(tasks, "VisDrone auxiliary", workers)


def regenerate_existing(root: Path, sft_root: Path, overwrite: bool, workers: int) -> dict[str, int]:
    tasks_by_dataset: dict[str, list[tuple[Path, Path, list[tuple[float, float, float, float]]]]] = {
        "TreeVGR-RL-37K": [],
        "Visual_CoT": [],
    }
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("TreeVGR regeneration requires pandas/pyarrow") from exc

    parquet = next((root / "raw" / "TreeVGR-RL-37K").glob("*.parquet"), None)
    if parquet is None:
        raise FileNotFoundError("TreeVGR parquet metadata not found")
    frame = pd.read_parquet(parquet)
    for _, row in frame.iterrows():
        image_value = row["images"]
        image_path = str(list(image_value)[0] if not isinstance(image_value, str) else image_value)
        relative = Path(image_path)
        if relative.parts and relative.parts[0] == "images":
            relative = Path(*relative.parts[1:])
        source = root / "images" / "original" / "TreeVGR-RL-37K" / relative
        target = root / "images" / "auxiliary" / "TreeVGR-RL-37K" / relative
        if target.exists() and not overwrite:
            continue
        boxes = target_instances_boxes(row["target_instances"], xywh=False)
        tasks_by_dataset["TreeVGR-RL-37K"].append((source, target, boxes))

    seen: set[str] = set()
    for split in ("train", "test"):
        path = sft_root / f"{split}.json"
        if not path.is_file():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            image_value = record.get("question_image")
            if not isinstance(image_value, str):
                continue
            name = Path(image_value).name
            if not name.endswith("_0.jpg") or name in seen:
                continue
            seen.add(name)
            source = root / "images" / "original" / "Visual_CoT" / name
            target = root / "images" / "auxiliary" / "Visual_CoT" / name.replace("_0.jpg", "_1.jpg")
            if not source.is_file() or not isinstance(record.get("box"), list):
                continue
            if target.exists() and not overwrite:
                continue
            box = as_xyxy(record["box"], xywh=False)
            if box is None:
                continue
            tasks_by_dataset["Visual_CoT"].append((source, target, [box]))
    return {
        name: render_tasks(tasks, f"{name} auxiliary", workers)
        for name, tasks in tasks_by_dataset.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--sft-data-root", type=Path, default=SFT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.data_root.resolve()
    overwrite = args.overwrite and not args.skip_existing
    print("regenerated existing:", regenerate_existing(root, args.sft_data_root.resolve(), overwrite, args.workers))
    print("COCO/GQA:", generate_bbox_json(root, overwrite, args.workers))
    print("Visual-CoT:", generate_visual_cot(root, overwrite, args.workers))
    print("VisDrone:", generate_visdrone(root, overwrite, args.workers))


if __name__ == "__main__":
    main()
