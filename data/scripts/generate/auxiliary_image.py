#!/usr/bin/env python3
"""Generate red-box auxiliary images from raw dataset annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from tqdm import tqdm

from dataset_utils import (
    boxes_from_target_instances,
    data_root,
    image_relative_name,
    load_metadata,
    raw_dataset_dirs,
)


def draw_auxiliary_pil(
    image: Image.Image,
    boxes: Iterable[tuple[float, float, float, float]],
) -> Image.Image:
    auxiliary = image.convert("RGB")
    width, height = auxiliary.size
    base_line_width = max(2, round(min(width, height) * 0.01))
    draw = ImageDraw.Draw(auxiliary)

    for x1, y1, x2, y2 in boxes:
        left = max(0, min(width - 1, round(min(x1, x2))))
        top = max(0, min(height - 1, round(min(y1, y2))))
        right = max(0, min(width - 1, round(max(x1, x2))))
        bottom = max(0, min(height - 1, round(max(y1, y2))))
        box_short_side = min(right - left + 1, bottom - top + 1)
        line_width = max(1, min(base_line_width, box_short_side // 3))
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=line_width)

    return auxiliary


def draw_auxiliary_image(source: Path, target: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    with Image.open(source) as image:
        auxiliary = draw_auxiliary_pil(image, boxes)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            auxiliary.save(target, format="JPEG", quality=95, subsampling=0)
        else:
            auxiliary.save(target, format="PNG", compress_level=1)


def generate_for_dataset(root: Path, raw_dataset_dir: Path, overwrite: bool) -> tuple[int, int]:
    dataset = raw_dataset_dir.name
    metadata = load_metadata(raw_dataset_dir)
    original_root = root / "images" / "original" / dataset
    auxiliary_root = root / "images" / "auxiliary" / dataset

    generated = 0
    skipped = 0
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=dataset):
        relative_name = image_relative_name(row["images"])
        source = original_root / relative_name
        target = auxiliary_root / Path(relative_name).with_suffix(".png")

        if target.exists() and not overwrite:
            skipped += 1
            continue
        if not source.exists():
            raise FileNotFoundError(f"missing original image: {source}")

        boxes = boxes_from_target_instances(row["target_instances"])
        draw_auxiliary_image(source, target, boxes)
        generated += 1

    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.data_root.resolve()
    dataset_dirs = raw_dataset_dirs(root, args.dataset)
    if not dataset_dirs:
        raise FileNotFoundError("no supported raw dataset found")

    for raw_dataset_dir in dataset_dirs:
        generated, skipped = generate_for_dataset(root, raw_dataset_dir, args.overwrite)
        print(f"{raw_dataset_dir.name}: generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
