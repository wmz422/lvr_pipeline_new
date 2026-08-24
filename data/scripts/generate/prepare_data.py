#!/usr/bin/env python3
"""Prepare original-image trees for the expanded LAM datasets.

The raw archives and annotation/text files are intentionally left in place.
Only already-extracted raw image directories are removed after their contents
have been copied into ``data/images/original``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def copy_zip_members(zip_path: Path, target_root: Path, strip_parts: int) -> int:
    count = 0
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts[strip_parts:]
            if not parts:
                continue
            target = target_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            count += 1
    return count


def copy_tree_images(source_root: Path, target_root: Path) -> int:
    count = 0
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        target = target_root / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def copy_visdrone_zip_images(zip_path: Path, target_root: Path) -> int:
    count = 0
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if "images" not in parts:
                continue
            image_index = parts.index("images")
            relative = Path(*parts[image_index + 1 :])
            if relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            count += 1
    return count


def extract_visual_cot_parts(parts_root: Path, target_root: Path) -> None:
    parts = sorted(parts_root.glob("cot_images_*"))
    if len(parts) != 13:
        raise RuntimeError(f"expected 13 Visual-CoT tar parts, found {len(parts)}")
    target_root.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(
        [
            "tar",
            "-x",
            "-f",
            "-",
            "-C",
            str(target_root),
            "--strip-components=1",
            "--no-same-owner",
            "--overwrite",
        ],
        stdin=subprocess.PIPE,
    )
    assert tar.stdin is not None
    try:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, tar.stdin, length=8 * 1024 * 1024)
        tar.stdin.close()
        return_code = tar.wait()
    except BaseException:
        tar.stdin.close()
        tar.kill()
        tar.wait()
        raise
    if return_code:
        raise RuntimeError(f"Visual-CoT tar extraction failed with exit code {return_code}")


def prepare(root: Path, cleanup_raw: bool, only_visdrone: bool = False) -> None:
    raw = root / "raw"
    original = root / "images" / "original"

    jobs = [
        (raw / "coco2014" / "train2014.zip", original / "coco2014", 1, raw / "coco2014" / "train2014"),
        (raw / "coco2017" / "train2017.zip", original / "coco2017", 1, raw / "coco2017" / "train2017"),
        (raw / "gqa" / "images.zip", original / "gqa", 1, raw / "gqa" / "images"),
    ]
    if not only_visdrone:
        for archive, target, strip_parts, extracted in jobs:
            if archive.is_file():
                count = copy_zip_members(archive, target, strip_parts)
                print(f"{archive.parent.name}: copied {count} images from {archive.name}")
            if cleanup_raw and extracted.is_dir():
                shutil.rmtree(extracted)
                print(f"removed raw image directory: {extracted}")

    visdrone_root = raw / "VisDrone"
    for split_root in sorted(visdrone_root.glob("VisDrone2019-DET-*")):
        source = split_root / "images"
        archive = visdrone_root / f"{split_root.name}.zip"
        target = original / "VisDrone" / split_root.name
        if archive.is_file():
            count = copy_visdrone_zip_images(archive, target)
            print(f"{split_root.name}: copied {count} images from {archive.name}")
        elif source.is_dir():
            count = copy_tree_images(source, target)
            print(f"{split_root.name}: copied {count} images")
        else:
            continue
        if cleanup_raw:
            if source.is_dir():
                shutil.rmtree(source)
                print(f"removed raw image directory: {source}")

    if not only_visdrone:
        visual_target = original / "visual_cot_400k"
        if not visual_target.exists() or not any(visual_target.iterdir()):
            extract_visual_cot_parts(raw / "Visual-CoT" / "cot_images_tar_split", visual_target)
            print(f"Visual-CoT extracted to {visual_target}")
        else:
            print(f"Visual-CoT destination already populated: {visual_target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--only-visdrone", action="store_true")
    args = parser.parse_args()
    prepare(args.data_root.resolve(), cleanup_raw=not args.no_cleanup, only_visdrone=args.only_visdrone)


if __name__ == "__main__":
    main()
