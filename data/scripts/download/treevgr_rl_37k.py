#!/usr/bin/env python3
"""Download and unpack the TreeVGR-RL-37K raw dataset."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path


DATASET = "TreeVGR-RL-37K"
REPO_ID = "HaochenWang/TreeVGR-RL-37K"
PARQUET_NAME = "vstar30k_visdrone6k_x1y1x2y2.parquet"
IMAGES_ARCHIVE = "images.tar.gz"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
EXPECTED_SIZES = {
    PARQUET_NAME: 2_926_363,
    IMAGES_ARCHIVE: 6_095_446_648,
}


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_file(filename: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / filename
    expected_size = EXPECTED_SIZES.get(filename)
    if expected_size is not None and target.exists() and target.stat().st_size == expected_size:
        return target

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "-C",
            "-",
            "-o",
            str(target),
            f"{BASE_URL}/{filename}",
        ],
        check=True,
    )
    return target


def safe_members(archive: tarfile.TarFile):
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        yield member


def extract_images(archive_path: Path, output_dir: Path, overwrite: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in safe_members(archive):
            if not member.isfile():
                continue

            parts = Path(member.name).parts
            if parts and parts[0] == "images":
                relative = Path(*parts[1:])
            else:
                relative = Path(*parts)
            if not relative.name:
                continue

            target = output_dir / relative
            if target.exists() and not overwrite:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                handle.write(source.read())
            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    root = args.data_root.resolve()
    raw_dir = root / "raw" / DATASET
    original_dir = root / "images" / "original" / DATASET

    parquet = download_file(PARQUET_NAME, raw_dir)
    print(f"metadata: {parquet}")

    if args.metadata_only:
        return

    archive = download_file(IMAGES_ARCHIVE, raw_dir)
    print(f"archive: {archive}")
    count = extract_images(archive, original_dir, overwrite=args.overwrite)
    print(f"extracted images: {count}")
    print(f"original images dir: {original_dir}")


if __name__ == "__main__":
    main()
