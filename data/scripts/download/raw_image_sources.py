#!/usr/bin/env python3
"""Download raw image archives used by SEAL VQA and VisDrone."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DOWNLOADS = {
    "gqa": [
        (
            "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "gqa/images.zip",
        ),
    ],
    "coco": [
        (
            "http://images.cocodataset.org/zips/train2017.zip",
            "coco2017/train2017.zip",
        ),
        (
            "http://images.cocodataset.org/zips/train2014.zip",
            "coco2014/train2014.zip",
        ),
    ],
    "visdrone": [
        (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
            "VisDrone/VisDrone2019-DET-train.zip",
        ),
        (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
            "VisDrone/VisDrone2019-DET-val.zip",
        ),
        (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
            "VisDrone/VisDrone2019-DET-test-dev.zip",
        ),
    ],
}


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "-C",
            "-",
            "-o",
            str(output),
            url,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument(
        "--group",
        choices=(*DOWNLOADS.keys(), "all"),
        default="all",
        help="Which source group to download.",
    )
    args = parser.parse_args()

    raw_root = args.data_root.resolve() / "raw"
    groups = DOWNLOADS.keys() if args.group == "all" else (args.group,)
    for group in groups:
        for url, relative_path in DOWNLOADS[group]:
            output = raw_root / relative_path
            print(f"download: {url}")
            print(f"      -> {output}")
            download(url, output)


if __name__ == "__main__":
    main()
