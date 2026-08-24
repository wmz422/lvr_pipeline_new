#!/usr/bin/env python3
"""Download the Visual-CoT dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "deepcs233/Visual-CoT"
DATASET_DIR = "Visual-CoT"


def data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=data_root())
    args = parser.parse_args()

    raw_dir = args.data_root.resolve() / "raw" / DATASET_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_dir),
        max_workers=4,
    )

    print(f"raw dir: {raw_dir}")
    print(f"source: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
