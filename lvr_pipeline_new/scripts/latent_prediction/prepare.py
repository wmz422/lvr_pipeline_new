#!/usr/bin/env python3
"""Verify the exact caption/split alignment and optionally every required original image."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from lvr.data.records import read_records

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT / "data")
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "data/latent_prediction"
    manifest = json.loads((directory / "manifest.json").read_text())
    compressed = (directory / manifest["file"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["sha256"]:
        raise ValueError("Caption archive checksum mismatch")
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != manifest["uncompressed_sha256"]:
        raise ValueError("Caption content checksum mismatch")
    captions = [json.loads(line) for line in raw.splitlines()]
    by_id = {row["id"]: row for row in captions}
    if len(by_id) != len(captions) or len(captions) != manifest["records"]:
        raise ValueError("Missing or duplicate caption IDs")
    seen, images = set(), set()
    for split, expected in (("train", 122896), ("test", 1242)):
        qa = read_records(ROOT / "data/sft" / split)
        annotations = read_records(ROOT / "data/align" / split)
        if len(qa) != len(annotations) or len(qa) != expected:
            raise ValueError(f"Incorrect {split} size")
        for item, annotation in zip(qa, annotations):
            identity = f"{item['question_image']}:{item['raw']['index']}"
            if identity in seen:
                raise ValueError(f"Duplicate training/validation ID: {identity}")
            seen.add(identity)
            caption = by_id[identity]
            if any(item[k] != annotation[k] for k in ("question_image", "auxiliary_image")):
                raise ValueError(f"QA/box order differs: {identity}")
            if caption["boxes"] != annotation["boxes"] or not caption["caption"].strip():
                raise ValueError(f"Caption/box mismatch: {identity}")
            images.add(item["question_image"])
        print(f"{split}: {expected} matched QA/box/description records")
    if seen != set(by_id):
        raise ValueError("Caption set differs from the exact train+validation split")
    if args.check_images:
        missing = [name for name in sorted(images) if not (args.image_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing originals; first examples: {missing[:5]}")
        print(f"Original images present: {len(images)}")
    print("Caption and split audit passed. Images were checked." if args.check_images else
          "Caption and split audit passed. Use --check-images after downloading images.")


if __name__ == "__main__":
    main()
