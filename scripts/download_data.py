#!/usr/bin/env python3
"""Rebuild training images from pinned public sources and the released JSON."""

import argparse
from collections import defaultdict
from io import BytesIO, RawIOBase
import json
import re
from pathlib import Path
import shutil
import tarfile
import zipfile

from PIL import Image, ImageDraw
from _downloads import BENCHMARKS, ROOT, download_source, prepare_benchmark, sources


class JoinedFiles(RawIOBase):
    """Read a split tar archive in order without creating another large file."""
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        chunks, remaining = [], size
        while remaining != 0:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = open(path, "rb")
            block = self.current.read(remaining)
            if block:
                chunks.append(block)
                if size >= 0:
                    remaining -= len(block)
            else:
                self.current.close()
                self.current = None
        return b"".join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_requested(files, destination, wanted, source):
    destination.mkdir(parents=True, exist_ok=True)
    remaining = {name for name in wanted if not (destination / name).is_file()}
    def match(name):
        if source == "visdrone":
            name = name.replace("/images/", "/")
        parts = Path(name).parts
        for i in range(len(parts)):
            candidate = "/".join(parts[i:])
            if candidate in remaining:
                return candidate
        return None
    def copy(name, handle):
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Invalid archive path: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with handle, target.open("wb") as output:
            shutil.copyfileobj(handle, output)
        remaining.remove(name)
    if not remaining:
        return
    if source == "visual_cot":
        with JoinedFiles(sorted(files)) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                name = match(member.name)
                if member.isfile() and name:
                    copy(name, archive.extractfile(member))
                if not remaining:
                    break
    else:
        for path in files:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        name = match(member.filename)
                        if not member.is_dir() and name:
                            copy(name, archive.open(member))
            else:
                with tarfile.open(path, "r|*") as archive:
                    for member in archive:
                        name = match(member.name)
                        if member.isfile() and name:
                            copy(name, archive.extractfile(member))
    if remaining:
        raise FileNotFoundError(f"{source}: {len(remaining)} originals absent from archive, e.g. {sorted(remaining)[:5]}")


def render_auxiliary(original, target, boxes):
    with Image.open(original) as image:
        auxiliary = image.convert("RGB")
    width, height = auxiliary.size
    base_width = max(2, round(min(width, height) * 0.01))
    draw = ImageDraw.Draw(auxiliary)
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(width-1, round(min(x1, x2))))
        top = max(0, min(height-1, round(min(y1, y2))))
        right = max(0, min(width-1, round(max(x1, x2))))
        bottom = max(0, min(height-1, round(max(y1, y2))))
        line_width = max(1, min(base_width, min(right-left+1, bottom-top+1)//3))
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=line_width)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        auxiliary.save(target, format="JPEG", quality=95, subsampling=0)
    else:
        auxiliary.save(target, format="PNG", compress_level=1)


def benchmark_originals(source, rows, bench_root, destination):
    from datasets import load_from_disk
    from lvr.eval.benchmarks.common import decode_image_value
    if source == "vstar":
        for row in rows:
            yield row, bench_root / "vstar_bench" / row["source_path"]
        return
    datasets, indices = {}, {}
    for row in rows:
        relative = Path(row["source_path"])
        if source == "blink":
            config, split = relative.parts[:2]
            dataset_key = (config, split)
            if dataset_key not in datasets:
                datasets[dataset_key] = load_from_disk(bench_root / "BLINK" / config)[split]
                indices[dataset_key] = {str(value): i for i, value in enumerate(datasets[dataset_key]["idx"])}
        else:
            dataset_key = source
            if dataset_key not in datasets:
                datasets[dataset_key] = load_from_disk(bench_root / source)
                indices[dataset_key] = {str(value): i for i, value in enumerate(datasets[dataset_key]["index"])}
        stem = relative.stem
        image_index = 1
        if source == "blink":
            multi = re.fullmatch(r"(.+)_image(\d+)", stem)
            if multi:
                stem, image_index = multi.group(1), int(multi.group(2))
        item = datasets[dataset_key][indices[dataset_key][stem]]
        target = destination / relative
        if not target.is_file():
            value = item[f"image_{image_index}"] if source == "blink" else item.get("image", item.get("bytes"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, dict) and value.get("bytes") is not None:
                target.write_bytes(value["bytes"])
            elif isinstance(value, (bytes, bytearray)):
                target.write_bytes(value)
            elif isinstance(value, str) and not Path(value).is_file():
                import base64
                target.write_bytes(base64.b64decode(value))
            else:
                decoded = decode_image_value(value)
                if isinstance(decoded, str):
                    shutil.copyfile(decoded, target)
                else:
                    decoded.convert("RGB").save(target, format="JPEG", quality=95, subsampling=0)
        yield row, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["lam", "align", "sft", "all"], default="sft")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--bench-root", type=Path, default=Path("benchmark/bench"))
    parser.add_argument("--list", action="store_true", help="List sources and image counts without downloading")
    args = parser.parse_args()
    from lvr.data.records import read_records
    rows = read_records(ROOT / "data/image_manifest")
    groups = defaultdict(list)
    for row in rows:
        if args.stage == "all" or args.stage in row["stages"]:
            groups[row["source"]].append(row)
    cache = args.data_root / "raw"
    for source, records in groups.items():
        print(f"{source}: {len(records)} image pairs", flush=True)
        if args.list:
            continue
        original_cache = cache / "extracted" / source
        if source in BENCHMARKS:
            prepare_benchmark(source, args.bench_root)
            prepared = benchmark_originals(source, records, args.bench_root, original_cache)
        else:
            files = download_source(source, cache / source)
            extract_requested(files, original_cache, {r["source_path"] for r in records}, source)
            prepared = ((row, original_cache / row["source_path"]) for row in records)
        from tqdm import tqdm
        for row, original in tqdm(prepared, total=len(records), desc=source):
            target = args.data_root / row["question_image"]
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            auxiliary = args.data_root / row["auxiliary_image"]
            if not auxiliary.is_file():
                render_auxiliary(target, auxiliary, row["boxes"])
    print("Image preparation complete." if not args.list else "No downloads performed.")


if __name__ == "__main__":
    main()
