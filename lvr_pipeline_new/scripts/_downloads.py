"""Pinned downloads and benchmark preparation shared by the public scripts."""

import hashlib
import json
from pathlib import Path
import subprocess
import shutil


ROOT = Path(__file__).resolve().parents[1]
BLINK_CONFIGS = ["Counting", "IQ_Test", "Jigsaw", "Relative_Reflectance", "Spatial_Relation"]
BENCHMARKS = ["vstar", "mmvp", "blink", "hr_bench_4k", "hr_bench_8k", "mme_realworld_lite"]


def sources():
    return json.loads((ROOT / "data/sources.json").read_text())["sources"]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download_source(name, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec = sources()[name]
    if "repo_id" in spec:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                          allow_patterns=[f["path"] for f in spec["files"]],
                          local_dir=destination, max_workers=4)
    else:
        for info in spec["files"]:
            path = destination / info["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != info["bytes"]:
                subprocess.run(["curl", "--fail", "--location", "--retry", "8", "--retry-delay", "5",
                                "--connect-timeout", "30", "--continue-at", "-", "--output", str(path), info["url"]], check=True)
    files = []
    for info in spec["files"]:
        path = destination / info["path"]
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise RuntimeError(f"Incomplete download: {path}")
        if info.get("sha256") and digest(path) != info["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        files.append(path)
    return files


def prepare_benchmark(name, root):
    from datasets import Dataset, DatasetDict, load_from_disk
    root = Path(root)
    folder = {"vstar": "vstar_bench", "mmvp": "MMVP", "blink": "BLINK"}.get(name, name)
    target = root / folder
    spec = sources()[name]
    stamp = target / "download_manifest.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("revision") == spec["revision"]:
        return target
    if name in {"vstar", "mmvp"}:
        download_source(name, target)
        if name == "vstar":
            count = len((target / "test_questions.jsonl").read_text().splitlines())
        else:
            import csv
            with (target / "Questions.csv").open() as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            image_dir = target / "MMVP_Images"
            image_dir.mkdir(exist_ok=True)
            for image in (target / "MMVP Images").glob("*.jpg"):
                if not (image_dir / image.name).is_file():
                    shutil.copyfile(image, image_dir / image.name)
            if len(list(image_dir.glob("*.jpg"))) != 300:
                raise RuntimeError("MMVP needs all 300 images")
        if count != {"vstar": 191, "mmvp": 300}[name]:
            raise RuntimeError(f"Unexpected {name} sample count: {count}")
    else:
        files = download_source(name, root / "_raw" / name)
        if name == "blink":
            count = {}
            for config in BLINK_CONFIGS:
                parts = {}
                for split in ("val", "test"):
                    selected = [str(f) for f in files if f.parent.name == config and f.name.startswith(split + "-")]
                    parts[split] = Dataset.from_parquet(selected)
                DatasetDict(parts).save_to_disk(target / config)
                count[config] = len(parts["val"])
            if sum(count.values()) != 697:
                raise RuntimeError(f"Unexpected BLINK validation counts: {count}")
        else:
            dataset = Dataset.from_parquet([str(f) for f in files])
            count = len(dataset)
            expected = 1919 if name == "mme_realworld_lite" else 800
            if count != expected:
                raise RuntimeError(f"Unexpected {name} sample count: {count}")
            dataset.save_to_disk(target)
    target.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"repo_id": spec["repo_id"], "revision": spec["revision"], "samples": count}, indent=2) + "\n")
    return target
