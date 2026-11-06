#!/usr/bin/env python3
"""Fetch revision-pinned public weights and the minimal official inference sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from common import HERE, safe_path, sha256, write_json


def check_file(path, spec):
    if not path.is_file() or ("bytes" in spec and path.stat().st_size != spec["bytes"]):
        raise ValueError(f"Missing file or wrong size: {path}")
    if spec.get("sha256"):
        actual = sha256(path)
        expected = spec["sha256"]
    else:
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        expected = spec["git_blob"]
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def verify_asset(root, spec, *, hash_files=True):
    receipt_path = root / "verified.json"
    if not hash_files:
        receipt = json.loads(receipt_path.read_text())
        if receipt["spec"] != spec:
            raise ValueError(f"Downloaded revision differs from models.lock.json: {root}")
    stats = {}
    for entry in spec["files"]:
        path = safe_path(root, entry["path"])
        if hash_files:
            check_file(path, entry)
        stat = path.stat()
        stats[entry["path"]] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not hash_files and stats != receipt["stats"]:
        raise ValueError(f"Assets changed since verification; rerun download_models.py --verify-only: {root}")
    if hash_files:
        write_json(receipt_path, {"spec": spec, "stats": stats})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / ".local/ablation-assets")
    parser.add_argument("--models", nargs="+", default=["ours", "qwen3-base", "monet", "monet-teacher", "lvr", "qwen25-processor"])
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads((HERE / "models.lock.json").read_text())
    for name in ([] if args.sources_only else args.models):
        spec = lock["models"][name]
        target = args.output / "models" / name
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                              local_dir=target, allow_patterns=[v["path"] for v in spec["files"]])
        verify_asset(target, spec)
        print("VERIFIED model", name, spec["revision"], flush=True)
    for name, spec in lock["sources"].items():
        target = args.output / "sources" / name
        for entry in spec["files"]:
            path = safe_path(target, entry["path"])
            if not args.verify_only and not path.exists():
                url = spec["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
                url += f"/{spec['commit']}/{entry['path']}"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while block := response.read(1024**2):
                        output.write(block)
                check_file(temporary, entry)
                temporary.replace(path)
        verify_asset(target, spec)
        print("VERIFIED source", name, spec["commit"], flush=True)


if __name__ == "__main__":
    main()
