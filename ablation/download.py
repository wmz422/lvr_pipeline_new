#!/usr/bin/env python3
"""Download only the frozen ablation benchmarks, or restore already downloaded shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from common import HERE, ROOT, load_data

sys.path.insert(0, str(ROOT / "tools"))
from release import fetch_archive, load_lock, restore_source, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument("--source", help="Already downloaded archive directory or HTTPS base URL")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/ablation-downloads")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        lock_path = HERE / "benchmark_release.json"
        lock, rows = load_lock(lock_path)
        source = restore_source(args.source, lock_path, HERE / "publication.json")
        for spec in lock["archives"]:
            archive = fetch_archive(source, spec, args.cache)
            unpack(archive, [row for row in rows if row["archive"] == spec["file"]], args.output)
            print("RESTORED", spec["file"], flush=True)
    rows, _, _, _ = load_data(args.output)
    print(json.dumps({"status": "passed", "records": len(rows), "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
