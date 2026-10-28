#!/usr/bin/env python3
"""Download the pinned Zebra-CoT selection and three reasoning benchmarks.

Raw files retain upstream bytes and directory names. This script does not alter
training indexes or register evaluation adapters. It needs only Python's standard
library for downloads; preparation uses the project's datasets/PyArrow/Pillow.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/reasoning_sources.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def check_file(path: Path, entry: dict) -> str:
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"File size mismatch: {path}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {entry['bytes']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            if "git_blob_sha1" in entry:
                git.update(chunk)
    actual = sha.hexdigest()
    if entry.get("sha256", actual) != actual:
        raise ValueError(f"SHA256 mismatch: {path}")
    if entry.get("git_blob_sha1", git.hexdigest()) != git.hexdigest():
        raise ValueError(f"Git blob SHA1 mismatch: {path}")
    return actual


def download(source: dict, entry: dict, destination: Path, verify_only: bool,
             endpoint: str = "https://huggingface.co", direct: bool = False) -> dict:
    target = destination / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        # An existing wrong file is an error, not permission to overwrite it.
        sha = check_file(target, entry)
        return {**entry, "verified_sha256": sha}
    if verify_only:
        raise FileNotFoundError(target)
    partial = target.with_name(target.name + ".partial")
    base = f"{endpoint.rstrip('/')}/datasets/{source['repo_id']}/resolve/{source['revision']}/"
    url = base + urllib.parse.quote(entry["path"], safe="/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    for attempt in range(12):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["bytes"]:
                raise ValueError(f"Oversized partial file: {partial}")
            if offset < entry["bytes"]:
                headers = {"User-Agent": "lvr-data-preparation/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request_url = url + (f"?download=true&retry={time.time_ns()}" if attempt else "?download=true")
                request = urllib.request.Request(request_url, headers=headers)
                with opener.open(request, timeout=60) as response:
                    if response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError(f"Invalid range response: {target}")
                        mode = "ab"
                    elif response.status == 200:
                        mode = "wb"  # Server did not honor Range; restart safely.
                    else:
                        raise ValueError(f"Unexpected HTTP status {response.status}")
                    with partial.open(mode) as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
            if partial.stat().st_size < entry["bytes"]:
                raise OSError("Incomplete response; resume from the retained partial file")
            sha = check_file(partial, entry)
            partial.replace(target)
            print(f"VERIFIED {source['repo_id']}/{entry['path']} ({entry['bytes']:,} bytes)", flush=True)
            return {**entry, "verified_sha256": sha}
        except (OSError, ValueError, http.client.HTTPException) as error:
            # Never print expiring signed redirect URLs or proxy credentials.
            status = getattr(error, "code", "")
            print(f"RETRY {attempt + 1}/12 {entry['path']}: {type(error).__name__} {status}", flush=True)
            if attempt == 11 or isinstance(error, ValueError):
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=("zebra_cot", "mathvision", "mathvista", "visulogic"), default=["mathvision", "mathvista", "visulogic", "zebra_cot"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--bench-root", type=Path, default=ROOT / "benchmark/bench")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"), default="https://huggingface.co", help="Download host; official pinned hashes are always checked")
    parser.add_argument("--direct", action="store_true", help="Bypass environment proxies for this download only")
    parser.add_argument("--verify-only", action="store_true", help="Check local raw files without accessing the network")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sources = {name: manifest["sources"][name] for name in args.sources}
    if args.list:
        for name, source in sources.items():
            print(name, source["revision"], len(source["files"]), sum(f["bytes"] for f in source["files"]), source.get("configs", source.get("splits")))
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    results = {name: [] for name in sources}
    destinations = {name: args.data_root / "raw/Zebra-CoT" if name == "zebra_cot" else args.bench_root / "_raw" / name for name in sources}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download, source, entry, destinations[name], args.verify_only, args.endpoint, args.direct): name for name, source in sources.items() for entry in source["files"]}
        for future in concurrent.futures.as_completed(jobs):
            results[jobs[future]].append(future.result())
    for name, source in sources.items():
        record = {"repo_id": source["repo_id"], "revision": source["revision"], "verified_at": datetime.now(timezone.utc).isoformat(), "files": sorted(results[name], key=lambda f: f["path"])}
        write_json(destinations[name] / "download_manifest.json", record)
        print(f"COMPLETE {name}: {len(record['files'])} verified files at {destinations[name]}", flush=True)


if __name__ == "__main__":
    main()
