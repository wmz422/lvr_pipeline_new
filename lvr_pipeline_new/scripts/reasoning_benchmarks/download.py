"""Download pinned benchmark files with upstream blob/LFS integrity checks."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "benchmark/bench/_raw/reasoning_v1"
SOURCES = {
    "MIRA": ("YiyangAiLab/MIRA", "43a43dc1d48b695e93168a7d9759362527a7020b"),
    "BLINK": ("BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040"),
}
SCORER_SOURCES = {
    "MathVista": ("lupantech/MathVista", "c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb",
                  ["evaluation/calculate_score.py", "evaluation/extract_answer.py"]),
    "MIRA": ("aiming-lab/MIRA", "200d9b57e93c8cf07f3011df35c540da0a9162b5", ["acc.py", "eval_api.py"]),
}


def fetch(url, *, method="GET"):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method=method)
            with opener.open(request, timeout=35) as response:
                return response.read(), dict(response.headers)
        except (OSError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 8))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify(data, entry):
    if len(data) != entry["bytes"]:
        raise ValueError(f"Size mismatch: {entry['path']}")
    sha = hashlib.sha256(data).hexdigest()
    if "sha256" in entry and sha != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch: {entry['path']}")
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if "git_blob_sha1" in entry and blob != entry["git_blob_sha1"]:
        raise ValueError(f"Git blob mismatch: {entry['path']}")
    return sha


def download_scorers(suite):
    directory = suite / "upstream"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (repo, revision, required) in SCORER_SOURCES.items():
        tree_path = directory / (name + "_tree.json")
        if tree_path.exists():
            tree = json.loads(tree_path.read_text())
        else:
            payload, _ = fetch(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")
            tree = json.loads(payload)
            write_json(tree_path, tree)
        if tree["sha"] != revision or tree.get("truncated"):
            raise ValueError("Scorer tree version mismatch")
        entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        paths = required + [p for p in ("LICENSE", "LICENSE.md", "LICENSE.txt") if p in entries]
        for filename in paths:
            entry = entries[filename]
            target = directory / name / filename
            if target.exists():
                payload = target.read_bytes()
            else:
                data, _ = fetch(f"https://api.github.com/repos/{repo}/git/blobs/{entry['sha']}")
                payload = base64.b64decode(json.loads(data)["content"])
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if blob != entry["sha"]:
                raise ValueError(f"Scorer blob mismatch: {repo}/{filename}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        print("VERIFIED SCORER", name, revision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--suite", type=Path, default=ROOT/"benchmark/reasoning_v1")
    parser.add_argument("--scorers-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    download_scorers(args.suite)
    if args.scorers_only:
        return
    for name in args.sources:
        repo, revision = SOURCES[name]
        dest = RAW / name
        lock_path = dest / "source_lock.json"
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
            if (lock["repo_id"], lock["revision"]) != (repo, revision):
                raise ValueError("Existing source lock changed")
        else:
            data, _ = fetch(f"{args.endpoint}/api/datasets/{repo}/revision/{revision}?blobs=1")
            info = json.loads(data)
            if info["sha"] != revision:
                raise ValueError("Upstream revision mismatch")
            entries = []
            for item in info["siblings"]:
                path = item["rfilename"]
                selected = (path.endswith((".png", ".jpg", ".jpeg", ".jsonl", ".md")) if name == "MIRA"
                            else path == "README.md" or path in [f"{task}/val-00000-of-00001.parquet"
                            for task in ("Jigsaw", "Multi-view_Reasoning", "Spatial_Relation")])
                if not selected:
                    continue
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError("Unsafe upstream path")
                entry = {"path": path, "bytes": item["size"]}
                if item.get("lfs"):
                    entry["sha256"] = item["lfs"]["sha256"]
                else:
                    entry["git_blob_sha1"] = item["blobId"]
                entries.append(entry)
            lock = {"repo_id": repo, "revision": revision, "files": entries}
            write_json(lock_path, lock)

        def one(entry):
            path = dest / entry["path"]
            if path.exists():
                payload = path.read_bytes()
            else:
                url = f"{args.endpoint}/datasets/{repo}/resolve/{revision}/" + urllib.parse.quote(entry["path"], safe="/")
                payload, _ = fetch(url)
            sha = verify(payload, entry)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".partial")
                temp.write_bytes(payload)
                temp.replace(path)
            return {**entry, "verified_sha256": sha}

        files = []
        with ThreadPoolExecutor(args.workers) as pool:
            for entry in pool.map(one, lock["files"]):
                files.append(entry)
                if len(files) % 100 == 0:
                    print(name, len(files), "/", len(lock["files"]), flush=True)
        write_json(dest / "download_manifest.json", {**lock, "files": files})
        print("COMPLETE", name, len(files), sum(e["bytes"] for e in files), flush=True)


if __name__ == "__main__":
    main()
