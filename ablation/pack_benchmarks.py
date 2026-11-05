#!/usr/bin/env python3
"""Maintainer: package the accepted inputs and publish an immutable HF dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from common import HERE, ROOT, COUNTS, sha256, write_json

sys.path.insert(0, str(ROOT / "tools"))
from release import load_lock, pack

REPO_ID = "Wing22/lvr-latent-ablation-benchmarks-v1"


def prepare(args):
    staging = ROOT / ".local/ablation-release/suite"
    staging.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.base_data / "manifest.json").read_text())
    for name, expected in original["files"].items():
        if sha256(args.base_data / name) != expected:
            raise ValueError(f"Original frozen metadata changed: {name}")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base_data / name, target)
    # The historical base suite has a task-grouped candidate; replace it explicitly.
    shutil.copyfile(args.global_donors, staging / "donors.json")
    original["files"]["donors.json"] = sha256(staging / "donors.json")
    original["release"] = {"version": "latent-five-mode-global-2026-11-05-v1",
                           "source_manifest_sha256": sha256(args.base_data / "manifest.json"),
                           "shared_shuffle_scope": "global", "shuffle_seed": 42}
    write_json(staging / "manifest.json", original)
    assets = json.loads((staging / "asset_manifest.json").read_text())
    # A temporary directory symlink avoids copying 1.3 GB before creating real tar members.
    asset_dir = staging / "assets"
    if not asset_dir.exists():
        asset_dir.symlink_to((args.base_data / "assets").resolve(), target_is_directory=True)
    inputs = [{"source": str(staging / name), "path": name, "group": "metadata"}
              for name in [*original["files"], "manifest.json"]]
    inputs += [{"source": str(args.base_data / name), "path": name, "group": "images",
                "sha256": image["sha256"]} for name, image in sorted(assets.items())]
    if args.lock.exists():
        lock, entries = load_lock(args.lock)
        if {row["path"] for row in entries} != {row["path"] for row in inputs}:
            raise ValueError("Existing release has different contents")
        for spec in lock["archives"]:
            if sha256(args.directory / spec["file"]) != spec["sha256"]:
                raise ValueError("Existing archive changed")
    else:
        pack(inputs, args.directory, args.lock, shard_mib=512, rate_mib=0, workers=4, resume=True)
    write_json(HERE / "benchmark.json", {"schema": "latent-ablation-benchmark-v1", "repo_id": REPO_ID,
               "suite_manifest_sha256": sha256(staging / "manifest.json"),
               "donors_sha256": sha256(args.global_donors), "benchmarks": COUNTS,
               "records": sum(COUNTS.values()), "images": len(assets)})
    # Do not dereference a staging symlink during path-confinement checks.
    # Full validation is run on an independently restored release instead.
    shutil.copyfile(HERE / "dataset_card.md", args.directory / "README.md")
    lock, _ = load_lock(args.lock)
    names = [spec["file"] for spec in lock["archives"]] + [args.lock.name, lock["files_manifest"], "README.md"]
    checksums = {name: sha256(args.directory / name) for name in names}
    (args.directory / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())))
    return [*names, "SHA256SUMS"]


def publish(args, names):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    identity = api.whoami()["name"]
    if identity.lower() != REPO_ID.split("/")[0].lower():
        raise ValueError(f"Log in to the dataset owner account {REPO_ID.split('/')[0]}")
    state_path = ROOT / ".local/ablation-release/hf_state.json"
    binding = {"repo_id": REPO_ID, "release_sha256": sha256(args.lock)}
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    try:
        info = api.repo_info(REPO_ID, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=False)
        write_json(state_path, binding)
        state = binding
        info = api.repo_info(REPO_ID, repo_type="dataset")
    if info.private:
        raise ValueError("The target exists as a private dataset; no visibility change was made")
    existing = {item.rfilename for item in info.siblings}
    if existing - set(names) - {".gitattributes"}:
        raise ValueError("Refuse to mix this immutable release with unrelated existing files")
    if args.lock.name in existing:
        remote = Path(hf_hub_download(REPO_ID, args.lock.name, repo_type="dataset", revision=info.sha))
        if sha256(remote) != sha256(args.lock):
            raise ValueError("Refuse to overwrite another frozen release")
    elif state != binding:
        raise ValueError("Existing dataset is not a recognized publication of this release")
    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=args.directory,
                      allow_patterns=names, commit_message="Publish four-benchmark five-mode latent ablation inputs")
    # Verify public access, all byte sizes and hashes at one pinned snapshot.
    public = HfApi(token=False)
    info = public.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    for name in names:
        local = args.directory / name
        if name not in remote or remote[name].size != local.stat().st_size:
            raise ValueError(f"Remote file is missing or incomplete: {name}")
        if remote[name].lfs:
            if remote[name].lfs.sha256 != sha256(local):
                raise ValueError(f"Remote SHA256 differs: {name}")
        else:
            path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=info.sha, token=False))
            if sha256(path) != sha256(local):
                raise ValueError(f"Remote metadata differs: {name}")
    write_json(HERE / "publication.json", {"huggingface": {"repo_id": REPO_ID, "status": "complete",
               "private": False, "revision": info.sha, "release_sha256": sha256(args.lock),
               "download_base_url": f"https://huggingface.co/datasets/{REPO_ID}/resolve/{info.sha}",
               "url": f"https://huggingface.co/datasets/{REPO_ID}", "files_verified": len(names)}})
    print("PUBLISHED", REPO_ID, info.sha, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path)
    parser.add_argument("--global-donors", type=Path)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/latent-ablation-v1")
    parser.add_argument("--lock", type=Path, default=HERE / "benchmark_release.json")
    parser.add_argument("--publish", action="store_true", help="Upload the already prepared package using the logged-in HF account")
    args = parser.parse_args()
    if args.publish:
        lock, _ = load_lock(args.lock)
        names = [spec["file"] for spec in lock["archives"]] + [args.lock.name, lock["files_manifest"], "README.md", "SHA256SUMS"]
        for spec in lock["archives"]:
            if sha256(args.directory / spec["file"]) != spec["sha256"]:
                raise ValueError("Local archive changed")
        publish(args, names)
    else:
        if args.base_data is None or args.global_donors is None:
            parser.error("Packaging requires --base-data and --global-donors")
        names = prepare(args)
        print(json.dumps({"status": "prepared_not_uploaded", "files": names}), flush=True)


if __name__ == "__main__":
    main()
