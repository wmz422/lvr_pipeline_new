#!/usr/bin/env python3
"""Download a published LVR bundle, optionally at a pinned Hub revision."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id")
    parser.add_argument("--base", action="store_true", help="Download the pinned Qwen base model for training")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.base:
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/base_model.json").read_text())
        repo_id = args.repo_id or config["repo_id"]
        revision = args.revision or config["revision"]
        output = args.output or Path("models/Qwen3-VL-4B-Instruct")
    else:
        if not args.repo_id:
            parser.error("Specify --repo-id for an LVR bundle, or --base for the training base model")
        repo_id, revision = args.repo_id, args.revision or "main"
        output = args.output or Path("models") / repo_id.rsplit("/", 1)[-1]
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=output)
    print(output)


if __name__ == "__main__":
    main()
