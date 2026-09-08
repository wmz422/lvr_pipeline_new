#!/usr/bin/env python3
"""Download a published LVR bundle, optionally at a pinned Hub revision."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("models/lvr-sft-v3-4000"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=args.output)
    print(args.output)


if __name__ == "__main__":
    main()
