"""Download the exact Qwen/FLUX files; verify every byte against the snapshot lock."""
import argparse
import json
from pathlib import Path
import shutil
import os

from common import ROOT, QWEN, FLUX
from release import sha256


def specs(name):
    lock_path = ROOT / 'manifests' / f'{name}_model.json'
    lock = json.loads(lock_path.read_text())
    if name == 'qwen':
        return lock_path, lock['repository'], lock['revision'], {
            row['name']: {'bytes': row['bytes'], 'sha256': row['sha256']}
            for row in lock['verified_files']}, QWEN, 'source.json'
    return lock_path, lock['repo_id'], lock['revision'], lock['files'], FLUX, 'weights_lock.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('qwen', 'flow'), default=['qwen', 'flow'])
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    for name in args.models:
        lock_path, repo, revision, files, target, lock_name = specs(name)
        print(repo, revision, sum(f['bytes'] for f in files.values()), 'bytes', flush=True)
        if args.list:
            continue
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, revision=revision, allow_patterns=list(files),
                              local_dir=target, max_workers=2)
        for relative, spec in files.items():
            path = target / relative
            if not path.is_file() or path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
                raise ValueError(f'Missing or changed model component: {path}')
        if not args.verify_only:
            shutil.copyfile(lock_path, target / lock_name)
        print('VERIFIED', repo, flush=True)


if __name__ == '__main__':
    main()

