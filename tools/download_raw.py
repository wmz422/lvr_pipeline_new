"""Pinned public raw downloads. Curated training annotations come from data-v1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, PROJECT, run, environment
from release import safe_target


def check(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        return False
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {spec['bytes']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            sha.update(block)
            blob.update(block)
    return ((not spec.get('sha256') or sha.hexdigest() == spec['sha256']) and
            (not spec.get('git_blob_sha1') or blob.hexdigest() == spec['git_blob_sha1']))


def download(spec, root, verify_only=False):
    path = safe_target(root, spec['path'])
    if path.exists():
        if not check(path, spec):
            raise ValueError(f'Existing raw file differs: {path}')
        return
    if verify_only:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == spec['bytes'] and check(partial, spec):
            partial.replace(path)
            return
        try:
            request = urllib.request.Request(spec['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Incorrect download resume range')
                with partial.open('ab' if response.status == 206 else 'wb') as output:
                    while block := response.read(4 * 1024**2):
                        output.write(block)
            if not check(partial, spec):
                raise ValueError(f'Raw source checksum mismatch: {spec["path"]}')
            partial.replace(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('training', 'benchmarks'), required=True)
    p.add_argument('--list', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/raw-benchmarks')
    args = p.parse_args()
    if args.workers < 1:
        p.error('Use a positive worker count')
    if args.kind == 'training':
        if args.verify_only:
            p.error('For restored training images use reproduce.py verify-data --groups legacy_images')
        env = environment(gpu=False)
        env.pop('HF_HUB_OFFLINE', None)
        env.pop('TRANSFORMERS_OFFLINE', None)
        run([sys.executable, PROJECT / 'scripts/download_data.py', '--stage', 'sft', '--originals-only',
             '--data-root', PROJECT / 'data', '--bench-root', PROJECT / 'benchmark/bench',
             *(['--list'] if args.list else [])], env=env, cwd=PROJECT)
        return
    lock = json.loads((ROOT / 'manifests/benchmark_downloads.json').read_text())
    if args.list:
        for name, source in lock['sources'].items():
            files = [f for f in lock['files'] if f['benchmark'] == name]
            print(name, source, len(files), sum(f['bytes'] for f in files), 'bytes')
        return
    with ThreadPoolExecutor(args.workers) as pool:
        for index, _ in enumerate(pool.map(lambda spec: download(spec, args.output, args.verify_only), lock['files'])):
            if (index + 1) % 100 == 0:
                print('VERIFIED', index + 1, '/', len(lock['files']), flush=True)
    print('Raw benchmarks verified:', args.output)


if __name__ == '__main__':
    main()

