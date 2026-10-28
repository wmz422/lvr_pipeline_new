"""Rebuild benchmark image assets from pinned raw sources using frozen protocol metadata.

Restore the metadata release first. Only images whose SHA256 is explicitly present
in the frozen records may be materialized. Questions, splits and options stay fixed.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from common import ROOT, SUITE, write_json
from release import safe_target, sha256

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.image'}


def binary_values(value):
    if isinstance(value, bytes):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from binary_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from binary_values(item)
    elif isinstance(value, str) and len(value) > 64 and value.startswith(('/9j/', 'iVBOR', 'UklGR', 'R0lGOD', 'data:image/')):
        try:
            yield base64.b64decode(value.split(',', 1)[-1] if value.startswith('data:') else value, validate=True)
        except ValueError:
            pass


def payloads(path):
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        yield path.read_bytes()
    elif suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    yield archive.read(name)
    elif path.name.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(path, 'r|*') as archive:
            for member in archive:
                if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES:
                    yield archive.extractfile(member).read()
    elif suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                yield from binary_values(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT / 'artifacts/raw-benchmarks')
    p.add_argument('--suite', type=Path, default=SUITE)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    from lvr.eval.paper_benchmark import load_suite
    rows, protocol, manifest = load_suite(args.suite)
    wanted = {im['sha256']: im['path'] for row in rows for im in row['images'] + row['helper_images']}
    missing = {}
    for digest, relative in wanted.items():
        path = safe_target(args.suite, relative)
        if not path.exists():
            missing[digest] = relative
        elif sha256(path) != digest:
            raise ValueError(f'Changed benchmark image: {path}')
    if args.verify_only and missing:
        raise ValueError(f'{len(missing)} missing benchmark images')
    if missing:
        from download_raw import check
        lock = json.loads((ROOT / 'manifests/benchmark_downloads.json').read_text())
        needed_benchmarks = {row['benchmark'] for row in rows
                             if any(im['sha256'] in missing for im in row['images'] + row['helper_images'])}
        for index, spec in enumerate(lock['files']):
            if spec['benchmark'] not in needed_benchmarks:
                continue
            path = safe_target(args.raw, spec['path'])
            if not check(path, spec):
                raise ValueError(f'Raw source is missing or corrupt: {path}')
            for raw in payloads(path):
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in missing:
                    continue
                target = safe_target(args.suite, missing.pop(digest))
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                temporary.write_bytes(raw)
                temporary.replace(target)
            if (index + 1) % 100 == 0:
                print('RAW', index + 1, '/', len(lock['files']), 'remaining', len(missing), flush=True)
        if missing:
            raise ValueError(f'Raw sources did not reproduce {len(missing)} frozen images: {list(missing)[:3]}')
    report = {'status': 'passed', 'records': manifest['records'], 'generations': len(rows),
              'images': len(wanted), 'counts': dict(Counter(row['benchmark'] for row in rows)),
              'records_sha256': sha256(args.suite / 'records.jsonl'), 'mode': 'verify' if args.verify_only else 'rebuild_or_verify'}
    write_json(ROOT / '.local/benchmark_preparation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
