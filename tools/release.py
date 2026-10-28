"""Portable, checksummed data releases. Standard library only; never uses CUDA."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def safe_target(root, relative):
    rel = PurePosixPath(relative)
    if not rel.parts or rel.is_absolute() or '..' in rel.parts or '\\' in relative:
        raise ValueError(f'Unsafe release path: {relative}')
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Release path escapes root: {relative}')
    return path


class Reader:
    def __init__(self, stream, limiter):
        self.stream, self.limiter = stream, limiter
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size):
        block = self.stream.read(size)
        self.digest.update(block)
        self.count += len(block)
        self.limiter(len(block))
        return block


class Limiter:
    def __init__(self, mib_per_second):
        self.rate = mib_per_second * 1024**2
        self.start, self.count = time.monotonic(), 0

    def __call__(self, count):
        self.count += count
        if self.rate:
            delay = self.count / self.rate - (time.monotonic() - self.start)
            if delay > 0:
                time.sleep(delay)


def pack(inputs, destination, lock_path, *, shard_mib=2048, rate_mib=32, workers=4, resume=False):
    """Inputs are explicit {source,path,group,sha256?}; archive names are portable."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if Path(lock_path).exists():
        raise FileExistsError('Use a new release path; existing releases are immutable')
    rows = sorted(inputs, key=lambda row: (row['group'], row['path']))
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate paths in release')
    limiter = Limiter(rate_mib)
    shards, files, batch, size, group = [], [], [], 0, None
    progress_path = destination / 'pack_progress.json'
    previous = {s['file']: s for s in json.loads(progress_path.read_text())['archives']} if resume and progress_path.exists() else {}

    def read_source(row):
        path = Path(row['source'])
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
            raise ValueError(f'Source changed while packing: {path}')
        digest = hashlib.sha256(raw).hexdigest()
        if row.get('sha256') and row['sha256'] != digest:
            raise ValueError(f'Frozen asset checksum mismatch: {path}')
        return raw, digest

    def flush():
        nonlocal batch, size
        if not batch:
            return
        name = f'{group}-{len(shards):03d}.tar'
        output = destination / name
        if output.exists():
            if name not in previous or not resume:
                raise FileExistsError(output)
            spec = previous[name]
            if output.stat().st_size != spec['bytes'] or sha256(output) != spec['sha256']:
                raise ValueError(f'Corrupt completed archive: {output}')
            expected = {row['path']: row for row in batch}
            found = []
            with tarfile.open(output, 'r|') as archive:
                for member in archive:
                    if not member.isfile() or member.name not in expected:
                        raise ValueError('Existing archive has unexpected entries')
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(4 * 1024**2), b''):
                            digest.update(block)
                    entry = expected[member.name]
                    if entry.get('sha256') and entry['sha256'] != digest.hexdigest():
                        raise ValueError('Existing archive member checksum mismatch')
                    files.append({'path': member.name, 'bytes': member.size, 'sha256': digest.hexdigest(),
                                  'archive': name, 'group': group})
                    found.append(member.name)
            if len(found) != len(expected) or set(found) != set(expected):
                raise ValueError('Existing archive membership differs from inputs')
            shards.append(spec)
            print('REUSED', name, flush=True)
            batch, size = [], 0
            return
        partial = output.with_name(output.name + '.partial')
        members = []
        with tarfile.open(partial, 'w', format=tarfile.USTAR_FORMAT) as archive, ThreadPoolExecutor(workers) as pool:
            pending = deque()
            iterator = iter(batch)
            def submit():
                row = next(iterator, None)
                if row is not None:
                    pending.append((row, pool.submit(read_source, row)))
            for _ in range(workers * 2):
                submit()
            while pending:
                row, future = pending.popleft()
                rel = PurePosixPath(row['path'])
                if rel.is_absolute() or '..' in rel.parts or '\\' in row['path']:
                    raise ValueError('Unsafe archive path')
                raw, digest = future.result()
                info = tarfile.TarInfo(row['path'])
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                archive.addfile(info, io.BytesIO(raw))
                limiter(len(raw))
                files.append({'path': row['path'], 'bytes': info.size, 'sha256': digest,
                              'archive': name, 'group': group})
                members.append(row['path'])
                del raw, future
                submit()
        partial.replace(output)
        shards.append({'file': name, 'bytes': output.stat().st_size, 'sha256': sha256(output),
                       'group': group, 'members': len(members)})
        write_json(destination / 'pack_progress.json', {'archives': shards, 'files': len(files)})
        print(f'PACKED {name}: {len(members)} files, {output.stat().st_size / 2**30:.2f} GiB', flush=True)
        batch, size = [], 0

    for row in rows:
        count = Path(row['source']).stat().st_size
        if batch and (row['group'] != group or size + count > shard_mib * 1024**2):
            flush()
        group = row['group']
        batch.append(row)
        size += count
    flush()
    manifest = Path(lock_path).with_suffix('.files.jsonl.gz')
    raw = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in files).encode()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    lock = {'schema': 'qwen25-data-release-v1', 'files_manifest': manifest.name,
            'files_manifest_sha256': sha256(manifest), 'file_count': len(files),
            'payload_bytes': sum(row['bytes'] for row in files), 'archives': shards,
            'public_base_url': None}
    write_json(lock_path, lock)
    shutil.copyfile(lock_path, destination / Path(lock_path).name)
    shutil.copyfile(manifest, destination / manifest.name)
    print(json.dumps({'status': 'complete', 'files': len(files), 'payload_bytes': lock['payload_bytes']}), flush=True)


def load_lock(path):
    path = Path(path)
    lock = json.loads(path.read_text())
    if lock['schema'] != 'qwen25-data-release-v1':
        raise ValueError('Unsupported release schema')
    manifest = safe_target(path.parent, lock['files_manifest'])
    if sha256(manifest) != lock['files_manifest_sha256']:
        raise ValueError('Release file manifest checksum mismatch')
    with gzip.open(manifest, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != lock['file_count'] or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Incorrect or duplicate file manifest entries')
    return lock, rows


def fetch_archive(source, spec, cache):
    target = safe_target(cache, spec['file'])
    if target.is_file():
        if target.stat().st_size != spec['bytes'] or sha256(target) != spec['sha256']:
            raise ValueError(f'Corrupt existing archive: {target}')
        return target
    local = Path(source).expanduser() / spec['file']
    if not source.startswith(('https://', 'http://')):
        if local.stat().st_size != spec['bytes'] or sha256(local) != spec['sha256']:
            raise ValueError(f'Corrupt source archive: {local}')
        return local
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(source.rstrip('/') + '/' + urllib.parse.quote(spec['file']),
                                     headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Incorrect resume range')
        mode = 'ab' if response.status == 206 else 'wb'
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, 4 * 1024**2)
    if partial.stat().st_size != spec['bytes'] or sha256(partial) != spec['sha256']:
        raise ValueError('Incomplete or corrupt download; rerun to resume')
    partial.replace(target)
    return target


def unpack(archive, entries, root):
    expected = {row['path']: row for row in entries}
    seen = set()
    with tarfile.open(archive, 'r:*') as stream:
        for member in stream:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f'Unexpected archive member: {member.name}')
            spec = expected[member.name]
            if member.size != spec['bytes']:
                raise ValueError(f'Wrong archive member size: {member.name}')
            seen.add(member.name)
            target = safe_target(root, member.name)
            if target.exists():
                if target.stat().st_size != spec['bytes'] or sha256(target) != spec['sha256']:
                    raise ValueError(f'Refuse to overwrite different data: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            digest = hashlib.sha256()
            with stream.extractfile(member) as source, partial.open('wb') as output:
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != spec['sha256']:
                partial.unlink()
                raise ValueError(f'Asset checksum mismatch: {member.name}')
            partial.replace(target)
    if seen != expected.keys():
        raise ValueError('Incomplete archive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--rate-mib', type=float, default=32)
    p.add_argument('--shard-mib', type=int, default=2048)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    for action in ('restore', 'verify'):
        p = sub.add_parser(action)
        p.add_argument('--lock', type=Path, default=ROOT / 'manifests/data_release.json')
        p.add_argument('--root', type=Path, default=ROOT)
        p.add_argument('--groups', nargs='+')
        if action == 'restore':
            p.add_argument('--source', required=True, help='Local release directory or HTTPS base URL')
    args = parser.parse_args()
    if args.action == 'pack':
        if args.rate_mib < 0 or args.shard_mib < 1 or args.workers < 1:
            parser.error('Invalid rate or shard size')
        pack(json.loads(args.inputs.read_text()), args.output, args.lock,
             shard_mib=args.shard_mib, rate_mib=args.rate_mib, workers=args.workers, resume=args.resume)
        return
    lock, rows = load_lock(args.lock)
    if args.groups:
        unknown = set(args.groups) - {r['group'] for r in rows}
        if unknown:
            parser.error(f'Unknown groups: {sorted(unknown)}')
        rows = [r for r in rows if r['group'] in args.groups]
    if args.action == 'restore':
        for spec in lock['archives']:
            entries = [r for r in rows if r['archive'] == spec['file']]
            if not entries:
                continue
            path = fetch_archive(args.source, spec, args.root / 'artifacts/downloads')
            unpack(path, entries, args.root)
            print('RESTORED', spec['file'], flush=True)
    else:
        for index, row in enumerate(rows):
            path = safe_target(args.root, row['path'])
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError(f'File checksum mismatch: {path}')
            if (index + 1) % 10000 == 0:
                print('VERIFIED', index + 1, '/', len(rows), flush=True)
    print(json.dumps({'status': 'passed', 'files': len(rows), 'action': args.action}))


if __name__ == '__main__':
    main()
