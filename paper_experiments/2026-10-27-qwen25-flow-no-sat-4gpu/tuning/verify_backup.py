"""Verify every payload byte before restoring one completed backup snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_snapshot(snapshot, workers=4):
    snapshot = Path(snapshot)
    if snapshot.is_symlink():
        raise ValueError('Snapshot cannot be a symlink')
    manifest_path = snapshot / 'backup_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('Manifest cannot be a symlink')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'flow-checkpoint-backup-v1' or not manifest.get('files'):
        raise ValueError('Unrecognized or empty backup snapshot')
    names = set(manifest['files'])
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
            raise ValueError('Invalid snapshot payload path')
    entries = list(snapshot.rglob('*'))
    if any(path.is_symlink() for path in entries):
        raise ValueError('Snapshot contains a symlink')
    actual = {path.relative_to(snapshot).as_posix() for path in entries
              if path.is_file() and path != manifest_path}
    if actual != names:
        raise ValueError('Backup payload inventory differs from manifest')
    def check(name):
        path, record = snapshot / name, manifest['files'][name]
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 2**20), b''):
                digest.update(chunk)
        after = path.stat()
        if ((before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns)
                or after.st_size != record['bytes'] or digest.hexdigest() != record['sha256']):
            raise ValueError(f'Backup payload corrupt or changed: {name}')
        return after.st_size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(check, sorted(names)))
    if total != manifest['copied_bytes']:
        raise ValueError('Backup total byte count differs')
    return {'status': 'verified', 'snapshot': str(snapshot), 'identity': manifest['identity'],
            'original_source': manifest['source'], 'files': len(names), 'bytes': total}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1: parser.error('Use a positive worker count')
    print(json.dumps(verify_snapshot(args.snapshot, args.workers), indent=2))
