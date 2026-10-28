import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from release import pack, load_lock, unpack, safe_target, fetch_archive, sha256
from download_raw import check
from prepare_benchmarks import binary_values


class ReleaseTests(unittest.TestCase):
    def test_pack_restore_relocation_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.bin'
            source.write_bytes(bytes(range(256)) * 100)
            pack([{'source': str(source), 'path': 'data/images/one.bin', 'group': 'images'}],
                 root / 'archive', root / 'lock.json', shard_mib=1, rate_mib=0)
            lock, rows = load_lock(root / 'lock.json')
            archive = fetch_archive(str(root / 'archive'), lock['archives'][0], root / 'cache')
            restored = root / 'different-directory'
            unpack(archive, rows, restored)
            self.assertEqual((restored / rows[0]['path']).read_bytes(), source.read_bytes())
            unpack(archive, rows, restored)  # idempotent restoration
            (restored / rows[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                unpack(archive, rows, restored)
            archive.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                fetch_archive(str(root / 'archive'), lock['archives'][0], root / 'cache')

    def test_traversal_symlinks_and_extra_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../escape', '/absolute', 'bad\\path'):
                with self.assertRaises(ValueError):
                    safe_target(root, name)
            (root / 'escape').symlink_to(root.parent)
            with self.assertRaises(ValueError):
                safe_target(root, 'escape/outside')
            path = root / 'malicious.tar'
            with tarfile.open(path, 'w') as archive:
                info = tarfile.TarInfo('data/link')
                info.type = tarfile.SYMTYPE
                info.linkname = '/tmp'
                archive.addfile(info)
            with self.assertRaises(ValueError):
                unpack(path, [{'path': 'data/link', 'bytes': 0}], root / 'destination')

    def test_manifest_and_download_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw'
            payload = b'fixed public input'
            path.write_bytes(payload)
            spec = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                    'git_blob_sha1': hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()}
            self.assertTrue(check(path, spec))
            path.write_bytes(b'X' * len(payload))
            self.assertFalse(check(path, spec))

    def test_pack_resume_preserves_completed_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'first', root / 'second'
            first.write_bytes(b'first payload')
            second.write_bytes(b'second payload')
            rows = [{'source': str(first), 'path': 'data/first', 'group': 'a'}]
            pack(rows, root / 'archives', root / 'lock.json', rate_mib=0)
            before = sha256(root / 'archives/a-000.tar')
            (root / 'lock.json').unlink()  # simulate interruption before final release publication
            rows.append({'source': str(second), 'path': 'data/second', 'group': 'b'})
            pack(rows, root / 'archives', root / 'lock.json', rate_mib=0, resume=True)
            lock, entries = load_lock(root / 'lock.json')
            self.assertEqual(before, sha256(root / 'archives/a-000.tar'))
            self.assertEqual(len(entries), 2)

    def test_nested_parquet_image_payloads(self):
        self.assertEqual(list(binary_values({'images': [{'bytes': b'a'}, {'bytes': b'b'}],
                                             'question': 'ordinary text'})), [b'a', b'b'])


if __name__ == '__main__':
    unittest.main()
