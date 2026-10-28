import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tuning.backup_checkpoints import copy_snapshot, prune_superseded_recoveries, candidates
from tuning.verify_backup import verify_snapshot


class BackupTests(unittest.TestCase):
    def test_cursor_at_checkpoint_step_does_not_prove_save_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase = root / 'teacher_joint'
            checkpoint = phase / 'checkpoints/candidate-000400.ckpt'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'rank0.pt').write_bytes(b'save still in progress')
            (phase / 'training_started.json').write_text(json.dumps({'recipe_sha256': 'recipe'}))
            def cursors(steps, status='training'):
                for rank, step in enumerate(steps):
                    (phase / f'runtime_rank{rank}.json').write_text(json.dumps({
                        'optimizer_steps': step, 'status': status}))
            cursors([400] * 4)
            self.assertEqual(list(candidates(root)), [])
            cursors([410, 410, 410, 400])
            self.assertEqual(list(candidates(root)), [])
            cursors([410] * 4)
            self.assertEqual([entry[1] for entry in candidates(root)], [checkpoint])
            cursors([400] * 4, 'training_complete')
            self.assertEqual(list(candidates(root)), [])
            (phase / 'complete.json').write_text(json.dumps({
                'recipe_sha256': 'recipe', 'optimizer_steps': 400, 'status': 'complete'}))
            self.assertEqual([entry[1] for entry in candidates(root)], [checkpoint])

    def test_new_source_shard_during_copy_prevents_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'rank0.pt').write_bytes(b'payload')
            destination = root / 'backup'
            with patch('tuning.backup_checkpoints.time.sleep',
                       side_effect=lambda _: (source / 'rank1.pt').write_bytes(b'late shard')):
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    copy_snapshot(source, destination, bytes_per_second=1, identity={'step': 1})
            self.assertFalse(destination.exists())

    def test_interrupted_manifest_write_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'rank.pt').write_bytes(b'checkpoint payload' * 100)
            destination = root / 'recovery-000400.ckpt'
            staging = root / 'recovery-000400.ckpt.copying'
            staging.mkdir()
            (staging / 'backup_manifest.json.partial').write_text('{interrupted')
            identity = {'recipe_sha256': 'recipe', 'step': 400, 'name': destination.name}
            copy_snapshot(source, destination, bytes_per_second=1e9, identity=identity)
            self.assertEqual((destination / 'rank.pt').read_bytes(), (source / 'rank.pt').read_bytes())
            self.assertEqual(json.loads((destination / 'backup_manifest.json').read_text())['identity'], identity)
            self.assertFalse(staging.exists())
            verified = verify_snapshot(destination)
            self.assertEqual(verified['identity'], identity)
            self.assertEqual(verified['files'], 1)
            (destination / 'rank.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupt or changed'):
                verify_snapshot(destination)

    def test_recovery_retention_preserves_candidates_and_other_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, recipe, step in [('recovery-000400.ckpt', 'recipe', 400),
                                       ('recovery-000800.ckpt', 'recipe', 800),
                                       ('recovery-000100.ckpt', 'other', 100),
                                       ('candidate-000200.ckpt', 'recipe', 200)]:
                path = root / name
                path.mkdir()
                (path / 'backup_manifest.json').write_text(json.dumps({
                    'format': 'flow-checkpoint-backup-v1',
                    'identity': {'recipe_sha256': recipe, 'step': step, 'name': name}}))
            prune_superseded_recoveries(root, {'recipe_sha256': 'recipe', 'step': 800,
                                              'name': 'recovery-000800.ckpt'})
            self.assertEqual({path.name for path in root.iterdir()},
                             {'recovery-000800.ckpt', 'recovery-000100.ckpt', 'candidate-000200.ckpt'})
            with self.assertRaisesRegex(ValueError, 'not been committed'):
                prune_superseded_recoveries(root, {'recipe_sha256': 'recipe', 'step': 1200,
                                                  'name': 'recovery-001200.ckpt'})
            self.assertTrue((root / 'recovery-000800.ckpt').exists())


if __name__ == '__main__':
    unittest.main()
