"""Check that the new launchers preserve the frozen experimental controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Qwen3ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {}
        code = '''
import json, sys
sys.path.insert(0, 'tools')
from common import *
from reproduce import FLOW_ARGS
from start_flow import launcher, bootstrap_binding
from download_models import specs
launch = launcher()
args = launch.parse_args(FLOW_ARGS)
args.defaults = flow_defaults()
args.output = DEFAULT_OUTPUT
plain = reference_recipe()
configs, steps = launch.build_recipes(args, plain, {'train':172224, 'val':7571}, {'protocol_sha256':'same-gt'})
print(json.dumps({'recipes':configs, 'steps':steps, 'output':str(DEFAULT_OUTPUT),
 'model':str(QWEN), 'local':str(LOCAL_FLOW), 'bootstrap':bootstrap_binding(),
 'evaluation':launch.evaluation_command(args, plain, steps), 'download_repo':specs('qwen')[1]}))
'''
        for variant in ('qwen25', 'qwen3', 'qwen3-no-flow-loss'):
            env = dict(os.environ, LVR_FLOW_VARIANT=variant, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
            cls.reports[variant] = json.loads(subprocess.check_output(
                [sys.executable, '-c', code], cwd=ROOT, env=env, text=True))

    @staticmethod
    def normalized(report):
        return json.loads(json.dumps(report['recipes']).replace(report['output'], '${OUTPUT}')
                          .replace(report['model'], '${BACKBONE}'))

    def test_backbone_replacement_preserves_every_other_effective_recipe_field(self):
        self.assertEqual(self.normalized(self.reports['qwen25']), self.normalized(self.reports['qwen3']))
        self.assertEqual(self.reports['qwen3']['download_repo'], 'Qwen/Qwen3-VL-4B-Instruct')

    def test_ablation_changes_only_teacher_flow_loss_weight(self):
        full = self.normalized(self.reports['qwen3'])
        ablation = self.normalized(self.reports['qwen3-no-flow-loss'])
        self.assertEqual(full['teacher_joint']['model']['flow_weight'], .1)
        self.assertEqual(ablation['teacher_joint']['model']['flow_weight'], 0)
        ablation['teacher_joint']['model']['flow_weight'] = .1
        self.assertEqual(full, ablation)

    def test_variants_bind_separate_outputs_and_correct_final_evaluation(self):
        self.assertEqual(len({r['local'] for r in self.reports.values()}), 3)
        self.assertEqual(len({r['output'] for r in self.reports.values()}), 3)
        for variant, report in self.reports.items():
            self.assertEqual(report['steps'], {'teacher_joint':1346, 'warmup':200, 'joint':1346})
            command = report['evaluation']
            self.assertEqual(command[command.index('--model') + 1], report['model'])
            self.assertEqual(command[command.index('--checkpoint') + 1],
                             report['output'] + '/joint/checkpoints/last_full.ckpt')
            self.assertEqual(command[command.index('--source-checkpoint') + 1],
                             report['output'] + '/teacher_joint/checkpoints/student_source.ckpt')
            self.assertEqual(report['bootstrap']['variant'], variant)
            self.assertIn('qwen3_model.json' if variant != 'qwen25' else 'qwen_model.json',
                          report['bootstrap']['locks'])

    def test_shell_entries_preview_correct_model_and_weight_from_another_directory(self):
        for script, weight, output in (('train_qwen3_flow.sh', .1, 'qwen3-flow'),
                                       ('train_qwen3_no_flow_loss.sh', 0, 'qwen3-no-flow-loss')):
            env = dict(os.environ, LVR_TRAIN_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
            report = json.loads(subprocess.check_output(
                ['bash', str(ROOT / script), '--dry-run'], cwd='/tmp', env=env, text=True))
            self.assertTrue(report['base_model'].endswith('/Qwen3-VL-4B-Instruct'))
            self.assertEqual(report['flow_loss_weight'], weight)
            self.assertEqual(report['output'], str(ROOT / 'runs' / output))
            self.assertFalse(report['gpu_started'])
            self.assertFalse(report['downloads_started'])


if __name__ == '__main__':
    unittest.main()
