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
        for variant in ('qwen25', 'qwen3', 'qwen3-no-flow-loss', 'qwen3-no-latent-align'):
            env = dict(os.environ, LVR_FLOW_VARIANT=variant, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
            # Each child previews its own profile, as the shell entry does.
            env.pop('LVR_FLOW_CONFIG', None)
            cls.reports[variant] = json.loads(subprocess.check_output(
                [sys.executable, '-c', code], cwd=ROOT, env=env, text=True))

    @staticmethod
    def normalized(report):
        return json.loads(json.dumps(report['recipes']).replace(report['output'], '${OUTPUT}')
                          .replace(report['model'], '${BACKBONE}'))

    def test_four_stages_keep_teacher_descriptions_and_disable_final_descriptions(self):
        configs = self.reports['qwen3']['recipes']
        self.assertEqual(list(configs), ['teacher_alignment', 'teacher_joint', 'warmup', 'joint'])
        for phase in ('teacher_alignment', 'teacher_joint'):
            self.assertEqual(configs[phase]['model']['description_weight'], 1)
            self.assertTrue(configs[phase]['data']['include_descriptions'])
            self.assertTrue(configs[phase]['data']['caption_path'].endswith('assets/qwen3_descriptions/captions.jsonl.gz'))
            self.assertEqual(configs[phase]['data']['description_image_mode'], 'none')
        self.assertEqual(configs['teacher_alignment']['model']['qa_loss_weight'], 0)
        self.assertEqual(configs['teacher_joint']['model']['qa_loss_weight'], 1)
        self.assertTrue(configs['teacher_joint']['init_checkpoint'].endswith('/teacher_alignment/checkpoints/last_full.ckpt'))
        for phase in ('warmup', 'joint'):
            self.assertEqual(configs[phase]['model']['description_weight'], 0)
            self.assertFalse(configs[phase]['data']['include_descriptions'])
            self.assertIsNone(configs[phase]['data']['caption_path'])
        self.assertTrue(configs['joint']['init_checkpoint'].endswith('/warmup/checkpoints/last_full.ckpt'))
        self.assertEqual(self.reports['qwen3']['download_repo'], 'Qwen/Qwen3-VL-4B-Instruct')

    def test_ablation_changes_only_teacher_flow_loss_weight(self):
        full = self.normalized(self.reports['qwen3'])
        ablation = self.normalized(self.reports['qwen3-no-flow-loss'])
        for phase in ('teacher_alignment', 'teacher_joint'):
            self.assertEqual(full[phase]['model']['flow_weight'], .1)
            self.assertEqual(ablation[phase]['model']['flow_weight'], 0)
            ablation[phase]['model']['flow_weight'] = .1
        self.assertEqual(full, ablation)

    def test_answer_only_reuses_flow_stage_two_and_has_no_teacher_inputs(self):
        report = self.reports['qwen3-no-latent-align']
        self.assertEqual(report['steps'], {'joint': 1346})
        self.assertEqual(list(report['recipes']), ['joint'])
        config = report['recipes']['joint']
        self.assertEqual(config['model']['latent_weight'], 0)
        self.assertEqual(config['model']['description_weight'], 0)
        self.assertEqual(config['model']['qa_loss_weight'], 1)
        self.assertNotIn('init_checkpoint', config)
        self.assertFalse(config['data']['include_encoder'])
        self.assertTrue(config['data']['predicted_latents_only'])
        self.assertFalse(config['data']['include_descriptions'])
        source = self.reports['qwen3']['output'] + '/teacher_joint/checkpoints/student_source.ckpt'
        self.assertEqual(config['model']['source_checkpoint'], source)
        self.assertEqual(report['evaluation'][report['evaluation'].index('--source-checkpoint') + 1], source)

    def test_variants_bind_separate_outputs_and_correct_final_evaluation(self):
        self.assertEqual(len({r['local'] for r in self.reports.values()}), 4)
        self.assertEqual(len({r['output'] for r in self.reports.values()}), 4)
        for variant, report in self.reports.items():
            expected = {'teacher_joint':1346, 'warmup':200, 'joint':1346}
            if variant in ('qwen3', 'qwen3-no-flow-loss'):
                expected = {'teacher_alignment':1346, **expected}
            elif variant == 'qwen3-no-latent-align':
                expected = {'joint':1346}
            self.assertEqual(report['steps'], expected)
            command = report['evaluation']
            self.assertEqual(command[command.index('--model') + 1], report['model'])
            self.assertEqual(command[command.index('--checkpoint') + 1],
                             report['output'] + '/joint/checkpoints/last_full.ckpt')
            source_output = self.reports['qwen3']['output'] if variant == 'qwen3-no-latent-align' else report['output']
            self.assertEqual(command[command.index('--source-checkpoint') + 1],
                             source_output + '/teacher_joint/checkpoints/student_source.ckpt')
            self.assertEqual(report['bootstrap']['variant'], variant)
            self.assertIn('qwen3_model.json' if variant != 'qwen25' else 'qwen_model.json',
                          report['bootstrap']['locks'])

    def test_shell_entries_preview_correct_model_and_weight_from_another_directory(self):
        for script, weight, output in (('train_qwen3_flow.sh', .1, 'qwen3-flow-4stage'),
                                       ('train_qwen3_no_flow_loss.sh', 0, 'qwen3-no-flow-loss-4stage'),
                                       ('train_qwen3_no_latent_align.sh', 0, 'qwen3-no-latent-align')):
            env = dict(os.environ, LVR_TRAIN_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
            report = json.loads(subprocess.check_output(
                ['bash', str(ROOT / script), '--dry-run'], cwd='/tmp', env=env, text=True))
            self.assertTrue(report['base_model'].endswith('/Qwen3-VL-4B-Instruct'))
            self.assertEqual(report['flow_loss_weight'], weight)
            self.assertEqual(report['output'], str(ROOT / 'runs' / output))
            self.assertFalse(report['gpu_started'])
            self.assertFalse(report['downloads_started'])

    def test_all_three_entries_keep_global_batch_and_step_counts_with_two_four_or_eight_gpus(self):
        for script in ('train_qwen3_flow.sh', 'train_qwen3_no_flow_loss.sh', 'train_qwen3_no_latent_align.sh'):
            for gpus in (2, 4, 8):
                with self.subTest(script=script, gpus=gpus):
                    env = dict(os.environ, LVR_TRAIN_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
                    report = json.loads(subprocess.check_output(
                        ['bash', str(ROOT / script), '--dry-run', '--gpus', str(gpus)],
                        cwd='/tmp', env=env, text=True))
                    for hardware in report['hardware']['phases'].values():
                        self.assertEqual(hardware['gpus'], gpus)
                        self.assertEqual(hardware['global_batch'], 128)
                    command = report['evaluation']
                    self.assertEqual(command[command.index('--mode') + 1], 'predicted')
                    self.assertEqual(int(command[command.index('--gpus') + 1]), gpus)


if __name__ == '__main__':
    unittest.main()
