"""Check that the new launchers preserve the frozen experimental controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
        for variant in ('qwen25', 'qwen3', 'qwen3-no-flow-loss', 'qwen3-no-latent-align',
                        'qwen3-latent-4', 'qwen3-latent-16'):
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
        self.assertEqual(len({r['local'] for r in self.reports.values()}), 6)
        self.assertEqual(len({r['output'] for r in self.reports.values()}), 6)
        for variant, report in self.reports.items():
            expected = {'teacher_joint':1346, 'warmup':200, 'joint':1346}
            if variant != 'qwen25' and variant != 'qwen3-no-latent-align':
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

    def test_length_ablation_changes_only_the_number_of_slots_from_standard_flow(self):
        full = self.normalized(self.reports['qwen3'])
        for length in (4, 16):
            with self.subTest(num_latents=length):
                report = self.reports[f'qwen3-latent-{length}']
                configs = self.normalized(report)
                for phase, config in configs.items():
                    self.assertEqual(config['data']['num_latents'], length)
                    config['data']['num_latents'] = 8
                    if phase.startswith('teacher_'):
                        self.assertEqual(config['model']['num_latents'], length)
                        config['model']['num_latents'] = 8
                self.assertEqual(configs, full)
                self.assertEqual(report['steps'], self.reports['qwen3']['steps'])

    @staticmethod
    def length_previews(*arguments):
        env = dict(os.environ, LVR_TRAIN_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
        output = subprocess.check_output(
            ['bash', str(ROOT / 'train_qwen3_latent_length.sh'), '--dry-run', *arguments],
            cwd='/tmp', env=env, text=True)
        decoder, reports = json.JSONDecoder(), []
        while output.strip():
            report, end = decoder.raw_decode(output.lstrip())
            reports.append(report)
            output = output.lstrip()[end:]
        return reports

    def test_length_entry_trains_and_evaluates_both_lengths_with_independent_outputs(self):
        for gpus in (2, 4, 8):
            with self.subTest(gpus=gpus):
                reports = self.length_previews('--gpus', str(gpus))
                self.assertEqual([r['num_latents'] for r in reports], [4, 16])
                for report in reports:
                    length = report['num_latents']
                    self.assertEqual(report['output'], str(ROOT / f'runs/qwen3-flow-4stage-latent-{length}'))
                    self.assertEqual(report['stages'], ['teacher_alignment', 'teacher_joint', 'warmup', 'joint'])
                    self.assertEqual(report['flow_loss_weight'], .1)
                    self.assertFalse(report['gpu_started'])
                    self.assertFalse(report['downloads_started'])
                    for hardware in report['hardware']['phases'].values():
                        self.assertEqual(hardware['gpus'], gpus)
                        self.assertEqual(hardware['global_batch'], 128)
                    command = report['evaluation']
                    self.assertEqual(command[command.index('--checkpoint') + 1],
                                     report['output'] + '/joint/checkpoints/last_full.ckpt')
                    self.assertEqual(command[command.index('--source-checkpoint') + 1],
                                     report['output'] + '/teacher_joint/checkpoints/student_source.ckpt')
                    self.assertEqual(int(command[command.index('--gpus') + 1]), gpus)

    def test_length_entry_separates_custom_parent_paths_and_forwards_eval_hardware(self):
        for option in (['--output', 'runs/length test'], ['--output=runs/length test']):
            reports = self.length_previews(*option, '--eval-gpus', '2', '--resume')
            self.assertEqual([r['output'] for r in reports],
                             [str(ROOT / f'runs/length test/latent-{n}') for n in (4, 16)])
            for report in reports:
                command = report['evaluation']
                self.assertEqual(command[command.index('--gpus') + 1], '2')

    def test_length_entry_forwards_resume_and_stops_if_a_run_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            journal, stub = directory / 'calls.jsonl', directory / 'python'
            stub.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
with open(os.environ['LVR_TEST_JOURNAL'], 'a') as stream:
    stream.write(json.dumps({'variant':os.environ['LVR_FLOW_VARIANT'], 'args':sys.argv[1:]}) + '\\n')
sys.exit(9 if os.environ['LVR_FLOW_VARIANT'] == os.environ.get('LVR_TEST_FAIL_VARIANT') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, LVR_TRAIN_PYTHON=str(stub), LVR_TEST_JOURNAL=str(journal))
            command = ['bash', str(ROOT / 'train_qwen3_latent_length.sh'), '--gpus', '2',
                       '--resume', '--eval-only', '--output', 'runs/length test']
            for failure, expected_variants in (('', ['qwen3-latent-4', 'qwen3-latent-16']),
                                                ('qwen3-latent-4', ['qwen3-latent-4']),
                                                ('qwen3-latent-16', ['qwen3-latent-4', 'qwen3-latent-16'])):
                with self.subTest(failure=failure):
                    journal.unlink(missing_ok=True)
                    result = subprocess.run(command, cwd='/tmp', env=dict(env, LVR_TEST_FAIL_VARIANT=failure),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 9 if failure else 0, result.stderr)
                    calls = [json.loads(line) for line in journal.read_text().splitlines()]
                    self.assertEqual([row['variant'] for row in calls], expected_variants)
                    for row in calls:
                        length = row['variant'].rsplit('-', 1)[1]
                        self.assertEqual(row['args'][1:], ['--gpus', '2', '--resume', '--eval-only',
                                                         '--output', f'runs/length test/latent-{length}'])

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
