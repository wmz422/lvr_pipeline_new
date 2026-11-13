"""Parse the current experiment recipes and exercise their public shell entries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    ('train_qwen3_latent2.sh', 'qwen3-latent-2', 'Qwen3-VL-4B-Instruct', 2),
    ('train_qwen3_reconstruction_only.sh', 'qwen3-reconstruction-only', 'Qwen3-VL-4B-Instruct', 8),
    ('train_qwen25_flow_4stage.sh', 'qwen25-four-stage', 'Qwen2.5-VL-7B-Instruct', 8),
)


class CurrentExperimentTests(unittest.TestCase):
    def test_entries_preview_independent_runs_and_forward_hardware_and_resume(self):
        outputs = set()
        for script, variant, model, length in ENTRIES:
            with self.subTest(script=script):
                env = dict(os.environ, LVR_TRAIN_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
                report = json.loads(subprocess.check_output(
                    ['bash', str(ROOT / script), '--dry-run', '--gpus', '2', '--eval-gpus', '1',
                     '--warmup-zero-stage', '2', '--resume'], cwd='/tmp', env=env, text=True))
                self.assertEqual(report['variant'], variant)
                self.assertTrue(report['base_model'].endswith('/' + model))
                self.assertEqual(report['num_latents'], length)
                self.assertEqual(report['optimizer_steps'], {
                    'teacher_alignment': 1346, 'teacher_joint': 1346, 'warmup': 200, 'joint': 1346})
                self.assertFalse(report['gpu_started'])
                self.assertFalse(report['downloads_started'])
                for phase, hardware in report['hardware']['phases'].items():
                    self.assertEqual(hardware['gpus'], 2)
                    self.assertEqual(hardware['global_batch'], 128)
                    self.assertEqual(hardware['zero_stage'], 2 if phase == 'warmup' else 3)
                for phase in ('teacher_alignment', 'teacher_joint'):
                    weights = report['loss_weights'][phase]
                    self.assertEqual(weights['flow_weight'], .1)
                    reconstruction_only = variant == 'qwen3-reconstruction-only'
                    self.assertEqual(weights['description_weight'], 0 if reconstruction_only else 1)
                    self.assertEqual(weights['qa_loss_weight'], 0 if reconstruction_only or phase == 'teacher_alignment' else 1)
                evaluation = report['evaluation']
                self.assertEqual(evaluation[evaluation.index('--gpus') + 1], '1')
                self.assertEqual(evaluation[evaluation.index('--mode') + 1], 'predicted')
                self.assertEqual(evaluation[evaluation.index('--checkpoint') + 1],
                                 report['output'] + '/joint/checkpoints/last_full.ckpt')
                outputs.add(report['output'])
        self.assertEqual(len(outputs), 3)

    def test_all_new_recipes_parse_for_formal_preflight_and_question_variants(self):
        code = '''
import importlib.util, json, sys
sys.path.insert(0, 'tools')
from common import EXPERIMENT, DEFAULT_OUTPUT, flow_defaults, reference_recipe
from reproduce import FLOW_ARGS
spec = importlib.util.spec_from_file_location('current_cpu_check', EXPERIMENT / 'script/check_cpu.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
args = check.launch.parse_args(FLOW_ARGS)
args.output, args.defaults = DEFAULT_OUTPUT, flow_defaults()
counts, settings = {'train':172224, 'val':7571}, {'protocol_sha256':'same-gt'}
checked = check.check_recipes(args, reference_recipe(), counts, settings)
configs, _ = check.launch.build_recipes(args, reference_recipe(), counts, settings)
commands = {phase:check.launch.training_command(args, phase, 'recipe.yaml')
            for phase in ('teacher_alignment', 'teacher_joint')}
print(json.dumps({'checked':len(checked), 'commands':commands}))
'''
        for _, variant, _, _ in ENTRIES:
            with self.subTest(variant=variant):
                env = dict(os.environ, LVR_FLOW_VARIANT=variant, CUDA_VISIBLE_DEVICES='')
                env.pop('LVR_FLOW_CONFIG', None)
                output = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, env=env, text=True)
                report = json.loads(output.splitlines()[-1])
                self.assertEqual(report['checked'], 16)
                module = ('reconstruction_experiment.stage1' if variant == 'qwen3-reconstruction-only'
                          else 'reconstruction_experiment.description_teacher')
                for command in report['commands'].values():
                    self.assertEqual(command[command.index('--module') + 1], module)


if __name__ == '__main__':
    unittest.main()
