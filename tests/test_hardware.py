"""Validate fixed-batch recipes, ZeRO compatibility and side-effect-free launch previews."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from common import EXPERIMENT
from reproduce import FLOW_ARGS
import start_flow


class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch = start_flow.launcher()
        cls.plain = yaml.safe_load((ROOT / 'configs/plain_reference.yaml').read_text().replace('${REPO}', str(ROOT)))

    def build(self, extra):
        args = self.launch.parse_args([*FLOW_ARGS, *extra])
        configs, steps = self.launch.build_recipes(args, self.plain, {'train': 172224, 'val': 7571},
                                                  {'protocol_sha256': 'test-hardware'})
        return args, configs, steps

    def test_gpu_counts_preserve_full_and_tail_updates_and_gradient_scale(self):
        for world in (1, 2, 4, 8, 16, 32, 64):
            with self.subTest(world=world):
                args, configs, steps = self.build(['--gpus', str(world), '--teacher-batch-size', '2',
                                                  '--warmup-batch-size', '2', '--joint-batch-size', '2'])
                self.assertEqual(args.eval_gpus, min(world, 8))
                self.assertEqual(steps, {'teacher_joint': 1346, 'warmup': 200, 'joint': 1346})
                for phase, config in configs.items():
                    batch = config['data']['batch_size']
                    gas = config['trainer']['accumulate_grad_batches']
                    self.assertEqual(world * batch * gas, 128)
                    tail_per_rank = (172224 // world) % (batch * gas)
                    self.assertEqual(tail_per_rank * world, 64)
                    self.assertEqual(tail_per_rank % gas, 0)
                    self.assertGreaterEqual(tail_per_rank, gas)
                    self.assertEqual(config['trainer']['plugins'][0]['init_args']['reference_accumulation'],
                                     16 if phase == 'teacher_joint' else 8)
                    self.assertEqual(config['data']['original_pixels'], 2000000)
                    self.assertEqual(config['model']['description_weight'], 0)

    def test_invalid_global_batch_and_nonempty_tail_are_rejected(self):
        for extra in (['--gpus', '3'], ['--gpus', '0'], ['--teacher-batch-size', '3'],
                      ['--warmup-batch-size', '0'], ['--flow-generator-batch-size', '0'],
                      ['--zero-stage', '2', '--parameter-offload', 'cpu'],
                      ['--skip-preflight', '--preflight-only']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.build(extra)
        with self.assertRaisesRegex(ValueError, 'equal nonempty'):
            self.build(['--teacher-batch-size', '1'])

    def test_installed_deepspeed_accepts_stage2_stage3_and_offload_configs(self):
        from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
        for stage in (2, 3):
            for device in ('cpu', 'cuda'):
                with self.subTest(stage=stage, optimizer_device=device):
                    args, configs, _ = self.build(['--zero-stage', str(stage),
                        *[value for phase in ('teacher', 'warmup', 'joint')
                          for value in (f'--{phase}-optimizer-device', device)]])
                    for config in configs.values():
                        value = config['trainer']['strategy']['init_args']['config']
                        ds = value if isinstance(value, dict) else json.loads(Path(value).read_text())
                        zero = ds['zero_optimization']
                        parsed = DeepSpeedZeroConfig(**zero)
                        self.assertEqual(parsed.stage, stage)
                        self.assertEqual(zero.get('offload_optimizer', {}).get('device', 'cuda'), device)
                        if stage == 2:
                            self.assertFalse(any(k.startswith('stage3_') for k in zero))
                            self.assertNotIn('offload_param', zero)
        _, configs, _ = self.build(['--parameter-offload', 'cpu'])
        for config in configs.values():
            zero = config['trainer']['strategy']['init_args']['config']['zero_optimization']
            parsed = DeepSpeedZeroConfig(**zero)
            self.assertEqual(parsed.offload_param.device, 'cpu')

    def test_zero2_audit_validates_original_parameters_instead_of_master_shards(self):
        import torch
        from lvr.training.audit import RunAudit
        model = torch.nn.Module()
        model.qa, model.encoder = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        model.hparams = SimpleNamespace(num_latents=8, description_weight=0, encoder_uses_question=False,
            encoder_type='qwen', encoder_includes_original=True, qformer_use_geometry=False, include_descriptions=False)
        class Dataset(list):
            @property
            def items(self):
                return self
        loader = SimpleNamespace(dataset=Dataset([{'id': 'one'}, {'id': 'two'}]), batch_size=2, sampler=[0, 1])
        original = list(model.parameters())
        master = torch.nn.Parameter(torch.zeros(sum(p.numel() for p in original)))
        optimizer = SimpleNamespace(bit16_groups=[original], param_groups=[{'params': [master], 'lr': 1e-5}], state={})
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(default_root_dir=directory, train_dataloader=loader, global_rank=0,
                datamodule=SimpleNamespace(hparams=model.hparams), strategy=SimpleNamespace(zero_stage_3=False,
                    config={'zero_optimization': {'stage': 2}}), optimizers=[optimizer],
                estimated_stepping_batches=1, accumulate_grad_batches=1, world_size=1, global_step=0, current_epoch=0)
            audit = RunAudit()
            audit.on_train_start(trainer, model)
            self.assertTrue(audit.record['optimizer_groups'][0]['partitioned'])
            self.assertEqual(audit.record['optimizer_groups'][0]['parameters'], master.numel())
            self.assertEqual(audit.record['qa_initial_parameter_hash_scope'], 'parameter_samples')
            optimizer.bit16_groups = [original[:-1]]
            with self.assertRaises(AssertionError):
                RunAudit().on_train_start(trainer, model)

    def test_phase_overrides_keep_teacher_loss_and_full_benchmark_protocol(self):
        args, configs, _ = self.build(['--gpus', '2', '--teacher-batch-size', '4',
                                      '--warmup-zero-stage', '2', '--eval-gpus', '1',
                                      '--flow-generator-batch-size', '4'])
        summary = self.launch.hardware_summary(args, configs)
        self.assertEqual([p['zero_stage'] for p in summary['phases'].values()], [3, 2, 3])
        self.assertEqual(summary['eval_gpus'], 1)
        self.assertEqual(configs['teacher_joint']['model']['flow_generator_batch_size'], 4)
        self.assertEqual(configs['teacher_joint']['model']['flow_weight'], .1)
        self.assertEqual(configs['teacher_joint']['model']['qa_loss_weight'], 1)
        self.assertFalse(configs['teacher_joint']['model']['encoder_uses_question'])
        self.assertFalse(summary['automatic_hardware_tuning'])

    def test_cpu_recipe_parser_accepts_alternative_hardware_in_all_phases(self):
        spec = importlib.util.spec_from_file_location('hardware_cpu_check', EXPERIMENT / 'script/check_cpu.py')
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        args, _, _ = self.build(['--gpus', '2', '--teacher-batch-size', '4', '--warmup-batch-size', '8',
                                 '--joint-batch-size', '2', '--zero-stage', '2'])
        rows = check.check_recipes(args, self.plain, {'train': 172224, 'val': 7571},
                                   {'protocol_sha256': 'test-parser'})
        self.assertEqual(len(rows), 12)

    def test_preview_never_bootstraps_downloads_or_launches_training(self):
        output = io.StringIO()
        with patch.dict(os.environ), patch.object(start_flow, 'bootstrap') as prepare, \
                patch.object(start_flow, 'run') as execute, contextlib.redirect_stdout(output):
            start_flow.main(['--gpus', '8', '--teacher-batch-size', '4', '--warmup-batch-size', '4',
                             '--joint-batch-size', '2', '--zero-stage', '2', '--dry-run'])
            prepare.assert_not_called()
            execute.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertFalse(report['gpu_started'])
        self.assertFalse(report['downloads_started'])
        self.assertFalse(report['gpu_preflight'])
        self.assertEqual(report['hardware']['global_batch'], 128)
        self.assertEqual(report['hardware']['eval_gpus'], 8)

    def test_single_command_bootstraps_then_forwards_manual_options(self):
        options = ['--gpus', '2', '--teacher-batch-size', '4', '--warmup-batch-size', '8',
                   '--joint-batch-size', '2', '--zero-stage', '2', '--skip-preflight']
        events = []
        with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(start_flow, 'bootstrap', side_effect=lambda source: events.append('inputs')) as prepare, \
                patch.object(start_flow, 'run', side_effect=lambda command: events.append(command)):
            start_flow.main(options)
            prepare.assert_called_once_with(None)
        self.assertEqual(events[0], 'inputs')
        self.assertEqual(events[-1][-len(options):], options)
        self.assertEqual(events[-1][2], 'train')


if __name__ == '__main__':
    unittest.main()
