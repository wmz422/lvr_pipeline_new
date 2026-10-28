"""Compare generated settings with the captured running experiment, after path relocation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from common import EXPERIMENT
from reproduce import FLOW_ARGS
from plain import recipe as plain_recipe


class RecipeTests(unittest.TestCase):
    def test_flow_matches_running_recipe_except_locations_and_bound_gt_hash(self):
        spec = importlib.util.spec_from_file_location('portable_flow_run', EXPERIMENT / 'script/run.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'LVR_FLOW_CONFIG': str(EXPERIMENT / 'configs/experiment.yaml')}):
            output = Path(directory).resolve()
            args = launch.parse_args([*FLOW_ARGS, '--output', str(output)])
            reference = yaml.safe_load((ROOT / 'configs/plain_reference.yaml').read_text().replace('${REPO}', str(ROOT)))
            configs, steps = launch.build_recipes(args, reference, {'train': 172224, 'val': 7571},
                                                  {'protocol_sha256': 'relocated-protocol'})
            def normalize(value):
                if isinstance(value, dict):
                    return {k: '${GT_PROTOCOL_SHA}' if k == 'flow_protocol_sha256' else
                               '${FLOW_LATENTS}' if k == 'flow_latent_root' else normalize(v)
                            for k, v in value.items()}
                if isinstance(value, list):
                    return [normalize(v) for v in value]
                if isinstance(value, str):
                    return value.replace(str(output), '${OUTPUT}').replace(str(ROOT), '${REPO}')
                return value
            frozen = json.loads((ROOT / 'manifests/reference_recipes.json').read_text())
            self.assertEqual(normalize(configs), frozen)
            self.assertEqual(steps, {'teacher_joint': 1346, 'warmup': 200, 'joint': 1346})

    def test_plain_recipe_parses_and_preserves_baseline_schedule(self):
        from lightning.pytorch import Trainer
        from lightning.pytorch.cli import LightningArgumentParser
        from lvr.models.visual_bottleneck import VisualBottleneck
        from lvr.stage1 import StageOneCLI
        from continuation_runtime import ContinuationDataModule
        config = plain_recipe(Path('/tmp/repro-plain-parser'), 2)
        parser = LightningArgumentParser(exit_on_error=False)
        parser.add_lightning_class_args(Trainer, 'trainer')
        parser.add_lightning_class_args(VisualBottleneck, 'model')
        parser.add_lightning_class_args(ContinuationDataModule, 'data')
        StageOneCLI.add_arguments_to_parser(object.__new__(StageOneCLI), parser)
        parser.add_argument('--ckpt_path', type=str, default=None)
        parser.add_argument('--seed_everything', default=42)
        parsed = parser.parse_object(config)
        self.assertEqual(parsed.trainer.max_steps, 2692)
        self.assertEqual(parsed.model.lr_schedule_steps, 2784)
        self.assertEqual(parsed.trainer.devices * parsed.data.batch_size * parsed.trainer.accumulate_grad_batches, 128)
        self.assertEqual(parsed.model.num_latents, 0)
        self.assertFalse(parsed.data.include_encoder)


if __name__ == '__main__':
    unittest.main()
