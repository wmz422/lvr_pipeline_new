"""Reject a wrong or unfinished shared stage-2 source before answer-only training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Qwen3HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('handoff_launch', ROOT /
            'paper_experiments/2026-10-27-qwen25-flow-no-sat-4gpu/script/run.py')
        cls.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launch)
        cls.workflow = cls.launch.import_script('handoff_workflow', ROOT /
            'lvr_pipeline_new/scripts/reasoning_data/train_and_evaluate.py')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        source = Path(self.directory.name)
        (source/'teacher_joint/checkpoints').mkdir(parents=True)
        (source/'teacher_joint/checkpoints/student_source.ckpt').write_bytes(b'test checkpoint identity')
        self.args = SimpleNamespace(source_run=source)
        self.counts = {'train':172224, 'val':7571}
        self.inputs = {'counts':self.counts, 'splits':{'train':{'sha256':'same-data'}},
                       'model_repository':'Qwen/Qwen3-VL-4B-Instruct', 'model_revision':'same-revision'}
        data = {key:str(ROOT/key) for key in ('train_path','val_path','train_boxes_path','val_boxes_path',
                'additional_train_path','additional_val_path','processor_name_or_path')}
        self.plain = {'data':data}
        self.config = {'model':{'flow_weight':.1,'description_weight':1.,'qa_loss_weight':1.},'data':data}
        self.plan = {'phases':['teacher_alignment','teacher_joint','warmup','joint'],
                     'recipes':{'teacher_joint':self.config},'inputs':self.inputs,
                     'optimizer_steps':{'teacher_joint':1346}}
        self.write_plan()
        self.done = {'status':'complete','optimizer_steps':1346,
                     'recipe_sha256':self.workflow.digest(self.config)}
        (source/'teacher_joint/complete.json').write_text(json.dumps(self.done))

    def write_plan(self):
        (self.args.source_run/'workflow_plan.json').write_text(json.dumps(self.plan))

    def bind(self, inputs=None):
        return self.launch.bind_source_run(self.args,self.plain,self.counts,inputs or self.inputs)

    def test_complete_flow_stage_two_is_sufficient_even_without_warmup_or_joint(self):
        bound = self.bind()
        self.assertEqual(bound['source_stage_2']['optimizer_steps'],1346)
        self.assertTrue(bound['checkpoint'].endswith('/teacher_joint/checkpoints/student_source.ckpt'))
        self.assertFalse((self.args.source_run/'warmup').exists())
        self.assertFalse((self.args.source_run/'joint').exists())

    def test_three_stage_or_zero_flow_sources_are_rejected(self):
        self.plan['phases'] = ['teacher_joint','warmup','joint']
        self.write_plan()
        with self.assertRaisesRegex(ValueError,'four-stage'):
            self.bind()
        self.plan['phases'].insert(0,'teacher_alignment')
        self.plan['recipes']['teacher_joint']['model']['flow_weight'] = 0
        self.write_plan()
        with self.assertRaisesRegex(ValueError,'Flow 0.1'):
            self.bind()

    def test_same_counts_with_different_data_or_backbone_are_rejected(self):
        for key in ('splits','model_revision'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.inputs)
                changed[key] = 'different'
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    self.bind(changed)

    def test_unfinished_stage_two_is_rejected(self):
        self.done['optimizer_steps'] = 100
        (self.args.source_run/'teacher_joint/complete.json').write_text(json.dumps(self.done))
        with self.assertRaisesRegex(ValueError,'not complete'):
            self.bind()

    def test_predicted_only_loader_requires_explicit_mode_and_forbids_auxiliary_branches(self):
        from reconstruction_experiment.data import StudentDataModule
        kwargs = dict(processor_name_or_path='unused', image_root='unused', num_latents=8,
                      include_encoder=False)
        with self.assertRaises(ValueError):
            StudentDataModule(**kwargs)
        module = StudentDataModule(**kwargs, predicted_latents_only=True)
        self.assertFalse(module.hparams.include_encoder)
        self.assertEqual(module.hparams.num_latents,8)
        for extra in ({'include_descriptions':True},{'include_reconstruction':True},{'qa_crops':True}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError,'Predicted-only data'):
                StudentDataModule(**kwargs, predicted_latents_only=True, **extra)


if __name__ == '__main__':
    unittest.main()
