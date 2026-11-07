"""Real tiny Qwen3/Flux gradients and train -> export -> student -> inference on CPU."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lvr.models.latent_prediction import LatentPrediction, prediction_prompt
from reconstruction_experiment.latent_cache import REPRESENTATION
from reconstruction_experiment.teacher import NoDescriptionFlowTeacher
from reconstruction_experiment.description_teacher import DescriptionFlowTeacher
from test_training import TinyFlowRuntime

ROOT = Path(__file__).resolve().parents[3]


class Qwen3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.flow = cls.root / 'flow'
        (cls.flow / 'transformer').mkdir(parents=True)
        (cls.flow / 'transformer/config.json').write_text('{"joint_attention_dim":7680}')
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=64,
                             rope_parameters={'rope_type':'default', 'mrope_section':[8,12,12]}),
            vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                               out_hidden_size=128, patch_size=16, temporal_patch_size=2,
                               spatial_merge_size=2, deepstack_visual_indexes=[0,1], num_position_embeddings=16),
            image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3)
        native = Qwen3VLForConditionalGeneration(config)
        native.generation_config.pad_token_id = 0
        native.generation_config.eos_token_id = 4
        native.save_pretrained(cls.root)
        spec = importlib.util.spec_from_file_location('qwen3_export',
            ROOT / 'doc/flow_reconstruction_parallel_20261015/scripts/export_teacher.py')
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(42)

    def teacher(self, weight, *, descriptions=False, alignment=False, num_latents=8):
        cls = DescriptionFlowTeacher if descriptions else NoDescriptionFlowTeacher
        model = cls(str(self.root), str(self.root), train_qa=True, num_latents=num_latents,
            description_weight=1.0 if descriptions else 0.0,
            qa_loss_weight=0.0 if alignment else 1.0,
            flow_model_path=str(self.flow), flow_protocol_sha256='same-gt', flow_weight=weight,
            lr_schedule_steps=2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model.__dict__['_flow_runtime'] = TinyFlowRuntime()
        return model

    @staticmethod
    def batch(num_latents=8):
        def inputs(answer=False):
            ids = torch.tensor([[2,1,3,10] + [0]*num_latents + ([11,12,4] if answer else [])]*2)
            slots = torch.zeros_like(ids, dtype=torch.bool)
            slots[:,4:4 + num_latents] = True
            result = dict(input_ids=ids, attention_mask=torch.ones_like(ids), slot_mask=slots,
                          mm_token_type_ids=(ids==1).long(), image_grid_thw=torch.tensor([[1,2,2]]*2),
                          pixel_values=torch.randn(8,1536))
            if answer:
                labels = torch.full_like(ids, -100)
                labels[:,4 + num_latents:] = ids[:,4 + num_latents:]
                result['labels'] = labels
            return result
        return {'ids':['a','b'], 'encoder':inputs(), 'qa':inputs(True),
                'flow':{'targets':[torch.randn(8,2,2) for _ in range(2)],
                        'references':[torch.randn(8,2,2) for _ in range(2)],
                        'representation':REPRESENTATION, 'gt_protocol_sha256':'same-gt'}}

    @classmethod
    def description_batch(cls, num_latents=8):
        batch = cls.batch(num_latents)
        ids = torch.tensor([[13,14] + [0]*num_latents + [15,16,4]]*2)
        slots = torch.zeros_like(ids, dtype=torch.bool)
        slots[:,2:2 + num_latents] = True
        labels = torch.full_like(ids, -100)
        labels[:,2 + num_latents:] = ids[:,2 + num_latents:]
        batch['description'] = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            slot_mask=slots, mm_token_type_ids=torch.zeros_like(ids), labels=labels)
        return batch

    @staticmethod
    def save(model, path, *, teacher=False):
        checkpoint = {'state_dict':model.state_dict(), 'hyper_parameters':dict(model.hparams),
                      'global_step':1, 'epoch':0}
        if teacher:
            model.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, path)
        return checkpoint

    def test_zero_weight_has_exact_qa_objective_and_zero_flow_gradient(self):
        full, ablation = self.teacher(.1), self.teacher(0)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, ablation.state_dict()[key], rtol=0, atol=0)
        inputs = self.batch()
        for model in (full, ablation):
            with patch.object(model, 'log_dict') as log, torch.autocast('cpu', dtype=torch.bfloat16):
                loss = model.training_step(inputs, 0)
            metrics = log.call_args.args[0]
            torch.testing.assert_close(loss, metrics['train/ce'] + model.hparams.flow_weight *
                                       metrics['train/flow_mse'], rtol=0, atol=0)
            loss.backward()
            mapper = model.flow_mapper.net[-1].weight.grad
            self.assertIsNotNone(mapper)
            self.assertEqual(bool(mapper.count_nonzero()), model.hparams.flow_weight > 0)
            for parameter in (model.queries, model.mu_head[1].weight,
                              model.encoder.model.language_model.layers[0].self_attn.q_proj.weight,
                              model.qa.model.language_model.layers[0].self_attn.q_proj.weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad
                                for p in model.__dict__['_flow_runtime'].transformer.parameters()))
        torch.testing.assert_close(loss, metrics['train/ce'], rtol=0, atol=0)

    def test_both_variants_train_all_stages_and_generate_after_checkpoint_reload(self):
        for weight in (.1, 0):
            with self.subTest(flow_weight=weight):
                directory = self.root / f'run-{weight}'
                directory.mkdir()
                teacher = self.teacher(weight)
                inputs = self.batch()
                optimizer = teacher.configure_optimizers()['optimizer']
                with patch.object(teacher, 'log_dict'), torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = teacher.training_step(inputs, 0)
                loss.backward()
                optimizer.step()
                teacher.eval()
                with patch.object(teacher, 'log_dict') as log, torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    teacher.validation_step(inputs, 0)
                self.assertTrue(all(torch.isfinite(value) for value in log.call_args.args[0].values()))
                teacher_path, source = directory / 'teacher.ckpt', directory / 'source.ckpt'
                checkpoint = self.save(teacher, teacher_path, teacher=True)
                self.assertEqual(checkpoint['flow_metadata']['flow_loss_weight'], weight)
                self.exporter.export_teacher(teacher_path, source)
                exported = torch.load(source, weights_only=False)
                self.assertFalse(any(key.startswith('flow_mapper.') for key in exported['state_dict']))
                previous = None
                for phase in ('warmup', 'joint'):
                    model = LatentPrediction(str(source), prediction_phase=phase,
                                             description_weight=0, lr_schedule_steps=2)
                    if previous:
                        model.initialize_from_checkpoint(previous)
                    optimizer = model.configure_optimizers()['optimizer']
                    before = model.prediction_queries.detach().clone()
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        values = model(inputs, step=500)
                    self.assertTrue(torch.isfinite(values['loss']))
                    values['loss'].backward()
                    optimizer.step()
                    self.assertFalse(torch.equal(before, model.prediction_queries))
                    model.eval()
                    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                        validation = model(inputs, validation=True)
                    self.assertTrue(torch.isfinite(validation['ce']))
                    previous = directory / f'{phase}.ckpt'
                    self.save(model, previous)
                restored = LatentPrediction.from_checkpoint(previous, 'cpu')
                qa = prediction_prompt(inputs['qa'], 0)
                with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
                    expected = model.predict_latents(qa)
                    actual = restored.predict_latents(qa)
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    tokens = restored.network.generate_answers(qa, actual, max_new_tokens=2, pad_token_id=0)
                self.assertEqual(tokens.shape[0], 2)
                self.assertGreater(tokens.shape[1], 0)

    def test_invalid_flow_weights_and_cross_variant_teacher_resume_are_rejected(self):
        for weight in (-.1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                self.teacher(weight)
        full, ablation = self.teacher(.1), self.teacher(0)
        checkpoint = self.save(full, self.root / 'resume.ckpt', teacher=True)
        full.on_load_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, 'Flow loss weight mismatch'):
            ablation.on_load_checkpoint(checkpoint)

    def test_four_stage_description_training_export_and_final_no_description(self):
        for num_latents, weight in ((8, .1), (8, 0), (4, .1), (16, .1)):
            with self.subTest(num_latents=num_latents, flow_weight=weight):
                directory = self.root / f'four-stage-{num_latents}-{weight}'
                directory.mkdir()
                inputs = self.description_batch(num_latents)
                first = self.teacher(weight, descriptions=True, alignment=True, num_latents=num_latents)
                optimizer = first.configure_optimizers()['optimizer']
                with patch.object(first, 'log_dict') as log, torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = first.training_step(inputs, 0)
                metrics = log.call_args.args[0]
                torch.testing.assert_close(loss, metrics['train/description_ce'] + weight * metrics['train/flow_mse'])
                self.assertEqual(metrics['train/ce'], 0)
                loss.backward()
                self.assertGreater(first.mu_head[1].weight.grad.abs().sum().item(), 0)
                optimizer.step()
                first_path = directory / 'alignment.ckpt'
                self.save(first, first_path, teacher=True)
                second = self.teacher(weight, descriptions=True, num_latents=num_latents)
                second.initialize_from_checkpoint(first_path)
                optimizer = second.configure_optimizers()['optimizer']
                with patch.object(second, 'log_dict') as log, torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = second.training_step(inputs, 0)
                metrics = log.call_args.args[0]
                torch.testing.assert_close(loss, metrics['train/ce'] + metrics['train/description_ce'] + weight * metrics['train/flow_mse'])
                loss.backward()
                optimizer.step()
                second.eval()
                with patch.object(second, 'log_dict') as log, torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    second.validation_step(inputs, 0)
                self.assertIn('val/description_ce', log.call_args.args[0])
                checkpoint, source = directory / 'teacher.ckpt', directory / 'source.ckpt'
                self.save(second, checkpoint, teacher=True)
                self.exporter.export_teacher(checkpoint, source)
                student_inputs = {key: value for key,value in inputs.items() if key not in ('description','flow')}
                previous = None
                for phase in ('warmup', 'joint'):
                    student = LatentPrediction(str(source), prediction_phase=phase, description_weight=0, lr_schedule_steps=2)
                    self.assertEqual(student.hparams.num_latents, num_latents)
                    self.assertEqual(tuple(student.prediction_queries.shape), (num_latents, 128))
                    if previous:
                        student.initialize_from_checkpoint(previous)
                    optimizer = student.configure_optimizers()['optimizer']
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        values = student(student_inputs, step=500)
                    self.assertEqual(tuple(values['predicted'].shape), (2, num_latents, 512))
                    self.assertNotIn('description_ce', values)
                    values['loss'].backward()
                    optimizer.step()
                    previous = directory / f'{phase}.ckpt'
                    self.save(student, previous)
                restored = LatentPrediction.from_checkpoint(previous, 'cpu')
                with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
                    prompt = prediction_prompt(student_inputs['qa'], 0)
                    latent = restored.predict_latents(prompt)
                    self.assertEqual(tuple(latent.shape), (2, num_latents, 512))
                    tokens = restored.network.generate_answers(prompt, latent, max_new_tokens=2, pad_token_id=0)
                self.assertEqual(tokens.shape[0], 2)

    def test_answer_only_ce_backpropagates_without_teacher_forward_or_alignment(self):
        directory = self.root / 'answer-only'
        directory.mkdir()
        teacher = self.teacher(.1, descriptions=True)
        checkpoint, source = directory / 'teacher.ckpt', directory / 'source.ckpt'
        self.save(teacher, checkpoint, teacher=True)
        self.exporter.export_teacher(checkpoint, source)
        student = LatentPrediction(str(source), prediction_phase='joint', latent_weight=0,
                                   description_weight=0, lr_schedule_steps=2)
        batch = self.batch()
        inputs = {'ids': batch['ids'], 'qa': batch['qa']}
        optimizer = student.configure_optimizers()['optimizer']
        with patch.object(student.network, 'encode', side_effect=AssertionError('Teacher must not execute')), \
             patch('lvr.models.latent_prediction.latent_alignment_loss', side_effect=AssertionError('No alignment')), \
             torch.autocast('cpu', dtype=torch.bfloat16):
            values = student(inputs, step=0)
            torch.testing.assert_close(values['loss'], values['ce'], rtol=0, atol=0)
            self.assertEqual(values['predicted_probability'], 1)
            values['predicted'].retain_grad()
            values['loss'].backward()
            for gradient in (values['predicted'].grad, student.prediction_queries.grad,
                             student.prediction_head[1].weight.grad):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in student.encoder.parameters()))
            optimizer.step()
            student.eval()
            with torch.no_grad():
                validation = student(inputs, validation=True)
            self.assertNotIn('teacher_ce', validation)
            self.assertNotIn('latent_loss', validation)
        final = directory / 'joint.ckpt'
        self.save(student, final)
        restored = LatentPrediction.from_checkpoint(final, 'cpu')
        self.assertEqual(restored.hparams.latent_weight, 0)
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            prompt = prediction_prompt(inputs['qa'], 0)
            latent = restored.predict_latents(prompt)
            tokens = restored.network.generate_answers(prompt, latent, max_new_tokens=2, pad_token_id=0)
        self.assertEqual(tokens.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
