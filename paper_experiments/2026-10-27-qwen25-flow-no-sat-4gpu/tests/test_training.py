"""Real tiny Qwen2.5/Flux gradient checks, including description-free validation."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from diffusers import Flux2Transformer2DModel
from flow_reconstruction.flow import conditional_flow_loss
from lvr.models.visual_bottleneck import VisualBottleneck
from lvr.models.latent_prediction import LatentPrediction
from reconstruction_experiment.teacher import NoDescriptionFlowTeacher
from reconstruction_experiment.latent_cache import REPRESENTATION

WORKSPACE = Path(__file__).resolve().parents[3]
FLOW = WORKSPACE / "doc/flow_reconstruction_parallel_20261015"


def batch():
    def inputs(answer=False):
        ids = torch.tensor([[2, 1, 3, 10] + [0] * 8 + ([11, 12, 4] if answer else [])] * 2)
        slots = torch.zeros_like(ids, dtype=torch.bool)
        slots[:, 4:12] = True
        out = dict(input_ids=ids, attention_mask=torch.ones_like(ids), slot_mask=slots,
                   mm_token_type_ids=(ids == 1).long(), image_grid_thw=torch.tensor([[1, 2, 2]] * 2),
                   pixel_values=torch.randn(8, 1176))
        if answer:
            labels = torch.full_like(ids, -100)
            labels[:, 12:] = ids[:, 12:]
            out["labels"] = labels
        return out
    return {"encoder": inputs(), "qa": inputs(True), "ids": ["a", "b"]}


class TinyFlowRuntime:
    def __init__(self):
        self.transformer = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=7680, timestep_guidance_channels=16, mlp_ratio=1,
            axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).requires_grad_(False).eval()
        self.transformer.enable_gradient_checkpointing()

    def loss(self, targets, references, condition, ids, **kwargs):
        clean = torch.stack(targets)
        noise = torch.ones_like(clean)
        return conditional_flow_loss(self.transformer, clean, [torch.stack(references)], condition,
            torch.tensor([.4] * len(ids)), torch.tensor([.4] * len(ids)), noise, checkpointing=True)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.flow_fixture = cls.root / "flow"
        (cls.flow_fixture / "transformer").mkdir(parents=True)
        (cls.flow_fixture / "transformer/config.json").write_text('{"joint_attention_dim": 7680}')
        config = Qwen2_5_VLConfig(
            text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2,
                             rope_parameters={"rope_type": "default", "mrope_section": [8, 12, 12]}),
            vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                               out_hidden_size=128, patch_size=14, temporal_patch_size=2,
                               spatial_merge_size=2, window_size=56, fullatt_block_indexes=[1]),
            image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3)
        native = Qwen2_5_VLForConditionalGeneration(config)
        native.generation_config.pad_token_id = 0
        native.generation_config.eos_token_id = 4
        native.save_pretrained(cls.root)
        cls.sources = {}
        for question in (False, True):
            teacher = VisualBottleneck(str(cls.root), str(cls.root), num_latents=8, latent_dim=512,
                gaussian=False, train_qa=True, description_weight=0, encoder_uses_question=question)
            path = cls.root / f"teacher_question_{question}.ckpt"
            torch.save({"state_dict": teacher.state_dict(), "hyper_parameters": dict(teacher.hparams),
                        "global_step": 1392, "epoch": 0}, path)
            cls.sources[question] = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(42)

    def teacher(self):
        return NoDescriptionFlowTeacher(str(self.root), str(self.root), num_latents=8, latent_dim=512,
            gaussian=False, train_qa=True, description_weight=0, kl_weight=0,
            flow_model_path=str(self.flow_fixture),
            flow_protocol_sha256="test-protocol", flow_weight=.1)

    def test_real_reconstruction_gradient_reaches_teacher_not_frozen_generator(self):
        model, inputs = self.teacher(), batch()
        runtime = TinyFlowRuntime()
        model.__dict__["_flow_runtime"] = runtime
        inputs["flow"] = {"targets": [torch.randn(8, 2, 2) for _ in range(2)],
                          "references": [torch.randn(8, 2, 2) for _ in range(2)],
                          "representation": REPRESENTATION, "gt_protocol_sha256": "test-protocol"}
        with patch.object(model, "answer_loss", wraps=model.answer_loss) as answer:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(inputs)
            self.assertEqual(answer.call_count, 1)
        out["flow_mse"].backward(retain_graph=True)
        for parameter in (model.queries, model.mu_head[1].weight, model.flow_mapper.net[-1].weight,
                          model.encoder.model.language_model.layers[0].self_attn.q_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.qa.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in runtime.transformer.parameters()))
        model.zero_grad(set_to_none=True)
        with patch.object(model, "log_dict") as log, patch.object(model, "answer_loss", wraps=model.answer_loss) as answer:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model.training_step(inputs, 0)
            self.assertEqual(answer.call_count, 1)
            self.assertFalse(any("description" in k for k in log.call_args.args[0]))
            loss.backward()
        self.assertGreater(model.qa.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.qa.model.visual.parameters()))
        with patch.object(model, "log_dict"), patch.object(model, "answer_loss", wraps=model.answer_loss) as answer:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                model.validation_step(inputs, 0)
            self.assertEqual(answer.call_count, 1)

    def test_both_student_phases_and_question_variants_skip_description(self):
        for question in (False, True):
            for phase in ("warmup", "joint"):
                with self.subTest(question=question, phase=phase):
                    model = LatentPrediction(self.sources[question], prediction_phase=phase,
                        description_weight=0, allow_question_conditioned_teacher=question)
                    inputs = batch()
                    with patch.object(model.network, "answer_loss", wraps=model.network.answer_loss) as answer:
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            out = model(inputs, step=500)
                        self.assertEqual(answer.call_count, 0 if phase == "warmup" else 1)
                    self.assertNotIn("description_ce", out)
                    out["loss"].backward()
                    for parameter in (model.prediction_queries, model.prediction_head[1].weight):
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))
                    if phase == "warmup":
                        self.assertTrue(all(p.grad is None for p in model.qa.parameters()))
                    else:
                        self.assertGreater(model.qa.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
                    with patch.object(model.network, "answer_loss", wraps=model.network.answer_loss) as answer:
                        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                            validation = model(inputs, validation=True)
                        self.assertEqual(answer.call_count, 2)  # predicted QA and teacher QA; no description
                    self.assertNotIn("description_ce", validation)

    def test_question_conditioning_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "teacher checkpoint"):
            LatentPrediction(self.sources[True], description_weight=0)
        model = LatentPrediction(self.sources[True], prediction_phase="joint", description_weight=0,
                                 allow_question_conditioned_teacher=True)
        path = self.root / "question_student.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams), "global_step": 1392}, path)
        restored = LatentPrediction.from_checkpoint(path, "cpu")
        self.assertTrue(restored.hparams.encoder_uses_question)
        self.assertTrue(restored.hparams.allow_question_conditioned_teacher)

    def test_export_initialization_roundtrip_and_answer_exclusion(self):
        model = self.teacher()
        checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                      "global_step": 1392, "epoch": 0}
        model.on_save_checkpoint(checkpoint)
        path, source = self.root / "flow.ckpt", self.root / "student_source.ckpt"
        torch.save(checkpoint, path)
        spec = importlib.util.spec_from_file_location("flow_export", FLOW / "scripts/export_teacher.py")
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        export.export_teacher(path, source)
        exported = torch.load(source, mmap=True, weights_only=False)
        self.assertFalse(any(k.startswith("flow_mapper.") for k in exported["state_dict"]))
        warmup = LatentPrediction(str(source), description_weight=0)
        warm_path = self.root / "warmup.ckpt"
        torch.save({"state_dict": warmup.state_dict(), "hyper_parameters": dict(warmup.hparams), "global_step": 200}, warm_path)
        student = LatentPrediction(str(source), prediction_phase="joint", description_weight=0)
        student.initialize_from_checkpoint(warm_path)
        inputs = batch()["qa"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            before = student.predict_latents(inputs)
            inputs["input_ids"][:, 12:] = 40
            after = student.predict_latents(inputs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        final = self.root / "joint.ckpt"
        torch.save({"state_dict": student.state_dict(), "hyper_parameters": dict(student.hparams), "global_step": 1392}, final)
        restored = LatentPrediction.from_checkpoint(final, "cpu")
        self.assertEqual(restored.checkpoint_step, 1392)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            torch.testing.assert_close(after, restored.predict_latents(inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
