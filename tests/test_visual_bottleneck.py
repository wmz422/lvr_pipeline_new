"""Exercise real Qwen gradient flow and the multi-crop input contracts."""

import tempfile
import os
import unittest
from pathlib import Path

from PIL import Image
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
import yaml

from lvr.data.crop_qa import CropQACollator, crop_regions, grid_boxes, resize_pixels
from lvr.models.visual_bottleneck import VisualBottleneck


RAW_QWEN = os.environ.get("QWEN_MODEL_PATH", "models/Qwen3-VL-4B-Instruct")


class VisualBottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=64,
                             rope_parameters={"rope_type": "default", "mrope_section": [8, 12, 12]}),
            vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                               out_hidden_size=128, patch_size=16, temporal_patch_size=2,
                               spatial_merge_size=2, deepstack_visual_indexes=[0, 1], num_position_embeddings=16),
            image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3,
        )
        model = Qwen3VLForConditionalGeneration(config)
        model.save_pretrained(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        # VLMEvalKit imports can disable gradients globally; training tests are independent.
        grad_context = torch.enable_grad()
        grad_context.__enter__()
        self.addCleanup(grad_context.__exit__, None, None, None)

    @staticmethod
    def batch(num_latents=2):
        def inputs(answer=False):
            ids = torch.tensor([[2, 1, 3, 10] + [0] * num_latents + ([11, 12, 4] if answer else [])] * 2)
            slots = torch.zeros_like(ids, dtype=torch.bool)
            slots[:, 4:4 + num_latents] = True
            batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), slot_mask=slots,
                         mm_token_type_ids=(ids == 1).long(), image_grid_thw=torch.tensor([[1, 2, 2]] * 2),
                         pixel_values=torch.randn(8, 1536))
            if answer:
                labels = torch.full_like(ids, -100)
                labels[:, 4 + num_latents:] = ids[:, 4 + num_latents:]
                batch["labels"] = labels
            return batch
        return {"encoder": inputs(), "qa": inputs(True), "ids": ["a", "b"]}

    def test_answer_gradient_reaches_encoder_but_not_frozen_backbones(self):
        for gaussian in (False, True):
            with self.subTest(gaussian=gaussian):
                model = VisualBottleneck(self.directory.name, self.directory.name,
                                         num_latents=2, latent_dim=8, gaussian=gaussian)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    outputs = model(self.batch())
                # CE alone must train A; KL must not hide a detached QA path.
                outputs["ce"].backward()
                self.assertTrue(torch.isfinite(outputs["ce"]))
                self.assertGreater(model.queries.grad.abs().sum().item(), 0)
                self.assertGreater(model.mu_head[1].weight.grad.abs().sum().item(), 0)
                parameters = list(model.encoder.model.language_model.layers[0].parameters())
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))
                self.assertTrue(all(p.grad is None for p in model.qa.parameters()))
                self.assertTrue(all(p.grad is None for p in model.encoder.model.visual.parameters()))
                self.assertEqual(tuple(outputs["mu"].shape), (2, 2, 8))
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                if gaussian:
                    self.assertGreater(model.logvar_head.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(outputs["kl"].item(), 0)
                else:
                    self.assertEqual(outputs["kl"].item(), 0)

    def test_description_only_gradients_reach_encoder_projector_and_qa_without_b_vision(self):
        from unittest.mock import patch

        model = VisualBottleneck(self.directory.name, self.directory.name, num_latents=2,
                                 latent_dim=8, gaussian=False, train_qa=True, description_weight=1.0)
        batch = self.batch()
        description = {key: value[:, 3:].clone() for key, value in batch["qa"].items()
                       if key not in ("pixel_values", "image_grid_thw")}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mu, _ = model.encode(batch["encoder"])
            with patch.object(model.qa.model, "get_image_features", side_effect=AssertionError("B read image")):
                loss = model.answer_loss(description, mu)
        loss.backward()
        for parameter in (model.queries, model.mu_head[1].weight, model.projector[0].weight,
                          model.encoder.model.language_model.layers[0].self_attn.q_proj.weight,
                          model.qa.model.language_model.layers[0].self_attn.q_proj.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.qa.model.visual.parameters()))

    def test_description_collation_has_only_fixed_instruction_latents_and_description(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        processor.tokenizer.padding_side = "right"
        root = Path(self.directory.name)
        Image.new("RGB", (128, 96)).save(root / "description.png")
        item = {"id": "description", "question_image": "description.png", "boxes": [[2, 3, 18, 25]],
                "question": "QUESTION_SENTINEL\nA. option_a\nB. option_b", "answer": "ANSWER_SENTINEL",
                "description": "A small blue object."}
        collator = CropQACollator(processor, root, num_latents=2, original_pixels=65536,
                                 crop_pixels=65536, include_descriptions=True)
        batch = collator([item])
        description = batch["description"]
        self.assertNotIn("pixel_values", description)
        self.assertNotIn("image_grid_thw", description)
        self.assertFalse(description["mm_token_type_ids"].any())
        self.assertEqual(description["slot_mask"].sum().item(), 2)
        text = processor.tokenizer.decode(description["input_ids"][0])
        for forbidden in ("QUESTION_SENTINEL", "ANSWER_SENTINEL", "option_a", "<|image_pad|>"):
            self.assertNotIn(forbidden, text)
        labels = description["labels"]
        self.assertEqual(processor.tokenizer.decode(labels[labels != -100]), item["description"] + "<|im_end|>")
        changed = collator([{**item, "question": "New question", "answer": "New answer"}])
        for key, value in description.items():
            self.assertTrue(torch.equal(value, changed["description"][key]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            collator([{**item, "description": ""}])

    def test_description_objective_shares_encoder_and_preserves_qa_loss(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        model = VisualBottleneck(self.directory.name, self.directory.name, num_latents=2,
                                 latent_dim=8, gaussian=False, train_qa=True, description_weight=0.5)
        batch = self.batch()
        batch["description"] = {key: value[:, 3:].clone() for key, value in batch["qa"].items()
                                if key not in ("pixel_values", "image_grid_thw")}
        model._trainer = SimpleNamespace(estimated_stepping_batches=10, global_step=0)
        with torch.autocast("cpu", dtype=torch.bfloat16), patch.object(model, "log_dict"), \
                patch.object(model, "encode", wraps=model.encode) as encode:
            actual = model.training_step(batch, 0)
            self.assertEqual(encode.call_count, 1)
            outputs = model(batch)
        torch.testing.assert_close(actual, outputs["ce"] + 0.5 * outputs["description_ce"])
        actual.backward()
        self.assertTrue(torch.isfinite(model.projector[0].weight.grad).all())
        batch["description"]["pixel_values"] = batch["qa"]["pixel_values"]
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "must not receive images"):
            model(batch)

    def test_aligned_initialization_preserves_all_weights_and_unfreezes_qa_in_fp32(self):
        from types import SimpleNamespace

        aligned = VisualBottleneck(self.directory.name, self.directory.name,
                                   num_latents=2, latent_dim=8, gaussian=False)
        aligned._trainer = SimpleNamespace(estimated_stepping_batches=10)
        old_optimizer = aligned.configure_optimizers()["optimizer"]
        with torch.no_grad():
            aligned.queries.add_(0.25)
            aligned.projector[0].weight.add_(0.125)
        path = Path(self.directory.name) / "aligned.ckpt"
        torch.save({"state_dict": aligned.state_dict(), "hyper_parameters": dict(aligned.hparams),
                    "global_step": 961, "epoch": 0, "optimizer_states": [old_optimizer.state_dict()]}, path)
        joint = VisualBottleneck(self.directory.name, self.directory.name,
                                 num_latents=2, latent_dim=8, gaussian=False, train_qa=True)
        report = joint.initialize_from_checkpoint(path)
        self.assertEqual(report["source_global_step"], 961)
        self.assertFalse(report["optimizer_restored"])
        self.assertFalse(report["scheduler_restored"])
        for name, value in joint.state_dict().items():
            self.assertTrue(torch.equal(value.float(), aligned.state_dict()[name].float()), name)
        self.assertTrue(all(p.dtype == torch.float32 for p in joint.parameters() if p.requires_grad))
        self.assertTrue(all(not p.requires_grad for p in joint.qa.model.visual.parameters()))
        joint._trainer = SimpleNamespace(estimated_stepping_batches=10)
        optimizer = joint.configure_optimizers()["optimizer"]
        self.assertFalse(optimizer.state)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in joint.parameters() if p.requires_grad})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            joint(self.batch())["ce"].backward()
        for parameter in (joint.queries, joint.projector[0].weight, joint.qa.lm_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_initialization_rejects_changed_interface_and_incomplete_weights(self):
        model = VisualBottleneck(self.directory.name, self.directory.name,
                                 num_latents=2, latent_dim=8, gaussian=False, train_qa=True)
        path = Path(self.directory.name) / "invalid_initialization.ckpt"
        checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                      "global_step": 961, "epoch": 0}
        checkpoint["hyper_parameters"]["encoder_uses_question"] = True
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "encoder_uses_question"):
            model.initialize_from_checkpoint(path)
        checkpoint["hyper_parameters"]["encoder_uses_question"] = False
        checkpoint["state_dict"].pop("queries")
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            model.initialize_from_checkpoint(path)

    def test_new_phase_initialization_cannot_resume_old_optimizer(self):
        from types import SimpleNamespace
        from lvr.stage1 import StageOneCLI

        cli = object.__new__(StageOneCLI)
        cli.config = SimpleNamespace(fit=SimpleNamespace(init_checkpoint="aligned.ckpt", ckpt_path="last.ckpt"))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            cli.before_fit()

    def test_all_crops_fixed_slots_question_blindness_and_answer_labels(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        processor.tokenizer.padding_side = "right"
        root = Path(self.directory.name)
        Image.new("RGB", (128, 96), (20, 80, 140)).save(root / "image.png")
        item = {"id": "sample", "question_image": "image.png", "boxes": [[2, 3, 18, 25], [40, 20, 80, 70]],
                "question": "What color is the object?", "answer": "Blue."}
        collator = CropQACollator(processor, root, num_latents=4, original_pixels=65536, crop_pixels=65536)
        first = collator([item])
        second = collator([{**item, "question": "Where is it?", "answer": "On the left."}])
        for key, value in first["encoder"].items():
            self.assertTrue(torch.equal(value, second["encoder"][key]), key)
        self.assertEqual(len(first["encoder"]["image_grid_thw"]), 3)
        self.assertEqual(first["encoder"]["slot_mask"].sum().item(), 4)
        qa = first["qa"]
        self.assertTrue((qa["labels"][qa["slot_mask"]] == -100).all())
        self.assertTrue((qa["attention_mask"][qa["slot_mask"]] == 1).all())
        supervised = qa["labels"][qa["labels"] != -100].tolist()
        self.assertEqual(processor.tokenizer.decode(supervised), "Blue.<|im_end|>")
        self.assertLess(first["encoder"]["input_ids"].shape[1], 8192)

    def test_question_conditioned_32_latents_preserve_options_exclude_answers_and_receive_gradients(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        processor.tokenizer.padding_side = "right"
        root = Path(self.directory.name)
        Image.new("RGB", (128, 96), (20, 80, 140)).save(root / "question.png")
        item = {"id": "question", "question_image": "question.png",
                "boxes": [[2, 3, 18, 25], [40, 20, 80, 70]],
                "question": "What color is it?\nA. Blue\nB. Red", "answer": "ANSWER_SENTINEL_123"}
        collator = CropQACollator(processor, root, num_latents=32, original_pixels=65536,
                                 crop_pixels=65536, encoder_uses_question=True)
        batch = collator([item])
        a = batch["encoder"]
        self.assertEqual(a["slot_mask"].sum().item(), 32)
        self.assertEqual(len(a["image_grid_thw"]), 3)
        first_slot = a["slot_mask"][0].nonzero()[0].item()
        prompt = processor.tokenizer.decode(a["input_ids"][0, :first_slot])
        self.assertIn(item["question"], prompt)
        self.assertNotIn(item["answer"], prompt)
        self.assertNotIn("labels", a)
        changed_answer = collator([{**item, "answer": "DIFFERENT_ANSWER_SENTINEL"}])["encoder"]
        for key, value in a.items():
            self.assertTrue(torch.equal(value, changed_answer[key]), key)
        changed_question = collator([{**item, "question": "Where is it?\nA. Left\nB. Right"}])["encoder"]
        self.assertFalse(torch.equal(a["input_ids"], changed_question["input_ids"]))
        self.assertTrue(torch.equal(a["pixel_values"], changed_question["pixel_values"]))

        model = VisualBottleneck(self.directory.name, self.directory.name, num_latents=32,
                                 latent_dim=512, gaussian=False, train_qa=True, encoder_uses_question=True)
        synthetic = self.batch(num_latents=32)
        changed = {key: value.clone() for key, value in synthetic["encoder"].items()}
        changed["input_ids"][:, 3] = 17
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model(synthetic)
            other_mu, _ = model.encode(changed)
        self.assertEqual(tuple(result["mu"].shape), (2, 32, 512))
        self.assertGreater((result["mu"] - other_mu).abs().max().item(), 0)
        result["ce"].backward()
        self.assertTrue((model.queries.grad.abs().sum(-1) > 0).all())
        self.assertGreater(model.qa.lm_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.model.visual.parameters()))

    def test_joint_and_plain_training_update_all_qa_language_parameters_only(self):
        from types import SimpleNamespace

        for count in (0, 2):
            with self.subTest(num_latents=count):
                model = VisualBottleneck(self.directory.name, self.directory.name,
                                         num_latents=count, latent_dim=8, gaussian=False, train_qa=True)
                batch = self.batch()
                if not count:
                    batch.pop("encoder")
                    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "slot_mask", "labels"):
                        batch["qa"][key] = batch["qa"][key][:, [0, 1, 2, 3, 6, 7, 8]]
                    self.assertIsNone(model.encoder)
                    self.assertIsNone(model.projector)
                model._trainer = SimpleNamespace(estimated_stepping_batches=10)
                optimizer = model.configure_optimizers()["optimizer"]
                optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimized), len(set(optimized)))
                self.assertEqual(set(optimized), {id(p) for p in model.parameters() if p.requires_grad})
                watched = [model.qa.model.language_model.layers[0].self_attn.q_proj.weight,
                           model.qa.model.language_model.embed_tokens.weight, model.qa.lm_head.weight]
                before = [parameter.detach().clone() for parameter in watched]
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model(batch)["ce"]
                loss.backward()
                self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in watched))
                self.assertTrue(all(p.grad is None for p in model.qa.model.visual.parameters()))
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                if count:
                    self.assertGreater(model.queries.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None for p in model.encoder.model.visual.parameters()))
                optimizer.step()
                self.assertTrue(all(not torch.equal(old, p) for old, p in zip(before, watched)))
                self.assertTrue(all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values()))

    def test_zero_latent_qa_loss_and_gradients_match_native_qwen(self):
        model = VisualBottleneck(self.directory.name, self.directory.name,
                                 num_latents=0, gaussian=False, train_qa=True).eval()
        batch = self.batch()["qa"]
        for key in ("input_ids", "attention_mask", "mm_token_type_ids", "slot_mask", "labels"):
            batch[key] = batch[key][:, [0, 1, 2, 3, 6, 7, 8]]
        native = {key: value for key, value in batch.items() if key != "slot_mask"}
        parameter = model.qa.model.language_model.layers[0].self_attn.q_proj.weight
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.answer_loss(batch)
            expected = model.qa(**native, use_cache=False).loss
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        grad_actual = torch.autograd.grad(actual, parameter)[0]
        grad_expected = torch.autograd.grad(expected, parameter)[0]
        torch.testing.assert_close(grad_actual, grad_expected, rtol=0.02, atol=0.0001)
        path = Path(self.directory.name) / "plain.ckpt"
        torch.save({"state_dict": model.state_dict(), "global_step": 3,
                    "hyper_parameters": dict(model.hparams)}, path)
        loaded = VisualBottleneck.from_checkpoint(path, "cpu")
        self.assertIsNone(loaded.encoder)
        self.assertEqual(loaded.checkpoint_step, 3)
        self.assertTrue(torch.equal(loaded.qa.lm_head.weight, model.qa.lm_head.weight))

    def test_crop_geometry_and_invalid_boxes(self):
        original = Image.new("RGB", (100, 80))
        crop = crop_regions(original, [[10.2, 20.2, 30.8, 50.7]])[0]
        self.assertEqual(crop.size, (21, 31))
        enlarged = resize_pixels(crop, 65536, enlarge=True)
        self.assertGreater(enlarged.width, crop.width)
        self.assertAlmostEqual(enlarged.width / enlarged.height, 21 / 31, delta=0.01)
        self.assertEqual(crop_regions(original, [[10, 20, 12, 20]])[0].size, (2, 1))
        with self.assertRaises(ValueError):
            crop_regions(original, [[200, 200, 210, 210]])

    def test_generation_matches_native_baseline_and_uncached_latent_decoding(self):
        from torch.nn.utils.rnn import pad_sequence
        from transformers import GenerationConfig

        model = VisualBottleneck(self.directory.name, self.directory.name,
                                 num_latents=2, latent_dim=8, gaussian=False).eval()
        # This untrained toy decoder can emit image placeholders. The fixture models text-only QA.
        def text_only_logits(module, inputs, logits):
            logits = logits.clone()
            logits[..., [1, 2, 3, 5]] = torch.finfo(logits.dtype).min
            return logits

        model.qa.lm_head.register_forward_hook(text_only_logits)
        batch = self.batch()["encoder"]
        for key in ("input_ids", "attention_mask", "mm_token_type_ids", "slot_mask"):
            batch[key] = torch.cat([torch.zeros_like(batch[key][:, :1]), batch[key]], dim=1)
        batch["input_ids"][1, 0] = 17
        batch["attention_mask"][1, 0] = 1
        keep = batch["attention_mask"].bool() & ~batch["slot_mask"]
        native = {key: pad_sequence([r[m] for r, m in zip(batch[key], keep)],
                                   batch_first=True, padding_side="left")
                  for key in ("input_ids", "attention_mask", "mm_token_type_ids")}
        native.update({key: batch[key] for key in ("pixel_values", "image_grid_thw")})
        config = GenerationConfig(do_sample=False, max_new_tokens=3, pad_token_id=0, eos_token_id=None)
        model.qa.generation_config.eos_token_id = None
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            baseline = model.generate_answers(batch, max_new_tokens=3, pad_token_id=0)
            reference = model.qa.generate(**native, generation_config=config)[:, native["input_ids"].shape[1]:]
            self.assertTrue(torch.equal(baseline, reference))
            latent = torch.randn(2, 2, 8)
            generated = model.generate_answers(batch, latent, max_new_tokens=3, pad_token_id=0)
            prefix = dict(batch)
            expected = []
            for _ in range(3):
                hidden = model._hidden(model.qa, prefix, model.projector(latent))
                token = model.qa.lm_head(hidden[:, -1]).argmax(-1, keepdim=True)
                expected.append(token)
                for key in ("input_ids", "attention_mask", "mm_token_type_ids", "slot_mask"):
                    value = token if key == "input_ids" else torch.full_like(token, int(key == "attention_mask"))
                    prefix[key] = torch.cat([prefix[key], value.to(prefix[key].dtype)], dim=1)
            self.assertTrue(torch.equal(generated, torch.cat(expected, dim=1)))
        with self.assertRaisesRegex(ValueError, "prompt-only"):
            model.generate_answers(self.batch()["qa"])

    def test_generation_collator_never_includes_reference_answers(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        root = Path(self.directory.name)
        Image.new("RGB", (128, 96)).save(root / "generation.png")
        item = {"id": "sample", "question_image": "generation.png", "boxes": [[2, 3, 18, 25]],
                "question": "What is visible?", "answer": "REFERENCE_SENTINEL_12345"}
        collator = CropQACollator(processor, root, original_pixels=65536, crop_pixels=65536,
                                 include_answers=False, include_encoder=False)
        first = collator([item])
        second = collator([{**item, "answer": "ANOTHER_REFERENCE_98765"}])
        self.assertNotIn("encoder", first)
        self.assertNotIn("labels", first["qa"])
        for key in first["qa"]:
            self.assertTrue(torch.equal(first["qa"][key], second["qa"][key]))
        text = processor.tokenizer.decode(first["qa"]["input_ids"][0])
        self.assertNotIn("REFERENCE_SENTINEL_12345", text)

    def test_multi_image_grid_inputs_keep_fixed_latents_and_question_blind_encoder(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        root = Path(self.directory.name)
        images = []
        for index, size in enumerate(((129, 97), (96, 128))):
            path = root / f"grid_{index}.png"
            Image.new("RGB", size, (index * 120, 60, 30)).save(path)
            boxes = grid_boxes(size)
            self.assertEqual(sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes), size[0] * size[1])
            images.append({"path": str(path), "boxes": boxes})
        collator = CropQACollator(processor, root, num_latents=8, original_pixels=65536,
                                 crop_pixels=16384, include_answers=False)
        items = [{"id": "multi", "images": images, "question": "Which image?", "answer": "A"},
                 {"id": "single", "images": images[:1], "question": "What color?", "answer": "B"}]
        batch = collator(items)
        changed = collator([{**item, "question": "Another question", "answer": "C"} for item in items])
        self.assertEqual(batch["crop_counts"].tolist(), [8, 4])
        self.assertEqual(len(batch["encoder"]["image_grid_thw"]), 15)
        self.assertEqual(len(batch["qa"]["image_grid_thw"]), 3)
        self.assertEqual(batch["encoder"]["slot_mask"].sum(1).tolist(), [8, 8])
        self.assertNotIn("labels", batch["qa"])
        for key in batch["encoder"]:
            self.assertTrue(torch.equal(batch["encoder"][key], changed["encoder"][key]), key)

    def test_native_rgb_crops_preserve_encoder_pixels_and_omit_latents_and_answers(self):
        processor = AutoProcessor.from_pretrained(RAW_QWEN)
        root = Path(self.directory.name)
        images = []
        for index, size in enumerate(((128, 96), (96, 128))):
            path = root / f"rgb_crop_{index}.png"
            Image.new("RGB", size, (index * 120, 60, 30)).save(path)
            images.append({"path": str(path), "boxes": [[2, 3, 18, 25], [40, 20, 80, 70]]})
        items = [{"id": "multi", "images": images, "question": "Which image?", "answer": "ANSWER_SENTINEL_123"},
                 {"id": "single", "images": images[:1], "question": "What color?", "answer": "ANSWER_SENTINEL_456"}]
        common = dict(original_pixels=65536, crop_pixels=16384, include_answers=False)
        latent = CropQACollator(processor, root, **common)(items)
        collator = CropQACollator(processor, root, num_latents=0, include_encoder=False, qa_crops=True, **common)
        rgb = collator(items)
        changed = collator([{**item, "answer": "DIFFERENT_ANSWER_789"} for item in items])
        self.assertNotIn("encoder", rgb)
        self.assertNotIn("labels", rgb["qa"])
        self.assertFalse(rgb["qa"]["slot_mask"].any())
        self.assertEqual(rgb["crop_counts"].tolist(), [4, 2])
        self.assertEqual(len(rgb["qa"]["image_grid_thw"]), 9)
        for key in ("pixel_values", "image_grid_thw"):
            self.assertTrue(torch.equal(rgb["qa"][key], latent["encoder"][key]), key)
        for key in rgb["qa"]:
            self.assertTrue(torch.equal(rgb["qa"][key], changed["qa"][key]), key)
        text = processor.tokenizer.decode(rgb["qa"]["input_ids"][0])
        self.assertIn("Original image 2:", text)
        self.assertIn("Enlarged crop 2 from original image 2:", text)
        self.assertIn("Which image?", text)
        self.assertNotIn("ANSWER_SENTINEL", text)
        self.assertEqual([len(labels) for labels in rgb["qa_image_labels"]], [6, 3])
        from lvr.eval.vlmeval import BottleneckModel
        with self.assertRaisesRegex(ValueError, "requires latent_mode"):
            BottleneckModel("unused", "unused", latent_mode="mu", qa_crops=True)

    def test_inference_checkpoint_load_ignores_lightning_metadata(self):
        model = VisualBottleneck(self.directory.name, self.directory.name,
                                 num_latents=32, latent_dim=8, gaussian=False, encoder_uses_question=True)
        path = Path(self.directory.name) / "inference.ckpt"
        torch.save({"state_dict": model.state_dict(), "global_step": 17,
                    "hyper_parameters": {**dict(model.hparams), "_instantiator": "lightning metadata"}}, path)
        loaded = VisualBottleneck.from_checkpoint(path, "cpu")
        self.assertEqual(loaded.checkpoint_step, 17)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertTrue(torch.equal(loaded.queries, model.queries))
        self.assertEqual(loaded.queries.dtype, torch.float32)
        self.assertTrue(loaded.hparams.encoder_uses_question)
        self.assertEqual(loaded.hparams.num_latents, 32)

    def test_final_checkpoint_includes_step_after_last_validation(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/latent_prediction/warmup.yaml").read_text())
        with tempfile.TemporaryDirectory() as output:
            callbacks = [pl.callbacks.ModelCheckpoint(**{**entry["init_args"], "dirpath": output})
                         for entry in config["trainer"]["callbacks"]
                         if entry["class_path"].endswith("ModelCheckpoint")]
            model = VisualBottleneck(self.directory.name, self.directory.name,
                                     num_latents=2, latent_dim=8, gaussian=False)
            trainer = pl.Trainer(accelerator="cpu", devices=1, precision="bf16-mixed", max_epochs=1,
                                 val_check_interval=0.5, num_sanity_val_steps=0,
                                 gradient_clip_val=1.0, callbacks=callbacks, logger=False,
                                 enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, train_dataloaders=DataLoader([self.batch() for _ in range(5)], batch_size=None),
                        val_dataloaders=DataLoader([self.batch()], batch_size=None))
            checkpoint = torch.load(Path(output) / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 5)
            self.assertTrue(torch.equal(checkpoint["state_dict"]["queries"], model.queries.detach()))
            self.assertTrue(checkpoint["optimizer_states"][0]["state"])
            self.assertTrue(checkpoint["lr_schedulers"])

    def test_benchmark_annotations_preserve_qa_identity_and_reject_missing_boxes(self):
        from lvr.eval.vlmeval import build_benchmark_sample

        path = Path(self.directory.name) / "annotated.png"
        Image.new("RGB", (100, 80)).save(path)
        message = [{"type": "image", "value": str(path)}, {"type": "text", "value": "Which object?"}]
        grid, _ = build_benchmark_sample(message, "example")
        annotation = {"benchmark": "example", "qa_prompt": grid["question"],
                      "images": [{"path": str(path.resolve()), "boxes": [[2, 3, 10, 12]]}],
                      "annotation_source": "official", "source_path": "original.png"}
        annotated, provenance = build_benchmark_sample(message, "example", {grid["id"]: annotation})
        self.assertEqual(annotated["id"], grid["id"])
        self.assertEqual(annotated["question"], grid["question"])
        self.assertEqual(annotated["images"][0]["boxes"], [[2, 3, 10, 12]])
        self.assertEqual(annotated["answer"], "")
        self.assertEqual(provenance["crop_policy"], "annotated_bbox")
        with self.assertRaises(KeyError):
            build_benchmark_sample(message, "example", {})
        annotation["images"][0]["boxes"] = []
        with self.assertRaises(ValueError):
            build_benchmark_sample(message, "example", {grid["id"]: annotation})

    def test_dataset_subset_preserves_native_prompt_and_scorer(self):
        from unittest.mock import patch
        import pandas as pd
        from lvr.eval.vlmeval import DatasetSubset

        class NativeDataset:
            dataset_name = "example"
            data = pd.DataFrame({"index": [10, 20, 30], "question": ["a", "b", "c"]})

            def __len__(self):
                return len(self.data)

            def build_prompt(self, row):
                return "native:" + row["question"]

            def evaluate(self, path):
                return {"native_metric": path}

        with patch("vlmeval.dataset.build_dataset", return_value=NativeDataset()):
            subset = DatasetSubset("example", ["30", "10"])
        self.assertEqual(subset.data["index"].tolist(), [10, 30])
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset.build_prompt(subset.data.iloc[1]), "native:c")
        self.assertEqual(subset.evaluate("predictions"), {"native_metric": "predictions"})


if __name__ == "__main__":
    unittest.main()
