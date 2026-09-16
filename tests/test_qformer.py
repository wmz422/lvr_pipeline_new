"""CPU tests for variable-resolution packing, question conditioning and joint QA."""

from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from transformers import AutoProcessor

from lvr.data.crop_qa import CropQACollator, crop_bounds
from lvr.models.qformer import QueryFormerEncoder
from lvr.models.visual_bottleneck import VisualBottleneck
import test_visual_bottleneck as fixtures


class QueryFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.VisualBottleneckTests.setUpClass()
        cls.directory = fixtures.VisualBottleneckTests.directory

    @classmethod
    def tearDownClass(cls):
        fixtures.VisualBottleneckTests.tearDownClass()

    def setUp(self):
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(42)

    def model(self, **kwargs):
        options = dict(num_latents=32, latent_dim=512,
            gaussian=False, train_qa=True, encoder_type="qformer", encoder_uses_question=True,
            qformer_hidden_size=32, qformer_num_layers=2, qformer_num_heads=4)
        return VisualBottleneck(self.directory.name, self.directory.name, **(options | kwargs))

    @staticmethod
    def batch():
        batch = fixtures.VisualBottleneckTests.batch(num_latents=32)
        batch["encoder"].update(
            image_counts=torch.tensor([1, 1]),
            image_metadata=torch.tensor([[0., 0., 1., 1., 0., 1., 0.]] * 2),
            question_input_ids=torch.tensor([[10, 11, 0], [12, 13, 14]]),
            question_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
        return batch

    def test_variable_resolution_batch_isolation_and_padding(self):
        encoder = QueryFormerEncoder(16, 32, 2, 4, uses_question=True,
                                     use_geometry=True, gradient_checkpointing=False).eval()
        # Two differently shaped crops in sample 0; a larger crop in sample 1.
        features = [torch.randn(2, 16), torch.randn(3, 16), torch.randn(8, 16)]
        batch = {
            "image_counts": torch.tensor([2, 1]),
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 6, 2], [1, 4, 8]]),
            "image_metadata": torch.tensor([[0., 0., .5, 1., 0, 1, 0],
                                             [.5, 0., 1., 1., 0, 2, 0],
                                             [0., 0., 1., 1., 1, 1, 0]]),
            "question_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
        }
        queries, text = torch.randn(32, 32), torch.randn(2, 4, 16)
        output = encoder(features, batch, queries, 2, text)
        self.assertEqual(tuple(output.shape), (2, 32, 32))
        single = {"image_counts": torch.tensor([2]), "image_grid_thw": batch["image_grid_thw"][:2],
                  "image_metadata": batch["image_metadata"][:2],
                  "question_attention_mask": torch.ones(1, 2)}
        expected = encoder(features[:2], single, queries, 2, text[:1, :2])
        torch.testing.assert_close(output[:1], expected, atol=2e-6, rtol=2e-5)
        altered = [features[0], features[1], torch.randn_like(features[2]) * 100]
        padded_text = text.clone()
        padded_text[0, 2:] = 10000
        other = encoder(altered, batch, queries, 2, padded_text)
        torch.testing.assert_close(output[:1], other[:1], atol=2e-6, rtol=2e-5)
        self.assertGreater((output[1] - other[1]).abs().max().item(), 1e-5)
        real_crop_change = encoder([features[0] + torch.randn_like(features[0]), *features[1:]],
                                   batch, queries, 2, text)
        self.assertGreater((output[0] - real_crop_change[0]).abs().max().item(), 1e-5)

    def test_collator_crop_only_geometry_and_question_answer_separation(self):
        from pathlib import Path

        root = Path(self.directory.name)
        Image.new("RGB", (160, 96), (10, 80, 140)).save(root / "qformer.png")
        processor = AutoProcessor.from_pretrained(fixtures.RAW_QWEN, local_files_only=True)
        item = {"id": "crop", "question_image": "qformer.png",
                "boxes": [[-4, 4, 33.2, 90], [60, 20, 150, 50]],
                "question": "What color?\nA. Blue\nB. Red", "answer": "ANSWER_SENTINEL_123"}
        collator = CropQACollator(processor, root, num_latents=32, original_pixels=16384,
                                 crop_pixels=8192, encoder_type="qformer", encoder_uses_question=True,
                                 encoder_includes_original=False)
        batch = collator([item, {**item, "id": "one", "boxes": item["boxes"][:1], "question": "Color?"}])
        a = batch["encoder"]
        self.assertEqual(a["image_counts"].tolist(), [2, 1])
        self.assertEqual(len(a["image_grid_thw"]), 3)
        self.assertEqual(len(batch["qa"]["image_grid_thw"]), 2)
        self.assertTrue((a["slot_mask"].sum(-1) == 32).all())
        torch.testing.assert_close(a["image_metadata"][0, :4], torch.tensor([0, 4/96, 34/160, 90/96]))
        question = processor.tokenizer.decode(a["question_input_ids"][0][a["question_attention_mask"][0].bool()])
        self.assertEqual(question, item["question"])
        first = collator([item])["encoder"]
        changed = collator([{**item, "answer": "different"}])["encoder"]
        for key in first:
            self.assertTrue(torch.equal(first[key], changed[key]), key)
        collator.encoder_includes_original = True
        original = collator([item])["encoder"]
        self.assertEqual(original["image_counts"].tolist(), [3])
        self.assertEqual(original["image_metadata"][0].tolist(), [0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(crop_bounds((160, 96), [5, 7, 5, 7]), [5, 7, 6, 8])

    def test_answer_ce_updates_all_trainable_parameters_and_optimizer_groups(self):
        model = self.model(qformer_use_geometry=True)
        model.trainer = SimpleNamespace(estimated_stepping_batches=4)
        optimizer = model.configure_optimizers()["optimizer"]
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-4, 1e-5, 1e-4])
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters() if p.requires_grad})
        before = model.queries.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(self.batch())
            self.assertEqual(tuple(output["mu"].shape), (2, 32, 512))
            self.assertEqual(output["kl"].item(), 0)
            output["ce"].backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.queries))
        self.assertFalse(hasattr(model.encoder, "model"))
        self.assertFalse(model.qa.model.visual.training)

    def test_encoder_responds_to_question_without_updating_shared_qa_embeddings(self):
        model = self.model(gradient_checkpointing=False)
        batch = self.batch()["encoder"]
        mu, _ = model.encode(batch)
        changed = {**batch, "question_input_ids": torch.full_like(batch["question_input_ids"], 17)}
        other, _ = model.encode(changed)
        self.assertGreater((mu - other).abs().max().item(), 1e-6)
        mu.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.qa.parameters()))
        self.assertGreater(model.encoder.text_projection[1].weight.grad.abs().sum().item(), 0)

    def test_checkpoint_round_trip_and_prompt_generation(self):
        from pathlib import Path

        model = self.model(encoder_includes_original=False, qformer_use_geometry=True,
                           gradient_checkpointing=False).eval()
        batch = self.batch()["encoder"]
        expected, _ = model.encode(batch)
        path = Path(self.directory.name) / "qformer.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "global_step": 2}, path)
        restored = VisualBottleneck.from_checkpoint(path, "cpu")
        actual, _ = restored.encode(batch)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        self.assertFalse(restored.hparams.encoder_includes_original)
        self.assertTrue(restored.hparams.qformer_use_geometry)
        self.assertEqual(restored.hparams.encoder_type, "qformer")
        for latent in (actual, torch.zeros_like(actual), None):
            tokens = restored.generate_answers(batch, latent, max_new_tokens=1, pad_token_id=0)
            self.assertEqual(tuple(tokens.shape), (2, 1))

    def test_zero_latent_keeps_positions_and_old_checkpoint_defaults(self):
        from pathlib import Path
        from unittest.mock import patch

        model = self.model(gradient_checkpointing=False).eval()
        batch = self.batch()["encoder"]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return torch.cat((kwargs["input_ids"], torch.ones(2, 1, dtype=torch.long)), dim=1)

        with patch.object(model.qa, "generate", side_effect=generate):
            model.generate_answers(batch, torch.zeros(2, 32, 512), max_new_tokens=1, pad_token_id=0)
            model.generate_answers(batch, None, max_new_tokens=1, pad_token_id=0)
        self.assertEqual(calls[0]["input_ids"].shape[1] - calls[1]["input_ids"].shape[1], 32)
        expected = model.projector(torch.zeros(2, 32, 512))
        torch.testing.assert_close(calls[0]["inputs_embeds"][:, -32:], expected)
        legacy = VisualBottleneck(self.directory.name, self.directory.name, num_latents=2,
                                  latent_dim=8, gaussian=False, gradient_checkpointing=False)
        hp = {k: v for k, v in legacy.hparams.items()
              if k not in ("encoder_type", "encoder_includes_original") and not k.startswith("qformer_")}
        path = Path(self.directory.name) / "legacy.ckpt"
        torch.save({"state_dict": legacy.state_dict(), "hyper_parameters": hp, "global_step": 1}, path)
        restored = VisualBottleneck.from_checkpoint(path, "cpu")
        self.assertEqual(restored.hparams.encoder_type, "qwen")
        self.assertTrue(restored.hparams.encoder_includes_original)

    def test_question_blind_qformer_and_invalid_grid(self):
        encoder = QueryFormerEncoder(16, 32, 2, 4, uses_question=False, gradient_checkpointing=False)
        batch = {"image_counts": torch.tensor([1]), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                 "image_metadata": torch.tensor([[0., 0., 1., 1., 0, 1, 0]])}
        output = encoder([torch.randn(1, 16)], batch, torch.randn(32, 32), 2)
        self.assertEqual(tuple(output.shape), (1, 32, 32))
        with self.assertRaisesRegex(ValueError, "length"):
            encoder([torch.randn(2, 16)], batch, torch.randn(32, 32), 2)

    def test_crop_only_inputs_ignore_question_answer_and_pixels_outside_crops(self):
        from pathlib import Path
        root = Path(self.directory.name)
        path = root / "pure_visual.png"
        image = Image.new("RGB", (160, 96), (10, 80, 140))
        image.save(path)
        processor = AutoProcessor.from_pretrained(fixtures.RAW_QWEN, local_files_only=True)
        collator = CropQACollator(
            processor, root, num_latents=32, original_pixels=16384, crop_pixels=8192,
            encoder_type="qformer", encoder_uses_question=False, encoder_includes_original=False)
        item = {"id": "visual", "question_image": path.name,
                "boxes": [[16, 16, 48, 48], [80, 16, 128, 48]],
                "question": "Color? A. Blue B. Red", "answer": "Blue"}
        first = collator([item])
        image.paste((240, 20, 10), (0, 70, 160, 96))
        image.save(path)
        other = collator([{**item, "question": "Position? A. Left B. Right", "answer": "Right"}])
        self.assertEqual(first["encoder"]["image_counts"].tolist(), [2])
        self.assertEqual(len(first["encoder"]["image_grid_thw"]), 2)
        self.assertEqual(len(first["qa"]["image_grid_thw"]), 1)
        self.assertFalse(any(key.startswith("question_") for key in first["encoder"]))
        for key in first["encoder"]:
            self.assertTrue(torch.equal(first["encoder"][key], other["encoder"][key]), key)
        self.assertFalse(torch.equal(first["qa"]["pixel_values"], other["qa"]["pixel_values"]))
        self.assertFalse(torch.equal(first["qa"]["input_ids"], other["qa"]["input_ids"]))

    def test_pure_visual_model_ce_gradient_and_checkpoint(self):
        from pathlib import Path

        model = self.model(encoder_uses_question=False, encoder_includes_original=False)
        self.assertIsNone(model.encoder.text_projection)
        self.assertTrue(all(not hasattr(layer, "intermediate") for layer in model.encoder.transformer.encoder.layer))
        batch = self.batch()
        batch["encoder"] = {k: v for k, v in batch["encoder"].items() if not k.startswith("question_")}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(batch)
        self.assertEqual(tuple(output["mu"].shape), (2, 32, 512))
        output["ce"].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        model.eval()
        with torch.no_grad():
            expected, _ = model.encode(batch["encoder"])
        path = Path(self.directory.name) / "pure_visual.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "global_step": 1}, path)
        restored = VisualBottleneck.from_checkpoint(path, "cpu")
        actual, _ = restored.encode(batch["encoder"])
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        self.assertFalse(restored.hparams.encoder_uses_question)
        self.assertFalse(restored.hparams.encoder_includes_original)

    def test_training_cli_and_benchmark_restore_the_encoder_contract(self):
        from unittest.mock import patch
        from lvr.data.crop_qa import CropQADataModule
        from lvr.eval.vlmeval import BottleneckModel
        from lvr.stage1 import StageOneCLI

        args = {
            "model": {"encoder_model_name_or_path": self.directory.name,
                      "qa_model_name_or_path": self.directory.name, "num_latents": 32,
                      "encoder_type": "qformer", "encoder_uses_question": False,
                      "encoder_includes_original": False, "qformer_use_geometry": False,
                      "qformer_hidden_size": 32, "qformer_num_layers": 2, "qformer_num_heads": 4,
                      "gaussian": False, "train_qa": True},
            "data": {"processor_name_or_path": fixtures.RAW_QWEN, "image_root": self.directory.name,
                     "num_latents": 32, "num_workers": 0},
            "trainer": {"accelerator": "cpu", "devices": 1, "logger": False,
                        "enable_checkpointing": False, "enable_progress_bar": False},
            "exp_name": self.directory.name, "version": "cli_contract",
        }
        cli = StageOneCLI(VisualBottleneck, CropQADataModule, args=args, run=False, save_config_callback=None)
        self.assertEqual(cli.datamodule.hparams.encoder_type, "qformer")
        self.assertFalse(cli.datamodule.hparams.encoder_uses_question)
        self.assertFalse(cli.datamodule.hparams.encoder_includes_original)
        from pathlib import Path
        config = Path(__file__).resolve().parents[1] / "configs/latent_prediction/warmup.yaml"
        with patch.object(VisualBottleneck, "from_checkpoint", return_value=cli.model):
            for mode in ("mu", "zero", "none"):
                adapter = BottleneckModel("unused", str(config), latent_mode=mode, base_model_path=fixtures.RAW_QWEN)
                self.assertEqual(adapter.collator.encoder_type, "qformer")
                self.assertFalse(adapter.collator.encoder_uses_question)
                self.assertFalse(adapter.collator.encoder_includes_original)
                self.assertEqual(adapter.collator.include_encoder, mode == "mu")


if __name__ == "__main__":
    unittest.main()
