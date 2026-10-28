"""Compare the real Qwen2.5-VL training/generation path against native Transformers."""
import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
from transformers import GenerationConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from lvr.models.visual_bottleneck import VisualBottleneck


class Qwen25PlainSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory()
        config = Qwen2_5_VLConfig(
            text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2,
                             rope_parameters={"rope_type": "default", "mrope_section": [8, 12, 12]}),
            vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                               out_hidden_size=128, patch_size=14, temporal_patch_size=2,
                               spatial_merge_size=2, window_size=56, fullatt_block_indexes=[1]),
            image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3,
        )
        Qwen2_5_VLForConditionalGeneration(config).save_pretrained(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        torch.manual_seed(42)

    @staticmethod
    def batch():
        ids = torch.tensor([[2, 1, 3, 10, 11, 12, 4, 0, 0, 0],
                            [2, 1, 3, 2, 1, 3, 10, 11, 12, 4]])
        labels = torch.full_like(ids, -100)
        labels[0, 4:7], labels[1, 7:] = ids[0, 4:7], ids[1, 7:]
        return dict(input_ids=ids, attention_mask=(ids != 0).long(), labels=labels,
                    slot_mask=torch.zeros_like(ids, dtype=torch.bool), mm_token_type_ids=(ids == 1).long(),
                    image_grid_thw=torch.tensor([[1, 2, 2]] * 3), pixel_values=torch.randn(12, 1176))

    def test_native_loss_and_language_gradients_with_frozen_vision(self):
        model = VisualBottleneck(self.directory.name, self.directory.name, num_latents=0,
                                 gaussian=False, train_qa=True, gradient_checkpointing=True)
        self.assertIsInstance(model.qa, Qwen2_5_VLForConditionalGeneration)
        self.assertIsNone(model.encoder)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
        batch = self.batch()
        # Native and chunked answer-only CE must agree, including multi-image MRoPE and padding.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.answer_loss(batch)
            expected = model.qa(**{k: v for k, v in batch.items() if k != "slot_mask"}, use_cache=False).loss
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        actual.backward()
        self.assertTrue(torch.isfinite(actual))
        for parameter in (model.qa.model.language_model.layers[0].self_attn.q_proj.weight,
                          model.qa.model.language_model.embed_tokens.weight, model.qa.lm_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertFalse(model.qa.model.visual.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.qa.model.visual.parameters()))

    def test_bf16_without_autocast_matches_deepspeed_compute_contract(self):
        model = VisualBottleneck(self.directory.name, self.directory.name, num_latents=0,
                                 gaussian=False, train_qa=True).bfloat16()
        with torch.autocast("cpu", enabled=False):
            loss = model.answer_loss(self.batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.qa.lm_head.weight.grad.abs().sum().item(), 0)

    def test_native_generation_and_checkpoint_roundtrip(self):
        model = VisualBottleneck.from_pretrained(self.directory.name, "cpu")
        native = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.directory.name, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        for key, value in model.qa.state_dict().items():
            torch.testing.assert_close(value, native.state_dict()[key], rtol=0, atol=0)
        for key, value in model.qa.named_buffers():
            torch.testing.assert_close(value, dict(native.named_buffers())[key], rtol=0, atol=0)
        batch = self.batch()
        # One prompt-only image; no target answer may enter generation.
        for key in ("input_ids", "attention_mask", "slot_mask", "mm_token_type_ids"):
            batch[key] = batch[key][:1, :4]
        batch.pop("labels")
        batch["image_grid_thw"] = batch["image_grid_thw"][:1]
        batch["pixel_values"] = batch["pixel_values"][:4]
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=4,
                                  pad_token_id=0, eos_token_id=None, use_cache=True)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = native.generate(**{k: v for k, v in batch.items() if k != "slot_mask"},
                                       generation_config=config)[:, 4:]
            actual = model.generate_answers(batch, max_new_tokens=4, pad_token_id=0)
        self.assertTrue(torch.equal(expected, actual))
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as handle:
            torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                        "global_step": 7, "epoch": 0}, handle.name)
            loaded = VisualBottleneck.from_checkpoint(handle.name, "cpu")
        self.assertEqual(loaded.checkpoint_step, 7)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.equal(actual, loaded.generate_answers(batch, max_new_tokens=4, pad_token_id=0)))

    def test_candidate_validation_scores_each_sample_without_padding_bias(self):
        path = Path(__file__).resolve().parents[1] / "scripts/reasoning_data/validate_plain_sft_candidates.py"
        spec = importlib.util.spec_from_file_location("candidate_validation_test", path)
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        model = VisualBottleneck.from_pretrained(self.directory.name, "cpu")
        batch = self.batch()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            rows = validation.score_batch(model, batch)
            expected = float(model.answer_loss(batch))
        weighted = sum(r["nll_sum"] for r in rows) / sum(r["target_tokens"] for r in rows)
        self.assertAlmostEqual(weighted, expected, places=5)
        self.assertEqual([r["target_tokens"] for r in rows], [3, 3])
        for i, (grid_start, grid_end) in enumerate(((0, 1), (1, 3))):
            single = {key: value[i:i + 1] for key, value in batch.items()
                      if key not in ("image_grid_thw", "pixel_values")}
            single.update(image_grid_thw=batch["image_grid_thw"][grid_start:grid_end],
                          pixel_values=batch["pixel_values"][grid_start * 4:grid_end * 4])
            with torch.autocast("cpu", dtype=torch.bfloat16):
                alone = validation.score_batch(model, single)[0]
            self.assertAlmostEqual(rows[i]["sample_ce"], alone["sample_ce"], places=5)
        summary = validation.summarize([dict(row, group=group, source=group)
                                       for row, group in zip(rows, ("old", "reasoning"))])
        self.assertEqual(summary["all"]["samples"], 2)
        self.assertEqual(summary["group/old"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
