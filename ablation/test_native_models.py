"""Real, randomly initialized tiny Qwen models; CPU coverage, not trained scores."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Run native model checks with the inference Python")

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "lvr_pipeline_new/src")]
from common import MODES, latent_source
from run import read_latent
from safetensors.torch import save_file
from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen3VLConfig, Qwen3VLForConditionalGeneration)
from lvr.models.visual_bottleneck import VisualBottleneck
from lvr.models.latent_prediction import LatentPrediction


class NativeModelTests(unittest.TestCase):
    @staticmethod
    def prompt(question_token=10, images=1):
        ids = torch.tensor([[2, 1, 3] * images + [question_token] + [0] * 8])
        slots = torch.zeros_like(ids, dtype=torch.bool)
        slots[:, -8:] = True
        return {"input_ids": ids, "slot_mask": slots, "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": (ids == 1).long(), "image_grid_thw": torch.tensor([[1, 2, 2]] * images),
                "pixel_values": torch.randn(4 * images, 1176)}

    def test_both_native_families_restore_and_generate_all_five_conditions(self):
        torch.set_num_threads(2)
        for config_class, model_class in ((Qwen3VLConfig, Qwen3VLForConditionalGeneration),
                                          (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration)):
            with self.subTest(family=config_class.model_type), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                root = Path(directory)
                base = root / "base"
                vision = dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                              out_hidden_size=128, patch_size=14, temporal_patch_size=2, spatial_merge_size=2)
                if config_class is Qwen3VLConfig:
                    vision.update(deepstack_visual_indexes=[0, 1], num_position_embeddings=64)
                else:
                    vision.update(window_size=56, fullatt_block_indexes=[1])
                config = config_class(text_config=dict(vocab_size=64, hidden_size=128,
                            intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=64,
                            rope_parameters={"rope_type": "default", "mrope_section": [8, 12, 12]}),
                            vision_config=vision, image_token_id=1, video_token_id=5,
                            vision_start_token_id=2, vision_end_token_id=3)
                native = model_class(config)
                native.generation_config.pad_token_id = 0
                native.generation_config.eos_token_id = 63
                native.save_pretrained(base)
                del native
                teacher = VisualBottleneck(str(base), str(base), gaussian=False,
                            gradient_checkpointing=False, encoder_includes_original=True)
                teacher_path = root / "teacher.ckpt"
                torch.save({"state_dict": teacher.state_dict(), "hyper_parameters": dict(teacher.hparams),
                            "global_step": 3, "epoch": 0}, teacher_path)
                del teacher
                model = LatentPrediction(str(teacher_path), prediction_phase="joint", description_weight=0,
                                         gradient_checkpointing=False).requires_grad_(False).eval()
                student_path = root / "student.ckpt"
                torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                            "global_step": 7, "epoch": 0}, student_path)
                del model
                relocated = root / "relocated base"
                base.rename(relocated)
                model = LatentPrediction.from_checkpoint(student_path, "cpu", source_checkpoint=teacher_path,
                                                        base_model_path=relocated)
                self.assertEqual(model.checkpoint_step, 7)
                qa = self.prompt()
                donor_qa = self.prompt(11)
                teacher_qa = self.prompt(images=2)
                donor_teacher_qa = self.prompt(11, images=2)
                unchanged = {key: value.clone() for key, value in qa.items()}
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    cache = {"target": {"teacher": model.network.encode(teacher_qa)[0],
                                        "predicted": model.predict_latents(qa)},
                             "donor": {"teacher": model.network.encode(donor_teacher_qa)[0],
                                       "predicted": model.predict_latents(donor_qa)}}
                    donors = {"target": {"id": "donor"}}
                    for mode in MODES:
                        source, kind = latent_source(mode, "target", donors)
                        latent = torch.zeros((1, 8, 512)) if mode == "zero" else cache[source][kind]
                        self.assertEqual(tuple(latent.shape), (1, 8, 512))
                        self.assertTrue(torch.isfinite(latent).all())
                        projected = []
                        handle = model.network.projector.register_forward_pre_hook(
                            lambda module, inputs: projected.append(inputs[0].detach().float().clone()))
                        try:
                            output = model.network.generate_answers(qa, latent, max_new_tokens=2)
                        finally:
                            handle.remove()
                        self.assertEqual(tuple(output.shape), (1, 2))
                        self.assertEqual(len(projected), 1)
                        torch.testing.assert_close(projected[0], latent.float(), rtol=.01, atol=.01)
                        for key in qa:
                            self.assertTrue(torch.equal(qa[key], unchanged[key]))

    def test_latent_cache_rejects_wrong_shapes_metadata_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latents.safetensors"
            values = {"teacher": torch.ones(1, 8, 512), "predicted": torch.zeros(1, 8, 512)}
            save_file(values, str(path), metadata={"id": "target", "run_sha256": "binding"})
            self.assertEqual(set(read_latent(path, "target", "binding")), set(values))
            with self.assertRaises(ValueError):
                read_latent(path, "donor", "binding")
            values["teacher"][0, 0, 0] = float("nan")
            save_file(values, str(path), metadata={"id": "target", "run_sha256": "binding"})
            with self.assertRaises(ValueError):
                read_latent(path, "target", "binding")
            values["teacher"] = torch.zeros(1, 7, 512)
            save_file(values, str(path), metadata={"id": "target", "run_sha256": "binding"})
            with self.assertRaises(ValueError):
                read_latent(path, "target", "binding")


if __name__ == "__main__":
    unittest.main()
