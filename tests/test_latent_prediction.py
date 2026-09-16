"""Exercise prediction and both CE gradient paths with an actual small Qwen stack."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from lvr.models.latent_prediction import LatentPrediction, latent_alignment_loss, prediction_prompt, predicted_probability, predicted_mask
from lvr.models.visual_bottleneck import VisualBottleneck
import test_visual_bottleneck as fixture


class LatentPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.VisualBottleneckTests.setUpClass()
        cls.root = Path(fixture.VisualBottleneckTests.directory.name)
        source = VisualBottleneck(str(cls.root), str(cls.root), num_latents=8, latent_dim=512,
                                  gaussian=False, train_qa=True, description_weight=1.0,
                                  encoder_includes_original=True)
        # This generation config is read from the base model directory on construction.
        source.qa.generation_config.pad_token_id = 0
        source.qa.generation_config.eos_token_id = 4
        source.qa.generation_config.save_pretrained(cls.root)
        cls.source = str(cls.root / "teacher.ckpt")
        torch.save({"hyper_parameters": dict(source.hparams), "state_dict": source.state_dict(),
                    "global_step": 961, "epoch": 0}, cls.source)

    @classmethod
    def tearDownClass(cls):
        fixture.VisualBottleneckTests.tearDownClass()

    def setUp(self):
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def model(self, phase="joint"):
        return LatentPrediction(self.source, prediction_phase=phase)

    @staticmethod
    def batch():
        batch = fixture.VisualBottleneckTests.batch(8)
        batch["description"] = {k: v[:, 3:].clone() for k, v in batch["qa"].items()
                                if k not in ("pixel_values", "image_grid_thw")}
        return batch

    def test_alignment_preserves_amplitude_and_stops_teacher_gradient(self):
        teacher = torch.randn(2, 8, 512, requires_grad=True)
        exact = latent_alignment_loss(teacher.clone(), teacher)
        self.assertLess(abs(exact["latent_loss"].item()), 1e-6)
        pred = (teacher.detach() * 2).requires_grad_()
        scaled = latent_alignment_loss(pred, teacher)
        torch.testing.assert_close(scaled["relative_mse"], torch.tensor(1.0))
        torch.testing.assert_close(scaled["cosine"], torch.tensor(1.0))
        scaled["latent_loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(pred.grad.norm(), 0)
        self.assertTrue(torch.isfinite(latent_alignment_loss(torch.zeros_like(pred), torch.zeros_like(pred))["latent_loss"]))

    def test_answer_is_physically_removed_and_cannot_change_prediction(self):
        model, batch = self.model(), self.batch()
        prompt = prediction_prompt(batch["qa"], 0)
        self.assertNotIn("labels", prompt)
        self.assertEqual(prompt["input_ids"].shape[1], 12)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            before = model.predict_latents(batch["qa"])
            batch["qa"]["input_ids"][:, 12:] = 40
            after = model.predict_latents(batch["qa"])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        # A short row must not retain its answer while a longer row still has slots.
        qa = {k: v.clone() for k, v in batch["qa"].items()}
        qa["slot_mask"][0] = False
        qa["slot_mask"][0, 3:11] = True
        qa["input_ids"][0, 11] = 40
        stripped = prediction_prompt(qa, 0)
        self.assertEqual(stripped["input_ids"][0, 11], 0)
        self.assertEqual(stripped["attention_mask"][0, 11], 0)

    def check_frozen(self, model):
        for part in (model.encoder, model.network.mu_head, model.network.projector, model.qa.model.visual):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in part.parameters()))
        self.assertIsNone(model.network.queries.grad)
        self.assertFalse(model.network.queries.requires_grad)
        self.assertNotEqual(model.prediction_queries.data_ptr(), model.network.queries.data_ptr())

    def test_warmup_retains_query_graph_through_frozen_b(self):
        model = self.model("warmup")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(self.batch())
        self.assertNotIn("ce", out)
        self.assertFalse(out["teacher"].requires_grad)
        out["loss"].backward()
        self.assertGreater(model.prediction_queries.grad.norm(), 0)
        self.assertGreater(model.prediction_head[1].weight.grad.norm(), 0)
        self.assertTrue(all(p.grad is None for p in model.qa.parameters()))
        self.check_frozen(model)

    def test_each_text_loss_trains_prediction_path_without_alignment_loss(self):
        model, batch = self.model(), self.batch()
        for part in ("qa", "description"):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                pred = model.predict_latents(batch["qa"])
                loss = model.network.answer_loss(batch[part], pred)
            loss.backward()
            self.assertGreater(model.prediction_queries.grad.norm(), 0, part)
            self.assertGreater(model.prediction_head[1].weight.grad.norm(), 0, part)
            self.assertGreater(model.qa.model.language_model.layers[0].self_attn.q_proj.weight.grad.norm(), 0, part)
            self.check_frozen(model)

    def test_curriculum_validation_and_fresh_joint_optimizer(self):
        self.assertEqual([predicted_probability(i) for i in (0, 199, 200, 499, 500, 960)], [.2, .2, .5, .5, 1., 1.])
        ids = [f"sample-{i}" for i in range(128)]
        together = predicted_mask(ids, 230, 42, .5, "cpu")
        split = torch.cat([predicted_mask(ids[i:i+4], 230, 42, .5, "cpu") for i in range(0, 128, 4)])
        self.assertTrue(torch.equal(together, split))
        self.assertGreater(together.sum(), 0)
        self.assertLess(together.sum(), len(ids))
        warm = self.model("warmup")
        path = self.root / "warmup.ckpt"
        torch.save({"hyper_parameters": dict(warm.hparams), "state_dict": warm.state_dict(),
                    "global_step": 200, "epoch": 0}, path)
        model = self.model()
        report = model.initialize_from_checkpoint(str(path))
        self.assertFalse(report["optimizer_restored"])
        self.assertTrue(model.qa.lm_head.weight.requires_grad)
        model._trainer = SimpleNamespace(estimated_stepping_batches=961)
        optimizer = model.configure_optimizers()["optimizer"]
        self.assertFalse(optimizer.state)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                         {id(p) for p in model.parameters() if p.requires_grad})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(self.batch(), validation=True)
        self.assertEqual(out["predicted_fraction"], 1)
        self.assertIn("teacher_ce", out)
        restored = LatentPrediction.from_checkpoint(path)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))

    def test_lightning_max_steps_writes_reusable_checkpoint(self):
        model = self.model("warmup")
        callback = pl.callbacks.ModelCheckpoint(dirpath=self.root / "fit", filename="final",
                                                save_on_train_epoch_end=True, save_last="link")
        trainer = pl.Trainer(accelerator="cpu", devices=1, precision="bf16-mixed", max_steps=2,
                             max_epochs=1, limit_train_batches=2, limit_val_batches=0,
                             callbacks=[callback], logger=False, enable_progress_bar=False,
                             enable_model_summary=False, gradient_clip_val=1.0)
        trainer.fit(model, train_dataloaders=DataLoader([self.batch()] * 2, batch_size=None))
        self.assertTrue(Path(callback.last_model_path).is_file())
        loaded = LatentPrediction.from_checkpoint(callback.last_model_path)
        self.assertEqual(loaded.checkpoint_step, 2)

    def test_cli_keeps_nested_teacher_hyperparameters_separate(self):
        from lvr.predict_latents import PredictionCLI
        from lvr.data.crop_qa import CropQADataModule

        cli = PredictionCLI(LatentPrediction, CropQADataModule, run=False, save_config_callback=None,
                            args={"model": {"source_checkpoint": self.source},
                                  "data": {"processor_name_or_path": str(self.root), "image_root": str(self.root)},
                                  "trainer": {"accelerator": "cpu", "devices": 1, "logger": False,
                                              "enable_checkpointing": False},
                                  "exp_name": str(self.root / "cli"), "version": "nested"})
        self.assertEqual(cli.model.network.hparams.encoder_type, "qwen")
        self.assertEqual(cli.model.hparams.source_checkpoint, self.source)
        self.assertNotIn("prediction_phase", cli.model.network.hparams)

    def test_relocated_checkpoint_uses_explicit_source_and_base_paths(self):
        import shutil

        moved = self.root / "moved_base"
        moved.mkdir(exist_ok=True)
        for path in self.root.iterdir():
            if path.suffix in (".json", ".safetensors"):
                shutil.copyfile(path, moved / path.name)
        original = self.model()
        teacher = torch.load(self.source, weights_only=False)
        teacher["hyper_parameters"].update(encoder_model_name_or_path="/nonexistent/old-base",
                                             qa_model_name_or_path="/nonexistent/old-base")
        source = self.root / "moved_teacher.ckpt"
        torch.save(teacher, source)
        hparams = dict(original.hparams, source_checkpoint="/nonexistent/old-teacher.ckpt")
        prediction = self.root / "moved_prediction.ckpt"
        torch.save({"hyper_parameters": hparams, "state_dict": original.state_dict(), "global_step": 961}, prediction)
        restored = LatentPrediction.from_checkpoint(prediction, source_checkpoint=source, base_model_path=moved)
        self.assertEqual(restored.checkpoint_step, 961)
        qa = self.batch()["qa"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.testing.assert_close(original.predict_latents(qa), restored.predict_latents(qa), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
