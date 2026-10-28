"""Compare compact transport with the original loss and condition gradients."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from flow_reconstruction.data import read_tensor, FlowCollator
from reconstruction_experiment.data import read_rgb8, CompactFlowCollator
from reconstruction_experiment.compact_flow import CompactFrozenFlowRuntime

FLOW = Path(__file__).resolve().parents[3] / "doc/flow_reconstruction_parallel_20261015"
spec = importlib.util.spec_from_file_location("original_parallel_test_helpers", FLOW / "tests/test_parallel.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class CompactFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pixels_collator_and_transfer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
            Image.fromarray(pixels).save(path)
            compact, original = read_rgb8(path), read_tensor(path)
            self.assertTrue(torch.equal(compact.float().div(127.5).sub(1), original))
            self.assertEqual(compact.numel() * compact.element_size() * 4,
                             original.numel() * original.element_size())
            row = {"target": {"path": "rgb.png", "pixel_sha256": "pixels"},
                   "references": [{"path": "rgb.png", "pixel_sha256": "pixels"}]}
            def base(examples):
                return {"ids": [item["id"] for item in examples], "qa": {"input_ids": torch.tensor([[1, 2]])}}
            old = FlowCollator(base, directory, {"a": row})([{"id": "a"}])
            new = CompactFlowCollator(base, directory, {"a": row})([{"id": "a"}])
            self.assertEqual(old["flow"]["target_keys"], new["flow"]["target_keys"])
            self.assertEqual(old["flow"]["reference_keys"], new["flow"]["reference_keys"])
            self.assertTrue(torch.equal(old["qa"]["input_ids"], new["qa"]["input_ids"]))
            self.assertEqual(new["flow"]["targets"][0].dtype, torch.uint8)
            stress = CompactFlowCollator(base, directory, {"a": row}, cache_keys=False)([{"id": "a"}])
            self.assertNotIn("target_keys", stress["flow"])

    def test_frozen_vae_cache_noise_loss_and_input_gradients_match(self):
        torch.manual_seed(4)
        model = helpers.tiny()
        targets = [torch.randint(256, (3, 32, width), dtype=torch.uint8) for width in (32, 48, 32)]
        shared = torch.randint(256, (3, 16, 32), dtype=torch.uint8)
        references = [[shared], [shared, shared], [shared]]
        initial_condition = torch.randn(3, 8, 16)
        for validation in (False, True):
            results = []
            for compact in (False, True):
                runtime = helpers.runtime(copy.deepcopy(model), batch=2, vae_batch=2)
                if compact:
                    runtime.__class__ = CompactFrozenFlowRuntime
                def convert(x):
                    return x if compact else x.float().div(127.5).sub(1)
                condition = initial_condition.clone().requires_grad_()
                torch.manual_seed(17)
                kwargs = dict(validation=validation, target_keys=["a", "b", "c"],
                              reference_keys=[["ref"] * len(refs) for refs in references])
                value = runtime.loss([convert(x) for x in targets],
                    [[convert(x) for x in refs] for refs in references], condition, ["a", "b", "c"], **kwargs)
                value.backward()
                rng_state = torch.get_rng_state().clone()
                cache = [x.clone() for x in runtime._encoded_cache.values()]
                calls = list(runtime.vae.calls)
                runtime.loss([convert(x) for x in targets],
                    [[convert(x) for x in refs] for refs in references], condition, ["a", "b", "c"], **kwargs)
                self.assertEqual(runtime.vae.calls, calls)
                self.assertTrue(all(p.grad is None for p in runtime.transformer.parameters()))
                self.assertGreater(condition.grad.abs().sum().item(), 0)
                results.append((value.detach(), condition.grad, rng_state, cache))
            for before, after in zip(results[0][:3], results[1][:3]):
                self.assertTrue(torch.equal(before, after))
            for before, after in zip(results[0][3], results[1][3]):
                self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
