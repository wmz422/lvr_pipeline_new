import unittest
from types import SimpleNamespace
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.plugins.precision import DeepSpeedPrecision
from lightning.pytorch.loops.optimization.automatic import ClosureResult
from deepspeed.runtime.engine import DeepSpeedEngine
from reconstruction_experiment.precision import MatchedDeepSpeedPrecision


class Engine:
    scale_wrt_gas = None

    def __init__(self, accumulation):
        self.accumulation = accumulation

    def gradient_accumulation_steps(self):
        return self.accumulation

    def backward(self, loss, **kwargs):
        # Use the installed DeepSpeed normalization implementation; optimizer
        # and distributed reduction are unnecessary to test this scalar factor.
        loss = DeepSpeedEngine._scale_loss_by_gas(self, loss.float())
        loss.backward()


def gradient(accumulation, plugin):
    model = LightningModule()
    model.weight = torch.nn.Parameter(torch.tensor([.4, -.2, .1]))
    model.trainer = SimpleNamespace(accumulate_grad_batches=accumulation, model=Engine(accumulation))
    x = torch.arange(96, dtype=torch.float32).reshape(32, 3) / 96
    y = torch.linspace(.1, .9, 32)
    for inputs, targets in zip(x.chunk(accumulation), y.chunk(accumulation)):
        loss = ((inputs @ model.weight - targets)**2).mean()
        normalized = ClosureResult.from_training_step_output(loss, normalize=accumulation).closure_loss
        plugin.backward(normalized, model, None)
    return model.weight.grad


class PrecisionScalingTests(unittest.TestCase):
    def test_default_recipe_gradient_preserved_across_microbatches(self):
        for reference in (8, 16):
            expected = gradient(reference, DeepSpeedPrecision("bf16-mixed"))
            for accumulation in (1, 2, 4, 8, 16):
                actual = gradient(accumulation, MatchedDeepSpeedPrecision(reference_accumulation=reference))
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)

    def test_installed_uncompensated_path_exposes_accumulation_scaling(self):
        default = gradient(16, DeepSpeedPrecision("bf16-mixed"))
        larger_batch = gradient(4, DeepSpeedPrecision("bf16-mixed"))
        torch.testing.assert_close(larger_batch, 4 * default, rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
