"""Plain SFT with an accumulation-aligned, resumable training batch sampler."""
from lvr.stage1 import StageOneCLI
from lvr.models.visual_bottleneck import VisualBottleneck
from lvr.training.diagnostics import diagnostic_session
from continuation_runtime import ContinuationDataModule


if __name__ == "__main__":
    with diagnostic_session():
        StageOneCLI(VisualBottleneck, ContinuationDataModule, seed_everything_default=42,
                    save_config_kwargs={"overwrite": True})
