"""Baseline warmup -> joint with current-Student output distillation."""
from lvr.data.crop_qa import CropQADataModule
from lvr.models.joint_qa_distillation import JointQAOutputDistillation
from lvr.predict_latents import PredictionCLI
from lvr.training.diagnostics import diagnostic_session


def main():
    with diagnostic_session():
        PredictionCLI(JointQAOutputDistillation, CropQADataModule,
                      seed_everything_default=42, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
