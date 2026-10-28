"""CLI for the frozen final-student QA output-distillation experiment."""
from lvr.data.crop_qa import CropQADataModule
from lvr.models.qa_distillation import FrozenQAOutputDistillation
from lvr.predict_latents import PredictionCLI
from lvr.training.diagnostics import diagnostic_session


def main():
    with diagnostic_session():
        PredictionCLI(FrozenQAOutputDistillation, CropQADataModule,
                      seed_everything_default=42, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
