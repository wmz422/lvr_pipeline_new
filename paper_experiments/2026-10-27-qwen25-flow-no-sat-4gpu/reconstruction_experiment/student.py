"""Original latent prediction model, with accumulation-aligned input batches."""
from lvr.models.latent_prediction import LatentPrediction
from lvr.predict_latents import PredictionCLI
from lvr.training.diagnostics import diagnostic_session
from .data import StudentDataModule


def main():
    with diagnostic_session():
        PredictionCLI(LatentPrediction, StudentDataModule, seed_everything_default=42,
                      save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
