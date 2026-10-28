"""Train prediction warm-up or QA/description joint adaptation in the shared run layout."""

from lvr.cli import LvrCLI
from lvr.data.crop_qa import CropQADataModule
from lvr.models.latent_prediction import LatentPrediction
from lvr.training.diagnostics import diagnostic_session


class PredictionCLI(LvrCLI):
    def __init__(self, *args, **kwargs):
        # CLI's global hparam injection otherwise overwrites the nested teacher's config.
        # This model has its own explicit checkpoint loader and needs no CLI instantiator metadata.
        kwargs.setdefault("load_from_checkpoint_support", False)
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--init_checkpoint", type=str, default=None)

    def before_fit(self):
        cfg = self.config.fit
        if cfg.init_checkpoint:
            if cfg.ckpt_path:
                raise ValueError("init_checkpoint and ckpt_path are mutually exclusive")
            print(f"[initialization] {self.model.initialize_from_checkpoint(cfg.init_checkpoint)}", flush=True)


def main():
    with diagnostic_session():
        PredictionCLI(LatentPrediction, CropQADataModule, seed_everything_default=42,
                      save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
