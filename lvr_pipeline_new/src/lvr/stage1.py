"""Stage-one visual bottleneck training with the shared Lightning run layout."""

from lvr.cli import LvrCLI
from lvr.data.crop_qa import CropQADataModule
from lvr.models.visual_bottleneck import VisualBottleneck
from lvr.training.diagnostics import diagnostic_session


class StageOneCLI(LvrCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("--init_checkpoint", type=str, default=None,
                            help="Initialize all model weights for a new training phase; reset optimizer and schedule")
        parser.link_arguments("model.encoder_uses_question", "data.encoder_uses_question")
        parser.link_arguments("model.encoder_type", "data.encoder_type")
        parser.link_arguments("model.encoder_includes_original", "data.encoder_includes_original")
        parser.link_arguments("model.description_image_mode", "data.description_image_mode")
        parser.link_arguments("model.description_weight", "data.include_descriptions", compute_fn=bool)
        parser.link_arguments("model.reconstruction_weight", "data.include_reconstruction", compute_fn=bool)

    def before_fit(self):
        cfg = self.config.fit
        if cfg.init_checkpoint:
            if cfg.ckpt_path:
                raise ValueError("init_checkpoint and ckpt_path are mutually exclusive")
            report = self.model.initialize_from_checkpoint(cfg.init_checkpoint)
            print(f"[initialization] {report}", flush=True)


def main():
    with diagnostic_session():
        StageOneCLI(VisualBottleneck, CropQADataModule, seed_everything_default=42,
                    save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
