#!/usr/bin/env python3
"""Register LVR adapters, then delegate execution to VLMEvalKit's run.py."""

import argparse
import runpy
import sys
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir")
    source.add_argument("--checkpoint")
    source.add_argument("--pretrained", action="store_true", help="Evaluate unchanged base Qwen weights")
    parser.add_argument("--bottleneck-config", default="configs/latent_prediction/warmup.yaml")
    parser.add_argument("--latent-mode", choices=("none", "mu", "sample", "zero", "teacher", "predicted", "shuffled", "teacher_shuffled"), default="mu")
    parser.add_argument("--prediction-checkpoint", action="store_true")
    parser.add_argument("--num-loops", type=int)
    parser.add_argument("--base-model", dest="base_model_path")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--crop-annotations", type=Path)
    parser.add_argument("--qa-crops", action="store_true", help="Send all RGB crops to the native QA vision path")
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system-prompt", help="Explicit system instruction; default preserves the existing protocol")
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--max-pixels", type=int, default=None)
    resolution.add_argument("--processor-default-resolution", action="store_true",
                            help="Skip project original-image resizing; retain the base processor's own limits")
    parser.add_argument("--data-config-file", type=Path)
    parser.add_argument("--vlmeval-root", type=Path, default=Path(__file__).resolve().parents[2] / "third_party/VLMEvalKit")
    args, remaining = parser.parse_known_args()
    if args.num_loops is not None and (args.num_loops < 1 or not args.prediction_checkpoint):
        parser.error("--num-loops requires --prediction-checkpoint and a positive value")
    if args.pretrained and (not args.base_model_path or args.latent_mode != "none" or args.prediction_checkpoint or args.source_checkpoint):
        parser.error("--pretrained requires --base-model, --latent-mode none, and no LVR checkpoint flags")
    if args.qa_crops and (not args.checkpoint or args.latent_mode != "none"):
        parser.error("--qa-crops requires --checkpoint and --latent-mode none")
    if args.max_pixels is not None and args.max_pixels < 1:
        parser.error("--max-pixels must be positive")
    entry_path = args.vlmeval_root.resolve() / "run.py"
    sys.argv = [str(entry_path), *remaining]
    if args.data_config_file:
        sys.argv.extend(["--data-config", args.data_config_file.read_text()])
    # The upstream entry point assigns GPUs before importing torch.
    entry = runpy.run_path(str(entry_path), run_name="vlmeval_entry")

    import vlmeval.dataset
    from vlmeval.config import supported_VLM
    from lvr.eval.vlmeval import BottleneckModel, DatasetSubset, ImageMCQSubset, LVRModel

    vlmeval.dataset.ImageMCQSubset = ImageMCQSubset
    vlmeval.dataset.DatasetSubset = DatasetSubset
    factory = BottleneckModel
    if args.prediction_checkpoint:
        from lvr.eval.latent_prediction import PredictedLatentModel
        factory = PredictedLatentModel
    if args.pretrained:
        from lvr.eval.vlmeval import PretrainedQwenModel
        supported_VLM["Qwen3-VL-4B-Instruct"] = partial(
            PretrainedQwenModel, checkpoint=None, config=args.bottleneck_config,
            latent_mode="none", max_new_tokens=args.max_new_tokens,
            crop_annotations=args.crop_annotations, max_sequence_tokens=args.max_sequence_tokens,
            base_model_path=args.base_model_path,
            system_prompt=args.system_prompt,
            processor_default_resolution=args.processor_default_resolution, max_pixels=args.max_pixels,
        )
    elif args.checkpoint:
        model_name = "Qwen-RGB-Crops" if args.qa_crops else f"Bottleneck-{args.latent_mode}"
        supported_VLM[model_name] = partial(
            factory, checkpoint=args.checkpoint, config=args.bottleneck_config,
            latent_mode=args.latent_mode, max_new_tokens=args.max_new_tokens,
            crop_annotations=args.crop_annotations, max_sequence_tokens=args.max_sequence_tokens,
            qa_crops=args.qa_crops, base_model_path=args.base_model_path, source_checkpoint=args.source_checkpoint,
            processor_default_resolution=args.processor_default_resolution, max_pixels=args.max_pixels,
            **({"system_prompt": args.system_prompt} if args.system_prompt is not None else {}),
            **({"num_loops": args.num_loops} if args.prediction_checkpoint else {}),
        )
    else:
        supported_VLM["LVR-SFT-v3-4000"] = partial(
            LVRModel, model_dir=args.model_dir,
            max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels,
        )
    entry["main"]()


if __name__ == "__main__":
    main()
