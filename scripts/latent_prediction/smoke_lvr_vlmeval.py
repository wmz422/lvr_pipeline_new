#!/usr/bin/env python3
"""Check published-bundle generation on a few examples before full evaluation."""

import argparse
import json
import time
from pathlib import Path

from vlmeval.dataset import build_dataset

from lvr.eval.vlmeval import BottleneckModel, DatasetSubset, ImageMCQSubset, LVRModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir")
    source.add_argument("--checkpoint")
    parser.add_argument("--bottleneck-config", default="configs/latent_prediction/warmup.yaml")
    parser.add_argument("--latent-mode", choices=("none", "mu", "sample", "zero", "teacher", "predicted", "shuffled"), default="mu")
    parser.add_argument("--prediction-checkpoint", action="store_true")
    parser.add_argument("--base-model", dest="base_model_path")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--crop-annotations", type=Path)
    parser.add_argument("--qa-crops", action="store_true")
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--data", nargs="+", default=["VStarBench", "MMVP", "BLINK", "HRBench4K", "HRBench8K", "MME-RealWorld-Lite"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if args.qa_crops and (not args.checkpoint or args.latent_mode != "none"):
        parser.error("--qa-crops requires --checkpoint and --latent-mode none")
    config = json.loads(args.data_config.read_text())
    factory = BottleneckModel
    if args.prediction_checkpoint:
        from lvr.eval.latent_prediction import PredictedLatentModel
        factory = PredictedLatentModel
    model = (factory(args.checkpoint, args.bottleneck_config, args.latent_mode,
                            crop_annotations=args.crop_annotations, max_sequence_tokens=args.max_sequence_tokens,
                            qa_crops=args.qa_crops, base_model_path=args.base_model_path,
                            source_checkpoint=args.source_checkpoint)
             if args.checkpoint else LVRModel(args.model_dir))
    annotations = json.loads(args.crop_annotations.read_text())["samples"] if args.crop_annotations else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for name in args.data:
            if name in config:
                kwargs = {k: v for k, v in config[name].items() if k != "class"}
                factory = {"ImageMCQSubset": ImageMCQSubset, "DatasetSubset": DatasetSubset}[config[name]["class"]]
                dataset = factory(**kwargs)
            else:
                dataset = build_dataset(name)
            positions = list(range(min(args.limit, len(dataset.data))))
            if annotations and positions:
                counts = {row["index"]: sum(len(item["boxes"]) for item in row["images"])
                          for row in annotations.values() if row["benchmark"] == name}
                positions[-1] = max(enumerate(dataset.data["index"]), key=lambda pair: counts[str(pair[1])])[0]
            elif args.checkpoint and name == "BLINK" and positions:
                # Exercise the largest image count, not only the first single-image task.
                positions[-1] = max(enumerate(dataset.data["image_path"]),
                                    key=lambda pair: len(pair[1]) if isinstance(pair[1], list) else 1)[0]
            for _, row in dataset.data.iloc[positions].iterrows():
                start = time.monotonic()
                result = model.generate(dataset.build_prompt(row), dataset=name)
                if not result["prediction"]:
                    raise RuntimeError(f"Empty generation: {name}/{row['index']}")
                record = {
                    "benchmark": name, "index": str(row["index"]), "answer": row["answer"],
                    **result, "seconds": round(time.monotonic() - start, 3),
                }
                line = json.dumps(record, ensure_ascii=False)
                out.write(line + "\n")
                out.flush()
                print(line, flush=True)


if __name__ == "__main__":
    main()
