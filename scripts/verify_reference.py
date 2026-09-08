#!/usr/bin/env python3
"""Compare the export's smoke predictions with a trusted source checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--export-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    import yaml
    from lvr.eval.runner import load_sft_model
    from lvr.eval.benchmarks import VSTARBenchmark
    from lvr.eval.inference import sft_generate

    torch.set_num_threads(8)
    config = yaml.safe_load(args.source_config.read_text())
    model_config = config.get("fit", config)["model"]
    model, processor = load_sft_model(model_config, args.checkpoint, args.device)
    benchmark = VSTARBenchmark({"test_jsonl": str(args.bench_root / "vstar_bench/test_questions.jsonl"),
                                "image_dir": str(args.bench_root / "vstar_bench")})
    samples = benchmark.load_data()[:2]
    outputs = []
    for sample in samples:
        with torch.inference_mode():
            prediction = sft_generate(model, processor, sample["images"], sample["question"],
                                      {"max_new_tokens": 64, "force_latent_end": True},
                                      image_size=None, system="You are a helpful assistant.")
        outputs.append({"id": sample["id"], "prediction": prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference_predictions": outputs}, indent=2) + "\n")
    if not args.export_results.is_file():
        raise FileNotFoundError(f"Reference saved, but exported predictions are missing: {args.export_results}")
    if args.export_results.is_file():
        exported = json.loads(args.export_results.read_text())
        matches = [{"id": a["id"], "exact": a["id"] == b["id"] and a["prediction"] == b["prediction"]}
                   for a, b in zip(outputs, exported)]
        result = {"reference_predictions": outputs, "matches": matches,
                  "all_exact": len(matches) == len(outputs) == len(exported) and all(m["exact"] for m in matches)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not result["all_exact"]:
            raise RuntimeError("Reference and export predictions differ")


if __name__ == "__main__":
    main()
