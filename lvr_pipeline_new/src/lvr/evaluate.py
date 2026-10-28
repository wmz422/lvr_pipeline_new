"""Evaluate an exported LVR bundle against locally prepared benchmarks."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bench-root", type=Path, default=Path("benchmark/bench"))
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--benchmarks", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="First N examples per benchmark; omit for the full evaluation")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    import torch
    import yaml
    from lvr.bundle import load_bundle
    from lvr.eval.benchmarks import BENCHMARKS
    from lvr.eval.inference import sft_generate

    torch.set_num_threads(8)
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation = dict(config["generation"])
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    model, processor = load_bundle(args.model_dir, args.device, config.get("qwen_max_pixels"))
    def infer(model, images, question):
        with torch.inference_mode():
            return sft_generate(model, processor, images, question, generation,
                                image_size=config.get("sft_image_size"),
                                system=config.get("system_prompt", ""))
    results = {}
    for name in args.benchmarks or config["benchmarks"]:
        if name not in config["data"] or name not in BENCHMARKS:
            parser.error(f"Unknown benchmark: {name}")
        data = dict(config["data"][name])
        for key in ("test_jsonl", "image_dir", "csv_file", "dataset_name", "dataset_path"):
            if key in data:
                relative = data[key].removeprefix("benchmark/bench/")
                data[key] = str(args.bench_root / relative)
        if args.limit is not None:
            data["limit"] = args.limit
        data["image_cache_dir"] = str(args.output_dir / "image_cache" / name)
        results[name] = BENCHMARKS[name](data).evaluate(model, infer, str(args.output_dir / "predictions"))
        (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    report = {"scope": "smoke" if args.limit else "full", "limit_per_benchmark": args.limit,
              "generation": generation, "config": config, "results": results}
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
