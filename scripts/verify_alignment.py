#!/usr/bin/env python3
"""Compare known-latent alignment inference using existing local image pairs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir")
    source.add_argument("--checkpoint")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--compare-bundle-parameters", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    import torch
    import yaml
    from lvr.eval.inference import known_latent_generate

    torch.set_num_threads(8)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    if args.model_dir:
        from lvr.bundle import load_bundle
        model, processor = load_bundle(args.model_dir, args.device)
    else:
        if args.source_config is None:
            parser.error("--source-config is required with --checkpoint")
        from lvr.eval.runner import load_sft_model
        config = yaml.safe_load(args.source_config.read_text())
        model, processor = load_sft_model(config.get("fit", config)["model"], args.checkpoint, args.device)
    if args.samples.is_dir():
        index = json.loads((args.samples / "index.json").read_text())
        rows = []
        for shard in index["shards"]:
            rows.extend(json.loads((args.samples / shard["file"]).read_text()))
            if len(rows) >= args.limit:
                break
    else:
        rows = json.loads(args.samples.read_text())
    predictions = []
    for row in rows[:args.limit]:
        images = [str(args.data_root / row[name]) for name in ("question_image", "auxiliary_image")]
        with torch.inference_mode():
            prediction = known_latent_generate(
                model, processor, images, row["question"],
                {"max_new_tokens": args.max_new_tokens, "do_sample": False, "temperature": 0.0},
                image_size=None, system="You are a helpful assistant.", lam_image_size=512,
            )
        predictions.append({"question_image": row["question_image"], "prediction": prediction})
    result = {"mode": "known_latent_alignment", "max_new_tokens": args.max_new_tokens,
              "predictions": predictions, "tied_lm_head": model.qwen.lm_head.weight is model.qwen.model.language_model.embed_tokens.weight}
    result["qwen_rotary_buffer_dtypes"] = {name: str(value.dtype) for name, value in model.qwen.named_buffers() if "rotary" in name}
    if args.compare_bundle_parameters:
        from safetensors.torch import load_file
        native = torch.nn.Module.state_dict(model)
        seen = set()
        mismatches = []
        for shard in sorted(args.compare_bundle_parameters.glob("model*.safetensors")):
            exported = load_file(shard, device=args.device)
            for name, value in exported.items():
                seen.add(name)
                if name not in native or native[name].dtype != value.dtype or not torch.equal(native[name], value):
                    mismatches.append(name)
            del exported
        result["all_parameters_exact"] = seen == set(native) and not mismatches
        result["compared_tensors"] = len(seen)
        result["parameter_mismatches"] = mismatches
    if args.reference:
        reference = json.loads(args.reference.read_text())["predictions"]
        result["all_exact"] = bool(predictions) and predictions == reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.compare_bundle_parameters and not result["all_parameters_exact"]:
        raise RuntimeError("Bundle parameters differ from the native loaded model")
    if args.reference and not result["all_exact"]:
        raise RuntimeError("Exported alignment predictions differ from the source checkpoint")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
