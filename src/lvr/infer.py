"""Run image-question inference using an exported local model bundle."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True, nargs="+")
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    from lvr.bundle import load_bundle
    from lvr.eval.inference import sft_generate
    torch.set_num_threads(8)
    model, processor = load_bundle(args.model_dir, args.device, args.max_pixels)
    with torch.inference_mode():
        output = sft_generate(model, processor, args.image, args.question,
                              {"max_new_tokens": args.max_new_tokens, "force_latent_end": True},
                              image_size=None, system="You are a helpful assistant.")
    result = {"question": args.question, "prediction": output}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
