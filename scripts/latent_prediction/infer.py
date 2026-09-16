#!/usr/bin/env python3
"""Predict eight latents from an original image/question, then answer with the same B."""

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor

from lvr.data.crop_qa import CropQACollator
from lvr.data.prompt import cap_qwen_pixels
from lvr.models.latent_prediction import LatentPrediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--base-model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--save-latents", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "right"
    cap_qwen_pixels(processor, 1048576)
    collator = CropQACollator(processor, image_root="/", num_latents=8, include_answers=False,
                              include_encoder=False, max_length=65536)
    batch = collator([{"id": "inference", "question_image": str(args.image.resolve()),
                       "boxes": [], "question": args.question, "answer": ""}])
    qa = {key: value.cuda() for key, value in batch["qa"].items()}
    model = LatentPrediction.from_checkpoint(args.checkpoint, "cuda:0", source_checkpoint=args.source_checkpoint,
                                             base_model_path=args.base_model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = model.predict_latents(qa)
        result = model.network.generate_answers(qa, latent, max_new_tokens=args.max_new_tokens)
    if args.save_latents:
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.cpu(), args.save_latents)
    print(processor.tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
