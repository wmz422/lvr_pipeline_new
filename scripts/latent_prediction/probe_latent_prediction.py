#!/usr/bin/env python3
"""Check full-size gradient boundaries and two complete optimizer steps before DDP."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoProcessor
import yaml

from lvr.data.crop_qa import CropQACollator, CropQADataset
from lvr.data.prompt import cap_qwen_pixels
from lvr.models.latent_prediction import LatentPrediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("warmup", "joint"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    dataset = CropQADataset("data/sft/train", "data/align/train", caption_path=data["caption_path"])
    examples = sorted(dataset.items, key=lambda row: (len(row["boxes"]), len(row["question"]) +
                      len(row["answer"]) + len(row["description"])), reverse=True)[:args.batch_size]
    processor = AutoProcessor.from_pretrained(data["processor_name_or_path"])
    processor.tokenizer.padding_side = "right"
    cap_qwen_pixels(processor, data["original_pixels"])
    collator = CropQACollator(processor, data["image_root"], num_latents=8, include_encoder=True,
                             encoder_includes_original=True, include_descriptions=True,
                             description_image_mode="none", original_pixels=data["original_pixels"],
                             crop_pixels=data["crop_pixels"], max_length=data["max_length"])
    batch = collator(examples)
    for part in ("encoder", "qa", "description"):
        batch[part] = {k: v.cuda() for k, v in batch[part].items()}
    model = LatentPrediction(**{**cfg["model"], "prediction_phase": args.phase}).cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0, weight_decay=0, foreach=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    records = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, step=960)
        out["loss"].backward()
        for parameter in (model.prediction_queries, model.prediction_head[1].weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
        assert not out["teacher"].requires_grad
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if args.phase == "joint":
            assert model.qa.lm_head.weight.grad is not None and model.qa.lm_head.weight.grad.norm() > 0
        else:
            assert all(p.grad is None for p in model.qa.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        records.append({**{k: v.item() for k, v in out.items() if v.ndim == 0}, "grad_norm": norm.item()})
        print(json.dumps(records[-1]), flush=True)
    assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in optimizer.state.values())
    report = {"status": "passed", "phase": args.phase, "batch_size": args.batch_size,
              "steps": records, "seconds": time.monotonic() - start, "optimizer_states_fp32": True,
              "gpu": torch.cuda.get_device_name(), "crop_counts": batch["crop_counts"].tolist(),
              "shapes": {part: list(batch[part]["input_ids"].shape) for part in ("encoder", "qa", "description")},
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "ddp_gradient_buffer_gib": sum(p.numel() * p.element_size() for p in parameters) / 2**30,
              "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
