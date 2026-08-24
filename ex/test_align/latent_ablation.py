#!/usr/bin/env python3
"""Evaluate whether an alignment projector uses the LAM latent information.

The three conditions have identical question images, prompts, latent placeholder
tokens and observation labels.  They differ *only* in the vectors injected at
the four latent-pad positions:

* ``correct``: LAM latent from the sample's paired auxiliary image.
* ``shuffled``: LAM latent from the next sample's auxiliary image, as defined
  by ``LatentDataset(include_shuffle_auxiliary=True)``.
* ``zero``: zero vectors at all latent positions (LAM is not evaluated).

The primary result is token-weighted observation CE.  A materially lower CE for
``correct`` than for both controls is evidence that the trained projector maps
LAM latents into a representation that Qwen uses for this task.

This program is evaluation-only: it calls ``model.eval()``, uses inference mode,
and never writes checkpoints or modifies the training run.  It is deliberately
not invoked by this file; run it only when a GPU is available, for example:

    CUDA_VISIBLE_DEVICES=0 python ex/test_align/latent_ablation.py \
      --checkpoint runs/align/align_full_lr1e4/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RUN_DIR = REPO_ROOT / "runs/align/align_full_lr1e4"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints/last.ckpt"
DEFAULT_TEST_JSON = REPO_ROOT / "data/align/v0/test.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/latent_ablation.json"
CONDITIONS = ("correct", "shuffled", "zero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate the first N validation samples (default: all).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("config", "checkpoint", "test_json", "image_root"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Root run configs dumped by LightningCLI nest the subcommand under ``fit``;
    # logger configs place the same fields at the top level.  Support both.
    if isinstance(config, dict) and isinstance(config.get("fit"), dict):
        config = config["fit"]
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Expected a Lightning CLI config with a model section: {path}")
    if not isinstance(config.get("data"), dict):
        raise ValueError(f"Expected a Lightning CLI config with a data section: {path}")
    return config


def load_model(model_config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    """Build base Qwen/LAM and overlay the projector state from a .ckpt file."""
    from lvr.models import LatentVLM

    valid_keys = set(inspect.signature(LatentVLM.__init__).parameters)
    init_kwargs = {key: value for key, value in model_config.items() if key in valid_keys}
    # Loading via init_checkpoint_path does not support Lightning's .ckpt format.
    init_kwargs["init_checkpoint_path"] = None
    model = LatentVLM(**init_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Lightning checkpoint has no state_dict: {checkpoint_path}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    missing_projector = [
        key for key in incompatible.missing_keys if key.startswith("latent_projector.")
    ]
    if missing_projector:
        raise RuntimeError(f"Checkpoint is missing projector parameters: {missing_projector}")

    model.to(device)
    model.eval()
    return model


def build_loader(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> DataLoader:
    from transformers import AutoProcessor

    from lvr.data.collator import AlignmentCollator
    from lvr.data.dataset import LatentDataset
    from lvr.data.prompt import cap_qwen_pixels
    from lvr.tokens import register_latent_tokens

    model_config = config["model"]
    data_config = config["data"]
    processor = AutoProcessor.from_pretrained(data_config["processor_name_or_path"])
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if data_config.get("add_latent_special_tokens", True):
        register_latent_tokens(tokenizer)
    if data_config.get("qwen_dynamic_resolution", False):
        cap_qwen_pixels(processor, data_config.get("qwen_max_pixels"))

    dataset = LatentDataset(
        args.test_json,
        limit=args.limit,
        target_key="observation",
        include_shuffle_auxiliary=True,
    )
    collator = AlignmentCollator(
        processor=processor,
        lam_image_processor=processor.image_processor,
        image_root=args.image_root,
        max_length=data_config.get("max_length"),
        latent_len=data_config.get("latent_len", model_config.get("lam_num_latent", 4)),
        lam_image_size=data_config.get("lam_image_size", 512),
        system_prompt=data_config.get("system_prompt", ""),
        qwen_dynamic_resolution=data_config.get("qwen_dynamic_resolution", False),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )


def move_qwen_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move only tensors consumed by the Qwen forward path; keep metadata on CPU."""
    keys = (
        "input_ids", "labels", "attention_mask", "pixel_values",
        "image_grid_thw", "mm_token_type_ids",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys if key in batch}


@torch.inference_mode()
def condition_ce(model: Any, qwen_batch: dict[str, Any], lam_inputs: dict[str, torch.Tensor] | None,
                 condition: str) -> tuple[float, int]:
    """Return summed CE and supervised-token count for one ablation condition."""
    from lvr.models import injection

    input_ids = qwen_batch["input_ids"]
    labels = qwen_batch["labels"]
    latent_mask = input_ids.eq(model.latent_pad_token_id)
    token_count = int((labels[:, 1:] != -100).sum().item())
    if token_count == 0:
        raise RuntimeError("Batch contains no supervised observation tokens.")

    # Training used Lightning bf16-mixed precision.  Match that here so the
    # bf16 Qwen/LAM weights receive compatible image and hidden-state dtypes.
    with torch.autocast(
        device_type=input_ids.device.type,
        dtype=model._compute_dtype(),
        enabled=input_ids.device.type == "cuda",
    ):
        if condition == "zero":
            mapped_latent = torch.zeros(
                input_ids.size(0),
                int(model.hparams.lam_num_latent),
                model.latent_projector.out_features,
                device=input_ids.device,
                dtype=model.latent_projector.weight.dtype,
            )
        else:
            if lam_inputs is None:
                raise ValueError(f"{condition} condition requires LAM inputs.")
            latent = injection.compute_latents(
                model.lam, lam_inputs, model.device, model._compute_dtype())
            mapped_latent = model.latent_projector(latent)

        inputs_embeds = injection.inject_latents(model.qwen, input_ids, mapped_latent, latent_mask)
        masked_labels = injection.mask_latent_labels(labels, latent_mask)
        outputs = model._qwen_forward_from_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=masked_labels,
            attention_mask=qwen_batch.get("attention_mask"),
            pixel_values=qwen_batch.get("pixel_values"),
            image_grid_thw=qwen_batch.get("image_grid_thw"),
            mm_token_type_ids=qwen_batch.get("mm_token_type_ids"),
            latent_mask=latent_mask,
        )
    return float(outputs["ce_loss"].item()) * token_count, token_count


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    config = load_config(args.config)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = load_model(config["model"], args.checkpoint, device)
    data_loader = build_loader(config, args, device)
    totals = {condition: {"ce_sum": 0.0, "tokens": 0} for condition in args.conditions}

    for batch_index, batch in enumerate(data_loader, start=1):
        qwen_batch = move_qwen_inputs(batch, device)
        condition_inputs = {
            "correct": batch.get("lam_inputs"),
            "shuffled": batch.get("shuffle_lam_inputs"),
            "zero": None,
        }
        for condition in args.conditions:
            ce_sum, tokens = condition_ce(
                model, qwen_batch, condition_inputs[condition], condition,
            )
            totals[condition]["ce_sum"] += ce_sum
            totals[condition]["tokens"] += tokens
        if batch_index % 25 == 0 or batch_index == len(data_loader):
            done = min(batch_index * args.batch_size, len(data_loader.dataset))
            print(f"Processed {done}/{len(data_loader.dataset)} examples", flush=True)

    metrics = {
        condition: {
            "observation_ce": totals[condition]["ce_sum"] / totals[condition]["tokens"],
            "supervised_tokens": totals[condition]["tokens"],
        }
        for condition in args.conditions
    }
    correct_ce = metrics.get("correct", {}).get("observation_ce")
    deltas_vs_correct = {
        condition: values["observation_ce"] - correct_ce
        for condition, values in metrics.items()
        if correct_ce is not None and condition != "correct"
    }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "test_json": str(args.test_json.resolve()),
        "num_examples": len(data_loader.dataset),
        "conditions": list(args.conditions),
        "metrics": metrics,
        "ce_increase_vs_correct": deltas_vs_correct,
        "interpretation": (
            "Positive CE increases for both shuffled and zero mean that correct LAM latents "
            "improve observation prediction beyond the unchanged question image and prompt."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
