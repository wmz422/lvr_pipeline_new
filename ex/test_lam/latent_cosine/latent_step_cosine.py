#!/usr/bin/env python3
"""Average adjacent-token cosine similarities for the four LAM latents.

For each question/auxiliary image pair, the LAM encoder produces four latent
tokens.  This script computes

    cos(z[0], z[1]), cos(z[1], z[2]), cos(z[2], z[3])

from the deterministic posterior means (``z_mu``), then averages each of the
three values over the test set.  The reconstruction decoder and the Qwen
language model are not loaded or executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs/lam/v1/chp_save/lam-epoch=001-step=00017000-val_loss=0.00000.ckpt"
)
DEFAULT_TEST_JSON = REPO_ROOT / "data/lam/v0/test.json"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "latent_step_cosine_step17000.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples; by default use the complete test set.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "test_json", "image_root"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.image_size != 512:
        raise ValueError("This checkpoint was trained at 512x512; --image-size must be 512.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")


def _load_image_processor(vision_model_path: str) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(vision_model_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError(f"No image_processor found at {vision_model_path}")
    return image_processor


def _load_lam(checkpoint_path: Path, device: torch.device):
    from lam.model import FeatureLAM

    load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        checkpoint = torch.load(checkpoint_path, mmap=True, **load_kwargs)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise KeyError("Lightning checkpoint has no hyper_parameters dictionary.")
    model = FeatureLAM(**hyper_parameters)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval().to(device=device, dtype=torch.float32)
    return model


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(
            device=device,
            dtype=torch.float32 if key == "pixel_values" else None,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


@torch.inference_mode()
def _posterior_mean_tokens(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return posterior means with shape [batch, num_latent, latent_dim]."""
    lam = model.lam
    features = lam.get_images_features(batch)
    batch_size, num_frames = features.shape[:2]
    if num_frames != 2:
        raise ValueError(f"Expected image pairs with two frames, got {num_frames}")

    projected = lam.encoder_proj(features)
    action_pad = lam.action_prompt.expand(batch_size, num_frames, -1, -1)
    encoded = lam.encoder(torch.cat([action_pad, projected], dim=2))
    encoded = encoded[:, 1:, : lam.num_latent]
    encoded = encoded.reshape(-1, lam.model_dim)
    z_mu, _z_log_variance = torch.chunk(lam.fc(encoded), 2, dim=-1)
    return z_mu.reshape(batch_size, lam.num_latent, lam.latent_dim)


def _adjacent_cosines(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(f"Expected latent tensor [B,N,D], got {tuple(latents.shape)}")
    if latents.shape[1] != 4:
        raise ValueError(
            f"Expected exactly four latent tokens, got {latents.shape[1]}. "
            "This script reports the three adjacent similarities requested."
        )
    return F.cosine_similarity(latents[:, :-1], latents[:, 1:], dim=-1)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)

    print(f"Loading LAM checkpoint from {args.checkpoint}", flush=True)
    model = _load_lam(args.checkpoint, device)
    image_processor = _load_image_processor(model.hparams.vision_model_path)

    from lam.data import LAMCollator, LAMPairDataset

    dataset = LAMPairDataset(args.test_json, limit=args.limit)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=LAMCollator(
            image_root=args.image_root,
            image_processor=image_processor,
            image_size=args.image_size,
        ),
    )

    cosine_sum = torch.zeros(3, dtype=torch.float64)
    sample_count = 0
    for batch_index, batch in enumerate(data_loader, start=1):
        batch = _to_device(batch, device)
        latents = _posterior_mean_tokens(model, batch)
        similarities = _adjacent_cosines(latents)
        cosine_sum += similarities.double().sum(dim=0).cpu()
        sample_count += similarities.shape[0]
        if batch_index % 25 == 0 or sample_count == len(dataset):
            print(f"Processed {sample_count}/{len(dataset)} examples", flush=True)

    means = cosine_sum / sample_count
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_json": str(args.test_json.resolve()),
        "num_examples": sample_count,
        "latent_source": "posterior_mean_z_mu",
        "adjacent_cosine_similarity_mean": {
            "step_0_step_1": float(means[0]),
            "step_1_step_2": float(means[1]),
            "step_2_step_3": float(means[2]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
