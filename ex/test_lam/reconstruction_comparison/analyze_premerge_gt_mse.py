#!/usr/bin/env python3
"""Measure original-vs-boxed-image MSE in Qwen's final pre-merger space.

The LAM vision tower is frozen, so loading the standalone Qwen vision tower is
equivalent to reading ``gt_feature0`` and ``gt_feature`` from FeatureLAM while
avoiding the trainable LAM encoder/decoder and the language model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_TEST_JSON = REPO_ROOT / "data/lam/v0/test.json"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"
DEFAULT_VISION_MODEL = Path(
    "/data/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
    "ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "premerge_gt_mse_first100.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--vision-model-path", type=Path, default=DEFAULT_VISION_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--focus-indices", type=int, nargs="*", default=[0, 36])
    return parser


def _load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records[:limit]


def _pixel_difference(original: np.ndarray, auxiliary: np.ndarray) -> dict[str, Any]:
    delta = np.max(
        np.abs(original.astype(np.int16) - auxiliary.astype(np.int16)), axis=-1
    )
    changed = delta > 20
    rows, columns = np.where(changed)
    bbox = None
    if columns.size:
        bbox = [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() + 1),
            int(rows.max() + 1),
        ]
    return {
        "threshold": 20,
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "bbox_xyxy": bbox,
        "max_channel_difference": int(delta.max()),
        "mean_max_channel_difference": float(delta.mean()),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    for path in (args.test_json, args.image_root, args.vision_model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.image_size != 512:
        raise ValueError("The current LAM was trained at 512x512")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable, cannot use {args.device}")

    from transformers import AutoProcessor

    from lvr.data.prompt import load_resized_rgb
    from lvr.modules.lam.lam_feature import _load_qwen3_vision_model

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(str(args.vision_model_path)).image_processor
    vision = _load_qwen3_vision_model(str(args.vision_model_path))
    vision.eval().to(device=device, dtype=dtype)

    records = _load_records(args.test_json, args.limit)
    focus_indices = set(args.focus_indices)
    results: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        original = load_resized_rgb(
            record["question_image"], image_root=args.image_root, size=args.image_size
        )
        auxiliary = load_resized_rgb(
            record["auxiliary_image"], image_root=args.image_root, size=args.image_size
        )
        encoded = processor(images=[original, auxiliary], return_tensors="pt")
        grid = encoded["image_grid_thw"].to(device)
        pixel_values = encoded["pixel_values"].to(device=device, dtype=dtype)
        output = vision(hidden_states=pixel_values, grid_thw=grid, return_dict=True)
        premerged = output.last_hidden_state
        split_sizes = grid.prod(dim=-1).tolist()
        if len(split_sizes) != 2 or split_sizes[0] != split_sizes[1]:
            raise ValueError(f"Case {index} has incompatible grids: {grid.tolist()}")
        original_feature, auxiliary_feature = torch.split(premerged, split_sizes)

        original_f = original_feature.float()
        auxiliary_f = auxiliary_feature.float()
        token_mse = (original_f - auxiliary_f).square().mean(dim=-1)
        global_mse = float(token_mse.mean().item())
        reference_energy = float(original_f.square().mean().item())
        sorted_token_mse, _ = token_mse.sort(descending=True)
        total_token_mse = token_mse.sum().clamp_min(torch.finfo(torch.float32).tiny)
        result: dict[str, Any] = {
            "index": index,
            "question_image": str(record["question_image"]),
            "auxiliary_image": str(record["auxiliary_image"]),
            "grid_thw": [int(value) for value in grid[0].tolist()],
            "premerge_shape_per_image": list(original_feature.shape),
            "premerge_mse": global_mse,
            "premerge_cosine_similarity": float(
                F.cosine_similarity(original_f, auxiliary_f, dim=-1).mean().item()
            ),
            "mse_over_original_feature_energy": global_mse / max(reference_energy, 1e-30),
            "token_mse": {
                "mean": global_mse,
                "std": float(token_mse.std(unbiased=False).item()),
                "max": float(token_mse.max().item()),
                "top1_share": float(sorted_token_mse[:1].sum().div(total_token_mse).item()),
                "top4_share": float(sorted_token_mse[:4].sum().div(total_token_mse).item()),
                "top16_share": float(sorted_token_mse[:16].sum().div(total_token_mse).item()),
            },
            "pixel_difference_at_512": _pixel_difference(original, auxiliary),
        }
        if index in focus_indices:
            result["token_mse_values"] = [float(value) for value in token_mse.cpu().tolist()]
        results.append(result)
        print(
            f"[{index + 1:03d}/{len(records):03d}] case={index} "
            f"mse={global_mse:.8f} cosine={result['premerge_cosine_similarity']:.8f}",
            flush=True,
        )

    mse_values = np.asarray([row["premerge_mse"] for row in results], dtype=np.float64)
    descending = sorted(range(len(results)), key=lambda i: results[i]["premerge_mse"], reverse=True)
    for rank, result_index in enumerate(descending, start=1):
        results[result_index]["mse_rank_descending"] = rank
        results[result_index]["mse_percentile"] = float(
            100.0 * np.mean(mse_values <= results[result_index]["premerge_mse"])
        )

    report = {
        "protocol": {
            "description": "Original image vs boxed auxiliary image, final Qwen ViT hidden states before spatial merger",
            "vision_model_path": str(args.vision_model_path),
            "test_json": str(args.test_json),
            "image_size": args.image_size,
            "dtype": args.dtype,
            "num_cases": len(results),
        },
        "premerge_mse_summary": _summary(mse_values),
        "focus_cases": {str(i): results[i] for i in args.focus_indices if i < len(results)},
        "highest_mse_cases": [results[i]["index"] for i in descending[:10]],
        "lowest_mse_cases": [results[i]["index"] for i in descending[-10:]],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
