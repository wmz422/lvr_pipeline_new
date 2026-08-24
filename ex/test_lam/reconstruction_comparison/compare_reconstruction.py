#!/usr/bin/env python3
"""Compare real and LAM-reconstructed auxiliary-image features with Qwen3-VL.

For every (question image, auxiliary image) pair this script runs two otherwise
identical Qwen3-VL prompts:

1. real question feature + real auxiliary feature;
2. real question feature + LAM-reconstructed auxiliary feature.

LAM reconstructs the final, pre-merger vision hidden state, not RGB pixels.
Both branches therefore pass those hidden states through Qwen's vision merger
and inject the resulting image embeddings into the language model.  DeepStack
features are disabled in both branches so the reconstruction branch cannot
leak the real auxiliary image through intermediate vision-layer features.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs/lam/v1/checkpoints/lam-epoch=001-step=00020000-val_loss=0.00000.ckpt"
)
DEFAULT_TEST_JSON = REPO_ROOT / "data/lam/v0/test.json"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_step20000.jsonl"
DEFAULT_QWEN_PATH = Path(
    "/data/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
    "ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
DEFAULT_PROMPT = (
    "Compare the two images. In the second image, red rectangular outlines may have been "
    "drawn to highlight objects. Identify every red rectangular outline that is newly "
    "drawn in the second image and is not present in the first image. For each new "
    "outline, provide a numbered description of the object it highlights. If no new red "
    "rectangular outline is visible, say: No new red outline."
)
PROTOCOL = "merged_final_vision_features_without_deepstack"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to evaluate; use -1 for all remaining records.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--qwen-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output instead of resuming by skipping indices already present.",
    )
    return parser


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "test_json", "image_root", "qwen_model_path"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.image_size != 512:
        raise ValueError("This checkpoint was trained at 512x512; --image-size must remain 512.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be positive or -1 for all records.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} is not a JSON object.")
        missing = {"question_image", "auxiliary_image"}.difference(record)
        if missing:
            raise KeyError(f"Record {index} is missing keys: {sorted(missing)}")
    return records


def _select_indices(total: int, start: int, limit: int) -> range:
    stop = total if limit == -1 else min(total, start + limit)
    if start >= total:
        raise IndexError(f"--start-index {start} is outside dataset of length {total}.")
    return range(start, stop)


def _completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return completed


def _reshape_pair_pixel_values(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Return Qwen processor patches as [1, 2, patches_per_image, patch_dim]."""
    if image_grid_thw.shape != (2, 3):
        raise ValueError(f"Expected exactly two image grids [2,3], got {tuple(image_grid_thw.shape)}")
    if pixel_values.ndim == 3 and pixel_values.shape[0] == 2:
        pair = pixel_values
    elif pixel_values.ndim == 2:
        if pixel_values.shape[0] % 2:
            raise ValueError(f"Odd flattened patch count: {pixel_values.shape[0]}")
        pair = pixel_values.reshape(2, -1, pixel_values.shape[-1])
    else:
        raise ValueError(
            "Expected pixel_values shaped [2,S,D] or [2*S,D], got "
            f"{tuple(pixel_values.shape)}"
        )
    if not torch.equal(image_grid_thw[0], image_grid_thw[1]):
        raise ValueError(
            "The two resized images produced different grids; LAM requires equal token geometry: "
            f"{image_grid_thw.tolist()}"
        )
    return pair.unsqueeze(0)


def _merged_split_sizes(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> list[int]:
    sizes = image_grid_thw.prod(dim=-1) // (spatial_merge_size**2)
    result = [int(value) for value in sizes.tolist()]
    if any(value <= 0 for value in result):
        raise ValueError(f"Invalid merged split sizes: {result}")
    return result


def _merge_feature_pair(
    merger: torch.nn.Module,
    feature_pair: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, ...]:
    """Merge [2,S,D] final vision states and split them back into two images."""
    if feature_pair.ndim != 3 or feature_pair.shape[0] != 2:
        raise ValueError(f"Expected feature pair [2,S,D], got {tuple(feature_pair.shape)}")
    merged = merger(feature_pair.reshape(-1, feature_pair.shape[-1]))
    split_sizes = _merged_split_sizes(image_grid_thw, spatial_merge_size)
    if sum(split_sizes) != merged.shape[0]:
        raise ValueError(
            f"Merged token mismatch: grid expects {sum(split_sizes)}, merger returned {merged.shape[0]}"
        )
    return tuple(torch.split(merged, split_sizes, dim=0))


def _feature_metrics(reference: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Metric tensors have different shapes: {tuple(reference.shape)} vs "
            f"{tuple(prediction.shape)}"
        )
    reference_f = reference.float()
    prediction_f = prediction.float()
    return {
        "mse": float(F.mse_loss(prediction_f, reference_f).item()),
        "cosine_similarity": float(
            F.cosine_similarity(prediction_f, reference_f, dim=-1).mean().item()
        ),
    }


def _load_lam(checkpoint_path: Path, device: torch.device):
    from lam.model import FeatureLAM

    load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    # mmap avoids eagerly reading the complete 6.9 GiB checkpoint into anonymous RAM.
    try:
        checkpoint = torch.load(checkpoint_path, mmap=True, **load_kwargs)
    except TypeError:  # Compatibility with older torch versions.
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise KeyError("Lightning checkpoint has no hyper_parameters dictionary.")
    model = FeatureLAM(**hyper_parameters)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval().to(device=device, dtype=torch.float32)
    return model


def _load_qwen(model_path: Path, device: torch.device, dtype: torch.dtype):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(model_path))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return model, processor


@contextmanager
def _precomputed_image_features(
    qwen_model: Any,
    features: Sequence[torch.Tensor],
) -> Iterator[None]:
    """Temporarily replace vision encoding with supplied merged image embeddings."""
    if len(features) != 2:
        raise ValueError(f"Expected two image feature tensors, got {len(features)}")
    model = qwen_model.model
    original = model.get_image_features
    frozen_features = tuple(feature.detach() for feature in features)

    def fake_get_image_features(
        _self: Any,
        _pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        if image_grid_thw is None or image_grid_thw.shape[0] != len(frozen_features):
            raise ValueError("Qwen requested image features with an unexpected image grid.")
        return SimpleNamespace(
            last_hidden_state=None,
            pooler_output=frozen_features,
            deepstack_features=None,
        )

    model.get_image_features = types.MethodType(fake_get_image_features, model)
    try:
        yield
    finally:
        model.get_image_features = original


@torch.inference_mode()
def _generate_with_features(
    qwen_model: Any,
    processor: Any,
    inputs: dict[str, torch.Tensor],
    merged_features: Sequence[torch.Tensor],
    max_new_tokens: int,
) -> str:
    prompt_length = int(inputs["input_ids"].shape[1])
    # rope_deltas is cached on the model in this transformers implementation.
    # Reset it because every comparison is an independent generation request.
    qwen_model.model.rope_deltas = None
    with _precomputed_image_features(qwen_model, merged_features):
        generated = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


@torch.inference_mode()
def _evaluate_record(
    *,
    index: int,
    record: dict[str, Any],
    lam_model: Any,
    qwen_model: Any,
    processor: Any,
    image_root: Path,
    image_size: int,
    prompt: str,
    system: str,
    max_new_tokens: int,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    from lvr.data.prompt import build_generation_prompt, load_resized_rgb

    question_path = Path(record["question_image"])
    auxiliary_path = Path(record["auxiliary_image"])
    question_rgb = load_resized_rgb(question_path, image_root=image_root, size=image_size)
    auxiliary_rgb = load_resized_rgb(auxiliary_path, image_root=image_root, size=image_size)

    prompt_text = build_generation_prompt(processor, prompt, system=system, num_images=2)
    qwen_inputs = processor(
        text=[prompt_text],
        images=[question_rgb, auxiliary_rgb],
        return_tensors="pt",
        padding=True,
    )
    qwen_inputs = _to_device(dict(qwen_inputs), device)
    if "pixel_values" not in qwen_inputs or "image_grid_thw" not in qwen_inputs:
        raise KeyError("Qwen processor did not return pixel_values and image_grid_thw.")

    pair_pixels = _reshape_pair_pixel_values(
        qwen_inputs["pixel_values"], qwen_inputs["image_grid_thw"]
    )
    lam_batch = {
        "pixel_values": pair_pixels.to(device=device, dtype=torch.float32),
        "image_grid_thw": qwen_inputs["image_grid_thw"].unsqueeze(0),
    }
    lam_outputs = lam_model(lam_batch)
    original_feature = lam_outputs["gt_feature0"][0]
    real_auxiliary_feature = lam_outputs["gt_feature"][0, 0]
    reconstructed_auxiliary_feature = lam_outputs["feature"][0, 0]

    real_pair = torch.stack([original_feature, real_auxiliary_feature], dim=0)
    reconstructed_pair = torch.stack(
        [original_feature, reconstructed_auxiliary_feature], dim=0
    )
    vision = lam_model.lam.visual_encoder
    real_merged = _merge_feature_pair(
        vision.merger,
        real_pair,
        qwen_inputs["image_grid_thw"],
        vision.spatial_merge_size,
    )
    reconstructed_merged = _merge_feature_pair(
        vision.merger,
        reconstructed_pair,
        qwen_inputs["image_grid_thw"],
        vision.spatial_merge_size,
    )

    real_answer = _generate_with_features(
        qwen_model, processor, qwen_inputs, real_merged, max_new_tokens
    )
    reconstructed_answer = _generate_with_features(
        qwen_model, processor, qwen_inputs, reconstructed_merged, max_new_tokens
    )

    return {
        "index": index,
        "question_image": str(question_path),
        "auxiliary_image": str(auxiliary_path),
        "checkpoint": str(checkpoint_path),
        "image_size": image_size,
        "feature_protocol": PROTOCOL,
        "prompt": prompt,
        "real_feature_answer": real_answer,
        "reconstructed_feature_answer": reconstructed_answer,
        "premerge_metrics": _feature_metrics(
            real_auxiliary_feature, reconstructed_auxiliary_feature
        ),
        "merged_metrics": _feature_metrics(real_merged[1], reconstructed_merged[1]),
        "latent_statistics": {
            "mu_mean": float(lam_outputs["z_mu"].float().mean().item()),
            "mu_std": float(lam_outputs["z_mu"].float().std().item()),
            "log_variance_mean": float(lam_outputs["z_var"].float().mean().item()),
            "log_variance_std": float(lam_outputs["z_var"].float().std().item()),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    records = _load_records(args.test_json)
    indices = _select_indices(len(records), args.start_index, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
    completed = _completed_indices(args.output)
    pending = [index for index in indices if index not in completed]
    if not pending:
        print(f"Nothing to do: all selected indices already exist in {args.output}")
        return

    print(f"Loading Qwen3-VL from {args.qwen_model_path}", flush=True)
    qwen_model, processor = _load_qwen(
        args.qwen_model_path, device, _torch_dtype(args.qwen_dtype)
    )
    print(f"Loading LAM checkpoint from {args.checkpoint}", flush=True)
    lam_model = _load_lam(args.checkpoint, device)

    with args.output.open("a", encoding="utf-8") as output_handle:
        for ordinal, index in enumerate(pending, start=1):
            print(f"[{ordinal}/{len(pending)}] evaluating record {index}", flush=True)
            result = _evaluate_record(
                index=index,
                record=records[index],
                lam_model=lam_model,
                qwen_model=qwen_model,
                processor=processor,
                image_root=args.image_root,
                image_size=args.image_size,
                prompt=args.prompt,
                system=args.system,
                max_new_tokens=args.max_new_tokens,
                device=device,
                checkpoint_path=args.checkpoint,
            )
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()

    print(f"Finished {len(pending)} records. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
