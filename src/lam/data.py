"""Dataset and collate function for feature-space LAM pretraining.

Each JSON record contains a question image and an auxiliary image.  The
auxiliary image is the second frame (in this project it normally contains a
red box).  Both images are resized to the same fixed square before they are
passed to the Qwen vision processor, which preserves the spatial
correspondence required by LAM's temporal attention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lvr.data.prompt import load_resized_rgb


class LAMPairDataset(Dataset):
    """JSON-backed pairs of ``question_image`` and ``auxiliary_image``."""

    def __init__(
        self,
        json_path: str | Path,
        *,
        limit: int | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        with self.json_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {self.json_path}")
        self.records = records[:limit] if limit is not None else records
        if not self.records:
            raise ValueError(f"No records found in {self.json_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        record = self.records[index]
        for key in ("question_image", "auxiliary_image"):
            if key not in record:
                raise KeyError(f"Record {index} in {self.json_path} has no {key!r}")
        return {
            "question_image": str(record["question_image"]),
            "auxiliary_image": str(record["auxiliary_image"]),
        }


def _check_processor_output(
    sample: dict[str, Any],
    *,
    index: int,
) -> None:
    if "pixel_values" not in sample or "image_grid_thw" not in sample:
        keys = ", ".join(sorted(sample.keys()))
        raise KeyError(
            "The Qwen image processor must return pixel_values and "
            f"image_grid_thw; sample {index} returned [{keys}]"
        )

    pixel_values = sample["pixel_values"]
    grid_thw = sample["image_grid_thw"]
    if not torch.is_tensor(pixel_values) or not torch.is_tensor(grid_thw):
        raise TypeError("Qwen processor outputs pixel_values/image_grid_thw must be tensors")
    if grid_thw.shape != (2, 3):
        raise ValueError(f"Expected image_grid_thw shape [2, 3], got {tuple(grid_thw.shape)}")
    if pixel_values.ndim == 2:
        # Transformers 5.x returns all image patches flattened across the
        # image list: [num_images * S, D]. Restore the pair dimension.
        if pixel_values.shape[0] % 2:
            raise ValueError(
                "Flattened pixel_values does not contain an integral number "
                f"of image token blocks: got {tuple(pixel_values.shape)}"
            )
        sample["pixel_values"] = pixel_values.reshape(2, -1, pixel_values.shape[-1])
    elif pixel_values.ndim != 3:
        raise ValueError(
            "Expected one pair's pixel_values to have shape [2, S, D] or "
            f"flattened [2*S, D], got {tuple(pixel_values.shape)}."
        )
    if sample["pixel_values"].shape[0] != 2:
        raise ValueError(f"Expected two images per pair, got {sample['pixel_values'].shape[0]}")


class LAMCollator:
    """Load two images and create a rectangular batch for ``lam_feature.py``.

    ``lam_feature.get_images_features`` currently reshapes the visual output
    as ``[B, 2, S, D]``.  Therefore this first implementation intentionally
    uses a fixed LAM input resolution.  Dynamic Qwen resolution is kept in the
    downstream LVR collator, where it does not alter the LAM training batch.
    """

    def __init__(
        self,
        *,
        image_root: str | Path,
        image_processor: Any,
        image_size: int = 256,
    ) -> None:
        if image_root is None:
            raise ValueError("LAMCollator requires image_root")
        if image_processor is None:
            raise ValueError("LAMCollator requires a Qwen image processor")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.image_size = image_size

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("LAMCollator received an empty batch")

        encoded: list[dict[str, torch.Tensor]] = []
        for index, example in enumerate(examples):
            question_image = load_resized_rgb(
                example["question_image"],
                image_root=self.image_root,
                size=self.image_size,
            )
            auxiliary_image = load_resized_rgb(
                example["auxiliary_image"],
                image_root=self.image_root,
                size=self.image_size,
            )
            sample = self.image_processor(#把图片切成patch，展开成[B,S（patch块的hxw）,D(一个patch里面的像素x通道)]
                images=[question_image, auxiliary_image],
                return_tensors="pt",
            )
            _check_processor_output(sample, index=index)
            encoded.append(
                {
                    "pixel_values": sample["pixel_values"],
                    "image_grid_thw": sample["image_grid_thw"],
                }
            )

        token_counts = {int(item["pixel_values"].shape[1]) for item in encoded}
        if len(token_counts) != 1:
            raise ValueError(
                "The image processor returned different token counts within one batch: "
                f"{sorted(token_counts)}. Use a fixed processor resolution or bucket batches "
                "by token count; padding is not silently allowed because LAM has no attention mask."
            )

        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in encoded], dim=0),
            "image_grid_thw": torch.stack([item["image_grid_thw"] for item in encoded], dim=0),
        }
