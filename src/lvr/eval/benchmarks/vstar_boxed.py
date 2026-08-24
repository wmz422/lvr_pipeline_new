"""Prepared VSTAR boxed-single-bbox diagnostic benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lvr.eval.benchmarks.base import BaseBenchmark


class VSTARBoxedSingleBBoxBenchmark(BaseBenchmark):
    """Prepared local VSTAR split with one bbox per sample.

    `image_mode` controls what the existing benchmark runner sends to Qwen:
    - `qwen_two_image`: original + boxed auxiliary (current eval-compatible).
    - `question_only`: original image only.
    - `boxed_only`: boxed image only.

    The prepared JSON keeps `question_image` and `auxiliary_image` as paired
    paths, so a future known-latent runner can feed the boxed image to LAM
    instead of Qwen.
    """

    name = "vstar_boxed_single"

    def load_data(self) -> list[dict[str, Any]]:
        json_file = Path(self.data_config["json_file"]).expanduser()
        payload = self._load_json(json_file)
        image_root = Path(
            self.data_config.get("image_root") or payload.get("image_root") or json_file.parent
        ).expanduser()
        categories = set(self.data_config.get("categories") or [])
        image_mode = self.data_config.get("image_mode", "qwen_two_image")
        if image_mode not in {"qwen_two_image", "question_only", "boxed_only"}:
            raise ValueError(
                "vstar_boxed_single image_mode must be one of "
                "{'qwen_two_image', 'question_only', 'boxed_only'}."
            )

        samples: list[dict[str, Any]] = []
        for item in payload.get("items", payload if isinstance(payload, list) else []):
            if categories and item.get("category") not in categories:
                continue
            question_image = self._resolve_path(image_root, item["question_image"])
            auxiliary_image = self._resolve_path(image_root, item["auxiliary_image"])
            if image_mode == "qwen_two_image":
                images = [question_image, auxiliary_image]
            elif image_mode == "boxed_only":
                images = [auxiliary_image]
            else:
                images = [question_image]

            samples.append(
                {
                    "id": item.get("id", ""),
                    "images": images,
                    "question_image": question_image,
                    "auxiliary_image": auxiliary_image,
                    "question": item["question"],
                    "label": item["label"],
                    "category": item.get("category", ""),
                    "choices": item.get("choices", []),
                    "bbox": item.get("bbox"),
                    "target_object": item.get("target_object", []),
                }
            )
        return samples

    def get_task_instruction(self) -> str:
        # 与 vstar_boxed 同款：data 段可用 task_instruction 覆盖（如 plain SFT 用 "" 走题目自带的
        # direct 指令）；显式写 "" 也算覆盖（返回空），只有不写这个键才落到默认 CoT scaffold。
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return ""

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _resolve_path(self, image_root: Path, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = image_root / path
        return str(path)
