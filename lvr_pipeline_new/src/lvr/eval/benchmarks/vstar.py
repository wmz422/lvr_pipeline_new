"""VSTAR multiple-choice questions with optional auxiliary images for diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from lvr.eval.benchmarks.base import BaseBenchmark
from lvr.eval.scoring import parse_choices_from_text


class VSTARBenchmark(BaseBenchmark):
    """VSTAR：2 选 1 视觉问答。数据：HF `craigwu/vstar_bench` test split，图片在本地目录。
    分类：direct_attributes、relative_position。

    注：数据集 `text` 字段已含问题、选项和「Answer with the option's letter ... directly.」，
    这里只追加 <answer> 标签指令。
    """

    name = "vstar"

    def load_data(self) -> list[dict[str, Any]]:
        test_ds = self._load_test_data()
        image_dir = self.data_config["image_dir"]
        auxiliary_image_dir = self.data_config.get("auxiliary_image_dir")

        samples = []
        for item in test_ds:
            img_path = os.path.join(image_dir, item["image"])
            images = [img_path]
            if auxiliary_image_dir:
                auxiliary_path = Path(auxiliary_image_dir) / Path(item["image"]).with_suffix(".png")
                if not auxiliary_path.is_file():
                    raise FileNotFoundError(
                        f"VSTAR auxiliary image is missing for {item['image']}: {auxiliary_path}"
                    )
                # known_latent_generate consumes [question_image, auxiliary_image].
                images.append(str(auxiliary_path))
            samples.append({
                "id": item["question_id"],
                "images": images,
                "question": item["text"],
                "label": item["label"],
                "category": item["category"],
                "choices": parse_choices_from_text(item["text"]),
            })
        return samples

    def _load_test_data(self) -> list[dict[str, Any]]:
        """Load the official test split, optionally from a checked-in JSONL copy.

        ``test_jsonl`` keeps a known-latent diagnostic independent of Hub access
        and guarantees that its 191 samples are the same local VSTAR set used by
        the regular benchmark loader.
        """
        test_jsonl = self.data_config.get("test_jsonl")
        if test_jsonl:
            import json

            path = Path(test_jsonl).expanduser()
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]

        ds = load_dataset(self.data_config["dataset_name"])
        return list(ds["test"])

    def get_task_instruction(self) -> str:
        # 与 vstar_boxed 同款：data 段可用 task_instruction 覆盖（如 plain SFT 用 "" 走题目自带的
        # direct 指令）；显式写 "" 也算覆盖（返回空），只有不写这个键才落到默认 CoT scaffold。
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return ""
