"""BLINK validation subsets with one to four images per question."""

from __future__ import annotations

import string
from typing import Any

from datasets import load_from_disk

from lvr.eval.benchmarks.base import BaseBenchmark


class BLINKBenchmark(BaseBenchmark):
    """BLINK：多选视觉推理（1-4 图）。数据：HF `BLINK-Benchmark/BLINK` val splits，图片为内嵌 PIL。"""

    name = "blink"

    def load_data(self) -> list[dict[str, Any]]:
        configs = self.data_config["configs"]
        samples = []

        for config in configs:
            ds = load_from_disk(f"{self.data_config['dataset_name']}/{config}")
            val_ds = ds["val"]
            for item in val_ds:
                if self.data_config.get("limit") is not None and len(samples) >= int(self.data_config["limit"]):
                    return samples
                answer = item["answer"]
                ans = answer[1].upper() if len(answer) > 1 else answer[0].upper()

                images = []
                for k in ["image_1", "image_2", "image_3", "image_4"]:
                    if k in item and item[k] is not None:
                        images.append(item[k])

                choices = item["choices"] or []
                letters = string.ascii_uppercase
                option_string = ""
                for letter, choice in zip(letters, choices):
                    option_string += f"{letter}. {choice}\n"

                question = item["question"] + "\nOptions:\n" + option_string
                samples.append({
                    "id": item["idx"],
                    "images": images,  # list of PIL Images
                    "question": question,
                    "label": ans,
                    "category": config,
                    "choices": choices,  # for answer extraction
                })
        return samples

    def get_task_instruction(self) -> str:
        # 与 vstar_boxed 同款：data 段可用 task_instruction 覆盖（如 plain SFT 用 "" 走题目自带的
        # direct 指令）；显式写 "" 也算覆盖（返回空），只有不写这个键才落到默认 CoT scaffold。
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return ""
