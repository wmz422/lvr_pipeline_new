"""VSTAR benchmark。搬自旧 sft/evaluation/benchmarks/vstar.py，逻辑不变。"""

from __future__ import annotations

import os
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
        ds = load_dataset(self.data_config["dataset_name"])
        test_ds = ds["test"]
        image_dir = self.data_config["image_dir"]

        samples = []
        for item in test_ds:
            img_path = os.path.join(image_dir, item["image"])
            samples.append({
                "id": item["question_id"],
                "images": [img_path],
                "question": item["text"],
                "label": item["label"],
                "category": item["category"],
                "choices": parse_choices_from_text(item["text"]),
            })
        return samples

    def get_task_instruction(self) -> str:
        # 与 vstar_boxed 同款：data 段可用 task_instruction 覆盖（如 plain SFT 用 "" 走题目自带的
        # direct 指令）；显式写 "" 也算覆盖（返回空），只有不写这个键才落到默认 CoT scaffold。
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return ""
