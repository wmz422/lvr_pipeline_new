"""MMVP questions and per-question accuracy, preserving the recorded evaluation protocol."""

from __future__ import annotations

import csv
import os
from typing import Any

from lvr.eval.benchmarks.base import BaseBenchmark
from lvr.eval.scoring import parse_choices_from_text


class MMVPBenchmark(BaseBenchmark):
    """MMVP：2 选 1 视觉问答。数据：本地 CSV + 图片。"""

    name = "mmvp"

    def load_data(self) -> list[dict[str, Any]]:
        csv_file = self.data_config["csv_file"]
        image_dir = self.data_config["image_dir"]

        samples = []
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["Index"])
                label = row["Correct Answer"]
                # label 形如 "(a)" / "(b)" -> "A" / "B"
                if label.startswith("(") and label.endswith(")"):
                    label = label.strip("()").upper()
                question = row["Question"] + "\nOptions:\n" + row["Options"]
                samples.append({
                    "id": idx,
                    "images": [os.path.join(image_dir, f"{idx}.jpg")],
                    "question": question,
                    "label": label,
                    "choices": parse_choices_from_text(row["Options"]),
                })
        return samples

    def get_task_instruction(self) -> str:
        # 与 vstar_boxed 同款：data 段可用 task_instruction 覆盖（如 plain SFT 用 "" 走题目自带的
        # direct 指令）；显式写 "" 也算覆盖（返回空），只有不写这个键才落到默认 CoT scaffold。
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return ""
