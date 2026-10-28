"""Shared benchmark evaluation loop and category statistics."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

from tqdm import tqdm

from lvr.eval.scoring import extract_answer


class BaseBenchmark(ABC):
    """所有 benchmark 评估器的抽象基类。"""

    name: str = ""

    def __init__(self, data_config: dict[str, Any]):
        self.data_config = data_config

    @abstractmethod
    def load_data(self) -> list[dict[str, Any]]:
        """返回样本列表。每条样本含 key：
        'images'（文件路径或 PIL Image 列表）、'question'(str)、'label'(str, ground truth 字母)，
        可选 'category'(str)、'choices'(list)。
        """

    @abstractmethod
    def get_task_instruction(self) -> str:
        """返回拼到每个 question 后面的任务指令。"""

    def evaluate(
        self,
        model: Any,
        infer_fn: Callable[[Any, list, str], str],
        out_dir: str,
    ) -> dict[str, Any]:
        """对单个模型跑评估。

        Args:
            model: 已加载的模型（LatentVLM 或 HF baseline）。
            infer_fn: (model, images, question) -> 生成文本。
            out_dir: 逐样本结果 JSON 的保存目录。

        Returns:
            含 overall accuracy 与各 category 细分的 dict。
        """
        os.makedirs(out_dir, exist_ok=True)
        data = self.load_data()
        if self.data_config.get("limit") is not None:
            data = data[:int(self.data_config["limit"])]
        task_instruction = self.get_task_instruction()

        results = []
        total, correct = 0, 0
        category_stats: dict[str, dict[str, int]] = {}

        for sample in tqdm(data, desc=f"Evaluating {self.name}"):
            output = infer_fn(model, sample["images"], sample["question"] + task_instruction)
            predicted = extract_answer(output, sample.get("choices", None))
            label = sample["label"]
            correct_flag = predicted == label

            record = {
                "id": sample.get("id", ""),
                "prediction": output,
                "predicted": predicted,
                "label": label,
                "correct": correct_flag,
            }
            if "category" in sample:
                record["category"] = sample["category"]
                cat = sample["category"]
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "correct": 0}
                category_stats[cat]["total"] += 1
                if correct_flag:
                    category_stats[cat]["correct"] += 1

            results.append(record)
            total += 1
            if correct_flag:
                correct += 1

        out_file = os.path.join(out_dir, f"{self.name}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        overall_acc = correct / total * 100 if total > 0 else 0.0
        return {
            "benchmark": self.name,
            "total": total,
            "correct": correct,
            "accuracy": overall_acc,
            "categories": {
                cat: {
                    "total": st["total"],
                    "correct": st["correct"],
                    "accuracy": st["correct"] / st["total"] * 100 if st["total"] > 0 else 0.0,
                }
                for cat, st in category_stats.items()
            },
        }
