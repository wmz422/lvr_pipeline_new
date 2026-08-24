"""HR-Bench 4K/8K benchmark loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from lvr.eval.benchmarks.base import BaseBenchmark
from lvr.eval.benchmarks.common import materialize_image_value, option_text


class HRBenchBenchmark(BaseBenchmark):
    """HR-Bench high-resolution multiple-choice VQA.

    The official HF dataset uses config ``hrbench_version_split`` and splits
    ``hrbench_4k`` / ``hrbench_8k``. The local downloader saves each split as a
    standalone ``datasets.Dataset`` under ``external/bench/hr_bench_{4k,8k}``.
    """

    split: str = ""

    def load_data(self) -> list[dict[str, Any]]:
        ds = self._load_dataset()
        cache_dir = self._image_cache_dir()
        samples: list[dict[str, Any]] = []
        limit = self.data_config.get("limit")
        for item in ds:
            if limit is not None and len(samples) >= int(limit):
                break
            choices = [item[key] for key in ["A", "B", "C", "D"] if item.get(key) is not None]
            options = option_text(choices)
            question = str(item["question"]).strip()
            if options:
                question = f"{question}\nOptions:\n{options}"
            samples.append(
                {
                    "id": item.get("index", ""),
                    "images": [
                        materialize_image_value(
                            item["image"],
                            cache_dir=cache_dir,
                            stem=str(item.get("index", len(samples))),
                        )
                    ],
                    "question": question,
                    "label": str(item["answer"]).strip().upper(),
                    "category": str(item.get("category") or item.get("cycle_category") or ""),
                    "choices": [str(choice) for choice in choices],
                }
            )
        return samples

    def get_task_instruction(self) -> str:
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return "\nAnswer with the option's letter from the given choices directly. Do not explain your reasoning — output only the letter."

    def _load_dataset(self):
        dataset_path = self.data_config.get("dataset_path")
        if dataset_path:
            ds = load_from_disk(str(Path(dataset_path).expanduser()))
            if hasattr(ds, "keys"):
                return ds[self.data_config.get("split", self.split)]
            return ds

        dataset_name = self.data_config.get("dataset_name", "parquet")
        split = self.data_config.get("split", self.split)
        data_files = self.data_config.get("data_files") or {
            split: f"hf://datasets/DreamMr/HR-Bench/{'hr_bench_4k.parquet' if split == 'hrbench_4k' else 'hr_bench_8k.parquet'}"
        }
        return load_dataset(dataset_name, data_files=data_files, split=split)

    def _image_cache_dir(self) -> Path:
        if self.data_config.get("image_cache_dir"):
            return Path(self.data_config["image_cache_dir"]).expanduser()
        if self.data_config.get("dataset_path"):
            return Path(self.data_config["dataset_path"]).expanduser() / "images"
        return Path("external/bench") / self.name / "images"


class HRBench4KBenchmark(HRBenchBenchmark):
    name = "hr_bench_4k"
    split = "hrbench_4k"


class HRBench8KBenchmark(HRBenchBenchmark):
    name = "hr_bench_8k"
    split = "hrbench_8k"
