"""MME-RealWorld-Lite benchmark loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from lvr.eval.benchmarks.base import BaseBenchmark
from lvr.eval.benchmarks.common import materialize_image_value, option_text, strip_option_labels


class MMERealWorldLiteBenchmark(BaseBenchmark):
    """MME-RealWorld-Lite multiple-choice VQA.

    Uses the compact ``yifanzhang114/MME-RealWorld-lite-lmms-eval`` parquet
    release by default. It contains embedded image data and 1,919 questions.
    """

    name = "mme_realworld_lite"

    def load_data(self) -> list[dict[str, Any]]:
        ds = self._load_dataset()
        samples: list[dict[str, Any]] = []
        category_field = self.data_config.get("category_field", "category")
        image_field = self.data_config.get("image_field")
        cache_dir = self._image_cache_dir()
        limit = self.data_config.get("limit")
        for item in ds:
            if limit is not None and len(samples) >= int(limit):
                break
            options = [str(option) for option in (item.get("options") or item.get("multi-choice options") or [])]
            option_block = option_text(options)
            question = str(item["question"]).strip()
            if option_block:
                question = f"{question}\nOptions:\n{option_block}"
            category = str(item.get(category_field) or item.get("category") or "")
            l2_category = item.get("l2-category") or item.get("l2_category")
            if self.data_config.get("use_l2_category", False) and l2_category:
                category = f"{category}/{l2_category}" if category else str(l2_category)

            samples.append(
                {
                    "id": item.get("index", ""),
                    "images": [
                        materialize_image_value(
                            item[self._image_field(item, image_field)],
                            cache_dir=cache_dir,
                            stem=str(item.get("index", len(samples))),
                            image_root=self.data_config.get("image_root"),
                        )
                    ],
                    "question": question,
                    "label": str(item["answer"]).strip().upper(),
                    "category": category,
                    "choices": strip_option_labels(options),
                }
            )
        return samples

    def get_task_instruction(self) -> str:
        override = self.data_config.get("task_instruction")
        if override is not None:
            return override
        return "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E)."

    def _load_dataset(self):
        dataset_path = self.data_config.get("dataset_path")
        if dataset_path:
            ds = load_from_disk(str(Path(dataset_path).expanduser()))
            if hasattr(ds, "keys"):
                return ds[self.data_config.get("split", "train")]
            return ds

        dataset_name = self.data_config.get(
            "dataset_name", "yifanzhang114/MME-RealWorld-lite-lmms-eval"
        )
        split = self.data_config.get("split", "train")
        return load_dataset(dataset_name, split=split)

    def _image_field(self, item: dict[str, Any], configured: str | None) -> str:
        if configured:
            return configured
        if "image" in item:
            return "image"
        if "bytes" in item:
            return "bytes"
        raise KeyError("MME-RealWorld-Lite sample has no image/bytes field.")

    def _image_cache_dir(self) -> Path:
        if self.data_config.get("image_cache_dir"):
            return Path(self.data_config["image_cache_dir"]).expanduser()
        if self.data_config.get("dataset_path"):
            return Path(self.data_config["dataset_path"]).expanduser() / "images"
        return Path("external/bench") / self.name / "images"
