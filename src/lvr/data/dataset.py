"""读取已整理好的 alignment / SFT json，stage-agnostic。

公共字段：question_image、auxiliary_image、question。auxiliary_image 不直接喂给 Qwen，
只给 LAM 用来在线生成 latent。target 字段按 stage 取：
- SFT：`answer`（默认）。
- Alignment：`observation`，并额外带 `shuffle_auxiliary_image`（下一条的 auxiliary，错配消融用）。

（搬自旧 sft/src/dataset.py SftDataset 与 alignment/src/dataset.py AlignmentDataset，逻辑不变。）
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        limit: int | None = None,
        *,
        target_key: str = "answer",
        include_shuffle_auxiliary: bool = False,
    ) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.items = data[:limit] if limit is not None else data
        self.target_key = target_key
        self.include_shuffle_auxiliary = include_shuffle_auxiliary

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, str]:
        item = self.items[index]
        sample = {
            "question_image": item["question_image"],
            "auxiliary_image": item["auxiliary_image"],
            "question": item["question"],
            self.target_key: item[self.target_key],
        }
        if self.include_shuffle_auxiliary:
            # 错配 auxiliary：取下一条的 auxiliary_image（与旧 AlignmentDataset 一致）。
            shifted = self.items[(index + 1) % len(self.items)]
            sample["shuffle_auxiliary_image"] = shifted["auxiliary_image"]
        return sample
