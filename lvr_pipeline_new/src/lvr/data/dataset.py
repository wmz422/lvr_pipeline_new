"""Read alignment and SFT records. Auxiliary images provide LAM targets; Qwen consumes question images."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset
from lvr.data.records import read_records


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
        self.items = read_records(self.path, limit=limit)
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
