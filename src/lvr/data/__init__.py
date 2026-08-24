"""数据层：datamodule + dataset + collator（搬自旧 sft/src/dataset.py，逻辑不变）。"""

from lvr.data.collator import AlignmentCollator, LatentCollator, PlainSFTCollator
from lvr.data.datamodule import LatentDataModule
from lvr.data.dataset import LatentDataset
from lvr.data.prompt import build_generation_prompt, load_resized_rgb, strip_auto_think_prompt

__all__ = [
    "LatentDataset",
    "LatentCollator",
    "AlignmentCollator",
    "PlainSFTCollator",
    "LatentDataModule",
    "build_generation_prompt",
    "load_resized_rgb",
    "strip_auto_think_prompt",
]
