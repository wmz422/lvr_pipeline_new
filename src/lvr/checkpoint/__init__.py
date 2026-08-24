"""checkpoint 统一 save / load（搬自旧 sft/src/model.py 的 _init_checkpoint，逻辑不变）。"""

from lvr.checkpoint.io import load_init_checkpoint

__all__ = ["load_init_checkpoint"]
