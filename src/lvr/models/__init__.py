"""模型层：组装 (latent_vlm) + 建模 (builder) + 注入 (injection) + loss / freezing。

LatentVLM 是 stage-agnostic 的瘦 LightningModule；建模细节在同目录子模块。
（搬自旧 sft/src/model.py 拆分而来。generate / test 在 M3 接入。）
"""

from lvr.models.latent_vlm import LatentVLM
from lvr.models.plain_vlm import PlainVLM

__all__ = ["LatentVLM", "PlainVLM"]
