"""LVR Pipeline — Latent Visual Reasoning 统一训练 / 推理 / 评估 pipeline.

整套方法分三阶段：LAM 预训练 → Alignment → SFT。本包统一 Alignment + SFT 的训练 /
推理 / 评估代码，LAM 预训练保持外部、只消费其 checkpoint。

详见仓库根目录 README.md（项目介绍）与 CLAUDE.md（重构计划与决策记录）。
"""

__version__ = "0.0.1"
