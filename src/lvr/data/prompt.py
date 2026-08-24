"""Prompt 构造 + 图像预处理的**单一真源**。

训练 collator、test_step 的生成路径、以及 M6 的 benchmark 评估，全部共用这里的实现，
消灭旧 `evaluation/inference.py` 自带的第二套 prompt 构造（§2.D 的漂移源）。

两件事统一在此：
1. **生成 prompt（path B）**：`build_generation_prompt` 产出与**训练前缀逐字一致**的 prompt
   —— 即 `…<|im_end|>\n<|im_start|>assistant\n`（含 assistant 起始标记，空 think 块已去）。
   做法是「插 sentinel → 渲染整段对话模板 → 去 think/observation 标签 → 按 sentinel 切，取前缀」，
   因此它**就是**模型训练时 answer/latent 之前真正看到的那段，天然不会和训练漂移
   Qwen3 Thinking 模板自动插入的空 `<think>` scaffold 由 `strip_auto_think_prompt` 统一清理。
2. **图像预处理**：`load_resized_rgb` —— cv2 读图 BGR→RGB、resize 到 256×256 INTER_AREA，
   与 LAM/训练完全一致；同时支持传入 PIL.Image（benchmark 加载器常给 PIL）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch

from lvr.tokens import LATENT_END_TOKEN, LATENT_PAD_TOKEN, LATENT_START_TOKEN

# assistant 段占位符：插进 assistant turn，用来定位 latent block / answer 应插入的位置，
# 也用来把生成 prompt（前缀）从整段模板里切出来。
ASSISTANT_SENTINEL = "__ALIGNMENT_OBSERVATION_TARGET__"


def strip_auto_think_prompt(text: str) -> str:
    """去掉 Qwen3 Thinking chat template 在生成起点自动加的空 think scaffold。

    只处理 prompt 阶段的空标记：
    - `add_generation_prompt=True` 产生的尾部 `<think>\n`
    - assistant sentinel path 产生的 `<think>\n\n</think>\n\n`

    不删除包含真实内容的 `<think>...</think>`，避免误改模型输出或 alignment 数据。
    """
    text = text.replace("<think>\n\n</think>\n\n", "")
    text = text.replace("<think>\n</think>\n", "")
    return re.sub(r"<think>\n\Z", "", text)


def strip_scaffold_tags(text: str) -> str:
    """去掉 observation 标签和空 think prompt scaffold（与训练 collator 行为一致）。"""
    text = text.replace("<observation>", "").replace("</observation>", "")
    return strip_auto_think_prompt(text)


def _messages(
    question: str, system: str, *, num_images: int, with_assistant_sentinel: bool
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": "placeholder"}  # tokenize=False 下不读图，仅占位
        for _ in range(num_images)
    ]
    user_content.append({"type": "text", "text": question})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]
    if with_assistant_sentinel:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_SENTINEL}]}
        )
    return messages


def build_generation_prompt(
    processor: Any, question: str, *, system: str = "", num_images: int = 1
) -> str:
    """生成 prompt（path B）= 训练时 sentinel 之前的前缀（含 `<|im_start|>assistant\\n`）。

    collator 的 generation 路径与 benchmark 评估都调它，保证「喂给 generate() 的 prompt」
    与「模型训练时的生成起点」逐字一致。`num_images` 控制 user 段的图像占位数
    （训练恒为单图=默认 1；多图 benchmark 如 BLINK 传实际张数）。
    """
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError("processor 必须提供 apply_chat_template。")
    template_text = processor.apply_chat_template(
        _messages(question, system, num_images=num_images, with_assistant_sentinel=True),
        tokenize=False,
    )
    template_text = strip_scaffold_tags(template_text)
    if ASSISTANT_SENTINEL not in template_text:
        raise ValueError("chat template 未保留 assistant sentinel。")
    prefix, _suffix = template_text.split(ASSISTANT_SENTINEL, 1)
    return prefix


def load_resized_rgb(
    image: str | Path,
    *,
    image_root: str | Path | None = None,
    size: int | None = 256,
) -> np.ndarray:
    """读图为 RGB array。`size=int` → resize 成 size×size；`size=None` → 保留原生分辨率。

    - str/Path：仅使用 cv2 读取（BGR）→ 转 RGB；相对路径按 image_root 拼接。
      OpenCV 不可用时在模块导入阶段直接报错，不使用 PIL fallback。
    - **size=None（动态分辨率）**：不 resize，原生 RGB 直接交给 Qwen processor 自做 smart-resize。
      LAM 路仍传固定 size（256），两路解耦——见 collator 的 qwen_dynamic_resolution。
    """
    path = Path(image)
    if image_root is not None and not path.is_absolute():
        path = Path(image_root) / path
    img_bgr = cv.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # Qwen processor 和 LAM image processor 都按 RGB 图像语义处理。
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    if size is None:
        return img_rgb
    return cv.resize(img_rgb, (size, size), interpolation=cv.INTER_AREA)


def cap_qwen_pixels(processor: Any, max_pixels: int | None) -> None:
    """给 Qwen image_processor 的动态分辨率设像素上限（cap），训练与 eval 共用、口径一致。

    Qwen2VLImageProcessor 用 `size.longest_edge` 当作 max_pixels（默认 ~16M），不设上限时
    单张大图会产出上万 vision token → 撑爆 max_length → 截断破坏 image token 计数 → 崩。
    这里把上限压到 `max_pixels`：超过的图等比缩到该上限，其余保持原生动态分辨率。
    `max_pixels=None` 时不动（用 Qwen 默认）。
    """
    if max_pixels is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    size = getattr(image_processor, "size", None)
    if size is not None and hasattr(size, "longest_edge"):
        image_processor.size.longest_edge = max_pixels  # 本版本走这条
    if hasattr(image_processor, "max_pixels"):  # 旧版 API 兜底
        image_processor.max_pixels = max_pixels


def build_latent_block(
    latent_len: int,
    *,
    latent_start_token: str = LATENT_START_TOKEN,
    latent_pad_token: str = LATENT_PAD_TOKEN,
    latent_end_token: str = LATENT_END_TOKEN,
) -> str:
    """latent 占位块：`<start>` + `<pad>`*latent_len + `<end>` + 换行。

    SFT 与 alignment collator 共用：SFT 里它是「待生成目标」，alignment 里它在 prompt（latent 已知）。
    真正 forward 时只替换 pad token 的 embedding（见 injection.inject_latents）。
    """
    return latent_start_token + latent_pad_token * latent_len + latent_end_token + "\n"


def build_lam_inputs(
    examples: list[dict[str, Any]],
    *,
    lam_image_processor: Any,
    image_root: str | Path | None,
    lam_image_size: int,
    auxiliary_key: str = "auxiliary_image",
) -> dict[str, torch.Tensor]:
    """把每条样本的 (question image, auxiliary image) 两帧整理成 LAM 输入（在线生成 latent）。

    SFT 与 alignment collator 共用。`auxiliary_key` 可换成 shuffle_auxiliary_image 做消融。
    """
    per_sample_inputs = []
    for example in examples:
        # LAM 训练时就是两帧输入：第一帧 question image，第二帧 auxiliary image。
        question_image = load_resized_rgb(example["question_image"], image_root=image_root, size=lam_image_size)
        auxiliary_image = load_resized_rgb(example[auxiliary_key], image_root=image_root, size=lam_image_size)
        per_sample_inputs.append(
            lam_image_processor(images=[question_image, auxiliary_image], return_tensors="pt")
        )

    lam_inputs: dict[str, torch.Tensor] = {}
    for key in per_sample_inputs[0]:
        values = [sample[key] for sample in per_sample_inputs]
        if torch.is_tensor(values[0]):
            lam_inputs[key] = torch.stack(values, dim=0)
    return lam_inputs
