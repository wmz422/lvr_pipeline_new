"""benchmark 推理后端。

- **SFT** (`sft_generate`)：复用训练侧的数据处理单一真源，再走 `LatentVLM.generate()`
  手写 latent 解码循环。
- **baseline** (`baseline_generate`)：vanilla Qwen3-VL，走 HF `model.generate()`，但输入构造
  与 SFT 完全一致（path B prompt + cv2 256² RGB），保证 benchmark 对照同口径。
"""

from __future__ import annotations

from typing import Any

import torch

from lvr.data.prompt import build_generation_prompt, build_lam_inputs, build_latent_block, load_resized_rgb

SFT_IMAGE_SIZE = 256  # 与训练 LAM image size 一致


def _to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def _to_rgb(img: Any):
    """path / PIL / ndarray → RGB PIL，**不 resize**（Qwen processor 自做动态分辨率）。"""
    from PIL import Image

    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    import numpy as np

    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    raise TypeError(type(img))


def build_eval_inputs(
    processor: Any,
    images: list[Any],
    question: str,
    device: Any,
    image_size: int | None = SFT_IMAGE_SIZE,
    system: str = "",
) -> dict[str, Any]:
    """Benchmark 输入单一真源：path B prompt + 图像预处理。

    SFT 与 baseline 都走这里，避免 prompt 或图像预处理口径漂移。
    - image_size=int：先 cv2 resize 到 size² RGB（与训练 LAM 预处理一致，SFT 必须）。
    - image_size=None：原生 RGB 直接交给 Qwen processor 做动态分辨率（no-resize / 动态分辨率口径）。
    - system：与训练 collator 一致的 system prompt（如 "You are a helpful assistant."），空=无 system。
    """
    if image_size is None:
        proc_images = [_to_rgb(img) for img in images]
    else:
        proc_images = [load_resized_rgb(img, size=image_size) for img in images]
    prompt = build_generation_prompt(processor, question, system=system, num_images=len(images))
    inputs = processor(
        text=[prompt],
        images=proc_images,
        return_tensors="pt",
        padding=True,
    )
    return _to_device(inputs, device)


def sft_generate(
    model: Any,
    processor: Any,
    images: list[Any],
    question: str,
    gen_cfg: dict[str, Any],
    image_size: int | None = SFT_IMAGE_SIZE,
    system: str = "",
) -> str:
    """用 LatentVLM.generate()（手写 latent 解码）跑推理。

    prompt 与图像预处理与训练 collator 同源：
    - prompt = build_generation_prompt（path B，含 `<|im_start|>assistant\\n`，空 think 块已去），system 同训练
    - 图像 = load_resized_rgb（image_size=256 与 LAM 同口径；=None 走 Qwen 动态分辨率，需与训练一致）
    """
    device = next(model.parameters()).device
    batch = build_eval_inputs(processor, images, question, device, image_size, system)

    # generate() 无 generation_* key 时回退用 input_ids/attention_mask 作 prompt（见 generation.generate）。
    generated_ids = model.generate(
        batch,
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=False,
        temperature=1.0,  # do_sample=False 下中性（argmax）
        force_latent_end=gen_cfg.get("force_latent_end", False),
    )
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return output[0] if isinstance(output, list) else output


def known_latent_generate(
    model: Any,
    processor: Any,
    images: list[Any],
    question: str,
    gen_cfg: dict[str, Any],
    image_size: int | None = SFT_IMAGE_SIZE,
    system: str = "",
    lam_image_size: int = SFT_IMAGE_SIZE,
) -> str:
    """Known-latent diagnostic generation.

    `images` must be `[question_image, auxiliary_image]`. Qwen receives only the
    question image, while LAM receives question+auxiliary and injects the latent
    before text-only autoregressive answer generation. ``lam_image_size`` must
    match the LAM input resolution used during the evaluated SFT training run.
    """
    if len(images) < 2:
        raise ValueError("known_latent_generate requires images=[question_image, auxiliary_image].")

    device = next(model.parameters()).device
    question_image, auxiliary_image = images[0], images[1]
    qwen_images = (
        [_to_rgb(question_image)]
        if image_size is None
        else [load_resized_rgb(question_image, size=image_size)]
    )
    latent_block = build_latent_block(
        int(model.hparams.lam_num_latent),
        latent_pad_token=model.hparams.latent_pad_token,
    )
    prompt = build_generation_prompt(processor, question, system=system, num_images=1) + latent_block
    batch = processor(
        text=[prompt],
        images=qwen_images,
        return_tensors="pt",
        padding=True,
    )
    batch = _to_device(batch, device)
    batch["lam_inputs"] = build_lam_inputs(
        [{"question_image": question_image, "auxiliary_image": auxiliary_image}],
        lam_image_processor=processor.image_processor,
        image_root=None,
        lam_image_size=lam_image_size,
    )

    generated_ids = model.generate_align(
        batch,
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        temperature=gen_cfg.get("temperature", 1.0) or 1.0,
        do_sample=gen_cfg.get("do_sample", False),
    )
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return output[0] if isinstance(output, list) else output


def baseline_generate(
    model: Any,
    processor: Any,
    images: list[Any],
    question: str,
    gen_cfg: dict[str, Any],
    image_size: int | None = SFT_IMAGE_SIZE,
    system: str = "",
) -> str:
    """vanilla Qwen3-VL：HF model.generate()。

    image_size=256 时与 SFT 同口径对照；image_size=None 时走 Qwen 原生动态分辨率（no-resize baseline）。
    """
    device = next(model.parameters()).device
    inputs = build_eval_inputs(processor, images, question, device, image_size, system)

    max_new_tokens = gen_cfg.get("max_new_tokens", 256)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    return output[0] if isinstance(output, list) else output
