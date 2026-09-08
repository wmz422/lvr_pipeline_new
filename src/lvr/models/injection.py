"""Shared latent and visual embedding injection for training and generation."""

from __future__ import annotations

import torch

from lvr.tokens import IGNORE_INDEX


def compute_latents(lam, lam_inputs: dict[str, torch.Tensor], device, dtype: torch.dtype) -> torch.Tensor:
    # 对齐 LAM 输入的设备与精度。
    lam_inputs = {
        key: value.to(device, dtype=dtype) if torch.is_floating_point(value) else value.to(device)
        for key, value in lam_inputs.items()
    }
    outputs = lam(lam_inputs)
    latent = outputs["z_rep"]
    if latent.ndim == 4 and latent.size(1) == 1:
        # LAM 返回 [B, T-1, latent_len, latent_dim]，当前两帧输入下 T-1=1。
        latent = latent[:, 0]
    if latent.ndim != 3:
        raise ValueError(f"Expected LAM latent shape [B, L, D], got {tuple(latent.shape)}.")
    return latent


def inject_latents(
    qwen,
    input_ids: torch.Tensor,
    mapped_latent: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    inputs_embeds = qwen.model.get_input_embeddings()(input_ids)
    expected_latent_tokens = mapped_latent.size(1)
    counts = latent_mask.sum(dim=1)
    if not torch.all(counts == expected_latent_tokens):
        # 强约束每条样本 latent token 数量，避免 silent misalignment。
        raise ValueError(
            "Each sample must contain exactly "
            f"{expected_latent_tokens} latent pad tokens, got {counts.detach().cpu().tolist()}."
        )
    inputs_embeds = inputs_embeds.clone()
    for batch_idx in range(input_ids.size(0)):
        inputs_embeds[batch_idx, latent_mask[batch_idx]] = mapped_latent[batch_idx].to(
            inputs_embeds.dtype
        )
    return inputs_embeds


def mask_latent_labels(labels: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
    labels = labels.clone()
    labels[latent_mask] = IGNORE_INDEX
    return labels


def inject_vision(
    model,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
):
    """把 Qwen3-VL 的 image feature 注入 inputs_embeds。

    因为我们传 inputs_embeds、不走 qwen.forward，这里手动补齐原 forward 的 vision 注入流程。
    返回 (inputs_embeds, image_mask, deepstack_image_embeds)。
    """
    image_mask = None
    deepstack_image_embeds = None
    if pixel_values is not None:
        # vision encoder 在本项目恒为冻结（freezing.set_trainable 从不解冻 visual），其输出 image
        # feature 对所有可训练模块(projector/LM/lm_head)都是常量。用 no_grad 跑它：loss 与对可训练
        # 参数的梯度逐位不变，但省掉整个 vision transformer 的激活——动态分辨率(大图→数千 patch、
        # 注意力 ~quadratic)下这是显存大头(projector-only 仍 OOM 的真因)。masked_scatter 留在
        # no_grad 外：inputs_embeds 在 latent 位带 projector 梯度，vision 位写入的是 detach 的常量。
        with torch.no_grad():
            image_outputs = model.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
            deepstack_image_embeds = image_outputs.deepstack_features
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask = image_mask[..., 0]
    return inputs_embeds, image_mask, deepstack_image_embeds
