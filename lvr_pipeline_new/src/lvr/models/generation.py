"""Autoregressive decoding with a fixed-length latent phase and known-latent diagnostic generation."""

from __future__ import annotations

from typing import Any

import torch

from lvr.models import injection
from lvr.tokens import LATENT_END_TOKEN, LATENT_START_TOKEN


@torch.no_grad()
def generate(
    model,
    batch: dict[str, Any],
    max_new_tokens: int = 256,
    latent_len: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool = True,
    force_latent_end: bool = False,
) -> torch.Tensor:
    """force_latent_end：latent self-rollout 结束后，强制喂入 `</abs_vis_token>`（teacher-force
    训练时的 latent 块分隔符），而非让模型自由采样。诊断用——self-rollout 把状态带偏后，本应
    生成的 `</abs_vis_token>` 退化成乱码；强制喂回真 end token 可把状态踩回 token embedding 流形、
    还原训练结构 `[start][latent×L][end]\\n[answer]`。默认 False（保持原 generate 行为，ADR-2）。"""
    if model.qwen is None or model.tokenizer is None:
        raise RuntimeError("Model not initialized for generation.")

    if latent_len is None:
        latent_len = model.hparams.lam_num_latent

    latent_start_id = model.tokenizer.convert_tokens_to_ids(LATENT_START_TOKEN)
    latent_end_id = model.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
    eos_id = model.tokenizer.eos_token_id
    if eos_id is None:
        eos_id = model.tokenizer.pad_token_id

    input_ids = batch.get("generation_input_ids")
    if input_ids is None:
        input_ids = batch["input_ids"]
    attention_mask = batch.get("generation_attention_mask")
    if attention_mask is None:
        attention_mask = batch.get("attention_mask")
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    mm_token_type_ids = batch.get("generation_mm_token_type_ids")
    if mm_token_type_ids is None:
        mm_token_type_ids = batch.get("mm_token_type_ids")

    batch_size = input_ids.size(0)
    device = input_ids.device

    # ---------- Prefill: image features + initial forward ----------
    inputs_embeds = model.qwen.model.get_input_embeddings()(input_ids)
    inputs_embeds, image_mask, deepstack_image_embeds = injection.inject_vision(
        model.qwen.model, input_ids, inputs_embeds, pixel_values, image_grid_thw,
    )

    position_ids = model.qwen.model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    prefill_out = model.qwen.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_image_embeds,
        use_cache=True,
    )
    past_key_values = prefill_out.past_key_values
    hidden = prefill_out[0][:, -1]  # (B, hidden_size)

    # ---------- Decode loop ----------
    generated = []  # list of (B, 1) tensors
    in_latent = torch.zeros(batch_size, dtype=torch.bool, device=device)
    latent_step = torch.zeros(batch_size, dtype=torch.long, device=device)
    # 刚退出 latent 模式、下一步需强制 </abs_vis_token> 的样本（force_latent_end 用）。
    force_end_pending = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):  # 本质：kv cache 与 next embedding 不停喂给 forward
        non_latent_mask = ~in_latent
        # Snapshot before the text branch changes in_latent: the start-token
        # embedding must not consume one of the continuous latent steps.
        latent_mask = in_latent.clone()

        # 默认：hidden 直接作为下一步（latent 样本用）
        next_embeds = hidden.unsqueeze(1).clone()

        # 文本样本：采样 token 并嵌入，覆盖 next_embeds
        if non_latent_mask.any():
            logits = model.qwen.lm_head(hidden[non_latent_mask].unsqueeze(1))
            logits = logits[:, -1] / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                mask = cum_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                logits[mask.scatter(1, sorted_idx, mask)] = float("-inf")

            if do_sample:
                probs = logits.softmax(-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_token = logits.argmax(-1)

            # force_latent_end：刚退出 latent 的样本，这一步用 </abs_vis_token> 覆盖采样结果。
            if force_latent_end:
                pend_sub = force_end_pending[non_latent_mask]
                if pend_sub.any():
                    next_token = next_token.clone()
                    next_token[pend_sub] = latent_end_id
                force_end_pending[non_latent_mask] = False

            # 注：batch>1 时若部分样本已 eos、部分未 eos 会继续生成（已知 bug，本次不修）。
            if (next_token == eos_id).all():
                break
            generated.append(next_token.unsqueeze(-1))

            text_embeds = model.qwen.model.get_input_embeddings()(next_token.unsqueeze(-1))
            next_embeds[non_latent_mask] = text_embeds.to(next_embeds.dtype)

            # 检测 <abs_vis_token> → 下轮进入 latent 模式
            in_latent[non_latent_mask] = (next_token == latent_start_id)
            latent_step[non_latent_mask] = 0

        # latent 步数累计，达 latent_len 退出
        if latent_mask.any():
            latent_step[latent_mask] += 1
            exiting = latent_mask & (latent_step >= latent_len)
            in_latent[latent_step >= latent_len] = False  # 下次碰到 <abs_vis_token> 再归零
            if force_latent_end:
                force_end_pending |= exiting  # 下一非 latent 步强制 </abs_vis_token>

        position_ids = position_ids[..., -1:] + 1  # batch=1 恒正确：末尾恒为文本 token，其位置即全局 max，逐 +1 == max+1（三维一致）；batch>1+padding 才需 rope_deltas 写法
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1,
            )

        out = model.qwen.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=next_embeds,
            past_key_values=past_key_values,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        hidden = out[0][:, -1]

    if not generated:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)
    return torch.cat(generated, dim=1)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool,
) -> torch.Tensor:
    """文本分支的下一个 token 采样（与 generate() 的文本分支同源，抽出复用）。"""
    logits = logits / temperature
    if top_k is not None:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits[logits < topk_vals[:, -1:]] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cum_probs = sorted_logits.softmax(-1).cumsum(-1)
        mask = cum_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        logits[mask.scatter(1, sorted_idx, mask)] = float("-inf")
    if do_sample:
        probs = logits.softmax(-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    return logits.argmax(-1)


@torch.no_grad()
def generate_align(
    model,
    batch: dict[str, Any],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool = False,
    latent_condition: str = "correct",
) -> torch.Tensor:
    """Alignment 推理模式 generate —— latent 作为**已知**在 prefill 注入，纯文本自回归生成。

    与 SFT generate() 的本质差别（CLAUDE.md M7 留待项 `generate_observation` 的接入）：
    - latent **不是模型生成的**：从 LAM 在线算 (compute_latents) → projector 映射 →
      注入到 prompt 里 latent_pad 位（injection.inject_latents，与训练 forward 同源）。
      ``latent_condition`` 可选择正确配对、错配辅助图或全零 latent，用于消融。
    - 解码循环**只走文本分支**：绝不进入 latent self-rollout（latent 已知，不需要生成）。
    要求 batch 含 `generation_input_ids`（prefix + latent_block，AlignmentCollator 产出）与
    `lam_inputs`（question+auxiliary 图像，benchmark 数据没有 auxiliary，故只能跑 test 集）。
    仍 batch=1 语义（ADR-2，eos/3D position_id 按 generate() 原样保留）。
    """
    if (
        model.qwen is None
        or model.tokenizer is None
        or model.lam is None
        or model.latent_projector is None
    ):
        raise RuntimeError("Model not initialized for alignment generation.")

    eos_id = model.tokenizer.eos_token_id
    if eos_id is None:
        eos_id = model.tokenizer.pad_token_id

    input_ids = batch.get("generation_input_ids")
    if input_ids is None:
        input_ids = batch["input_ids"]
    attention_mask = batch.get("generation_attention_mask")
    if attention_mask is None:
        attention_mask = batch.get("attention_mask")
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    mm_token_type_ids = batch.get("generation_mm_token_type_ids")
    if mm_token_type_ids is None:
        mm_token_type_ids = batch.get("mm_token_type_ids")
    if latent_condition not in {"correct", "shuffled", "zero"}:
        raise ValueError(f"Unknown latent_condition: {latent_condition}")
    lam_inputs = batch.get(
        "shuffle_lam_inputs" if latent_condition == "shuffled" else "lam_inputs"
    )
    if latent_condition != "zero" and lam_inputs is None:
        raise RuntimeError(
            "generate_align requires LAM inputs (question+auxiliary image) to compute "
            "the known latent. Benchmark data has no auxiliary image — run on the test split."
        )

    batch_size = input_ids.size(0)
    device = input_ids.device

    # ---------- latent: LAM → projector → 注入 latent_pad 位（与训练 forward 同源）----------
    if latent_condition == "zero":
        mapped_latent = torch.zeros(
            input_ids.size(0),
            int(model.hparams.lam_num_latent),
            model.latent_projector.out_features,
            device=device,
            dtype=next(model.latent_projector.parameters()).dtype,
        )
    else:
        latent = injection.compute_latents(model.lam, lam_inputs, device, model._compute_dtype())
        mapped_latent = model.latent_projector(latent)
    latent_mask = input_ids.eq(model.latent_pad_token_id)
    inputs_embeds = injection.inject_latents(model.qwen, input_ids, mapped_latent, latent_mask)

    # ---------- Prefill: image features 注入 + 初次 forward ----------
    inputs_embeds, image_mask, deepstack_image_embeds = injection.inject_vision(
        model.qwen.model, input_ids, inputs_embeds, pixel_values, image_grid_thw,
    )
    position_ids = model.qwen.model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    prefill_out = model.qwen.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_image_embeds,
        use_cache=True,
    )
    past_key_values = prefill_out.past_key_values
    hidden = prefill_out[0][:, -1]  # (B, hidden_size)

    # ---------- Decode loop: 纯文本（无 latent self-rollout）----------
    generated = []
    for _ in range(max_new_tokens):
        logits = model.qwen.lm_head(hidden.unsqueeze(1))[:, -1]
        next_token = _sample_next_token(logits, temperature, top_p, top_k, do_sample)
        if (next_token == eos_id).all():  # batch=1 语义；batch>1 部分 eos 的 bug 不修（ADR-2）
            break
        generated.append(next_token.unsqueeze(-1))

        next_embeds = model.qwen.model.get_input_embeddings()(next_token.unsqueeze(-1))
        position_ids = position_ids[..., -1:] + 1  # batch=1 恒正确：末尾恒为文本 token，其位置即全局 max，逐 +1 == max+1（三维一致）；batch>1+padding 才需 rope_deltas 写法
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1,
            )
        out = model.qwen.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=next_embeds,
            past_key_values=past_key_values,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        hidden = out[0][:, -1]

    if not generated:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)
    return torch.cat(generated, dim=1)
