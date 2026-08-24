"""组件构建：Qwen3-VL / LAM / dtype / hidden size。

搬自旧 sft/src/model.py 的 _build_qwen / _build_lam / _qwen_hidden_size / _compute_dtype。
special token 注册改为调用 lvr.tokens.register_latent_tokens（与数据侧共用唯一入口）。
"""

from __future__ import annotations

import torch

from lvr.tokens import register_latent_tokens


def compute_dtype(qwen_torch_dtype: str) -> torch.dtype:
    if qwen_torch_dtype == "bfloat16":
        return torch.bfloat16
    if qwen_torch_dtype == "float16":
        return torch.float16
    return torch.float32


def qwen_hidden_size(qwen) -> int:
    config = qwen.config
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    return int(config.hidden_size)


def build_qwen(
    model_name_or_path: str,
    qwen_torch_dtype: str,
    add_latent_special_tokens: bool,
    latent_pad_token: str,
    gradient_checkpointing: bool,
):
    """加载 HF 原生 Qwen3-VL，注册 latent special token 并 resize embedding。

    返回 (qwen, tokenizer, latent_pad_token_id)。
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = compute_dtype(qwen_torch_dtype)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if add_latent_special_tokens:
        register_latent_tokens(tokenizer, latent_pad_token)
        # tokenizer 与 embedding 层需保持词汇量一致；新增行用随机值初始化。
        qwen.resize_token_embeddings(len(tokenizer))

    latent_pad_token_id = tokenizer.convert_tokens_to_ids(latent_pad_token)
    if latent_pad_token_id is None or latent_pad_token_id < 0:
        raise ValueError(f"Could not resolve latent token id for {latent_pad_token!r}.")

    if gradient_checkpointing:
        if hasattr(qwen, "gradient_checkpointing_enable"):
            qwen.gradient_checkpointing_enable()
        elif hasattr(qwen.model.language_model, "gradient_checkpointing_enable"):
            qwen.model.language_model.gradient_checkpointing_enable()
        if hasattr(qwen.config, "use_cache"):
            qwen.config.use_cache = False
        if hasattr(qwen.config, "text_config") and hasattr(qwen.config.text_config, "use_cache"):
            qwen.config.text_config.use_cache = False

    return qwen, tokenizer, latent_pad_token_id


def build_lam(
    checkpoint_path: str | None,
    vision_model_path: str,
    model_dim: int,
    latent_dim: int,
    patch_size: int,
    enc_blocks: int,
    dec_blocks: int,
    num_heads: int,
    num_latent: int,
):
    """构造 feature-space LAM（复用 Qwen3 vision encoder），加载 checkpoint（冻结使用）。"""
    from lvr.modules.lam import LatentActionModel

    lam = LatentActionModel(
        in_dim=3,
        model_dim=model_dim,
        latent_dim=latent_dim,
        patch_size=patch_size,
        enc_blocks=enc_blocks,
        dec_blocks=dec_blocks,
        num_heads=num_heads,
        num_latent=num_latent,
        feature_space=True,
        save_dir=vision_model_path,
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # 外部训练 checkpoint 可能带 lam. 前缀，两种格式都兼容。
        stripped_state_dict = {
            key.removeprefix("lam."): value
            for key, value in state_dict.items()
            if key.startswith("lam.")
        }
        if not stripped_state_dict:
            stripped_state_dict = state_dict
        lam.load_state_dict(stripped_state_dict, strict=True)
    return lam
