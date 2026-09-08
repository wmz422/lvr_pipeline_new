"""Construct Qwen3-VL, LAM and the latent projector from weights or local configurations."""

from __future__ import annotations

import torch
from torch import nn

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


class LatentProjectorMLP(nn.Sequential):
    """Two-layer latent projector with the same shape metadata as ``nn.Linear``."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.in_features = input_dim
        self.out_features = output_dim


def build_latent_projector(
    projector_type: str,
    input_dim: int,
    output_dim: int,
    mlp_hidden_dim: int | None = None,
) -> nn.Module:
    """Build the latent-to-Qwen embedding projector.

    ``linear`` preserves the original alignment architecture.  ``mlp`` uses a
    two-layer GELU MLP; when not specified, its hidden width is 1024.
    """
    if projector_type == "linear":
        return nn.Linear(input_dim, output_dim)
    if projector_type == "mlp":
        hidden_dim = 1024 if mlp_hidden_dim is None else mlp_hidden_dim
        if hidden_dim <= 0:
            raise ValueError("latent_projector_mlp_hidden_dim must be positive.")
        return LatentProjectorMLP(input_dim, hidden_dim, output_dim)
    raise ValueError(
        f"Unsupported latent_projector_type: {projector_type!r}; expected 'linear' or 'mlp'."
    )


def build_qwen(
    model_name_or_path: str,
    qwen_torch_dtype: str,
    add_latent_special_tokens: bool,
    latent_pad_token: str,
    gradient_checkpointing: bool,
    initialize_from_config: bool = False,
):
    """加载 HF 原生 Qwen3-VL，注册 latent special token 并 resize embedding。

    返回 (qwen, tokenizer, latent_pad_token_id)。
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = compute_dtype(qwen_torch_dtype)
    if initialize_from_config:
        config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        # Initialize at the requested weight dtype while retaining FP32 rotary
        # buffers, as from_pretrained does. Casting the whole model rounds them.
        qwen = AutoModelForImageTextToText.from_config(config, dtype=dtype)
    else:
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
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
    initialize_from_config: bool = False,
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
        load_vision_weights=not (initialize_from_config or checkpoint_path),
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
