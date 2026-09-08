"""Shared latent special tokens and idempotent tokenizer registration."""

from __future__ import annotations

from typing import Any

LATENT_START_TOKEN = "<abs_vis_token>"
LATENT_PAD_TOKEN = "<abs_vis_token_pad>"
LATENT_END_TOKEN = "</abs_vis_token>"
IGNORE_INDEX = -100


def register_latent_tokens(tokenizer: Any, latent_pad_token: str = LATENT_PAD_TOKEN) -> int:
    """把 3 个 latent special token 加入 tokenizer（幂等），返回 latent pad token id。

    与旧 sft 行为一致：作为 additional_special_tokens 追加
    （`replace_additional_special_tokens=False`），旧版 transformers 无该参数时回退。
    注意：本函数不负责 `resize_token_embeddings`，模型侧 builder 在加完 token 后自行 resize。
    """
    special_tokens = {
        "additional_special_tokens": [
            LATENT_START_TOKEN,
            latent_pad_token,
            LATENT_END_TOKEN,
        ]
    }
    try:
        tokenizer.add_special_tokens(special_tokens, replace_additional_special_tokens=False)
    except TypeError:  # 旧版 transformers 没有该参数
        tokenizer.add_special_tokens(special_tokens)

    pad_id = tokenizer.convert_tokens_to_ids(latent_pad_token)
    if pad_id is None or pad_id < 0:
        raise ValueError(f"Could not resolve latent token id for {latent_pad_token!r}.")
    return pad_id
