"""loss：文本 CE + latent 位 MSE。

搬自旧 sft/src/model.py：CE 走 Qwen 自带 loss_function；MSE 比较 latent 位注入 embedding
与 Qwen 输出 hidden state（鼓励 latent 位 hidden 与注入表示一致，为推理 latent 自回归做准备）。
λ 组合（loss = ce + λ·mse）在 LatentVLM 里完成。
"""

from __future__ import annotations

import torch
from torch import nn


def text_ce_loss(qwen, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return qwen.loss_function(
        logits=logits,
        labels=labels,
        vocab_size=qwen.config.text_config.vocab_size,
    )


def text_ce_loss_from_hidden(qwen, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """显存高效 CE：只在被监督(labels!=-100)的位置跑 lm_head + CE。

    动机：动态分辨率下序列长(~1357)、词表大(~152k)，在**全序列**上 materialize logits ~数 GB，
    但 CE 只监督 ~50 个 observation token，其余全是 -100。lm_head 是线性层，
    `lm_head(hidden[mask]) == lm_head(hidden)[mask]`，故只在被监督位算 logits 与
    `qwen.loss_function(全logits, labels)` **数值等价**（同样 causal shift、忽略 -100、mean 归约），
    但峰值显存从 batch×seq×vocab 降到 K×vocab(K≈被监督 token 数)。
    """
    # causal shift：position i 的 hidden 预测 labels[i+1]（与 HF loss_function 内部一致）。
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    supervised = shift_labels != -100
    if not bool(supervised.any()):
        return (hidden_states.sum() * 0.0)  # 无监督 token：返回 0 且保持计算图连通
    selected_hidden = shift_hidden[supervised]          # (K, H)
    selected_labels = shift_labels[supervised]          # (K,)
    selected_logits = qwen.lm_head(selected_hidden)     # (K, vocab)，K≈50，极小
    return nn.functional.cross_entropy(selected_logits.float(), selected_labels)


def latent_mse_loss(
    latent_mask: torch.Tensor,
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    latent_inputs = inputs_embeds[latent_mask]
    latent_hidden = hidden_states[latent_mask]
    return nn.functional.mse_loss(latent_inputs, latent_hidden)
