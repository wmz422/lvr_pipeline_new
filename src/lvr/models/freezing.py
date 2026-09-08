"""可训练模块控制。

先 `model.eval()` 全关，再按需 `module.train()` 解冻。`LatentVLM.train()` 会在
Lightning/DeepSpeed 切换模式后再次强制冻结的 LAM 保持 eval，避免 VAE
重参数化在下游训练中随机采样。
"""

from __future__ import annotations

from torch import nn


def _unfreeze_module(module: nn.Module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad = True


def set_trainable(
    model,
    *,
    train_qwen_lm: bool,
    train_qwen_lm_head: bool,
    train_latent_projector: bool,
    train_latent_head: bool,
    train_lam: bool,
) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # 先全部关闭，再按需打开
    if train_qwen_lm:
        _unfreeze_module(model.qwen.model.language_model)
    if train_qwen_lm_head:
        _unfreeze_module(model.qwen.lm_head)
    if train_latent_projector:
        _unfreeze_module(model.latent_projector)
    if train_latent_head:  # 注意 train_latent_head=True 时需先初始化 model.latent_head
        _unfreeze_module(model.latent_head)
    if train_lam:
        _unfreeze_module(model.lam)
