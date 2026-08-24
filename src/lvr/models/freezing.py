"""可训练模块控制。

搬自旧 sft/src/model.py 的 _set_trainable / _unfreeze_module，**行为保持不变**（ADR-2）：
先 `model.eval()` 全关，再按需 `module.train()` 解冻。注意 Lightning 训练循环启动时会调
`model.train()` 覆盖这里的 eval —— 这是已知 TODO，本次重构不修，只忠实搬运。
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
    #breakpoint()#确认eval情况，权重加载情况
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
