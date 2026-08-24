"""checkpoint 统一加载：HF 分片 / DeepSpeed ZeRO / 单 pt 三格式。

搬自旧 sft/src/model.py 的 _init_checkpoint，**逻辑按原样**（ADR-2：本次不修"先建模再加载"
的双份显存问题）。用于从 alignment 或上一阶段加载权重（load_state_dict strict=False），
与 Lightning 原生 ckpt_path（恢复 optimizer/scheduler）不同。
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_init_checkpoint(model, init_checkpoint_path: str) -> None:
    path = Path(init_checkpoint_path).expanduser().resolve()
    state_dict: dict[str, torch.Tensor] = {}

    if path.is_file() and path.suffix in (".pt", ".pth"):
        # 单文件权重：DeepSpeed zero_to_fp32 合并产物或普通 .pt
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    elif path.is_dir():
        # DeepSpeed ZeRO checkpoint: checkpoint/ 子目录下有 zero_pp_rank_* 分片
        ds_checkpoint_dir = path / "checkpoint"
        ds_shards = sorted(ds_checkpoint_dir.glob("zero_pp_rank_*_mp_rank_*_model_states.pt"))
        if ds_shards:
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )

            state_dict = get_fp32_state_dict_from_zero_checkpoint(str(path))
        else:
            # 标准分片权重目录：pytorch_model-*-of-*.bin
            shard_files = sorted(path.glob("pytorch_model-*-of-*.bin"))
            if not shard_files:
                raise FileNotFoundError(f"No checkpoint found in {path}")
            for shard in shard_files:
                shard_data = torch.load(str(shard), map_location="cpu", weights_only=True)
                state_dict.update(shard_data)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    n_params = sum(t.numel() for t in state_dict.values())
    print(f"[checkpoint] state dict 就绪（{n_params / 1e9:.1f}B 参数 fp32），开始灌入模型"
          f"（fp32→模型 dtype 逐张拷贝，约需数分钟，期间无输出）...", flush=True)
    model.load_state_dict(state_dict=state_dict, strict=False)
    print("[checkpoint] 权重灌入完成", flush=True)
