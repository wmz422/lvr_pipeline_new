"""Load complete safetensors bundles strictly, or initialize stages from native training checkpoints."""

from __future__ import annotations

from pathlib import Path
import json

import torch


def load_init_checkpoint(model, init_checkpoint_path: str) -> None:
    path = Path(init_checkpoint_path).expanduser().resolve()
    state_dict: dict[str, torch.Tensor] = {}

    index_file = path / "model.safetensors.index.json"
    safe_file = path / "model.safetensors"
    if path.suffix == ".safetensors" and path.is_file():
        safe_files = [path]
    elif index_file.is_file():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        safe_files = [path / name for name in sorted(set(weight_map.values()))]
    elif safe_file.is_file():
        safe_files = [safe_file]
    else:
        safe_files = []
    if safe_files:
        from safetensors.torch import load_file

        for shard in safe_files:
            tensors = load_file(shard, device="cpu")
            duplicate = set(state_dict).intersection(tensors)
            if duplicate:
                raise ValueError(f"Duplicate checkpoint keys: {sorted(duplicate)[:3]}")
            state_dict.update(tensors)
        if index_file.is_file() and set(state_dict) != set(weight_map):
            raise ValueError("Safetensors index and shard keys do not match")
        # Public bundles contain the whole model. Never silently accept missing
        # frozen modules or a mismatched architecture.
        torch.nn.Module.load_state_dict(model, state_dict, strict=True)
        print(f"[checkpoint] Loaded {len(state_dict)} safetensors tensors (strict).", flush=True)
        return

    if path.is_file() and path.suffix in (".pt", ".pth", ".ckpt"):
        # 单文件权重：DeepSpeed zero_to_fp32 合并产物、普通 .pt，或 Lightning
        # ModelCheckpoint 的 .ckpt。后者包含 optimizer 等元数据，模型权重在
        # ``state_dict`` 字段；这里是阶段间初始化而非 resume，故只取该字段。
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

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
