"""Minimal single-node NCCL collective smoke test."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=45),
        device_id=device,
    )
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {rank}: all_reduce={value.item()}, expected={expected}")
    print(f"rank={rank} gpu={local_rank} all_reduce={value.item()}", flush=True)
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
