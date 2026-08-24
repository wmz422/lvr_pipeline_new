#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Shared-memory transport is stable for the four-rank collective on this host.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/private/wmz/tools/miniforge3/envs/lvr/bin/python}"
cd "${ROOT_DIR}"

TRAIN_ARGS=(
  --train-json data/lam/v1/train.json
  --val-json data/lam/v0/test.json
  --image-root data
  --vision-model-path /data/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
  --output-dir runs/lam/v1
  --logger-version 6
  --batch-size 8
  --num-workers 4
  --max-epochs 10
  --max-steps -1
  # CUDA_VISIBLE_DEVICES remaps physical GPUs 4--7 to local IDs 0--3.
  --devices 0,1,2,3
  # Preserve the original effective batch size: 8 GPUs x 8 x 1 = 4 GPUs x 8 x 2.
  --accumulate-grad-batches 2
  --precision 32-true
  --log-every-n-steps 50
  --val-every-n-steps 1000
  --enc-blocks 16
  --dec-blocks 16
  --image-size 512
)

# Extra command-line arguments are appended so they can override defaults when
# the same option is supplied later by argparse.
exec "${PYTHON_BIN}" -m lam.main "${TRAIN_ARGS[@]}" "$@"
