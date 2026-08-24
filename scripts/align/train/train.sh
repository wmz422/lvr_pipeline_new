#!/usr/bin/env bash
# Alignment (Stage-1) training entry point.
#
# The final arguments are deliberately appended after TRAIN_ARGS, so callers can
# override any default, for example:
#   bash scripts/align/train/train.sh --trainer.max_steps 2 --data.train_limit 16
#
# The data paths are relative to DATA_ROOT.  `image_root` must be DATA_ROOT (not
# DATA_ROOT/images), because records contain paths such as images/original/....

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# This host's NCCL P2P path is unreliable; use shared-memory transport unless
# the caller explicitly supplies a different setting.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

PYTHON_BIN="${PYTHON_BIN:-/data/private/wmz/tools/miniforge3/envs/lvr/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
LAM_CHECKPOINT_PATH="${LAM_CHECKPOINT_PATH:-${ROOT_DIR}/runs/lam/v1/checkpoints/lam-epoch=003-step=00035000-val_loss=0.00000.ckpt}"
# This must be the same 4B vision tower recorded in the LAM checkpoint.  The
# standalone qwen3_vision directory is a different 27-layer/1152-dim tower.
LAM_VISION_MODEL_PATH="${LAM_VISION_MODEL_PATH:-${QWEN_MODEL_PATH}}"
VERSION="${VERSION:-v0}"

for required_path in \
  "${DATA_ROOT}/align/v0/train.json" \
  "${DATA_ROOT}/align/v0/test.json" \
  "${QWEN_MODEL_PATH}" \
  "${LAM_CHECKPOINT_PATH}" \
  "${LAM_VISION_MODEL_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[align/train] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

cd "${ROOT_DIR}"

TRAIN_ARGS=(
  --trainer.accelerator gpu
  --trainer.devices 4
  # Lightning selects DDP for the default four-GPU run and remains usable for
  # a one-GPU smoke test when callers override --trainer.devices 1.
  --trainer.strategy auto
  --trainer.precision bf16-mixed
  --trainer.max_epochs 5
  # 1,000 micro-batches = 250 optimizer steps with accumulation=4.
  --trainer.val_check_interval 1000
  --trainer.limit_val_batches 256
  --trainer.accumulate_grad_batches 4
  --trainer.gradient_clip_val 1.0
  --trainer.log_every_n_steps 50
  --trainer.callbacks lightning.pytorch.callbacks.ModelCheckpoint
  --trainer.callbacks.monitor "val/ce_loss_epoch"
  --trainer.callbacks.mode min
  --trainer.callbacks.filename "align-{epoch:03d}-{step:08d}"
  --trainer.callbacks.auto_insert_metric_name false
  --trainer.callbacks.save_top_k 3
  --trainer.callbacks.save_last true
  --trainer.callbacks.save_on_train_epoch_end false

  --model.learning_rate 1.0e-4
  --model.qwen_model_name_or_path "${QWEN_MODEL_PATH}"
  --model.qwen_torch_dtype bfloat16
  --model.lam_checkpoint_path "${LAM_CHECKPOINT_PATH}"
  --model.lam_vision_model_path "${LAM_VISION_MODEL_PATH}"
  --model.lam_model_dim 1024
  --model.lam_latent_dim 32
  --model.lam_patch_size 16
  --model.lam_enc_blocks 16
  --model.lam_dec_blocks 16
  --model.lam_num_heads 16
  --model.lam_num_latent 4
  --model.train_qwen_lm false
  --model.train_qwen_lm_head false
  --model.train_latent_projector true
  --model.train_latent_head false
  --model.train_lam false
  --model.qwen_gradient_checkpointing true
  --model.save_trainable_only true
  --model.loss_lambda 0.0
  --model.generation_mode align

  --data.stage align
  --data.train_path "${DATA_ROOT}/align/v0/train.json"
  --data.test_path "${DATA_ROOT}/align/v0/test.json"
  --data.image_root "${DATA_ROOT}"
  --data.processor_name_or_path "${QWEN_MODEL_PATH}"
  --data.batch_size 4
  --data.num_workers 4
  --data.max_length 4096
  --data.latent_len 4
  --data.lam_image_size 512
  --data.system_prompt "You are a helpful assistant."
  --data.qwen_dynamic_resolution true
  --data.qwen_max_pixels 2408448
)

# Extra command-line arguments are appended so they can override defaults.
exec "${PYTHON_BIN}" -m lvr.cli fit \
  --exp_name align --version "${VERSION}" \
  "${TRAIN_ARGS[@]}" "$@"
