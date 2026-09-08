#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-models/Qwen3-VL-4B-Instruct}"
LAM_CHECKPOINT_PATH="${LAM_CHECKPOINT_PATH:-models/lvr-sft-v3-4000/lam/lam.ckpt}"
ALIGN_CHECKPOINT_PATH="${ALIGN_CHECKPOINT_PATH:-runs/align/v2/checkpoints/align-000-00001000.ckpt}"
DATA_ROOT="${DATA_ROOT:-data}"
JSON_ROOT="${JSON_ROOT:-${ROOT_DIR}/data}"
exec "${PYTHON_BIN:-python}" -m lvr.cli fit --config configs/sft.yaml \
  --model.qwen_model_name_or_path "${QWEN_MODEL_PATH}" \
  --model.lam_vision_model_path "${QWEN_MODEL_PATH}" \
  --model.lam_checkpoint_path "${LAM_CHECKPOINT_PATH}" \
  --model.init_checkpoint_path "${ALIGN_CHECKPOINT_PATH}" \
  --data.processor_name_or_path "${QWEN_MODEL_PATH}" \
  --data.image_root "${DATA_ROOT}" \
  --data.train_path "${JSON_ROOT}/sft/train" --data.test_path "${JSON_ROOT}/sft/test" "$@"
