#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
DATA_ROOT="${DATA_ROOT:-data}"
JSON_ROOT="${JSON_ROOT:-${ROOT_DIR}/data}"
exec "${PYTHON_BIN:-python}" -m lam.main \
  --train-json "${JSON_ROOT}/lam/train" --val-json "${JSON_ROOT}/lam/test" \
  --image-root "${DATA_ROOT}" \
  --vision-model-path "${QWEN_MODEL_PATH:-models/Qwen3-VL-4B-Instruct}" \
  --output-dir runs/lam/v1 --batch-size 8 --num-workers 4 \
  --max-epochs 10 --max-steps 55001 --devices 4 --accumulate-grad-batches 2 \
  --precision 32-true --seed 32 --log-every-n-steps 50 \
  --val-every-n-steps 1000 --enc-blocks 16 --dec-blocks 16 --image-size 512 "$@"
