#!/usr/bin/env bash
# Generate all v1 train/test align labels with eight Qwen3-VL-4B data-parallel
# workers.  Workers write resumable JSONL; each split is merged into the JSON
# arrays consumed directly by the alignment datamodule.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-/data/private/wmz/tools/miniforge3/envs/qwen3-grounding/bin/python}
GENERATOR="$ROOT/data/scripts/generate/align.py"
MODEL=${MODEL:-/data/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
OUT=${OUT:-$ROOT/data/align/v0}
CROP_SUBDIR=${CROP_SUBDIR:-crops_qwen4b_direct_content}

mkdir -p "$OUT/logs"
start_seconds=$(date +%s)

run_split() {
  local split=$1
  local -a pids=()
  for gpu in $(seq 0 7); do
    # Individual sessions keep each shard alive and resumable after disconnects.
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" -u "$GENERATOR" \
      --data-root "$ROOT/data" --split "$split" --num-shards 8 --shard-index "$gpu" \
      --model "$MODEL" --observation-source qwen3-vl-4b-instruct-crop-only \
      --crop-subdir "$CROP_SUBDIR" --output-dir "$OUT" \
      >"$OUT/logs/${split}.qwen4b.gpu${gpu}.log" 2>&1 < /dev/null &
    pids+=("$!")
  done
  wait "${pids[@]}"
  "$PYTHON" "$GENERATOR" --data-root "$ROOT/data" --split "$split" \
    --num-shards 8 --merge --output-dir "$OUT"
}

run_split train
run_split test
end_seconds=$(date +%s)
printf '{"wall_seconds": %s, "model": "Qwen3-VL-4B-Instruct", "workers": 8, "splits": ["train", "test"]}\n' \
  "$((end_seconds - start_seconds))" > "$OUT/run_metrics_qwen4b_full.json"
