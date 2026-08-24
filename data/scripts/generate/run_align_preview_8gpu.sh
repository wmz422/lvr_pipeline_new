#!/usr/bin/env bash
# Generate the first 100 v1 test records with one Qwen worker sharded across
# all eight A100s.  This reads the 32B checkpoint once (rather than once/GPU),
# then merges the resumable JSONL into a training-ready JSON array.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-/data/private/wmz/tools/miniforge3/envs/qwen3-grounding/bin/python}
GENERATOR="$ROOT/data/scripts/generate/align.py"
OUT="$ROOT/data/align/v0"

mkdir -p "$OUT/logs"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid "$PYTHON" -u "$GENERATOR" \
  --data-root "$ROOT/data" --split test --limit 100 \
  --num-shards 1 --shard-index 0 --device-map balanced --output-dir "$OUT" \
  >"$OUT/logs/test.first100.tp8.log" 2>&1 < /dev/null &
worker_pid=$!
wait "$worker_pid"

"$PYTHON" "$GENERATOR" --data-root "$ROOT/data" --split test --limit 100 \
  --num-shards 1 --merge --output-dir "$OUT"
