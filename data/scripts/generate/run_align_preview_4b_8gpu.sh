#!/usr/bin/env bash
# Compare Qwen3-VL-4B against the 32B preview on exactly the first 100 test
# records.  Eight independent workers provide genuine data parallelism.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-/data/private/wmz/tools/miniforge3/envs/qwen3-grounding/bin/python}
GENERATOR="$ROOT/data/scripts/generate/align.py"
MODEL=${MODEL:-/data/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
OUT=${OUT:-$ROOT/data/align/v0/qwen4b_preview100}

mkdir -p "$OUT/logs"
start_seconds=$(date +%s)
pids=()
for gpu in $(seq 0 7); do
  # A separate session preserves each resumable JSONL if the caller disconnects.
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" -u "$GENERATOR" \
    --data-root "$ROOT/data" --split test --limit 100 \
    --num-shards 8 --shard-index "$gpu" --model "$MODEL" \
    --observation-source qwen3-vl-4b-instruct-crop-only --output-dir "$OUT" \
    >"$OUT/logs/test.first100.gpu${gpu}.log" 2>&1 < /dev/null &
  pids+=("$!")
done
wait "${pids[@]}"

"$PYTHON" "$GENERATOR" --data-root "$ROOT/data" --split test --limit 100 \
  --num-shards 8 --merge --output-dir "$OUT"
end_seconds=$(date +%s)
printf '{"wall_seconds": %s, "records": 100, "model": "Qwen3-VL-4B-Instruct", "workers": 8}\n' \
  "$((end_seconds - start_seconds))" > "$OUT/run_metrics.json"
