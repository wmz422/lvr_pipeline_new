#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/data/private/wmz/model_weights/Qwen3-VL-32B-Instruct
DIRECT_DIR=/data/private/wmz/model_weights/Qwen3-VL-32B-Instruct.direct-download-01-08
PYTHON=/data/private/wmz/tools/miniforge3/envs/qwen3-grounding/bin/python
GENERATOR=/data/private/wmz/lvr_pipeline_new/data/scripts/generate/benchmark_auxiliary.py
BENCHMARK_ROOT=/data/private/wmz/lvr_pipeline_new/benchmark/bench
OUTPUT_ROOT=/data/private/wmz/lvr_pipeline_new/data/images/auxiliary/benchmark
SMOKE_ROOT=/data/private/wmz/lvr_pipeline_new/data/images/auxiliary/benchmark_smoke
RUN_ROOT=/data/private/wmz/lvr_pipeline_new/data/images/auxiliary/benchmark_run
LOG_ROOT="$RUN_ROOT/logs"

mkdir -p "$DIRECT_DIR" "$MODEL_DIR" "$LOG_ROOT"
export PYTHONPATH=/data/private/wmz/lvr_pipeline_new/data/scripts/generate
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

download_shard() {
    local shard="$1"
    local target="$DIRECT_DIR/model-000${shard}-of-00014.safetensors"
    local url="https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/resolve/main/model-000${shard}-of-00014.safetensors"
    local total current end expected actual got_start got_end
    local chunk="$DIRECT_DIR/.range-${shard}.part"
    local headers="$DIRECT_DIR/.range-${shard}.headers"

    while :; do
        if total=$(curl --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 2 \
            -r 0-0 -D - -o /dev/null "$url" --silent --show-error |
            awk 'BEGIN{IGNORECASE=1} /^content-range:/{split($3, parts, "/"); gsub(/\r/, "", parts[2]); print parts[2]; exit}'); then
            break
        fi
        echo "shard ${shard} size probe failed; retrying"
        sleep 2
    done
    [[ "$total" =~ ^[0-9]+$ ]]

    while :; do
        current=$(stat -c '%s' "$target" 2>/dev/null || echo 0)
        if (( current >= total )); then
            break
        fi
        end=$((current + 16777216 - 1))
        if (( end >= total )); then
            end=$((total - 1))
        fi
        expected=$((end - current + 1))
        if ! curl --http1.1 -L --fail --retry 100 --retry-all-errors --retry-delay 2 \
            -r "${current}-${end}" -D "$headers" -o "$chunk" "$url" --silent --show-error; then
            echo "shard ${shard} range ${current}-${end} failed; retrying"
            continue
        fi
        actual=$(stat -c '%s' "$chunk" 2>/dev/null || echo 0)
        got_start=$(awk 'BEGIN{IGNORECASE=1} /^content-range:/{split($3, parts, "-"); print parts[1]}' "$headers" | tail -1)
        got_end=$(awk 'BEGIN{IGNORECASE=1} /^content-range:/{split($3, parts, "-"); split(parts[2], endparts, "/"); print endparts[1]}' "$headers" | tail -1)
        if [[ "$actual" != "$expected" || "$got_start" != "$current" || "$got_end" != "$end" ]]; then
            echo "shard ${shard} range mismatch requested=${current}-${end} got=${got_start}-${got_end} bytes=${actual}; retrying"
            continue
        fi
        cat "$chunk" >> "$target"
    done
    printf 'shard %s complete bytes=%s\n' "$shard" "$(stat -c '%s' "$target")"
}

echo "[$(date -Is)] resuming model shards 01-08"
download_pids=()
for shard in 01 02 03 04 05 06 07 08; do
    download_shard "$shard" &
    download_pids+=("$!")
done
for pid in "${download_pids[@]}"; do
    wait "$pid"
done

echo "[$(date -Is)] validating downloaded shards 01-08"
(
    cd "$DIRECT_DIR"
    printf '%s\n' \
        '6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb  model-00001-of-00014.safetensors' \
        'd8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa  model-00002-of-00014.safetensors' \
        '54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940  model-00003-of-00014.safetensors' \
        'ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f  model-00004-of-00014.safetensors' \
        'fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2  model-00005-of-00014.safetensors' \
        '82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974  model-00006-of-00014.safetensors' \
        'fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934  model-00007-of-00014.safetensors' \
        '431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f  model-00008-of-00014.safetensors' | sha256sum -c -
)

mv "$DIRECT_DIR"/*.safetensors "$MODEL_DIR"/

echo "[$(date -Is)] validating all 14 model shards"
(
    cd "$MODEL_DIR"
    printf '%s\n' \
        '6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb  model-00001-of-00014.safetensors' \
        'd8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa  model-00002-of-00014.safetensors' \
        '54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940  model-00003-of-00014.safetensors' \
        'ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f  model-00004-of-00014.safetensors' \
        'fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2  model-00005-of-00014.safetensors' \
        '82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974  model-00006-of-00014.safetensors' \
        'fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934  model-00007-of-00014.safetensors' \
        '431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f  model-00008-of-00014.safetensors' \
        '3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47  model-00009-of-00014.safetensors' \
        'aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f  model-00010-of-00014.safetensors' \
        '3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f  model-00011-of-00014.safetensors' \
        '05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e  model-00012-of-00014.safetensors' \
        'b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c  model-00013-of-00014.safetensors' \
        'e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b  model-00014-of-00014.safetensors' | sha256sum -c -
)

benchmarks=(
    hr_bench_4k
    hr_bench_8k
    mme_realworld_lite
    blink_counting
    blink_spatial_relation
)

common_args=(
    --benchmark-root "$BENCHMARK_ROOT"
    --output-root "$OUTPUT_ROOT"
    --model "$MODEL_DIR"
    --max-pixels 4194304
    --max-new-tokens 1024
    --device-map balanced
    --benchmarks "${benchmarks[@]}"
)

echo "[$(date -Is)] testing one A100 with a high-resolution HR-Bench sample"
if CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$GENERATOR" \
    --benchmark-root "$BENCHMARK_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --model "$MODEL_DIR" \
    --max-pixels 4194304 \
    --device-map balanced \
    --benchmarks hr_bench_8k \
    --limit 1 \
    --fail-fast >"$LOG_ROOT/smoke-one-gpu.log" 2>&1; then
    workers=8
    gpu_mode=single
    echo "[$(date -Is)] one-GPU smoke test passed; using 8 workers"
else
    workers=4
    gpu_mode=paired
    echo "[$(date -Is)] one-GPU smoke test failed; using 4 two-GPU workers"
fi

run_workers() {
    local retry_failed="$1"
    local label="$2"
    local worker_pids=()
    local shard visible first_gpu second_gpu

    for ((shard = 0; shard < workers; shard++)); do
        if [[ "$gpu_mode" == single ]]; then
            visible="$shard"
        else
            first_gpu=$((shard * 2))
            second_gpu=$((first_gpu + 1))
            visible="${first_gpu},${second_gpu}"
        fi

        retry_args=()
        if [[ "$retry_failed" == yes ]]; then
            retry_args+=(--retry-failed)
        fi
        CUDA_VISIBLE_DEVICES="$visible" "$PYTHON" "$GENERATOR" \
            "${common_args[@]}" \
            --num-shards "$workers" \
            --shard-index "$shard" \
            "${retry_args[@]}" \
            >"$LOG_ROOT/${label}-shard-${shard}.log" 2>&1 &
        worker_pids+=("$!")
    done

    set +e
    local failed_workers=0
    for pid in "${worker_pids[@]}"; do
        wait "$pid" || failed_workers=$((failed_workers + 1))
    done
    set -e
    echo "[$(date -Is)] ${label} finished; failed_workers=${failed_workers}"
}

echo "[$(date -Is)] starting full answer-conditioned grounding"
run_workers no first-pass
echo "[$(date -Is)] retrying failed samples once"
run_workers yes retry-pass

echo "[$(date -Is)] final PNG counts"
for directory in hr_bench_4k hr_bench_8k mme_realworld_lite blink/Counting/val blink/Spatial_Relation/val; do
    count=$(find "$OUTPUT_ROOT/$directory" -type f -name '*.png' 2>/dev/null | wc -l)
    echo "$directory $count"
done
echo "[$(date -Is)] benchmark auxiliary generation pipeline finished"
