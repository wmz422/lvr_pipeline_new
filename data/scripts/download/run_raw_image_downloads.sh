#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="/data/private/wmz/lvr_pipeline_new/data/raw"
ARIA2="/data/private/wmz/tools/miniforge3/envs/lvr-download/bin/aria2c"
PROXY_URL="http://127.0.0.1:7897"
LOG_FILE="${RAW_ROOT}/download_raw_images.log"

mkdir -p "${RAW_ROOT}/gqa" "${RAW_ROOT}/coco2014" "${RAW_ROOT}/coco2017" "${RAW_ROOT}/VisDrone"

# Avoid inherited proxy settings confusing aria2; use the requested 7897 proxy explicitly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

run_aria2() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  local split="$4"

  mkdir -p "${out_dir}"
  echo "[$(date -Is)] start ${out_dir}/${out_name}" | tee -a "${LOG_FILE}"
  "${ARIA2}" \
    --all-proxy="${PROXY_URL}" \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --max-tries=0 \
    --retry-wait=10 \
    --timeout=60 \
    --connect-timeout=30 \
    --summary-interval=60 \
    --max-connection-per-server="${split}" \
    --split="${split}" \
    --min-split-size=8M \
    --dir="${out_dir}" \
    --out="${out_name}" \
    "${url}" 2>&1 | tee -a "${LOG_FILE}"
  echo "[$(date -Is)] done ${out_dir}/${out_name}" | tee -a "${LOG_FILE}"
}

run_aria2 "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
  "${RAW_ROOT}/gqa" "images.zip" 8

run_aria2 "http://images.cocodataset.org/zips/train2014.zip" \
  "${RAW_ROOT}/coco2014" "train2014.zip" 8

run_aria2 "http://images.cocodataset.org/zips/train2017.zip" \
  "${RAW_ROOT}/coco2017" "train2017.zip" 8

# GitHub release assets have returned range/SSL errors with aggressive splitting in this environment.
run_aria2 "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip" \
  "${RAW_ROOT}/VisDrone" "VisDrone2019-DET-val.zip" 1

run_aria2 "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip" \
  "${RAW_ROOT}/VisDrone" "VisDrone2019-DET-test-dev.zip" 1
