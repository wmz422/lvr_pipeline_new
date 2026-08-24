#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/private/wmz/lvr_pipeline_new"
ARIA2="/data/private/wmz/tools/miniforge3/envs/lvr-download/bin/aria2c"
RAW_ROOT="${PROJECT_ROOT}/data/raw"
VISUAL_COT_DIR="${RAW_ROOT}/Visual-CoT"
LOG_FILE="${VISUAL_COT_DIR}/download.log"
PROXY_URL="http://127.0.0.1:7897"
# Use the mirror endpoint to obtain fresh download redirects, while the
# explicitly configured local HTTP proxy handles the actual transfer.
BASE_URL="https://hf-mirror.com/datasets/deepcs233/Visual-CoT/resolve/main"

mkdir -p "${VISUAL_COT_DIR}"

# Avoid inherited SOCKS proxy settings; use the requested 7897 proxy explicitly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

cd "${PROJECT_ROOT}"

download_one() {
  local relative_path="$1"
  local out_dir="${VISUAL_COT_DIR}/$(dirname "${relative_path}")"
  local out_name
  out_name="$(basename "${relative_path}")"

  if [[ "$(dirname "${relative_path}")" == "." ]]; then
    out_dir="${VISUAL_COT_DIR}"
  fi

  mkdir -p "${out_dir}"

  if [[ -f "${out_dir}/${out_name}" && ! -f "${out_dir}/${out_name}.aria2" ]]; then
    echo "[$(date -Is)] skip existing ${relative_path}" | tee -a "${LOG_FILE}"
    return
  fi

  echo "[$(date -Is)] start ${relative_path}" | tee -a "${LOG_FILE}"
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
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=8M \
    --dir="${out_dir}" \
    --out="${out_name}" \
    "${BASE_URL}/${relative_path}?download=true" 2>&1 | tee -a "${LOG_FILE}"
  echo "[$(date -Is)] done ${relative_path}" | tee -a "${LOG_FILE}"
}

FILES=(
  ".gitattributes"
  "README.md"
  "assets/dataset.png"
  "assets/dataset_gqa.png"
  "assets/pipeline.jpg"
  "assets/supp_demo_1.png"
  "assets/supp_demo_2.png"
  "assets/supp_demo_4.png"
  "cot_images_tar_split/cot_images_00"
  "cot_images_tar_split/cot_images_01"
  "cot_images_tar_split/cot_images_02"
  "cot_images_tar_split/cot_images_03"
  "cot_images_tar_split/cot_images_04"
  "cot_images_tar_split/cot_images_05"
  "cot_images_tar_split/cot_images_06"
  "cot_images_tar_split/cot_images_07"
  "cot_images_tar_split/cot_images_08"
  "cot_images_tar_split/cot_images_09"
  "cot_images_tar_split/cot_images_10"
  "cot_images_tar_split/cot_images_11"
  "cot_images_tar_split/cot_images_12"
  "cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl"
  "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
  "metadata/cub_cot_train.jsonl"
  "metadata/docvqa_cot_train.jsonl"
  "metadata/dude_cot_train.jsonl"
  "metadata/flickr30k_cot_train.jsonl"
  "metadata/gqa_cot_train.jsonl"
  "metadata/infographicsvqa_cot_train.jsonl"
  "metadata/openimages_cot_train.jsonl"
  "metadata/sroie_cot_train.jsonl"
  "metadata/textcap_cot_train.jsonl"
  "metadata/textvqa_cot_train.jsonl"
  "metadata/visual7w_cot_train.jsonl"
  "metadata/vsr_cot_train.jsonl"
  "viscot_363k.json"
  "viscot_mixed_2m.json"
)

echo "[$(date -Is)] start Visual-CoT aria2 download" | tee -a "${LOG_FILE}"
for relative_path in "${FILES[@]}"; do
  download_one "${relative_path}"
done
echo "[$(date -Is)] done Visual-CoT aria2 download" | tee -a "${LOG_FILE}"
