#!/usr/bin/env bash
set -euo pipefail
# Run independent four-stage Flow experiments with 4 and 16 latents, each followed by evaluation.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_OUTPUT_PARENT=""
REPRO_FORWARDED=()
while (( $# )); do
  case "$1" in
    --help|-h)
      cat <<'HELP'
Usage: bash train_qwen3_latent_length.sh [options]

Run standard four-stage Qwen3 Flow with 4 latents, evaluate all 11 benchmarks,
then repeat with 16 latents. The 8-latent baseline is train_qwen3_flow.sh.
Each length trains its own Teacher and Student; only latent length changes.

Default outputs: runs/qwen3-flow-4stage-latent-4 and runs/qwen3-flow-4stage-latent-16.
--output PATH sets a parent directory: PATH/latent-4 and PATH/latent-16.
--resume and --eval-only apply to both lengths. --dry-run previews both runs.
All other options, including --gpus and --eval-gpus, apply to both runs:
HELP
      LVR_FLOW_VARIANT=qwen3-latent-4 LVR_LAUNCHER_NAME=train_qwen3_latent_length.sh \
        bash "$REPRO_ROOT/train_flow.sh" --help
      exit 0
      ;;
    --output)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        printf '%s\n' '--output requires a parent directory' >&2
        exit 2
      fi
      REPRO_OUTPUT_PARENT="$2"
      shift 2
      ;;
    --output=*)
      REPRO_OUTPUT_PARENT="${1#--output=}"
      if [[ -z "$REPRO_OUTPUT_PARENT" ]]; then
        printf '%s\n' '--output requires a parent directory' >&2
        exit 2
      fi
      shift
      ;;
    *) REPRO_FORWARDED+=("$1"); shift ;;
  esac
done
for REPRO_LENGTH in 4 16; do
  REPRO_OUTPUT_ARGS=()
  if [[ -n "$REPRO_OUTPUT_PARENT" ]]; then
    REPRO_OUTPUT_ARGS=(--output "${REPRO_OUTPUT_PARENT%/}/latent-$REPRO_LENGTH")
  fi
  LVR_FLOW_VARIANT="qwen3-latent-$REPRO_LENGTH" LVR_LAUNCHER_NAME=train_qwen3_latent_length.sh \
    bash "$REPRO_ROOT/train_flow.sh" "${REPRO_FORWARDED[@]}" "${REPRO_OUTPUT_ARGS[@]}"
done
