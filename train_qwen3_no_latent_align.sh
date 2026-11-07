#!/usr/bin/env bash
set -euo pipefail
# Reuse Flow stage 2; skip warmup; optimize answers with predicted latents only.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen3-no-latent-align
export LVR_LAUNCHER_NAME=train_qwen3_no_latent_align.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
