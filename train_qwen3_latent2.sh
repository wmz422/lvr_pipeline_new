#!/usr/bin/env bash
set -euo pipefail
# Standard four-stage Qwen3 Flow with two latents, followed by all 11 benchmarks.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen3-latent-2
export LVR_LAUNCHER_NAME=train_qwen3_latent2.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
