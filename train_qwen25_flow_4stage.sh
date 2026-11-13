#!/usr/bin/env bash
set -euo pipefail
# The complete Qwen3 four-stage Flow recipe on Qwen2.5-VL-7B-Instruct.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen25-four-stage
export LVR_LAUNCHER_NAME=train_qwen25_flow_4stage.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
