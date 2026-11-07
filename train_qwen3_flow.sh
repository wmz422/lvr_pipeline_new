#!/usr/bin/env bash
set -euo pipefail
# Teacher alignment (Description + Flow) -> Teacher joint -> warmup -> joint (no Description).
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen3
export LVR_LAUNCHER_NAME=train_qwen3_flow.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
