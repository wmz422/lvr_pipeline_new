#!/usr/bin/env bash
set -euo pipefail
# Same four stages as train_qwen3_flow.sh; both Teacher stages use Flow weight zero.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen3-no-flow-loss
export LVR_LAUNCHER_NAME=train_qwen3_no_flow_loss.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
