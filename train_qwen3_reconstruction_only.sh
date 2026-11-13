#!/usr/bin/env bash
set -euo pipefail
# Four stages; both Teacher stages use Flow 0.1, QA 0 and Description 0.
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LVR_FLOW_VARIANT=qwen3-reconstruction-only
export LVR_LAUNCHER_NAME=train_qwen3_reconstruction_only.sh
exec bash "$REPRO_ROOT/train_flow.sh" "$@"
