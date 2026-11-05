#!/usr/bin/env bash
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${LVR_ABLATION_PYTHON:-$ABLATION_ROOT/.venv/bin/python}" "$ABLATION_ROOT/ablation/run.py" "$@"
