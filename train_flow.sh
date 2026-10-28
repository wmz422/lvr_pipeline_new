#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DRY_RUN=0
for REPRO_ARG in "$@"; do
  case "$REPRO_ARG" in
    --help|-h)
      cat <<'HELP'
Usage: bash train_flow.sh [options]

Prepare pinned data/models, build the offline VAE cache, train Teacher -> Warmup
-> Joint with validation, then evaluate all 11 benchmarks. Global batch is 128.
Hardware settings are manual; no configuration search or automatic batch changes.

  --gpus N                         Training GPUs (default: 4)
  --teacher-batch-size N            Teacher batch per GPU (default: 8)
  --warmup-batch-size N             Warmup batch per GPU (default: 8)
  --joint-batch-size N              Joint batch per GPU (default: 4)
  --zero-stage {2,3}                Default ZeRO stage for all phases (default: 3)
  --teacher-zero-stage {2,3}        Optional per-phase ZeRO override
  --warmup-zero-stage {2,3}
  --joint-zero-stage {2,3}
  --teacher-optimizer-device {cpu,cuda}  Default: cpu
  --warmup-optimizer-device {cpu,cuda}   Default: cuda
  --joint-optimizer-device {cpu,cuda}    Default: cuda
  --parameter-offload {none,cpu}    Parameter offload; CPU requires all ZeRO-3
  --flow-generator-batch-size N    Frozen Flow sub-batch (default: 2)
  --eval-gpus N                    Defaults to min(--gpus, 8)
  --output PATH                    New run directory (default: runs/flow)
  --resume                         Resume this output with identical settings
  --preflight                      Opt in to a separate four-update GPU smoke check
  --skip-bootstrap                 Use data/models already prepared in this checkout
  --data-source PATH_OR_URL        Optional local or HTTPS data release directory
  --dry-run                        Preview stages/settings; no downloads or GPU use

accumulation = 128 / (GPUs * per-GPU batch). The divisor must be exact.
For the unchanged 64-sample epoch tail, each per-GPU batch must be at least 2.
Other advanced options are listed by: python reproduce.py train --help

First execution installs pinned environments if absent (Python 3.12 + 3.13 required).
LVR_TRAIN_PYTHON and LVR_SCORING_PYTHON can point to existing pinned environments.
HELP
      exit 0
      ;;
    --dry-run) REPRO_DRY_RUN=1 ;;
  esac
done
if [[ -z "${LVR_TRAIN_PYTHON:-}" ]] && { [[ ! -x "$REPRO_ROOT/.venv/bin/python" ]] || [[ ! -x "$REPRO_ROOT/.venv-score/bin/python" ]]; }; then
  if [[ "$REPRO_DRY_RUN" == 1 ]]; then
    printf '%s\n' 'Create the environments first: bash tools/setup.sh' >&2
    exit 2
  fi
  bash "$REPRO_ROOT/tools/setup.sh"
fi
REPRO_PYTHON="${LVR_TRAIN_PYTHON:-$REPRO_ROOT/.venv/bin/python}"
exec "$REPRO_PYTHON" "$REPRO_ROOT/tools/start_flow.py" "$@"
