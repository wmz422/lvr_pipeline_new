#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_TRAIN_PYTHON="${REPRO_TRAIN_PYTHON:-python3.12}"
REPRO_SCORE_PYTHON="${REPRO_SCORE_PYTHON:-python3.13}"
export CUDA_VISIBLE_DEVICES=""
export SETUPTOOLS_USE_DISTUTILS=local
export DS_BUILD_OPS=0
"$REPRO_TRAIN_PYTHON" -m venv "$REPRO_ROOT/.venv"
"$REPRO_ROOT/.venv/bin/python" -m pip install 'setuptools==78.1.0' wheel
"$REPRO_ROOT/.venv/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
"$REPRO_ROOT/.venv/bin/python" -m pip install --no-build-isolation -r "$REPRO_ROOT/requirements/training.txt"
"$REPRO_SCORE_PYTHON" -m venv "$REPRO_ROOT/.venv-score"
"$REPRO_ROOT/.venv-score/bin/python" -m pip install -r "$REPRO_ROOT/requirements/scoring.txt"
"$REPRO_ROOT/.venv/bin/python" "$REPRO_ROOT/reproduce.py" configure
"$REPRO_ROOT/.venv/bin/python" "$REPRO_ROOT/reproduce.py" doctor --environment-only

