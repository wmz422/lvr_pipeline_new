#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON_BIN:-python}
revision=34a64e619d5466ab3a900f1720041ea88d4a1ed6
target="$repo_root/third_party/VLMEvalKit"
if [[ ! -d "$target" ]]; then
  git clone --no-checkout https://github.com/open-compass/VLMEvalKit.git "$target"
  git -C "$target" checkout --detach "$revision"
fi
if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]] || [[ -n "$(git -C "$target" status --porcelain)" ]]; then
  echo "VLMEvalKit must be an unmodified checkout at $revision: $target" >&2
  exit 2
fi
"$python_bin" -m pip install -e "$target" -c "$repo_root/constraints-prediction.txt"
