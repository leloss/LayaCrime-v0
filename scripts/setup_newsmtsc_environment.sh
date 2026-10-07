#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${NEWSMTSC_PYTHON_BIN:-python3}"
VENV_DIR="${LAYA_NEWSMTSC_VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/newsmtsc-venv}"
SOURCE_DIR="${LAYA_NEWSMTSC_SOURCE:-$PROJECT_ROOT/third_party/NewsMTSC}"
SOURCE_REPOSITORY="https://github.com/fhamborg/NewsMTSC"
SOURCE_COMMIT="b9d9b79704ed1b35cecaf1d7c2343dc1bd734fb7"

"$PYTHON_BIN" -c 'import sys; assert (3, 8) <= sys.version_info[:2] < (3, 12), "NewsMTSC requires Python 3.8-3.11"'

if [[ ! -d "$SOURCE_DIR/NewsSentiment" ]]; then
    mkdir -p "$(dirname -- "$SOURCE_DIR")"
    git clone "$SOURCE_REPOSITORY" "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch --depth 1 origin "$SOURCE_COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install truststore
"$PYTHON" -m pip install --editable "$SOURCE_DIR"
"$PYTHON" - <<'PY'
from importlib.metadata import version

import NewsSentiment
import torch
import transformers

assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 1)
assert tuple(map(int, transformers.__version__.split(".")[:2])) <= (4, 24)
print({
    "NewsSentiment": version("NewsSentiment"),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
})
PY

printf 'NewsMTSC environment ready: %s\n' "$PYTHON"