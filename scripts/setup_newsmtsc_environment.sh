#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "NewsMTSC environment setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${NEWSMTSC_PYTHON_BIN:-}"
VENV_DIR="${LAYA_NEWSMTSC_VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/newsmtsc-venv}"
MAIN_VENV_DIR="${LAYA_MAIN_VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
SOURCE_DIR="${LAYA_NEWSMTSC_SOURCE:-$PROJECT_ROOT/third_party/NewsMTSC}"
SOURCE_REPOSITORY="https://github.com/fhamborg/NewsMTSC"
SOURCE_COMMIT="b9d9b79704ed1b35cecaf1d7c2343dc1bd734fb7"

# The pinned NewsSentiment release requires Python 3.8-3.11.
python_is_supported() {
    "$1" -c 'import sys; raise SystemExit(not (3, 8) <= sys.version_info[:2] < (3, 12))' \
        >/dev/null 2>&1
}

if [[ ! -d "$SOURCE_DIR/NewsSentiment" ]]; then
    mkdir -p "$(dirname -- "$SOURCE_DIR")"
    git clone "$SOURCE_REPOSITORY" "$SOURCE_DIR"
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$SOURCE_COMMIT"
    git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"
fi

if [[ -x "$VENV_DIR/bin/python" ]] && ! python_is_supported "$VENV_DIR/bin/python"; then
    echo "Replacing $VENV_DIR because it does not use Python 3.8-3.11."
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -z "$PYTHON_BIN" ]]; then
        for candidate in python3.11 python3.10 python3.9 python3.8 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
                PYTHON_BIN="$candidate"
                break
            fi
        done
    fi
    mkdir -p "$(dirname -- "$VENV_DIR")"
    if [[ -n "$PYTHON_BIN" ]]; then
        if ! python_is_supported "$PYTHON_BIN"; then
            echo "NewsMTSC requires Python 3.8-3.11; $PYTHON_BIN is not supported." >&2
            exit 1
        fi
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        # No compatible system interpreter (for example Ubuntu 24.04 ships 3.12):
        # let uv provision a standalone CPython 3.11 for this isolated environment.
        uv_bin="$(command -v uv 2>/dev/null || true)"
        if [[ -z "$uv_bin" && -x "$MAIN_VENV_DIR/bin/python" ]]; then
            "$MAIN_VENV_DIR/bin/python" -m pip install --quiet uv
            uv_bin="$MAIN_VENV_DIR/bin/uv"
        fi
        if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
            echo "NewsMTSC requires Python 3.8-3.11, and none was found." >&2
            echo "Install python3.11 or uv, or set NEWSMTSC_PYTHON_BIN, then rerun this script." >&2
            exit 1
        fi
        echo "No Python 3.8-3.11 found; provisioning CPython 3.11 with uv"
        "$uv_bin" venv --seed --python 3.11 "$VENV_DIR"
    fi
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install truststore
# torch<2.1 is built against NumPy 1.x.
"$PYTHON" -m pip install --editable "$SOURCE_DIR" "numpy<2"
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

# NewsSentiment loads its encoder from NewsSentiment/pretrained_models/<name>,
# which upstream does not ship; save the Hugging Face weights there once.
PRETRAINED_MODEL="${LAYA_NEWSMTSC_PRETRAINED_MODEL:-roberta-base}"
PRETRAINED_DIR="$SOURCE_DIR/NewsSentiment/pretrained_models/$PRETRAINED_MODEL"
if [[ ! -f "$PRETRAINED_DIR/config.json" ]]; then
    echo "Downloading the $PRETRAINED_MODEL encoder into $PRETRAINED_DIR"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$PYTHON" - "$PRETRAINED_MODEL" "$PRETRAINED_DIR" <<'PY'
import sys

from transformers import RobertaModel, RobertaTokenizer

name, target = sys.argv[1], sys.argv[2]
RobertaTokenizer.from_pretrained(name).save_pretrained(target)
RobertaModel.from_pretrained(name).save_pretrained(target)
PY
fi

# The application starts the worker with network downloads disabled, so load it
# once here to confirm the GRU-TSC checkpoint and encoder load from local files.
echo "Loading the NewsMTSC GRU-TSC checkpoint"
warmup_output="$(printf '{"command":"shutdown"}\n' \
    | env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$PYTHON" "$PROJECT_ROOT/scripts/newsmtsc_worker.py" 2>&1)" \
    || true
if [[ "$warmup_output" != *"ready=1"* ]]; then
    printf '%s\n' "$warmup_output" | tail -n 20 >&2
    echo "The NewsMTSC worker could not load its checkpoint." >&2
    exit 1
fi

printf 'NewsMTSC environment ready: %s\n' "$PYTHON"
