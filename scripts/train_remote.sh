#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Remote training setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-gpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PIP_BOOTSTRAP_URL="${PIP_BOOTSTRAP_URL:-https://bootstrap.pypa.io/get-pip.py}"
HF_MODEL_ID="${HF_MODEL_ID:-convaiinnovations/laya}"
HF_REVISION="${HF_REVISION:-main}"
LAYA_REPOSITORY="${LAYA_REPOSITORY:-https://github.com/NandhaKishorM/laya.git}"
LAYA_VERSION="${LAYA_VERSION:-v0.3.10}"
LAYA_SOURCE="${LAYA_SOURCE:-$PROJECT_ROOT/third_party/laya}"
PREPARED_DIR="${PREPARED_DIR:-$PROJECT_ROOT/artifacts/adverse-media-data-azure}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/laya-adverse-media-azure}"
RESUME_FROM="${RESUME_FROM:-}"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$PROJECT_ROOT/scripts/fine_tune_laya.py" ]]; then
    echo "Training script not found: $PROJECT_ROOT/scripts/fine_tune_laya.py" >&2
    echo "Upload the repository's scripts directory or set PROJECT_ROOT to its parent directory." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), "Python 3.10-3.13 is required"'

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
TORCHRUN="$VENV_DIR/bin/torchrun"

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "Bootstrapping pip in the virtual environment"
    if ! "$PYTHON" -m ensurepip --upgrade; then
        bootstrap_file="$(mktemp)"
        trap 'rm -f "${bootstrap_file:-}"' EXIT
        PIP_BOOTSTRAP_URL="$PIP_BOOTSTRAP_URL" "$PYTHON" - "$bootstrap_file" <<'PY'
import os
import sys
import urllib.request

urllib.request.urlretrieve(os.environ["PIP_BOOTSTRAP_URL"], sys.argv[1])
PY
        "$PYTHON" "$bootstrap_file"
        rm -f "$bootstrap_file"
        trap - EXIT
    fi
fi

if [[ -z "$TORCH_INDEX_URL" ]]; then
    driver_version=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    elif [[ -r /proc/driver/nvidia/version ]]; then
        driver_version="$(grep -oE 'Kernel Module[[:space:]]+[0-9.]+' /proc/driver/nvidia/version | awk '{print $3}' | head -n 1)"
    fi
    if [[ -z "$driver_version" ]]; then
        echo "No working NVIDIA driver was detected." >&2
        echo "Confirm that this is a GPU server, install one coherent NVIDIA driver version, reboot, and verify nvidia-smi before retrying." >&2
        echo "TORCH_INDEX_URL cannot make a GPU visible when the host driver is missing." >&2
        exit 1
    fi
    driver_major="${driver_version%%.*}"
    if ! [[ "$driver_major" =~ ^[0-9]+$ ]]; then
        echo "Could not determine the NVIDIA driver version: $driver_version" >&2
        exit 1
    elif (( driver_major >= 550 )); then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    elif (( driver_major >= 525 )); then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    else
        echo "NVIDIA driver $driver_version is too old; version 525 or newer is required." >&2
        exit 1
    fi
    echo "NVIDIA driver $driver_version selected $TORCH_INDEX_URL"
fi

echo "Installing Python and CUDA dependencies"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install torch --index-url "$TORCH_INDEX_URL"

if [[ ! -f "$LAYA_SOURCE/pyproject.toml" ]]; then
    echo "Cloning Laya $LAYA_VERSION"
    mkdir -p "$(dirname -- "$LAYA_SOURCE")"
    git clone --depth 1 --branch "$LAYA_VERSION" "$LAYA_REPOSITORY" "$LAYA_SOURCE"
fi

"$PYTHON" -m pip install --no-deps --editable "$LAYA_SOURCE"
if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    "$PYTHON" -m pip install --editable "${PROJECT_ROOT}[train]"
else
    echo "pyproject.toml not found; installing the minimal standalone training dependencies"
    "$PYTHON" -m pip install \
        "huggingface-hub>=0.20" \
        "numpy>=1.26,<3" \
        "safetensors>=0.4,<1" \
        "transformers>=4.48,<5"
fi

echo "Downloading $HF_MODEL_ID at revision $HF_REVISION"
export PROJECT_ROOT HF_MODEL_ID HF_REVISION
"$PYTHON" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    revision=os.environ["HF_REVISION"],
    cache_dir=Path(os.environ["PROJECT_ROOT"]) / "models",
    allow_patterns=[
        "encoder/**",
        "model.safetensors",
        "rl_agent_config.json",
        "tokenizer/**",
    ],
)
print(f"English checkpoint ready: {snapshot}")
PY

required_prepared_files=(train.pt calibration.pt test.pt manifest.json split-assignments.jsonl)
prepared=true
for filename in "${required_prepared_files[@]}"; do
    if [[ ! -f "$PREPARED_DIR/$filename" ]]; then
        prepared=false
        break
    fi
done

if [[ "$prepared" == false ]]; then
    corpus="$PROJECT_ROOT/artifacts/independent-annotations/blind_articles.jsonl"
    annotations="$PROJECT_ROOT/artifacts/independent-annotations/batches/azure.annotations.jsonl"
    if [[ ! -f "$corpus" || ! -f "$annotations" ]]; then
        echo "Prepared tensors and raw Azure inputs are both incomplete." >&2
        echo "Upload artifacts/adverse-media-data-azure or the blind corpus and Azure ledger." >&2
        exit 1
    fi
    echo "Preparing consolidated Azure training tensors"
    "$PYTHON" scripts/fine_tune_laya.py prepare --output-dir "$PREPARED_DIR"
else
    echo "Using prepared tensors: $PREPARED_DIR"
fi

CUDA_SUMMARY="$("$PYTHON" - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
PY
)"
echo "CUDA environment: $CUDA_SUMMARY"

GPU_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < 1 )); then
    echo "No CUDA GPU is visible to PyTorch. Check the NVIDIA driver and CUDA wheel." >&2
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || (( NPROC_PER_NODE > GPU_COUNT )); then
    echo "NPROC_PER_NODE must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
    exit 1
fi

if [[ -f "$OUTPUT_DIR/model.safetensors" && -z "$RESUME_FROM" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "A completed model already exists at $OUTPUT_DIR." >&2
    echo "Set OUTPUT_DIR to a new path, RESUME_FROM to a checkpoint, or ALLOW_OVERWRITE=1." >&2
    exit 1
fi

train_args=(
    scripts/fine_tune_laya.py train
    --prepared-dir "$PREPARED_DIR"
    --output-dir "$OUTPUT_DIR"
)
if [[ -n "$RESUME_FROM" ]]; then
    train_args+=(--resume-from "$RESUME_FROM")
fi
train_args+=("$@")

echo "Starting fine-tuning with $NPROC_PER_NODE GPU process(es)"
if (( NPROC_PER_NODE == 1 )); then
    "$PYTHON" "${train_args[@]}"
else
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${train_args[@]}"
fi