#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Remote environment setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PIP_BOOTSTRAP_URL="${PIP_BOOTSTRAP_URL:-https://bootstrap.pypa.io/get-pip.py}"
LAYA_HF_REPO_ID="${LAYA_HF_REPO_ID:-convaiinnovations/laya}"
LAYA_HF_REVISION="${LAYA_HF_REVISION:-main}"
LAYACRIME_HF_REPO_ID="${LAYACRIME_HF_REPO_ID:-leloss/LayaCrime-v0}"
LAYACRIME_HF_REVISION="${LAYACRIME_HF_REVISION:-main}"
LAYA_REPOSITORY="${LAYA_REPOSITORY:-https://github.com/NandhaKishorM/laya.git}"
LAYA_VERSION="${LAYA_VERSION:-v0.3.10}"
LAYA_SOURCE="${LAYA_SOURCE:-$PROJECT_ROOT/third_party/laya}"
LAYA_AUTO_INSTALL_SYSTEM_DEPS="${LAYA_AUTO_INSTALL_SYSTEM_DEPS:-true}"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), "Python 3.10-3.13 is required"'

if [[ "$LAYA_AUTO_INSTALL_SYSTEM_DEPS" == "true" && -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" == "ubuntu" ]]; then
        system_packages=(build-essential git cmake wget ca-certificates python3-venv)
        if (( EUID == 0 )); then
            apt-get update
            apt-get install -y "${system_packages[@]}"
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y "${system_packages[@]}"
        else
            echo "System dependencies require root access or sudo: ${system_packages[*]}" >&2
            exit 1
        fi
    fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment: $VENV_DIR"
    mkdir -p "$(dirname -- "$VENV_DIR")"
    if ! "$PYTHON_BIN" -m venv --clear --copies "$VENV_DIR"; then
        echo "Could not create a Python virtual environment." >&2
        echo "On Ubuntu, install venv support for $PYTHON_BIN, for example:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y python3-venv" >&2
        exit 1
    fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Virtual environment creation did not produce $VENV_DIR/bin/python." >&2
    echo "Remove $VENV_DIR, install python3-venv, and retry." >&2
    exit 1
fi

PYTHON="$VENV_DIR/bin/python"

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
        echo "Install a coherent NVIDIA driver, reboot, and verify nvidia-smi before retrying." >&2
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

if [[ ! -f "$LAYA_SOURCE/pyproject.toml" ]]; then
    echo "Cloning Laya $LAYA_VERSION"
    mkdir -p "$(dirname -- "$LAYA_SOURCE")"
    git clone --depth 1 --branch "$LAYA_VERSION" "$LAYA_REPOSITORY" "$LAYA_SOURCE"
fi

echo "Installing Python and CUDA dependencies"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install torch --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install --no-deps --editable "$LAYA_SOURCE"
"$PYTHON" -m pip install --editable "${PROJECT_ROOT}[train]"

echo "Building and validating the local CUDA language-model runtime"
LAYA_INSTALL_CUDA_TOOLKIT=true bash "$PROJECT_ROOT/scripts/setup_llama_cpp.sh"

echo "Downloading Laya from https://huggingface.co/$LAYA_HF_REPO_ID"
"$PYTHON" scripts/download_huggingface_snapshot.py \
    --repo-id "$LAYA_HF_REPO_ID" \
    --revision "$LAYA_HF_REVISION" \
    --cache-dir "$PROJECT_ROOT/models" \
    --allow-pattern 'encoder/**' \
    --allow-pattern 'model.safetensors' \
    --allow-pattern 'rl_agent_config.json' \
    --allow-pattern 'tokenizer/**' \
    --allow-pattern 'multilingual/**' \
    --allow-pattern 'typed-decisions/**'

echo "Downloading LayaCrime from https://huggingface.co/$LAYACRIME_HF_REPO_ID"
"$PYTHON" scripts/download_huggingface_snapshot.py \
    --repo-id "$LAYACRIME_HF_REPO_ID" \
    --revision "$LAYACRIME_HF_REVISION" \
    --local-dir "$PROJECT_ROOT/models/fine-tuned/layacrime-public" \
    --allow-pattern 'encoder/**' \
    --allow-pattern 'model.safetensors' \
    --allow-pattern 'rl_agent_config.json' \
    --allow-pattern 'tokenizer/**' \
    --allow-pattern 'training_report.json' \
    --allow-pattern 'data_manifest.json' \
    --allow-pattern 'README.md'

"$PYTHON" - <<'PY'
import json
import torch

print("CUDA environment:", json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible to PyTorch.")
PY

printf '\nEnvironment ready. Start the benchmark and fine-tuning UI with:\n  %q\n' \
    "$PROJECT_ROOT/scripts/run_ui.sh"
