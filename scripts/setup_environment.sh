#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Environment setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
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
LAYA_INSTALL_CUDA_TOOLKIT="${LAYA_INSTALL_CUDA_TOOLKIT:-true}"
# Set to false to skip building llama.cpp; local GGUF language models are then
# unavailable, but decision models, hosted language models, and datasets work.
LAYA_SETUP_LLAMA_CPP="${LAYA_SETUP_LLAMA_CPP:-true}"
# Set to false to skip the Khandpur, Tartu, and NewsMTSC academic baselines.
LAYA_SETUP_ACADEMIC_MODELS="${LAYA_SETUP_ACADEMIC_MODELS:-true}"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

python_is_supported() {
    "$1" -c 'import sys; raise SystemExit(not (3, 10) <= sys.version_info[:2] < (3, 14))' \
        >/dev/null 2>&1
}

if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.10-3.13 was not found. Install one, or set PYTHON_BIN to its path." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! python_is_supported "$PYTHON_BIN"; then
    echo "$PYTHON_BIN is not Python 3.10-3.13; set PYTHON_BIN to a supported interpreter." >&2
    exit 1
fi

if [[ "$LAYA_AUTO_INSTALL_SYSTEM_DEPS" == "true" && -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" == "ubuntu" || "${ID:-}" == "debian" || " ${ID_LIKE:-} " == *" debian "* ]]; then
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
    else
        echo "Automatic system package installation supports Debian and Ubuntu only." >&2
        echo "Make sure git, cmake, a C++ compiler, make, and wget are installed." >&2
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

driver_version=""
gpu_compute_major=""
if [[ "$(uname -s)" != "Darwin" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
            | head -n 1 | tr -d '[:space:]')" || driver_version=""
        gpu_compute_major="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | sed -nE 's/^[[:space:]]*([0-9]+)\..*/\1/p' | sort -n | tail -n 1)" || gpu_compute_major=""
    elif [[ -r /proc/driver/nvidia/version ]]; then
        driver_version="$(grep -oE 'Kernel Module[[:space:]]+[0-9.]+' /proc/driver/nvidia/version \
            | awk '{print $3}' | head -n 1)" || driver_version=""
    fi
fi
driver_major="${driver_version%%.*}"
[[ "$driver_major" =~ ^[0-9]+$ ]] || driver_major=""

if [[ -z "$TORCH_INDEX_URL" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "macOS detected; installing the default PyTorch build (CPU and Apple MPS)."
    elif [[ -z "$driver_major" ]]; then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        echo "No working NVIDIA driver was detected; installing the CPU build of PyTorch."
        echo "Inference and benchmarks run on CPU; fine-tuning requires an NVIDIA GPU."
    elif (( driver_major < 525 )); then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        echo "NVIDIA driver $driver_version is older than 525; installing the CPU build of PyTorch." >&2
        echo "Upgrade the driver and rerun this script to use the GPU." >&2
    elif [[ -n "$gpu_compute_major" ]] && (( gpu_compute_major >= 10 && driver_major >= 570 )); then
        # Blackwell GPUs are not supported by the cu124 and cu121 builds.
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    elif (( driver_major >= 550 )); then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    else
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    fi
    if [[ -n "$driver_major" && -n "$TORCH_INDEX_URL" ]]; then
        echo "NVIDIA driver $driver_version selected $TORCH_INDEX_URL"
    fi
fi

if [[ ! -f "$LAYA_SOURCE/pyproject.toml" ]]; then
    echo "Cloning Laya $LAYA_VERSION"
    mkdir -p "$(dirname -- "$LAYA_SOURCE")"
    git clone --depth 1 --branch "$LAYA_VERSION" "$LAYA_REPOSITORY" "$LAYA_SOURCE"
fi

echo "Installing Python dependencies"
"$PYTHON" -m pip install --upgrade pip wheel
if [[ -n "$TORCH_INDEX_URL" ]]; then
    "$PYTHON" -m pip install torch --index-url "$TORCH_INDEX_URL"
else
    "$PYTHON" -m pip install torch
fi
"$PYTHON" -m pip install --no-deps --editable "$LAYA_SOURCE"
"$PYTHON" -m pip install --editable "${PROJECT_ROOT}[train]"

llama_cpp_ready=false
if [[ "$LAYA_SETUP_LLAMA_CPP" == "true" ]]; then
    echo "Building and validating the local language-model runtime (llama.cpp)"
    if LAYA_INSTALL_CUDA_TOOLKIT="$LAYA_INSTALL_CUDA_TOOLKIT" \
        bash "$PROJECT_ROOT/scripts/setup_llama_cpp.sh"; then
        llama_cpp_ready=true
    else
        echo "WARNING: llama.cpp could not be set up; local GGUF language models are unavailable." >&2
        echo "Everything else will still be installed. Retry later with:" >&2
        echo "  $PROJECT_ROOT/scripts/setup_llama_cpp.sh" >&2
    fi
fi

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

academic_ready=true
if [[ "$LAYA_SETUP_ACADEMIC_MODELS" == "true" ]]; then
    echo "Setting up the academic baseline models"
    if ! VENV_DIR="$VENV_DIR" bash "$PROJECT_ROOT/scripts/setup_academic_models.sh"; then
        academic_ready=false
        echo "WARNING: some academic baseline models could not be set up." >&2
        echo "Retry later with: $PROJECT_ROOT/scripts/setup_academic_models.sh" >&2
    fi
fi

DRIVER_VERSION="$driver_version" "$PYTHON" - <<'PY'
import json
import os

import torch

mps = getattr(torch.backends, "mps", None)
print("Compute environment:", json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": bool(mps and mps.is_available()),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
if not torch.cuda.is_available():
    if os.environ.get("DRIVER_VERSION"):
        print(
            "WARNING: an NVIDIA driver is installed but PyTorch cannot use the GPU. "
            "Laya will run on CPU. Check nvidia-smi, or set TORCH_INDEX_URL to a "
            "PyTorch CUDA build that matches the driver and rerun this script."
        )
    print("Laya inference and benchmarks will run on CPU; fine-tuning requires a CUDA GPU.")
PY

if [[ "$llama_cpp_ready" != "true" ]]; then
    echo "Local GGUF language models: unavailable (run scripts/setup_llama_cpp.sh to retry)."
fi
if [[ "$academic_ready" != "true" ]]; then
    echo "Academic baselines: incomplete (run scripts/setup_academic_models.sh to retry)."
fi

printf '\nEnvironment ready. Start the benchmark and fine-tuning UI with:\n  %q\n' \
    "$PROJECT_ROOT/scripts/run_ui.sh"
