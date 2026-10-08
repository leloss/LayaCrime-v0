#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Laya UI failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
SERVER="$VENV_DIR/bin/laya-adverse-media"

if [[ ! -x "$SERVER" ]]; then
    echo "Laya environment not found at $VENV_DIR." >&2
    echo "Run $PROJECT_ROOT/scripts/setup_environment.sh first." >&2
    exit 1
fi

if [[ -z "${LAYA_DEVICE:-}" ]]; then
    LAYA_DEVICE="$("$VENV_DIR/bin/python" -c '
import torch
mps = getattr(torch.backends, "mps", None)
print("cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu")
' 2>/dev/null)" || LAYA_DEVICE="cpu"
fi

export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
export LAYA_DEVICE
export LAYA_ENABLE_FINE_TUNING="${LAYA_ENABLE_FINE_TUNING:-true}"
export LAYA_GGUF_MODELS_DIR="${LAYA_GGUF_MODELS_DIR:-$PROJECT_ROOT/models/gguf}"
DEFAULT_LLAMA_SERVER="$PROJECT_ROOT/third_party/llama.cpp/build/bin/llama-server"
LLAMA_CMAKE_CACHE="$PROJECT_ROOT/third_party/llama.cpp/build/CMakeCache.txt"
LLAMA_RUNTIME_STAMP="$PROJECT_ROOT/third_party/llama.cpp/build/laya-cuda-runtime.ok"
# Stamps written before the backend line existed always describe CUDA builds.
llama_backend="$(sed -n 's/^backend=//p' "$LLAMA_RUNTIME_STAMP" 2>/dev/null | head -n 1)" || true
llama_backend="${llama_backend:-cuda}"
if [[ "$llama_backend" == "cuda" ]] \
    && ! grep -q '^GGML_CUDA:BOOL=ON$' "$LLAMA_CMAKE_CACHE" 2>/dev/null; then
    llama_backend=""
fi
if [[ -z "${LAYA_LLAMA_SERVER:-}" && -x "$DEFAULT_LLAMA_SERVER" ]] \
    && [[ -f "$LLAMA_RUNTIME_STAMP" && -n "$llama_backend" ]]; then
    export LAYA_LLAMA_SERVER="$DEFAULT_LLAMA_SERVER"
elif [[ -z "${LAYA_LLAMA_SERVER:-}" && -x "$DEFAULT_LLAMA_SERVER" ]]; then
    echo "Ignoring project llama-server because it has no verified initialization stamp." >&2
    echo "Rebuild it with: $PROJECT_ROOT/scripts/setup_llama_cpp.sh" >&2
fi
# Only a CUDA llama.cpp build can satisfy the GPU offload check.
if [[ -n "${LAYA_LLAMA_SERVER:-}" && "$LAYA_LLAMA_SERVER" == "$DEFAULT_LLAMA_SERVER" \
    && "$llama_backend" == "cuda" ]]; then
    default_require_cuda=true
else
    default_require_cuda=false
fi
export LAYA_REQUIRE_CUDA="${LAYA_REQUIRE_CUDA:-$default_require_cuda}"

cd "$PROJECT_ROOT"
echo "Starting the Laya UI at http://$LAYA_HOST:$LAYA_PORT/"
echo "Benchmark:   http://$LAYA_HOST:$LAYA_PORT/"
echo "Fine-tuning: http://$LAYA_HOST:$LAYA_PORT/fine-tuning"
echo "Laya device: $LAYA_DEVICE"
if [[ -n "${LAYA_LLAMA_SERVER:-}" ]]; then
    if [[ "$LAYA_LLAMA_SERVER" != "$DEFAULT_LLAMA_SERVER" ]]; then llama_backend="external"; fi
    echo "Local LLMs:  $LAYA_LLAMA_SERVER ($llama_backend)"
else
    echo "Local LLMs:  unavailable (run scripts/setup_llama_cpp.sh)"
fi
if [[ "$LAYA_DEVICE" != "cuda" && "$LAYA_ENABLE_FINE_TUNING" == "true" ]]; then
    echo "Fine-tuning jobs require an NVIDIA CUDA GPU and will not start on this host." >&2
fi
echo "Fine-tuning paths are restricted by LAYA_FINE_TUNING_ALLOWED_ROOTS (project root by default)."
if [[ "$LAYA_HOST" != "127.0.0.1" && "$LAYA_HOST" != "localhost" && "$LAYA_HOST" != "::1" ]]; then
    echo "WARNING: This application has no built-in authentication. Use an authenticated reverse proxy." >&2
fi
exec "$SERVER"
