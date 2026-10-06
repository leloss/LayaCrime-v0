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

export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
export LAYA_DEVICE="${LAYA_DEVICE:-cuda}"
export LAYA_REQUIRE_CUDA="${LAYA_REQUIRE_CUDA:-true}"
export LAYA_ENABLE_FINE_TUNING="${LAYA_ENABLE_FINE_TUNING:-true}"
export LAYA_GGUF_MODELS_DIR="${LAYA_GGUF_MODELS_DIR:-$PROJECT_ROOT/models/gguf}"
DEFAULT_LLAMA_SERVER="$PROJECT_ROOT/third_party/llama.cpp/build/bin/llama-server"
LLAMA_CMAKE_CACHE="$PROJECT_ROOT/third_party/llama.cpp/build/CMakeCache.txt"
LLAMA_RUNTIME_STAMP="$PROJECT_ROOT/third_party/llama.cpp/build/laya-cuda-runtime.ok"
if [[ -z "${LAYA_LLAMA_SERVER:-}" && -x "$DEFAULT_LLAMA_SERVER" ]] \
    && [[ -f "$LLAMA_RUNTIME_STAMP" ]] \
    && grep -q '^GGML_CUDA:BOOL=ON$' "$LLAMA_CMAKE_CACHE" 2>/dev/null; then
    export LAYA_LLAMA_SERVER="$DEFAULT_LLAMA_SERVER"
elif [[ -z "${LAYA_LLAMA_SERVER:-}" && -x "$DEFAULT_LLAMA_SERVER" ]]; then
    echo "Ignoring project llama-server because it has no verified CUDA initialization stamp." >&2
    echo "Rebuild it with: $PROJECT_ROOT/scripts/setup_llama_cpp.sh" >&2
fi

cd "$PROJECT_ROOT"
echo "Starting the Laya UI at http://$LAYA_HOST:$LAYA_PORT/"
echo "Benchmark:   http://$LAYA_HOST:$LAYA_PORT/"
echo "Fine-tuning: http://$LAYA_HOST:$LAYA_PORT/fine-tuning"
if [[ -n "${LAYA_LLAMA_SERVER:-}" ]]; then
    echo "Local LLMs:  $LAYA_LLAMA_SERVER"
else
    echo "Local LLMs:  unavailable (run scripts/setup_llama_cpp.sh)"
fi
echo "Fine-tuning paths are restricted by LAYA_FINE_TUNING_ALLOWED_ROOTS (project root by default)."
if [[ "$LAYA_HOST" != "127.0.0.1" && "$LAYA_HOST" != "localhost" && "$LAYA_HOST" != "::1" ]]; then
    echo "WARNING: This application has no built-in authentication. Use an authenticated reverse proxy." >&2
fi
exec "$SERVER"
