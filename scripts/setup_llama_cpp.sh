#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "llama.cpp setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_CPP_SOURCE="${LLAMA_CPP_SOURCE:-$PROJECT_ROOT/third_party/llama.cpp}"
LLAMA_CPP_REPOSITORY="${LLAMA_CPP_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-bed0a8566}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"
LAYA_INSTALL_CUDA_TOOLKIT="${LAYA_INSTALL_CUDA_TOOLKIT:-true}"
RUNTIME_STAMP="$LLAMA_CPP_SOURCE/build/laya-cuda-runtime.ok"
LLAMA_SERVER="$LLAMA_CPP_SOURCE/build/bin/llama-server"
CMAKE_CACHE="$LLAMA_CPP_SOURCE/build/CMakeCache.txt"

stamp_value() {
    sed -n "s/^$1=//p" "$RUNTIME_STAMP" 2>/dev/null | head -n 1
}

source_matches_requested_revision() {
    command -v git >/dev/null 2>&1 || return 1
    [[ -d "$LLAMA_CPP_SOURCE/.git" ]] || return 1
    local current_revision requested_revision
    current_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse HEAD 2>/dev/null)" || return 1
    requested_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse "$LLAMA_CPP_REVISION^{commit}" 2>/dev/null)" \
        || return 1
    [[ "$current_revision" == "$requested_revision" ]]
}

runtime_is_reusable() {
    [[ -x "$LLAMA_SERVER" && -f "$CMAKE_CACHE" ]] || return 1
    if [[ "$(stamp_value revision)" != "$LLAMA_CPP_REVISION" ]]; then
        source_matches_requested_revision || return 1
        local requested_revision binary_version
        requested_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse "$LLAMA_CPP_REVISION^{commit}")"
        binary_version="$($LLAMA_SERVER --version 2>&1)" || return 1
        [[ "$binary_version" == *"${requested_revision:0:7}"* ]] || return 1
    fi
    grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE" || return 1
    grep -Eq "^CMAKE_CUDA_ARCHITECTURES[^=]*=${CUDA_ARCHITECTURES}$" "$CMAKE_CACHE" \
        || return 1
    local device_output cuda_device
    device_output="$($LLAMA_SERVER --list-devices 2>&1)" || return 1
    cuda_device="$(printf '%s\n' "$device_output" \
        | sed -nE 's/^[[:space:]]*(CUDA[0-9]+):.*/\1/p' | head -n 1)"
    [[ -n "$cuda_device" ]] || return 1
    if [[ ! -f "$RUNTIME_STAMP" || "$(stamp_value revision)" != "$LLAMA_CPP_REVISION" ]]; then
        printf 'revision=%s\narchitecture=%s\ndevice=%s\nadopted=true\n' \
            "$LLAMA_CPP_REVISION" "$CUDA_ARCHITECTURES" "$cuda_device" >"$RUNTIME_STAMP"
    fi
}

if runtime_is_reusable; then
    echo "Verified llama.cpp CUDA runtime is already installed; nothing to rebuild."
    "$LLAMA_SERVER" --version
    printf 'Using existing runtime: %s\n' "$LLAMA_SERVER"
    exit 0
fi

for command in git cmake c++ make; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        echo "On Ubuntu, install the host build tools with:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y build-essential git cmake" >&2
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Required NVIDIA driver utility not found: nvidia-smi" >&2
    exit 1
fi
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
driver_major="${driver_version%%.*}"
if [[ ! "$driver_major" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the NVIDIA driver version from nvidia-smi." >&2
    exit 1
fi

nvcc_major() {
    "$1" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n 1
}

nvcc_is_compatible() {
    local compiler="$1" toolkit_major minimum_driver
    toolkit_major="$(nvcc_major "$compiler")"
    case "$toolkit_major" in
        13) minimum_driver=580 ;;
        12) minimum_driver=525 ;;
        *) return 1 ;;
    esac
    (( driver_major >= minimum_driver ))
}

find_nvcc() {
    local candidate resolved seen=""
    if [[ -n "${CUDACXX:-}" ]]; then
        resolved="$(command -v "$CUDACXX" 2>/dev/null || true)"
        if [[ -z "$resolved" || ! -x "$resolved" ]]; then
            echo "Configured CUDACXX is not executable: $CUDACXX" >&2
            return 1
        fi
        if ! nvcc_is_compatible "$resolved"; then
            echo "Configured CUDA compiler $resolved is unsupported; current llama.cpp " >&2
            echo "requires CUDA 12+ and a compatible NVIDIA driver (found $driver_version)." >&2
            return 1
        fi
        printf '%s\n' "$resolved"
        return
    fi
    for candidate in "$(command -v nvcc 2>/dev/null || true)" \
        /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        resolved="$(readlink -f "$candidate")"
        [[ " $seen " == *" $resolved "* ]] && continue
        seen+=" $resolved"
        if nvcc_is_compatible "$resolved"; then
            printf '%s\n' "$resolved"
            return
        fi
        echo "Skipping $resolved: current llama.cpp requires CUDA 12+ and a compatible driver." >&2
    done
}

CUDACXX="$(find_nvcc || true)"
if [[ -z "$CUDACXX" ]]; then
    toolkit_package="cuda-toolkit-12-1"
    if (( driver_major >= 550 )); then
        toolkit_package="cuda-toolkit-12-4"
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
    fi
    if [[ "$LAYA_INSTALL_CUDA_TOOLKIT" == "true" \
        && "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
        echo "Installing $toolkit_package for NVIDIA driver $driver_version"
        if (( EUID == 0 )); then
            root_command=()
        elif command -v sudo >/dev/null 2>&1; then
            root_command=(sudo)
        else
            echo "CUDA toolkit installation requires root access or sudo." >&2
            exit 1
        fi
        wget -qO /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        "${root_command[@]}" dpkg -i /tmp/cuda-keyring.deb
        "${root_command[@]}" apt-get update
        "${root_command[@]}" apt-get install -y "$toolkit_package"
        CUDACXX="$(find_nvcc || true)"
    fi

    if [[ -z "$CUDACXX" && "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
        echo "No installed CUDA compiler is compatible with NVIDIA driver $driver_version." >&2
        cat >&2 <<EOF
Install the NVIDIA CUDA toolkit without replacing the working GPU driver:
  wget -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y build-essential ${toolkit_package}

Then rerun this script. It automatically discovers nvcc under /usr/local/cuda*.
EOF
    elif [[ -z "$CUDACXX" ]]; then
        echo "No installed CUDA compiler is compatible with NVIDIA driver $driver_version." >&2
        echo "Install a CUDA toolkit compatible with the NVIDIA driver; a driver alone is insufficient." >&2
        echo "See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/" >&2
    fi
    if [[ -z "$CUDACXX" ]]; then
        exit 1
    fi
fi

export CUDACXX
export PATH="$(dirname -- "$CUDACXX"):$PATH"
CUDA_TOOLKIT_ROOT="$(cd -- "$(dirname -- "$CUDACXX")/.." && pwd)"
echo "Using CUDA compiler: $CUDACXX"
echo "Using CUDA toolkit root: $CUDA_TOOLKIT_ROOT"
"$CUDACXX" --version

smoke_dir="$(mktemp -d)"
cat >"$smoke_dir/cuda-smoke.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count < 1) {
        std::fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    std::printf("CUDA initialization succeeded with %d GPU(s).\n", count);
    return 0;
}
EOF
if ! "$CUDACXX" -arch="sm_$CUDA_ARCHITECTURES" "$smoke_dir/cuda-smoke.cu" \
    -o "$smoke_dir/cuda-smoke"; then
    rm -rf "$smoke_dir"
    echo "The selected CUDA toolkit could not compile the GPU initialization probe." >&2
    exit 1
fi
if ! "$smoke_dir/cuda-smoke"; then
    rm -rf "$smoke_dir"
    echo "CUDA cannot initialize with driver $driver_version and compiler $CUDACXX." >&2
    echo "Install a compatible CUDA 12 toolkit, then rerun this script." >&2
    exit 1
fi
rm -rf "$smoke_dir"

if [[ ! -d "$LLAMA_CPP_SOURCE/.git" ]]; then
    mkdir -p "$(dirname -- "$LLAMA_CPP_SOURCE")"
    git clone "$LLAMA_CPP_REPOSITORY" "$LLAMA_CPP_SOURCE"
fi

if [[ ! "$LLAMA_CPP_REVISION" =~ ^[0-9a-fA-F]{7,40}$ ]] \
    || ! git -C "$LLAMA_CPP_SOURCE" cat-file -e "$LLAMA_CPP_REVISION^{commit}" 2>/dev/null; then
    git -C "$LLAMA_CPP_SOURCE" fetch --tags origin
fi
previous_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse HEAD)"
git -C "$LLAMA_CPP_SOURCE" checkout --detach "$LLAMA_CPP_REVISION"
selected_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse HEAD)"

clean_reason=""
if [[ -f "$CMAKE_CACHE" ]]; then
    if [[ "$previous_revision" != "$selected_revision" ]]; then
        clean_reason="llama.cpp revision changed"
    elif ! grep -Eq "^CMAKE_CUDA_COMPILER[^=]*=${CUDACXX}$" "$CMAKE_CACHE"; then
        clean_reason="CUDA compiler changed"
    elif ! grep -Eq "^CMAKE_CUDA_ARCHITECTURES[^=]*=${CUDA_ARCHITECTURES}$" "$CMAKE_CACHE"; then
        clean_reason="CUDA architecture changed"
    elif ! grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE"; then
        clean_reason="existing build is not CUDA-enabled"
    fi
fi
if [[ -n "$clean_reason" ]]; then
    echo "Cleaning incompatible llama.cpp build: $clean_reason"
    rm -rf "$LLAMA_CPP_SOURCE/build"
elif [[ -f "$CMAKE_CACHE" ]]; then
    echo "Resuming compatible llama.cpp build incrementally."
fi
cmake -S "$LLAMA_CPP_SOURCE" -B "$LLAMA_CPP_SOURCE/build" \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
cmake --build "$LLAMA_CPP_SOURCE/build" --config Release --target llama-server --parallel "$(nproc)"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "Build completed without the expected executable: $LLAMA_SERVER" >&2
    exit 1
fi

"$LLAMA_SERVER" --version
device_output="$($LLAMA_SERVER --list-devices 2>&1)"
printf '%s\n' "$device_output"
cuda_device="$(printf '%s\n' "$device_output" | sed -nE 's/^[[:space:]]*(CUDA[0-9]+):.*/\1/p' | head -n 1)"
if [[ -z "$cuda_device" ]]; then
    echo "Built llama-server does not enumerate a CUDA device." >&2
    exit 1
fi
printf 'driver=%s\nnvcc=%s\nrevision=%s\narchitecture=%s\ndevice=%s\n' \
    "$driver_version" "$CUDACXX" "$LLAMA_CPP_REVISION" "$CUDA_ARCHITECTURES" \
    "$cuda_device" >"$RUNTIME_STAMP"
printf '\nllama.cpp is ready. The UI launcher will use:\n  LAYA_LLAMA_SERVER=%q\n  LAYA_GGUF_MODELS_DIR=%q\n' \
    "$LLAMA_SERVER" "$PROJECT_ROOT/models/gguf"