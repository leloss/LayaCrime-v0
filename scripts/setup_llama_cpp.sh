#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "llama.cpp setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_CPP_SOURCE="${LLAMA_CPP_SOURCE:-$PROJECT_ROOT/third_party/llama.cpp}"
LLAMA_CPP_REPOSITORY="${LLAMA_CPP_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-bed0a8566}"
# auto: build for CUDA when an NVIDIA GPU and a compatible CUDA 12+ toolkit are
# available, otherwise fall back to a CPU build (Metal on macOS).
# cuda: fail instead of falling back. cpu: never build for CUDA.
LAYA_LLAMA_BACKEND="${LAYA_LLAMA_BACKEND:-auto}"
# Empty means detect from the installed GPUs (the Tesla T4 reports 75); a GPU
# whose architecture cannot be queried falls back to 75.
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
LAYA_INSTALL_CUDA_TOOLKIT="${LAYA_INSTALL_CUDA_TOOLKIT:-true}"
RUNTIME_STAMP="$LLAMA_CPP_SOURCE/build/laya-cuda-runtime.ok"
LLAMA_SERVER="$LLAMA_CPP_SOURCE/build/bin/llama-server"
CMAKE_CACHE="$LLAMA_CPP_SOURCE/build/CMakeCache.txt"

case "$LAYA_LLAMA_BACKEND" in
    auto | cuda | cpu) ;;
    *)
        echo "LAYA_LLAMA_BACKEND must be auto, cuda, or cpu (got $LAYA_LLAMA_BACKEND)." >&2
        exit 1
        ;;
esac

if [[ "$(uname -s)" == "Darwin" ]]; then
    NATIVE_BACKEND="metal"
else
    NATIVE_BACKEND="cpu"
fi

cpu_count() {
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2
}

stamp_value() {
    sed -n "s/^$1=//p" "$RUNTIME_STAMP" 2>/dev/null | head -n 1
}

# Stamps written before the backend line existed always describe CUDA builds.
stamp_backend() {
    local backend
    backend="$(stamp_value backend)"
    printf '%s\n' "${backend:-cuda}"
}

driver_version=""
driver_major=""
if [[ "$LAYA_LLAMA_BACKEND" != "cpu" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d '[:space:]')" || driver_version=""
    driver_major="${driver_version%%.*}"
    if [[ ! "$driver_major" =~ ^[0-9]+$ ]]; then
        driver_version=""
        driver_major=""
    fi
fi

if [[ -n "$driver_major" && -z "$CUDA_ARCHITECTURES" ]]; then
    CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d '. ' | grep -E '^[0-9]+$' | sort -un | paste -sd ';' -)" || CUDA_ARCHITECTURES=""
fi
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"

numeric_cuda_architectures() {
    local arch
    for arch in ${CUDA_ARCHITECTURES//;/ }; do
        [[ "$arch" =~ ^[0-9]+ ]] && printf '%s\n' "${BASH_REMATCH[0]}"
    done
    return 0
}

highest_cuda_architecture() {
    numeric_cuda_architectures | sort -n | tail -n 1
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

first_cuda_device() {
    printf '%s\n' "$1" | sed -nE 's/^[[:space:]]*(CUDA[0-9]+):.*/\1/p' | head -n 1
}

runtime_is_reusable() {
    [[ -x "$LLAMA_SERVER" && -f "$CMAKE_CACHE" ]] || return 1
    local existing_backend
    if [[ -f "$RUNTIME_STAMP" ]]; then
        existing_backend="$(stamp_backend)"
    elif grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE"; then
        existing_backend="cuda"
    else
        return 1
    fi
    case "$LAYA_LLAMA_BACKEND:$existing_backend" in
        cpu:cuda | cuda:cpu | cuda:metal) return 1 ;;
    esac
    # Retry CUDA when a GPU is present; an unchanged CPU build resumes as a no-op.
    if [[ "$existing_backend" != "cuda" && -n "$driver_major" ]]; then
        return 1
    fi
    if [[ "$(stamp_value revision)" != "$LLAMA_CPP_REVISION" ]]; then
        source_matches_requested_revision || return 1
        local requested_revision binary_version
        requested_revision="$(git -C "$LLAMA_CPP_SOURCE" rev-parse "$LLAMA_CPP_REVISION^{commit}")"
        binary_version="$("$LLAMA_SERVER" --version 2>&1)" || return 1
        [[ "$binary_version" == *"${requested_revision:0:7}"* ]] || return 1
    fi
    local device_output cuda_device=""
    device_output="$("$LLAMA_SERVER" --list-devices 2>&1)" || return 1
    if [[ "$existing_backend" == "cuda" ]]; then
        grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE" || return 1
        grep -Eq "^CMAKE_CUDA_ARCHITECTURES[^=]*=${CUDA_ARCHITECTURES}$" "$CMAKE_CACHE" \
            || return 1
        cuda_device="$(first_cuda_device "$device_output")"
        [[ -n "$cuda_device" ]] || return 1
    elif grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE"; then
        return 1
    fi
    if [[ ! -f "$RUNTIME_STAMP" || "$(stamp_value revision)" != "$LLAMA_CPP_REVISION" ]]; then
        printf 'backend=%s\nrevision=%s\narchitecture=%s\ndevice=%s\nadopted=true\n' \
            "$existing_backend" "$LLAMA_CPP_REVISION" "$CUDA_ARCHITECTURES" "$cuda_device" \
            >"$RUNTIME_STAMP"
    fi
    REUSED_BACKEND="$existing_backend"
}

if runtime_is_reusable; then
    echo "Verified llama.cpp $REUSED_BACKEND runtime is already installed; nothing to rebuild."
    "$LLAMA_SERVER" --version
    printf 'Using existing runtime: %s\n' "$LLAMA_SERVER"
    exit 0
fi

for command in git cmake c++ make; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        echo "On Ubuntu, install the host build tools with:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y build-essential git cmake" >&2
        echo "On macOS, install the Xcode command line tools and CMake." >&2
        exit 1
    fi
done

nvcc_major() {
    "$1" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n 1
}

nvcc_supports_architectures() {
    local compiler="$1" supported arch
    # Compilers without --list-gpu-arch are left to the initialization probe.
    supported="$("$compiler" --list-gpu-arch 2>/dev/null)" || return 0
    for arch in $(numeric_cuda_architectures); do
        [[ "$supported" == *"compute_${arch}"* ]] || return 1
    done
}

nvcc_is_compatible() {
    local compiler="$1" toolkit_major minimum_driver
    toolkit_major="$(nvcc_major "$compiler")"
    case "$toolkit_major" in
        13) minimum_driver=580 ;;
        12) minimum_driver=525 ;;
        *) return 1 ;;
    esac
    (( driver_major >= minimum_driver )) && nvcc_supports_architectures "$compiler"
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
            echo "requires CUDA 12+ for GPU architecture $CUDA_ARCHITECTURES and a compatible" >&2
            echo "NVIDIA driver (found $driver_version)." >&2
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
        echo "Skipping $resolved: current llama.cpp requires CUDA 12+ for GPU architecture" \
            "$CUDA_ARCHITECTURES and a compatible driver." >&2
    done
}

os_id=""
os_version=""
if [[ -r /etc/os-release ]]; then
    os_id="$(. /etc/os-release && printf '%s' "${ID:-}")"
    os_version="$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")"
fi

is_wsl() {
    [[ -e /usr/lib/wsl/lib/libcuda.so.1 ]] || grep -qi microsoft /proc/version 2>/dev/null
}

# NVIDIA's apt repository for this host. WSL uses the driver-free wsl-ubuntu
# repository so the Windows driver is never shadowed by a Linux one.
cuda_repository() {
    local arch
    case "$(uname -m)" in
        x86_64) arch="x86_64" ;;
        aarch64) arch="sbsa" ;;
        *) return 1 ;;
    esac
    if is_wsl && [[ "$os_id" == "ubuntu" && "$arch" == "x86_64" ]]; then
        printf 'wsl-ubuntu/x86_64\n'
    elif [[ "$os_id" == "ubuntu" || "$os_id" == "debian" ]]; then
        printf '%s%s/%s\n' "$os_id" "${os_version//./}" "$arch"
    else
        return 1
    fi
}

# Candidate toolkits as version:minimum-driver, in preference order. The tested
# 12.4 and 12.1 toolkits come first; newer 12.x releases cover distributions
# that do not package them (Ubuntu 24.04) and Blackwell GPUs, which need 12.8+.
CUDA_TOOLKIT_CANDIDATES=(12-4:550 12-1:525 12-5:555 12-6:560 12-8:570 12-9:575)

install_cuda_toolkit() {
    local repository keyring candidate version minimum highest_arch
    local root_command=()
    if ! repository="$(cuda_repository)"; then
        echo "No NVIDIA CUDA apt repository is known for ${os_id:-this OS} ${os_version}." >&2
        return 1
    fi
    command -v apt-get >/dev/null 2>&1 || return 1
    if (( EUID != 0 )); then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "CUDA toolkit installation requires root access or sudo." >&2
            return 1
        fi
        root_command=(sudo)
    fi
    keyring="$(mktemp)"
    if ! wget -qO "$keyring" \
        "https://developer.download.nvidia.com/compute/cuda/repos/$repository/cuda-keyring_1.1-1_all.deb"; then
        rm -f "$keyring"
        echo "Could not download the NVIDIA CUDA keyring for $repository." >&2
        return 1
    fi
    "${root_command[@]}" dpkg -i "$keyring" || { rm -f "$keyring"; return 1; }
    rm -f "$keyring"
    "${root_command[@]}" apt-get update || return 1
    highest_arch="$(highest_cuda_architecture)"
    for candidate in "${CUDA_TOOLKIT_CANDIDATES[@]}"; do
        version="${candidate%%:*}"
        minimum="${candidate##*:}"
        (( driver_major >= minimum )) || continue
        if [[ -n "$highest_arch" ]] && (( highest_arch >= 100 )) && [[ "$version" == 12-[0-7] ]]; then
            continue
        fi
        apt-cache show "cuda-toolkit-$version" >/dev/null 2>&1 || continue
        toolkit_package="cuda-toolkit-$version"
        echo "Installing $toolkit_package from $repository for NVIDIA driver $driver_version"
        "${root_command[@]}" apt-get install -y "$toolkit_package"
        return
    done
    echo "No CUDA 12 toolkit in $repository supports driver $driver_version" \
        "and GPU architecture $CUDA_ARCHITECTURES." >&2
    return 1
}

select_cuda_compiler() {
    CUDACXX="$(find_nvcc || true)"
    if [[ -z "$CUDACXX" && "$LAYA_INSTALL_CUDA_TOOLKIT" == "true" ]]; then
        if install_cuda_toolkit; then
            CUDACXX="$(find_nvcc || true)"
        fi
    fi
    if [[ -z "$CUDACXX" ]]; then
        echo "No installed CUDA compiler is compatible with NVIDIA driver $driver_version" \
            "and GPU architecture $CUDA_ARCHITECTURES." >&2
        echo "Install a CUDA 12 toolkit without replacing the working GPU driver; see" >&2
        if is_wsl; then
            echo "  https://docs.nvidia.com/cuda/wsl-user-guide/" >&2
        else
            echo "  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/" >&2
        fi
        echo "This script discovers nvcc on PATH and under /usr/local/cuda*." >&2
        return 1
    fi
}

probe_cuda_compiler() {
    local smoke_dir arch arch_flags=()
    export CUDACXX
    export PATH="$(dirname -- "$CUDACXX"):$PATH"
    CUDA_TOOLKIT_ROOT="$(cd -- "$(dirname -- "$CUDACXX")/.." && pwd)"
    echo "Using CUDA compiler: $CUDACXX"
    echo "Using CUDA toolkit root: $CUDA_TOOLKIT_ROOT"
    echo "Using CUDA architectures: $CUDA_ARCHITECTURES"
    "$CUDACXX" --version || return 1

    for arch in $(numeric_cuda_architectures); do
        arch_flags+=(-gencode "arch=compute_${arch},code=sm_${arch}")
    done
    (( ${#arch_flags[@]} )) || arch_flags=(-arch=native)
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
    if ! "$CUDACXX" "${arch_flags[@]}" "$smoke_dir/cuda-smoke.cu" -o "$smoke_dir/cuda-smoke"; then
        rm -rf "$smoke_dir"
        echo "The selected CUDA toolkit could not compile the GPU initialization probe." >&2
        return 1
    fi
    if ! "$smoke_dir/cuda-smoke"; then
        rm -rf "$smoke_dir"
        echo "CUDA cannot initialize with driver $driver_version and compiler $CUDACXX." >&2
        return 1
    fi
    rm -rf "$smoke_dir"
}

# Without --backend=cuda, a failed CUDA step degrades to a CPU build so the
# rest of the platform stays usable; local LLMs are then slower but available.
cuda_unavailable() {
    if [[ "$LAYA_LLAMA_BACKEND" == "cuda" ]]; then
        echo "LAYA_LLAMA_BACKEND=cuda was requested, so setup stops here." >&2
        exit 1
    fi
    echo "Falling back to a $NATIVE_BACKEND llama.cpp build: $1" >&2
    echo "Set LAYA_LLAMA_BACKEND=cpu to skip CUDA detection on this host." >&2
    BACKEND="$NATIVE_BACKEND"
}

BACKEND="$NATIVE_BACKEND"
if [[ "$LAYA_LLAMA_BACKEND" != "cpu" ]]; then
    if [[ -z "$driver_major" ]]; then
        cuda_unavailable "no working NVIDIA driver was detected."
    elif select_cuda_compiler && probe_cuda_compiler; then
        BACKEND="cuda"
    else
        cuda_unavailable "the CUDA toolchain is not usable on this host."
    fi
fi

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

build_llama_server() {
    local backend="$1" clean_reason="" cmake_args=()
    if [[ -f "$CMAKE_CACHE" ]]; then
        if [[ "$previous_revision" != "$selected_revision" ]]; then
            clean_reason="llama.cpp revision changed"
        elif [[ "$backend" == "cuda" ]]; then
            if ! grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE"; then
                clean_reason="existing build is not CUDA-enabled"
            elif ! grep -Eq "^CMAKE_CUDA_COMPILER[^=]*=${CUDACXX}$" "$CMAKE_CACHE"; then
                clean_reason="CUDA compiler changed"
            elif ! grep -Eq "^CMAKE_CUDA_ARCHITECTURES[^=]*=${CUDA_ARCHITECTURES}$" "$CMAKE_CACHE"; then
                clean_reason="CUDA architecture changed"
            fi
        elif grep -q '^GGML_CUDA:BOOL=ON$' "$CMAKE_CACHE"; then
            clean_reason="existing build is CUDA-enabled"
        fi
    fi
    if [[ -n "$clean_reason" ]]; then
        echo "Cleaning incompatible llama.cpp build: $clean_reason"
        rm -rf "$LLAMA_CPP_SOURCE/build"
    elif [[ -f "$CMAKE_CACHE" ]]; then
        echo "Resuming compatible llama.cpp build incrementally."
    fi
    rm -f "$RUNTIME_STAMP"

    if [[ "$backend" == "cuda" ]]; then
        cmake_args=(
            -DGGML_CUDA=ON
            -DCMAKE_CUDA_COMPILER="$CUDACXX"
            -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT"
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
        )
    else
        cmake_args=(-DGGML_CUDA=OFF)
    fi
    echo "Building llama.cpp for the $backend backend"
    cmake -S "$LLAMA_CPP_SOURCE" -B "$LLAMA_CPP_SOURCE/build" \
        "${cmake_args[@]}" \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release || return 1
    cmake --build "$LLAMA_CPP_SOURCE/build" --config Release --target llama-server \
        --parallel "$(cpu_count)" || return 1

    if [[ ! -x "$LLAMA_SERVER" ]]; then
        echo "Build completed without the expected executable: $LLAMA_SERVER" >&2
        return 1
    fi
    "$LLAMA_SERVER" --version || return 1
    local device_output cuda_device="" architecture=""
    device_output="$("$LLAMA_SERVER" --list-devices 2>&1)" || {
        printf '%s\n' "$device_output" >&2
        echo "Built llama-server could not list its devices." >&2
        return 1
    }
    printf '%s\n' "$device_output"
    if [[ "$backend" == "cuda" ]]; then
        cuda_device="$(first_cuda_device "$device_output")"
        if [[ -z "$cuda_device" ]]; then
            echo "Built llama-server does not enumerate a CUDA device." >&2
            return 1
        fi
        architecture="$CUDA_ARCHITECTURES"
    fi
    printf 'backend=%s\ndriver=%s\nnvcc=%s\nrevision=%s\narchitecture=%s\ndevice=%s\n' \
        "$backend" "$driver_version" "${CUDACXX:-}" "$LLAMA_CPP_REVISION" \
        "$architecture" "$cuda_device" >"$RUNTIME_STAMP"
}

if ! build_llama_server "$BACKEND"; then
    if [[ "$BACKEND" != "cuda" || "$LAYA_LLAMA_BACKEND" == "cuda" ]]; then
        echo "llama.cpp build failed for the $BACKEND backend." >&2
        exit 1
    fi
    cuda_unavailable "the CUDA build of llama.cpp failed."
    build_llama_server "$BACKEND"
fi

printf '\nllama.cpp (%s) is ready. The UI launcher will use:\n  LAYA_LLAMA_SERVER=%q\n  LAYA_GGUF_MODELS_DIR=%q\n' \
    "$BACKEND" "$LLAMA_SERVER" "$PROJECT_ROOT/models/gguf"
