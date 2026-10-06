from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


def script(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_environment_setup_only_provisions_dependencies_and_models() -> None:
    content = script("setup_environment.sh")

    assert "-m venv --clear --copies" in content
    assert "sudo apt-get install -y python3-venv" in content
    assert "pip install" in content
    assert "LAYA_AUTO_INSTALL_SYSTEM_DEPS" in content
    assert "build-essential git cmake wget ca-certificates python3-venv" in content
    assert "LAYA_INSTALL_CUDA_TOOLKIT=true" in content
    assert "scripts/setup_llama_cpp.sh" in content
    assert "scripts/download_huggingface_snapshot.py" in content
    assert "convaiinnovations/laya" in content
    assert "leloss/layacrime" in content
    assert "fine_tune_laya.py train" not in content
    assert "run_ui.sh" in content


def test_backend_launcher_enables_ui_without_starting_training() -> None:
    content = script("run_ui.sh")

    assert 'LAYA_HOST="${LAYA_HOST:-127.0.0.1}"' in content
    assert 'LAYA_ENABLE_FINE_TUNING="${LAYA_ENABLE_FINE_TUNING:-true}"' in content
    assert 'LAYA_REQUIRE_CUDA="${LAYA_REQUIRE_CUDA:-true}"' in content
    assert 'exec "$SERVER"' in content
    assert "Benchmark:" in content
    assert "Fine-tuning:" in content
    assert "export LAYA_FINE_TUNING_ALLOWED_ROOTS" not in content
    assert "fine_tune_laya.py" not in content
    assert "pip install" not in content
    assert "LAYA_LLAMA_SERVER" in content
    assert "third_party/llama.cpp/build/bin/llama-server" in content
    assert "GGML_CUDA:BOOL=ON" in content
    assert "laya-cuda-runtime.ok" in content
    assert "Ignoring project llama-server" in content


def test_model_installer_does_not_accept_an_unverified_path_runtime() -> None:
    content = script("download_huggingface_file.py")

    assert 'shutil.which("llama-server")' not in content
    assert '"GGML_CUDA:BOOL=ON"' in content
    assert "laya-cuda-runtime.ok" in content
    assert "setup_llama_cpp.sh" in content


def test_llama_cpp_setup_builds_cuda_server_for_t4() -> None:
    content = script("setup_llama_cpp.sh")

    assert "-DGGML_CUDA=ON" in content
    assert 'LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-bed0a8566}"' in content
    assert 'CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"' in content
    assert "/usr/local/cuda/bin/nvcc" in content
    assert 'export CUDACXX' in content
    assert "cuda-keyring_1.1-1_all.deb" in content
    assert "LAYA_INSTALL_CUDA_TOOLKIT" in content
    assert 'LAYA_INSTALL_CUDA_TOOLKIT="${LAYA_INSTALL_CUDA_TOOLKIT:-true}"' in content
    assert 'apt-get install -y "$toolkit_package"' in content
    assert 'toolkit_package="cuda-toolkit-12-1"' in content
    assert 'toolkit_package="cuda-toolkit-12-4"' in content
    assert "cudaGetDeviceCount" in content
    assert "minimum_driver=580" in content
    assert "11) minimum_driver" not in content
    assert 'requires CUDA 12+' in content
    assert '-DCMAKE_CUDA_COMPILER="$CUDACXX"' in content
    assert '-DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT"' in content
    assert "laya-cuda-runtime.ok" in content
    assert "--list-devices" in content
    assert "Built llama-server does not enumerate a CUDA device" in content
    assert "runtime_is_reusable" in content
    assert "source_matches_requested_revision" in content
    assert 'binary_version="$($LLAMA_SERVER --version 2>&1)"' in content
    assert "adopted=true" in content
    assert "already installed; nothing to rebuild" in content
    assert "Resuming compatible llama.cpp build incrementally" in content
    assert "Cleaning incompatible llama.cpp build" in content
    assert 'rm -rf "$LLAMA_CPP_SOURCE/build"' in content
    assert content.index('if [[ -n "$clean_reason" ]]') < content.index(
        'rm -rf "$LLAMA_CPP_SOURCE/build"'
    )
    assert "--target llama-server" in content
    assert "build/bin/llama-server" in content


def test_legacy_backend_launcher_delegates_to_ui() -> None:
    content = script("run_finetuning_backend.sh")

    assert 'exec "$PROJECT_ROOT/scripts/run_ui.sh" "$@"' in content


def test_optional_cli_trainer_requires_provisioned_environment() -> None:
    content = script("train_remote.sh")

    assert "setup_environment.sh first" in content
    assert 'source "$PROJECT_ROOT/.env.local"' in content
    assert "LAYA_ACTIVE_DATASET" in content
    assert "LAYA_BENCHMARK_LABEL_SET" in content
    assert "fine_tune_laya.py prepare" in content
    assert '"$PYTHON" "${train_args[@]}"' in content
    assert "pip install" not in content
    assert "snapshot_download" not in content


def test_fine_tuning_hub_repository_pattern_escapes_separator() -> None:
    content = (ROOT / "src" / "laya_adverse_media" / "static" / "fine-tuning.html").read_text(
        encoding="utf-8"
    )

    assert "{0,95}\\/[A-Za-z0-9]" in content
