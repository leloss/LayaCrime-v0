import sys
import time
from pathlib import Path

import pytest

from laya_adverse_media.model_installation import ModelInstallationJob
from scripts.download_huggingface_file import (
    finalize_gguf_installation,
    resolve_gguf_filename,
)


def test_gguf_filename_resolution_handles_case_and_repository_paths() -> None:
    files = ["weights/GPT-OSS-20B-MXFP4.gguf", "README.md"]

    assert resolve_gguf_filename("gpt-oss-20b-mxfp4.gguf", files) == files[0]


def test_gguf_filename_resolution_uses_unique_quantization() -> None:
    files = ["renamed-release-MXFP4.gguf", "renamed-release-Q4_0.gguf"]

    assert resolve_gguf_filename("gpt-oss-20b-mxfp4.gguf", files) == files[0]


def test_gguf_filename_resolution_uses_only_available_quantized_model() -> None:
    files = ["model-BF16.gguf", "model-Q8_0.gguf", "README.md"]

    assert resolve_gguf_filename("model-Q6_K.gguf", files) == "model-Q8_0.gguf"


def test_gguf_filename_resolution_rejects_ambiguous_quantization() -> None:
    with pytest.raises(FileNotFoundError, match="could not uniquely resolve"):
        resolve_gguf_filename(
            "gpt-oss-20b-mxfp4.gguf",
            ["part-a-mxfp4.gguf", "part-b-mxfp4.gguf"],
        )


def test_finalize_gguf_installation_uses_canonical_name_and_atomic_receipt(
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / "weights" / "MODEL-Q6_K.gguf"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"GGUFcomplete")
    target = tmp_path / "model-q6-k.gguf"

    installed = finalize_gguf_installation(
        downloaded,
        target,
        repo_id="owner/model",
        resolved_filename="weights/MODEL-Q6_K.gguf",
        expected_size=12,
    )

    assert installed == target
    assert target.read_bytes() == b"GGUFcomplete"
    receipt = target.with_name(target.name + ".install.json")
    assert receipt.is_file()
    assert '"size": 12' in receipt.read_text(encoding="utf-8")


def test_finalize_gguf_installation_rejects_partial_file(tmp_path: Path) -> None:
    downloaded = tmp_path / "model.gguf"
    downloaded.write_bytes(b"GGUFpartial")

    with pytest.raises(RuntimeError, match="repository metadata requires 99 bytes"):
        finalize_gguf_installation(
            downloaded,
            downloaded,
            repo_id="owner/model",
            resolved_filename="model.gguf",
            expected_size=99,
        )

    assert not (tmp_path / "model.gguf.install.json").exists()


def test_model_installation_job_streams_progress_and_completion(tmp_path: Path) -> None:
    installed = tmp_path / "model.gguf"
    program = (
        "import pathlib,sys; "
        "sys.stdin.read(); "
        "print('phase=building_runtime', flush=True); "
        "print('compiler output', flush=True); "
        "print('download_total=4', flush=True); "
        f"path=pathlib.Path({str(installed)!r}); path.write_bytes(b'gguf'); "
        "print('phase=downloading_gguf', flush=True); "
        "print(f'installed_path={path}', flush=True)"
    )
    job = ModelInstallationJob()

    job.start(
        [sys.executable, "-u", "-c", program],
        "test-model",
        tmp_path,
        tmp_path,
        stdin_data='{"token":"secret"}',
    )
    deadline = time.monotonic() + 5
    while job.snapshot()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    snapshot = job.snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["phase"] == "ready"
    assert snapshot["downloaded_bytes"] == 4
    assert snapshot["total_bytes"] == 4
    assert snapshot["progress"] == 1.0
    assert snapshot["installed_path"] == str(installed)
    assert snapshot["logs"] == [{"index": 1, "message": "compiler output"}]