from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_RECEIPT_SUFFIX = ".install.json"


def resolve_gguf_filename(requested: str, repository_files: list[str]) -> str:
    requested_name = Path(requested).name.casefold()
    exact = [
        candidate
        for candidate in repository_files
        if Path(candidate).name.casefold() == requested_name
    ]
    if len(exact) == 1:
        return exact[0]

    quantizations = (
        "mxfp4",
        "q2_k",
        "q3_k_m",
        "q4_0",
        "q4_k_m",
        "q5_k_m",
        "q6_k",
        "q8_0",
    )
    quantization = next(
        (value for value in quantizations if value in requested_name),
        None,
    )
    matches = [
        candidate
        for candidate in repository_files
        if Path(candidate).suffix.casefold() == ".gguf"
        and quantization is not None
        and quantization in Path(candidate).name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    quantized = [
        candidate
        for candidate in repository_files
        if Path(candidate).suffix.casefold() == ".gguf"
        and not any(
            precision in Path(candidate).name.casefold()
            for precision in ("bf16", "f16", "f32")
        )
    ]
    if len(quantized) == 1:
        return quantized[0]
    raise FileNotFoundError(
        f"could not uniquely resolve {requested!r}; available GGUF files: "
        + ", ".join(repository_files)
    )


def finalize_gguf_installation(
    downloaded_path: Path,
    target_path: Path,
    *,
    repo_id: str,
    resolved_filename: str,
    expected_size: int | None,
) -> Path:
    actual_size = downloaded_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"downloaded GGUF is {actual_size} bytes; repository metadata requires "
            f"{expected_size} bytes"
        )
    with downloaded_path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("downloaded file does not have a valid GGUF signature")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != target_path.resolve():
        staging_path = target_path.with_name(target_path.name + ".installing")
        staging_path.unlink(missing_ok=True)
        try:
            os.link(downloaded_path, staging_path)
        except OSError:
            shutil.copyfile(downloaded_path, staging_path)
        os.replace(staging_path, target_path)

    receipt_path = target_path.with_name(target_path.name + INSTALL_RECEIPT_SUFFIX)
    staging_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
    staging_receipt.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": repo_id,
                "resolved_filename": resolved_filename,
                "size": target_path.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(staging_receipt, receipt_path)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--target-filename")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    credentials = json.loads(sys.stdin.read() or "{}")
    token = credentials.get("token")

    try:
        print("phase=reading_metadata", flush=True)
        from huggingface_hub import (
            HfApi,
            get_hf_file_metadata,
            hf_hub_download,
            hf_hub_url,
        )
        from huggingface_hub.errors import EntryNotFoundError

        filename = args.filename
        try:
            metadata = get_hf_file_metadata(
                hf_hub_url(repo_id=args.repo_id, filename=filename),
                token=token,
            )
        except EntryNotFoundError:
            filename = resolve_gguf_filename(
                filename,
                HfApi().list_repo_files(args.repo_id, token=token),
            )
            print(f"resolved_filename={filename}", flush=True)
            metadata = get_hf_file_metadata(
                hf_hub_url(repo_id=args.repo_id, filename=filename),
                token=token,
            )
        if metadata.size is not None:
            print(f"download_total={metadata.size}", flush=True)

        print("phase=checking_runtime", flush=True)
        configured_runtime = os.getenv("LAYA_LLAMA_SERVER")
        project_runtime = (
            args.project_root
            / "third_party"
            / "llama.cpp"
            / "build"
            / "bin"
            / "llama-server"
        )
        cmake_cache = project_runtime.parents[1] / "CMakeCache.txt"
        runtime_stamp = project_runtime.parents[1] / "laya-cuda-runtime.ok"
        project_runtime_verified = False
        if project_runtime.is_file() and runtime_stamp.is_file() and cmake_cache.is_file():
            stamp = runtime_stamp.read_text(encoding="utf-8", errors="replace")
            # Stamps without a backend line predate CPU/Metal builds and describe CUDA.
            cuda_build = "backend=" not in stamp or "backend=cuda" in stamp.splitlines()
            project_runtime_verified = not cuda_build or (
                "GGML_CUDA:BOOL=ON"
                in cmake_cache.read_text(encoding="utf-8", errors="replace")
            )
        runtime = (
            configured_runtime
            if configured_runtime and Path(configured_runtime).is_file()
            else str(project_runtime) if project_runtime_verified else None
        )
        if runtime is None:
            if os.name == "nt":
                raise RuntimeError(
                    "llama-server is missing. Build it on a Linux or macOS host with "
                    "scripts/setup_llama_cpp.sh"
                )
            print("phase=building_runtime", flush=True)
            subprocess.run(
                ["bash", str(args.project_root / "scripts" / "setup_llama_cpp.sh")],
                cwd=args.project_root,
                check=True,
            )
            runtime = str(project_runtime)
        if not Path(runtime).is_file():
            raise RuntimeError(f"llama-server was not produced at {runtime}")
        if runtime == str(project_runtime) and not runtime_stamp.is_file():
            raise RuntimeError("llama-server did not pass its device initialization probe")

        print("phase=downloading_gguf", flush=True)

        downloaded_path = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=args.local_dir,
            token=token,
        ))
        path = finalize_gguf_installation(
            downloaded_path,
            args.local_dir / (args.target_filename or Path(args.filename).name),
            repo_id=args.repo_id,
            resolved_filename=filename,
            expected_size=metadata.size,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"installed_path={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())