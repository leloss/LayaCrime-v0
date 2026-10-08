from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATASETS = {
    "adverse-media-public-tuning-2000",
    "adverse-media-public-holdout-1000",
}
REQUIRED_RELEASE_FILES = {
    "datasets/README.md",
    "datasets/adverse-media-public-tuning-2000/README.md",
    "datasets/adverse-media-public-tuning-2000/annotations/human.jsonl",
    "datasets/adverse-media-public-tuning-2000/corpus.jsonl",
    "datasets/adverse-media-public-tuning-2000/dataset.json",
    "datasets/adverse-media-public-holdout-1000/README.md",
    "datasets/adverse-media-public-holdout-1000/annotations/human.jsonl",
    "datasets/adverse-media-public-holdout-1000/corpus.jsonl",
    "datasets/adverse-media-public-holdout-1000/dataset.json",
    "scripts/benchmark_academic_baselines.py",
    "scripts/benchmark_newsmtsc_checkpoint.py",
    "scripts/benchmark_public_holdout_llms.py",
    "scripts/benchmark_tartu_baseline.py",
    "scripts/compare_academic_baselines.py",
    "tests/test_benchmark_academic_baselines.py",
    "tests/test_benchmark_newsmtsc_checkpoint.py",
    "tests/test_benchmark_public_holdout_llms.py",
    "tests/test_benchmark_tartu_baseline.py",
    "tests/test_compare_academic_baselines.py",
}
MODEL_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
PRIVATE_ANNOTATION_FILES = {
    "docs/annotation-rubric.md",
    "scripts/annotate_blind_corpus_azure.py",
    "scripts/build_annotation_corpus.py",
    "scripts/build_holdout_annotation.py",
    "scripts/build_public_datasets.py",
    "scripts/build_relational_dataset.py",
    "scripts/compare_source_labels.py",
    "scripts/extract_named_entities.py",
    "scripts/merge_annotations.py",
    "tests/test_annotate_blind_corpus_azure.py",
    "tests/test_build_annotation_corpus.py",
    "tests/test_build_holdout_annotation.py",
    "tests/test_build_public_datasets.py",
    "tests/test_build_relational_dataset.py",
    "tests/test_compare_source_labels.py",
    "tests/test_extract_named_entities.py",
    "tests/test_merge_annotations.py",
}


def release_candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / value
        for value in result.stdout.splitlines()
        if value and (ROOT / value).is_file()
    ]


def violation(path: Path) -> str | None:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if relative.as_posix() in PRIVATE_ANNOTATION_FILES:
        return "annotation production code belongs in ../laya-annotations-private"
    if parts[0] == "paper":
        return "article sources and generated manuscript files must not ship"
    if parts[0] == "artifacts":
        return "generated artifacts and saved runs are private"
    if parts[0] == "models" and relative.as_posix() != "models/README.md":
        return "model weights and checkpoints must come from Hugging Face"
    if parts[0] == "datasets" and len(parts) > 1:
        if parts[1] != "README.md" and parts[1] not in PUBLIC_DATASETS:
            return "only public tuning and holdout datasets may ship"
        if "training-subsets" in parts or path.stem.casefold() == "consensus":
            return "only human annotations ship; consensus labels and training subsets are private"
    if path.suffix.casefold() in MODEL_SUFFIXES:
        return "model weight files must not ship in the repository"
    return None


def main() -> int:
    candidates = release_candidates()
    candidate_names = {path.relative_to(ROOT).as_posix() for path in candidates}
    violations = [
        (path.relative_to(ROOT), reason)
        for path in candidates
        if (reason := violation(path)) is not None
    ]
    if violations:
        for path, reason in violations:
            print(f"ERROR {path.as_posix()}: {reason}")
        return 1

    missing = sorted(REQUIRED_RELEASE_FILES - candidate_names)
    if missing:
        for path in missing:
            print(f"ERROR {path}: required reproducibility code or public dataset metadata is missing")
        return 1

    for dataset_id in sorted(PUBLIC_DATASETS):
        manifest_path = ROOT / "datasets" / dataset_id / "dataset.json"
        if not manifest_path.is_file():
            print(f"ERROR {manifest_path.relative_to(ROOT).as_posix()}: public manifest is missing")
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("license") != "Apache-2.0"
            or manifest.get("redistribution_status") != "cleared-for-public-release"
        ):
            print(
                f"ERROR {manifest_path.relative_to(ROOT).as_posix()}: "
                "public release terms are not cleared"
            )
            return 1
        declared_files = [("corpus", manifest.get("corpus") or {})] + [
            (f"annotations[{index}]", row)
            for index, row in enumerate(manifest.get("annotations") or [])
        ]
        for field, entry in declared_files:
            declared = entry.get("sha256")
            data_path = manifest_path.parent / str(entry.get("path", ""))
            if declared is None or not data_path.is_file():
                continue
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != declared:
                print(
                    f"ERROR {manifest_path.relative_to(ROOT).as_posix()}: "
                    f"{field} checksum does not match {data_path.relative_to(ROOT).as_posix()}"
                )
                return 1

    staged_modes = subprocess.run(
        ["git", "ls-files", "--stage", "--", "*.sh"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    non_executable = [
        line.split("\t", 1)[1] for line in staged_modes if not line.startswith("100755")
    ]
    if non_executable:
        for path in non_executable:
            print(f"ERROR {path}: shell scripts must be executable (git update-index --chmod=+x)")
        return 1

    public_files = [
        path
        for path in candidates
        if path.relative_to(ROOT).parts[:1] == ("datasets",)
        and len(path.relative_to(ROOT).parts) > 1
        and path.relative_to(ROOT).parts[1] in PUBLIC_DATASETS
    ]
    public_bytes = sum(path.stat().st_size for path in public_files)
    print(
        f"Release boundary clean: {len(public_files)} public dataset files "
        f"({public_bytes:,} bytes); no article sources, models, artifacts, saved runs, "
        "or private datasets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
