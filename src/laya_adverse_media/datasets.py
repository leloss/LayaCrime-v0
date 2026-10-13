from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
DEFAULT_QUESTION = (
    "How does this article portray {entity_name} regarding criminal behavior or intent? "
    "Judge only the named entity, not other people or organizations."
)
DEFAULT_CRITERIA = [
    {
        "decision": "negative",
        "text": (
            "negative: the article credibly associates the entity with alleged, investigated, "
            "charged, convicted, sanctioned, or admitted criminal behavior or intent"
        ),
    },
    {
        "decision": "positive",
        "text": (
            "positive: the article does not associate the entity with criminal behavior or intent, "
            "or identifies the entity only as a victim, witness, investigator, or unrelated party"
        ),
    },
]


@dataclass(frozen=True)
class AnnotationSet:
    id: str
    name: str
    path: Path
    records: int | None = None
    user_managed: bool = False


@dataclass(frozen=True)
class DatasetBundle:
    id: str
    name: str
    root: Path
    corpus: Path
    annotations: tuple[AnnotationSet, ...]
    prompt: dict[str, Any]
    manifest: dict[str, Any]


def validate_prompt(value: object | None) -> dict[str, Any]:
    if value is None:
        return {
            "question": DEFAULT_QUESTION,
            "criteria": [dict(item) for item in DEFAULT_CRITERIA],
        }
    if not isinstance(value, dict):
        raise ValueError("dataset prompt must be an object")
    question = value.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("dataset prompt.question must be a non-empty string up to 4000 characters")
    criteria = value.get("criteria")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 20:
        raise ValueError("dataset prompt.criteria must contain 2 to 20 entries")
    normalized = []
    decisions = set()
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError(f"dataset prompt.criteria[{index}] must be an object")
        decision = criterion.get("decision")
        text = criterion.get("text")
        if decision not in {"negative", "positive"}:
            raise ValueError(f"dataset prompt.criteria[{index}].decision is invalid")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError(
                f"dataset prompt.criteria[{index}].text must be a non-empty string up to 4000 characters"
            )
        decisions.add(decision)
        normalized.append({"decision": decision, "text": text.strip()})
    if decisions != {"negative", "positive"}:
        raise ValueError("dataset prompt criteria must include negative and positive decisions")
    return {"question": question.strip(), "criteria": normalized}


def _bundle_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the dataset bundle")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} points outside the dataset bundle")
    return resolved


def _verify_declared_file(path: Path, sha256: object, field: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    if sha256 is None:
        return
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256 digest")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha256:
        raise ValueError(f"{field} checksum does not match dataset.json")


def load_dataset_bundle(manifest_path: Path) -> DatasetBundle:
    root = manifest_path.resolve().parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest must use schema_version 1")
    dataset_id = manifest.get("id")
    if not isinstance(dataset_id, str) or not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("dataset id must contain lowercase letters, numbers, or hyphens")
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset name must be a non-empty string")
    purpose = manifest.get("purpose", "training-and-evaluation")
    if purpose not in {
        "training-and-evaluation",
        "training-only",
        "evaluation-only",
    }:
        raise ValueError(
            "dataset purpose must be training-and-evaluation, training-only, or evaluation-only"
        )
    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("dataset corpus must be an object")
    corpus_path = _bundle_path(root, corpus.get("path"), "corpus.path")
    _verify_declared_file(corpus_path, corpus.get("sha256"), "corpus")
    annotation_rows = manifest.get("annotations")
    if not isinstance(annotation_rows, list) or not annotation_rows:
        raise ValueError("dataset must define at least one annotation set")
    annotations = []
    seen_ids: set[str] = set()
    for index, row in enumerate(annotation_rows):
        if not isinstance(row, dict):
            raise ValueError(f"annotations[{index}] must be an object")
        annotation_id = row.get("id")
        if not isinstance(annotation_id, str) or not DATASET_ID_PATTERN.fullmatch(annotation_id):
            raise ValueError(f"annotations[{index}].id is invalid")
        if annotation_id in seen_ids:
            raise ValueError(f"duplicate annotation id {annotation_id}")
        seen_ids.add(annotation_id)
        annotation_name = row.get("name")
        if not isinstance(annotation_name, str) or not annotation_name.strip():
            raise ValueError(f"annotations[{index}].name must be a non-empty string")
        records = row.get("records")
        if records is not None and (not isinstance(records, int) or records < 0):
            raise ValueError(f"annotations[{index}].records must be a nonnegative integer")
        annotation_path = _bundle_path(
            root, row.get("path"), f"annotations[{index}].path"
        )
        _verify_declared_file(
            annotation_path, row.get("sha256"), f"annotations[{index}]"
        )
        annotations.append(
            AnnotationSet(
                id=annotation_id,
                name=annotation_name.strip(),
                path=annotation_path,
                records=records,
                user_managed=row.get("user_managed") is True,
            )
        )
    return DatasetBundle(
        id=dataset_id,
        name=name.strip(),
        root=root,
        corpus=corpus_path,
        annotations=tuple(annotations),
        prompt=validate_prompt(manifest.get("prompt")),
        manifest=manifest,
    )


def discover_dataset_bundles(root: Path) -> dict[str, DatasetBundle]:
    if not root.is_dir():
        return {}
    bundles: dict[str, DatasetBundle] = {}
    for manifest_path in sorted(root.glob("*/dataset.json")):
        bundle = load_dataset_bundle(manifest_path)
        if bundle.id in bundles:
            raise ValueError(f"duplicate dataset id {bundle.id}")
        bundles[bundle.id] = bundle
    return bundles


def register_annotation_set(
    bundle: DatasetBundle,
    annotation_id: str,
    name: str,
    path: Path,
    records: int,
) -> DatasetBundle:
    resolved = path.resolve()
    if not resolved.is_relative_to(bundle.root):
        raise ValueError("annotation path points outside the dataset bundle")
    manifest = dict(bundle.manifest)
    annotations = [dict(row) for row in manifest["annotations"]]
    if any(row.get("id") == annotation_id for row in annotations):
        raise ValueError(f"annotation id {annotation_id} already exists")
    annotations.append({
        "id": annotation_id,
        "name": name.strip(),
        "path": resolved.relative_to(bundle.root).as_posix(),
        "records": records,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "user_managed": True,
    })
    manifest["annotations"] = annotations
    manifest_path = bundle.root / "dataset.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return load_dataset_bundle(manifest_path)


def remove_annotation_set(bundle: DatasetBundle, annotation_id: str) -> DatasetBundle:
    manifest = dict(bundle.manifest)
    annotations = [dict(row) for row in manifest["annotations"]]
    selected = next(
        (row for row in annotations if row.get("id") == annotation_id), None
    )
    if selected is None:
        raise FileNotFoundError("annotation set not found")
    if selected.get("user_managed") is not True:
        raise PermissionError("built-in annotation sets cannot be deleted")
    remaining = [row for row in annotations if row.get("id") != annotation_id]
    if not remaining:
        raise ValueError("a dataset must retain at least one annotation set")
    annotation_path = _bundle_path(
        bundle.root, selected.get("path"), f"annotation {annotation_id} path"
    )
    manifest["annotations"] = remaining
    manifest_path = bundle.root / "dataset.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    annotation_path.unlink(missing_ok=True)
    return load_dataset_bundle(manifest_path)