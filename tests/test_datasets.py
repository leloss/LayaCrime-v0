import json

import pytest

from laya_adverse_media.datasets import (
    discover_dataset_bundles,
    load_dataset_bundle,
    register_annotation_set,
    remove_annotation_set,
    update_dataset_prompt,
)


def test_dataset_bundle_keeps_corpus_and_annotations_together(tmp_path) -> None:
    root = tmp_path / "datasets" / "example"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("", encoding="utf-8")
    (annotations / "reviewers.jsonl").write_text("", encoding="utf-8")
    (root / "dataset.json").write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "corpus": {"path": "corpus.jsonl", "records": 10},
            "annotations": [{
                "id": "reviewers",
                "name": "Reviewer consensus",
                "path": "annotations/reviewers.jsonl",
                "records": 10,
            }],
        }),
        encoding="utf-8",
    )

    bundles = discover_dataset_bundles(tmp_path / "datasets")

    assert list(bundles) == ["example"]
    assert bundles["example"].corpus == (root / "corpus.jsonl").resolve()
    assert bundles["example"].annotations[0].path == (
        annotations / "reviewers.jsonl"
    ).resolve()


def test_dataset_bundle_rejects_paths_outside_bundle(tmp_path) -> None:
    root = tmp_path / "example"
    root.mkdir()
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "corpus": {"path": "../corpus.jsonl"},
            "annotations": [{
                "id": "labels",
                "name": "Labels",
                "path": "annotations/labels.jsonl",
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the dataset bundle"):
        load_dataset_bundle(manifest)


def test_dataset_bundle_rejects_unknown_purpose(tmp_path) -> None:
    root = tmp_path / "example"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("", encoding="utf-8")
    (annotations / "labels.jsonl").write_text("", encoding="utf-8")
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "purpose": "training",
            "corpus": {"path": "corpus.jsonl"},
            "annotations": [{
                "id": "labels",
                "name": "Labels",
                "path": "annotations/labels.jsonl",
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset purpose"):
        load_dataset_bundle(manifest)


def test_dataset_bundle_rejects_declared_checksum_mismatch(tmp_path) -> None:
    root = tmp_path / "example"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (annotations / "labels.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "corpus": {"path": "corpus.jsonl", "sha256": "0" * 64},
            "annotations": [{
                "id": "labels",
                "name": "Labels",
                "path": "annotations/labels.jsonl",
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        load_dataset_bundle(manifest)


def test_dataset_bundle_accepts_training_only_purpose(tmp_path) -> None:
    root = tmp_path / "training"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("", encoding="utf-8")
    (annotations / "labels.jsonl").write_text("", encoding="utf-8")
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "training",
            "name": "Training dataset",
            "purpose": "training-only",
            "corpus": {"path": "corpus.jsonl"},
            "annotations": [{
                "id": "labels",
                "name": "Labels",
                "path": "annotations/labels.jsonl",
            }],
        }),
        encoding="utf-8",
    )

    assert load_dataset_bundle(manifest).manifest["purpose"] == "training-only"


def test_register_annotation_set_updates_manifest_atomically(tmp_path) -> None:
    root = tmp_path / "example"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("", encoding="utf-8")
    original = annotations / "original.jsonl"
    original.write_text("", encoding="utf-8")
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "corpus": {"path": "corpus.jsonl"},
            "annotations": [{
                "id": "original",
                "name": "Original",
                "path": "annotations/original.jsonl",
            }],
        }),
        encoding="utf-8",
    )
    uploaded = annotations / "reviewers.jsonl"
    uploaded.write_text('{"article_id":"one","label":1}\n', encoding="utf-8")

    updated = register_annotation_set(
        load_dataset_bundle(manifest), "reviewers", "Reviewer consensus", uploaded, 1
    )

    assert [annotation.id for annotation in updated.annotations] == [
        "original",
        "reviewers",
    ]
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["annotations"][1]["path"] == "annotations/reviewers.jsonl"
    assert saved["annotations"][1]["records"] == 1
    assert len(saved["annotations"][1]["sha256"]) == 64
    assert saved["annotations"][1]["user_managed"] is True

    removed = remove_annotation_set(updated, "reviewers")

    assert [annotation.id for annotation in removed.annotations] == ["original"]
    assert not uploaded.exists()
    with pytest.raises(PermissionError, match="built-in"):
        remove_annotation_set(removed, "original")


def test_update_dataset_prompt_persists_with_bundle(tmp_path) -> None:
    root = tmp_path / "example"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (root / "corpus.jsonl").write_text("", encoding="utf-8")
    (annotations / "labels.jsonl").write_text("", encoding="utf-8")
    manifest = root / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "id": "example",
            "name": "Example dataset",
            "corpus": {"path": "corpus.jsonl"},
            "annotations": [{
                "id": "labels",
                "name": "Labels",
                "path": "annotations/labels.jsonl",
            }],
        }),
        encoding="utf-8",
    )
    prompt = {
        "question": "Does {entity_name} show criminal intent?",
        "criteria": [
            {"decision": "negative", "text": "planned an offense"},
            {"decision": "positive", "text": "no offense was planned"},
        ],
    }

    updated = update_dataset_prompt(load_dataset_bundle(manifest), prompt)

    assert updated.prompt == prompt
    assert json.loads(manifest.read_text(encoding="utf-8"))["prompt"] == prompt