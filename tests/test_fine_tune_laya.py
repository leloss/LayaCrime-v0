import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "fine_tune_laya.py"
SPEC = importlib.util.spec_from_file_location("fine_tune_laya", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_record(path: Path, entity: str, disposition: str, content: str) -> None:
    path.write_text(
        f"###entityName: {entity}\n"
        f"###disspositionReason: {disposition}\n"
        f"###content: {content}\n",
        encoding="utf-8",
    )


def test_parse_record_maps_adverse_media_labels(tmp_path) -> None:
    hit = tmp_path / "hit"
    false_positive = tmp_path / "false-positive"
    write_record(hit, "Acme Corp", "Hit", "Acme was charged with fraud.")
    write_record(false_positive, "Acme Corp", "False Positive", "Acme was the victim.")

    assert MODULE.parse_record(hit).label == 0
    assert MODULE.parse_record(false_positive).label == 1


def test_load_records_drops_conflicts_and_deduplicates(tmp_path) -> None:
    write_record(tmp_path / "same-1", "Acme", "Hit", "Same article")
    write_record(tmp_path / "same-2", "Acme", "Hit", "Same article")
    write_record(tmp_path / "conflict-1", "Beta", "Hit", "Conflicting article")
    write_record(tmp_path / "conflict-2", "Beta", "False Positive", "Conflicting article")

    records, report = MODULE.load_records(tmp_path)

    assert len(records) == 1
    assert report["duplicate_copies_removed"] == 3
    assert report["conflicting_duplicate_groups_removed"] == 1


def test_entity_split_prevents_leakage() -> None:
    records = [
        MODULE.Record(entity=f"Entity {index // 2}", article=f"Article {index}", label=index % 2, source=str(index))
        for index in range(100)
    ]
    splits = MODULE.split_records(records)
    entity_sets = {
        name: {record.entity.casefold() for record in values}
        for name, values in splits.items()
    }

    assert entity_sets["train"].isdisjoint(entity_sets["calibration"])
    assert entity_sets["train"].isdisjoint(entity_sets["test"])
    assert entity_sets["calibration"].isdisjoint(entity_sets["test"])


def test_dataset_fingerprint_covers_human_labels() -> None:
    records = [MODULE.Record(entity="Acme", article="Article", label=0, source="one")]
    relabeled = [MODULE.Record(entity="Acme", article="Article", label=1, source="two")]

    assert MODULE.dataset_sha256(records) != MODULE.dataset_sha256(relabeled)


def test_article_id_matches_blind_corpus_identity() -> None:
    record = MODULE.Record(
        entity="Acme Corp",
        article="Acme   was charged.\n",
        label=0,
        source="human-record",
    )

    normalized = "\n".join(
        (record.entity.casefold(), " ".join(record.article.split()).casefold())
    )
    expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]

    assert MODULE.article_id(record) == expected


def test_load_azure_records_maps_labels_and_audits_missing_rows(tmp_path) -> None:
    negative = MODULE.Record("Acme", "Acme was charged.", 0, "test")
    positive = MODULE.Record("Beta", "Beta was the victim.", 0, "test")
    missing = MODULE.Record("Gamma", "Gamma was mentioned.", 0, "test")
    corpus = tmp_path / "corpus.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "article_id": MODULE.article_id(record),
                    "entity_name": record.entity,
                    "article": record.article,
                }
            )
            for record in (negative, positive, missing)
        )
        + "\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "article_id": MODULE.article_id(negative),
                        "label": 2,
                        "annotator": "gpt-5.6-terra",
                    }
                ),
                json.dumps(
                    {
                        "article_id": MODULE.article_id(positive),
                        "label": 1,
                        "annotator": "gpt-5.6-terra",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records, audit = MODULE.load_azure_records(corpus, annotations)

    assert [record.label for record in records] == [0, 1]
    assert [MODULE.article_id(record) for record in records] == [
        MODULE.article_id(negative),
        MODULE.article_id(positive),
    ]
    assert audit["missing_annotations"] == 1
    assert audit["missing_article_ids"] == [MODULE.article_id(missing)]
    assert audit["annotators"] == {"gpt-5.6-terra": 2}


def test_fine_tuning_config_matches_released_recipe() -> None:
    configured = MODULE.fine_tuning_config(
        {"max_len": 512, "head_max_len": 192, "temperature": [1.0, 1.0, 1.0]}
    )

    assert configured["max_len"] == 1024
    assert configured["head_max_len"] == 256
    assert configured["max_tokens_per_batch"] == 4096
    assert configured["gradient_checkpointing"] is True
    assert configured["temperature"] == [1.0, 1.0, 1.0]


def test_validate_prepared_config_rejects_stale_sequence_lengths(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"dataset_schema_version": 1, "label_source": "original_human_disposition", '
        '"dataset_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"sequence_config": {"max_len": 512, "head_max_len": 192}}',
        encoding="utf-8",
    )

    try:
        MODULE.validate_prepared_config(
            tmp_path,
            {"max_len": 1024, "head_max_len": 256},
        )
    except RuntimeError as exc:
        assert "rerun the prepare command" in str(exc)
    else:
        raise AssertionError("stale prepared data should be rejected")


def test_validate_prepared_config_rejects_unknown_label_source(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"dataset_schema_version": 1, "label_source": "generated", '
        '"dataset_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"sequence_config": {"max_len": 1024, "head_max_len": 256}}',
        encoding="utf-8",
    )

    try:
        MODULE.validate_prepared_config(
            tmp_path,
            {"max_len": 1024, "head_max_len": 256},
        )
    except RuntimeError as exc:
        assert "unsupported label_source" in str(exc)
    else:
        raise AssertionError("unknown label sources should be rejected")


def test_prepare_and_train_defaults_use_azure_artifact_paths() -> None:
    parser = MODULE.build_parser()

    prepare_args = parser.parse_args(["prepare"])
    train_args = parser.parse_args(["train"])

    assert prepare_args.label_source == "azure"
    assert prepare_args.output_dir == MODULE.default_prepared_dir()
    assert train_args.prepared_dir == MODULE.default_prepared_dir()
    assert train_args.output_dir == MODULE.default_output_model_dir()


def test_validate_train_args_rejects_zero_sigma() -> None:
    args = MODULE.build_parser().parse_args(["train", "--sigma-end", "0"])

    try:
        MODULE.validate_train_args(args)
    except ValueError as exc:
        assert "sigma values" in str(exc)
    else:
        raise AssertionError("zero exploration sigma should be rejected")