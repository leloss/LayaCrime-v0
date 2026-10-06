import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

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


def test_count_correct_choices_compares_indices_with_integer_labels() -> None:
    logits = MODULE.torch.tensor([
        [0.2, 0.8, -4.0, -4.0],
        [0.7, 0.3, -4.0, -4.0],
        [0.4, 0.6, -4.0, -4.0],
    ])
    mask = MODULE.torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
    ])
    labels = MODULE.torch.tensor([1, 0, 0])

    assert MODULE.count_correct_choices(logits, mask, labels) == 2


def test_training_prompt_builds_question_from_dataset_manifest(tmp_path) -> None:
    manifest = tmp_path / "dataset.json"
    prompt = {
        "question": "Assess whether {entity_name} planned an offense.",
        "criteria": [
            {"decision": "negative", "text": "charged with an offense"},
            {"decision": "negative", "text": "planned an offense"},
            {"decision": "positive", "text": "only a witness"},
        ],
    }
    manifest.write_text(json.dumps({"prompt": prompt}), encoding="utf-8")

    loaded = MODULE.training_prompt(manifest)
    question = MODULE.adverse_question("Acme", loaded)

    assert loaded == prompt
    assert question["ins"] == "Assess whether 'Acme' planned an offense."
    assert question["crit"] == {
        "A": "charged with an offense | planned an offense",
        "B": "only a witness",
    }


def test_load_records_drops_conflicts_and_deduplicates(tmp_path) -> None:
    write_record(tmp_path / "same-1", "Acme", "Hit", "Same article")
    write_record(tmp_path / "same-2", "Acme", "Hit", "Same article")
    write_record(tmp_path / "conflict-1", "Beta", "Hit", "Conflicting article")
    write_record(tmp_path / "conflict-2", "Beta", "False Positive", "Conflicting article")

    records, report = MODULE.load_records(tmp_path)

    assert len(records) == 1
    assert report["duplicate_copies_removed"] == 3
    assert report["conflicting_duplicate_groups_removed"] == 1


def test_load_external_records_maps_bundle_labels(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"article_id": "negative", "entity_name": "Acme", "article": "Acme was charged."},
                {"article_id": "positive", "entity_name": "Beta", "article": "Beta was a witness."},
                {"article_id": "unlabeled", "entity_name": "Gamma", "article": "Gamma was mentioned."},
            ]
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"article_id": "negative", "label": 2, "annotator": "reviewer"},
                {"article_id": "positive", "label": 1, "annotator": "reviewer"},
            ]
        ),
        encoding="utf-8",
    )

    records, report = MODULE.load_external_records(corpus, labels)

    assert [(record.identifier, record.label) for record in records] == [
        ("negative", 0),
        ("positive", 1),
    ]
    assert report["labels"] == {"negative": 1, "positive": 1}
    assert report["annotators"] == {"reviewer": 2}
    assert report["missing_annotations"] == 1
    assert report["missing_article_ids_sample"] == ["unlabeled"]


def test_build_sequence_truncates_state_during_tokenization() -> None:
    class FakeTokenizer:
        mask_token = "[MASK]"
        mask_token_id = 3
        cls_token_id = 1
        sep_token_id = 2

        def __init__(self) -> None:
            self.state_call = None

        def __call__(self, text, **kwargs):
            if text.startswith("{"):
                self.state_call = kwargs
                length = kwargs.get("max_length", 20_000)
                return {"input_ids": [4] * length}
            return {"input_ids": [5] * 8}

    tokenizer = FakeTokenizer()
    ids, markers = MODULE.build_training_sequence(
        tokenizer,
        {"article": "long article", "entity_name": "Acme"},
        MODULE.adverse_question("Acme"),
        max_len=128,
        head_max_len=64,
        option_order=[0, 1],
    )

    assert len(ids) == 128
    assert markers == [10, 19]
    assert tokenizer.state_call["truncation"] is True
    assert tokenizer.state_call["max_length"] < 128


def test_adverse_question_includes_criminal_behavior_and_intent() -> None:
    question = MODULE.adverse_question("Acme")

    assert "criminal behavior or intent" in question["ins"]
    assert "criminal behavior or intent" in question["crit"]["B"]
    assert question["crit"]["A"].endswith("criminal behavior or intent")


def test_cosine_lr_multiplier_keeps_frozen_group_at_zero() -> None:
    assert MODULE.cosine_lr_multiplier(0.0, 100, 50) == 1.0
    assert 0.0 * MODULE.cosine_lr_multiplier(0.0, 100, 100) == 0.0
    assert MODULE.cosine_lr_multiplier(5e-5, 100, 100) == pytest.approx(0.02)


def test_staged_lr_multiplier_delays_then_warms_encoder() -> None:
    assert MODULE.staged_lr_multiplier(1e-6, 100, 19, 10, 20) == 0.0
    assert MODULE.staged_lr_multiplier(1e-6, 100, 20, 10, 20) == 0.0
    assert MODULE.staged_lr_multiplier(1e-6, 100, 25, 10, 20) == 0.5
    assert MODULE.staged_lr_multiplier(1e-6, 100, 30, 10, 20) == 1.0


def test_smooth_choice_targets_preserves_mass_on_valid_options() -> None:
    targets = MODULE.torch.tensor([[1.0, 0.0, 0.0]])
    mask = MODULE.torch.tensor([[True, True, False]])

    smoothed = MODULE.smooth_choice_targets(targets, mask, 0.1)

    assert smoothed.tolist()[0] == pytest.approx([0.95, 0.05, 0.0])


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
    assert set(entity_sets) == {"train", "calibration"}


def test_split_keeps_shared_articles_together() -> None:
    records = [
        MODULE.Record(entity="Negative Target", article="Shared article", label=0, source="negative"),
        MODULE.Record(entity="Unrelated Entity", article="Shared article", label=1, source="positive"),
    ]

    splits = MODULE.split_records(records)

    assert sorted(len(values) for values in splits.values()) == [0, 2]


@pytest.mark.parametrize(
    ("strategy", "train_percentage"),
    [
        ("holdout_50_50", 50),
        ("holdout_60_40", 60),
        ("holdout_70_30", 70),
        ("holdout_80_20", 80),
    ],
)
def test_holdout_presets_emit_only_train_and_calibration(
    strategy: str, train_percentage: int
) -> None:
    records = [
        MODULE.Record(
            entity=f"Entity {index}",
            article=f"Article {index}",
            label=index % 2,
            source=str(index),
        )
        for index in range(1000)
    ]

    splits, metadata = MODULE.partition_records(records, strategy=strategy)

    assert set(splits) == {"train", "calibration"}
    assert sum(map(len, splits.values())) == len(records)
    assert metadata["train_percentage"] == train_percentage
    assert metadata["calibration_percentage"] == 100 - train_percentage
    assert abs(len(splits["train"]) - train_percentage * 10) < 100


@pytest.mark.parametrize("strategy", ["group_k_fold", "leave_one_group_out"])
def test_partition_strategies_keep_entity_and_article_groups_disjoint(strategy) -> None:
    records = [
        MODULE.Record(
            entity=f"Entity {index // 2}",
            article=f"Article {index // 3}",
            label=index % 2,
            source=str(index),
        )
        for index in range(60)
    ]

    splits, metadata = MODULE.partition_records(
        records,
        strategy=strategy,
        fold_count=5,
        fold_index=1,
    )

    for field in ("entity", "article"):
        values = {
            name: {getattr(record, field).casefold() for record in split}
            for name, split in splits.items()
        }
        assert values["train"].isdisjoint(values["calibration"])
        assert set(values) == {"train", "calibration"}
    assert metadata["strategy"] == strategy


@pytest.mark.parametrize("strategy", ["group_k_fold", "leave_one_group_out"])
def test_rotating_calibration_folds_cover_every_component_once(strategy: str) -> None:
    records = [
        MODULE.Record(
            entity=f"Entity {index}",
            article=f"Article {index}",
            label=index % 2,
            source=str(index),
        )
        for index in range(100)
    ]

    held_out = []
    fold_count = 5 if strategy == "group_k_fold" else len(records)
    for fold_index in range(fold_count):
        splits, _ = MODULE.partition_records(
            records,
            strategy=strategy,
            fold_count=5,
            fold_index=fold_index,
        )
        held_out.append({record.source for record in splits["calibration"]})

    assert set().union(*held_out) == {record.source for record in records}
    assert sum(len(fold) for fold in held_out) == len(records)


def test_training_option_order_is_deterministic_and_balanced() -> None:
    orders = [MODULE.training_option_order(f"article-{index}") for index in range(100)]

    assert orders == [
        MODULE.training_option_order(f"article-{index}") for index in range(100)
    ]
    assert set(map(tuple, orders)) == {(0, 1), (1, 0)}
    assert 35 <= orders.count([1, 0]) <= 65


def test_ordered_choice_target_follows_marker_order() -> None:
    assert MODULE.ordered_choice_target(0, [0, 1]) == [1.0, 0.0]
    assert MODULE.ordered_choice_target(0, [1, 0]) == [0.0, 1.0]
    assert MODULE.ordered_choice_target(1, [1, 0]) == [1.0, 0.0]


def test_balanced_class_weights_equalize_aggregate_weight() -> None:
    labels = [0, 1, 1, 1]
    weights = MODULE.balanced_class_weights(labels)

    assert weights == [2.0, 2 / 3]
    assert weights[0] == sum(weights[label] for label in labels if label == 0)
    assert abs(weights[0] - sum(weights[label] for label in labels if label == 1)) < 1e-12


def test_validation_improvement_accepts_every_strictly_lower_loss() -> None:
    assert MODULE.validation_improved(0.89, 0.90) is True
    assert MODULE.validation_improved(0.8995, 0.90) is True
    assert MODULE.validation_improved(0.90, 0.90) is False


def test_adaptive_epoch_limit_defaults_to_planned_and_rejects_lower_cap() -> None:
    assert MODULE.adaptive_epoch_limit(10, 0) == 10
    assert MODULE.adaptive_epoch_limit(10, 14) == 14
    with pytest.raises(ValueError, match="at least the planned epochs"):
        MODULE.adaptive_epoch_limit(10, 8)


def test_early_stopping_respects_minimum_epochs_and_patience() -> None:
    assert MODULE.should_stop_early(3, 2, minimum_epochs=4, patience=2) is False
    assert MODULE.should_stop_early(4, 1, minimum_epochs=4, patience=2) is False
    assert MODULE.should_stop_early(4, 2, minimum_epochs=4, patience=2) is True
    assert MODULE.should_stop_early(10, 10, minimum_epochs=4, patience=0) is False


def test_epoch_budget_extends_only_for_new_improving_boundary_best() -> None:
    improving = [
        {"validation_loss": 0.40},
        {"validation_loss": 0.35},
        {"validation_loss": 0.32},
    ]
    plateau = [
        {"validation_loss": 0.40},
        {"validation_loss": 0.35},
        {"validation_loss": 0.351},
    ]

    assert MODULE.extended_epoch_budget(improving, 10, 14, 2, 0.001) == 12
    assert MODULE.extended_epoch_budget(improving, 12, 14, 2, 0.001) == 14
    assert MODULE.extended_epoch_budget(improving, 14, 14, 2, 0.001) == 14
    assert MODULE.extended_epoch_budget(plateau, 10, 14, 2, 0.001) == 10


def test_train_parser_keeps_adaptive_epochs_opt_in() -> None:
    args = MODULE.build_parser().parse_args(["train"])

    assert args.epochs == 4
    assert args.max_epochs == 0
    assert args.minimum_epochs == 1
    assert args.early_stopping_patience == 0
    assert args.early_stopping_min_delta == 0.0
    assert args.extension_epochs == 0


def test_upload_checkpoint_excludes_rolling_checkpoints(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", token))

        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

    MODULE.upload_checkpoint(tmp_path, "owner/model", private=True)

    assert calls[0] == ("token", "secret")
    assert calls[1] == (
        "create",
        {"repo_id": "owner/model", "private": True, "exist_ok": True},
    )
    assert calls[2][1]["repo_id"] == "owner/model"
    assert calls[2][1]["ignore_patterns"] == [
        "checkpoint_best/**",
        "checkpoint_latest/**",
    ]


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


def test_fine_tuning_config_matches_released_recipe() -> None:
    configured = MODULE.fine_tuning_config(
        {"max_len": 512, "head_max_len": 192, "temperature": [1.0, 1.0, 1.0]}
    )

    assert configured["max_len"] == 1024
    assert configured["head_max_len"] == 256
    assert configured["max_tokens_per_batch"] == 4096
    assert configured["gradient_checkpointing"] is True
    assert configured["temperature"] == [1.0, 1.0, 1.0]


def test_no_init_weights_context_supports_installed_transformers() -> None:
    with MODULE.no_init_weights_context():
        pass


def test_t4_uses_fp16_instead_of_emulated_bfloat16() -> None:
    class FakeCuda:
        @staticmethod
        def get_device_capability(device_index):
            assert device_index == 0
            return 7, 5

        @staticmethod
        def is_bf16_supported():
            return True

    assert MODULE.cuda_supports_native_bfloat16(FakeCuda, 0) is False


def test_ampere_uses_bfloat16_when_supported() -> None:
    class FakeCuda:
        @staticmethod
        def get_device_capability(device_index):
            assert device_index == 1
            return 8, 0

        @staticmethod
        def is_bf16_supported():
            return True

    assert MODULE.cuda_supports_native_bfloat16(FakeCuda, 1) is True


def test_validate_prepared_config_rejects_stale_sequence_lengths(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"dataset_schema_version": 3, "label_source": "original_human_disposition", '
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


def test_validate_prepared_config_accepts_external_labels(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"dataset_schema_version": 3, "label_source": "external_labels", '
        '"dataset_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"sequence_config": {"max_len": 1024, "head_max_len": 256}}',
        encoding="utf-8",
    )

    MODULE.validate_prepared_config(
        tmp_path,
        {"max_len": 1024, "head_max_len": 256},
    )