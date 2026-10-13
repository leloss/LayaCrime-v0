import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_script("benchmark_laya_checkpoint.py")
SUMMARY = load_script("summarize_public_experiment_matrix.py")
ALL_CHECKPOINTS = load_script("benchmark_all_laya_checkpoints.py")
RELEASE = load_script("check_release_contents.py")


@pytest.mark.parametrize("relative", [
    "paper/layacrime.tex",
    "artifacts/benchmark-runs/run.json",
    "models/fine-tuned/run/model.safetensors",
    "datasets/other-bundle/corpus.jsonl",
    "datasets/adverse-media-public-tuning-2000/annotations/extra.jsonl",
    "datasets/adverse-media-public-tuning-2000/training-subsets/human/manifest.json",
])
def test_release_rejects_private_paths(relative: str) -> None:
    assert RELEASE.violation(ROOT / relative)
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative], cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0


def test_release_keeps_article_training_and_selection_files() -> None:
    for relative in (
        "scripts/run_public_experiment_matrix.sh",
        "scripts/fine_tune_laya.py",
        "scripts/summarize_public_experiment_matrix.py",
        "scripts/benchmark_all_laya_checkpoints.py",
    ):
        assert relative in RELEASE.REQUIRED_RELEASE_FILES
        assert RELEASE.violation(ROOT / relative) is None


def make_batch_run(matrix: Path, run_id: str, loss: float = 0.3) -> Path:
    model = matrix / "models" / run_id
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"test-weights")
    (model / "rl_agent_config.json").write_text(json.dumps({"training": {"seed": 17}}), encoding="utf-8")
    (model / "data_manifest.json").write_text(json.dumps({
        "partition": {"strategy": "group_k_fold", "fold_count": 5, "calibration_fold": 0},
        "splits": {"train": {"records": 1600}, "calibration": {"records": 400}},
    }), encoding="utf-8")
    (model / "training_report.json").write_text(json.dumps({
        "strategy": "test-strategy", "best_epoch": 1, "best_validation_loss": loss,
        "history": [{"epoch": 1, "validation_loss": loss, "validation_accuracy": 0.8}],
    }), encoding="utf-8")
    return model


def test_batch_inventory_does_not_read_benchmark_scores(tmp_path: Path) -> None:
    make_batch_run(tmp_path, "run-b", 0.2)
    make_batch_run(tmp_path, "run-a", 0.4)
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "run-a.json").write_text("not valid JSON", encoding="utf-8")
    incomplete = tmp_path / "runs" / "missing-run"
    incomplete.mkdir(parents=True)
    (incomplete / "run.json").write_text("{}", encoding="utf-8")

    rows, skipped = ALL_CHECKPOINTS.discover_runs(tmp_path)

    assert [row["run_id"] for row in rows] == ["run-a", "run-b"]
    assert rows[0]["best_validation_loss"] == 0.4
    assert rows[0]["calibration_examples"] == 400
    assert skipped[0]["run_id"] == "missing-run"
    assert not set(ALL_CHECKPOINTS.VALIDATION_COLUMNS) & {"accuracy", "precision", "recall", "f1"}


def test_batch_rejects_unknown_selection_scope() -> None:
    with pytest.raises(ValueError, match="all candidates"):
        ALL_CHECKPOINTS.select_runs([], "unknown", minimum_runs=1)
    with pytest.raises(SystemExit) as failure:
        ALL_CHECKPOINTS.build_parser().parse_args(["--selection-scope", "unknown"])
    assert failure.value.code == 2


def test_batch_selects_winner_from_selection_f1_only() -> None:
    validation = [
        {"run_id": "higher-f1", "best_validation_loss": 0.4},
        {"run_id": "lower-loss", "best_validation_loss": 0.2},
        {"run_id": "failed", "best_validation_loss": 0.1},
    ]
    selection = [
        {"run_id": "higher-f1", "status": "complete", "f1": 0.91, "accuracy": 0.90},
        {"run_id": "lower-loss", "status": "complete", "f1": 0.90, "accuracy": 0.92},
        {"run_id": "failed", "status": "failed", "f1": 0.99, "accuracy": 0.99},
    ]

    winner = ALL_CHECKPOINTS.select_benchmark_winner(validation, selection)

    assert winner["run_id"] == "higher-f1"
    assert winner["selection_f1"] == 0.91


def test_batch_defaults_to_new_selection_and_established_benchmark() -> None:
    args = ALL_CHECKPOINTS.build_parser().parse_args([])

    assert args.selection_scope == "matrix"
    assert args.minimum_runs == 1
    assert args.corpus.parent.name == "adverse-media-public-selection-1000"
    assert args.labels.parent.parent.name == "adverse-media-public-selection-1000"
    assert args.test_corpus.parent.name == "adverse-media-public-holdout-1000"
    assert args.test_labels.parent.parent.name == "adverse-media-public-holdout-1000"


def test_batch_matrix_scope_excludes_legacy_candidates() -> None:
    rows = [
        {"run_id": "matrix", "partition": "group_k_fold"},
        {"run_id": "legacy", "partition": "holdout_80_20"},
    ]

    assert ALL_CHECKPOINTS.select_runs(rows, "matrix", minimum_runs=15) == [rows[0]]


def test_batch_breaks_selection_ties_without_test_metrics() -> None:
    validation = [
        {"run_id": "higher-loss", "best_validation_loss": 0.3},
        {"run_id": "lower-loss", "best_validation_loss": 0.2},
    ]
    selection = [
        {"run_id": "higher-loss", "status": "complete", "f1": 0.9, "accuracy": 0.91},
        {"run_id": "lower-loss", "status": "complete", "f1": 0.9, "accuracy": 0.91},
    ]

    assert ALL_CHECKPOINTS.select_benchmark_winner(validation, selection)["run_id"] == "lower-loss"


def test_batch_rejects_mixed_checkpoint_ledgers(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({
        "article_id": "a", "gold_label": 1, "decision": "positive",
        "elapsed_seconds": 0.1, "routing": {"model": "wrong-run"},
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different checkpoint"):
        ALL_CHECKPOINTS.validate_ledger(ledger, "expected-run", {"a": 1})


def test_batch_lock_does_not_remove_an_existing_owner(tmp_path: Path) -> None:
    lock = tmp_path / ".lock"
    lock.mkdir()
    owner = lock / "owner"
    owner.write_text("someone-else", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        with ALL_CHECKPOINTS.matrix_lock(tmp_path):
            pytest.fail("Must not enter an existing lock")
    assert owner.read_text(encoding="utf-8") == "someone-else"


def test_batch_prefers_configured_training_environment(
    tmp_path: Path, monkeypatch
) -> None:
    environment = tmp_path / "venv"
    executable = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setenv("VENV_DIR", str(environment))

    assert ALL_CHECKPOINTS.benchmark_python() == executable.resolve()


def test_batch_runs_every_checkpoint_and_resumes(tmp_path: Path, monkeypatch) -> None:
    make_batch_run(tmp_path, "run-a")
    make_batch_run(tmp_path, "run-b")
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text(json.dumps({"article_id": "a", "article": "text", "entity_name": "entity"}) + "\n", encoding="utf-8")
    labels.write_text(json.dumps({"article_id": "a", "label": 1}) + "\n", encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path), "--corpus", str(corpus),
        "--labels", str(labels), "--device", "cpu",
    ])
    assert args.selection_scope == "matrix"
    calls = []

    def fake_benchmark(command, **kwargs):
        assert (tmp_path / "benchmarks" / "selection-checkpoints" / "inputs.json").is_file()
        assert (tmp_path / "benchmarks" / "selection-checkpoints" / "validation-runs.tsv").is_file()
        run_id = command[command.index("--run-id") + 1]
        model = Path(command[command.index("--model-dir") + 1])
        output = Path(command[command.index("--output") + 1])
        report = Path(command[command.index("--report") + 1])
        calls.append(run_id)
        output.write_text(json.dumps({
            "article_id": "a", "gold_label": 1, "decision": "positive",
            "elapsed_seconds": 0.1, "routing": {"model": run_id},
        }) + "\n", encoding="utf-8")
        BENCHMARK.write_json(report, {
            "run_id": run_id, "status": "complete", "ledger_sha256": BENCHMARK.sha256(output),
            "model_weights_sha256": BENCHMARK.sha256(model / "model.safetensors"),
            "model_config_sha256": BENCHMARK.sha256(model / "rl_agent_config.json"),
            "corpus_sha256": BENCHMARK.sha256(corpus), "labels_sha256": BENCHMARK.sha256(labels),
        })

    monkeypatch.setattr(ALL_CHECKPOINTS.subprocess, "run", fake_benchmark)

    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == ["run-a", "run-b"]
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == ["run-a", "run-b"]
    make_batch_run(tmp_path, "run-c")
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == ["run-a", "run-b", "run-c"]
    result = tmp_path / "benchmarks" / "selection-checkpoints" / "benchmark-results.tsv"
    with result.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, dialect="excel-tab"))
    assert [row["run_id"] for row in rows] == calls
    assert float(rows[0]["pairs_per_second"]) == 10
    assert float(rows[0]["f1"]) == 0.5
    frozen_validation = (result.parent / "validation-runs.tsv").read_bytes()
    (tmp_path / "models" / "run-a" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Benchmarked checkpoint changed or disappeared"):
        ALL_CHECKPOINTS.run_batch(args)
    assert (result.parent / "validation-runs.tsv").read_bytes() == frozen_validation


@pytest.mark.parametrize("preliminary", [False, True])
def test_batch_finalizes_selection_winner_on_test_set_only(
    tmp_path: Path, monkeypatch, preliminary: bool
) -> None:
    make_batch_run(tmp_path, "run-a", 0.4)
    make_batch_run(tmp_path, "run-b", 0.2)
    selection_corpus = tmp_path / "selection-corpus.jsonl"
    selection_labels = tmp_path / "selection-labels.jsonl"
    test_corpus = tmp_path / "test-corpus.jsonl"
    test_labels = tmp_path / "test-labels.jsonl"
    corpus_text = "\n".join(
        json.dumps({"article_id": identifier, "article": identifier, "entity_name": identifier})
        for identifier in ("positive", "negative")
    ) + "\n"
    labels_text = "\n".join(
        json.dumps({"article_id": identifier, "label": label})
        for identifier, label in (("positive", 1), ("negative", 2))
    ) + "\n"
    selection_corpus.write_text(corpus_text, encoding="utf-8")
    selection_labels.write_text(labels_text, encoding="utf-8")
    test_corpus.write_text(corpus_text, encoding="utf-8")
    test_labels.write_text(labels_text, encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path),
        "--corpus", str(selection_corpus),
        "--labels", str(selection_labels),
        "--test-corpus", str(test_corpus),
        "--test-labels", str(test_labels),
        "--device", "cpu",
        "--finalize-winner",
        "--expected-selection-runs", "210" if preliminary else "2",
        *(["--benchmark-strategy-winners"] if preliminary else []),
    ])
    calls = []

    def fake_benchmark(command, **kwargs):
        run_id = command[command.index("--run-id") + 1]
        corpus = Path(command[command.index("--corpus") + 1])
        labels = Path(command[command.index("--labels") + 1])
        model = Path(command[command.index("--model-dir") + 1])
        ledger = Path(command[command.index("--output") + 1])
        report = Path(command[command.index("--report") + 1])
        dataset = "test" if corpus == test_corpus.resolve() else "selection"
        calls.append((dataset, run_id))
        decisions = (
            {"positive": "positive", "negative": "negative"}
            if run_id != "run-b"
            else {"positive": "negative", "negative": "negative"}
        )
        ledger.write_text("\n".join(json.dumps({
            "article_id": identifier,
            "gold_label": label,
            "decision": decisions[identifier],
            "elapsed_seconds": 0.1,
            "routing": {"model": run_id},
        }) for identifier, label in (("positive", 1), ("negative", 2))) + "\n", encoding="utf-8")
        BENCHMARK.write_json(report, {
            "run_id": run_id,
            "status": "complete",
            "ledger_sha256": BENCHMARK.sha256(ledger),
            "model_weights_sha256": BENCHMARK.sha256(model / "model.safetensors"),
            "model_config_sha256": BENCHMARK.sha256(model / "rl_agent_config.json"),
            "corpus_sha256": BENCHMARK.sha256(corpus),
            "labels_sha256": BENCHMARK.sha256(labels),
        })

    monkeypatch.setattr(ALL_CHECKPOINTS.subprocess, "run", fake_benchmark)

    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == [("selection", "run-a"), ("selection", "run-b"), ("test", "run-a")]
    output = tmp_path / "benchmarks" / "selection-checkpoints"
    if preliminary:
        assert not (output / "selection-winner.json").exists()
        assert not (tmp_path / "benchmarks" / "final-test-winners").exists()
        winner_path = (
            tmp_path / "benchmarks" / "strategy-benchmark-winners"
            / "test-strategy__config-legacy" / "current-winner.json"
        )
    else:
        winner_path = output / "selection-winner.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    assert winner["run_id"] == "run-a"
    if preliminary:
        assert winner["status"] == "provisional"
    current = json.loads(
        (output / "current-selection-leader.json").read_text(encoding="utf-8")
    )
    assert current["status"] == ("provisional" if preliminary else "complete")
    assert current["run_id"] == "run-a"
    assert current["completed_grouped_runs"] == 2
    assert not (tmp_path / "benchmarks" / "final-test-winners" / "run-b.jsonl").exists()
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == [("selection", "run-a"), ("selection", "run-b"), ("test", "run-a")]
    if preliminary:
        make_batch_run(tmp_path, "run-c", 0.1)
        assert ALL_CHECKPOINTS.run_batch(args) == 0
        assert calls[-2:] == [("selection", "run-c"), ("test", "run-c")]
        assert ALL_CHECKPOINTS.run_batch(args) == 0
        assert len(calls) == 5
        snapshot = json.loads((winner_path.parent / "inputs.json").read_text(encoding="utf-8"))
        assert [row["run_id"] for row in snapshot["runs"]] == ["run-a", "run-c"]
        current_strategy = json.loads(winner_path.read_text(encoding="utf-8"))
        assert current_strategy["run_id"] == "run-c"


def test_batch_writes_preliminary_leader_before_finalization_gate(
    tmp_path: Path, monkeypatch
) -> None:
    make_batch_run(tmp_path, "run-a", 0.4)
    make_batch_run(tmp_path, "run-b", 0.2)
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text('{"article_id": "a"}\n', encoding="utf-8")
    labels.write_text('{"article_id": "a", "label": 1}\n', encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path),
        "--corpus", str(corpus),
        "--labels", str(labels),
        "--device", "cpu",
        "--finalize-winner",
        "--expected-selection-runs", "3",
    ])

    monkeypatch.setattr(ALL_CHECKPOINTS, "benchmark_records", lambda *args, **kwargs: ([
        {"run_id": "run-a", "status": "complete", "f1": 0.9, "accuracy": 0.9},
        {"run_id": "run-b", "status": "complete", "f1": 0.8, "accuracy": 0.8},
    ], 0))

    assert ALL_CHECKPOINTS.run_batch(args) == 0

    output = tmp_path / "benchmarks" / "selection-checkpoints"
    current = json.loads((output / "current-selection-leader.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert current["status"] == "provisional"
    assert current["run_id"] == "run-a"
    assert current["completed_grouped_runs"] == 2
    assert current["expected_grouped_runs"] == 3
    assert completion["completed_runs"] == 2
    assert not (output / "selection-winner.json").exists()
    assert not (tmp_path / "benchmarks" / "final-test-winners").exists()


def test_batch_benchmarks_partial_strategy_winners_without_finalizing(
    tmp_path: Path, monkeypatch
) -> None:
    for run_id, strategy, loss in (
        ("run-a", "strategy-a", 0.4),
        ("run-b", "strategy-a", 0.2),
        ("run-c", "strategy-b", 0.3),
    ):
        model = make_batch_run(tmp_path, run_id, loss)
        report = json.loads((model / "training_report.json").read_text(encoding="utf-8"))
        report["strategy"] = strategy
        (model / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text('{"article_id": "a"}\n', encoding="utf-8")
    labels.write_text('{"article_id": "a", "label": 1}\n', encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path), "--corpus", str(corpus),
        "--labels", str(labels), "--device", "cpu",
        "--finalize-winner", "--benchmark-strategy-winners",
    ])
    selection = [
        {"run_id": "run-a", "status": "complete", "f1": 0.9, "accuracy": 0.9},
        {"run_id": "run-b", "status": "complete", "f1": 0.8, "accuracy": 0.8},
        {"run_id": "run-c", "status": "complete", "f1": 0.85, "accuracy": 0.85},
    ]
    monkeypatch.setattr(ALL_CHECKPOINTS, "benchmark_records", lambda *args, **kwargs: (selection, 0))
    calls = []

    def fake_comparison(comparison_args, winner, record, *args, provisional=False):
        assert provisional
        assert "strategy-benchmark-winners" in comparison_args.test_output_dir.parts
        assert record["run_id"] == winner["run_id"]
        calls.append(winner["run_id"])
        return 0

    monkeypatch.setattr(ALL_CHECKPOINTS, "benchmark_final_winner", fake_comparison)
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert calls == ["run-a", "run-c"]
    manifest = json.loads((
        tmp_path / "benchmarks" / "strategy-benchmark-winners" / "current-strategy-winners.json"
    ).read_text(encoding="utf-8"))
    assert manifest["status"] == "provisional"
    assert [row["completed_strategy_runs"] for row in manifest["winners"]] == [2, 1]
    assert not (tmp_path / "benchmarks" / "selection-checkpoints" / "selection-winner.json").exists()
    assert not (tmp_path / "benchmarks" / "final-test-winners").exists()


def test_batch_treats_unfinished_exports_as_nonfatal_for_rolling_results(
    tmp_path: Path, monkeypatch
) -> None:
    make_batch_run(tmp_path, "run-a")
    unfinished = tmp_path / "runs" / "run-b"
    unfinished.mkdir(parents=True)
    (unfinished / "run.json").write_text('{"status": "running"}', encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text('{"article_id": "a"}\n', encoding="utf-8")
    labels.write_text('{"article_id": "a", "label": 1}\n', encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path),
        "--corpus", str(corpus),
        "--labels", str(labels),
        "--device", "cpu",
    ])

    monkeypatch.setattr(ALL_CHECKPOINTS, "benchmark_records", lambda *args, **kwargs: ([
        {"run_id": "run-a", "status": "complete", "f1": 0.9, "accuracy": 0.9},
    ], 0))

    assert ALL_CHECKPOINTS.run_batch(args) == 0
    output = tmp_path / "benchmarks" / "selection-checkpoints"
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    current = json.loads((output / "current-selection-leader.json").read_text(encoding="utf-8"))
    assert inventory["skipped"] == [
        {"run_id": "run-b", "reason": "training is still marked running"}
    ]
    assert current["status"] == "provisional"


def test_checkpoint_sha256_streams_weights(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "weights"
    content = b"weights" * 200000
    weights.write_bytes(content)

    def fail_read_bytes(path):
        pytest.fail("Weight hashing must not load the entire checkpoint into memory")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    assert BENCHMARK.sha256(weights) == hashlib.sha256(content).hexdigest()


def test_batch_continues_after_failure_and_retries_partial_ledger(tmp_path: Path, monkeypatch) -> None:
    make_batch_run(tmp_path, "run-a")
    make_batch_run(tmp_path, "run-b")
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text("\n".join(json.dumps({"article_id": identifier}) for identifier in ("a", "b")) + "\n", encoding="utf-8")
    labels.write_text("\n".join(json.dumps({"article_id": identifier, "label": 1}) for identifier in ("a", "b")) + "\n", encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path), "--corpus", str(corpus),
        "--labels", str(labels), "--device", "cpu", "--selection-scope", "all",
    ])
    attempts = []

    def fake_benchmark(command, **kwargs):
        run_id = command[command.index("--run-id") + 1]
        model = Path(command[command.index("--model-dir") + 1])
        ledger = Path(command[command.index("--output") + 1])
        report = Path(command[command.index("--report") + 1])
        attempts.append(run_id)
        previous = BENCHMARK.read_jsonl(ledger) if ledger.exists() else []
        if run_id == "run-a" and attempts.count(run_id) == 1:
            ledger.write_text(json.dumps({
                "article_id": "a", "gold_label": 1, "decision": "positive",
                "elapsed_seconds": 0.1, "routing": {"model": run_id},
            }) + "\n", encoding="utf-8")
            raise subprocess.CalledProcessError(1, command)
        if run_id == "run-a":
            assert len(previous) == 1
            assert "--resume" in command
        completed = {row["article_id"] for row in previous}
        with ledger.open("a", encoding="utf-8") as destination:
            for identifier in ("a", "b"):
                if identifier not in completed:
                    destination.write(json.dumps({
                        "article_id": identifier, "gold_label": 1, "decision": "positive",
                        "elapsed_seconds": 0.1, "routing": {"model": run_id},
                    }) + "\n")
        BENCHMARK.write_json(report, {
            "run_id": run_id, "status": "complete", "ledger_sha256": BENCHMARK.sha256(ledger),
            "model_weights_sha256": BENCHMARK.sha256(model / "model.safetensors"),
            "model_config_sha256": BENCHMARK.sha256(model / "rl_agent_config.json"),
            "corpus_sha256": BENCHMARK.sha256(corpus), "labels_sha256": BENCHMARK.sha256(labels),
        })

    monkeypatch.setattr(ALL_CHECKPOINTS.subprocess, "run", fake_benchmark)
    assert ALL_CHECKPOINTS.run_batch(args) == 1
    assert attempts == ["run-a", "run-b"]
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert attempts == ["run-a", "run-b", "run-a"]
    assert not (tmp_path / ".lock").exists()


def test_batch_dry_run_never_starts_inference(tmp_path: Path, monkeypatch) -> None:
    make_batch_run(tmp_path, "run-a")
    corpus = tmp_path / "corpus.jsonl"
    labels = tmp_path / "labels.jsonl"
    corpus.write_text('{"article_id": "a"}\n', encoding="utf-8")
    labels.write_text('{"article_id": "a", "label": 1}\n', encoding="utf-8")
    args = ALL_CHECKPOINTS.build_parser().parse_args([
        "--matrix-root", str(tmp_path), "--corpus", str(corpus),
        "--labels", str(labels), "--dry-run", "--selection-scope", "all",
    ])

    def fail_subprocess(*args, **kwargs):
        pytest.fail("Dry run must not start a subprocess or check the GPU")

    monkeypatch.setattr(ALL_CHECKPOINTS.subprocess, "run", fail_subprocess)
    assert ALL_CHECKPOINTS.run_batch(args) == 0
    assert (tmp_path / "benchmarks" / "all-checkpoints" / "validation-aggregate.tsv").exists()
    assert not (tmp_path / "benchmarks" / "all-checkpoints" / "benchmark-results.tsv").exists()


def test_benchmark_metrics_uses_macro_scores_and_annotation_labels() -> None:
    rows = [
        {"gold_label": 1, "decision": "positive", "elapsed_seconds": 1.0},
        {"gold_label": 1, "decision": "negative", "elapsed_seconds": 2.0},
        {"gold_label": 2, "decision": "negative", "elapsed_seconds": 3.0},
        {"gold_label": 2, "decision": "negative", "elapsed_seconds": 4.0},
    ]

    report = BENCHMARK.benchmark_metrics(rows)

    assert report["accuracy"] == 0.75
    assert report["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert report["recall"] == 0.75
    assert report["f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert report["precision_positive"] == 1.0
    assert report["recall_positive"] == 0.5
    assert report["precision_negative"] == pytest.approx(2 / 3)
    assert report["recall_negative"] == 1.0
    assert report["true_positive"] == 1
    assert report["false_positive"] == 0
    assert report["true_negative"] == 2
    assert report["false_negative"] == 1
    assert report["latency_seconds"] == {"mean": 2.5, "median": 2.5, "p95": 4.0}


def test_summary_writes_training_values(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    run_dir = matrix / "runs" / "run-one"
    model_dir = matrix / "models" / "run-one"
    prepared_dir = matrix / "prepared" / "holdout"
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": "run-one",
        "status": "complete",
        "strategy": "public_natural_staged",
        "partition": "holdout_80_20",
        "seed": 17,
        "prepared_dir": str(prepared_dir),
        "model_dir": str(model_dir),
        "log": "run.log",
    }), encoding="utf-8")
    (prepared_dir / "manifest.json").write_text(json.dumps({
        "partition": {"strategy": "holdout_80_20"},
        "splits": {"train": {"records": 1600}, "calibration": {"records": 400}},
    }), encoding="utf-8")
    (model_dir / "training_report.json").write_text(json.dumps({
        "history": [
            {"epoch": 1, "train_loss": 0.5, "validation_loss": 0.4, "train_accuracy": 0.8, "validation_accuracy": 0.75, "elapsed_seconds": 10},
            {"epoch": 2, "train_loss": 0.3, "validation_loss": 0.35, "train_accuracy": 0.9, "validation_accuracy": 0.82, "elapsed_seconds": 20},
        ],
        "best_epoch": 2,
        "training_control": {
            "planned_epochs": 10,
            "max_epochs": 14,
            "epochs_completed": 2,
            "stopping_reason": "early_stopping",
        },
        "temperature_choice": 1.2,
    }), encoding="utf-8")
    output = matrix / "results.tsv"

    rows = SUMMARY.write_table(matrix, output)

    assert rows[0]["best_epoch"] == 2
    assert rows[0]["config_id"] is None
    assert rows[0]["planned_epochs"] == 10
    assert rows[0]["max_epochs"] == 14
    assert rows[0]["epochs_completed"] == 2
    assert rows[0]["stopping_reason"] == "early_stopping"
    assert rows[0]["best_train_loss"] == 0.3
    assert rows[0]["best_validation_loss"] == 0.35
    with output.open(encoding="utf-8", newline="") as source:
        stored = next(csv.DictReader(source, dialect="excel-tab"))
    assert stored["partition"] == "holdout_80_20"
    assert "f1" not in stored

    aggregate_output = matrix / "aggregate.tsv"
    aggregates = SUMMARY.write_aggregate_table(rows, aggregate_output)
    assert aggregates[0]["strategy"] == "public_natural_staged"
    assert aggregates[0]["config_id"] is None
    assert aggregates[0]["runs"] == 1
    assert aggregates[0]["mean_validation_loss"] == 0.35
    assert aggregates[0]["early_stopped_runs"] == 1
    with aggregate_output.open(encoding="utf-8", newline="") as source:
        aggregate = next(csv.DictReader(source, dialect="excel-tab"))
    assert aggregate["mean_validation_accuracy"] == "0.82"


def test_summary_localizes_downloaded_remote_paths(tmp_path: Path) -> None:
    matrix_root = tmp_path / "public-experiment-matrix"
    local_report = matrix_root / "models" / "run" / "training_report.json"
    local_report.parent.mkdir(parents=True)
    local_report.write_text("{}", encoding="utf-8")

    resolved = SUMMARY.local_artifact_path(
        "/home/ubuntu/laya/artifacts/public-experiment-matrix/models/run/training_report.json",
        matrix_root,
    )

    assert resolved == local_report


def test_matrix_script_covers_presets_and_uses_adaptive_training() -> None:
    content = (ROOT / "scripts" / "run_public_experiment_matrix.sh").read_text(
        encoding="utf-8"
    )
    for strategy in (
        "preserve_entity_matching",
        "balanced",
        "maximum_task_adaptation",
        "adaptation_encoder_lr_5e6",
        "adaptation_encoder_lr_2e5",
        "adaptation_head_lr_2e5",
        "adaptation_head_lr_1e4",
        "adaptation_weight_decay_0",
        "adaptation_weight_decay_2e2",
        "relational_low_drift",
        "relational_adaptation",
        "relational_extended",
        "relational_rlcd",
        "public_natural_staged",
    ):
        assert strategy in content
    assert (
        "group_k_fold:5:0 group_k_fold:5:1 group_k_fold:5:2 "
        "group_k_fold:5:3 group_k_fold:5:4"
    ) in content
    assert 'MATRIX_SEEDS="${MATRIX_SEEDS:-20260923 20260924 20260925}"' in content
    assert "holdout-1000" not in content
    assert "results.tsv" in content
    assert "aggregate.tsv" in content
    assert "--aggregate-output" in content
    assert 'MATRIX_PLANNED_EPOCHS="${MATRIX_PLANNED_EPOCHS:-4}"' in content
    assert 'MATRIX_MAX_EPOCHS="${MATRIX_MAX_EPOCHS:-10}"' in content
    assert "--early-stopping-patience" in content
    assert "--early-stopping-min-delta" in content
    assert "--extension-epochs" in content
    assert "existing_config_ids" in content
    assert '"adaptive-v1"' in content
    assert "__config-${CONFIG_ID}__seed-" in content
    assert "MATRIX_ALLOW_SHARED_GPU" in content
    assert "MATRIX_DRY_RUN" in content
    assert "MATRIX_MAX_RUNS" in content
    assert "matrix_start mode=exhaustive" in content
    assert 'run_cells "$MATRIX_STRATEGIES" "$MATRIX_PARTITIONS" "$MATRIX_SEEDS" exhaustive' in content
    assert 'MATRIX_RETRY_FAILED="${MATRIX_RETRY_FAILED:-1}"' in content
    assert "reason=previous_failure" in content


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_matrix_script_has_valid_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "run_public_experiment_matrix.sh")],
        check=True,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_matrix_custom_grid_is_exhaustive(tmp_path: Path) -> None:
    tuning = tmp_path / "tuning"
    model = tmp_path / "model"
    venv = tmp_path / "venv"
    tuning.mkdir()
    model.mkdir()
    (venv / "bin").mkdir(parents=True)
    for name in ("corpus.jsonl", "dataset.json"):
        (tuning / name).touch()
    (tuning / "annotations").mkdir()
    (tuning / "annotations" / "human.jsonl").touch()
    (model / "model.safetensors").touch()
    fake_python = venv / "bin" / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_public_experiment_matrix.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": str(Path(shutil.which("bash")).parent),
            "PROJECT_ROOT": str(tmp_path),
            "VENV_DIR": str(venv),
            "TUNING_DIR": str(tuning),
            "LAYA_MODEL_PATH": str(model),
            "MATRIX_ROOT": str(tmp_path / "matrix"),
            "MATRIX_DRY_RUN": "1",
            "MATRIX_STRATEGIES": (
                "maximum_task_adaptation adaptation_encoder_lr_2e5"
            ),
        },
    )

    plans = [
        line.split(" run=", 1)[1]
        for line in completed.stdout.splitlines()
        if "plan stage=exhaustive index=" in line
    ]
    assert len(plans) == len(set(plans)) == 30
    assert "matrix_plan_complete runs=30" in completed.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_matrix_default_plan_is_exhaustive_and_round_robin(tmp_path: Path) -> None:
    tuning = tmp_path / "tuning"
    model = tmp_path / "model"
    venv = tmp_path / "venv"
    tuning.mkdir()
    model.mkdir()
    (venv / "bin").mkdir(parents=True)
    for name in ("corpus.jsonl", "dataset.json"):
        (tuning / name).touch()
    (tuning / "annotations").mkdir()
    (tuning / "annotations" / "human.jsonl").touch()
    (model / "model.safetensors").touch()
    fake_python = venv / "bin" / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_public_experiment_matrix.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": str(Path(shutil.which("bash")).parent),
            "PROJECT_ROOT": str(tmp_path),
            "VENV_DIR": str(venv),
            "TUNING_DIR": str(tuning),
            "LAYA_MODEL_PATH": str(model),
            "MATRIX_ROOT": str(tmp_path / "matrix"),
            "MATRIX_DRY_RUN": "1",
        },
    )

    plans = [
        line for line in completed.stdout.splitlines()
        if "plan stage=exhaustive index=" in line
    ]
    assert len(plans) == 210
    assert len({line.split(" run=", 1)[1] for line in plans}) == 210
    first_round = plans[:14]
    assert all("group_k_fold-5-0__" in line for line in first_round)
    assert all("seed-20260923" in line for line in first_round)
    assert len({line.split("__")[1] for line in first_round}) == 14
    assert "group_k_fold-5-1__adaptation_encoder_lr_2e5__seed-20260923" in plans[14]
    assert "matrix_plan_complete runs=210" in completed.stdout
