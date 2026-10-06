import csv
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


def test_benchmark_metrics_treats_negative_as_event() -> None:
    rows = [
        {"gold_label": 1, "decision": "positive", "elapsed_seconds": 1.0},
        {"gold_label": 1, "decision": "negative", "elapsed_seconds": 2.0},
        {"gold_label": 2, "decision": "negative", "elapsed_seconds": 3.0},
        {"gold_label": 2, "decision": "negative", "elapsed_seconds": 4.0},
    ]

    report = BENCHMARK.benchmark_metrics(rows)

    assert report["accuracy"] == 0.75
    assert report["precision"] == 2 / 3
    assert report["recall"] == 1.0
    assert report["f1"] == 0.8
    assert report["latency_seconds"] == {"mean": 2.5, "median": 2.5, "p95": 4.0}


def test_summary_writes_training_and_benchmark_values(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    run_dir = matrix / "runs" / "run-one"
    model_dir = matrix / "models" / "run-one"
    prepared_dir = matrix / "prepared" / "holdout"
    benchmark_report = matrix / "benchmarks" / "run-one.json"
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    benchmark_report.parent.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": "run-one",
        "status": "complete",
        "strategy": "public_natural_staged",
        "partition": "holdout_80_20",
        "seed": 17,
        "prepared_dir": str(prepared_dir),
        "model_dir": str(model_dir),
        "benchmark_report": str(benchmark_report),
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
    benchmark_report.write_text(json.dumps({
        "ledger_sha256": "abc",
        "metrics": {
            "examples": 1000,
            "accuracy": 0.83,
            "precision": 0.8,
            "recall": 0.85,
            "f1": 0.824,
            "latency_seconds": {"mean": 0.06, "median": 0.05, "p95": 0.09},
        },
    }), encoding="utf-8")
    output = matrix / "results.tsv"

    rows = SUMMARY.write_table(matrix, output)

    assert rows[0]["best_epoch"] == 2
    assert rows[0]["planned_epochs"] == 10
    assert rows[0]["max_epochs"] == 14
    assert rows[0]["epochs_completed"] == 2
    assert rows[0]["stopping_reason"] == "early_stopping"
    assert rows[0]["best_train_loss"] == 0.3
    assert rows[0]["best_validation_loss"] == 0.35
    assert rows[0]["f1"] == 0.824
    with output.open(encoding="utf-8", newline="") as source:
        stored = next(csv.DictReader(source, dialect="excel-tab"))
    assert stored["partition"] == "holdout_80_20"
    assert stored["mean_latency_ms"] == "60.0"

    aggregate_output = matrix / "aggregate.tsv"
    aggregates = SUMMARY.write_aggregate_table(rows, aggregate_output)
    assert aggregates[0]["strategy"] == "public_natural_staged"
    assert aggregates[0]["runs"] == 1
    assert aggregates[0]["mean_validation_loss"] == 0.35
    assert aggregates[0]["early_stopped_runs"] == 1
    with aggregate_output.open(encoding="utf-8", newline="") as source:
        aggregate = next(csv.DictReader(source, dialect="excel-tab"))
    assert aggregate["mean_validation_accuracy"] == "0.82"


def test_matrix_script_covers_presets_and_uses_adaptive_training() -> None:
    content = (ROOT / "scripts" / "run_public_experiment_matrix.sh").read_text(
        encoding="utf-8"
    )
    for strategy in (
        "preserve_entity_matching",
        "balanced",
        "maximum_task_adaptation",
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
    train = content.index('"$PYTHON" scripts/fine_tune_laya.py train')
    benchmark = content.index('"$PYTHON" scripts/benchmark_laya_checkpoint.py')
    assert train < benchmark
    assert "results.tsv" in content
    assert "aggregate.tsv" in content
    assert "--aggregate-output" in content
    assert 'MATRIX_RUN_BENCHMARK="${MATRIX_RUN_BENCHMARK:-0}"' in content
    assert 'MATRIX_PLANNED_EPOCHS="${MATRIX_PLANNED_EPOCHS:-10}"' in content
    assert 'MATRIX_MAX_EPOCHS="${MATRIX_MAX_EPOCHS:-14}"' in content
    assert "--early-stopping-patience" in content
    assert "--early-stopping-min-delta" in content
    assert "--extension-epochs" in content
    assert '"adaptive-v1"' in content
    assert "__config-${config_id}__seed-" in content
    assert 'event=benchmark_skip reason=exploration_protocol' in content
    assert "MATRIX_ALLOW_SHARED_GPU" in content
    assert "MATRIX_DRY_RUN" in content
    assert "MATRIX_MAX_RUNS" in content
    assert content.index("public_natural_staged balanced") < content.index(
        "relational_low_drift relational_adaptation"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_matrix_script_has_valid_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "run_public_experiment_matrix.sh")],
        check=True,
    )
