from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = (
    "run_id",
    "status",
    "strategy",
    "partition",
    "seed",
    "train_examples",
    "calibration_examples",
    "planned_epochs",
    "max_epochs",
    "epochs_completed",
    "stopping_reason",
    "best_epoch",
    "best_train_loss",
    "best_validation_loss",
    "best_train_accuracy",
    "best_validation_accuracy",
    "temperature",
    "training_seconds",
    "benchmark_examples",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "model_dir",
    "ledger_sha256",
    "log",
    "error",
)
AGGREGATE_COLUMNS = (
    "strategy",
    "partition",
    "runs",
    "calibration_examples",
    "mean_validation_loss",
    "std_validation_loss",
    "mean_validation_accuracy",
    "mean_best_epoch",
    "mean_epochs_completed",
    "early_stopped_runs",
    "extended_runs",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def selected_epoch(report: dict[str, Any]) -> dict[str, Any]:
    best_epoch = report.get("best_epoch")
    history = report.get("history") or []
    return next(
        (
            row
            for row in history
            if isinstance(row, dict) and row.get("epoch") == best_epoch
        ),
        {},
    )


def row_for(run_file: Path) -> dict[str, Any]:
    run = load_json(run_file)
    model_dir = Path(run.get("model_dir", ""))
    prepared_dir = Path(run.get("prepared_dir", ""))
    benchmark_report = Path(run.get("benchmark_report", ""))
    training = load_json(model_dir / "training_report.json") if model_dir else {}
    manifest = load_json(prepared_dir / "manifest.json") if prepared_dir else {}
    benchmark = load_json(benchmark_report) if benchmark_report else {}
    selected = selected_epoch(training)
    splits = manifest.get("splits") or {}
    metrics = benchmark.get("metrics") or {}
    latency = metrics.get("latency_seconds") or {}
    history = training.get("history") or []
    training_control = training.get("training_control") or {}
    return {
        "run_id": run.get("run_id", run_file.parent.name),
        "status": run.get("status", "unknown"),
        "strategy": run.get("strategy"),
        "partition": (manifest.get("partition") or {}).get("strategy", run.get("partition")),
        "seed": run.get("seed"),
        "train_examples": (splits.get("train") or {}).get("records"),
        "calibration_examples": (splits.get("calibration") or {}).get("records"),
        "planned_epochs": training_control.get("planned_epochs", len(history) or None),
        "max_epochs": training_control.get("max_epochs", len(history) or None),
        "epochs_completed": training_control.get("epochs_completed", len(history) or None),
        "stopping_reason": training_control.get("stopping_reason"),
        "best_epoch": training.get("best_epoch"),
        "best_train_loss": selected.get("train_loss"),
        "best_validation_loss": selected.get("validation_loss"),
        "best_train_accuracy": selected.get("train_accuracy"),
        "best_validation_accuracy": selected.get("validation_accuracy"),
        "temperature": training.get("temperature_choice"),
        "training_seconds": history[-1].get("elapsed_seconds") if history else None,
        "benchmark_examples": metrics.get("examples"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "mean_latency_ms": 1000 * latency["mean"] if "mean" in latency else None,
        "median_latency_ms": 1000 * latency["median"] if "median" in latency else None,
        "p95_latency_ms": 1000 * latency["p95"] if "p95" in latency else None,
        "model_dir": str(model_dir) if model_dir else None,
        "ledger_sha256": benchmark.get("ledger_sha256"),
        "log": run.get("log"),
        "error": run.get("error"),
    }


def write_table(matrix_root: Path, output: Path) -> list[dict[str, Any]]:
    rows = [row_for(path) for path in (matrix_root / "runs").glob("*/run.json")]
    rows.sort(
        key=lambda row: (
            row["best_validation_loss"] is None,
            row["best_validation_loss"] or float("inf"),
            str(row["run_id"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return rows


def weighted_mean(values: list[tuple[float, float]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if not values or total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete" or row.get("best_validation_loss") is None:
            continue
        groups[(str(row.get("strategy")), str(row.get("partition")))].append(row)

    aggregates = []
    for (strategy, partition), group in groups.items():
        weights = [float(row.get("calibration_examples") or 1) for row in group]
        losses = [float(row["best_validation_loss"]) for row in group]
        mean_loss = weighted_mean(list(zip(losses, weights)))
        variance = weighted_mean([
            ((loss - mean_loss) ** 2, weight)
            for loss, weight in zip(losses, weights)
        ])
        accuracies = [
            (float(row["best_validation_accuracy"]), weight)
            for row, weight in zip(group, weights)
            if row.get("best_validation_accuracy") is not None
        ]
        best_epochs = [
            (float(row["best_epoch"]), weight)
            for row, weight in zip(group, weights)
            if row.get("best_epoch") is not None
        ]
        completed_epochs = [
            (float(row["epochs_completed"]), weight)
            for row, weight in zip(group, weights)
            if row.get("epochs_completed") is not None
        ]
        aggregates.append({
            "strategy": strategy,
            "partition": partition,
            "runs": len(group),
            "calibration_examples": int(sum(weights)),
            "mean_validation_loss": mean_loss,
            "std_validation_loss": math.sqrt(variance or 0.0),
            "mean_validation_accuracy": weighted_mean(accuracies),
            "mean_best_epoch": weighted_mean(best_epochs),
            "mean_epochs_completed": weighted_mean(completed_epochs),
            "early_stopped_runs": sum(
                row.get("stopping_reason") == "early_stopping" for row in group
            ),
            "extended_runs": sum(
                (row.get("epochs_completed") or 0) > (row.get("planned_epochs") or 0)
                for row in group
            ),
        })
    aggregates.sort(key=lambda row: (row["mean_validation_loss"], row["strategy"]))
    return aggregates


def write_aggregate_table(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    aggregates = aggregate_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=AGGREGATE_COLUMNS, dialect="excel-tab"
        )
        writer.writeheader()
        writer.writerows(aggregates)
    temporary.replace(output)
    return aggregates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the calibration-ranked LayaCrime exploration table"
    )
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = write_table(args.matrix_root, args.output)
    aggregates = (
        write_aggregate_table(rows, args.aggregate_output)
        if args.aggregate_output
        else []
    )
    print(json.dumps({
        "rows": len(rows),
        "output": str(args.output),
        "aggregate_rows": len(aggregates),
        "aggregate_output": str(args.aggregate_output) if args.aggregate_output else None,
    }))


if __name__ == "__main__":
    main()
