from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmark_laya_checkpoint import benchmark_metrics, read_jsonl, sha256, write_json
from summarize_public_experiment_matrix import (
    AGGREGATE_COLUMNS,
    CONFIG_ID_PATTERN,
    aggregate_rows,
    selected_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_COLUMNS = (
    "run_id", "status", "strategy", "config_id", "partition", "fold_count",
    "calibration_fold", "seed", "train_examples", "calibration_examples",
    "best_epoch", "best_validation_loss", "best_validation_accuracy",
    "planned_epochs", "epochs_completed", "stopping_reason", "model_dir",
)
RESULT_COLUMNS = (
    "run_id", "status", "examples", "accuracy", "precision", "recall", "f1",
    "true_positive", "false_positive", "true_negative", "false_negative",
    "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
    "pairs_per_second", "pairs_per_minute", "ledger_sha256", "log", "error",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=columns, dialect="excel-tab", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover_runs(matrix_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    run_files = {path.parent.name: path for path in (matrix_root / "runs").glob("*/run.json")}
    model_dirs = {
        path.name: path for path in (matrix_root / "models").glob("*")
        if path.is_dir() and ".failed-" not in path.name
    }
    candidates = []
    skipped = []
    for run_id in sorted(run_files.keys() | model_dirs.keys()):
        model_dir = matrix_root / "models" / run_id
        try:
            run = load_object(run_files[run_id]) if run_id in run_files else {}
            if run.get("status") == "running":
                raise ValueError("training is still marked running")
            required = ("model.safetensors", "rl_agent_config.json", "training_report.json", "data_manifest.json")
            missing = [name for name in required if not (model_dir / name).is_file()]
            if missing:
                raise ValueError("missing export files: " + ", ".join(missing))
            training = load_object(model_dir / "training_report.json")
            config = load_object(model_dir / "rl_agent_config.json")
            manifest = load_object(model_dir / "data_manifest.json")
            if not isinstance(training.get("best_epoch"), int):
                raise ValueError("selected checkpoint epoch is missing")
            selected = selected_epoch(training)
            baseline = training.get("baseline_validation") or {}
            loss = training.get("best_validation_loss", selected.get("validation_loss"))
            if training.get("best_epoch") == 0 and loss is None:
                loss = baseline.get("negative_log_likelihood")
            if loss is None or not math.isfinite(float(loss)):
                raise ValueError("selected checkpoint has no finite validation NLL")
            partition = manifest.get("partition") or training.get("partition") or {}
            splits = manifest.get("splits") or {}
            control = training.get("training_control") or {}
            config_match = CONFIG_ID_PATTERN.search(run_id)
            record = {
                "run_id": run_id,
                "status": "complete",
                "strategy": run.get("strategy") or training.get("strategy"),
                "config_id": run.get("config_id") or (config_match.group(1) if config_match else None),
                "partition": partition.get("strategy", run.get("partition")),
                "fold_count": partition.get("fold_count"),
                "calibration_fold": partition.get("calibration_fold"),
                "seed": run.get("seed", (config.get("training") or {}).get("seed")),
                "train_examples": (splits.get("train") or {}).get("records"),
                "calibration_examples": (splits.get("calibration") or {}).get("records"),
                "best_epoch": training.get("best_epoch"),
                "best_validation_loss": float(loss),
                "best_validation_accuracy": baseline.get("accuracy") if training.get("best_epoch") == 0 else selected.get("validation_accuracy"),
                "planned_epochs": control.get("planned_epochs"),
                "epochs_completed": control.get("epochs_completed"),
                "stopping_reason": control.get("stopping_reason"),
                "model_dir": str(model_dir.resolve()),
            }
            if not record["calibration_examples"]:
                raise ValueError("calibration example count is missing")
            candidates.append(record)
        except (OSError, ValueError, TypeError) as error:
            skipped.append({"run_id": run_id, "reason": str(error)})
    return candidates, skipped


def select_runs(
    rows: list[dict[str, Any]], selection_scope: str, minimum_runs: int
) -> list[dict[str, Any]]:
    if selection_scope == "matrix":
        return [row for row in rows if row.get("partition") == "group_k_fold"]
    if selection_scope == "all":
        return rows

    raise ValueError("Selection scope must evaluate all candidates: matrix or all")


def select_benchmark_winner(
    validation_rows: list[dict[str, Any]],
    selection_results: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_by_id = {str(row["run_id"]): row for row in validation_rows}
    eligible = []
    for result in selection_results:
        run_id = str(result.get("run_id"))
        validation = validation_by_id.get(run_id)
        if result.get("status") != "complete" or validation is None:
            continue
        f1 = float(result.get("f1"))
        accuracy = float(result.get("accuracy"))
        loss = float(validation.get("best_validation_loss"))
        if all(math.isfinite(value) for value in (f1, accuracy, loss)):
            eligible.append({**validation, "selection_f1": f1, "selection_accuracy": accuracy})
    if not eligible:
        raise ValueError("No completed selection-set benchmark is available")
    return min(eligible, key=lambda row: (
        -float(row["selection_f1"]),
        -float(row["selection_accuracy"]),
        float(row["best_validation_loss"]),
        str(row["run_id"]),
    ))


def read_gold(corpus: Path, labels: Path) -> dict[str, int]:
    articles = read_jsonl(corpus)
    annotations = read_jsonl(labels)
    article_ids = {row["article_id"] for row in articles}
    gold = {row["article_id"]: int(row["label"]) for row in annotations}
    if not gold or len(article_ids) != len(articles) or len(gold) != len(annotations):
        raise ValueError("Corpus and labels must be nonempty and contain unique article IDs")
    if article_ids != gold.keys() or any(label not in (1, 2) for label in gold.values()):
        raise ValueError("Corpus and labels must match exactly, with labels 1 or 2")
    return gold


def validate_ledger(path: Path, run_id: str, gold: dict[str, int], complete: bool = False) -> list[dict[str, Any]]:
    rows = read_jsonl(path) if path.is_file() else []
    seen = set()
    for row in rows:
        identifier = row["article_id"]
        if identifier in seen or identifier not in gold or row["gold_label"] != gold[identifier]:
            raise ValueError(f"Ledger has duplicate, unknown, or mismatched gold IDs: {path}")
        if (row.get("routing") or {}).get("model") != run_id:
            raise ValueError(f"Ledger belongs to a different checkpoint: {path}")
        if row["decision"] not in ("positive", "negative"):
            raise ValueError(f"Invalid benchmark decision: {path}")
        elapsed = float(row["elapsed_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Invalid benchmark timing: {path}")
        seen.add(identifier)
    if complete and seen != gold.keys():
        raise ValueError(f"Incomplete benchmark ledger: {path}")
    return rows


@contextmanager
def matrix_lock(matrix_root: Path):
    lock = matrix_root / ".lock"
    matrix_root.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Matrix is locked: {lock}. Stop the owner before removing a stale lock.") from error
    try:
        (lock / "owner").write_text(f"{socket.gethostname()}\n{os.getpid()}\n", encoding="utf-8")
        yield
    finally:
        (lock / "owner").unlink(missing_ok=True)
        lock.rmdir()


def check_gpu(device: str) -> None:
    if not device.startswith("cuda"):
        return
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("Cannot check GPU isolation: nvidia-smi is unavailable")
    result = subprocess.run(
        [executable, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("GPU compute processes are active. Stop training/inference before benchmarking.")


def benchmark_python(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else []
    if os.environ.get("VENV_DIR"):
        candidates.append(Path(os.environ["VENV_DIR"]) / "bin" / "python")
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python")
    candidates.append(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "laya-adverse-media" / "venv" / "bin" / "python"
    )
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise ValueError("No benchmark Python executable found; pass --python explicitly")


def validate_rolling_snapshot(
    previous: dict[str, Any], current: dict[str, Any], output: Path
) -> None:
    immutable = (
        "corpus_sha256",
        "labels_sha256",
        "device",
        "review_threshold",
        "runner_sha256",
    )
    changed = [key for key in immutable if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError("Benchmark inputs changed: " + ", ".join(changed))

    current_runs = {row["run_id"]: row for row in current["runs"]}
    for previous_run in previous.get("runs", []):
        run_id = previous_run["run_id"]
        has_output = (output / f"{run_id}.jsonl").exists() or (
            output / f"{run_id}.json"
        ).exists()
        if has_output and current_runs.get(run_id) != previous_run:
            raise ValueError(f"Benchmarked checkpoint changed or disappeared: {run_id}")


def benchmark_records(
    records: list[dict[str, Any]],
    output: Path,
    corpus: Path,
    labels: Path,
    gold: dict[str, int],
    snapshot: dict[str, Any],
    args: argparse.Namespace,
    inference_python: Path,
    environment: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    results = []
    failures = 0
    for record in records:
        run_id = record["run_id"]
        ledger = output / f"{run_id}.jsonl"
        report_path = output / f"{run_id}.json"
        log = output / "logs" / f"{run_id}.log"
        status = "complete"
        error_text = None
        metrics = {}
        ledger_hash = None
        try:
            validate_ledger(ledger, run_id, gold)
            if not report_path.is_file():
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(inference_python), "-u", str(ROOT / "scripts" / "benchmark_laya_checkpoint.py"),
                    "--run-id", run_id, "--model-dir", record["model_dir"],
                    "--corpus", str(corpus), "--labels", str(labels),
                    "--output", str(ledger), "--report", str(report_path),
                    "--device", args.device, "--review-threshold", str(args.review_threshold),
                    "--report-every", str(args.report_every), "--resume",
                ]
                print(json.dumps({"event": "benchmark_start", "run_id": run_id, "log": str(log)}), flush=True)
                with log.open("a", encoding="utf-8") as destination:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=destination, stderr=subprocess.STDOUT, check=True)
            report = load_object(report_path)
            rows = validate_ledger(ledger, run_id, gold, complete=True)
            ledger_hash = sha256(ledger)
            expected = {
                "run_id": run_id, "status": "complete", "ledger_sha256": ledger_hash,
                "model_weights_sha256": record["model_weights_sha256"],
                "model_config_sha256": record["model_config_sha256"],
                "corpus_sha256": snapshot["corpus_sha256"],
                "labels_sha256": snapshot["labels_sha256"],
            }
            if any(report.get(key) != value for key, value in expected.items()):
                raise ValueError("Benchmark report fingerprints do not match pinned inputs")
            metrics = benchmark_metrics(rows)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            status = "failed"
            error_text = str(error)
            failures += 1
        latency = metrics.get("latency_seconds") or {}
        mean = latency.get("mean")
        results.append({
            "run_id": run_id, "status": status,
            **{key: metrics.get(key) for key in RESULT_COLUMNS if key in metrics},
            "mean_latency_ms": 1000 * mean if mean is not None else None,
            "median_latency_ms": 1000 * latency["median"] if "median" in latency else None,
            "p95_latency_ms": 1000 * latency["p95"] if "p95" in latency else None,
            "pairs_per_second": 1 / mean if mean else None,
            "pairs_per_minute": 60 / mean if mean else None,
            "ledger_sha256": ledger_hash, "log": str(log), "error": error_text,
        })
        write_table(output / "benchmark-results.tsv", RESULT_COLUMNS, results)
        print(json.dumps({"event": "benchmark_finished", "run_id": run_id, "status": status, "error": error_text}), flush=True)
    return results, failures


def benchmark_final_winner(
    args: argparse.Namespace,
    winner: dict[str, Any],
    winner_record: dict[str, Any],
    inference_python: Path,
    environment: dict[str, str],
    *,
    provisional: bool = False,
) -> int:
    test_corpus = args.test_corpus.resolve()
    test_labels = args.test_labels.resolve()
    output = (
        args.test_output_dir
        or args.matrix_root / "benchmarks" / "final-test-winners"
    ).resolve()
    gold = read_gold(test_corpus, test_labels)
    candidate = {
        **winner_record,
        "selection_f1": winner["selection_f1"],
        "selection_accuracy": winner["selection_accuracy"],
    }
    snapshot = {
        "corpus_sha256": sha256(test_corpus),
        "labels_sha256": sha256(test_labels),
        "device": args.device,
        "review_threshold": args.review_threshold,
        "python": str(inference_python),
        "runner_sha256": sha256(ROOT / "scripts" / "benchmark_laya_checkpoint.py"),
        "batch_runner_sha256": sha256(Path(__file__)),
        "selection_rule": "selection macro F1, selection accuracy, validation NLL, run ID",
        "runs": [candidate],
    }
    snapshot_path = output / "inputs.json"
    if snapshot_path.is_file():
        previous = load_object(snapshot_path)
        previous_by_id = {row["run_id"]: row for row in previous.get("runs", [])}
        snapshot["runs"] = list(previous.get("runs", []))
        if candidate["run_id"] not in previous_by_id:
            snapshot["runs"].append(candidate)
        else:
            snapshot["runs"] = [
                candidate if row["run_id"] == candidate["run_id"] else row
                for row in snapshot["runs"]
            ]
        validate_rolling_snapshot(previous, snapshot, output)
    elif any(output.glob("*.jsonl")):
        raise ValueError("Unpinned final-test ledgers exist; use a new --test-output-dir")
    write_json(snapshot_path, snapshot)
    write_json(output / "current-winner.json", {
        "status": "provisional" if provisional else "final",
        "run_id": winner["run_id"],
        "selection_f1": winner["selection_f1"],
        "selection_accuracy": winner["selection_accuracy"],
        "best_validation_loss": winner["best_validation_loss"],
        "selection_rule": snapshot["selection_rule"],
    })
    results, failures = benchmark_records(
        snapshot["runs"], output, test_corpus, test_labels, gold, snapshot,
        args, inference_python, environment,
    )
    write_json(output / "completion.json", {
        "evaluated_winners": len(snapshot["runs"]),
        "completed_runs": len(results) - failures,
        "failed_runs": failures,
        "averaging": "macro",
        "label_convention": {"positive": "non-adverse", "negative": "adverse"},
    })
    return failures


def benchmark_strategy_winners(
    args: argparse.Namespace,
    validation_rows: list[dict[str, Any]],
    selection_results: list[dict[str, Any]],
    records: list[dict[str, Any]],
    inference_python: Path,
    environment: dict[str, str],
) -> int:
    output = args.matrix_root / "benchmarks" / "strategy-benchmark-winners"
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    completed_ids = {
        str(row["run_id"]) for row in selection_results if row.get("status") == "complete"
    }
    records_by_id = {str(row["run_id"]): row for row in records}
    for row in validation_rows:
        if str(row["run_id"]) in completed_ids:
            key = (str(row["strategy"]), str(row.get("config_id") or "legacy"))
            groups.setdefault(key, []).append(row)
    failures = 0
    winners = []
    for (strategy, config_id), rows in sorted(groups.items()):
        winner = select_benchmark_winner(rows, selection_results)
        comparison_args = argparse.Namespace(**{
            **vars(args),
            "test_output_dir": output / f"{strategy}__config-{config_id}",
        })
        failures += benchmark_final_winner(
            comparison_args, winner, records_by_id[str(winner["run_id"])],
            inference_python, environment, provisional=True,
        )
        winners.append({
            "strategy": strategy, "config_id": config_id, "run_id": winner["run_id"],
            "selection_f1": winner["selection_f1"],
            "selection_accuracy": winner["selection_accuracy"],
            "completed_strategy_runs": len(rows),
        })
    write_json(output / "current-strategy-winners.json", {
        "status": "provisional", "winners": winners, "failed_runs": failures,
        "selection_rule": "selection macro F1, selection accuracy, validation NLL, run ID",
    })
    return failures


def run_batch(args: argparse.Namespace) -> int:
    args.matrix_root = args.matrix_root.resolve()
    args.corpus = args.corpus.resolve()
    args.labels = args.labels.resolve()
    if args.benchmark_strategy_winners and args.selection_scope not in ("all", "matrix"):
        raise ValueError("--benchmark-strategy-winners requires --selection-scope matrix or all")
    default_output_names = {
        "matrix": "selection-checkpoints",
        "all": "all-checkpoints",
    }
    output = (
        args.output_dir
        or args.matrix_root / "benchmarks" / default_output_names[args.selection_scope]
    ).resolve()
    inference_python = benchmark_python(args.python)
    gold = read_gold(args.corpus, args.labels)
    with matrix_lock(args.matrix_root):
        discovered, skipped = discover_runs(args.matrix_root)
        candidates = select_runs(discovered, args.selection_scope, args.minimum_runs)
        print(json.dumps({
            "event": "inventory",
            "discovered_runs": len(discovered),
            "eligible_runs": len(candidates),
            "selection_scope": args.selection_scope,
            "skipped_runs": len(skipped),
            "output_dir": str(output),
        }), flush=True)
        if not discovered:
            write_json(output / "inventory.json", {"eligible_runs": 0, "skipped": skipped})
            raise RuntimeError("No finished exported checkpoints found; see inventory.json")
        if not candidates:
            write_json(output / "inventory.json", {
                "discovered_runs": len(discovered),
                "eligible_runs": 0,
                "selection_scope": args.selection_scope,
                "minimum_runs": args.minimum_runs,
                "skipped": skipped,
            })
            raise RuntimeError("No checkpoint satisfies the validation-only selection scope; see inventory.json")
        if args.dry_run:
            if (output / "inputs.json").exists():
                raise ValueError("Use a separate --output-dir for a dry run after benchmarking has started")
            write_json(output / "inventory.json", {
                "discovered_runs": len(discovered),
                "eligible_runs": len(candidates),
                "selection_scope": args.selection_scope,
                "minimum_runs": args.minimum_runs,
                "skipped": skipped,
            })
            write_table(output / "validation-runs.tsv", VALIDATION_COLUMNS, discovered)
            write_table(output / "validation-aggregate.tsv", AGGREGATE_COLUMNS, aggregate_rows(discovered))
            write_table(output / "selected-runs.tsv", VALIDATION_COLUMNS, candidates)
            for record in candidates:
                print(json.dumps({"event": "planned", "run_id": record["run_id"]}), flush=True)
            return 0
        if not args.allow_shared_gpu:
            check_gpu(args.device)
        snapshot = {
            "corpus_sha256": sha256(args.corpus),
            "labels_sha256": sha256(args.labels),
            "device": args.device,
            "review_threshold": args.review_threshold,
            "python": str(inference_python),
            "selection_scope": args.selection_scope,
            "minimum_runs": args.minimum_runs,
            "runner_sha256": sha256(ROOT / "scripts" / "benchmark_laya_checkpoint.py"),
            "batch_runner_sha256": sha256(Path(__file__)),
            "runs": [
                {**record, **{
                    name: sha256(Path(record["model_dir"]) / filename)
                    for name, filename in (
                        ("model_weights_sha256", "model.safetensors"),
                        ("model_config_sha256", "rl_agent_config.json"),
                        ("training_report_sha256", "training_report.json"),
                        ("data_manifest_sha256", "data_manifest.json"),
                    )
                }}
                for record in candidates
            ],
        }
        snapshot_path = output / "inputs.json"
        if snapshot_path.is_file():
            previous_snapshot = load_object(snapshot_path)
            if args.selection_scope in ("all", "matrix"):
                validate_rolling_snapshot(previous_snapshot, snapshot, output)
                write_json(snapshot_path, snapshot)
            elif previous_snapshot != snapshot:
                if any(output.glob("*.jsonl")):
                    raise ValueError("Inputs or candidate runs changed; use a new --output-dir instead of mixing benchmarks")
                write_json(snapshot_path, snapshot)
        else:
            if any(output.glob("*.jsonl")):
                raise ValueError("Unpinned ledgers exist; use a new --output-dir")
            write_json(snapshot_path, snapshot)
        write_json(output / "inventory.json", {"eligible_runs": len(candidates), "skipped": skipped})
        write_table(output / "validation-runs.tsv", VALIDATION_COLUMNS, discovered)
        write_table(output / "validation-aggregate.tsv", AGGREGATE_COLUMNS, aggregate_rows(discovered))
        write_table(output / "selected-runs.tsv", VALIDATION_COLUMNS, candidates)
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        results, failures = benchmark_records(
            snapshot["runs"], output, args.corpus, args.labels, gold, snapshot,
            args, inference_python, environment,
        )
        completed_runs = sum(result.get("status") == "complete" for result in results)
        write_json(output / "completion.json", {
            "eligible_runs": len(candidates), "completed_runs": completed_runs,
            "failed_runs": failures, "skipped_runs": len(skipped),
            "averaging": "macro", "label_convention": {"positive": "non-adverse", "negative": "adverse"},
        })

        grouped_validation = [
            row for row in discovered if row.get("partition") == "group_k_fold"
        ]
        grouped_run_ids = {str(row["run_id"]) for row in grouped_validation}
        grouped_results = [
            result for result in results if str(result.get("run_id")) in grouped_run_ids
        ]
        grouped_runs = len(grouped_validation)
        completed_grouped_runs = sum(
            result.get("status") == "complete" for result in grouped_results
        )
        current_winner = None
        if args.selection_scope in ("all", "matrix") and completed_grouped_runs:
            current_winner = select_benchmark_winner(grouped_validation, grouped_results)
            write_json(output / "current-selection-leader.json", {
                "status": (
                    "complete"
                    if grouped_runs >= args.expected_selection_runs
                    and completed_grouped_runs == grouped_runs
                    else "provisional"
                ),
                "run_id": current_winner["run_id"],
                "selection_f1": current_winner["selection_f1"],
                "selection_accuracy": current_winner["selection_accuracy"],
                "best_validation_loss": current_winner["best_validation_loss"],
                "eligible_grouped_runs": grouped_runs,
                "completed_grouped_runs": completed_grouped_runs,
                "expected_grouped_runs": args.expected_selection_runs,
                "selection_corpus_sha256": snapshot["corpus_sha256"],
                "selection_labels_sha256": snapshot["labels_sha256"],
                "selection_rule": "selection macro F1, selection accuracy, validation NLL, run ID",
            })
        if args.benchmark_strategy_winners:
            failures += benchmark_strategy_winners(
                args, grouped_validation, grouped_results, snapshot["runs"],
                inference_python, environment,
            )
        if args.finalize_winner:
            if args.selection_scope not in ("all", "matrix"):
                raise ValueError("--finalize-winner requires --selection-scope matrix or all")
            if grouped_runs < args.expected_selection_runs:
                print(json.dumps({
                    "event": "finalization_deferred",
                    "eligible_grouped_runs": grouped_runs,
                    "expected_grouped_runs": args.expected_selection_runs,
                    "preliminary_leader": str(output / "current-selection-leader.json"),
                    "message": "Available benchmarks completed; no final winner frozen.",
                }), flush=True)
                return 1 if failures else 0
            if failures:
                raise ValueError("Cannot finalize while selection-set benchmarks have failed")
            if current_winner is None or completed_grouped_runs != grouped_runs:
                raise ValueError("Cannot finalize without complete selection-set benchmarks")
            winner = current_winner
            winner_record = next(
                row for row in snapshot["runs"] if row["run_id"] == winner["run_id"]
            )
            write_json(output / "selection-winner.json", {
                "run_id": winner["run_id"],
                "selection_f1": winner["selection_f1"],
                "selection_accuracy": winner["selection_accuracy"],
                "best_validation_loss": winner["best_validation_loss"],
                "selection_corpus_sha256": snapshot["corpus_sha256"],
                "selection_labels_sha256": snapshot["labels_sha256"],
                "selection_rule": "selection macro F1, selection accuracy, validation NLL, run ID",
            })
            failures += benchmark_final_winner(
                args, winner, winner_record, inference_python, environment
            )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark all candidates on the selection set and optionally evaluate the frozen winner on the test set")
    parser.add_argument("--matrix-root", type=Path, default=ROOT / "artifacts" / "public-experiment-matrix")
    parser.add_argument("--corpus", type=Path, default=ROOT / "datasets" / "adverse-media-public-selection-1000" / "corpus.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "datasets" / "adverse-media-public-selection-1000" / "annotations" / "human.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-corpus", type=Path, default=ROOT / "datasets" / "adverse-media-public-holdout-1000" / "corpus.jsonl")
    parser.add_argument("--test-labels", type=Path, default=ROOT / "datasets" / "adverse-media-public-holdout-1000" / "annotations" / "human.jsonl")
    parser.add_argument("--test-output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--review-threshold", type=float, default=0.8)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--python", type=Path, help="Python executable containing the Laya runtime")
    parser.add_argument(
        "--selection-scope", choices=("matrix", "all"), default="matrix",
        help="Evaluate all grouped matrix candidates on the selection set, or every exported run",
    )
    parser.add_argument("--minimum-runs", type=int, default=1)
    parser.add_argument("--expected-selection-runs", type=int, default=210)
    parser.add_argument("--finalize-winner", action="store_true", help="Freeze the selection-set winner and evaluate only it on the benchmark set")
    parser.add_argument(
        "--benchmark-strategy-winners", action="store_true",
        help="Evaluate each strategy's selection-F1 leader on the benchmark set, as in the paper's appendix table",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write validation-only inventory and tables without inference")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.report_every < 1 or args.minimum_runs < 1 or args.expected_selection_runs < 1 or not 0 <= args.review_threshold <= 1:
        raise SystemExit("--report-every, --minimum-runs, and --expected-selection-runs must be positive, and --review-threshold must be between 0 and 1")

    def interrupt(signum, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(run_batch(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()