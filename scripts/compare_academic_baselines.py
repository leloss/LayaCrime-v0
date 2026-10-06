from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_label(row: dict[str, Any]) -> int:
    if row.get("label") in (1, 2):
        return int(row["label"])
    if row.get("decision") in ("positive", "negative"):
        return 2 if row["decision"] == "negative" else 1
    raise ValueError(f"row {row.get('article_id')} has no supported prediction label")


def ledger(path: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for row in read_jsonl(path):
        identifier = str(row["article_id"])
        if identifier in result:
            raise ValueError(f"duplicate article ID in {path}: {identifier}")
        gold = int(row["gold_label"])
        if gold not in (1, 2):
            raise ValueError(f"row {identifier} has invalid gold label {gold}")
        result[identifier] = (gold, normalized_label(row))
    return result


def binary_metrics(gold: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    adverse_gold = gold == 2
    adverse_predictions = predictions == 2
    true_positive = int(np.sum(adverse_gold & adverse_predictions))
    false_positive = int(np.sum(~adverse_gold & adverse_predictions))
    true_negative = int(np.sum(~adverse_gold & ~adverse_predictions))
    false_negative = int(np.sum(adverse_gold & ~adverse_predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "examples": int(len(gold)),
        "accuracy": (true_positive + true_negative) / len(gold),
        "precision_negative": precision,
        "recall_negative": recall,
        "f1_negative": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def exact_mcnemar(reference_only: int, candidate_only: int) -> float:
    disagreements = reference_only + candidate_only
    if disagreements == 0:
        return 1.0
    tail = min(reference_only, candidate_only)
    probability = math.ldexp(sum(math.comb(disagreements, value) for value in range(tail + 1)), -disagreements)
    return min(1.0, 2.0 * probability)


def f1_negative(gold: np.ndarray, predictions: np.ndarray) -> float:
    adverse_gold = gold == 2
    adverse_predictions = predictions == 2
    true_positive = np.sum(adverse_gold & adverse_predictions)
    false_positive = np.sum(~adverse_gold & adverse_predictions)
    false_negative = np.sum(adverse_gold & ~adverse_predictions)
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def bootstrap_differences(
    gold: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = np.random.default_rng(seed)
    accuracy_differences = np.empty(samples)
    f1_differences = np.empty(samples)
    for sample in range(samples):
        indices = generator.integers(0, len(gold), size=len(gold))
        sampled_gold = gold[indices]
        sampled_reference = reference[indices]
        sampled_candidate = candidate[indices]
        accuracy_differences[sample] = np.mean(sampled_candidate == sampled_gold) - np.mean(
            sampled_reference == sampled_gold
        )
        f1_differences[sample] = f1_negative(sampled_gold, sampled_candidate) - f1_negative(
            sampled_gold, sampled_reference
        )

    def summarize(values: np.ndarray, observed: float) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {"difference": observed, "ci95_lower": float(lower), "ci95_upper": float(upper)}

    return {
        "accuracy": summarize(
            accuracy_differences,
            float(np.mean(candidate == gold) - np.mean(reference == gold)),
        ),
        "f1_negative": summarize(
            f1_differences,
            f1_negative(gold, candidate) - f1_negative(gold, reference),
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def compare(
    reference_path: Path,
    candidates: dict[str, Path],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    reference_rows = ledger(reference_path)
    identifiers = sorted(reference_rows)
    gold = np.array([reference_rows[identifier][0] for identifier in identifiers])
    reference = np.array([reference_rows[identifier][1] for identifier in identifiers])
    comparisons = {}
    p_values = {}
    for index, (name, path) in enumerate(candidates.items()):
        candidate_rows = ledger(path)
        if set(candidate_rows) != set(identifiers):
            raise ValueError(f"{name} article IDs do not match the reference ledger")
        candidate_gold = np.array([candidate_rows[identifier][0] for identifier in identifiers])
        if not np.array_equal(gold, candidate_gold):
            raise ValueError(f"{name} gold labels do not match the reference ledger")
        candidate = np.array([candidate_rows[identifier][1] for identifier in identifiers])
        reference_correct = reference == gold
        candidate_correct = candidate == gold
        reference_only = int(np.sum(reference_correct & ~candidate_correct))
        candidate_only = int(np.sum(~reference_correct & candidate_correct))
        p_value = exact_mcnemar(reference_only, candidate_only)
        p_values[name] = p_value
        comparisons[name] = {
            "ledger": str(path),
            "ledger_sha256": sha256(path),
            "metrics": binary_metrics(gold, candidate),
            "paired_correctness": {
                "both_correct": int(np.sum(reference_correct & candidate_correct)),
                "both_incorrect": int(np.sum(~reference_correct & ~candidate_correct)),
                "reference_only_correct": reference_only,
                "candidate_only_correct": candidate_only,
                "mcnemar_exact_two_sided_p": p_value,
            },
            "bootstrap": bootstrap_differences(
                gold,
                reference,
                candidate,
                samples=bootstrap_samples,
                seed=seed + index,
            ),
        }

    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        comparisons[name]["paired_correctness"]["holm_adjusted_p"] = value
    return {
        "reference": {
            "ledger": str(reference_path),
            "ledger_sha256": sha256(reference_path),
            "metrics": binary_metrics(gold, reference),
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
    }


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must have the form NAME=PATH")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare academic baseline ledgers with LayaCrime")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare(
        args.reference,
        dict(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()