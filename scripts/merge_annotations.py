from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable


LABEL_NAMES = {1: "good_guy", 2: "bad_guy"}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    return rows


def normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_predictions(articles: Iterable[dict], predictions: Iterable[dict]) -> list[str]:
    article_rows = list(articles)
    prediction_rows = list(predictions)
    articles_by_id = {row["article_id"]: row for row in article_rows}
    article_ids = set(articles_by_id)
    prediction_ids = [row.get("article_id") for row in prediction_rows]
    counts = Counter(prediction_ids)
    errors: list[str] = []

    duplicate_ids = sorted(str(article_id) for article_id, count in counts.items() if count > 1)
    missing_ids = sorted(article_ids - set(prediction_ids))
    unknown_ids = sorted(str(article_id) for article_id in set(prediction_ids) - article_ids)
    if duplicate_ids:
        errors.append(f"duplicate predictions: {', '.join(duplicate_ids)}")
    if missing_ids:
        errors.append(f"missing predictions: {', '.join(missing_ids)}")
    if unknown_ids:
        errors.append(f"unknown predictions: {', '.join(unknown_ids)}")

    for row in prediction_rows:
        article_id = row.get("article_id")
        if article_id not in articles_by_id:
            continue
        label = row.get("label")
        if type(label) is not int or label not in LABEL_NAMES:
            errors.append(f"{article_id}: label must be integer 1 or 2")
        elif row.get("label_name") != LABEL_NAMES[label]:
            errors.append(f"{article_id}: label_name does not match label")

        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{article_id}: confidence must be between 0 and 1")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            errors.append(f"{article_id}: rationale must be nonempty")
        if not isinstance(row.get("annotator"), str) or not row["annotator"].strip():
            errors.append(f"{article_id}: annotator must be nonempty")
        if row.get("status") not in {"ai_annotated", "human_reviewed"}:
            errors.append(f"{article_id}: invalid annotation status")

        evidence = row.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(quote, str) or not quote.strip() for quote in evidence):
            errors.append(f"{article_id}: evidence must be a list of nonempty strings")
            continue
        article_text = normalized_text(articles_by_id[article_id]["article"])
        for quote in evidence:
            if normalized_text(quote) not in article_text:
                errors.append(f"{article_id}: evidence is not present in article: {quote!r}")

    return errors


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def merge(args: argparse.Namespace) -> dict:
    articles = read_jsonl(args.batch)
    predictions = [row for path in args.predictions for row in read_jsonl(path)]
    errors = validate_predictions(articles, predictions)
    if errors:
        raise ValueError("annotation validation failed:\n- " + "\n- ".join(errors))

    predictions_by_id = {row["article_id"]: row for row in predictions}
    ordered_predictions = [predictions_by_id[row["article_id"]] for row in articles]
    ledger = read_jsonl(args.ledger)
    ledger_ids = {row["article_id"] for row in ledger}
    absent_from_ledger = sorted(predictions_by_id.keys() - ledger_ids)
    if absent_from_ledger:
        raise ValueError(f"predictions absent from ledger: {', '.join(absent_from_ledger)}")

    updated_ledger = [predictions_by_id.get(row["article_id"], row) for row in ledger]
    write_jsonl_atomic(args.output, ordered_predictions)
    write_jsonl_atomic(args.ledger, updated_ledger)

    labels = Counter(row["label_name"] for row in ordered_predictions)
    review_queue = [row["article_id"] for row in ordered_predictions if row["confidence"] < args.review_threshold]
    return {
        "predictions": len(ordered_predictions),
        "labels": dict(sorted(labels.items())),
        "review_threshold": args.review_threshold,
        "review_queue": review_queue,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate blind annotations and merge them into the annotation ledger")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-threshold", type=float, default=0.6)
    return parser


def main() -> None:
    try:
        report = merge(build_parser().parse_args())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()