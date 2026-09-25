from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_LABELS = {"false positive": 1, "hit": 2}
LABEL_NAMES = {1: "good_guy", 2: "bad_guy"}


def article_id(entity: str, article: str) -> str:
    normalized = "\n".join((entity.casefold(), " ".join(article.split()).casefold()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def parse_source(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    headers: dict[str, str] = {}
    content_lines: list[str] = []
    in_content = False
    for line in text.splitlines():
        if not in_content and line.startswith("###content:"):
            content_lines.append(line.partition(":")[2].lstrip())
            in_content = True
        elif in_content:
            content_lines.append(line)
        elif line.startswith("###") and ":" in line:
            key, _, value = line[3:].partition(":")
            headers[key.strip().casefold()] = value.strip()

    entity = headers.get("entityname", "").strip()
    disposition = (
        headers.get("disspositionreason") or headers.get("dispositionreason") or ""
    ).strip().casefold()
    article = "\n".join(content_lines).strip()
    if not entity or not article:
        raise ValueError("missing entityName or content")
    if disposition not in SOURCE_LABELS:
        raise ValueError(f"unsupported disposition {disposition!r}")
    return article_id(entity, article), SOURCE_LABELS[disposition]


def load_source_labels(paths: Iterable[Path]) -> tuple[dict[str, int], dict]:
    grouped: dict[str, set[int]] = defaultdict(set)
    parse_errors = 0
    files_seen = 0
    for path in paths:
        if not path.is_file():
            continue
        files_seen += 1
        try:
            identifier, label = parse_source(path)
        except ValueError:
            parse_errors += 1
            continue
        grouped[identifier].add(label)

    conflicts = {identifier for identifier, labels in grouped.items() if len(labels) > 1}
    labels = {
        identifier: next(iter(values))
        for identifier, values in grouped.items()
        if identifier not in conflicts
    }
    return labels, {
        "files_seen": files_seen,
        "unique_labeled_articles": len(labels),
        "conflicting_articles_excluded": len(conflicts),
        "parse_errors": parse_errors,
    }


def read_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    counts = Counter(row.get("article_id") for row in rows)
    duplicates = sorted(str(identifier) for identifier, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate prediction IDs: {', '.join(duplicates)}")
    return rows


def metrics(predictions: Iterable[dict], source_labels: dict[str, int]) -> dict:
    rows = list(predictions)
    compared = [row for row in rows if row.get("article_id") in source_labels]
    matrix = [[0, 0], [0, 0]]
    for row in compared:
        predicted = row.get("label")
        if type(predicted) is not int or predicted not in LABEL_NAMES:
            raise ValueError(f"{row.get('article_id')}: prediction label must be integer 1 or 2")
        actual = source_labels[row["article_id"]]
        matrix[actual - 1][predicted - 1] += 1

    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    total = sum(sum(row) for row in matrix)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    original_counts = Counter(LABEL_NAMES[source_labels[row["article_id"]]] for row in compared)
    predicted_counts = Counter(LABEL_NAMES[row["label"]] for row in compared)
    return {
        "prediction_records": len(rows),
        "compared_records": total,
        "unmatched_predictions": len(rows) - total,
        "orientation": "rows=original, columns=prediction",
        "labels": ["good_guy", "bad_guy"],
        "confusion_matrix": matrix,
        "counts": {
            "original": dict(sorted(original_counts.items())),
            "prediction": dict(sorted(predicted_counts.items())),
        },
        "bad_guy_as_positive": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
    }


def compare(args: argparse.Namespace) -> dict:
    predictions = read_predictions(args.predictions)
    source_labels, source_audit = load_source_labels(sorted(args.data_dir.iterdir()))
    report = {
        "source_audit": source_audit,
        "comparison": metrics(predictions, source_labels),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare frozen blind predictions with original source dispositions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(compare(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()