from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from laya_adverse_media.academic_models import (
    ACADEMIC_KHANDPUR_ID,
    ACADEMIC_NEWSMTSC_ID,
    ACADEMIC_TARTU_ID,
    AcademicModelRuntime,
    _read_jsonl,
)

MODEL_IDS = {
    "khandpur": ACADEMIC_KHANDPUR_ID,
    "newsmtsc": ACADEMIC_NEWSMTSC_ID,
    "tartu": ACADEMIC_TARTU_ID,
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.corpus)
    labels = {
        row["article_id"]: int(row["label"])
        for row in _read_jsonl(args.labels)
    }
    model_id = MODEL_IDS[args.model]
    runtime = AcademicModelRuntime(args.project_root)
    predictions = []
    elapsed = []
    try:
        runtime.activate(model_id)
        for row in rows:
            started = time.perf_counter()
            result = runtime.predict(
                str(row["article"]), str(row["entity_name"]), model_id
            )
            elapsed.append(time.perf_counter() - started)
            choice = result["answers"]["criminal_association"]["choice"]
            predictions.append(2 if choice == "A" else 1)
    finally:
        runtime.close()

    truth = [labels[row["article_id"]] for row in rows]
    total_seconds = sum(elapsed)
    report = {
        "model_id": model_id,
        "examples": len(rows),
        "inference_seconds": total_seconds,
        "average_seconds": total_seconds / len(rows),
        "items_per_second": len(rows) / total_seconds,
        "true_positive": sum(prediction == 2 and gold == 2 for prediction, gold in zip(predictions, truth, strict=True)),
        "false_positive": sum(prediction == 2 and gold == 1 for prediction, gold in zip(predictions, truth, strict=True)),
        "true_negative": sum(prediction == 1 and gold == 1 for prediction, gold in zip(predictions, truth, strict=True)),
        "false_negative": sum(prediction == 1 and gold == 2 for prediction, gold in zip(predictions, truth, strict=True)),
    }
    if args.canonical_predictions is not None:
        canonical = [
            int(row["label"])
            for row in _read_jsonl(args.canonical_predictions)
        ]
        if len(canonical) != len(predictions):
            raise ValueError("canonical ledger has a different number of predictions")
        report["canonical_prediction_mismatches"] = sum(
            actual != expected
            for actual, expected in zip(predictions, canonical, strict=True)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure an academic model through the interactive UI runtime"
    )
    parser.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--canonical-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()