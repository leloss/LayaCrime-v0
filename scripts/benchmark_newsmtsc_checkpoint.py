from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

SOURCE_REPOSITORY = "https://github.com/fhamborg/NewsMTSC"
SOURCE_COMMIT = "b9d9b79704ed1b35cecaf1d7c2343dc1bd734fb7"
MODEL_ID = "newsmtsc-grutsc-v1-sentiment-transfer"
SENTENCE_BOUNDARIES = ".!?\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def join_examples(corpus_path: Path, labels_path: Path) -> list[dict[str, Any]]:
    corpus = read_jsonl(corpus_path)
    labels = {row["article_id"]: int(row["label"]) for row in read_jsonl(labels_path)}
    if len(corpus) != len(labels) or any(row["article_id"] not in labels for row in corpus):
        raise ValueError("corpus and labels must contain the same unique article IDs")
    return [{**row, "gold_label": 1 if labels[row["article_id"]] == 2 else 0} for row in corpus]


def target_sentence(
    article: str, entity: str, *, maximum_chars_per_side: int = 300
) -> tuple[str, str, str] | None:
    compact = re.sub(r"[\t\r ]+", " ", article).strip()
    match = re.search(re.escape(entity.strip()), compact, re.IGNORECASE)
    if match is None:
        return None

    sentence_start = max(compact.rfind(boundary, 0, match.start()) for boundary in SENTENCE_BOUNDARIES)
    sentence_start = 0 if sentence_start < 0 else sentence_start + 1
    following = [compact.find(boundary, match.end()) for boundary in SENTENCE_BOUNDARIES]
    sentence_ends = [position for position in following if position >= 0]
    sentence_end = min(sentence_ends) + 1 if sentence_ends else len(compact)

    left = compact[sentence_start : match.start()]
    target = compact[match.start() : match.end()]
    right = compact[match.end() : sentence_end]
    return left[-maximum_chars_per_side:], target, right[:maximum_chars_per_side]


def metrics(gold: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        gold, predictions, labels=[0, 1]
    ).ravel()
    return {
        "examples": int(len(gold)),
        "accuracy": float(accuracy_score(gold, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(gold, predictions)),
        "precision_negative": float(precision_score(gold, predictions, zero_division=0)),
        "recall_negative": float(recall_score(gold, predictions, zero_division=0)),
        "f1_negative": float(f1_score(gold, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(gold, predictions)),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
    }


def classify_results(results: list[tuple[dict[str, Any], ...]]) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    predictions = []
    for result in results:
        probabilities = {entry["class_label"]: float(entry["class_prob"]) for entry in result}
        scores.append(probabilities["negative"])
        predictions.append(int(max(result, key=lambda entry: entry["class_prob"])["class_label"] == "negative"))
    return np.array(scores), np.array(predictions, dtype=np.int64)


def infer_in_batches(
    classifier: Any,
    contexts: list[tuple[str, str, str]],
    *,
    batch_size: int,
) -> tuple[list[tuple[dict[str, Any], ...] | None], list[dict[str, Any]]]:
    outputs: list[tuple[dict[str, Any], ...] | None] = [None] * len(contexts)
    failures = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        try:
            results = classifier.infer(targets=batch, batch_size=batch_size, disable_tqdm=True)
            outputs[start : start + len(results)] = results
        except Exception as batch_error:
            for offset, context in enumerate(batch):
                try:
                    outputs[start + offset] = classifier.infer(
                        text_left=context[0],
                        target_mention=context[1],
                        text_right=context[2],
                        disable_tqdm=True,
                    )
                except Exception as item_error:
                    failures.append(
                        {
                            "context_index": start + offset,
                            "error": type(item_error).__name__,
                            "message": str(item_error),
                            "batch_error": type(batch_error).__name__,
                        }
                    )
    return outputs, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    import truststore
    from NewsSentiment.download import Download
    from NewsSentiment.infer import TargetSentimentClassifier, parse_arguments
    from NewsSentiment.models.singletarget.grutscsingle import GRUTSCSingle

    truststore.inject_into_ssl()
    rows = join_examples(args.test_corpus, args.test_labels)
    if args.limit_test is not None:
        rows = rows[: args.limit_test]

    contexts = []
    context_row_indices = []
    missing_mentions = []
    for row_index, row in enumerate(rows):
        context = target_sentence(
            str(row["article"]),
            str(row["entity_name"]),
            maximum_chars_per_side=args.maximum_chars_per_side,
        )
        if context is None:
            missing_mentions.append(row_index)
        else:
            context_row_indices.append(row_index)
            contexts.append(context)

    options = parse_arguments(override_args=True)
    options.pretrained_model_name = args.pretrained_model_name
    classifier = TargetSentimentClassifier(opts_from_infer=options)
    inference_started = time.perf_counter()
    context_outputs, failures = infer_in_batches(classifier, contexts, batch_size=args.batch_size)
    inference_seconds = time.perf_counter() - inference_started

    scores = np.zeros(len(rows), dtype=np.float64)
    predictions = np.zeros(len(rows), dtype=np.int64)
    successful_contexts = 0
    for context_index, output in enumerate(context_outputs):
        if output is None:
            continue
        output_scores, output_predictions = classify_results([output])
        row_index = context_row_indices[context_index]
        scores[row_index] = output_scores[0]
        predictions[row_index] = output_predictions[0]
        successful_contexts += 1

    ledger = [
        {
            "article_id": row["article_id"],
            "entity_name": row["entity_name"],
            "gold_label": 2 if row["gold_label"] else 1,
            "label": 2 if predictions[index] else 1,
            "label_name": "negative" if predictions[index] else "positive",
            "adverse_score": float(scores[index]),
            "status": "success" if index in set(context_row_indices) and scores[index] > 0 else "fallback_no_prediction",
            "model_id": MODEL_ID,
        }
        for index, row in enumerate(rows)
    ]
    run_directory = args.output_root / MODEL_ID
    write_jsonl(run_directory / "predictions.jsonl", ledger)

    encoder_directory = Path(classifier.opt.pretrained_model_name)
    state_dict_path = Path(Download.model_path(GRUTSCSingle))
    gold = np.array([row["gold_label"] for row in rows])
    report = {
        "model_id": MODEL_ID,
        "comparison_status": "official released checkpoint with sentiment-to-CSA diagnostic mapping",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint": "GRU-TSC v1.0.0",
        "task_relation": {
            "source_task": "three-class target sentiment in news sentences",
            "evaluation_task": "binary entity-conditioned criminal association in articles",
            "uses_csa_training_labels": False,
            "mapping": "predict adverse only when the released checkpoint's argmax class is negative",
            "fallback": "predict non-adverse when no literal target mention is available or inference fails",
            "interpretation": "sentiment-transfer diagnostic, not a CSA-trained baseline",
        },
        "context_adapter": {
            "selection": "sentence containing the first case-insensitive literal target mention",
            "maximum_chars_per_side": args.maximum_chars_per_side,
            "matched_examples": len(contexts),
            "successful_predictions": successful_contexts,
            "missing_literal_mentions": len(missing_mentions),
            "inference_failures": len(failures),
            "failure_details": failures,
        },
        "metrics": metrics(gold, predictions),
        "timing": {
            "inference_seconds": inference_seconds,
            "matched_examples_per_second": len(contexts) / inference_seconds if inference_seconds else None,
        },
        "fingerprints": {
            "test_corpus_sha256": sha256(args.test_corpus),
            "test_labels_sha256": sha256(args.test_labels),
            "predictions_sha256": sha256(run_directory / "predictions.jsonl"),
            "source_checkpoint_sha256": sha256(state_dict_path),
            "encoder_files": {
                path.name: sha256(path)
                for path in sorted(encoder_directory.iterdir())
                if path.is_file()
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    write_json(run_directory / "report.json", report)
    print(json.dumps({"event": "baseline_complete", **report["metrics"]}), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official NewsMTSC GRU-TSC checkpoint as a CSA transfer diagnostic")
    parser.add_argument("--test-corpus", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretrained-model-name", default="roberta-base")
    parser.add_argument("--maximum-chars-per-side", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-test", type=int)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
