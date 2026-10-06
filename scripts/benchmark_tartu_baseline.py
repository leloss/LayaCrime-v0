from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

SOURCE_REPOSITORY = "https://github.com/kristjanr/ut-ml-adverse-media"
SOURCE_COMMIT = "12fa6ada0a6f46ce098a654e39aea0ebbf1d55f5"
MODEL_ID = "tartu-tfidf-multinomial-nb-source-transfer"
CLEANUP_PATTERN = re.compile(r"(http\S+)|(#(\w+))|(@(\w+))|[^\w\s]|(\w*\d\w*)")
WHITESPACE_PATTERN = re.compile(r"(\s+)|(\n+)")


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


def read_zip_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path}, found {members}")
        with archive.open(members[0]) as raw_source:
            source = (line.decode("utf-8-sig") for line in raw_source)
            return list(csv.DictReader(source))


def source_text(row: dict[str, str]) -> str:
    return f"{row.get('title', '')} {row.get('article', '')}"


def load_released_training(repository: Path) -> tuple[list[str], np.ndarray, dict[str, int]]:
    adverse_rows = read_zip_csv(repository / "adverse_media_training.csv.zip")
    non_adverse_rows = read_zip_csv(repository / "non_adverse_media_training.csv.zip")
    examples: list[tuple[str, int]] = []

    for row in adverse_rows + non_adverse_rows:
        label = row.get("label", "").strip()
        if label == "am":
            examples.append((source_text(row), 1))
        elif label in {"nam", "random"}:
            examples.append((source_text(row), 0))

    labels = np.array([label for _, label in examples], dtype=np.int64)
    counts = {
        "released_rows": len(adverse_rows) + len(non_adverse_rows),
        "included_rows": len(examples),
        "adverse_rows": int(labels.sum()),
        "non_adverse_rows": int(len(labels) - labels.sum()),
    }
    return [text for text, _ in examples], labels, counts


def clean_text(text: str) -> str:
    text = CLEANUP_PATTERN.sub("", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip().lower()


def lemmatize_texts(texts: list[str], nlp: Any, batch_size: int) -> list[str]:
    cleaned = [clean_text(text) for text in texts]
    return [
        " ".join(token.lemma_ for token in document if not token.is_stop)
        for document in nlp.pipe(cleaned, batch_size=batch_size)
    ]


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=40_000,
        min_df=5,
        max_df=0.5,
        analyzer="word",
        stop_words="english",
        ngram_range=(1, 3),
    )


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


def prediction_rows(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    predictions = scores >= 0.5
    return [
        {
            "article_id": row["article_id"],
            "entity_name": row["entity_name"],
            "gold_label": 2 if row["gold_label"] else 1,
            "label": 2 if prediction else 1,
            "label_name": "negative" if prediction else "positive",
            "adverse_score": float(score),
            "model_id": MODEL_ID,
        }
        for row, score, prediction in zip(rows, scores, predictions, strict=True)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn
    import spacy

    source_texts, source_labels, source_counts = load_released_training(args.repository)
    test_rows = join_examples(args.test_corpus, args.test_labels)
    if args.limit_test is not None:
        test_rows = test_rows[: args.limit_test]

    nlp = spacy.load(args.spacy_model)
    preprocessing_started = time.perf_counter()
    source_documents = lemmatize_texts(source_texts, nlp, args.preprocessing_batch_size)
    test_documents = lemmatize_texts(
        [str(row["article"]) for row in test_rows], nlp, args.preprocessing_batch_size
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    vectorizer = build_vectorizer()
    train_matrix = vectorizer.fit_transform(source_documents)
    classifier = MultinomialNB(alpha=0.3)
    classifier.fit(train_matrix, source_labels)

    inference_started = time.perf_counter()
    scores = classifier.predict_proba(vectorizer.transform(test_documents))[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    predictions = scores >= 0.5

    run_directory = args.output_root / MODEL_ID
    ledger = prediction_rows(test_rows, scores)
    write_jsonl(run_directory / "predictions.jsonl", ledger)
    report = {
        "model_id": MODEL_ID,
        "comparison_status": "released article-level adverse-media method transferred without CSA retraining",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_method": "spaCy lemmatization, word TF-IDF, and multinomial Naive Bayes",
        "task_relation": {
            "source_unit": "article",
            "evaluation_unit": "entity/article pair",
            "uses_target_entity": False,
            "uses_csa_training_labels": False,
            "interpretation": "article-level adverse-media transfer baseline, not an entity-conditioned CSA system",
        },
        "configuration": {
            "cleanup_regex": CLEANUP_PATTERN.pattern,
            "lowercase": True,
            "lemmatizer": args.spacy_model,
            "remove_spacy_stop_words": True,
            "tfidf_max_features": 40_000,
            "tfidf_min_df": 5,
            "tfidf_max_df": 0.5,
            "tfidf_stop_words": "english",
            "tfidf_ngram_range": [1, 3],
            "multinomial_nb_alpha": 0.3,
            "decision_threshold": 0.5,
        },
        "released_training_data": source_counts,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "metrics": metrics(np.array([row["gold_label"] for row in test_rows]), predictions),
        "timing": {
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "examples_per_second": len(test_rows) / inference_seconds if inference_seconds else None,
        },
        "fingerprints": {
            "source_adverse_archive_sha256": sha256(args.repository / "adverse_media_training.csv.zip"),
            "source_non_adverse_archive_sha256": sha256(
                args.repository / "non_adverse_media_training.csv.zip"
            ),
            "test_corpus_sha256": sha256(args.test_corpus),
            "test_labels_sha256": sha256(args.test_labels),
            "predictions_sha256": sha256(run_directory / "predictions.jsonl"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "spacy": spacy.__version__,
            "spacy_model": nlp.meta.get("version"),
        },
    }
    write_json(run_directory / "report.json", report)
    print(json.dumps({"event": "baseline_complete", **report["metrics"]}), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the released Tartu article-level adverse-media baseline")
    parser.add_argument("--repository", type=Path, default=Path("third_party/ut-ml-adverse-media"))
    parser.add_argument("--test-corpus", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--preprocessing-batch-size", type=int, default=32)
    parser.add_argument("--limit-test", type=int)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
