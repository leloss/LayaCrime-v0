from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0: "positive", 1: "negative"}
TOKEN_PATTERN = re.compile(r"\[/?TARGET\]|\[SEP\]|\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


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
    return [
        {
            **row,
            "gold_label": 1 if labels[row["article_id"]] == 2 else 0,
        }
        for row in corpus
    ]


def target_context(
    article: str,
    entity: str,
    *,
    radius: int = 900,
    maximum_mentions: int = 3,
) -> tuple[str, bool, int]:
    compact = re.sub(r"\s+", " ", article).strip()
    pattern = re.compile(re.escape(entity.strip()), re.IGNORECASE)
    matches = list(pattern.finditer(compact))
    if not matches:
        prefix = compact[: radius * 2]
        return f"[TARGET] {entity.strip()} [/TARGET] [SEP] {prefix}", False, 0

    windows = []
    for match in matches[:maximum_mentions]:
        start = max(0, match.start() - radius)
        end = min(len(compact), match.end() + radius)
        fragment = compact[start : match.start()]
        fragment += f" [TARGET] {match.group(0)} [/TARGET] "
        fragment += compact[match.end() : end]
        windows.append(fragment.strip())
    return " [SEP] ".join(windows) + f" [SEP] {entity.strip()}", True, len(matches)


def tokenize_target_context(text: str, maximum_tokens: int) -> tuple[list[str], list[int]]:
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens: list[str] = []
    target_mask: list[int] = []
    inside_target = False
    for token in raw_tokens:
        if token == "[TARGET]":
            inside_target = True
            continue
        if token == "[/TARGET]":
            inside_target = False
            continue
        tokens.append(token.casefold())
        target_mask.append(int(inside_target))

    if len(tokens) <= maximum_tokens:
        return tokens, target_mask
    target_positions = [index for index, value in enumerate(target_mask) if value]
    center = target_positions[0] if target_positions else 0
    start = max(0, min(center - maximum_tokens // 2, len(tokens) - maximum_tokens))
    return tokens[start : start + maximum_tokens], target_mask[start : start + maximum_tokens]


def choose_threshold(gold: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.10, 0.90, 161)
    ranked = [
        (f1_score(gold, scores >= threshold, zero_division=0), -abs(threshold - 0.5), threshold)
        for threshold in candidates
    ]
    best_f1, _, best_threshold = max(ranked)
    return float(best_threshold), float(best_f1)


def metrics(gold: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(gold, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
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


def split_training(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(len(rows))
    labels = np.array([row["gold_label"] for row in rows])
    train, validation = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    return train.tolist(), validation.tolist()


def contexts(rows: list[dict[str, Any]], radius: int) -> tuple[list[str], dict[str, int]]:
    values = []
    found = 0
    mention_counts = []
    for row in rows:
        text, exact_match, mention_count = target_context(
            str(row["article"]), str(row["entity_name"]), radius=radius
        )
        values.append(text)
        found += int(exact_match)
        mention_counts.append(mention_count)
    return values, {
        "exact_entity_match_examples": found,
        "fallback_examples": len(rows) - found,
        "median_entity_mentions": int(statistics.median(mention_counts)) if mention_counts else 0,
        "maximum_entity_mentions": max(mention_counts, default=0),
    }


def prediction_rows(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, model_id: str
) -> list[dict[str, Any]]:
    predictions = scores >= threshold
    return [
        {
            "article_id": row["article_id"],
            "entity_name": row["entity_name"],
            "gold_label": 2 if row["gold_label"] else 1,
            "label": 2 if prediction else 1,
            "label_name": LABEL_NAMES[int(prediction)],
            "adverse_score": float(score),
            "model_id": model_id,
        }
        for row, score, prediction in zip(rows, scores, predictions, strict=True)
    ]


def run_khandpur(
    training_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_indices: list[int],
    validation_indices: list[int],
    radius: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_contexts, training_context_stats = contexts(training_rows, radius)
    test_contexts, test_context_stats = contexts(test_rows, radius)
    training_labels = np.array([row["gold_label"] for row in training_rows])

    best: tuple[float, float, float, CountVectorizer, LogisticRegression] | None = None
    for regularization in (0.25, 1.0, 4.0):
        vectorizer = CountVectorizer(
            lowercase=True,
            binary=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=75_000,
        )
        train_matrix = vectorizer.fit_transform([training_contexts[index] for index in train_indices])
        validation_matrix = vectorizer.transform(
            [training_contexts[index] for index in validation_indices]
        )
        classifier = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(train_matrix, training_labels[train_indices])
        validation_scores = classifier.predict_proba(validation_matrix)[:, 1]
        threshold, validation_f1 = choose_threshold(
            training_labels[validation_indices], validation_scores
        )
        candidate = (validation_f1, -abs(threshold - 0.5), regularization, vectorizer, classifier)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None
    validation_f1, _, regularization, vectorizer, classifier = best
    validation_matrix = vectorizer.transform([training_contexts[index] for index in validation_indices])
    threshold, _ = choose_threshold(
        training_labels[validation_indices], classifier.predict_proba(validation_matrix)[:, 1]
    )
    started = time.perf_counter()
    scores = classifier.predict_proba(vectorizer.transform(test_contexts))[:, 1]
    elapsed = time.perf_counter() - started
    predictions = scores >= threshold
    report = {
        "model_id": "khandpur-entity-relevance-component-reimplementation",
        "comparison_status": "one-component reimplementation trained on CSA labels",
        "source_method": "entity-context bag-of-words logistic regression",
        "configuration": {
            "context_radius_chars": radius,
            "maximum_mentions": 3,
            "features": "binary word unigrams and bigrams",
            "minimum_document_frequency": 2,
            "maximum_features": 75_000,
            "regularization_c": regularization,
            "threshold": threshold,
            "seed": seed,
        },
        "selection": {
            "training_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "validation_f1_negative": validation_f1,
        },
        "context": {"training": training_context_stats, "test": test_context_stats},
        "vocabulary_size": len(vectorizer.vocabulary_),
        "metrics": metrics(np.array([row["gold_label"] for row in test_rows]), predictions),
        "timing": {
            "inference_seconds": elapsed,
            "examples_per_second": len(test_rows) / elapsed if elapsed else None,
        },
    }
    return prediction_rows(test_rows, scores, threshold, report["model_id"]), report


class Vocabulary:
    def __init__(self, sequences: list[list[str]], maximum_size: int = 30_000) -> None:
        frequencies = Counter(token for sequence in sequences for token in sequence)
        ordered = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        self.tokens = ["[PAD]", "[UNK]"] + [token for token, _ in ordered[: maximum_size - 2]]
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.indices.get(token, 1) for token in tokens]


def run_target_bigru(
    training_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_indices: list[int],
    validation_indices: list[int],
    radius: int,
    maximum_tokens: int,
    seed: int,
    epochs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    training_contexts, training_context_stats = contexts(training_rows, radius)
    test_contexts, test_context_stats = contexts(test_rows, radius)
    training_tokenized = [tokenize_target_context(text, maximum_tokens) for text in training_contexts]
    test_tokenized = [tokenize_target_context(text, maximum_tokens) for text in test_contexts]
    vocabulary = Vocabulary([training_tokenized[index][0] for index in train_indices])

    class TargetDataset(Dataset):
        def __init__(self, tokenized: list[tuple[list[str], list[int]]], labels: list[int]) -> None:
            self.tokenized = tokenized
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[list[int], list[int], int]:
            tokens, target_mask = self.tokenized[index]
            return vocabulary.encode(tokens), target_mask, self.labels[index]

    def collate(batch: list[tuple[list[int], list[int], int]]) -> tuple[Any, Any, Any, Any]:
        width = max(len(item[0]) for item in batch)
        token_ids = torch.zeros((len(batch), width), dtype=torch.long)
        target_masks = torch.zeros((len(batch), width), dtype=torch.float32)
        sequence_masks = torch.zeros((len(batch), width), dtype=torch.bool)
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        for row_index, (ids, target_mask, _) in enumerate(batch):
            length = len(ids)
            token_ids[row_index, :length] = torch.tensor(ids)
            target_masks[row_index, :length] = torch.tensor(target_mask)
            sequence_masks[row_index, :length] = True
        return token_ids, target_masks, sequence_masks, labels

    class TargetAwareBiGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(vocabulary.tokens), 64, padding_idx=0)
            self.gru = nn.GRU(65, 64, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.25)
            self.output = nn.Linear(128 * 3, 2)

        def forward(self, token_ids: Any, target_mask: Any, sequence_mask: Any) -> Any:
            embedded = self.embedding(token_ids)
            encoded, final = self.gru(torch.cat((embedded, target_mask.unsqueeze(-1)), dim=-1))
            lengths = sequence_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            mean_pool = (encoded * sequence_mask.unsqueeze(-1)).sum(dim=1) / lengths
            max_pool = encoded.masked_fill(~sequence_mask.unsqueeze(-1), -1e9).max(dim=1).values
            final_pool = torch.cat((final[-2], final[-1]), dim=1)
            return self.output(self.dropout(torch.cat((final_pool, mean_pool, max_pool), dim=1)))

    labels = [int(row["gold_label"]) for row in training_rows]
    train_dataset = TargetDataset([training_tokenized[index] for index in train_indices], [labels[index] for index in train_indices])
    validation_dataset = TargetDataset(
        [training_tokenized[index] for index in validation_indices],
        [labels[index] for index in validation_indices],
    )
    test_dataset = TargetDataset(test_tokenized, [int(row["gold_label"]) for row in test_rows])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate)

    model = TargetAwareBiGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    def score(loader: Any) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        scores: list[float] = []
        gold: list[int] = []
        with torch.inference_mode():
            for token_ids, target_mask, sequence_mask, batch_labels in loader:
                probabilities = torch.softmax(model(token_ids, target_mask, sequence_mask), dim=1)[:, 1]
                scores.extend(probabilities.tolist())
                gold.extend(batch_labels.tolist())
        return np.array(gold), np.array(scores)

    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_validation_f1 = -math.inf
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for token_ids, target_mask, sequence_mask, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(token_ids, target_mask, sequence_mask), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_gold, validation_scores = score(validation_loader)
        threshold, validation_f1 = choose_threshold(validation_gold, validation_scores)
        history.append({
            "epoch": epoch,
            "training_loss": statistics.fmean(losses),
            "validation_f1_negative": validation_f1,
            "threshold": threshold,
        })
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started
    inference_started = time.perf_counter()
    test_gold, test_scores = score(test_loader)
    inference_seconds = time.perf_counter() - inference_started
    predictions = test_scores >= best_threshold
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "model_id": "local-target-aware-bigru",
        "comparison_status": "local neural architecture control trained on CSA labels",
        "source_method": "learned embeddings, target mask, bidirectional GRU, and concatenated last/mean/max pooling",
        "configuration": {
            "context_radius_chars": radius,
            "maximum_mentions": 3,
            "maximum_tokens": maximum_tokens,
            "embedding_dimensions": 64,
            "gru_hidden_dimensions_per_direction": 64,
            "dropout": 0.25,
            "batch_size": 32,
            "learning_rate": 0.002,
            "maximum_epochs": epochs,
            "seed": seed,
        },
        "selection": {
            "training_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "selected_epoch": best_epoch,
            "threshold": best_threshold,
            "validation_f1_negative": best_validation_f1,
            "history": history,
        },
        "context": {"training": training_context_stats, "test": test_context_stats},
        "vocabulary_size": len(vocabulary.tokens),
        "parameters": parameters,
        "metrics": metrics(test_gold, predictions),
        "timing": {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "examples_per_second": len(test_rows) / inference_seconds if inference_seconds else None,
        },
    }
    return prediction_rows(test_rows, test_scores, best_threshold, report["model_id"]), report


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_rows = join_examples(args.training_corpus, args.training_labels)
    test_rows = join_examples(args.test_corpus, args.test_labels)
    if args.limit_training is not None:
        training_rows = training_rows[: args.limit_training]
    if args.limit_test is not None:
        test_rows = test_rows[: args.limit_test]
    if len({row["gold_label"] for row in training_rows}) != 2:
        raise ValueError("limited training data must contain both labels")

    train_indices, validation_indices = split_training(training_rows, args.seed)
    reports = {}
    for model_name, runner in (
        ("khandpur", lambda: run_khandpur(training_rows, test_rows, train_indices, validation_indices, args.context_radius, args.seed)),
        ("bigru", lambda: run_target_bigru(training_rows, test_rows, train_indices, validation_indices, args.context_radius, args.maximum_tokens, args.seed, args.epochs)),
    ):
        if args.model not in ("all", model_name):
            continue
        predictions, report = runner()
        run_directory = args.output_root / report["model_id"]
        write_jsonl(run_directory / "predictions.jsonl", predictions)
        report["fingerprints"] = {
            "training_corpus_sha256": sha256(args.training_corpus),
            "training_labels_sha256": sha256(args.training_labels),
            "test_corpus_sha256": sha256(args.test_corpus),
            "test_labels_sha256": sha256(args.test_labels),
            "predictions_sha256": sha256(run_directory / "predictions.jsonl"),
        }
        report["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        write_json(run_directory / "report.json", report)
        reports[model_name] = report
        print(json.dumps({"event": "baseline_complete", "model": model_name, **report["metrics"]}), flush=True)

    summary = {
        "status": "complete",
        "training_examples": len(training_rows),
        "test_examples": len(test_rows),
        "seed": args.seed,
        "models": {name: report["metrics"] for name, report in reports.items()},
    }
    write_json(args.output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and benchmark CSA comparison baselines")
    parser.add_argument("--training-corpus", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--test-corpus", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("all", "khandpur", "bigru"), default="all")
    parser.add_argument("--context-radius", type=int, default=900)
    parser.add_argument("--maximum-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--limit-training", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()