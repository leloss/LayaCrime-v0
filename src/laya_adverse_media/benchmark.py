from __future__ import annotations

import json
import hashlib
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_NAMES = {1: "positive", 2: "negative"}
SOURCE_LABELS = {"false positive": 1, "hit": 2}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def article_id(entity: str, article: str) -> str:
    normalized = "\n".join((entity.casefold(), " ".join(article.split()).casefold()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def load_source_labels(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    grouped: dict[str, set[int]] = {}
    files_seen = 0
    parse_errors = 0
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        files_seen += 1
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        headers: dict[str, str] = {}
        content: list[str] = []
        in_content = False
        for line in text.splitlines():
            if not in_content and line.startswith("###content:"):
                content.append(line.partition(":")[2].lstrip())
                in_content = True
            elif in_content:
                content.append(line)
            elif line.startswith("###") and ":" in line:
                key, _, value = line[3:].partition(":")
                headers[key.strip().casefold()] = value.strip()
        entity = headers.get("entityname", "").strip()
        disposition = (headers.get("disspositionreason") or headers.get("dispositionreason") or "").strip().casefold()
        article = "\n".join(content).strip()
        if not entity or not article or disposition not in SOURCE_LABELS:
            parse_errors += 1
            continue
        grouped.setdefault(article_id(entity, article), set()).add(SOURCE_LABELS[disposition])
    conflicts = {identifier for identifier, labels in grouped.items() if len(labels) > 1}
    labels = {
        identifier: {"article_id": identifier, "label": next(iter(values)), "annotator": "source disposition"}
        for identifier, values in grouped.items()
        if identifier not in conflicts
    }
    return labels, {"files_seen": files_seen, "usable_rows": len(labels), "conflicts": len(conflicts), "parse_errors": parse_errors}


def load_gpt_labels(gold_path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["article_id"]: row
        for row in read_jsonl(gold_path)
        if type(row.get("label")) is int and row["label"] in LABEL_NAMES
    }


def load_benchmark(corpus_path: Path, gold_path: Path, source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    articles = read_jsonl(corpus_path)
    gpt_by_id = load_gpt_labels(gold_path)
    human_by_id, source_audit = load_source_labels(source_dir)
    annotators = Counter(str(row.get("annotator") or "unknown") for row in gpt_by_id.values())
    return articles, gpt_by_id, human_by_id, {
        "blind_articles": len(articles),
        "usable_gold_rows": len(gpt_by_id),
        "human_gold_rows": len(human_by_id),
        "annotators": dict(sorted(annotators.items())),
        "source_audit": source_audit,
        "corpus_path": corpus_path.as_posix() if not corpus_path.is_absolute() else corpus_path.name,
        "gold_path": gold_path.as_posix() if not gold_path.is_absolute() else gold_path.name,
        "anti_leakage": "Laya receives only article and entity_name; gold is joined after inference.",
    }


class BenchmarkState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.status = "idle"
            self.total = 0
            self.completed = 0
            self.started_at: float | None = None
            self.finished_at: float | None = None
            self.inference_seconds = 0.0
            self.matrices = {"human": [[0, 0], [0, 0]], "gpt": [[0, 0], [0, 0]]}
            self.compared = {"human": 0, "gpt": 0}
            self.current: dict[str, Any] | None = None
            self.pending: dict[str, Any] | None = None
            self.predictions: dict[str, int] = {}
            self.probability_series: list[dict[str, float | int]] = []
            self.error: str | None = None
            self.pause_requested = False
            self.audit: dict[str, Any] = {}

    def start(self, total: int, audit: dict[str, Any], *, resume: bool = False) -> None:
        with self._lock:
            if not resume:
                self.total = total
                self.completed = 0
                self.started_at = time.perf_counter()
                self.finished_at = None
                self.inference_seconds = 0.0
                self.matrices = {"human": [[0, 0], [0, 0]], "gpt": [[0, 0], [0, 0]]}
                self.compared = {"human": 0, "gpt": 0}
                self.current = None
                self.pending = None
                self.predictions = {}
                self.probability_series = []
                self.error = None
                self.audit = audit
            self.status = "running"
            self.pause_requested = False

    def begin_item(self, article: dict[str, Any]) -> None:
        with self._lock:
            self.pending = {
                "article_id": article["article_id"],
                "entity_name": article["entity_name"],
            }

    def record(self, article: dict[str, Any], prediction: dict[str, Any], truths: dict[str, dict[str, Any] | None], seconds: float) -> None:
        predicted_label = 2 if prediction["decision"] == "negative" else 1
        with self._lock:
            for source, truth in truths.items():
                if truth is not None:
                    self.matrices[source][truth["label"] - 1][predicted_label - 1] += 1
                    self.compared[source] += 1
            self.completed += 1
            self.inference_seconds += seconds
            self.predictions[article["article_id"]] = predicted_label
            self.probability_series.append({
                "index": self.completed,
                "negative": -100 * prediction["probabilities"]["negative"],
                "positive": 100 * prediction["probabilities"]["positive"],
            })
            display_gold = truths["gpt"] or truths["human"]
            self.current = {
                "article_id": article["article_id"],
                "entity_name": article["entity_name"],
                "article": article["article"],
                "phase": "complete",
                "elapsed_seconds": seconds,
                "correct": predicted_label == display_gold["label"] if display_gold else None,
                "prediction": {
                    "label": predicted_label,
                    "label_name": LABEL_NAMES[predicted_label],
                    **prediction,
                },
                "gold": ({
                    "label": display_gold["label"],
                    "label_name": display_gold.get("label_name") or LABEL_NAMES[display_gold["label"]],
                    "confidence": display_gold.get("confidence"),
                    "rationale": display_gold.get("rationale"),
                    "evidence": display_gold.get("evidence", []),
                    "annotator": display_gold.get("annotator"),
                    "source": "gpt" if truths["gpt"] else "human",
                } if display_gold else None),
            }
            self.pending = None

    def refresh_gpt(self, labels: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            matrix = [[0, 0], [0, 0]]
            compared = 0
            for identifier, predicted in self.predictions.items():
                truth = labels.get(identifier)
                if truth:
                    matrix[truth["label"] - 1][predicted - 1] += 1
                    compared += 1
            self.matrices["gpt"] = matrix
            self.compared["gpt"] = compared
            self.audit["usable_gold_rows"] = len(labels)
            self.audit["annotators"] = dict(sorted(Counter(str(row.get("annotator") or "unknown") for row in labels.values()).items()))

    def restore(
        self,
        rows: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        gpt_labels: dict[str, dict[str, Any]],
        human_labels: dict[str, dict[str, Any]],
        audit: dict[str, Any],
        *,
        total: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        articles_by_id = {row["article_id"]: row for row in articles}
        self.reset()
        self.start(total or len(rows), audit)
        for row in rows:
            article = articles_by_id.get(row.get("article_id"))
            if article is None:
                continue
            probabilities = row.get("probabilities") or {}
            decision = row.get("decision")
            if decision not in {"negative", "positive"}:
                continue
            prediction = {
                "decision": decision,
                "confidence": float(row.get("confidence", 0.0)),
                "probabilities": {
                    "negative": float(probabilities.get("negative", 0.0)),
                    "positive": float(probabilities.get("positive", 0.0)),
                },
                "needs_review": bool(row.get("needs_review", False)),
                "routing": row.get("routing"),
            }
            truths = {
                "gpt": gpt_labels.get(article["article_id"]),
                "human": human_labels.get(article["article_id"]),
            }
            self.record(article, prediction, truths, float(row.get("elapsed_seconds", 0.0)))
        with self._lock:
            now = time.perf_counter()
            wall_seconds = max(float(elapsed_seconds or self.inference_seconds), 0.0)
            self.started_at = now - wall_seconds
            self.finished_at = now if self.completed >= self.total else None
            self.status = "complete" if self.completed >= self.total else "paused"

    def update_audit(self, audit: dict[str, Any]) -> None:
        with self._lock:
            self.audit.update(audit)

    def request_pause(self) -> None:
        with self._lock:
            if self.status == "running":
                self.pause_requested = True

    def should_pause(self) -> bool:
        with self._lock:
            return self.pause_requested

    def paused(self) -> None:
        with self._lock:
            self.status = "paused"
            self.pause_requested = False

    def finish(self) -> None:
        with self._lock:
            self.status = "complete"
            self.finished_at = time.perf_counter()

    def fail(self, error: Exception) -> None:
        with self._lock:
            self.status = "error"
            self.error = str(error)
            self.finished_at = time.perf_counter()

    def snapshot(self, series_after: int = 0) -> dict[str, Any]:
        with self._lock:
            matrices = {name: [row[:] for row in matrix] for name, matrix in self.matrices.items()}
            compared = dict(self.compared)
            completed = self.completed
            started_at = self.started_at
            finished_at = self.finished_at
            inference_seconds = self.inference_seconds
            payload = {
                "status": self.status,
                "total": self.total,
                "completed": completed,
                "current": self.current,
                "pending": self.pending,
                "probability_series": self.probability_series[series_after:],
                "error": self.error,
                "audit": dict(self.audit),
            }
        metrics = {}
        for source, matrix in matrices.items():
            true_positive, true_negative = matrix[1][1], matrix[0][0]
            false_positive, false_negative = matrix[0][1], matrix[1][0]
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            metrics[source] = {
                "compared": compared[source],
                "accuracy": (true_positive + true_negative) / compared[source] if compared[source] else 0.0,
                "precision_negative": precision,
                "recall_negative": recall,
                "f1_negative": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        now = finished_at or time.perf_counter()
        wall_seconds = now - started_at if started_at is not None else 0.0
        payload.update({
            "progress": completed / self.total if self.total else 0.0,
            "confusion_matrices": {
                source: {
                "orientation": f"rows={source} ground truth, columns=Laya prediction",
                "labels": ["positive", "negative"],
                "values": matrix,
                "compared": compared[source],
                } for source, matrix in matrices.items()
            },
            "metrics": metrics,
            "timing": {
                "elapsed_seconds": wall_seconds,
                "inference_seconds": inference_seconds,
                "average_seconds": inference_seconds / completed if completed else 0.0,
                "items_per_second": completed / inference_seconds if inference_seconds else 0.0,
            },
        })
        return payload