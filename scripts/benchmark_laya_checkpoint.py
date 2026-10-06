from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        (truth, prediction): 0
        for truth in ("positive", "negative")
        for prediction in ("positive", "negative")
    }
    for row in rows:
        truth = "negative" if row["gold_label"] == 2 else "positive"
        counts[truth, row["decision"]] += 1
    true_positive = counts["negative", "negative"]
    false_positive = counts["positive", "negative"]
    false_negative = counts["negative", "positive"]
    true_negative = counts["positive", "positive"]
    total = len(rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "examples": total,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "latency_seconds": {
            "mean": statistics.mean(elapsed) if elapsed else 0.0,
            "median": statistics.median(elapsed) if elapsed else 0.0,
            "p95": sorted(elapsed)[max(0, math.ceil(0.95 * total) - 1)] if elapsed else 0.0,
        },
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya import Agent

    from laya_adverse_media.app import _parse_result, _questions

    articles = read_jsonl(args.corpus)
    gold_rows = read_jsonl(args.labels)
    gold = {row["article_id"]: int(row["label"]) for row in gold_rows}
    if len(articles) != len(gold) or any(row["article_id"] not in gold for row in articles):
        raise ValueError("corpus and labels must contain the same article IDs")

    config = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    prompt = config.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("exported checkpoint does not contain prompt metadata")
    criteria = prompt.get("criteria")
    question = prompt.get("question")
    if not isinstance(question, str) or not isinstance(criteria, list):
        raise ValueError("exported checkpoint prompt metadata is invalid")

    existing = read_jsonl(args.output) if args.resume and args.output.is_file() else []
    completed = {row["article_id"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("existing benchmark ledger contains duplicate article IDs")
    pending = [row for row in articles if row["article_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    print(json.dumps({
        "event": "benchmark_start",
        "model_dir": str(args.model_dir),
        "examples": len(articles),
        "previously_completed": len(existing),
        "pending": len(pending),
        "device": args.device,
    }), flush=True)

    agent = Agent(str(args.model_dir), device=args.device)
    started = time.perf_counter()
    try:
        with args.output.open(mode, encoding="utf-8", newline="\n") as destination:
            for index, article in enumerate(pending, start=1):
                state = {
                    "article": article["article"],
                    "entity_name": article["entity_name"],
                }
                questions = _questions(article["entity_name"], question, criteria)
                item_started = time.perf_counter()
                result = agent.system_one(state, questions)
                response = _parse_result(
                    result,
                    article["entity_name"],
                    review_threshold=args.review_threshold,
                )
                elapsed_seconds = time.perf_counter() - item_started
                row = {
                    "article_id": article["article_id"],
                    "entity_name": article["entity_name"],
                    "decision": response.decision,
                    "confidence": response.confidence,
                    "probabilities": response.probabilities,
                    "needs_review": response.needs_review,
                    "elapsed_seconds": elapsed_seconds,
                    "routing": {
                        "model": args.run_id,
                        "provider": "laya-checkpoint",
                        "model_dir": str(args.model_dir),
                    },
                    "gold_label": gold[article["article_id"]],
                }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                if index % args.report_every == 0 or index == len(pending):
                    elapsed = time.perf_counter() - started
                    print(json.dumps({
                        "event": "benchmark_progress",
                        "completed_now": index,
                        "completed_total": len(existing) + index,
                        "total": len(articles),
                        "rate_per_second": index / elapsed if elapsed else None,
                    }), flush=True)
    finally:
        del agent
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    rows = read_jsonl(args.output)
    if len(rows) != len(articles):
        raise RuntimeError(f"benchmark ledger is incomplete: {len(rows)} of {len(articles)}")
    report = {
        "run_id": args.run_id,
        "status": "complete",
        "model_dir": str(args.model_dir),
        "model_config_sha256": sha256(args.model_dir / "rl_agent_config.json"),
        "model_weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "corpus_sha256": sha256(args.corpus),
        "labels_sha256": sha256(args.labels),
        "ledger_sha256": sha256(args.output),
        "metrics": benchmark_metrics(rows),
    }
    write_json(args.report, report)
    print(json.dumps({"event": "benchmark_complete", **report}, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one exported Laya checkpoint against labeled public data"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--review-threshold", type=float, default=0.8)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
