from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable

LABELS = {"positive": (1, "positive"), "negative": (2, "negative")}
BATCH_QUESTIONS = {
    "criminal_association": {
        "type": "choice",
        "instructions": (
            "How does this article portray the named entity in the entity_name field regarding "
            "criminal behavior? Judge only that entity, not other people or organizations."
        ),
        "criteria": {
            "A": (
                "negative: the article credibly associates the entity with alleged, investigated, "
                "charged, convicted, sanctioned, or admitted criminal behavior"
            ),
            "B": (
                "positive: the article does not associate the entity with criminal behavior, or "
                "identifies the entity only as a victim, witness, investigator, or unrelated party"
            ),
        },
    }
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def add_reference_resolution(article: dict, prediction: dict) -> dict:
    entity_found = normalized_text(article["entity_name"]) in normalized_text(article["article"])
    enriched = dict(prediction)
    enriched["entity_name_literal_match"] = entity_found
    enriched["reference_resolution"] = "literal_match" if entity_found else "needs_entity_resolution"
    if not entity_found:
        enriched["needs_review"] = True
    return enriched


def format_prediction(article: dict, result: dict) -> dict:
    decision = result.get("decision")
    if decision not in LABELS:
        raise ValueError(f"unsupported server decision {decision!r}")
    label, label_name = LABELS[decision]
    return add_reference_resolution(article, {
        "article_id": article["article_id"],
        "label": label,
        "label_name": label_name,
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "needs_review": result["needs_review"],
        "routing": result.get("routing"),
        "producer": "laya-router",
        "status": "model_predicted",
    })


def request_prediction(endpoint: str, article: dict, timeout: float) -> dict:
    body = json.dumps(
        {
            "article": article["article"],
            "entity_name": article["entity_name"],
            "routing_mode": "auto",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"prediction request failed with HTTP {error.code}: {detail}") from error
    return format_prediction(article, result)


def direct_predictions(articles: list[dict], batch_size: int) -> Iterable[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya_adverse_media.app import _build_predictor, _parse_result

    router = _build_predictor()
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for article in articles:
        state = {"article": article["article"], "entity_name": article["entity_name"]}
        route = router.route(state, BATCH_QUESTIONS)
        grouped[route["model"]].append((article, dict(route)))

    for model_name, routed_articles in grouped.items():
        agent = router.load(model_name)
        for start in range(0, len(routed_articles), batch_size):
            part = routed_articles[start : start + batch_size]
            states = [
                {"article": article["article"], "entity_name": article["entity_name"]}
                for article, _ in part
            ]
            results = agent.predict_batch(states, BATCH_QUESTIONS, batch_size=batch_size)
            for (article, route), result in zip(part, results):
                result["routing"] = route
                response = _parse_result(result, article["entity_name"], review_threshold=0.8)
                yield format_prediction(article, response.model_dump())


def predict(args: argparse.Namespace) -> dict:
    articles = read_jsonl(args.corpus)
    if args.limit is not None:
        articles = articles[: args.limit]

    completed: set[str] = set()
    mode = "w"
    if args.resume and args.output.exists():
        articles_by_id = {row["article_id"]: row for row in articles}
        existing = read_jsonl(args.output)
        upgraded = [
            add_reference_resolution(articles_by_id[row["article_id"]], row)
            for row in existing
            if row["article_id"] in articles_by_id
        ]
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            for row in upgraded:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed = {row["article_id"] for row in upgraded}
        mode = "a"

    pending = [row for row in articles if row["article_id"] not in completed]
    predictions = (
        direct_predictions(pending, args.batch_size)
        if args.direct
        else (request_prediction(args.endpoint, article, args.timeout) for article in pending)
    )
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8", newline="\n") as output:
        for index, prediction in enumerate(predictions, start=1):
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            if index % args.report_every == 0:
                elapsed = time.perf_counter() - started
                print(f"predicted={index} rate={index / elapsed:.2f}/s", flush=True)

    elapsed = time.perf_counter() - started
    return {
        "selected": len(articles),
        "previously_completed": len(completed),
        "predicted_now": len(pending),
        "elapsed_seconds": round(elapsed, 2),
        "predictions_per_second": round(len(pending) / elapsed, 3) if pending else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run blind adverse-media predictions through the local Laya server")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/adverse-media")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Use the local Router batch API instead of HTTP")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    print(json.dumps(predict(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()