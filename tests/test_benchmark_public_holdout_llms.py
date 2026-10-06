import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_public_holdout_llms.py"
SPEC = importlib.util.spec_from_file_location("benchmark_public_holdout_llms", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Responses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps({
                "label": 2,
                "confidence": 0.9,
                "rationale": "Acme is charged.",
            }),
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_classification_is_unstored_structured_and_excludes_gold() -> None:
    responses = Responses()
    article = {
        "article_id": "one",
        "entity_name": "Acme",
        "article": "Authorities charged Acme with fraud.",
    }

    row = MODULE.classify_article(SimpleNamespace(responses=responses), "gpt-5.4-nano", article, 500)

    assert row["label_name"] == "negative"
    assert row["usage"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert "label" not in json.loads(responses.kwargs["input"])


def test_classification_records_explicit_article_truncation() -> None:
    responses = Responses()
    article = {"article_id": "one", "entity_name": "Acme", "article": "0123456789"}

    row = MODULE.classify_article(
        SimpleNamespace(responses=responses), "gpt-5.4-nano", article, 500, 4
    )

    assert json.loads(responses.kwargs["input"])["article"] == "0123"
    assert row["input"] == {
        "original_article_chars": 10,
        "submitted_article_chars": 4,
        "article_truncated": True,
    }


def test_report_computes_negative_class_metrics_tokens_and_cost(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(corpus, [
        {"article_id": "a", "entity_name": "A", "article": "A"},
        {"article_id": "b", "entity_name": "B", "article": "B"},
        {"article_id": "c", "entity_name": "C", "article": "C"},
        {"article_id": "d", "entity_name": "D", "article": "D"},
    ])
    write_jsonl(gold, [
        {"article_id": "a", "label": 1},
        {"article_id": "b", "label": 1},
        {"article_id": "c", "label": 2},
        {"article_id": "d", "label": 2},
    ])
    predictions = [
        {
            "article_id": identifier,
            "label": label,
            "elapsed_seconds": elapsed,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        for identifier, label, elapsed in (
            ("a", 1, 1.0),
            ("b", 2, 2.0),
            ("c", 2, 3.0),
            ("d", 2, 4.0),
        )
    ]

    report = MODULE.benchmark_report(
        "model",
        corpus,
        gold,
        predictions,
        5.0,
        {"input_usd_per_million": 2.0, "output_usd_per_million": 10.0},
    )

    assert report["confusion_matrix"]["values"] == [[1, 1], [0, 2]]
    assert report["coverage"] == 1.0
    assert report["abstentions"] == {"total": 0, "gold_labels": {}}
    assert report["metrics"]["accuracy"] == 0.75
    assert report["metrics"]["end_to_end_accuracy"] == 0.75
    assert report["metrics"]["precision_negative"] == 2 / 3
    assert report["metrics"]["recall_negative"] == 1.0
    assert report["tokens"]["total"] == 440
    assert report["cost"]["estimated_total"] == 0.0012