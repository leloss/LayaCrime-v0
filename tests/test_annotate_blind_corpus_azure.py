import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "annotate_blind_corpus_azure.py"
SPEC = importlib.util.spec_from_file_location("annotate_blind_corpus_azure", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Responses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            status="completed", output_text=json.dumps(self.payload)
        )


def test_classification_is_structured_unstored_and_grounded() -> None:
    responses = Responses({
        "label": 2,
        "confidence": 0.91,
        "rationale": "Acme is charged.",
        "evidence": ["Acme was charged", "not in the article"],
        "reference_resolution": "literal_match",
    })
    client = SimpleNamespace(responses=responses)
    article = {
        "article_id": "one",
        "entity_name": "Acme",
        "article": "Authorities said Acme was charged with fraud.",
    }

    row = MODULE.classify_article(client, "gpt-5.6-terra", article, 1500)

    assert row["label_name"] == "bad_guy"
    assert row["evidence"] == ["Acme was charged"]
    assert row["annotator"] == "gpt-5.6-terra"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"]["format"]["strict"] is True


def test_publish_requires_exact_coverage_and_preserves_previous_ledger(tmp_path) -> None:
    corpus = [{"article_id": "one"}, {"article_id": "two"}]
    ledger = tmp_path / "annotations.jsonl"
    ledger.write_text('{"old": true}\n', encoding="utf-8")
    rows = [
        {
            "article_id": identifier,
            "label": 1,
            "label_name": "good_guy",
            "confidence": 0.9,
            "rationale": "No adverse conduct.",
        }
        for identifier in ("one", "two")
    ]

    MODULE.publish_annotations(corpus, rows, ledger)

    assert len(MODULE.read_jsonl(ledger)) == 2
    assert ledger.with_name("annotations.pre-azure-backup.jsonl").read_text(
        encoding="utf-8"
    ) == '{"old": true}\n'