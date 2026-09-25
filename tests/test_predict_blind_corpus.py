import importlib.util
import io
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "predict_blind_corpus.py"
SPEC = importlib.util.spec_from_file_location("predict_blind_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeResponse(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_request_prediction_maps_negative_to_bad_guy(monkeypatch) -> None:
    response = {
        "decision": "negative",
        "confidence": 0.8,
        "probabilities": {"negative": 0.9, "positive": 0.1},
        "needs_review": False,
        "routing": {"model": "english"},
    }
    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(json.dumps(response)),
    )

    prediction = MODULE.request_prediction(
        "http://localhost/predict",
        {"article_id": "abc", "entity_name": "Acme", "article": "Acme was charged."},
        1,
    )

    assert prediction["article_id"] == "abc"
    assert prediction["label"] == 2
    assert prediction["label_name"] == "bad_guy"
    assert prediction["annotator"] == "laya-router"


def test_format_prediction_maps_positive_to_good_guy() -> None:
    prediction = MODULE.format_prediction(
        {"article_id": "xyz", "entity_name": "Acme", "article": "Acme was cleared."},
        {
            "decision": "positive",
            "confidence": 0.7,
            "probabilities": {"negative": 0.2, "positive": 0.8},
            "needs_review": True,
            "routing": {"model": "multilingual"},
        },
    )

    assert prediction["label"] == 1
    assert prediction["label_name"] == "good_guy"
    assert prediction["entity_name_literal_match"] is True


def test_missing_reference_name_forces_review_without_changing_label() -> None:
    prediction = MODULE.format_prediction(
        {"article_id": "xyz", "entity_name": "Acme Corporation", "article": "The company was charged."},
        {
            "decision": "negative",
            "confidence": 0.95,
            "probabilities": {"negative": 0.99, "positive": 0.01},
            "needs_review": False,
            "routing": {"model": "english"},
        },
    )

    assert prediction["label"] == 2
    assert prediction["confidence"] == 0.95
    assert prediction["entity_name_literal_match"] is False
    assert prediction["reference_resolution"] == "needs_entity_resolution"
    assert prediction["needs_review"] is True