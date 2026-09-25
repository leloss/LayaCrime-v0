import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_annotations.py"
SPEC = importlib.util.spec_from_file_location("merge_annotations", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def article() -> dict:
    return {"article_id": "abc", "entity_name": "Acme", "article": "Acme was charged with fraud."}


def prediction(**changes) -> dict:
    row = {
        "article_id": "abc",
        "label": 2,
        "label_name": "bad_guy",
        "confidence": 0.9,
        "rationale": "Acme was charged.",
        "evidence": ["charged with fraud"],
        "annotator": "tester",
        "status": "ai_annotated",
    }
    row.update(changes)
    return row


def test_valid_prediction_accepts_normalized_evidence() -> None:
    row = prediction(evidence=["ACME   was charged"])

    assert MODULE.validate_predictions([article()], [row]) == []


def test_validation_rejects_invalid_label_and_unsupported_evidence() -> None:
    row = prediction(label=1, label_name="bad_guy", evidence=["convicted of fraud"])

    errors = MODULE.validate_predictions([article()], [row])

    assert "abc: label_name does not match label" in errors
    assert any("evidence is not present" in error for error in errors)


def test_validation_requires_exact_id_coverage() -> None:
    errors = MODULE.validate_predictions([article()], [])

    assert errors == ["missing predictions: abc"]