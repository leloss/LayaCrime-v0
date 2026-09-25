import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_source_labels.py"
SPEC = importlib.util.spec_from_file_location("compare_source_labels", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_source_maps_original_labels_to_annotation_labels(tmp_path) -> None:
    source = tmp_path / "article"
    source.write_text(
        "###entityName: Acme\n"
        "###disspositionReason: Hit\n"
        "###content: Acme was charged with fraud.\n",
        encoding="utf-8",
    )

    identifier, label = MODULE.parse_source(source)

    assert identifier == MODULE.article_id("Acme", "Acme was charged with fraud.")
    assert label == 2


def test_metrics_uses_original_rows_and_prediction_columns() -> None:
    predictions = [
        {"article_id": "tn", "label": 1},
        {"article_id": "fp", "label": 2},
        {"article_id": "fn", "label": 1},
        {"article_id": "tp", "label": 2},
    ]
    source = {"tn": 1, "fp": 1, "fn": 2, "tp": 2}

    report = MODULE.metrics(predictions, source)

    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    assert report["bad_guy_as_positive"]["accuracy"] == 0.5
    assert report["bad_guy_as_positive"]["precision"] == 0.5
    assert report["bad_guy_as_positive"]["recall"] == 0.5