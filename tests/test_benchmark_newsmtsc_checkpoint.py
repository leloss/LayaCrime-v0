import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_newsmtsc_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("benchmark_newsmtsc_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

WORKER_SCRIPT = Path(__file__).parents[1] / "scripts" / "newsmtsc_worker.py"
WORKER_SPEC = importlib.util.spec_from_file_location("newsmtsc_worker", WORKER_SCRIPT)
assert WORKER_SPEC and WORKER_SPEC.loader
WORKER_MODULE = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(WORKER_MODULE)


def test_target_sentence_selects_and_bounds_first_literal_mention() -> None:
    article = "Earlier sentence. " + "left " * 100 + "Acme was charged with fraud. Later sentence."

    context = MODULE.target_sentence(article, "ACME", maximum_chars_per_side=40)

    assert context is not None
    left, target, right = context
    assert len(left) <= 40
    assert target == "Acme"
    assert right == " was charged with fraud."


def test_target_sentence_requires_literal_mention() -> None:
    assert MODULE.target_sentence("The company was charged.", "Acme") is None


def test_classify_results_uses_negative_argmax_not_binary_threshold() -> None:
    results = [
        (
            {"class_label": "negative", "class_prob": 0.40},
            {"class_label": "neutral", "class_prob": 0.35},
            {"class_label": "positive", "class_prob": 0.25},
        ),
        (
            {"class_label": "negative", "class_prob": 0.40},
            {"class_label": "neutral", "class_prob": 0.50},
            {"class_label": "positive", "class_prob": 0.10},
        ),
    ]

    scores, predictions = MODULE.classify_results(results)

    assert scores.tolist() == [0.4, 0.4]
    assert predictions.tolist() == [1, 0]


def test_worker_uses_three_class_argmax_for_ui_choice() -> None:
    class Classifier:
        def infer(self, **kwargs):
            return (
                {"class_label": "negative", "class_prob": 0.40},
                {"class_label": "neutral", "class_prob": 0.50},
                {"class_label": "positive", "class_prob": 0.10},
            )

    result = WORKER_MODULE._result(
        Classifier(), "Acme announced its results.", "Acme"
    )

    decision = result["answers"]["criminal_association"]
    assert decision["choice"] == "B"
    assert decision["probabilities"] == {"A": 0.4, "B": 0.6}
