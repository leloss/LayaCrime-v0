import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_academic_baselines.py"
SPEC = importlib.util.spec_from_file_location("benchmark_academic_baselines", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_target_context_marks_entity_and_bounds_long_input() -> None:
    article = "prefix " * 1_000 + "Acme Corp was charged with fraud. " + "suffix " * 1_000

    context, found, mentions = MODULE.target_context(article, "Acme Corp", radius=100)

    assert found is True
    assert mentions == 1
    assert "[TARGET] Acme Corp [/TARGET]" in context
    assert len(context) < 300


def test_target_context_has_explicit_fallback_for_missing_entity() -> None:
    context, found, mentions = MODULE.target_context("Other people were charged.", "Acme", radius=20)

    assert found is False
    assert mentions == 0
    assert context.startswith("[TARGET] Acme [/TARGET] [SEP]")


def test_tokenizer_preserves_target_mask_when_truncating() -> None:
    context = " ".join(["before"] * 50) + " [TARGET] Acme [/TARGET] " + " ".join(["after"] * 50)

    tokens, mask = MODULE.tokenize_target_context(context, maximum_tokens=20)

    assert len(tokens) == 20
    assert sum(mask) == 1
    assert tokens[mask.index(1)] == "acme"


def test_threshold_and_metrics_use_negative_as_event() -> None:
    gold = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])

    threshold, _ = MODULE.choose_threshold(gold, scores)
    report = MODULE.metrics(gold, scores >= threshold)

    assert threshold > 0.1
    assert report["true_positive"] == 2
    assert report["false_negative"] == 0
    assert report["f1_negative"] >= 0.8