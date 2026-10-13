import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_academic_baselines.py"
SPEC = importlib.util.spec_from_file_location("compare_academic_baselines", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_mcnemar_is_symmetric_and_handles_no_disagreement() -> None:
    assert MODULE.exact_mcnemar(0, 0) == 1.0
    assert MODULE.exact_mcnemar(8, 2) == MODULE.exact_mcnemar(2, 8)
    assert 0 < MODULE.exact_mcnemar(8, 2) < 1


def test_holm_adjustment_is_monotonic_in_rank() -> None:
    adjusted = MODULE.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})

    assert adjusted["a"] == 0.003
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_binary_metrics_are_macro_with_annotation_literal_confusion() -> None:
    metrics = MODULE.binary_metrics(np.array([1, 1, 1, 2]), np.array([1, 1, 2, 2]))

    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(5 / 6)
    assert metrics["f1"] == pytest.approx(11 / 15)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1