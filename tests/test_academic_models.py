from laya_adverse_media.academic_models import KHANDPUR_SEED, _clean_tartu_text


def test_khandpur_runtime_uses_canonical_paper_seed() -> None:
    assert KHANDPUR_SEED == 20261005


def test_tartu_runtime_normalizes_whitespace_like_canonical_runner() -> None:
    assert _clean_tartu_text("First\n\n  second!") == "first second"