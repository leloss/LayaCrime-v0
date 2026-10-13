import csv
import importlib.util
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_tartu_baseline.py"
SPEC = importlib.util.spec_from_file_location("benchmark_tartu_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_archive(path: Path, rows: list[dict[str, str]]) -> None:
    csv_path = path.with_suffix("")
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["title", "article", "label"])
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(csv_path, csv_path.name)


def test_load_released_training_applies_source_label_filter(tmp_path: Path) -> None:
    write_archive(
        tmp_path / "adverse_media_training.csv.zip",
        [
            {"title": "A", "article": "adverse", "label": "am "},
            {"title": "B", "article": "not adverse", "label": "nam"},
            {"title": "C", "article": "excluded", "label": "doubt"},
        ],
    )
    write_archive(
        tmp_path / "non_adverse_media_training.csv.zip",
        [
            {"title": "D", "article": "cross-labeled adverse", "label": "am"},
            {"title": "E", "article": "random negative", "label": "random"},
        ],
    )

    texts, labels, counts = MODULE.load_released_training(tmp_path)

    assert texts == ["A adverse", "B not adverse", "D cross-labeled adverse", "E random negative"]
    assert labels.tolist() == [1, 0, 1, 0]
    assert counts == {
        "released_rows": 5,
        "included_rows": 4,
        "adverse_rows": 2,
        "non_adverse_rows": 2,
    }


def test_vectorizer_matches_released_notebook_configuration() -> None:
    vectorizer = MODULE.build_vectorizer()

    assert vectorizer.max_features == 40_000
    assert vectorizer.min_df == 5
    assert vectorizer.max_df == 0.5
    assert vectorizer.analyzer == "word"
    assert vectorizer.stop_words == "english"
    assert vectorizer.ngram_range == (1, 3)


def test_clean_text_matches_released_regular_expressions() -> None:
    assert MODULE.clean_text("Visit http://example.test Foo-123!\nNow") == "visit foo now"
