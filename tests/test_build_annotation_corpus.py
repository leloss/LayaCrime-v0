import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_annotation_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_annotation_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_blind_article_excludes_untrusted_metadata(tmp_path) -> None:
    source = tmp_path / "article"
    source.write_text(
        "###url: https://example.test\n"
        "###relevancy_score: 1.0\n"
        "###entityName: Acme Corp\n"
        "###disspositionReason: Hit\n"
        "###content: Acme Corp was charged with fraud.\n",
        encoding="utf-8",
    )

    article = MODULE.parse_blind_article(source)

    assert article.entity_name == "Acme Corp"
    assert article.article == "Acme Corp was charged with fraud."
    assert "Hit" not in repr(article)
    assert "relevancy" not in repr(article)


def test_blind_corpus_deduplicates_without_reading_labels(tmp_path) -> None:
    content = "###entityName: Acme\n###content: Same article\n"
    (tmp_path / "one").write_text("###disspositionReason: Hit\n" + content, encoding="utf-8")
    (tmp_path / "two").write_text("###disspositionReason: False Positive\n" + content, encoding="utf-8")

    articles, report = MODULE.unique_articles(sorted(tmp_path.iterdir()))

    assert len(articles) == 1
    assert report["duplicate_copies_removed"] == 1