import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_named_entities.py"
SPEC = importlib.util.spec_from_file_location("extract_named_entities", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_project_endpoint_resolves_to_openai_v1_without_project_name() -> None:
    endpoint = "https://example.services.ai.azure.com/api/projects/private-project"

    result = MODULE.openai_base_url(endpoint)

    assert result == "https://example.openai.azure.com/openai/v1/"
    assert "private-project" not in result


def test_extract_chunk_disables_response_storage() -> None:
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {"status": "completed", "output_text": json.dumps({"entities": []})},
            )()

    client = type("Client", (), {"responses": Responses()})()

    assert MODULE.extract_chunk(client, "deployment", "article", 100) == []
    assert captured["store"] is False


def test_resilient_extraction_splits_incomplete_output() -> None:
    def extractor(client, deployment, text, max_output_tokens):
        if len(text) > 20:
            raise MODULE.IncompleteExtractionError("too long")
        return [{"name": text[:1], "entity_type": "other_named_entity", "mentions": [text[:1]]}]

    groups, filtered, failed = MODULE.extract_fragment_resilient(
        None, "deployment", "A" * 60, 100, 15, extractor
    )

    assert len(groups) > 1
    assert filtered == 0
    assert failed == 0


def test_irreducible_content_filter_is_marked_partial() -> None:
    class FilteredError(Exception):
        code = "content_filter"

    def extractor(client, deployment, text, max_output_tokens):
        raise FilteredError()

    groups, filtered, failed = MODULE.extract_fragment_resilient(
        None, "deployment", "blocked", 100, 20, extractor
    )

    assert groups == []
    assert filtered == 1
    assert failed == 1


def test_chunks_cover_long_text_with_overlap() -> None:
    text = "A" * 250

    parts = MODULE.chunks(text, max_chars=100, overlap=10)

    assert len(parts) == 3
    assert parts[0][-10:] == parts[1][:10]
    assert parts[1][-10:] == parts[2][:10]


def test_merge_entities_discards_ungrounded_names_and_mentions() -> None:
    article = "Alice works at Acme Bank. Alice met Bob."
    groups = [[
        {"name": "Alice", "entity_type": "person", "mentions": ["Alice", "Alicia"]},
        {"name": "Acme Bank", "entity_type": "bank_or_financial_institution", "mentions": ["Acme Bank"]},
        {"name": "Invented Corp", "entity_type": "company", "mentions": []},
    ]]

    entities = MODULE.merge_entities(groups, article)

    assert [entity["name"] for entity in entities] == ["Acme Bank", "Alice"]
    assert entities[1]["mentions"] == ["Alice"]


def test_build_targets_creates_stable_entity_article_pairs() -> None:
    corpus = [{"article_id": "article-1", "article": "Alice joined Acme.", "entity_name": "Other"}]
    extractions = [{
        "article_id": "article-1",
        "entities": [
            {"name": "Alice", "entity_type": "person", "mentions": ["Alice"]},
            {"name": "Acme", "entity_type": "company", "mentions": ["Acme"]},
        ],
    }]

    targets = MODULE.build_targets(corpus, extractions)

    assert len(targets) == 2
    assert {target["entity_name"] for target in targets} == {"Alice", "Acme"}
    assert all(target["article"] == "Alice joined Acme." for target in targets)