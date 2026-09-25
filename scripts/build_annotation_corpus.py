from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BlindArticle:
    article_id: str
    entity_name: str
    article: str


def default_data_dir() -> Path:
    return Path(os.getenv("LAYA_SOURCE_DATA_DIR", "data/source"))


def parse_blind_article(path: Path) -> BlindArticle:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    entity = ""
    content_lines: list[str] = []
    in_content = False
    for line in text.splitlines():
        if not in_content and line.startswith("###content:"):
            content_lines.append(line.partition(":")[2].lstrip())
            in_content = True
        elif in_content:
            content_lines.append(line)
        elif line.lower().startswith("###entityname:"):
            entity = line.partition(":")[2].strip()

    article = "\n".join(content_lines).strip()
    if not entity:
        raise ValueError("missing entityName")
    if not article:
        raise ValueError("missing content")
    normalized = "\n".join((entity.casefold(), " ".join(article.split()).casefold()))
    article_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return BlindArticle(article_id=article_id, entity_name=entity, article=article)


def unique_articles(paths: Iterable[Path]) -> tuple[list[BlindArticle], dict[str, int]]:
    by_id: dict[str, BlindArticle] = {}
    errors = 0
    duplicates = 0
    for path in paths:
        try:
            article = parse_blind_article(path)
        except ValueError:
            errors += 1
            continue
        if article.article_id in by_id:
            duplicates += 1
        else:
            by_id[article.article_id] = article
    articles = sorted(by_id.values(), key=lambda item: item.article_id)
    return articles, {
        "unique_articles": len(articles),
        "duplicate_copies_removed": duplicates,
        "parse_errors": errors,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> None:
    files = sorted(path for path in args.data_dir.iterdir() if path.is_file())
    articles, report = unique_articles(files)
    write_jsonl(args.output_dir / "blind_articles.jsonl", (asdict(article) for article in articles))
    write_jsonl(
        args.output_dir / "annotations.jsonl",
        (
            {
                "article_id": article.article_id,
                "label": None,
                "label_name": None,
                "confidence": None,
                "rationale": None,
                "evidence": [],
                "annotator": None,
                "status": "unreviewed",
            }
            for article in articles
        ),
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **report,
                "data_dir": str(args.data_dir.resolve()),
                "source_labels_exposed": False,
                "labels": {"1": "good_guy", "2": "bad_guy"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a blind corpus for independent adverse-media annotation")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/independent-annotations"))
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()