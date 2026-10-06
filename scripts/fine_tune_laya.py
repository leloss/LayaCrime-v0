from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

LABELS = {"hit": 0, "false positive": 1}
LABEL_NAMES = ["negative", "positive"]
HUMAN_LABEL_SOURCE = "original_human_disposition"
EXTERNAL_LABEL_SOURCE = "external_labels"
SUPPORTED_LABEL_SOURCES = {HUMAN_LABEL_SOURCE, EXTERNAL_LABEL_SOURCE}
DATASET_SCHEMA_VERSION = 3
FINE_TUNING_MAX_LEN = 1024
FINE_TUNING_HEAD_MAX_LEN = 256
FINE_TUNING_MAX_TOKENS_PER_BATCH = 4096
DEFAULT_PROMPT = {
    "question": (
        "How does this article portray {entity_name} regarding criminal behavior or intent? "
        "Judge only the named entity, not other people or organizations."
    ),
    "criteria": [
        {
            "decision": "negative",
            "text": (
                "negative: the article credibly associates the entity with alleged, investigated, "
                "charged, convicted, sanctioned, or admitted criminal behavior or intent"
            ),
        },
        {
            "decision": "positive",
            "text": (
                "positive: the article does not associate the entity with criminal behavior or intent, "
                "or identifies the entity only as a victim, witness, investigator, or unrelated party"
            ),
        },
    ],
}


@dataclass(frozen=True)
class Record:
    entity: str
    article: str
    label: int
    source: str
    identifier: str | None = None
    annotator: str | None = None


def default_data_dir() -> Path:
    return Path(os.getenv("LAYA_TRAINING_DATA_DIR", "data/source"))


def default_model_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    explicit = os.getenv("LAYA_FINE_TUNING_MODEL_DIR") or os.getenv("LAYA_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()
    bundle = Path(
        os.getenv(
            "LAYA_MODEL_BUNDLE",
            project_root / "models" / "models--convaiinnovations--laya",
        )
    ).expanduser()
    revision_file = bundle / "refs" / "main"
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="utf-8").strip()
        return bundle / "snapshots" / revision
    return bundle


def parse_record(path: Path) -> Record:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    headers: dict[str, str] = {}
    content_lines: list[str] = []
    in_content = False
    for line in text.splitlines():
        if not in_content and line.startswith("###content:"):
            content_lines.append(line.partition(":")[2].lstrip())
            in_content = True
        elif in_content:
            content_lines.append(line)
        elif line.startswith("###") and ":" in line:
            key, _, value = line[3:].partition(":")
            headers[key.strip().lower()] = value.strip()

    entity = headers.get("entityname", "").strip()
    disposition = (
        headers.get("disspositionreason")
        or headers.get("dispositionreason")
        or ""
    ).strip()
    article = "\n".join(content_lines).strip()
    if not entity:
        raise ValueError("missing entityName")
    if not article:
        raise ValueError("missing content")
    label = LABELS.get(disposition.casefold())
    if label is None:
        raise ValueError(f"unsupported disposition {disposition!r}")
    return Record(entity=entity, article=article, label=label, source=path.name)


def _duplicate_key(record: Record) -> str:
    normalized = "\n".join(
        (record.entity.casefold().strip(), " ".join(record.article.split()).casefold())
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def article_id(record: Record) -> str:
    return record.identifier or _duplicate_key(record)[:20]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_external_records(
    corpus_path: Path, labels_path: Path
) -> tuple[list[Record], dict[str, Any]]:
    articles = read_jsonl(corpus_path)
    labels = read_jsonl(labels_path)
    articles_by_id: dict[str, dict[str, Any]] = {}
    for row in articles:
        identifier = str(row.get("article_id") or "")
        entity = str(row.get("entity_name") or "").strip()
        article = str(row.get("article") or "").strip()
        if not identifier or not entity or not article:
            raise ValueError("corpus rows require article_id, entity_name, and article")
        if identifier in articles_by_id:
            raise ValueError(f"duplicate corpus article_id: {identifier}")
        articles_by_id[identifier] = row

    labels_by_id: dict[str, dict[str, Any]] = {}
    annotators: Counter[str] = Counter()
    for row in labels:
        identifier = str(row.get("article_id") or "")
        if identifier not in articles_by_id:
            raise ValueError(f"label article_id is absent from corpus: {identifier}")
        if identifier in labels_by_id:
            raise ValueError(f"duplicate label article_id: {identifier}")
        label = row.get("label")
        if type(label) is not int or label not in {1, 2}:
            raise ValueError(f"invalid label for article_id: {identifier}")
        annotator = str(row.get("annotator") or "unknown")
        annotators[annotator] += 1
        labels_by_id[identifier] = row

    records = []
    for identifier, article_row in articles_by_id.items():
        label_row = labels_by_id.get(identifier)
        if label_row is None:
            continue
        records.append(
            Record(
                entity=str(article_row["entity_name"]).strip(),
                article=str(article_row["article"]).strip(),
                label=0 if label_row["label"] == 2 else 1,
                source=f"{labels_path.name}:{identifier}",
                identifier=identifier,
                annotator=str(label_row.get("annotator") or "unknown"),
            )
        )

    missing_ids = sorted(set(articles_by_id) - set(labels_by_id))
    return records, {
        "corpus_rows": len(articles),
        "annotation_rows": len(labels),
        "records_retained": len(records),
        "missing_annotations": len(missing_ids),
        "missing_article_ids_sample": missing_ids[:100],
        "labels": dict(Counter(LABEL_NAMES[record.label] for record in records)),
        "annotators": dict(sorted(annotators.items())),
        "unique_entities": len({record.entity.casefold() for record in records}),
    }


def load_records(data_dir: Path) -> tuple[list[Record], dict[str, Any]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            record = parse_record(path)
            groups[_duplicate_key(record)].append(record)
        except ValueError as exc:
            errors.append({"file": path.name, "error": str(exc)})

    records: list[Record] = []
    duplicate_copies_removed = 0
    conflicting_groups = 0
    for copies in groups.values():
        labels = {record.label for record in copies}
        if len(labels) > 1:
            conflicting_groups += 1
            duplicate_copies_removed += len(copies)
            continue
        records.append(copies[0])
        duplicate_copies_removed += len(copies) - 1

    report = {
        "files_seen": sum(len(copies) for copies in groups.values()) + len(errors),
        "records_retained": len(records),
        "duplicate_copies_removed": duplicate_copies_removed,
        "conflicting_duplicate_groups_removed": conflicting_groups,
        "parse_errors": errors,
        "labels": dict(Counter(LABEL_NAMES[record.label] for record in records)),
        "unique_entities": len({record.entity.casefold() for record in records}),
    }
    return records, report


def dataset_sha256(records: Iterable[Record]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=_duplicate_key):
        canonical = {
            "article": " ".join(record.article.split()),
            "entity": record.entity.strip(),
            "label": record.label,
        }
        digest.update(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def record_components(records: Iterable[Record]) -> list[list[Record]]:
    records = list(records)
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    entity_owner: dict[str, int] = {}
    article_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        entity = record.entity.casefold().strip()
        article = hashlib.sha256(" ".join(record.article.split()).casefold().encode("utf-8")).hexdigest()
        if entity in entity_owner:
            union(index, entity_owner[entity])
        else:
            entity_owner[entity] = index
        if article in article_owner:
            union(index, article_owner[article])
        else:
            article_owner[article] = index

    components: dict[int, list[Record]] = defaultdict(list)
    for index, record in enumerate(records):
        components[find(index)].append(record)
    return list(components.values())


def partition_records(
    records: Iterable[Record],
    strategy: str = "holdout_80_20",
    seed: int = 20260923,
    fold_count: int = 10,
    fold_index: int = 0,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    components = record_components(records)
    splits: dict[str, list[Record]] = {"train": [], "calibration": []}
    holdout_percentages = {
        "holdout_50_50": 50,
        "holdout_60_40": 60,
        "holdout_70_30": 70,
        "holdout_80_20": 80,
        "entity_article_holdout": 80,
    }
    if strategy in holdout_percentages:
        train_percentage = holdout_percentages[strategy]
        for component in components:
            key = min(_duplicate_key(record) for record in component)
            digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % 100
            split = "train" if bucket < train_percentage else "calibration"
            splits[split].extend(component)
        return splits, {
            "strategy": f"holdout_{train_percentage}_{100 - train_percentage}",
            "label": (
                "Entity/article-disjoint train/calibration holdout "
                f"({train_percentage}/{100 - train_percentage})"
            ),
            "train_percentage": train_percentage,
            "calibration_percentage": 100 - train_percentage,
            "component_count": len(components),
        }

    if strategy == "group_k_fold":
        if not 2 <= fold_count <= len(components):
            raise ValueError(
                f"fold_count must be between 2 and the {len(components)} available groups"
            )
        if not 0 <= fold_index < fold_count:
            raise ValueError(f"fold_index must be between 0 and {fold_count - 1}")
        for component in components:
            key = min(_duplicate_key(record) for record in component)
            digest = hashlib.sha256(f"{seed}:fold:{key}".encode("utf-8")).digest()
            component_fold = int.from_bytes(digest[:4], "big") % fold_count
            split = "calibration" if component_fold == fold_index else "train"
            splits[split].extend(component)
        return splits, {
            "strategy": strategy,
            "label": "Grouped k-fold train/calibration",
            "component_count": len(components),
            "fold_count": fold_count,
            "calibration_fold": fold_index,
        }

    if strategy == "leave_one_group_out":
        if len(components) < 2:
            raise ValueError("leave-one-group-out requires at least 2 groups")
        if not 0 <= fold_index < len(components):
            raise ValueError(f"fold_index must be between 0 and {len(components) - 1}")
        ordered = sorted(
            components,
            key=lambda component: hashlib.sha256(
                f"{seed}:leave-one-out:{min(_duplicate_key(record) for record in component)}".encode(
                    "utf-8"
                )
            ).digest(),
        )
        for index, component in enumerate(ordered):
            split = "calibration" if index == fold_index else "train"
            splits[split].extend(component)
        return splits, {
            "strategy": strategy,
            "label": "Leave-one-entity/article-group-out for calibration",
            "component_count": len(components),
            "calibration_group": fold_index,
        }

    raise ValueError(f"unknown partition strategy {strategy!r}")


def split_records(
    records: Iterable[Record], seed: int = 20260923
) -> dict[str, list[Record]]:
    splits, _ = partition_records(records, seed=seed)
    return splits


def training_option_order(identifier: str) -> list[int]:
    digest = hashlib.sha256(f"option-order:{identifier}".encode("utf-8")).digest()
    return [1, 0] if digest[0] & 1 else [0, 1]


def ordered_choice_target(label: int, option_order: list[int]) -> list[float]:
    semantic_target = [1.0, 0.0] if label == 0 else [0.0, 1.0]
    return [semantic_target[index] for index in option_order]


def balanced_class_weights(labels: Iterable[int]) -> list[float]:
    counts = Counter(labels)
    if set(counts) != {0, 1}:
        raise ValueError("training data must contain both negative and positive labels")
    total = sum(counts.values())
    return [total / (2 * counts[index]) for index in range(2)]


def validation_improved(current: float, best: float) -> bool:
    return current < best


def adaptive_epoch_limit(planned_epochs: int, max_epochs: int) -> int:
    if planned_epochs < 1:
        raise ValueError("epochs must be at least 1")
    if max_epochs == 0:
        return planned_epochs
    if max_epochs < planned_epochs:
        raise ValueError("max epochs must be zero or at least the planned epochs")
    return max_epochs


def should_stop_early(
    epochs_completed: int,
    epochs_without_improvement: int,
    minimum_epochs: int,
    patience: int,
) -> bool:
    return (
        patience > 0
        and epochs_completed >= minimum_epochs
        and epochs_without_improvement >= patience
    )


def extended_epoch_budget(
    history: list[dict[str, Any]],
    current_budget: int,
    max_epochs: int,
    extension_epochs: int,
    min_delta: float,
) -> int:
    if extension_epochs <= 0 or current_budget >= max_epochs or len(history) < 2:
        return current_budget
    current_loss = float(history[-1]["validation_loss"])
    previous_loss = float(history[-2]["validation_loss"])
    prior_best = min(float(row["validation_loss"]) for row in history[:-1])
    if (
        current_loss < previous_loss - min_delta
        and current_loss < prior_best - min_delta
    ):
        return min(max_epochs, current_budget + extension_epochs)
    return current_budget


def cosine_lr_multiplier(initial_lr: float, total_steps: int, step: int) -> float:
    if initial_lr <= 0:
        return 1.0
    floor = min(1.0, 1e-6 / initial_lr)
    progress = min(max(step, 0), max(1, total_steps)) / max(1, total_steps)
    return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0


def staged_lr_multiplier(
    initial_lr: float,
    total_steps: int,
    step: int,
    warmup_steps: int = 0,
    start_step: int = 0,
) -> float:
    if initial_lr <= 0:
        return 1.0
    if step < start_step:
        return 0.0
    active_step = step - start_step
    active_total = max(1, total_steps - start_step)
    active_warmup = min(max(0, warmup_steps), max(0, active_total - 1))
    if active_warmup and active_step < active_warmup:
        return active_step / active_warmup
    decay_step = active_step - active_warmup
    decay_total = max(1, active_total - active_warmup)
    return cosine_lr_multiplier(initial_lr, decay_total, decay_step)


def smooth_choice_targets(
    targets: torch.Tensor,
    mask: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    option_count = mask.sum(-1, keepdim=True).float()
    return targets * (1.0 - smoothing) + smoothing * mask / option_count


def count_correct_choices(
    logits: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    predictions = logits.masked_fill(~mask, -1e4).argmax(-1)
    return int((predictions == labels).sum().detach())


def training_prompt(prompt_file: Path | None) -> dict[str, Any]:
    if prompt_file is None:
        return json.loads(json.dumps(DEFAULT_PROMPT))
    source = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompt = source.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError(f"prompt metadata is missing from {prompt_file}")
    question = prompt.get("question")
    criteria = prompt.get("criteria")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("prompt question must be a non-empty string")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("prompt criteria must be a non-empty list")
    decisions = {item.get("decision") for item in criteria if isinstance(item, dict)}
    if decisions != {"negative", "positive"}:
        raise ValueError("prompt criteria must include negative and positive decisions")
    if any(
        not isinstance(item.get("text"), str) or not item["text"].strip()
        for item in criteria
        if isinstance(item, dict)
    ) or any(not isinstance(item, dict) for item in criteria):
        raise ValueError("prompt criteria must contain non-empty text")
    return {
        "question": question.strip(),
        "criteria": [
            {"decision": item["decision"], "text": item["text"].strip()}
            for item in criteria
        ],
    }


def adverse_question(entity: str, prompt: dict[str, Any] | None = None) -> dict[str, Any]:
    configured = prompt or DEFAULT_PROMPT
    grouped = {
        decision: " | ".join(
            item["text"]
            for item in configured["criteria"]
            if item["decision"] == decision
        )
        for decision in ("negative", "positive")
    }
    return {
        "t": "choice",
        "ins": configured["question"].replace("{entity_name}", repr(entity)),
        "crit": {
            "A": grouped["negative"],
            "B": grouped["positive"],
        },
    }


def build_training_sequence(
    tokenizer,
    state: dict[str, str],
    question: dict[str, Any],
    max_len: int,
    head_max_len: int,
    option_order: list[int],
) -> tuple[list[int], list[int]]:
    mask_token = tokenizer.mask_token
    options = [
        key if value in (None, "") else f"{key}: {value}"
        for key, value in question["crit"].items()
    ]
    head_ids = tokenizer(
        f"{question['t']} question: {str(question['ins']).replace(mask_token, ' ')}",
        add_special_tokens=False,
    )["input_ids"]
    option_ids = [
        [tokenizer.mask_token_id]
        + tokenizer(
            " " + options[index].replace(mask_token, " "),
            add_special_tokens=False,
        )["input_ids"][:48]
        for index in option_order
    ]
    option_budget = head_max_len - sum(len(option) for option in option_ids)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [option[:per_option] for option in option_ids]
        option_budget = head_max_len - sum(len(option) for option in option_ids)
    head_ids = head_ids[: max(8, option_budget)]
    ids = [tokenizer.cls_token_id, *head_ids, tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = tokenizer(
        json.dumps(state, ensure_ascii=False).replace(mask_token, " "),
        add_special_tokens=False,
        truncation=True,
        max_length=room,
    )["input_ids"] if room else []
    return [*ids, *state_ids, tokenizer.sep_token_id], markers


def fine_tuning_config(config: dict[str, Any]) -> dict[str, Any]:
    configured = dict(config)
    configured.update(
        {
            "gradient_checkpointing": True,
            "max_tokens_per_batch": FINE_TUNING_MAX_TOKENS_PER_BATCH,
            "max_len": FINE_TUNING_MAX_LEN,
            "head_max_len": FINE_TUNING_HEAD_MAX_LEN,
        }
    )
    return configured


def validate_prepared_config(prepared_dir: Path, config: dict[str, Any]) -> None:
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            "prepared data has an unsupported or missing dataset schema; "
            "rerun the prepare command before training"
        )
    if manifest.get("label_source") not in SUPPORTED_LABEL_SOURCES:
        raise RuntimeError(
            "prepared data has an unsupported label_source; rerun the prepare command"
        )
    fingerprint = manifest.get("dataset_sha256", "")
    if len(fingerprint) != 64:
        raise RuntimeError(
            "prepared data has no valid dataset fingerprint; "
            "rerun the prepare command before training"
        )
    prepared_config = manifest.get("sequence_config")
    expected = {
        "max_len": config["max_len"],
        "head_max_len": config["head_max_len"],
    }
    if prepared_config != expected:
        raise RuntimeError(
            f"prepared data uses sequence_config={prepared_config!r}, expected {expected!r}; "
            "rerun the prepare command before training"
        )


def prepare(args: argparse.Namespace) -> None:
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES
    from transformers import AutoTokenizer

    data_dir = args.data_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _fix_tokenizer_config(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    config = fine_tuning_config(
        json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    )

    if args.label_source == EXTERNAL_LABEL_SOURCE:
        if args.corpus is None or args.labels is None:
            raise ValueError("external_labels preparation requires --corpus and --labels")
        records, audit = load_external_records(args.corpus.resolve(), args.labels.resolve())
    else:
        records, audit = load_records(data_dir)
    splits, partition = partition_records(
        records,
        strategy=args.partition_strategy,
        seed=args.seed,
        fold_count=args.fold_count,
        fold_index=args.fold_index,
    )
    empty_splits = [name for name, values in splits.items() if not values]
    if empty_splits:
        raise ValueError(
            f"partition produced empty splits: {', '.join(empty_splits)}; "
            "choose another seed, fold count, or strategy"
        )
    prompt = training_prompt(args.prompt_file)
    manifest: dict[str, Any] = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "label_source": args.label_source,
        "dataset_sha256": dataset_sha256(records),
        "model_dir": str(model_dir),
        "seed": args.seed,
        "partition": partition,
        "sequence_config": {
            "max_len": config["max_len"],
            "head_max_len": config["head_max_len"],
        },
        "audit": audit,
        "prompt": prompt,
        "splits": {},
    }
    if args.label_source == EXTERNAL_LABEL_SOURCE:
        manifest["corpus"] = str(args.corpus.resolve())
        manifest["labels"] = str(args.labels.resolve())
    else:
        manifest["data_dir"] = str(data_dir)
    split_assignments = []

    for split_name, split_records_list in splits.items():
        items = []
        for record in split_records_list:
            question = adverse_question(record.entity, prompt)
            state = {"article": record.article, "entity_name": record.entity}
            option_order = (
                training_option_order(article_id(record))
                if split_name == "train"
                else [0, 1]
            )
            ids, markers = build_training_sequence(
                tokenizer,
                state,
                question,
                config["max_len"],
                config["head_max_len"],
                option_order,
            )
            if len(markers) != 2:
                raise ValueError(f"question markers were truncated for {record.source}")
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES["choice"],
                    "target": ordered_choice_target(record.label, option_order),
                    "label": record.label,
                    "option_order": option_order,
                    "article_id": article_id(record),
                    "entity": record.entity,
                    "source": record.source,
                }
            )
            split_assignments.append(
                {
                    "article_id": article_id(record),
                    "split": split_name,
                    "label": record.label,
                    "label_name": LABEL_NAMES[record.label],
                }
            )
        torch.save(items, output_dir / f"{split_name}.pt")
        manifest["splits"][split_name] = {
            "records": len(items),
            "entities": len({item["entity"].casefold() for item in items}),
            "labels": dict(Counter(LABEL_NAMES[item["label"]] for item in items)),
        }

    with (output_dir / "split-assignments.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for assignment in sorted(split_assignments, key=lambda item: item["article_id"]):
            output.write(json.dumps(assignment, ensure_ascii=False) + "\n")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    from laya.common import collate_items

    batch = collate_items([[item] for item in items], pad_id)
    if batch is None:
        raise ValueError("cannot collate an empty batch")
    return batch


def no_init_weights_context():
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def cuda_supports_native_bfloat16(cuda, device_index: int) -> bool:
    major, _ = cuda.get_device_capability(device_index)
    return major >= 8 and cuda.is_bf16_supported()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_trainable_model(model_dir: Path, config: dict[str, Any], device: torch.device):
    from laya.common import build_model
    from safetensors.torch import load_file

    with no_init_weights_context():
        model = build_model(config, encoder_dir=str(model_dir / "encoder"), pretrained=False)
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    return model.to(device)


def evaluate_model(
    model,
    items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    batch_size: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    brier = 0.0
    nll = 0.0
    confusion = [[0, 0], [0, 0]]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start : start + batch_size], pad_id)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            probabilities = torch.softmax(logits[:, :2].float() / temperature, -1).cpu()
            labels = batch["label"]
            predictions = probabilities.argmax(-1)
            correct += int((predictions == labels).sum())
            total += len(labels)
            target = torch.nn.functional.one_hot(labels, 2).float()
            brier += float(((probabilities - target) ** 2).sum(-1).sum())
            nll += float(
                -torch.log(probabilities[torch.arange(len(labels)), labels].clamp_min(1e-9)).sum()
            )
            for gold, prediction in zip(labels.tolist(), predictions.tolist()):
                confusion[gold][prediction] += 1
    return {
        "records": total,
        "accuracy": correct / max(1, total),
        "balanced_accuracy": sum(
            confusion[index][index] / max(1, sum(confusion[index])) for index in range(2)
        )
        / 2,
        "brier_score": brier / max(1, total),
        "negative_log_likelihood": nll / max(1, total),
        "confusion_matrix": {
            "rows": LABEL_NAMES,
            "columns": LABEL_NAMES,
            "values": confusion,
        },
    }


def fit_temperature(model, items, pad_id, device, batch_size) -> float:
    model.eval()
    logits_parts = []
    labels_parts = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start : start + batch_size], pad_id)
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            logits_parts.append(logits[:, :2].float().cpu())
            labels_parts.append(batch["label"])
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.5, 5.0).item())


def save_checkpoint(model, tokenizer, config, output_dir: Path, metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(state, output_dir / "model.safetensors")
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def upload_checkpoint(output_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        commit_message="Upload LayaCrime fine-tuned checkpoint",
        ignore_patterns=["checkpoint_best/**", "checkpoint_latest/**"],
    )


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Laya fine-tuning requires an NVIDIA CUDA GPU. This machine has no CUDA device; "
            "run this command on an approved GPU host with the project, prepared data, and local model bundle."
        )

    import torch.distributed as dist
    from laya.common import proper_reward
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    max_epochs = adaptive_epoch_limit(args.epochs, args.max_epochs)
    if not 1 <= args.minimum_epochs <= max_epochs:
        raise ValueError("minimum epochs must be between 1 and max epochs")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early stopping minimum delta cannot be negative")
    if args.extension_epochs < 0:
        raise ValueError("extension epochs cannot be negative")

    model_dir = args.model_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = fine_tuning_config(
        json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    )
    validate_prepared_config(prepared_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = load_trainable_model(model_dir, config, device)
    train_model = model
    if distributed:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    train_items = torch.load(prepared_dir / "train.pt", weights_only=False)
    calibration_items = torch.load(prepared_dir / "calibration.pt", weights_only=False)
    class_weights = torch.tensor(
        balanced_class_weights(item["label"] for item in train_items),
        device=device,
    )
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.")]
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_rank = math.ceil(len(train_items) / max(1, world_size * args.batch_size))
    updates_per_epoch = math.ceil(steps_per_rank / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    encoder_start_update = min(
        total_updates - 1,
        updates_per_epoch * args.encoder_warmup_epochs,
    )
    head_warmup_updates = math.ceil(total_updates * args.lr_warmup_ratio)
    encoder_warmup_updates = math.ceil(
        max(1, total_updates - encoder_start_update) * args.lr_warmup_ratio
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: staged_lr_multiplier(
                args.encoder_lr,
                total_updates,
                step,
                encoder_warmup_updates,
                encoder_start_update,
            ),
            lambda step: staged_lr_multiplier(
                args.head_lr,
                total_updates,
                step,
                head_warmup_updates,
            ),
        ],
    )
    use_bfloat16 = cuda_supports_native_bfloat16(torch.cuda, local_rank)
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = make_grad_scaler(enabled=not use_bfloat16)
    if rank == 0:
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(local_rank),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(local_rank)
                    ),
                    "amp_dtype": "bfloat16" if use_bfloat16 else "float16",
                }
            ),
            flush=True,
        )
    history = []
    started = time.time()
    baseline_validation = None
    if rank == 0:
        baseline_validation = evaluate_model(
            model,
            calibration_items,
            tokenizer.pad_token_id,
            device,
            args.eval_batch_size,
        )
        best_validation_loss = baseline_validation["negative_log_likelihood"]
        baseline_config = dict(config)
        baseline_config["training"] = {
            "epoch": 0,
            "validation_loss": best_validation_loss,
            "validation_accuracy": baseline_validation["accuracy"],
            "validation_balanced_accuracy": baseline_validation["balanced_accuracy"],
        }
        save_checkpoint(
            model,
            tokenizer,
            baseline_config,
            output_dir / "checkpoint_best",
            {"history": history, "best_epoch": 0, "baseline_validation": baseline_validation},
        )
        print(
            "baseline_metrics "
            f"validation_loss={best_validation_loss:.8f} "
            f"validation_accuracy={baseline_validation['accuracy']:.8f}",
            flush=True,
        )
    else:
        best_validation_loss = math.inf
    best_epoch = 0
    epoch_budget = args.epochs
    early_stopping_best_loss = best_validation_loss
    epochs_without_improvement = 0
    stopping_reason = "max_epochs_reached"
    if distributed:
        dist.barrier()

    for epoch in range(max_epochs):
        encoder_trainable = epoch >= args.encoder_warmup_epochs
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_trainable)
        model.train()
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        per_rank = math.ceil(len(order) / world_size)
        order.extend(order[: per_rank * world_size - len(order)])
        rank_indices = order[rank::world_size]
        optimizer.zero_grad(set_to_none=True)
        epoch_nll_sum = 0.0
        epoch_optimization_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0
        batches = 0
        sigma_progress = min(1.0, epoch / max(1, args.epochs - 1))
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * sigma_progress

        for offset in range(0, len(rank_indices), args.batch_size):
            selected = rank_indices[offset : offset + args.batch_size]
            batch = collate([train_items[index] for index in selected], tokenizer.pad_token_id)
            final_batch = offset + args.batch_size >= len(rank_indices)
            accumulation_index = batches % args.gradient_accumulation
            synchronize = accumulation_index == args.gradient_accumulation - 1 or final_batch
            sync_context = nullcontext() if synchronize or not distributed else train_model.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits, action = train_model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["marker_pos"].to(device),
                        batch["marker_mask"].to(device),
                        batch["qtype"].to(device),
                    )
                logits = logits.float()
                mask = batch["marker_mask"].to(device)
                reward_targets = batch["target"].to(device)
                targets = smooth_choice_targets(
                    reward_targets, mask, args.label_smoothing
                )
                reinforcement_loss = logits.new_zeros(())
                if args.rl_weight:
                    option_count = mask.sum(-1, keepdim=True).float()
                    noise = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
                    noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
                    sampled_logits = logits.detach().unsqueeze(0) + noise
                    distributions = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), -1)
                    with torch.no_grad():
                        rewards = proper_reward(
                            distributions,
                            reward_targets.unsqueeze(0),
                            batch["qtype"].to(device),
                            mask,
                            w_sph=0.75,
                            w_rps=1.0,
                        )
                        advantages = rewards - rewards.mean(0, keepdim=True)
                        advantages = advantages / (advantages.std() + 1e-6)
                    log_probability = -(
                        ((sampled_logits - logits.unsqueeze(0)) ** 2 * mask).sum(-1)
                        / (2 * sigma**2)
                    )
                    reinforcement_loss = -(advantages * log_probability).mean()
                per_item_cross_entropy = -(
                    targets * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
                ).sum(-1)
                sample_weights = class_weights[batch["label"].to(device)]
                cross_entropy = (per_item_cross_entropy * sample_weights).mean()
                loss = (
                    args.rl_weight * reinforcement_loss
                    + cross_entropy
                    + 0.0 * action.sum()
                ) / args.gradient_accumulation
                scaler.scale(loss).backward()

            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_examples = len(selected)
            epoch_nll_sum += float(per_item_cross_entropy.sum().detach())
            epoch_optimization_loss_sum += (
                float(loss.detach()) * args.gradient_accumulation * batch_examples
            )
            epoch_correct += count_correct_choices(
                logits,
                mask,
                batch["label"].to(device),
            )
            epoch_examples += batch_examples
            batches += 1
            if rank == 0 and batches % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{epoch_budget} batch={batches} "
                    f"loss={epoch_nll_sum / max(1, epoch_examples):.4f} sigma={sigma:.3f}",
                    flush=True,
                )

        if distributed:
            epoch_totals = torch.tensor(
                [
                    epoch_nll_sum,
                    epoch_optimization_loss_sum,
                    epoch_correct,
                    epoch_examples,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)
            epoch_nll_sum = float(epoch_totals[0].item())
            epoch_optimization_loss_sum = float(epoch_totals[1].item())
            epoch_correct = int(epoch_totals[2].item())
            epoch_examples = int(epoch_totals[3].item())
            dist.barrier()
        if rank == 0:
            validation = evaluate_model(
                model,
                calibration_items,
                tokenizer.pad_token_id,
                device,
                args.eval_batch_size,
            )
            epoch_report = {
                "epoch": epoch + 1,
                "train_loss": epoch_nll_sum / max(1, epoch_examples),
                "train_accuracy": epoch_correct / max(1, epoch_examples),
                "validation_loss": validation["negative_log_likelihood"],
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "optimization_loss": epoch_optimization_loss_sum / max(1, epoch_examples),
                "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
                "encoder_trainable": encoder_trainable,
                "elapsed_seconds": round(time.time() - started, 1),
            }
            history.append(epoch_report)
            checkpoint_config = dict(config)
            checkpoint_config["training"] = epoch_report
            save_checkpoint(
                model,
                tokenizer,
                checkpoint_config,
                output_dir / "checkpoint_latest",
                {"history": history},
            )
            if validation_improved(
                epoch_report["validation_loss"],
                early_stopping_best_loss - args.early_stopping_min_delta,
            ):
                early_stopping_best_loss = epoch_report["validation_loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if validation_improved(
                epoch_report["validation_loss"],
                best_validation_loss,
            ):
                best_validation_loss = epoch_report["validation_loss"]
                best_epoch = epoch + 1
                save_checkpoint(
                    model,
                    tokenizer,
                    checkpoint_config,
                    output_dir / "checkpoint_best",
                    {"history": history, "best_epoch": best_epoch},
                )
            print(
                "epoch_metrics "
                f"epoch={epoch_report['epoch']} "
                f"train_loss={epoch_report['train_loss']:.8f} "
                f"validation_loss={epoch_report['validation_loss']:.8f} "
                f"train_accuracy={epoch_report['train_accuracy']:.8f} "
                f"validation_accuracy={epoch_report['validation_accuracy']:.8f}",
                flush=True,
            )
            print(json.dumps(epoch_report), flush=True)
            epochs_completed = epoch + 1
            stop_training = should_stop_early(
                epochs_completed,
                epochs_without_improvement,
                args.minimum_epochs,
                args.early_stopping_patience,
            )
            if stop_training:
                stopping_reason = "early_stopping"
                print(
                    "training_control "
                    f"event=early_stop epoch={epochs_completed} "
                    f"best_epoch={best_epoch} patience={args.early_stopping_patience}",
                    flush=True,
                )
            elif epochs_completed >= epoch_budget:
                extended_budget = extended_epoch_budget(
                    history,
                    epoch_budget,
                    max_epochs,
                    args.extension_epochs,
                    args.early_stopping_min_delta,
                )
                if extended_budget > epoch_budget:
                    print(
                        "training_control "
                        f"event=extend from_epoch={epoch_budget} "
                        f"to_epoch={extended_budget}",
                        flush=True,
                    )
                    epoch_budget = extended_budget
                else:
                    stop_training = True
                    stopping_reason = (
                        "max_epochs_reached"
                        if epochs_completed >= max_epochs
                        else "planned_epochs_complete"
                    )
        else:
            stop_training = False
        if distributed:
            stop_control = torch.tensor(
                [int(stop_training)], dtype=torch.int32, device=device
            )
            dist.broadcast(stop_control, src=0)
            stop_training = bool(stop_control.item())
            dist.barrier()
        if stop_training:
            break

    if rank == 0:
        best_checkpoint = output_dir / "checkpoint_best" / "model.safetensors"
        if not best_checkpoint.is_file():
            raise RuntimeError("training produced no best validation checkpoint")
        model.load_state_dict(load_file(best_checkpoint), strict=True)
        temperature = fit_temperature(
            model, calibration_items, tokenizer.pad_token_id, device, args.eval_batch_size
        )
        final_config = dict(config)
        temperatures = list(final_config.get("temperature", [1.0, 1.0, 1.0]))
        temperatures[0] = temperature
        final_config["temperature"] = temperatures
        final_config.pop("temperature_by_options", None)
        final_config["fine_tuned"] = True
        final_config["model_name"] = "laya-adverse-media"
        prepared_manifest = json.loads(
            (prepared_dir / "manifest.json").read_text(encoding="utf-8")
        )
        final_config["prompt"] = prepared_manifest.get("prompt", DEFAULT_PROMPT)
        final_config["training"] = {
            "task": "adverse-media",
            "strategy": args.strategy,
            "epochs_requested": args.epochs,
            "max_epochs": max_epochs,
            "epochs_completed": len(history),
            "stopping_reason": stopping_reason,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "checkpoint_selection": "lowest_validation_loss_across_all_epochs",
            "world_size": world_size,
            "seed": args.seed,
            "class_balanced": True,
            "rl_weight": args.rl_weight,
            "encoder_warmup_epochs": args.encoder_warmup_epochs,
            "lr_warmup_ratio": args.lr_warmup_ratio,
            "label_smoothing": args.label_smoothing,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
            "minimum_epochs": args.minimum_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "extension_epochs": args.extension_epochs,
            "baseline_validation": baseline_validation,
            "partition": prepared_manifest.get("partition"),
        }
        report = {
            "strategy": args.strategy,
            "partition": prepared_manifest.get("partition"),
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "checkpoint_selection": "lowest_validation_loss_across_all_epochs",
            "training_control": {
                "planned_epochs": args.epochs,
                "max_epochs": max_epochs,
                "epochs_completed": len(history),
                "stopping_reason": stopping_reason,
                "minimum_epochs": args.minimum_epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "extension_epochs": args.extension_epochs,
            },
            "baseline_validation": baseline_validation,
            "optimization": {
                "encoder_lr": args.encoder_lr,
                "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "encoder_warmup_epochs": args.encoder_warmup_epochs,
                "lr_warmup_ratio": args.lr_warmup_ratio,
                "label_smoothing": args.label_smoothing,
                "batch_size_per_gpu": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
            },
            "temperature_choice": temperature,
        }
        save_checkpoint(model, tokenizer, final_config, output_dir, report)
        shutil.copy2(prepared_dir / "manifest.json", output_dir / "data_manifest.json")
        if args.hub_repo_id:
            upload_checkpoint(output_dir, args.hub_repo_id, args.hub_private)
        print(json.dumps(report, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def evaluate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    validate_prepared_config(args.prepared_dir.resolve(), config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = load_trainable_model(model_dir, config, device)
    items = torch.load(args.prepared_dir / f"{args.split}.pt", weights_only=False)
    temperature = float(config.get("temperature", [1.0])[0])
    report = evaluate_model(model, items, tokenizer.pad_token_id, device, args.batch_size, temperature)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and fine-tune Laya for adverse-media decisions")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="audit, split, and tokenize local training data")
    prepare_parser.add_argument(
        "--label-source",
        choices=(EXTERNAL_LABEL_SOURCE, HUMAN_LABEL_SOURCE),
        default=HUMAN_LABEL_SOURCE,
    )
    prepare_parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    prepare_parser.add_argument("--corpus", type=Path)
    prepare_parser.add_argument("--labels", type=Path)
    prepare_parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adverse-media-data"))
    prepare_parser.add_argument("--seed", type=int, default=20260923)
    prepare_parser.add_argument(
        "--partition-strategy",
        choices=(
            "entity_article_holdout",
            "holdout_50_50",
            "holdout_60_40",
            "holdout_70_30",
            "holdout_80_20",
            "group_k_fold",
            "leave_one_group_out",
        ),
        default="holdout_80_20",
    )
    prepare_parser.add_argument("--fold-count", type=int, default=10)
    prepare_parser.add_argument("--fold-index", type=int, default=0)
    prepare_parser.add_argument("--prompt-file", type=Path)
    prepare_parser.set_defaults(func=prepare)

    train_parser = commands.add_parser("train", help="fine-tune on prepared data (launch with torchrun for DDP)")
    train_parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/adverse-media-data"))
    train_parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    train_parser.add_argument("--output-dir", type=Path, default=Path("models/laya-adverse-media"))
    train_parser.add_argument(
        "--strategy",
        choices=(
            "preserve_entity_matching",
            "balanced",
            "maximum_task_adaptation",
            "relational_low_drift",
            "relational_adaptation",
            "relational_extended",
            "relational_rlcd",
            "public_natural_staged",
            "custom",
        ),
        default="custom",
    )
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Hard epoch ceiling; zero uses --epochs without extension",
    )
    train_parser.add_argument("--minimum-epochs", type=int, default=1)
    train_parser.add_argument("--early-stopping-patience", type=int, default=0)
    train_parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train_parser.add_argument("--extension-epochs", type=int, default=0)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8)
    train_parser.add_argument("--group-size", type=int, default=4)
    train_parser.add_argument("--rl-weight", type=float, default=0.0)
    train_parser.add_argument("--encoder-lr", type=float, default=1e-5)
    train_parser.add_argument("--head-lr", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--sigma-start", type=float, default=0.4)
    train_parser.add_argument("--sigma-end", type=float, default=0.1)
    train_parser.add_argument("--encoder-warmup-epochs", type=int, default=0)
    train_parser.add_argument("--lr-warmup-ratio", type=float, default=0.0)
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=20260923)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--hub-repo-id")
    train_parser.add_argument("--hub-private", action="store_true")
    train_parser.set_defaults(func=train)

    evaluate_parser = commands.add_parser(
        "evaluate", help="evaluate an exported checkpoint on a prepared split"
    )
    evaluate_parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/adverse-media-data"))
    evaluate_parser.add_argument("--model-dir", type=Path, default=Path("models/laya-adverse-media"))
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument(
        "--split", choices=("train", "calibration"), default="calibration"
    )
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()