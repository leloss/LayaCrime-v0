from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
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
AZURE_LABEL_SOURCE = "consolidated_azure_annotations"
SUPPORTED_LABEL_SOURCES = {HUMAN_LABEL_SOURCE, AZURE_LABEL_SOURCE}
DATASET_SCHEMA_VERSION = 1
FINE_TUNING_MAX_LEN = 1024
FINE_TUNING_HEAD_MAX_LEN = 256
FINE_TUNING_MAX_TOKENS_PER_BATCH = 4096


@dataclass(frozen=True)
class Record:
    entity: str
    article: str
    label: int
    source: str
    identifier: str | None = None
    annotator: str | None = None


def default_data_dir() -> Path:
    return Path(os.getenv("LAYA_SOURCE_DATA_DIR", "data/source"))


def default_model_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    bundle = project_root / "models" / "models--convaiinnovations--laya"
    revision = (bundle / "refs" / "main").read_text(encoding="utf-8").strip()
    return bundle / "snapshots" / revision


def default_corpus_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "independent-annotations"
        / "blind_articles.jsonl"
    )


def default_annotations_path() -> Path:
    return default_corpus_path().parent / "batches" / "azure.annotations.jsonl"


def default_prepared_dir() -> Path:
    return Path("artifacts/adverse-media-data-azure")


def default_output_model_dir() -> Path:
    return Path("models/laya-adverse-media-azure")


def manifest_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


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


def load_azure_records(
    corpus_path: Path, annotations_path: Path
) -> tuple[list[Record], dict[str, Any]]:
    articles = read_jsonl(corpus_path)
    annotations = read_jsonl(annotations_path)
    articles_by_id: dict[str, dict[str, Any]] = {}
    for row in articles:
        identifier = str(row.get("article_id") or "")
        entity = str(row.get("entity_name") or "").strip()
        article = str(row.get("article") or "").strip()
        if not identifier or not entity or not article:
            raise ValueError("corpus rows require article_id, entity_name, and article")
        expected = _duplicate_key(
            Record(entity=entity, article=article, label=0, source=corpus_path.name)
        )[:20]
        if identifier != expected:
            raise ValueError(f"corpus article_id does not match content: {identifier}")
        if identifier in articles_by_id:
            raise ValueError(f"duplicate corpus article_id: {identifier}")
        articles_by_id[identifier] = row

    annotations_by_id: dict[str, dict[str, Any]] = {}
    annotators: Counter[str] = Counter()
    for row in annotations:
        identifier = str(row.get("article_id") or "")
        if identifier not in articles_by_id:
            raise ValueError(f"annotation article_id is absent from corpus: {identifier}")
        if identifier in annotations_by_id:
            raise ValueError(f"duplicate annotation article_id: {identifier}")
        label = row.get("label")
        if type(label) is not int or label not in {1, 2}:
            raise ValueError(f"annotation has invalid label for article_id: {identifier}")
        annotator = str(row.get("annotator") or "unknown")
        annotators[annotator] += 1
        annotations_by_id[identifier] = row

    records = []
    for identifier, article_row in articles_by_id.items():
        annotation = annotations_by_id.get(identifier)
        if annotation is None:
            continue
        azure_label = int(annotation["label"])
        records.append(
            Record(
                entity=str(article_row["entity_name"]).strip(),
                article=str(article_row["article"]).strip(),
                label=0 if azure_label == 2 else 1,
                source=f"{annotations_path.name}:{identifier}",
                identifier=identifier,
                annotator=str(annotation.get("annotator") or "unknown"),
            )
        )

    missing_ids = sorted(set(articles_by_id) - set(annotations_by_id))
    report = {
        "corpus_rows": len(articles),
        "annotation_rows": len(annotations),
        "records_retained": len(records),
        "missing_annotations": len(missing_ids),
        "missing_article_ids": missing_ids,
        "labels": dict(Counter(LABEL_NAMES[record.label] for record in records)),
        "annotators": dict(sorted(annotators.items())),
        "unique_entities": len({record.entity.casefold() for record in records}),
    }
    return records, report


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


def split_records(
    records: Iterable[Record], seed: int = 20260923
) -> dict[str, list[Record]]:
    splits: dict[str, list[Record]] = {"train": [], "calibration": [], "test": []}
    entity_splits: dict[str, str] = {}
    for record in records:
        entity = record.entity.casefold().strip()
        if entity not in entity_splits:
            digest = hashlib.sha256(f"{seed}:{entity}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % 100
            entity_splits[entity] = (
                "train" if bucket < 80 else "calibration" if bucket < 90 else "test"
            )
        splits[entity_splits[entity]].append(record)
    return splits


def adverse_question(entity: str) -> dict[str, Any]:
    return {
        "t": "choice",
        "ins": (
            f"How does this article portray {entity!r} regarding criminal behavior? "
            "Judge only the named entity, not other people or organizations."
        ),
        "crit": {
            "A": (
                "negative: the article credibly associates the entity with alleged, "
                "investigated, charged, convicted, sanctioned, or admitted criminal behavior"
            ),
            "B": (
                "positive: the article does not associate the entity with criminal behavior, "
                "or identifies the entity only as a victim, witness, investigator, or unrelated party"
            ),
        },
    }


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
            "prepared data has an unsupported label_source; rerun the prepare command "
            "with original human or consolidated Azure annotations"
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
    from transformers import AutoTokenizer

    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_sequence

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _fix_tokenizer_config(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    config = fine_tuning_config(
        json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    )

    if args.label_source == "azure":
        corpus_path = args.corpus.resolve()
        annotations_path = args.annotations.resolve()
        records, audit = load_azure_records(corpus_path, annotations_path)
        label_source = AZURE_LABEL_SOURCE
        inputs = {
            "corpus_path": manifest_path(corpus_path),
            "annotations_path": manifest_path(annotations_path),
        }
    else:
        data_dir = args.data_dir.resolve()
        records, audit = load_records(data_dir)
        label_source = HUMAN_LABEL_SOURCE
        inputs = {"data_dir": manifest_path(data_dir)}
    splits = split_records(records, args.seed)
    manifest: dict[str, Any] = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "label_source": label_source,
        "dataset_sha256": dataset_sha256(records),
        **inputs,
        "model_dir": manifest_path(model_dir),
        "seed": args.seed,
        "sequence_config": {
            "max_len": config["max_len"],
            "head_max_len": config["head_max_len"],
        },
        "audit": audit,
        "splits": {},
    }
    split_assignments = []

    for split_name, split_records_list in splits.items():
        items = []
        for record in split_records_list:
            question = adverse_question(record.entity)
            state = {"article": record.article, "entity_name": record.entity}
            ids, markers = build_sequence(
                tokenizer,
                state,
                question,
                config["max_len"],
                config["head_max_len"],
            )
            if len(markers) != 2:
                raise ValueError(f"question markers were truncated for {record.source}")
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES["choice"],
                    "target": [1.0, 0.0] if record.label == 0 else [0.0, 1.0],
                    "label": record.label,
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


def load_trainable_model(model_dir: Path, config: dict[str, Any], device: torch.device):
    from safetensors.torch import load_file
    from transformers.initialization import no_init_weights

    from laya.common import build_model

    with no_init_weights():
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


def validate_train_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "group_size",
        "log_every",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be greater than zero")
    if args.encoder_lr <= 0 or args.head_lr <= 0:
        raise ValueError("learning rates must be greater than zero")
    if args.sigma_start <= 0 or args.sigma_end <= 0:
        raise ValueError("sigma values must be greater than zero")


def cuda_report(world_size: int, rank: int, local_rank: int, args: argparse.Namespace) -> dict[str, Any]:
    visible_devices = torch.cuda.device_count()
    if local_rank >= visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_devices} CUDA device(s) are visible"
        )
    devices = []
    for index in range(visible_devices):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_gib": round(properties.total_memory / 1024**3, 1),
            }
        )
    return {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "visible_devices": devices,
        "mixed_precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_global_batch_size": (
            world_size * args.batch_size * args.gradient_accumulation
        ),
    }


def train(args: argparse.Namespace) -> None:
    validate_train_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Laya fine-tuning requires an NVIDIA CUDA GPU. This machine has no CUDA device; "
            "run this command on a CUDA GPU host with the project, prepared data, and local model bundle."
        )

    import torch.distributed as dist
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from laya.common import proper_reward

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    hardware = cuda_report(world_size, rank, local_rank, args)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    model_dir = args.model_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = fine_tuning_config(
        json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    )
    validate_prepared_config(prepared_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    resume_dir = args.resume_from.resolve() if args.resume_from else None
    if resume_dir and not (resume_dir / "trainer_state.pt").is_file():
        raise FileNotFoundError(f"resume checkpoint has no trainer_state.pt: {resume_dir}")
    model = load_trainable_model(resume_dir or model_dir, config, device)
    train_model = model
    if distributed:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    train_items = torch.load(prepared_dir / "train.pt", weights_only=False)
    calibration_items = torch.load(prepared_dir / "calibration.pt", weights_only=False)
    test_items = torch.load(prepared_dir / "test.pt", weights_only=False)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * args.epochs), eta_min=1e-6
    )
    use_bfloat16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bfloat16)
    history = []
    start_epoch = 0
    if resume_dir:
        trainer_state = torch.load(
            resume_dir / "trainer_state.pt", map_location=device, weights_only=False
        )
        prepared_manifest = json.loads(
            (prepared_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if trainer_state.get("dataset_sha256") != prepared_manifest.get("dataset_sha256"):
            raise RuntimeError("resume checkpoint was trained on a different prepared dataset")
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        scaler.load_state_dict(trainer_state["scaler"])
        history = list(trainer_state["history"])
        start_epoch = int(trainer_state["completed_epochs"])
        if start_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint already completed {start_epoch} epoch(s); "
                f"--epochs must be greater than {start_epoch}"
            )
    started = time.time()
    if rank == 0:
        print(json.dumps({"hardware": hardware, "resume_epoch": start_epoch}, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        per_rank = math.ceil(len(order) / world_size)
        order.extend(order[: per_rank * world_size - len(order)])
        rank_indices = order[rank::world_size]
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        batches = 0
        sigma_progress = epoch / max(1, args.epochs - 1)
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
                targets = batch["target"].to(device)
                option_count = mask.sum(-1, keepdim=True).float()
                noise = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
                noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
                sampled_logits = logits.detach().unsqueeze(0) + noise
                distributions = torch.softmax(sampled_logits.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    rewards = proper_reward(
                        distributions,
                        targets.unsqueeze(0),
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
                cross_entropy = -(
                    targets * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
                ).sum(-1).mean()
                loss = (reinforcement_loss + cross_entropy + 0.0 * action.sum()) / args.gradient_accumulation
                scaler.scale(loss).backward()

            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach()) * args.gradient_accumulation
            batches += 1
            if rank == 0 and batches % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} batch={batches} "
                    f"loss={epoch_loss / batches:.4f} sigma={sigma:.3f}",
                    flush=True,
                )

        if distributed:
            dist.barrier()
        if rank == 0:
            epoch_report = {
                "epoch": epoch + 1,
                "average_loss": epoch_loss / max(1, batches),
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
            prepared_manifest = json.loads(
                (prepared_dir / "manifest.json").read_text(encoding="utf-8")
            )
            torch.save(
                {
                    "completed_epochs": epoch + 1,
                    "dataset_sha256": prepared_manifest.get("dataset_sha256"),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                },
                output_dir / "checkpoint_latest" / "trainer_state.pt",
            )
            print(json.dumps(epoch_report), flush=True)
        if distributed:
            dist.barrier()

    if rank == 0:
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
        final_config["training"] = {
            "task": "adverse-media",
            "epochs": args.epochs,
            "world_size": world_size,
            "seed": args.seed,
        }
        metrics = evaluate_model(
            model,
            test_items,
            tokenizer.pad_token_id,
            device,
            args.eval_batch_size,
            temperature,
        )
        report = {"history": history, "temperature_choice": temperature, "test": metrics}
        save_checkpoint(model, tokenizer, final_config, output_dir, report)
        shutil.copy2(prepared_dir / "manifest.json", output_dir / "data_manifest.json")
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
    items = torch.load(args.prepared_dir / "test.pt", weights_only=False)
    temperature = float(config.get("temperature", [1.0])[0])
    report = evaluate_model(model, items, tokenizer.pad_token_id, device, args.batch_size, temperature)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and fine-tune Laya for adverse-media decisions")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="audit, split, and tokenize local training data")
    prepare_parser.add_argument("--label-source", choices=("azure", "human"), default="azure")
    prepare_parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    prepare_parser.add_argument("--corpus", type=Path, default=default_corpus_path())
    prepare_parser.add_argument("--annotations", type=Path, default=default_annotations_path())
    prepare_parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    prepare_parser.add_argument("--output-dir", type=Path, default=default_prepared_dir())
    prepare_parser.add_argument("--seed", type=int, default=20260923)
    prepare_parser.set_defaults(func=prepare)

    train_parser = commands.add_parser("train", help="fine-tune on prepared data (launch with torchrun for DDP)")
    train_parser.add_argument("--prepared-dir", type=Path, default=default_prepared_dir())
    train_parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    train_parser.add_argument("--output-dir", type=Path, default=default_output_model_dir())
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8)
    train_parser.add_argument("--group-size", type=int, default=4)
    train_parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    train_parser.add_argument("--head-lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--sigma-start", type=float, default=0.4)
    train_parser.add_argument("--sigma-end", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=20260923)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.set_defaults(func=train)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate an exported checkpoint on the held-out test split")
    evaluate_parser.add_argument("--prepared-dir", type=Path, default=default_prepared_dir())
    evaluate_parser.add_argument("--model-dir", type=Path, default=default_output_model_dir())
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()