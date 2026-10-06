from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr

from .benchmark import BenchmarkState, load_benchmark, load_gold_labels, read_jsonl
from .benchmark_models import (
    GenerativeBenchmarkRuntime,
    benchmark_model_options,
    model_installation_receipt_path,
    register_custom_model,
    registered_model_by_id,
    registered_model_catalog,
    remove_registered_model,
    update_registered_source,
)
from .datasets import (
    DatasetBundle,
    discover_dataset_bundles,
    register_annotation_set,
    remove_annotation_set,
    update_dataset_prompt,
    validate_prompt,
)
from .fine_tuning import FineTuningJob
from .model_installation import ModelInstallationJob


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Predictor(Protocol):
    @property
    def loaded(self) -> list[str]: ...

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Mapping[str, Any]: ...


class PromptCriterion(BaseModel):
    decision: Literal["negative", "positive"]
    text: str = Field(min_length=1, max_length=4000)


class AdverseMediaRequest(BaseModel):
    article: str = Field(min_length=1, max_length=100_000)
    entity_name: str = Field(min_length=1, max_length=500)
    routing_mode: Literal["auto", "english", "multilingual"] = "auto"
    model_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    criteria: list[PromptCriterion] | None = Field(default=None, min_length=2, max_length=20)


class AdverseMediaResponse(BaseModel):
    entity_name: str
    decision: Literal["negative", "positive"]
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    needs_review: bool
    routing: dict[str, Any] | None = None


class BenchmarkStartRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=100_000)
    routing_mode: str = Field(default="auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,79})$")
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    criteria: list[PromptCriterion] | None = Field(default=None, min_length=2, max_length=20)


class ModelActivationRequest(BaseModel):
    model_id: str = Field(pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,79})$")


class BenchmarkModelActivationRequest(ModelActivationRequest):
    endpoint: str | None = Field(default=None, max_length=2000)
    api_key: SecretStr | None = None
    deployment: str | None = Field(default=None, max_length=200)


class BenchmarkModelInstallRequest(ModelActivationRequest):
    huggingface_token: SecretStr | None = None
    repo_id: str | None = Field(default=None, min_length=3, max_length=500)
    huggingface_path: str | None = Field(default=None, min_length=5, max_length=1000)


class BenchmarkCustomModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    repo_id: str = Field(min_length=3, max_length=500)
    huggingface_path: str = Field(min_length=5, max_length=1000)


class BenchmarkSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class BenchmarkLoadRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,159}$")


class BenchmarkLabelSetRequest(BaseModel):
    label_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    dataset_id: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$"
    )


class BenchmarkDatasetRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")


class BenchmarkLabelSetUploadRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=25_000_000)


class FineTuningPrepareRequest(BaseModel):
    label_source: Literal["external"] = "external"
    data_dir: Path | None = None
    corpus: Path
    labels: Path
    model_dir: Path
    output_dir: Path
    seed: int = Field(ge=0)
    partition_strategy: Literal[
        "entity_article_holdout",
        "holdout_50_50",
        "holdout_60_40",
        "holdout_70_30",
        "holdout_80_20",
        "group_k_fold",
        "leave_one_group_out",
    ] = "holdout_80_20"
    fold_count: int = Field(default=5, ge=2, le=100)
    fold_index: int = Field(default=0, ge=0)
    dataset_id: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$"
    )


class FineTuningPromptRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    question: str = Field(min_length=1, max_length=4000)
    criteria: list[PromptCriterion] = Field(min_length=2, max_length=20)


class FineTuningTrainRequest(BaseModel):
    strategy: Literal[
        "preserve_entity_matching",
        "balanced",
        "maximum_task_adaptation",
        "relational_low_drift",
        "relational_adaptation",
        "relational_extended",
        "relational_rlcd",
        "public_natural_staged",
        "custom",
    ] = "balanced"
    partition_strategy: Literal[
        "entity_article_holdout",
        "holdout_50_50",
        "holdout_60_40",
        "holdout_70_30",
        "holdout_80_20",
        "group_k_fold",
        "leave_one_group_out",
    ] = "holdout_80_20"
    prepared_dir: Path
    model_dir: Path
    output_dir: Path
    gpu_count: int = Field(ge=1, le=16)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    eval_batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    group_size: int = Field(ge=1)
    rl_weight: float = Field(default=0.0, ge=0)
    encoder_lr: float = Field(ge=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    sigma_start: float = Field(ge=0)
    sigma_end: float = Field(ge=0)
    encoder_warmup_epochs: int = Field(default=0, ge=0)
    lr_warmup_ratio: float = Field(default=0.0, ge=0, le=0.5)
    label_smoothing: float = Field(default=0.0, ge=0, lt=0.5)
    seed: int = Field(ge=0)
    log_every: int = Field(ge=1)
    huggingface_repo_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$",
    )
    huggingface_token: SecretStr | None = None
    huggingface_private: bool = False


class FineTuningLoadRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$")


TRAINING_STRATEGIES = {
    "preserve_entity_matching": {
        "label": "Preserve entity matching",
        "description": "Train only the decision head so the base encoder keeps its name-resolution behavior.",
        "parameters": {
            "epochs": 4,
            "group_size": 4,
            "encoder_lr": 0.0,
            "head_lr": 5e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.01,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "balanced": {
        "label": "Balanced adaptation",
        "description": "Adapt the encoder cautiously while prioritizing retention of general entity understanding.",
        "parameters": {
            "epochs": 4,
            "group_size": 4,
            "encoder_lr": 3e-6,
            "head_lr": 5e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.01,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "maximum_task_adaptation": {
        "label": "Maximum task adaptation",
        "description": "Update the full encoder at the original rate for the strongest criminal-intent specialization.",
        "parameters": {
            "epochs": 4,
            "group_size": 4,
            "encoder_lr": 1e-5,
            "head_lr": 5e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.01,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "relational_low_drift": {
        "label": "Relational low drift",
        "description": "Use cautious encoder updates and stronger regularization for the 1,000-row relational subset.",
        "parameters": {
            "epochs": 4,
            "group_size": 4,
            "encoder_lr": 1e-6,
            "head_lr": 3e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.02,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "relational_adaptation": {
        "label": "Relational adaptation",
        "description": "Adapt the full encoder moderately for the complete 2,000-row relational dataset.",
        "parameters": {
            "epochs": 4,
            "group_size": 4,
            "encoder_lr": 5e-6,
            "head_lr": 5e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.02,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "relational_extended": {
        "label": "Relational extended",
        "description": "Train longer at a lower encoder rate so later epochs can improve relational reasoning with less representation drift.",
        "parameters": {
            "epochs": 8,
            "group_size": 4,
            "encoder_lr": 3e-6,
            "head_lr": 5e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.02,
            "sigma_start": 0.4,
            "sigma_end": 0.1,
        },
    },
    "relational_rlcd": {
        "label": "Relational RLCD pilot",
        "description": "Experiment with a small RLCD objective alongside cautious encoder adaptation; compare it against the strict relational holdout.",
        "parameters": {
            "epochs": 6,
            "group_size": 4,
            "encoder_lr": 3e-6,
            "head_lr": 5e-5,
            "rl_weight": 0.1,
            "weight_decay": 0.02,
            "sigma_start": 0.3,
            "sigma_end": 0.05,
        },
    },
    "public_natural_staged": {
        "label": "Public natural staged",
        "description": "Warm the decision head first, then adapt the encoder slowly with label smoothing and stronger regularization for the public natural dataset.",
        "parameters": {
            "epochs": 10,
            "group_size": 4,
            "encoder_lr": 1e-6,
            "head_lr": 2e-5,
            "rl_weight": 0.0,
            "weight_decay": 0.05,
            "sigma_start": 0.3,
            "sigma_end": 0.05,
            "encoder_warmup_epochs": 1,
            "lr_warmup_ratio": 0.1,
            "label_smoothing": 0.05,
        },
    },
}

for strategy in TRAINING_STRATEGIES.values():
    strategy["parameters"] = {
        "encoder_warmup_epochs": 0,
        "lr_warmup_ratio": 0.0,
        "label_smoothing": 0.0,
        **strategy["parameters"],
    }


AUTHOR_TRAINING_PRESET = {
    "gpu_count": 2,
    "batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation": 4,
    "seed": 42,
    "log_every": 50,
    **TRAINING_STRATEGIES["balanced"]["parameters"],
}


def _resolve_training_parameters(payload: FineTuningTrainRequest) -> dict[str, Any]:
    return payload.model_dump()


def _run_id(name: str, created_at: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:80] or "run"
    timestamp = created_at.strftime("%Y%m%dt%H%M%S%fz")
    return f"{slug}-{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _enabled(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _require_allowed_paths(paths: Mapping[str, Path], roots: tuple[Path, ...]) -> None:
    outside = [name for name, path in paths.items() if not any(path.expanduser().resolve().is_relative_to(root) for root in roots)]
    if outside:
        raise HTTPException(
            status_code=422,
            detail=f"Paths outside LAYA_FINE_TUNING_ALLOWED_ROOTS: {', '.join(outside)}",
        )


def _require_input_files(paths: Mapping[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Fine-tuning input files are missing. "
                f"{'; '.join(missing)}. Copy or mount the private JSONL files, then set "
                "LAYA_BENCHMARK_CORPUS and LAYA_BENCHMARK_LABELS to those paths."
            ),
        )


def _saved_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    if not runs_dir.exists():
        return runs
    for path in runs_dir.glob("*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (runs_dir / f"{metadata.get('run_id')}.jsonl").is_file():
            runs.append({**metadata, "deletable": True})
    return sorted(runs, key=lambda row: row.get("created_at", ""), reverse=True)


def _training_reports(models_root: Path) -> list[dict[str, Any]]:
    if not models_root.is_dir():
        return []
    reports = []
    for output_dir in models_root.iterdir():
        report_path = output_dir / "training_report.json"
        if not output_dir.is_dir() or not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                continue
            history = report.get("history", [])
            if not isinstance(history, list):
                continue
            latest = history[-1] if history and isinstance(history[-1], dict) else {}
            modified = datetime.fromtimestamp(
                report_path.stat().st_mtime, timezone.utc
            ).isoformat()
        except (OSError, json.JSONDecodeError, TypeError, IndexError):
            continue
        reports.append({
            "run_id": output_dir.name,
            "label": output_dir.name.removeprefix("laya-adverse-media-").replace("-", " ").title(),
            "updated_at": modified,
            "epochs": len(history),
            "validation_loss": latest.get("validation_loss"),
            "best_validation_loss": report.get("best_validation_loss"),
            "deletable": True,
        })
    return sorted(reports, key=lambda row: row["updated_at"], reverse=True)


def _historical_training_snapshot(models_root: Path, run_id: str) -> dict[str, Any]:
    reports = {report["run_id"]: report for report in _training_reports(models_root)}
    summary = reports.get(run_id)
    if summary is None:
        raise FileNotFoundError("completed training report not found")
    report = json.loads(
        (models_root / run_id / "training_report.json").read_text(encoding="utf-8")
    )
    history = report.get("history", [])
    if not isinstance(history, list):
        raise ValueError("training report history must be a list")
    latest = history[-1] if history else {}
    loss_history = []
    required_metrics = (
        "epoch", "train_loss", "validation_loss", "validation_accuracy"
    )
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict) or any(name not in row for name in required_metrics):
            raise ValueError(f"training report epoch {index} is incomplete")
        normalized = {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "validation_loss": float(row["validation_loss"]),
            "train_accuracy": (
                float(row["train_accuracy"])
                if row.get("train_accuracy") is not None
                else None
            ),
            "validation_accuracy": float(row["validation_accuracy"]),
        }
        if not all(
            value is None or math.isfinite(value)
            for value in normalized.values()
        ):
            raise ValueError(f"training report epoch {index} has non-finite metrics")
        loss_history.append(normalized)
    return {
        "historical": True,
        "status": "complete",
        "phase": "historical training",
        "pid": None,
        "started_at": summary["updated_at"],
        "command": [],
        "exit_code": 0,
        "error": None,
        "progress": {
            "epoch": latest.get("epoch"),
            "epochs": len(history),
            "loss": latest.get("train_loss"),
        },
        "loss_history": loss_history,
        "logs": [{
            "index": 1,
            "message": "Historical report loaded. Process output was not persisted for this training run.",
        }],
        "report": report,
        "run": summary,
    }


REQUIRED_MODEL_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


class _SelectablePredictor:
    def __init__(
        self,
        router: Predictor,
        fine_tuned_models: Mapping[str, Path],
        agent_factory: Callable[[str], Any],
    ) -> None:
        self._router = router
        self._agent_factory = agent_factory
        self._agents: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._active_model = "auto"
        self.models = dict(getattr(router, "models", {}))
        self.models.update({name: str(path) for name, path in fine_tuned_models.items()})

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return [*self._router.loaded, *self._agents]

    @property
    def active_model(self) -> str:
        return self._active_model

    def _release_fine_tuned(self) -> None:
        if not self._agents:
            return
        self._agents.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self) -> None:
        with self._lock:
            self._router.unload()
            self._release_fine_tuned()
            self._active_model = "unloaded"

    def activate(self, model: str | None) -> str:
        selected = "auto" if model in (None, "auto") else model
        with self._lock:
            if selected == self._active_model:
                return selected
            if selected.startswith("crime-"):
                model_path = self.models.get(selected)
                if model_path is None:
                    raise ValueError(f"unknown fine-tuned model {selected!r}")
                self._router.unload()
                self._release_fine_tuned()
                self._active_model = "loading"
                try:
                    self._agents[selected] = self._agent_factory(model_path)
                except Exception:
                    self._active_model = "unavailable"
                    raise
            else:
                self._release_fine_tuned()
                self._active_model = "loading"
                try:
                    self._router.preload(["english", "multilingual"])
                except Exception:
                    self._active_model = "unavailable"
                    raise
            self._active_model = selected
            return selected

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        with self._lock:
            if model is None or not model.startswith("crime-"):
                self.activate(None)
                return self._router.predict(state, questions, model=model)
            self.activate(model)
            model_path = self.models[model]
            result = dict(self._agents[model].system_one(state, questions))
            result["routing"] = {
                "model": model,
                "repo": model_path,
                "reason": f"explicit model={model!r}",
            }
            return result


def _fine_tuned_model_paths(models_root: Path) -> dict[str, Path]:
    if not models_root.is_dir():
        return {}
    paths = [
        path
        for path in models_root.iterdir()
        if path.is_dir()
        and all((path / required).is_file() for required in REQUIRED_MODEL_FILES)
    ]
    paths.sort(key=lambda path: (-_checkpoint_modified_at(path), path.name.casefold()))
    return {
        f"crime-{re.sub(r'[^a-z0-9]+', '-', path.name.casefold().removeprefix('laya-adverse-media-')).strip('-')}": path
        for path in paths
    }


def _checkpoint_modified_at(path: Path) -> int:
    candidates = [path / "training_report.json", path / "model.safetensors"]
    try:
        return max(
            (candidate.stat().st_mtime_ns for candidate in candidates if candidate.is_file()),
            default=path.stat().st_mtime_ns,
        )
    except OSError:
        return 0


def _refresh_fine_tuned_models(predictor: Predictor, models_root: Path) -> None:
    configured = getattr(predictor, "models", None)
    if not isinstance(configured, dict):
        return
    discovered = _fine_tuned_model_paths(models_root)
    def refresh() -> None:
        for name in list(configured):
            if name.startswith("crime-") and name not in discovered:
                del configured[name]
        configured.update({name: str(path) for name, path in discovered.items()})

    lock = getattr(predictor, "_lock", None)
    if lock is None:
        refresh()
        return
    with lock:
        refresh()


def _build_predictor() -> Predictor:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya import Agent, Router

    device = os.getenv("LAYA_DEVICE") or None
    project_root = Path(__file__).resolve().parents[2]
    bundle = Path(
        os.getenv(
            "LAYA_MODEL_BUNDLE",
            project_root / "models" / "models--convaiinnovations--laya",
        )
    ).expanduser().resolve()
    revision_file = bundle / "refs" / "main"
    if not revision_file.is_file():
        raise FileNotFoundError(
            f"Local Laya bundle has no refs/main at {bundle}. "
            "Set LAYA_MODEL_BUNDLE to the copied Hugging Face cache directory. "
            "Runtime network downloads are disabled."
        )
    revision = revision_file.read_text(encoding="utf-8").strip()
    snapshot = bundle / "snapshots" / revision
    model_paths = {
        "english": snapshot,
        "multilingual": snapshot / "multilingual",
        "typed-decisions": snapshot / "typed-decisions",
    }
    explicit_model = os.getenv("LAYA_MODEL_PATH")
    if explicit_model:
        model_paths["english"] = Path(explicit_model).expanduser().resolve()

    fine_tuned_root = Path(
        os.getenv(
            "LAYA_FINE_TUNED_MODELS_DIR",
            project_root / "models" / "fine-tuned",
        )
    ).expanduser().resolve()
    fine_tuned_models = _fine_tuned_model_paths(fine_tuned_root)
    missing = [
        f"{name}: {path / required}"
        for name, path in model_paths.items()
        for required in REQUIRED_MODEL_FILES
        if not (path / required).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Local Laya checkpoint bundle is incomplete. Missing: "
            f"{', '.join(missing)}. Runtime network downloads are disabled."
        )

    router = Router(
        models={name: str(path) for name, path in model_paths.items()},
        device=device,
    )
    router.preload(["english", "multilingual"])
    return _SelectablePredictor(
        router,
        fine_tuned_models,
        lambda path: Agent(path, device=device),
    )


def _model_prompt(predictor: Predictor, model: str | None) -> dict[str, Any]:
    if model and model.startswith("crime-"):
        configured = getattr(predictor, "models", {})
        model_path = configured.get(model)
        if model_path is not None:
            try:
                config = json.loads(
                    (Path(str(model_path)) / "rl_agent_config.json").read_text(
                        encoding="utf-8"
                    )
                )
                return validate_prompt(config.get("prompt"))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                pass
    return validate_prompt(None)


def _model_options(predictor: Predictor) -> list[dict[str, Any]]:
    configured = getattr(predictor, "models", {})
    fine_tuned = sorted(
        (name for name in configured if name.startswith("crime-")),
        key=lambda name: (
            -_checkpoint_modified_at(Path(str(configured[name]))),
            name.casefold(),
        ),
    )
    return [
        {
            "id": name,
            "label": f"LayaCrime · {name.removeprefix('crime-').replace('-', ' ').title()}",
            "family": "fine-tuned",
            "category": "decision",
            "deletable": True,
            "prompt": _model_prompt(predictor, name),
        }
        for name in fine_tuned
    ] + [{
        "id": "auto",
        "label": "Laya · Router",
        "family": "original",
        "category": "decision",
        "deletable": False,
        "prompt": _model_prompt(predictor, None),
    }]


def _available_model_ids(predictor: Predictor) -> set[str]:
    configured = getattr(predictor, "models", {})
    return {"english", "multilingual", *configured}


def _questions(
    entity_name: str,
    question: str | None = None,
    criteria: list[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    instruction = question or (
        "How does this article portray {entity_name} regarding criminal behavior or intent? "
        "Judge only the named entity, not other people or organizations."
    )
    configured_criteria = criteria or [
        {
            "decision": "negative",
            "text": (
                "negative: the article credibly associates the entity with alleged, "
                "investigated, charged, convicted, sanctioned, or admitted criminal behavior "
                "or intent"
            ),
        },
        {
            "decision": "positive",
            "text": (
                "positive: the article does not associate the entity with criminal behavior or intent, "
                "or identifies the entity only as a victim, witness, investigator, or unrelated party"
            ),
        },
    ]
    grouped = {
        decision: " | ".join(
            criterion["text"]
            for criterion in configured_criteria
            if criterion["decision"] == decision
        )
        for decision in ("negative", "positive")
    }
    if not all(grouped.values()):
        raise ValueError("prompt criteria must include negative and positive decisions")
    return {
        "criminal_association": {
            "type": "choice",
            "instructions": instruction.replace("{entity_name}", repr(entity_name)),
            "criteria": {
                "A": grouped["negative"],
                "B": grouped["positive"],
            },
        }
    }


def _parse_result(
    result: Mapping[str, Any], entity_name: str, review_threshold: float
) -> AdverseMediaResponse:
    try:
        answer = result["answers"]["criminal_association"]
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        raw_probabilities = answer["probabilities"]
        probabilities = {
            "negative": float(raw_probabilities.get("A", 0.0)),
            "positive": float(raw_probabilities.get("B", 0.0)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Laya returned an unexpected adverse-media response") from exc

    if choice not in {"A", "B"}:
        raise ValueError(f"Laya returned unsupported choice {choice!r}")

    return AdverseMediaResponse(
        entity_name=entity_name,
        decision="negative" if choice == "A" else "positive",
        confidence=confidence,
        probabilities=probabilities,
        needs_review=confidence < review_threshold,
        routing=dict(result["routing"]) if result.get("routing") else None,
    )


def create_app(
    predictor_factory: Callable[[], Predictor] = _build_predictor,
    benchmark_corpus: Path | None = None,
    benchmark_gold: Path | None = None,
    benchmark_source_dir: Path | None = None,
    benchmark_output: Path | None = None,
) -> FastAPI:
    review_threshold = float(os.getenv("LAYA_REVIEW_THRESHOLD", "0.80"))
    if not 0 <= review_threshold <= 1:
        raise ValueError("LAYA_REVIEW_THRESHOLD must be between 0 and 1")
    project_root = Path(__file__).resolve().parents[2]
    server_host = os.getenv("LAYA_HOST", "127.0.0.1").strip().casefold()
    configured_fine_tuning = os.getenv("LAYA_ENABLE_FINE_TUNING")
    fine_tuning_enabled = (
        _enabled(configured_fine_tuning)
        if configured_fine_tuning is not None
        else server_host in {"127.0.0.1", "localhost", "::1"}
    )
    configured_roots = os.getenv("LAYA_FINE_TUNING_ALLOWED_ROOTS", str(project_root))
    fine_tuning_roots = tuple(
        Path(value.strip()).expanduser().resolve()
        for value in configured_roots.split(os.pathsep)
        if value.strip()
    )
    if not fine_tuning_roots:
        raise ValueError("LAYA_FINE_TUNING_ALLOWED_ROOTS must contain at least one path")
    datasets_root = Path(
        os.getenv("LAYA_DATASETS_DIR", project_root / "datasets")
    ).expanduser().resolve()
    dataset_bundles = discover_dataset_bundles(datasets_root)
    benchmark_dataset_bundles = {
        dataset_id: bundle
        for dataset_id, bundle in dataset_bundles.items()
        if bundle.manifest.get("purpose", "training-and-evaluation")
        != "training-only"
    }
    active_dataset_id = os.getenv("LAYA_ACTIVE_DATASET", "")
    if active_dataset_id and active_dataset_id not in dataset_bundles:
        raise ValueError(f"LAYA_ACTIVE_DATASET {active_dataset_id!r} was not found")
    active_dataset: DatasetBundle | None = (
        benchmark_dataset_bundles.get(active_dataset_id)
        if active_dataset_id in benchmark_dataset_bundles
        else next(iter(benchmark_dataset_bundles.values()), None)
    )
    external_data_root = project_root / "external-data"
    configured_corpus = os.getenv("LAYA_BENCHMARK_CORPUS")
    corpus_path = benchmark_corpus or (
        Path(configured_corpus)
        if configured_corpus
        else active_dataset.corpus
        if active_dataset
        else external_data_root / "benchmark-corpus.jsonl"
    )
    configured_labels = os.getenv("LAYA_BENCHMARK_LABELS")
    configured_gold_path = benchmark_gold or (
        Path(configured_labels)
        if configured_labels
        else active_dataset.annotations[0].path
        if active_dataset
        else external_data_root / "benchmark-labels.jsonl"
    )
    label_sets_root = Path(
        os.getenv(
            "LAYA_LABEL_SETS_DIR",
            active_dataset.root / "annotations"
            if active_dataset
            else project_root / "artifacts" / "label-sets",
        )
    ).expanduser().resolve()

    def label_set_catalog() -> dict[str, dict[str, Any]]:
        catalog: dict[str, dict[str, Any]] = {}
        if benchmark_gold is None and active_dataset is not None:
            user_managed_ids = {
                row.get("id")
                for row in active_dataset.manifest.get("annotations", [])
                if isinstance(row, dict) and row.get("user_managed") is True
            }
            for annotation in active_dataset.annotations:
                if annotation.path.is_file():
                    catalog[annotation.id] = {
                        "id": annotation.id,
                        "label": annotation.name,
                        "path": annotation.path,
                        "built_in": annotation.id not in user_managed_ids,
                    }
        if benchmark_gold is None or os.getenv("LAYA_LABEL_SETS_DIR"):
            for path in sorted(label_sets_root.glob("*.jsonl")):
                label_set_id = re.sub(r"[^a-z0-9]+", "-", path.stem.casefold()).strip("-")
                if label_set_id and label_set_id not in catalog:
                    catalog[label_set_id] = {
                        "id": label_set_id,
                        "label": path.stem.replace("-", " ").title(),
                        "path": path.resolve(),
                        "built_in": path.name == "human.jsonl",
                    }
        if configured_gold_path.is_file() and (
            benchmark_gold is not None or configured_labels or active_dataset is None
        ):
            configured_id = "default" if benchmark_gold is not None else os.getenv(
                "LAYA_BENCHMARK_LABEL_SET_ID", "configured"
            )
            catalog[configured_id] = {
                "id": configured_id,
                "label": "Default" if benchmark_gold is not None else os.getenv(
                    "LAYA_BENCHMARK_LABEL_SET_LABEL", "Configured labels"
                ),
                "path": configured_gold_path.resolve(),
                "built_in": benchmark_gold is None,
            }
        return catalog

    initial_catalog = label_set_catalog()
    configured_active_set = os.getenv(
        "LAYA_BENCHMARK_LABEL_SET", next(iter(initial_catalog), "")
    )
    active_label_set_id = (
        configured_active_set
        if configured_active_set in initial_catalog
        else next(iter(initial_catalog), "")
    )
    gold_path = (
        initial_catalog[active_label_set_id]["path"]
        if active_label_set_id
        else configured_gold_path
    )
    configured_source_dir = os.getenv("LAYA_BENCHMARK_SOURCE_DIR")
    source_dir = benchmark_source_dir or (
        Path(configured_source_dir) if configured_source_dir else None
    )

    def load_configured_benchmark(
        annotation_path: Path,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ]:
        rows, labels, human_labels, audit = load_benchmark(
            corpus_path, annotation_path, source_dir
        )
        if active_dataset is not None:
            audit["dataset"] = {
                "id": active_dataset.id,
                "name": active_dataset.name,
            }
        return rows, labels, human_labels, audit

    default_output = (
        project_root
        / "artifacts"
        / "benchmark-runs"
        / (active_dataset.id if active_dataset else "external")
        / "active.predictions.jsonl"
        if benchmark_corpus is None
        else corpus_path.with_name("laya-live.predictions.jsonl")
    )
    output_path = benchmark_output or Path(os.getenv("LAYA_BENCHMARK_OUTPUT", default_output))
    runs_dir = output_path.parent / "saved-runs"
    active_metadata_path = output_path.with_suffix(".meta.json")

    def activate_dataset_paths(dataset_id: str) -> None:
        nonlocal active_dataset, corpus_path, configured_gold_path
        nonlocal label_sets_root, active_label_set_id, gold_path
        nonlocal output_path, runs_dir, active_metadata_path
        if benchmark_corpus is not None or benchmark_gold is not None:
            raise ValueError("dataset switching is unavailable with explicit benchmark paths")
        bundle = benchmark_dataset_bundles.get(dataset_id)
        if bundle is None:
            raise ValueError(f"unknown dataset {dataset_id!r}")
        active_dataset = bundle
        corpus_path = bundle.corpus
        configured_gold_path = bundle.annotations[0].path
        label_sets_root = bundle.root / "annotations"
        catalog = label_set_catalog()
        active_label_set_id = next(iter(catalog), "")
        gold_path = (
            catalog[active_label_set_id]["path"]
            if active_label_set_id
            else configured_gold_path
        )
        configured_output = benchmark_output or os.getenv("LAYA_BENCHMARK_OUTPUT")
        output_path = Path(configured_output) if configured_output else (
            project_root
            / "artifacts"
            / "benchmark-runs"
            / bundle.id
            / "active.predictions.jsonl"
        )
        runs_dir = output_path.parent / "saved-runs"
        active_metadata_path = output_path.with_suffix(".meta.json")
    fine_tuning_script = project_root / "scripts" / "fine_tune_laya.py"
    labels_root_value = os.getenv("LAYA_TRAINING_LABELS_DIR")
    labels_root = Path(labels_root_value).expanduser().resolve() if labels_root_value else None
    fine_tuned_models_root = Path(
        os.getenv(
            "LAYA_FINE_TUNED_MODELS_DIR", project_root / "models" / "fine-tuned"
        )
    ).expanduser().resolve()
    model_bundle = project_root / "models" / "models--convaiinnovations--laya"
    model_revision = (
        (model_bundle / "refs" / "main").read_text(encoding="utf-8").strip()
        if (model_bundle / "refs" / "main").is_file()
        else ""
    )
    fine_tuning_model = Path(
        os.getenv(
            "LAYA_FINE_TUNING_MODEL_DIR",
            model_bundle / "snapshots" / model_revision,
        )
    )
    def fine_tuning_model_presets() -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "label": f"LayaCrime · {model_id.removeprefix('crime-').replace('-', ' ').title()}",
                "path": str(path),
                "deletable": True,
            }
            for model_id, path in _fine_tuned_model_paths(fine_tuned_models_root).items()
        ] + [{
            "id": "auto",
            "label": "Laya · Router",
            "path": str(fine_tuning_model),
            "deletable": False,
        }]
    full_dataset_presets = []
    for dataset in dataset_bundles.values():
        if dataset.manifest.get("purpose", "training-and-evaluation") == "evaluation-only":
            continue
        for annotation in dataset.annotations:
            sample_count = len(load_gold_labels(annotation.path))
            if sample_count == 0:
                continue
            full_dataset_presets.append({
                "id": f"{dataset.id}-{annotation.id}-full",
                "label": f"{dataset.name} · {annotation.name} · Full set",
                "samples": sample_count,
                "dataset_id": dataset.id,
                "label_set_id": annotation.id,
                "deletable": annotation.user_managed,
                "prompt": dataset.prompt,
                "recommended_strategy": dataset.manifest.get(
                    "recommended_training_strategy"
                ),
                "corpus": str(dataset.corpus),
                "labels": str(annotation.path),
                "prepared_dir": str(
                    project_root
                    / "artifacts"
                    / "prepared-data"
                    / dataset.id
                    / annotation.id
                ),
                "output_dir": str(
                    fine_tuned_models_root
                    / f"laya-adverse-media-{dataset.id}-{annotation.id}"
                ),
            })
    if active_dataset is None:
        for item in initial_catalog.values():
            sample_count = len(load_gold_labels(item["path"]))
            if sample_count == 0:
                continue
            full_dataset_presets.append({
                "id": f"external-{item['id']}-full",
                "label": f"{item['label']} · Full set",
                "samples": sample_count,
                "dataset_id": None,
                "label_set_id": item["id"],
                "deletable": not item["built_in"],
                "prompt": None,
                "recommended_strategy": None,
                "corpus": str(corpus_path),
                "labels": str(item["path"]),
                "prepared_dir": str(
                    project_root / "artifacts" / "prepared-data" / "external" / item["id"]
                ),
                "output_dir": str(
                    fine_tuned_models_root / f"laya-adverse-media-{item['id']}"
                ),
            })
    bundle_subset_presets = []
    for dataset in dataset_bundles.values():
        if dataset.manifest.get("purpose", "training-and-evaluation") == "evaluation-only":
            continue
        for group in dataset.manifest.get("training_subsets", []):
            if (
                not isinstance(group, dict)
                or not isinstance(group.get("annotation_set_id"), str)
                or not isinstance(group.get("path"), str)
            ):
                continue
            annotation_set_id = group["annotation_set_id"]
            annotation_name = next(
                (
                    annotation.name
                    for annotation in dataset.annotations
                    if annotation.id == annotation_set_id
                ),
                annotation_set_id,
            )
            for size in group.get("sizes", []):
                if not isinstance(size, int):
                    continue
                labels_path = dataset.root / group["path"] / f"labels-{size:05d}.jsonl"
                if not labels_path.is_file():
                    continue
                bundle_subset_presets.append({
                    "id": f"{dataset.id}-{annotation_set_id}-{size}",
                    "label": f"{dataset.name} · {annotation_name} · {size:,} samples",
                    "samples": size,
                    "dataset_id": dataset.id,
                    "prompt": dataset.prompt,
                    "recommended_strategy": dataset.manifest.get(
                        "recommended_training_strategy"
                    ),
                    "corpus": str(dataset.corpus),
                    "labels": str(labels_path),
                    "prepared_dir": str(
                        project_root
                        / "artifacts"
                        / "prepared-data"
                        / dataset.id
                        / f"{annotation_set_id}-{size}"
                    ),
                    "output_dir": str(
                        fine_tuned_models_root
                        / f"laya-adverse-media-{dataset.id}-{annotation_set_id}-{size}"
                    ),
                })
    legacy_subset_presets = [
        {
            "id": f"external-{size}",
            "label": f"External labels · {size:,} samples",
            "samples": size,
            "corpus": str(corpus_path),
            "labels": str(labels_root / f"labels-{size:05d}.jsonl"),
            "prepared_dir": str(project_root / "artifacts" / "prepared-data" / f"external-{size}"),
            "output_dir": str(project_root / "models" / f"laya-adverse-media-external-{size}"),
        }
        for size in (1_000, 2_000, 5_000, 10_000)
        if labels_root is not None and (labels_root / f"labels-{size:05d}.jsonl").is_file()
    ]
    dataset_presets = full_dataset_presets + bundle_subset_presets + legacy_subset_presets
    default_preset = next(
        (
            preset
            for preset in full_dataset_presets
            if active_dataset is not None
            if preset["id"]
            == f"{active_dataset.id}-{active_label_set_id}-full"
        ),
        dataset_presets[0] if dataset_presets else None,
    ) or {
        "labels": str(gold_path),
        "prepared_dir": str(
            project_root
            / "artifacts"
            / "prepared-data"
            / (active_dataset.id if active_dataset else "external")
        ),
        "output_dir": str(fine_tuned_models_root / "laya-adverse-media"),
    }
    fine_tuning_defaults = {
        "dataset": (
            {"id": active_dataset.id, "name": active_dataset.name}
            if active_dataset
            else None
        ),
        "corpus": str(corpus_path),
        "labels": default_preset["labels"],
        "model_dir": str(fine_tuning_model),
        "prepared_dir": default_preset["prepared_dir"],
        "output_dir": default_preset["output_dir"],
        "dataset_presets": dataset_presets,
        "author_training_preset": AUTHOR_TRAINING_PRESET,
        "training_strategies": [
            {"id": strategy_id, **strategy}
            for strategy_id, strategy in TRAINING_STRATEGIES.items()
        ],
        "default_training_strategy": "balanced",
        "enabled": fine_tuning_enabled,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.time()
        app.state.predictor = None
        app.state.startup_error = None
        app.state.inference_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="laya-infer"
        )
        app.state.benchmark = BenchmarkState()
        app.state.benchmark_task = None
        app.state.benchmark_rows = []
        app.state.benchmark_gold = {}
        app.state.benchmark_human = {}
        app.state.benchmark_gold_mtime = None
        app.state.benchmark_gold_path = gold_path
        app.state.benchmark_label_set_id = active_label_set_id
        app.state.benchmark_routing_mode = "auto"
        default_prompt = validate_prompt(None)
        app.state.benchmark_question = default_prompt["question"]
        app.state.benchmark_criteria = default_prompt["criteria"]
        app.state.benchmark_active_model = "auto"
        app.state.benchmark_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "truncated_examples": 0,
        }
        app.state.benchmark_saved_run = None
        app.state.generative_runtime = None
        app.state.model_installation = ModelInstallationJob()
        app.state.fine_tuning = FineTuningJob()
        if output_path.is_file():
            try:
                active_metadata = (
                    json.loads(active_metadata_path.read_text(encoding="utf-8"))
                    if active_metadata_path.is_file()
                    else {}
                )
                restored_set_id = active_metadata.get("label_set_id", active_label_set_id)
                restored_set = label_set_catalog().get(restored_set_id)
                restored_path = restored_set["path"] if restored_set else gold_path
                articles, gold_labels, human_labels, audit = load_configured_benchmark(
                    restored_path
                )
                if restored_set:
                    audit["label_set"] = {
                        "id": restored_set["id"],
                        "label": restored_set["label"],
                    }
                rows = read_jsonl(output_path)
                if rows:
                    app.state.benchmark.restore(
                        rows,
                        articles,
                        gold_labels,
                        human_labels,
                        audit,
                        total=int(active_metadata.get("total") or len(articles)),
                        elapsed_seconds=active_metadata.get("elapsed_seconds"),
                    )
                    app.state.benchmark_rows = articles
                    app.state.benchmark_gold = gold_labels
                    app.state.benchmark_human = human_labels
                    app.state.benchmark_gold_path = restored_path
                    app.state.benchmark_gold_mtime = restored_path.stat().st_mtime_ns
                    app.state.benchmark_label_set_id = restored_set_id
                    app.state.benchmark_routing_mode = active_metadata.get(
                        "routing_mode", "auto"
                    )
                    app.state.benchmark_question = active_metadata.get(
                        "question", default_prompt["question"]
                    )
                    app.state.benchmark_criteria = active_metadata.get(
                        "criteria", default_prompt["criteria"]
                    )
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
                pass
        app.state.warmup_future = app.state.inference_pool.submit(predictor_factory)
        try:
            yield
        finally:
            if app.state.benchmark_task and not app.state.benchmark_task.done():
                app.state.benchmark_task.cancel()
                try:
                    await app.state.benchmark_task
                except asyncio.CancelledError:
                    pass
            app.state.fine_tuning.stop()
            app.state.model_installation.stop()
            if app.state.generative_runtime is not None:
                app.state.generative_runtime.close()
            app.state.inference_pool.shutdown(wait=True)

    def resolve_predictor(request: Request, wait: float = 0) -> Predictor | None:
        if request.app.state.predictor is not None:
            return request.app.state.predictor
        if request.app.state.startup_error is not None:
            return None

        future: Future[Predictor] = request.app.state.warmup_future
        try:
            request.app.state.predictor = future.result(timeout=wait)
        except TimeoutError:
            return None
        except Exception as exc:  # Keep diagnostics and the test UI available.
            request.app.state.startup_error = str(exc)
            return None
        return request.app.state.predictor

    app = FastAPI(
        title="Laya Adverse Media",
        version="0.1.0",
        lifespan=lifespan,
    )
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "individual-test.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(static_dir / "favicon.svg", media_type="image/svg+xml")

    @app.get("/benchmark", include_in_schema=False)
    def benchmark_page() -> FileResponse:
        return FileResponse(static_dir / "benchmark.html")

    @app.get("/fine-tuning", include_in_schema=False)
    def fine_tuning_page() -> FileResponse:
        return FileResponse(static_dir / "fine-tuning.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request, wait=0.05)
        if predictor is None:
            return {
                "status": (
                    "degraded" if request.app.state.startup_error else "warming"
                ),
                "loaded": [],
                "error": request.app.state.startup_error,
            }
        active_model = getattr(predictor, "active_model", "auto")
        if active_model in {"loading", "unavailable"}:
            return {
                "status": "warming" if active_model == "loading" else "degraded",
                "loaded": [],
                "error": None if active_model == "loading" else "Selected model failed to load",
            }
        return {"status": "ok", "loaded": predictor.loaded, "error": None}

    @app.get("/diagnostics")
    def diagnostics(request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request, wait=0.05)
        if predictor is None:
            return {
                "warm_start": False,
                "status": "degraded" if request.app.state.startup_error else "warming",
                "error": request.app.state.startup_error,
            }

        agents = getattr(predictor, "_agents", {})
        model_specs = getattr(predictor, "models", {})
        return {
            "warm_start": True,
            "predictor": f"{type(predictor).__module__}.{type(predictor).__name__}",
            "uptime_seconds": round(time.time() - request.app.state.started_at, 1),
            "resident_models": {
                name: {
                    "instance_id": hex(id(agent)),
                    "device": str(getattr(agent, "device", "unknown")),
                    "source": Path(str(model_specs.get(name, "unknown"))).name,
                }
                for name, agent in agents.items()
            },
            "network_model_downloads": "disabled",
        }

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request, wait=0.05)
        if predictor is None:
            warming = request.app.state.startup_error is None
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Laya model is warming up"
                        if warming
                        else "Laya model is unavailable"
                    ),
                    "startup_error": request.app.state.startup_error,
                },
            )
        _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
        return {
            "models": _model_options(predictor),
            "active_model": getattr(predictor, "active_model", "auto"),
        }

    @app.post("/v1/models/activate")
    async def activate_model(
        payload: ModelActivationRequest, request: Request
    ) -> dict[str, Any]:
        if request.app.state.benchmark.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409,
                detail="reset or finish the active benchmark before changing models",
            )
        predictor = resolve_predictor(request)
        if predictor is None:
            raise HTTPException(status_code=503, detail="Laya model is not ready")
        _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
        visible_models = {option["id"] for option in _model_options(predictor)}
        if payload.model_id not in visible_models:
            raise HTTPException(status_code=422, detail="Unknown model selection")
        activate = getattr(predictor, "activate", None)
        if callable(activate):
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    request.app.state.inference_pool,
                    lambda: activate(payload.model_id),
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not load selected model: {exc}",
                ) from exc
        return {
            "model_id": payload.model_id,
            "active_model": getattr(predictor, "active_model", payload.model_id),
            "loaded": list(predictor.loaded),
        }

    def generative_runtime(
        request: Request, predictor: Predictor | None
    ) -> GenerativeBenchmarkRuntime:
        runtime = request.app.state.generative_runtime
        if runtime is None:
            def unload_laya() -> None:
                unload = getattr(predictor, "unload", None)
                if callable(unload):
                    unload()
                    return
                router = getattr(predictor, "_router", None)
                router_unload = getattr(router, "unload", None)
                if callable(router_unload):
                    router_unload()

            runtime = GenerativeBenchmarkRuntime(
                Path(os.getenv("LAYA_GGUF_MODELS_DIR", project_root / "models" / "gguf"))
                .expanduser()
                .resolve(),
                unload_laya,
            )
            request.app.state.generative_runtime = runtime
        return runtime

    def gguf_models_root() -> Path:
        return Path(
            os.getenv("LAYA_GGUF_MODELS_DIR", project_root / "models" / "gguf")
        ).expanduser().resolve()

    @app.get("/v1/benchmark/models")
    def benchmark_models(request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request, wait=0.05)
        laya_models: list[dict[str, Any]] = []
        if predictor is not None:
            _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
            laya_models = _model_options(predictor)
        gguf_root = gguf_models_root()
        generative_models = benchmark_model_options(gguf_root)
        generative_ids = {item["id"] for item in generative_models}
        active_model = request.app.state.benchmark_active_model
        if active_model in generative_ids:
            selected = next(item for item in generative_models if item["id"] == active_model)
            runtime = request.app.state.generative_runtime
            if (
                not selected["installed"]
                or runtime is None
                or runtime.active_model != active_model
                or not runtime.is_healthy
            ):
                if runtime is not None:
                    runtime.close()
                active_model = "auto"
                request.app.state.benchmark_active_model = active_model
        return {
            "models": laya_models + generative_models,
            "active_model": active_model,
            "laya_error": request.app.state.startup_error,
        }

    @app.post("/v1/benchmark/models/custom")
    def benchmark_add_custom_model(
        payload: BenchmarkCustomModelRequest, request: Request
    ) -> dict[str, Any]:
        if request.app.state.benchmark.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409,
                detail="reset or finish the active benchmark before adding models",
            )
        try:
            spec = register_custom_model(
                gguf_models_root(), payload.name, payload.repo_id, payload.huggingface_path
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"model": next(
            item for item in benchmark_model_options(gguf_models_root())
            if item["id"] == spec.id
        )}

    @app.post("/v1/benchmark/models/activate")
    async def benchmark_activate_model(
        payload: BenchmarkModelActivationRequest, request: Request
    ) -> dict[str, Any]:
        if request.app.state.benchmark.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409,
                detail="reset or finish the active benchmark before changing models",
            )
        predictor = resolve_predictor(request)
        loop = asyncio.get_running_loop()
        try:
            gguf_root = gguf_models_root()
            spec = registered_model_by_id(gguf_root, payload.model_id)
            if spec is not None:
                credentials = None
                if payload.endpoint or payload.api_key or payload.deployment:
                    credentials = {
                        "endpoint": payload.endpoint or "",
                        "api_key": (
                            payload.api_key.get_secret_value() if payload.api_key else ""
                        ),
                        "deployment": payload.deployment or "",
                    }
                runtime = generative_runtime(request, predictor)
                await loop.run_in_executor(
                    request.app.state.inference_pool,
                    lambda: runtime.activate(payload.model_id, credentials, spec),
                )
            else:
                if predictor is None:
                    raise RuntimeError("Laya model is not ready")
                runtime = request.app.state.generative_runtime
                if runtime is not None:
                    await loop.run_in_executor(
                        request.app.state.inference_pool, runtime.close
                    )
                visible = {option["id"] for option in _model_options(predictor)}
                if payload.model_id not in visible:
                    raise ValueError("Unknown model selection")
                activate = getattr(predictor, "activate", None)
                if callable(activate):
                    await loop.run_in_executor(
                        request.app.state.inference_pool,
                        lambda: activate(payload.model_id),
                    )
            request.app.state.benchmark_active_model = payload.model_id
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not activate model: {exc}") from exc
        return {
            "model_id": payload.model_id,
            "active_model": request.app.state.benchmark_active_model,
        }

    @app.post("/v1/benchmark/models/delete")
    def benchmark_delete_model(
        payload: ModelActivationRequest, request: Request
    ) -> dict[str, Any]:
        if request.app.state.benchmark.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409, detail="finish or reset the benchmark before deleting models"
            )
        if request.app.state.fine_tuning.snapshot()["status"] in {"running", "stopping"}:
            raise HTTPException(
                status_code=409, detail="stop fine-tuning before deleting models"
            )
        if request.app.state.benchmark_active_model == payload.model_id:
            raise HTTPException(status_code=409, detail="the active model cannot be deleted")
        gguf_root = gguf_models_root()
        spec = registered_model_by_id(gguf_root, payload.model_id)
        predictor = resolve_predictor(request, wait=0.05)
        try:
            if spec is not None:
                if spec.provider != "llama.cpp" or not spec.filename:
                    raise HTTPException(
                        status_code=403, detail="hosted catalog models cannot be deleted"
                    )
                model_path = (gguf_root / spec.filename).resolve()
                if not model_path.is_relative_to(gguf_root):
                    raise HTTPException(status_code=404, detail="installed model not found")
                if not model_path.is_file() and not spec.custom:
                    raise HTTPException(status_code=404, detail="installed model not found")
                model_path.unlink(missing_ok=True)
                model_installation_receipt_path(model_path).unlink(missing_ok=True)
                if spec.custom:
                    remove_registered_model(gguf_root, spec.id)
            else:
                model_path = _fine_tuned_model_paths(fine_tuned_models_root).get(
                    payload.model_id
                )
                if model_path is None:
                    raise HTTPException(status_code=404, detail="deletable model not found")
                shutil.rmtree(model_path)
                if predictor is not None:
                    _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
        except OSError as exc:
            raise HTTPException(status_code=500, detail="model could not be deleted") from exc
        catalog = benchmark_models(request)
        return {"deleted": payload.model_id, **catalog}

    @app.post("/v1/benchmark/models/install")
    def benchmark_install_model(
        payload: BenchmarkModelInstallRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning()
        gguf_root = gguf_models_root()
        spec = registered_model_by_id(gguf_root, payload.model_id)
        if spec is None or spec.provider != "llama.cpp" or not spec.repo_id or not spec.filename:
            raise HTTPException(status_code=422, detail="Selected model is not a local GGUF")
        try:
            spec = update_registered_source(
                gguf_root,
                spec,
                payload.repo_id or spec.repo_id,
                payload.huggingface_path or spec.huggingface_path or spec.filename,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        token = (
            payload.huggingface_token.get_secret_value()
            if payload.huggingface_token else None
        )

        gguf_root.mkdir(parents=True, exist_ok=True)
        model_installation_receipt_path(gguf_root / spec.filename).unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.pop("HF_HUB_OFFLINE", None)
        try:
            request.app.state.model_installation.start(
                [
                    sys.executable,
                    str(project_root / "scripts" / "download_huggingface_file.py"),
                    "--repo-id",
                    spec.repo_id,
                    "--filename",
                    spec.huggingface_path or spec.filename,
                    "--target-filename",
                    spec.filename,
                    "--local-dir",
                    str(gguf_root),
                    "--project-root",
                    str(project_root),
                ],
                spec.id,
                gguf_root,
                project_root,
                stdin_data=json.dumps({"token": token}),
                env=environment,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return request.app.state.model_installation.snapshot()

    @app.get("/v1/benchmark/models/install")
    def benchmark_install_status(request: Request, logs_after: int = 0) -> dict[str, Any]:
        return request.app.state.model_installation.snapshot(max(0, logs_after))

    @app.post("/v1/benchmark/models/install/stop")
    def benchmark_install_stop(request: Request) -> dict[str, Any]:
        request.app.state.model_installation.stop()
        return request.app.state.model_installation.snapshot()

    @app.post("/v1/adverse-media", response_model=AdverseMediaResponse)
    async def adverse_media(
        payload: AdverseMediaRequest, request: Request
    ) -> AdverseMediaResponse:
        predictor = resolve_predictor(request)
        if predictor is None:
            warming = request.app.state.startup_error is None
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Laya model is warming up"
                        if warming
                        else "Laya model is unavailable"
                    ),
                    "startup_error": request.app.state.startup_error,
                },
            )
        loop = asyncio.get_running_loop()
        try:
            _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
            selected_model = (None if payload.model_id == "auto" else payload.model_id) or (
                None if payload.routing_mode == "auto" else payload.routing_mode
            )
            available_models = _available_model_ids(predictor)
            if selected_model is not None and selected_model not in available_models:
                raise HTTPException(status_code=422, detail="Unknown model selection")
            model_prompt = _model_prompt(predictor, selected_model)
            result = await loop.run_in_executor(
                request.app.state.inference_pool,
                lambda: predictor.predict(
                    {"article": payload.article, "entity_name": payload.entity_name},
                    _questions(
                        payload.entity_name,
                        payload.question or model_prompt["question"],
                        [criterion.model_dump() for criterion in payload.criteria]
                        if payload.criteria else model_prompt["criteria"],
                    ),
                    model=selected_model,
                ),
            )
            return _parse_result(result, payload.entity_name, review_threshold)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def persist_active_benchmark(state: BenchmarkState, app_state: Any) -> None:
        active = state.snapshot()
        _write_json(active_metadata_path, {
            "total": active["total"],
            "completed": active["completed"],
            "status": active["status"],
            "routing_mode": app_state.benchmark_routing_mode,
            "label_set_id": app_state.benchmark_label_set_id,
            "question": app_state.benchmark_question,
            "criteria": app_state.benchmark_criteria,
            "elapsed_seconds": active["timing"]["elapsed_seconds"],
            "error": active["error"],
            "usage": app_state.benchmark_usage,
        })

    def usage_from_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, int | float]:
        totals: dict[str, int | float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "truncated_examples": 0,
        }
        for row in rows:
            routing = row.get("routing") or {}
            usage = routing.get("usage") or {}
            totals["input_tokens"] += int(usage.get("input_tokens", 0))
            totals["output_tokens"] += int(usage.get("output_tokens", 0))
            totals["cost_usd"] += float(routing.get("cost_usd", 0.0))
            totals["truncated_examples"] += int(
                bool((routing.get("input") or {}).get("article_truncated"))
            )
        return totals

    def restore_active_ledger(request: Request, current: dict[str, Any]) -> dict[str, Any]:
        persisted_rows = read_jsonl(output_path) if output_path.is_file() else []
        known_ids = {row["article_id"] for row in request.app.state.benchmark_rows}
        persisted_ids = [row.get("article_id") for row in persisted_rows]
        if len(persisted_ids) != len(set(persisted_ids)) or any(
            identifier not in known_ids for identifier in persisted_ids
        ):
            raise HTTPException(
                status_code=409,
                detail="prediction ledger contains duplicate or unknown article IDs; reset the run",
            )
        request.app.state.benchmark.restore(
            persisted_rows,
            request.app.state.benchmark_rows,
            request.app.state.benchmark_gold,
            request.app.state.benchmark_human,
            current["audit"],
            total=current["total"],
            elapsed_seconds=current["timing"]["elapsed_seconds"],
        )
        request.app.state.benchmark_usage = usage_from_prediction_rows(persisted_rows)
        return request.app.state.benchmark.snapshot()

    def automatic_run_name(request: Request) -> str:
        model_id = request.app.state.benchmark_routing_mode
        model = registered_model_by_id(gguf_models_root(), model_id)
        label = model.label if model else next(
            (
                item["label"]
                for item in _model_options(request.app.state.predictor)
                if item["id"] == model_id
            ),
            model_id,
        )
        dataset = active_dataset.id if active_dataset else corpus_path.stem
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H%M UTC")
        return (
            f"{label} · {dataset} · "
            f"{request.app.state.benchmark_label_set_id} · {timestamp}"
        )[:120]

    async def run_benchmark(request: Request) -> None:
        state: BenchmarkState = request.app.state.benchmark
        predictor = request.app.state.predictor
        rows = request.app.state.benchmark_rows
        human_by_id = request.app.state.benchmark_human
        routing_mode = request.app.state.benchmark_routing_mode
        question = request.app.state.benchmark_question
        criteria = request.app.state.benchmark_criteria
        loop = asyncio.get_running_loop()
        try:
            for article in rows:
                if state.has_prediction(article["article_id"]):
                    continue
                state.begin_item(article)
                started = time.perf_counter()
                if registered_model_by_id(gguf_models_root(), routing_mode) is not None:
                    runtime = generative_runtime(request, predictor)
                    result = await loop.run_in_executor(
                        request.app.state.inference_pool,
                        lambda article=article: runtime.predict(
                            article["article"], article["entity_name"]
                        ),
                    )
                else:
                    result = await loop.run_in_executor(
                        request.app.state.inference_pool,
                        lambda article=article: predictor.predict(
                            {"article": article["article"], "entity_name": article["entity_name"]},
                            _questions(article["entity_name"], question, criteria),
                            model=None if routing_mode == "auto" else routing_mode,
                        ),
                    )
                prediction = _parse_result(result, article["entity_name"], review_threshold).model_dump()
                inference_seconds = time.perf_counter() - started
                routing = prediction.get("routing") or {}
                usage = routing.get("usage") or {}
                request.app.state.benchmark_usage["input_tokens"] += int(
                    usage.get("input_tokens", 0)
                )
                request.app.state.benchmark_usage["output_tokens"] += int(
                    usage.get("output_tokens", 0)
                )
                request.app.state.benchmark_usage["cost_usd"] += float(
                    routing.get("cost_usd", 0.0)
                )
                request.app.state.benchmark_usage["truncated_examples"] += int(
                    bool((routing.get("input") or {}).get("article_truncated"))
                )
                truths = {
                    "gold": request.app.state.benchmark_gold.get(article["article_id"]),
                    "human": human_by_id.get(article["article_id"]),
                }
                state.record(article, prediction, truths, inference_seconds)
                output_row = {
                    "article_id": article["article_id"],
                    "entity_name": article["entity_name"],
                    "decision": prediction["decision"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"],
                    "elapsed_seconds": inference_seconds,
                    "routing": prediction.get("routing"),
                    "human_label": truths["human"].get("label") if truths["human"] else None,
                    "gold_label": truths["gold"].get("label") if truths["gold"] else None,
                }
                with output_path.open("a", encoding="utf-8", newline="\n") as output:
                    output.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if state.should_pause():
                    state.paused()
                    persist_active_benchmark(state, request.app.state)
                    return
            state.finish()
            persist_active_benchmark(state, request.app.state)
            benchmark_save(
                BenchmarkSaveRequest(name=automatic_run_name(request)), request
            )
        except asyncio.CancelledError:
            state.paused()
            persist_active_benchmark(state, request.app.state)
            raise
        except Exception as exc:
            state.fail(exc)
            persist_active_benchmark(state, request.app.state)
            if state.snapshot()["completed"]:
                benchmark_save(
                    BenchmarkSaveRequest(name=automatic_run_name(request)), request
                )

    def current_gold_path(request: Request) -> Path:
        return request.app.state.benchmark_gold_path

    def label_set_options() -> list[dict[str, Any]]:
        return [
            {
                "id": item["id"],
                "label": item["label"],
                "rows": len(load_gold_labels(item["path"])),
                "built_in": item["built_in"],
                "deletable": not item["built_in"],
            }
            for item in label_set_catalog().values()
        ]

    def ground_truth_options() -> list[dict[str, Any]]:
        if benchmark_corpus is not None or benchmark_gold is not None:
            return [
                {
                    **item,
                    "dataset_id": active_dataset.id if active_dataset else None,
                    "dataset_label": active_dataset.name if active_dataset else "External corpus",
                }
                for item in label_set_options()
            ]
        options: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for bundle in benchmark_dataset_bundles.values():
            user_managed_ids = {
                row.get("id")
                for row in bundle.manifest.get("annotations", [])
                if isinstance(row, dict) and row.get("user_managed") is True
            }
            for annotation in bundle.annotations:
                if not annotation.path.is_file():
                    continue
                seen.add((bundle.id, annotation.id))
                options.append(
                    {
                        "id": annotation.id,
                        "label": annotation.name,
                        "rows": len(load_gold_labels(annotation.path)),
                        "built_in": annotation.id not in user_managed_ids,
                        "deletable": annotation.id in user_managed_ids,
                        "dataset_id": bundle.id,
                        "dataset_label": bundle.name,
                    }
                )
        if active_dataset is not None:
            for item in label_set_options():
                key = (active_dataset.id, item["id"])
                if key not in seen:
                    options.append(
                        {
                            **item,
                            "dataset_id": active_dataset.id,
                            "dataset_label": active_dataset.name,
                        }
                    )
        return options

    def activate_label_set(request: Request, label_set_id: str) -> dict[str, Any]:
        state: BenchmarkState = request.app.state.benchmark
        if state.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409, detail="reset the current benchmark before changing labels"
            )
        selected = label_set_catalog().get(label_set_id)
        if selected is None:
            raise HTTPException(status_code=404, detail="label set not found")
        rows, labels, human_labels, audit = load_configured_benchmark(selected["path"])
        audit["label_set"] = {"id": selected["id"], "label": selected["label"]}
        state.reset()
        state.update_audit(audit)
        request.app.state.benchmark_rows = rows
        request.app.state.benchmark_gold = labels
        request.app.state.benchmark_human = human_labels
        request.app.state.benchmark_gold_path = selected["path"]
        request.app.state.benchmark_gold_mtime = selected["path"].stat().st_mtime_ns
        request.app.state.benchmark_label_set_id = selected["id"]
        return {
            "active_label_set": selected["id"],
            "active_dataset": active_dataset.id if active_dataset else None,
            "label_sets": ground_truth_options(),
            "audit": audit,
        }

    def refresh_gold_labels(request: Request) -> None:
        selected_path = current_gold_path(request)
        try:
            modified = selected_path.stat().st_mtime_ns
        except OSError:
            return
        if modified == request.app.state.benchmark_gold_mtime:
            return
        labels = load_gold_labels(selected_path)
        request.app.state.benchmark_gold = labels
        request.app.state.benchmark_gold_mtime = modified
        request.app.state.benchmark.refresh_gold(labels)

    @app.get("/v1/benchmark/label-sets")
    def benchmark_label_sets(request: Request) -> dict[str, Any]:
        return {
            "dataset": (
                {"id": active_dataset.id, "name": active_dataset.name}
                if active_dataset
                else None
            ),
            "active_label_set": request.app.state.benchmark_label_set_id,
            "label_sets": ground_truth_options(),
        }

    @app.get("/v1/benchmark/datasets")
    def benchmark_datasets() -> dict[str, Any]:
        return {
            "datasets": [
                {
                    "id": bundle.id,
                    "label": bundle.name,
                    "purpose": bundle.manifest.get(
                        "purpose", "training-and-evaluation"
                    ),
                    "records": bundle.manifest.get("corpus", {}).get("records"),
                }
                for bundle in benchmark_dataset_bundles.values()
            ],
            "active_dataset": active_dataset.id if active_dataset else None,
        }

    @app.post("/v1/benchmark/datasets/activate")
    async def benchmark_activate_dataset(
        payload: BenchmarkDatasetRequest, request: Request
    ) -> dict[str, Any]:
        state: BenchmarkState = request.app.state.benchmark
        if state.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409, detail="reset the current benchmark before changing datasets"
            )
        try:
            activate_dataset_paths(payload.dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        activated = activate_label_set(request, active_label_set_id)
        return {**benchmark_datasets(), **activated}

    @app.post("/v1/benchmark/label-sets/activate")
    def benchmark_activate_label_set(
        payload: BenchmarkLabelSetRequest, request: Request
    ) -> dict[str, Any]:
        if payload.dataset_id and (
            active_dataset is None or payload.dataset_id != active_dataset.id
        ):
            bundle = benchmark_dataset_bundles.get(payload.dataset_id)
            if bundle is None or payload.label_set_id not in {
                annotation.id for annotation in bundle.annotations
            }:
                raise HTTPException(status_code=404, detail="ground truth not found")
            try:
                activate_dataset_paths(payload.dataset_id)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return activate_label_set(request, payload.label_set_id)

    @app.post("/v1/benchmark/label-sets/upload")
    def benchmark_upload_label_set(
        payload: BenchmarkLabelSetUploadRequest, request: Request
    ) -> dict[str, Any]:
        nonlocal active_dataset
        state: BenchmarkState = request.app.state.benchmark
        if state.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409, detail="reset the current benchmark before uploading labels"
            )
        parsed_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        expected_names = {1: "positive", 2: "negative"}
        try:
            for line_number, line in enumerate(payload.content.splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = row.get("article_id")
                label = row.get("label")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError(f"line {line_number} has no article_id")
                if identifier in seen:
                    raise ValueError(f"line {line_number} duplicates article_id {identifier}")
                if type(label) is not int or label not in expected_names:
                    raise ValueError(f"line {line_number} has an invalid label")
                if row.get("label_name") is not None and not isinstance(row["label_name"], str):
                    raise ValueError(f"line {line_number} has an invalid label_name")
                seen.add(identifier)
                parsed_rows.append({**row, "label_name": expected_names[label]})
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid label JSONL: {exc}") from exc
        if not parsed_rows:
            raise HTTPException(status_code=422, detail="label file contains no usable rows")
        corpus_ids = {row["article_id"] for row in read_jsonl(corpus_path)}
        unknown_ids = seen - corpus_ids
        if unknown_ids:
            raise HTTPException(
                status_code=422,
                detail=f"label file contains {len(unknown_ids)} article IDs outside the corpus",
            )
        base_id = re.sub(r"[^a-z0-9]+", "-", payload.name.casefold()).strip("-")
        if not base_id:
            raise HTTPException(status_code=422, detail="label set name has no usable characters")
        label_set_id = base_id[:72]
        existing = label_set_catalog()
        suffix = 2
        while label_set_id in existing:
            label_set_id = f"{base_id[:68]}-{suffix}"
            suffix += 1
        label_sets_root.mkdir(parents=True, exist_ok=True)
        destination = label_sets_root / f"{label_set_id}.jsonl"
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            for row in parsed_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if active_dataset is not None and destination.is_relative_to(active_dataset.root):
            try:
                active_dataset = register_annotation_set(
                    active_dataset,
                    label_set_id,
                    payload.name.strip(),
                    destination,
                    len(parsed_rows),
                )
            except (OSError, ValueError) as exc:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"annotation manifest could not be updated: {exc}",
                ) from exc
        return activate_label_set(request, label_set_id)

    @app.post("/v1/benchmark/label-sets/delete")
    def benchmark_delete_label_set(
        payload: BenchmarkLabelSetRequest, request: Request
    ) -> dict[str, Any]:
        nonlocal active_dataset
        state: BenchmarkState = request.app.state.benchmark
        if state.snapshot()["status"] in {"running", "paused"}:
            raise HTTPException(
                status_code=409, detail="reset the current benchmark before deleting labels"
            )
        dataset_id = payload.dataset_id
        if (
            dataset_id is None
            and benchmark_corpus is None
            and benchmark_gold is None
            and active_dataset is not None
        ):
            dataset_id = active_dataset.id
        bundle = dataset_bundles.get(dataset_id or "")
        deleting_active = (
            request.app.state.benchmark_label_set_id == payload.label_set_id
            and (
                payload.dataset_id is None
                or payload.dataset_id == (active_dataset.id if active_dataset else None)
            )
        )
        if bundle is None:
            selected = label_set_catalog().get(payload.label_set_id)
            if selected is None:
                raise HTTPException(status_code=404, detail="label set not found")
            if selected["built_in"]:
                raise HTTPException(
                    status_code=403, detail="built-in label sets cannot be deleted"
                )
            if deleting_active and not any(
                identifier != payload.label_set_id for identifier in label_set_catalog()
            ):
                raise HTTPException(
                    status_code=409, detail="no fallback label set is available"
                )
            try:
                selected["path"].unlink()
            except OSError as exc:
                raise HTTPException(
                    status_code=500, detail="label set could not be deleted"
                ) from exc
        else:
            try:
                updated = remove_annotation_set(bundle, payload.label_set_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            dataset_bundles[updated.id] = updated
            if updated.id in benchmark_dataset_bundles:
                benchmark_dataset_bundles[updated.id] = updated
            if active_dataset is not None and active_dataset.id == updated.id:
                active_dataset = updated
        dataset_presets[:] = [
            preset
            for preset in dataset_presets
            if not (
                preset.get("label_set_id") == payload.label_set_id
                and (
                    payload.dataset_id is None
                    or preset.get("dataset_id") == payload.dataset_id
                )
            )
        ]
        if deleting_active:
            fallback = next(iter(label_set_catalog()), "")
            if not fallback:
                raise HTTPException(status_code=409, detail="no fallback label set is available")
            activate_label_set(request, fallback)
        return {
            "deleted": payload.label_set_id,
            "active_label_set": request.app.state.benchmark_label_set_id,
            "label_sets": ground_truth_options(),
        }

    @app.get("/v1/benchmark")
    def benchmark_status(request: Request, series_after: int = 0) -> dict[str, Any]:
        state: BenchmarkState = request.app.state.benchmark
        snapshot = state.snapshot(max(0, series_after))
        if "blind_articles" not in snapshot["audit"]:
            selected_path = current_gold_path(request)
            try:
                _, _, _, audit = load_configured_benchmark(selected_path)
                selected = label_set_catalog().get(request.app.state.benchmark_label_set_id)
                if selected:
                    audit["label_set"] = {"id": selected["id"], "label": selected["label"]}
                state.update_audit(audit)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                state.update_audit({
                    "usable_gold_rows": 0,
                    "corpus_path": str(corpus_path),
                    "gold_path": str(selected_path),
                    "error": str(exc),
                })
        refresh_gold_labels(request)
        snapshot = state.snapshot(max(0, series_after))
        snapshot["usage"] = dict(request.app.state.benchmark_usage)
        snapshot["saved_run"] = request.app.state.benchmark_saved_run
        snapshot["model_id"] = request.app.state.benchmark_routing_mode
        snapshot["question"] = request.app.state.benchmark_question
        snapshot["criteria"] = request.app.state.benchmark_criteria
        return snapshot

    @app.get("/v1/benchmark/results")
    def benchmark_results(
        request: Request,
        truth_label: int = Query(ge=1, le=2),
        prediction_label: int = Query(ge=1, le=2),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        refresh_gold_labels(request)
        state: BenchmarkState = request.app.state.benchmark
        return state.matching_results(
            request.app.state.benchmark_rows,
            request.app.state.benchmark_gold,
            truth_label,
            prediction_label,
            offset,
            limit,
        )

    @app.post("/v1/benchmark/start")
    async def benchmark_start(payload: BenchmarkStartRequest, request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request)
        generative_ids = {model.id for model in registered_model_catalog(gguf_models_root())}
        if predictor is None and payload.routing_mode not in generative_ids:
            raise HTTPException(status_code=503, detail="Laya model is not ready")
        if predictor is not None:
            _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
        available_models = (
            _available_model_ids(predictor) if predictor is not None else set()
        )
        available_models.update(generative_ids)
        if payload.routing_mode != "auto" and payload.routing_mode not in available_models:
            raise HTTPException(status_code=422, detail="Unknown model selection")
        if (
            payload.routing_mode in generative_ids
            and payload.routing_mode != request.app.state.benchmark_active_model
        ):
            raise HTTPException(
                status_code=409, detail="activate the selected model before starting"
            )
        state: BenchmarkState = request.app.state.benchmark
        current = state.snapshot()
        task = request.app.state.benchmark_task
        if task and not task.done():
            raise HTTPException(status_code=409, detail="benchmark is already running")

        if (
            current["status"] in {"paused", "error"}
            and request.app.state.benchmark_rows
        ):
            current = restore_active_ledger(request, current)

        if (
            current["status"] in {"paused", "error"}
            and current["completed"] < current["total"]
            and request.app.state.benchmark_rows
        ):
            if payload.routing_mode != request.app.state.benchmark_routing_mode:
                raise HTTPException(
                    status_code=409,
                    detail="reset the interrupted benchmark before changing models",
                )
            state.start(current["total"], current["audit"], resume=True)
        else:
            if payload.criteria and {
                criterion.decision for criterion in payload.criteria
            } != {"negative", "positive"}:
                raise HTTPException(
                    status_code=422,
                    detail="Prompt criteria must include negative and positive decisions",
                )
            model_prompt = _model_prompt(predictor, payload.routing_mode)
            request.app.state.benchmark_question = (
                payload.question or model_prompt["question"]
            )
            request.app.state.benchmark_criteria = (
                [criterion.model_dump() for criterion in payload.criteria]
                if payload.criteria else model_prompt["criteria"]
            )
            selected_path = current_gold_path(request)
            rows, gold_by_id, human_by_id, audit = load_configured_benchmark(
                selected_path
            )
            selected = label_set_catalog().get(request.app.state.benchmark_label_set_id)
            if selected:
                audit["label_set"] = {"id": selected["id"], "label": selected["label"]}
            if payload.limit is not None:
                rows = rows[: payload.limit]
            request.app.state.benchmark_rows = rows
            request.app.state.benchmark_gold = gold_by_id
            request.app.state.benchmark_human = human_by_id
            request.app.state.benchmark_gold_mtime = selected_path.stat().st_mtime_ns
            request.app.state.benchmark_routing_mode = payload.routing_mode
            request.app.state.benchmark_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "truncated_examples": 0,
            }
            request.app.state.benchmark_saved_run = None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            audit["output_path"] = str(output_path)
            state.start(len(rows), audit)
            _write_json(active_metadata_path, {
                "total": len(rows),
                "completed": 0,
                "status": "running",
                "routing_mode": payload.routing_mode,
                "label_set_id": request.app.state.benchmark_label_set_id,
                "question": request.app.state.benchmark_question,
                "criteria": request.app.state.benchmark_criteria,
                "elapsed_seconds": 0.0,
            })
        request.app.state.benchmark_task = asyncio.create_task(run_benchmark(request))
        snapshot = state.snapshot()
        snapshot["question"] = request.app.state.benchmark_question
        snapshot["criteria"] = request.app.state.benchmark_criteria
        return snapshot

    @app.post("/v1/benchmark/pause")
    def benchmark_pause(request: Request) -> dict[str, Any]:
        state: BenchmarkState = request.app.state.benchmark
        state.request_pause()
        return state.snapshot()

    @app.post("/v1/benchmark/reset")
    async def benchmark_reset(request: Request) -> dict[str, Any]:
        task = request.app.state.benchmark_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        request.app.state.benchmark_rows = []
        request.app.state.benchmark_gold = {}
        request.app.state.benchmark_human = {}
        default_prompt = validate_prompt(None)
        request.app.state.benchmark_question = default_prompt["question"]
        request.app.state.benchmark_criteria = default_prompt["criteria"]
        request.app.state.benchmark_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "truncated_examples": 0,
        }
        request.app.state.benchmark_saved_run = None
        request.app.state.benchmark.reset()
        output_path.write_text("", encoding="utf-8")
        persist_active_benchmark(request.app.state.benchmark, request.app.state)
        return request.app.state.benchmark.snapshot()

    @app.get("/v1/benchmark/runs")
    def benchmark_runs() -> dict[str, Any]:
        return {"runs": _saved_runs(runs_dir)}

    @app.post("/v1/benchmark/runs/delete")
    def benchmark_delete_run(
        payload: BenchmarkLoadRequest, request: Request
    ) -> dict[str, Any]:
        task = request.app.state.benchmark_task
        if task and not task.done():
            raise HTTPException(status_code=409, detail="pause the active run before deleting")
        paths = [
            runs_dir / f"{payload.run_id}.json",
            runs_dir / f"{payload.run_id}.jsonl",
        ]
        if not any(path.is_file() for path in paths):
            raise HTTPException(status_code=404, detail="saved run not found")
        try:
            for path in paths:
                path.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail="saved run could not be deleted"
            ) from exc
        saved = request.app.state.benchmark_saved_run
        if saved and saved.get("run_id") == payload.run_id:
            request.app.state.benchmark_saved_run = None
        return {"deleted": payload.run_id, "runs": _saved_runs(runs_dir)}

    @app.post("/v1/benchmark/runs/save")
    def benchmark_save(
        payload: BenchmarkSaveRequest, request: Request
    ) -> dict[str, Any]:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="run name cannot be blank")
        snapshot = request.app.state.benchmark.snapshot()
        if snapshot["completed"] == 0 or not output_path.is_file():
            raise HTTPException(status_code=409, detail="there are no results to save")
        rows = read_jsonl(output_path)[: snapshot["completed"]]
        if len(rows) != snapshot["completed"]:
            raise HTTPException(status_code=409, detail="prediction ledger is incomplete")
        created_at = datetime.now(timezone.utc)
        runs_dir.mkdir(parents=True, exist_ok=True)
        base_run_id = _run_id(name, created_at)
        run_id = base_run_id
        suffix = 2
        while (runs_dir / f"{run_id}.json").exists() or (
            runs_dir / f"{run_id}.jsonl"
        ).exists():
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1
        ledger_path = runs_dir / f"{run_id}.jsonl"
        with ledger_path.open("x", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata = {
            "run_id": run_id,
            "name": name,
            "created_at": created_at.isoformat(),
            "status": snapshot["status"],
            "completed": snapshot["completed"],
            "total": snapshot["total"],
            "routing_mode": request.app.state.benchmark_routing_mode,
            "dataset_id": active_dataset.id if active_dataset else None,
            "label_set_id": request.app.state.benchmark_label_set_id,
            "corpus_sha256": _file_sha256(corpus_path),
            "labels_sha256": _file_sha256(current_gold_path(request)),
            "timing": snapshot["timing"],
            "usage": request.app.state.benchmark_usage,
            "metrics": snapshot["metrics"],
            "confusion_matrices": snapshot["confusion_matrices"],
            "question": request.app.state.benchmark_question,
            "criteria": request.app.state.benchmark_criteria,
        }
        _write_json(runs_dir / f"{run_id}.json", metadata)
        request.app.state.benchmark_saved_run = metadata
        return {"saved": metadata, "runs": _saved_runs(runs_dir)}

    @app.post("/v1/benchmark/runs/load")
    async def benchmark_load(
        payload: BenchmarkLoadRequest, request: Request
    ) -> dict[str, Any]:
        task = request.app.state.benchmark_task
        if task and not task.done():
            raise HTTPException(status_code=409, detail="pause the active run before loading")
        metadata_path = runs_dir / f"{payload.run_id}.json"
        ledger_path = runs_dir / f"{payload.run_id}.jsonl"
        if not metadata_path.is_file() or not ledger_path.is_file():
            raise HTTPException(status_code=404, detail="saved run not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_dataset_id = metadata.get("dataset_id")
        if (
            saved_dataset_id
            and active_dataset is not None
            and saved_dataset_id != active_dataset.id
        ):
            raise HTTPException(
                status_code=422,
                detail=f"saved run dataset {saved_dataset_id!r} is not active",
            )
        saved_corpus_sha256 = metadata.get("corpus_sha256")
        if saved_corpus_sha256 and saved_corpus_sha256 != _file_sha256(corpus_path):
            raise HTTPException(
                status_code=409,
                detail="saved run corpus fingerprint does not match the active dataset",
            )
        restored_set_id = metadata.get(
            "label_set_id", request.app.state.benchmark_label_set_id
        )
        restored_set = label_set_catalog().get(restored_set_id)
        if restored_set is None:
            raise HTTPException(
                status_code=422,
                detail=f"saved run label set {restored_set_id!r} is unavailable",
            )
        saved_labels_sha256 = metadata.get("labels_sha256")
        if saved_labels_sha256 and saved_labels_sha256 != _file_sha256(
            restored_set["path"]
        ):
            raise HTTPException(
                status_code=409,
                detail="saved run label fingerprint does not match the active annotation set",
            )
        articles, gold_labels, human_labels, audit = load_configured_benchmark(
            restored_set["path"]
        )
        audit["label_set"] = {
            "id": restored_set["id"],
            "label": restored_set["label"],
        }
        audit["loaded_run"] = {
            "run_id": metadata["run_id"],
            "name": metadata["name"],
            "created_at": metadata["created_at"],
        }
        request.app.state.benchmark.restore(
            read_jsonl(ledger_path),
            articles,
            gold_labels,
            human_labels,
            audit,
            total=int(metadata["total"]),
            elapsed_seconds=metadata.get("timing", {}).get("elapsed_seconds"),
        )
        request.app.state.benchmark_rows = articles
        request.app.state.benchmark_gold = gold_labels
        request.app.state.benchmark_human = human_labels
        request.app.state.benchmark_gold_path = restored_set["path"]
        request.app.state.benchmark_gold_mtime = restored_set["path"].stat().st_mtime_ns
        request.app.state.benchmark_label_set_id = restored_set_id
        request.app.state.benchmark_routing_mode = metadata.get("routing_mode", "auto")
        request.app.state.benchmark_usage = metadata.get("usage", {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "truncated_examples": 0,
        })
        request.app.state.benchmark_saved_run = metadata
        default_prompt = validate_prompt(None)
        request.app.state.benchmark_question = metadata.get(
            "question", default_prompt["question"]
        )
        request.app.state.benchmark_criteria = metadata.get(
            "criteria", default_prompt["criteria"]
        )
        snapshot = request.app.state.benchmark.snapshot()
        snapshot["question"] = request.app.state.benchmark_question
        snapshot["criteria"] = request.app.state.benchmark_criteria
        return snapshot

    def fine_tuning_snapshot(request: Request, logs_after: int = 0) -> dict[str, Any]:
        snapshot = request.app.state.fine_tuning.snapshot(logs_after)
        presets = []
        for preset in fine_tuning_defaults["dataset_presets"]:
            prepared_partition = None
            try:
                prepared_manifest = json.loads(
                    (Path(preset["prepared_dir"]) / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                prepared_partition = prepared_manifest.get("partition")
            except (OSError, json.JSONDecodeError, TypeError):
                pass
            presets.append({**preset, "prepared_partition": prepared_partition})
        snapshot["defaults"] = {
            **fine_tuning_defaults,
            "dataset_presets": presets,
            "model_presets": fine_tuning_model_presets(),
        }
        return snapshot

    def require_fine_tuning() -> None:
        if not fine_tuning_enabled:
            raise HTTPException(
                status_code=403,
                detail="Fine-tuning is disabled. Set LAYA_ENABLE_FINE_TUNING=true to enable it.",
            )

    def release_inference_for_training(request: Request) -> None:
        if request.app.state.benchmark.snapshot()["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="pause or finish the active benchmark before fine-tuning",
            )
        training_status = request.app.state.fine_tuning.snapshot()["status"]
        if training_status in {"running", "stopping"}:
            raise HTTPException(status_code=409, detail="a fine-tuning job is already running")
        predictor = resolve_predictor(request, wait=0.05)
        if predictor is None and not request.app.state.warmup_future.done():
            raise HTTPException(
                status_code=409,
                detail="wait for model warm-up to finish before fine-tuning",
            )
        runtime = request.app.state.generative_runtime
        try:
            if runtime is not None:
                runtime.close()
                request.app.state.generative_runtime = None
            if predictor is not None:
                unload = getattr(predictor, "unload", None)
                if callable(unload):
                    unload()
            request.app.state.benchmark_active_model = "auto"
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"could not release inference GPU resources: {exc}",
            ) from exc

    @app.get("/v1/fine-tuning")
    def fine_tuning_status(request: Request, logs_after: int = 0) -> dict[str, Any]:
        return fine_tuning_snapshot(request, max(0, logs_after))

    @app.get("/v1/fine-tuning/reports")
    def fine_tuning_reports() -> dict[str, Any]:
        return {"reports": _training_reports(fine_tuned_models_root)}

    @app.post("/v1/fine-tuning/reports/load")
    def fine_tuning_load_report(payload: FineTuningLoadRequest) -> dict[str, Any]:
        try:
            snapshot = _historical_training_snapshot(
                fine_tuned_models_root, payload.run_id
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail=f"invalid training report: {exc}"
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail="training report could not be read"
            ) from exc
        snapshot["defaults"] = {
            **fine_tuning_defaults,
            "model_presets": fine_tuning_model_presets(),
        }
        return snapshot

    @app.post("/v1/fine-tuning/reports/delete")
    def fine_tuning_delete_report(
        payload: FineTuningLoadRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning()
        if request.app.state.fine_tuning.snapshot()["status"] in {"running", "stopping"}:
            raise HTTPException(
                status_code=409, detail="stop fine-tuning before deleting training output"
            )
        output_dir = (fine_tuned_models_root / payload.run_id).resolve()
        if (
            not output_dir.is_relative_to(fine_tuned_models_root)
            or not (output_dir / "training_report.json").is_file()
        ):
            raise HTTPException(status_code=404, detail="completed training not found")
        model_id = next(
            (
                identifier
                for identifier, path in _fine_tuned_model_paths(
                    fine_tuned_models_root
                ).items()
                if path == output_dir
            ),
            None,
        )
        if model_id and request.app.state.benchmark_active_model == model_id:
            raise HTTPException(status_code=409, detail="the active model cannot be deleted")
        try:
            shutil.rmtree(output_dir)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail="training output could not be deleted"
            ) from exc
        predictor = resolve_predictor(request, wait=0.05)
        if predictor is not None:
            _refresh_fine_tuned_models(predictor, fine_tuned_models_root)
        return {
            "deleted": payload.run_id,
            "reports": _training_reports(fine_tuned_models_root),
            "model_presets": fine_tuning_model_presets(),
        }

    @app.post("/v1/fine-tuning/prepare")
    def fine_tuning_prepare(
        payload: FineTuningPrepareRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning()
        if (
            payload.partition_strategy == "group_k_fold"
            and payload.fold_index >= payload.fold_count
        ):
            raise HTTPException(
                status_code=422,
                detail="Selected fold must be lower than the number of folds",
            )
        _require_allowed_paths(
            {
                "corpus": payload.corpus,
                "labels": payload.labels,
                "model_dir": payload.model_dir,
                "output_dir": payload.output_dir,
            },
            fine_tuning_roots,
        )
        _require_input_files({"corpus": payload.corpus, "labels": payload.labels})
        prompt_file: Path | None = None
        if payload.dataset_id is not None:
            bundle = dataset_bundles.get(payload.dataset_id)
            if bundle is None:
                raise HTTPException(status_code=422, detail="Unknown training dataset")
            if payload.corpus.resolve() != bundle.corpus:
                raise HTTPException(
                    status_code=422,
                    detail="Selected dataset does not own the submitted corpus",
                )
            prompt_file = bundle.root / "dataset.json"
        command = [
            sys.executable,
            str(fine_tuning_script),
            "prepare",
            "--label-source",
            "external_labels",
            "--corpus",
            str(payload.corpus),
            "--labels",
            str(payload.labels),
            "--model-dir",
            str(payload.model_dir),
            "--output-dir",
            str(payload.output_dir),
            "--seed",
            str(payload.seed),
            "--partition-strategy",
            payload.partition_strategy,
            "--fold-count",
            str(payload.fold_count),
            "--fold-index",
            str(payload.fold_index),
        ]
        if prompt_file is not None:
            command.extend(["--prompt-file", str(prompt_file)])
        try:
            request.app.state.fine_tuning.start(command, "preparation", project_root)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return fine_tuning_snapshot(request)

    @app.post("/v1/fine-tuning/prompt")
    def fine_tuning_prompt(
        payload: FineTuningPromptRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning()
        bundle = dataset_bundles.get(payload.dataset_id)
        if bundle is None:
            raise HTTPException(status_code=422, detail="Unknown training dataset")
        try:
            updated = update_dataset_prompt(
                bundle,
                {
                    "question": payload.question,
                    "criteria": [criterion.model_dump() for criterion in payload.criteria],
                },
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        dataset_bundles[payload.dataset_id] = updated
        for preset in dataset_presets:
            if preset.get("dataset_id") == payload.dataset_id:
                preset["prompt"] = updated.prompt
        return {"dataset_id": updated.id, "prompt": updated.prompt}

    @app.post("/v1/fine-tuning/train")
    def fine_tuning_train(
        payload: FineTuningTrainRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning()
        training = _resolve_training_parameters(payload)
        if (payload.huggingface_repo_id is None) != (payload.huggingface_token is None):
            raise HTTPException(
                status_code=422,
                detail="Hugging Face repository and token must be provided together.",
            )
        _require_allowed_paths(
            {
                "prepared_dir": payload.prepared_dir,
                "model_dir": payload.model_dir,
                "output_dir": payload.output_dir,
            },
            fine_tuning_roots,
        )
        required_prepared_files = (
            "train.pt",
            "calibration.pt",
            "manifest.json",
            "split-assignments.jsonl",
        )
        missing = [
            name
            for name in required_prepared_files
            if not (payload.prepared_dir / name).is_file()
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Prepared dataset is incomplete at {payload.prepared_dir}; "
                    f"missing {', '.join(missing)}. Run Prepare dataset first."
                ),
            )
        try:
            prepared_manifest = json.loads(
                (payload.prepared_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Prepared dataset manifest is invalid. Run Prepare dataset again.",
            ) from exc
        prepared_partition = (prepared_manifest.get("partition") or {}).get("strategy")
        if prepared_partition != payload.partition_strategy:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Prepared data uses {prepared_partition or 'an unknown partition'}, "
                    f"but the form selects {payload.partition_strategy}. "
                    "Run Prepare dataset again before training."
                ),
            )
        if payload.gpu_count > 1:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={payload.gpu_count}",
                str(fine_tuning_script),
                "train",
            ]
        else:
            command = [sys.executable, str(fine_tuning_script), "train"]
        command.extend([
            "--prepared-dir", str(payload.prepared_dir),
            "--model-dir", str(payload.model_dir),
            "--output-dir", str(payload.output_dir),
            "--strategy", payload.strategy,
            "--epochs", str(training["epochs"]),
            "--batch-size", str(payload.batch_size),
            "--eval-batch-size", str(payload.eval_batch_size),
            "--gradient-accumulation", str(payload.gradient_accumulation),
            "--group-size", str(training["group_size"]),
            "--rl-weight", str(training["rl_weight"]),
            "--encoder-lr", str(training["encoder_lr"]),
            "--head-lr", str(training["head_lr"]),
            "--weight-decay", str(training["weight_decay"]),
            "--sigma-start", str(training["sigma_start"]),
            "--sigma-end", str(training["sigma_end"]),
            "--encoder-warmup-epochs", str(training["encoder_warmup_epochs"]),
            "--lr-warmup-ratio", str(training["lr_warmup_ratio"]),
            "--label-smoothing", str(training["label_smoothing"]),
            "--seed", str(payload.seed),
            "--log-every", str(payload.log_every),
        ])
        process_env = None
        if payload.huggingface_repo_id and payload.huggingface_token:
            command.extend(["--hub-repo-id", payload.huggingface_repo_id])
            if payload.huggingface_private:
                command.append("--hub-private")
            process_env = {
                "HF_TOKEN": payload.huggingface_token.get_secret_value(),
                "HF_HUB_OFFLINE": "0",
            }
        release_inference_for_training(request)
        try:
            request.app.state.fine_tuning.start(
                command, "training", project_root, env=process_env
            )
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return fine_tuning_snapshot(request)

    @app.post("/v1/fine-tuning/stop")
    def fine_tuning_stop(request: Request) -> dict[str, Any]:
        require_fine_tuning()
        request.app.state.fine_tuning.stop()
        return fine_tuning_snapshot(request)

    return app


app = create_app()
