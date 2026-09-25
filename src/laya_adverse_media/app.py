from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
import re
import secrets
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .benchmark import BenchmarkState, load_benchmark, load_gpt_labels, read_jsonl


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


class AdverseMediaRequest(BaseModel):
    article: str = Field(min_length=1, max_length=100_000)
    entity_name: str = Field(min_length=1, max_length=500)
    routing_mode: Literal["auto", "english", "multilingual"] = "auto"


class AdverseMediaResponse(BaseModel):
    entity_name: str
    decision: Literal["negative", "positive"]
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    needs_review: bool
    routing: dict[str, Any] | None = None


class BenchmarkStartRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=100_000)
    routing_mode: Literal["auto", "english", "multilingual"] = "auto"


class BenchmarkSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class BenchmarkLoadRequest(BaseModel):
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,159}$")


class FineTuningPrepareRequest(BaseModel):
    label_source: Literal["azure", "human"] = "azure"
    data_dir: str = Field(default="data/source", min_length=1, max_length=4096)
    corpus: str = Field(
        default="artifacts/independent-annotations/blind_articles.jsonl",
        min_length=1,
        max_length=4096,
    )
    annotations: str = Field(
        default="artifacts/independent-annotations/batches/azure.annotations.jsonl",
        min_length=1,
        max_length=4096,
    )
    model_dir: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(
        default="artifacts/adverse-media-data-azure", min_length=1, max_length=4096
    )
    seed: int = Field(default=20260923, ge=0, le=2_147_483_647)


class FineTuningTrainRequest(BaseModel):
    prepared_dir: str = Field(
        default="artifacts/adverse-media-data-azure", min_length=1, max_length=4096
    )
    model_dir: str = Field(min_length=1, max_length=4096)
    output_dir: str = Field(
        default="models/laya-adverse-media-azure", min_length=1, max_length=4096
    )
    resume_from: str | None = Field(default=None, max_length=4096)
    gpu_count: int = Field(default=1, ge=1, le=16)
    epochs: int = Field(default=4, ge=1, le=10_000)
    batch_size: int = Field(default=4, ge=1, le=100_000)
    eval_batch_size: int = Field(default=16, ge=1, le=100_000)
    gradient_accumulation: int = Field(default=8, ge=1, le=100_000)
    group_size: int = Field(default=4, ge=1, le=1_000)
    encoder_lr: float = Field(default=2.5e-5, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    sigma_start: float = Field(default=0.4, gt=0)
    sigma_end: float = Field(default=0.1, gt=0)
    seed: int = Field(default=20260923, ge=0, le=2_147_483_647)
    log_every: int = Field(default=50, ge=1, le=100_000)


class FineTuningJob:
    def __init__(self) -> None:
        self.status = "idle"
        self.phase: str | None = None
        self.pid: int | None = None
        self.command: list[str] = []
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.stop_requested = False
        self.process: asyncio.subprocess.Process | None = None
        self.logs: deque[dict[str, Any]] = deque(maxlen=2_000)
        self.next_log_index = 1
        self.progress: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self.status in {"running", "stopping"}

    def append_log(self, message: str) -> None:
        progress = re.search(
            r"epoch=(\d+)/(\d+) batch=(\d+) loss=([\d.]+) sigma=([\d.]+)",
            message,
        )
        if progress:
            self.progress = {
                "epoch": int(progress.group(1)),
                "epochs": int(progress.group(2)),
                "batch": int(progress.group(3)),
                "loss": float(progress.group(4)),
                "sigma": float(progress.group(5)),
            }
        self.logs.append(
            {
                "index": self.next_log_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": message.rstrip(),
            }
        )
        self.next_log_index += 1

    def start(self, phase: str, command: Sequence[str]) -> None:
        self.status = "running"
        self.phase = phase
        self.pid = None
        self.command = list(command)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at = None
        self.exit_code = None
        self.error = None
        self.stop_requested = False
        self.logs.clear()
        self.next_log_index = 1
        self.progress = {}

    def snapshot(self, logs_after: int = 0) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "pid": self.pid,
            "command": self.command,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "progress": self.progress,
            "logs": [row for row in self.logs if row["index"] > logs_after],
            "latest_log_index": self.next_log_index - 1,
        }


def _run_id(name: str, created_at: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:80] or "run"
    timestamp = created_at.strftime("%Y%m%dt%H%M%S%fz")
    return f"{slug}-{timestamp}-{secrets.token_hex(4)}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _public_path(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _project_root() -> Path:
    explicit = os.getenv("LAYA_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "pyproject.toml").is_file() else Path.cwd().resolve()


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
            runs.append(metadata)
    return sorted(runs, key=lambda row: row.get("created_at", ""), reverse=True)


def _build_predictor() -> Predictor:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya import Router

    device = os.getenv("LAYA_DEVICE") or None
    project_root = _project_root()
    bundle = Path(
        os.getenv(
            "LAYA_MODEL_BUNDLE",
            project_root / "models" / "models--convaiinnovations--laya",
        )
    ).expanduser().resolve()
    revision_file = bundle / "refs" / "main"
    if not revision_file.is_file():
        raise FileNotFoundError(
            "Local Laya bundle has no refs/main. "
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

    required_names = (
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    missing = [
        f"{name}: {required}"
        for name, path in model_paths.items()
        for required in required_names
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
    return router


def _questions(entity_name: str) -> dict[str, Any]:
    return {
        "criminal_association": {
            "type": "choice",
            "instructions": (
                f"How does this article portray {entity_name!r} regarding criminal behavior? "
                "Judge only the named entity, not other people or organizations."
            ),
            "criteria": {
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
    fine_tuning_script: Path | None = None,
) -> FastAPI:
    review_threshold = float(os.getenv("LAYA_REVIEW_THRESHOLD", "0.80"))
    if not 0 <= review_threshold <= 1:
        raise ValueError("LAYA_REVIEW_THRESHOLD must be between 0 and 1")
    project_root = _project_root()
    benchmark_root = project_root / "artifacts" / "independent-annotations"
    corpus_path = benchmark_corpus or Path(
        os.getenv("LAYA_BENCHMARK_CORPUS", benchmark_root / "blind_articles.jsonl")
    )
    gold_path = benchmark_gold or Path(
        os.getenv(
            "LAYA_BENCHMARK_GOLD",
            benchmark_root / "batches" / "azure.annotations.jsonl",
        )
    )
    source_dir = benchmark_source_dir or Path(
        os.getenv("LAYA_BENCHMARK_SOURCE_DIR", project_root / "data" / "source")
    )
    default_output = (
        benchmark_root / "batches" / "laya-live.predictions.jsonl"
        if benchmark_corpus is None
        else corpus_path.with_name("laya-live.predictions.jsonl")
    )
    output_path = benchmark_output or Path(os.getenv("LAYA_BENCHMARK_OUTPUT", default_output))
    runs_dir = output_path.parent / "saved-runs"
    active_metadata_path = output_path.with_suffix(".meta.json")
    training_script = fine_tuning_script or project_root / "scripts" / "fine_tune_laya.py"
    fine_tuning_setting = os.getenv("LAYA_ENABLE_FINE_TUNING")
    bind_host = os.getenv("LAYA_HOST", "127.0.0.1")
    fine_tuning_enabled = (
        fine_tuning_setting.casefold() in {"1", "true", "yes", "on"}
        if fine_tuning_setting is not None
        else bind_host in {"127.0.0.1", "localhost", "::1"}
    )

    def resolve_training_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()

    def require_training_path(value: str, kind: Literal["file", "directory"]) -> Path:
        path = resolve_training_path(value)
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            raise HTTPException(
                status_code=422,
                detail=f"{kind} does not exist: {_public_path(path, project_root)}",
            )
        return path

    def default_training_model_dir() -> str:
        explicit = os.getenv("LAYA_MODEL_PATH")
        if explicit:
            return _public_path(Path(explicit), project_root)
        try:
            bundle = project_root / "models" / "models--convaiinnovations--laya"
            revision = (bundle / "refs" / "main").read_text(encoding="utf-8").strip()
            return _public_path(bundle / "snapshots" / revision, project_root)
        except OSError:
            return ""

    def public_command(command: Sequence[str]) -> list[str]:
        displayed = []
        for part in command:
            if part == sys.executable:
                displayed.append("python")
                continue
            path = Path(part)
            displayed.append(_public_path(path, project_root) if path.is_absolute() else part)
        return displayed

    def sanitize_training_output(
        message: str, command: Sequence[str], displayed_command: Sequence[str]
    ) -> str:
        replacements = {
            actual: displayed
            for actual, displayed in zip(command, displayed_command)
            if actual != displayed
        }
        replacements[str(Path(sys.executable).resolve().parents[1])] = "<python-env>"
        for actual in sorted(replacements, key=len, reverse=True):
            message = message.replace(actual, replacements[actual])
        return message

    def require_fine_tuning_enabled() -> None:
        if not fine_tuning_enabled:
            raise HTTPException(
                status_code=403,
                detail="fine-tuning is disabled; set LAYA_ENABLE_FINE_TUNING=true on a trusted administrative deployment",
            )

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
        app.state.benchmark_routing_mode = "auto"
        app.state.fine_tuning_job = FineTuningJob()
        app.state.fine_tuning_task = None
        if output_path.is_file():
            try:
                articles, gpt_labels, human_labels, audit = load_benchmark(
                    corpus_path, gold_path, source_dir
                )
                rows = read_jsonl(output_path)
                active_metadata = (
                    json.loads(active_metadata_path.read_text(encoding="utf-8"))
                    if active_metadata_path.is_file()
                    else {}
                )
                if rows:
                    app.state.benchmark.restore(
                        rows,
                        articles,
                        gpt_labels,
                        human_labels,
                        audit,
                        total=int(active_metadata.get("total") or len(articles)),
                        elapsed_seconds=active_metadata.get("elapsed_seconds"),
                    )
                    app.state.benchmark_rows = articles
                    app.state.benchmark_gold = gpt_labels
                    app.state.benchmark_human = human_labels
                    app.state.benchmark_routing_mode = active_metadata.get(
                        "routing_mode", "auto"
                    )
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
                pass
        app.state.warmup_future = app.state.inference_pool.submit(predictor_factory)
        try:
            yield
        finally:
            if app.state.benchmark_task and not app.state.benchmark_task.done():
                app.state.benchmark_task.cancel()
            fine_tuning_job: FineTuningJob = app.state.fine_tuning_job
            if fine_tuning_job.process and fine_tuning_job.process.returncode is None:
                fine_tuning_job.process.terminate()
                await fine_tuning_job.process.wait()
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
        except Exception:  # Keep diagnostics and the test UI available.
            request.app.state.startup_error = (
                "model startup failed; check the CrimeLaya server logs"
            )
            return None
        return request.app.state.predictor

    app = FastAPI(
        title="Laya Adverse Media",
        version="0.1.0",
        lifespan=lifespan,
    )
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "benchmark.html")

    @app.get("/fine-tuning", include_in_schema=False)
    def fine_tuning_page() -> FileResponse:
        require_fine_tuning_enabled()
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
            "process_id": os.getpid(),
            "uptime_seconds": round(time.time() - request.app.state.started_at, 1),
            "resident_models": {
                name: {
                    "instance_id": hex(id(agent)),
                    "device": str(getattr(agent, "device", "unknown")),
                    "source": _public_path(
                        Path(model_specs.get(name, "unknown")), project_root
                    ),
                }
                for name, agent in agents.items()
            },
            "network_model_downloads": "disabled",
        }

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
            result = await loop.run_in_executor(
                request.app.state.inference_pool,
                lambda: predictor.predict(
                    {"article": payload.article, "entity_name": payload.entity_name},
                    _questions(payload.entity_name),
                    model=(
                        None if payload.routing_mode == "auto" else payload.routing_mode
                    ),
                ),
            )
            return _parse_result(result, payload.entity_name, review_threshold)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def run_fine_tuning(
        request: Request,
        phase: str,
        command: list[str],
        displayed_command: list[str],
    ) -> None:
        job: FineTuningJob = request.app.state.fine_tuning_job
        if job.stop_requested:
            job.status = "stopped"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            return
        job.append_log(f"Starting {phase}")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            job.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=project_root,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            job.pid = job.process.pid
            assert job.process.stdout is not None
            while line := await job.process.stdout.readline():
                job.append_log(
                    sanitize_training_output(
                        line.decode("utf-8", errors="replace"),
                        command,
                        displayed_command,
                    )
                )
            job.exit_code = await job.process.wait()
            if job.stop_requested:
                job.status = "stopped"
                job.append_log("Job stopped")
            elif job.exit_code == 0:
                job.status = "complete"
                job.append_log(f"{phase.capitalize()} completed")
            else:
                job.status = "failed"
                job.error = f"{phase.capitalize()} exited with code {job.exit_code}"
                job.append_log(job.error)
        except asyncio.CancelledError:
            if job.process and job.process.returncode is None:
                job.process.terminate()
                await job.process.wait()
            job.status = "stopped"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.append_log(f"Unable to run {phase}: {exc}")
        finally:
            job.pid = None
            job.process = None
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def launch_fine_tuning(
        request: Request, phase: str, command: list[str]
    ) -> dict[str, Any]:
        job: FineTuningJob = request.app.state.fine_tuning_job
        task = request.app.state.fine_tuning_task
        if job.active or (task and not task.done()):
            raise HTTPException(status_code=409, detail="a fine-tuning job is already running")
        displayed_command = public_command(command)
        job.start(phase, displayed_command)
        request.app.state.fine_tuning_task = asyncio.create_task(
            run_fine_tuning(request, phase, command, displayed_command)
        )
        return job.snapshot()

    @app.get("/v1/fine-tuning")
    async def fine_tuning_status(
        request: Request, logs_after: int = 0
    ) -> dict[str, Any]:
        require_fine_tuning_enabled()
        job: FineTuningJob = request.app.state.fine_tuning_job
        return {
            **job.snapshot(max(0, logs_after)),
            "defaults": {
                "model_dir": default_training_model_dir(),
                "corpus": "artifacts/independent-annotations/blind_articles.jsonl",
                "annotations": "artifacts/independent-annotations/batches/azure.annotations.jsonl",
                "data_dir": _public_path(source_dir, project_root),
                "prepared_dir": "artifacts/adverse-media-data-azure",
                "output_dir": "models/laya-adverse-media-azure",
            },
        }

    @app.post("/v1/fine-tuning/prepare")
    async def fine_tuning_prepare(
        payload: FineTuningPrepareRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning_enabled()
        if not training_script.is_file():
            raise HTTPException(status_code=503, detail="fine-tuning script is unavailable")
        model_dir = require_training_path(payload.model_dir, "directory")
        command = [
            sys.executable,
            str(training_script),
            "prepare",
            "--label-source",
            payload.label_source,
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(resolve_training_path(payload.output_dir)),
            "--seed",
            str(payload.seed),
        ]
        if payload.label_source == "azure":
            command.extend(
                [
                    "--corpus",
                    str(require_training_path(payload.corpus, "file")),
                    "--annotations",
                    str(require_training_path(payload.annotations, "file")),
                ]
            )
        else:
            command.extend(
                ["--data-dir", str(require_training_path(payload.data_dir, "directory"))]
            )
        return launch_fine_tuning(request, "prepare", command)

    @app.post("/v1/fine-tuning/train")
    async def fine_tuning_train(
        payload: FineTuningTrainRequest, request: Request
    ) -> dict[str, Any]:
        require_fine_tuning_enabled()
        if not training_script.is_file():
            raise HTTPException(status_code=503, detail="fine-tuning script is unavailable")
        prepared_dir = require_training_path(payload.prepared_dir, "directory")
        model_dir = require_training_path(payload.model_dir, "directory")
        if payload.gpu_count == 1:
            command = [sys.executable, str(training_script), "train"]
        else:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={payload.gpu_count}",
                str(training_script),
                "train",
            ]
        command.extend(
            [
                "--prepared-dir",
                str(prepared_dir),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(resolve_training_path(payload.output_dir)),
                "--epochs",
                str(payload.epochs),
                "--batch-size",
                str(payload.batch_size),
                "--eval-batch-size",
                str(payload.eval_batch_size),
                "--gradient-accumulation",
                str(payload.gradient_accumulation),
                "--group-size",
                str(payload.group_size),
                "--encoder-lr",
                str(payload.encoder_lr),
                "--head-lr",
                str(payload.head_lr),
                "--weight-decay",
                str(payload.weight_decay),
                "--sigma-start",
                str(payload.sigma_start),
                "--sigma-end",
                str(payload.sigma_end),
                "--seed",
                str(payload.seed),
                "--log-every",
                str(payload.log_every),
            ]
        )
        if payload.resume_from:
            resume_from = require_training_path(payload.resume_from, "directory")
            command.extend(["--resume-from", str(resume_from)])
        return launch_fine_tuning(request, "train", command)

    @app.post("/v1/fine-tuning/stop")
    async def fine_tuning_stop(request: Request) -> dict[str, Any]:
        require_fine_tuning_enabled()
        job: FineTuningJob = request.app.state.fine_tuning_job
        if not job.active:
            return job.snapshot()
        job.stop_requested = True
        job.status = "stopping"
        job.append_log("Stop requested")
        if not job.process:
            return job.snapshot()
        job.process.terminate()
        try:
            await asyncio.wait_for(job.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            job.process.kill()
        return job.snapshot()

    async def run_benchmark(request: Request) -> None:
        state: BenchmarkState = request.app.state.benchmark
        predictor = request.app.state.predictor
        rows = request.app.state.benchmark_rows
        human_by_id = request.app.state.benchmark_human
        routing_mode = request.app.state.benchmark_routing_mode
        loop = asyncio.get_running_loop()
        try:
            for article in rows[state.snapshot()["completed"] :]:
                state.begin_item(article)
                started = time.perf_counter()
                result = await loop.run_in_executor(
                    request.app.state.inference_pool,
                    lambda article=article: predictor.predict(
                        {"article": article["article"], "entity_name": article["entity_name"]},
                        _questions(article["entity_name"]),
                        model=None if routing_mode == "auto" else routing_mode,
                    ),
                )
                prediction = _parse_result(result, article["entity_name"], review_threshold).model_dump()
                inference_seconds = time.perf_counter() - started
                truths = {
                    "gpt": request.app.state.benchmark_gold.get(article["article_id"]),
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
                    "gpt_label": truths["gpt"].get("label") if truths["gpt"] else None,
                }
                with output_path.open("a", encoding="utf-8", newline="\n") as output:
                    output.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                if state.should_pause():
                    state.paused()
                    return
            state.finish()
            active = state.snapshot()
            _write_json(active_metadata_path, {
                "total": active["total"],
                "completed": active["completed"],
                "status": active["status"],
                "routing_mode": routing_mode,
                "elapsed_seconds": active["timing"]["elapsed_seconds"],
            })
        except asyncio.CancelledError:
            state.paused()
            raise
        except Exception as exc:
            state.fail(exc)

    def refresh_gpt_labels(request: Request) -> None:
        try:
            modified = gold_path.stat().st_mtime_ns
        except OSError:
            return
        if modified == request.app.state.benchmark_gold_mtime:
            return
        labels = load_gpt_labels(gold_path)
        request.app.state.benchmark_gold = labels
        request.app.state.benchmark_gold_mtime = modified
        request.app.state.benchmark.refresh_gpt(labels)

    @app.get("/v1/benchmark")
    def benchmark_status(request: Request, series_after: int = 0) -> dict[str, Any]:
        state: BenchmarkState = request.app.state.benchmark
        snapshot = state.snapshot(max(0, series_after))
        if "blind_articles" not in snapshot["audit"]:
            try:
                _, _, _, audit = load_benchmark(corpus_path, gold_path, source_dir)
                state.update_audit(audit)
            except (OSError, ValueError, json.JSONDecodeError):
                state.update_audit({
                    "usable_gold_rows": 0,
                    "corpus_path": _public_path(corpus_path, project_root),
                    "gold_path": _public_path(gold_path, project_root),
                    "error": "benchmark inputs are unavailable; check the CrimeLaya server logs",
                })
        refresh_gpt_labels(request)
        return state.snapshot(max(0, series_after))

    @app.post("/v1/benchmark/start")
    async def benchmark_start(payload: BenchmarkStartRequest, request: Request) -> dict[str, Any]:
        predictor = resolve_predictor(request)
        if predictor is None:
            raise HTTPException(status_code=503, detail="Laya model is not ready")
        state: BenchmarkState = request.app.state.benchmark
        current = state.snapshot()
        task = request.app.state.benchmark_task
        if task and not task.done():
            raise HTTPException(status_code=409, detail="benchmark is already running")

        if current["status"] == "paused" and request.app.state.benchmark_rows:
            state.start(current["total"], current["audit"], resume=True)
        else:
            rows, gold_by_id, human_by_id, audit = load_benchmark(corpus_path, gold_path, source_dir)
            if payload.limit is not None:
                rows = rows[: payload.limit]
            request.app.state.benchmark_rows = rows
            request.app.state.benchmark_gold = gold_by_id
            request.app.state.benchmark_human = human_by_id
            request.app.state.benchmark_gold_mtime = gold_path.stat().st_mtime_ns
            request.app.state.benchmark_routing_mode = payload.routing_mode
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            audit["output_path"] = str(output_path)
            state.start(len(rows), audit)
            _write_json(active_metadata_path, {
                "total": len(rows),
                "completed": 0,
                "status": "running",
                "routing_mode": payload.routing_mode,
                "elapsed_seconds": 0.0,
            })
        request.app.state.benchmark_task = asyncio.create_task(run_benchmark(request))
        return state.snapshot()

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
        request.app.state.benchmark.reset()
        return request.app.state.benchmark.snapshot()

    @app.get("/v1/benchmark/runs")
    def benchmark_runs() -> dict[str, Any]:
        return {"runs": _saved_runs(runs_dir)}

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
        run_id = _run_id(name, created_at)
        runs_dir.mkdir(parents=True, exist_ok=True)
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
            "timing": snapshot["timing"],
        }
        _write_json(runs_dir / f"{run_id}.json", metadata)
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
        articles, gpt_labels, human_labels, audit = load_benchmark(
            corpus_path, gold_path, source_dir
        )
        audit["loaded_run"] = {
            "run_id": metadata["run_id"],
            "name": metadata["name"],
            "created_at": metadata["created_at"],
        }
        request.app.state.benchmark.restore(
            read_jsonl(ledger_path),
            articles,
            gpt_labels,
            human_labels,
            audit,
            total=int(metadata["total"]),
            elapsed_seconds=metadata.get("timing", {}).get("elapsed_seconds"),
        )
        request.app.state.benchmark_rows = articles
        request.app.state.benchmark_gold = gpt_labels
        request.app.state.benchmark_human = human_labels
        request.app.state.benchmark_routing_mode = metadata.get("routing_mode", "auto")
        return request.app.state.benchmark.snapshot()

    return app


app = create_app()
