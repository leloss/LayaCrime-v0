import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from laya_adverse_media.app import (
    FineTuningPrepareRequest,
    FineTuningTrainRequest,
    _build_predictor,
    _fine_tuned_model_paths,
    _model_options,
    _resolve_training_parameters,
    _SelectablePredictor,
    create_app,
)
from laya_adverse_media.benchmark import BenchmarkState, article_id
from laya_adverse_media.benchmark_models import TemporaryCloudModelError


class FakePredictor:
    loaded = ["english"]

    def __init__(self) -> None:
        self.calls = []

    def predict(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        return {
            "answers": {
                "criminal_association": {
                    "choice": "A",
                    "confidence": 0.91,
                    "probabilities": {"A": 0.91, "B": 0.09},
                }
            },
            "routing": {"model": "english", "reason": "explicit model"},
        }


class PositivePredictor(FakePredictor):
    def predict(self, state, questions, *, model=None):
        result = super().predict(state, questions, model=model)
        result["answers"]["criminal_association"] = {
            "choice": "B",
            "confidence": 0.72,
            "probabilities": {"A": 0.28, "B": 0.72},
        }
        return result

def test_fine_tuning_prepare_defaults_to_80_20_partition(tmp_path: Path) -> None:
    request = FineTuningPrepareRequest(
        corpus=tmp_path / "corpus.jsonl",
        labels=tmp_path / "labels.jsonl",
        model_dir=tmp_path / "model",
        output_dir=tmp_path / "prepared",
        seed=20260923,
    )

    assert request.partition_strategy == "holdout_80_20"
    assert request.fold_count == 5


def test_fine_tuning_releases_inference_resources_before_training(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("train.pt", "calibration.pt", "split-assignments.jsonl"):
        (prepared / name).write_text("prepared", encoding="utf-8")
    (prepared / "manifest.json").write_text(
        json.dumps({"partition": {"strategy": "holdout_80_20"}}),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    events = []

    class Predictor(FakePredictor):
        def unload(self):
            events.append("unload-predictor")

    class Runtime:
        def close(self):
            events.append("close-runtime")

    def capture_start(self, command, phase, cwd, env=None):
        events.append("start-training")

    monkeypatch.setenv("LAYA_FINE_TUNING_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("LAYA_ENABLE_FINE_TUNING", "true")
    monkeypatch.setattr("laya_adverse_media.app.FineTuningJob.start", capture_start)

    with TestClient(create_app(Predictor)) as client:
        client.get("/health")
        client.app.state.generative_runtime = Runtime()
        client.app.state.benchmark_active_model = "ministral"
        response = client.post("/v1/fine-tuning/train", json={
            **training_request(
                prepared_dir=prepared,
                model_dir=model_dir,
                output_dir=tmp_path / "output",
                epochs=1,
            ).model_dump(mode="json"),
        })

        assert response.status_code == 200, response.text
        assert events == ["close-runtime", "unload-predictor", "start-training"]
        assert client.app.state.generative_runtime is None
        assert client.app.state.benchmark_active_model == "auto"


def test_fine_tuning_does_not_interrupt_running_benchmark(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("train.pt", "calibration.pt", "split-assignments.jsonl"):
        (prepared / name).write_text("prepared", encoding="utf-8")
    (prepared / "manifest.json").write_text(
        json.dumps({"partition": {"strategy": "holdout_80_20"}}),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("LAYA_FINE_TUNING_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("LAYA_ENABLE_FINE_TUNING", "true")

    with TestClient(create_app(FakePredictor)) as client:
        client.app.state.benchmark.start(1, {})
        response = client.post("/v1/fine-tuning/train", json={
            **training_request(
                prepared_dir=prepared,
                model_dir=model_dir,
                output_dir=tmp_path / "output",
                epochs=1,
            ).model_dump(mode="json"),
        })

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "pause or finish the active benchmark before fine-tuning"
    )


def test_fine_tuning_rejects_stale_prepared_partition(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("train.pt", "calibration.pt", "split-assignments.jsonl"):
        (prepared / name).write_text("prepared", encoding="utf-8")
    (prepared / "manifest.json").write_text(
        json.dumps({"partition": {"strategy": "holdout_70_30"}}),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("LAYA_FINE_TUNING_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("LAYA_ENABLE_FINE_TUNING", "true")

    with TestClient(create_app(FakePredictor)) as client:
        response = client.post("/v1/fine-tuning/train", json={
            **training_request(
                prepared_dir=prepared,
                model_dir=model_dir,
                output_dir=tmp_path / "output",
            ).model_dump(mode="json"),
        })

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Prepared data uses holdout_70_30, but the form selects holdout_80_20. "
        "Run Prepare dataset again before training."
    )


class SlowPredictor(FakePredictor):
    def predict(self, state, questions, *, model=None):
        time.sleep(0.03)
        return super().predict(state, questions, model=model)


class FailsOncePredictor(FakePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def predict(self, state, questions, *, model=None):
        if len(self.calls) == 1 and not self.failed:
            self.failed = True
            raise RuntimeError("temporary inference failure")
        return super().predict(state, questions, model=model)


def training_request(**overrides) -> FineTuningTrainRequest:
    values = {
        "prepared_dir": "prepared",
        "model_dir": "model",
        "output_dir": "output",
        "gpu_count": 1,
        "epochs": 99,
        "batch_size": 4,
        "eval_batch_size": 16,
        "gradient_accumulation": 8,
        "group_size": 9,
        "rl_weight": 0.5,
        "encoder_lr": 0.5,
        "head_lr": 0.5,
        "weight_decay": 0.5,
        "sigma_start": 0.9,
        "sigma_end": 0.8,
        "encoder_warmup_epochs": 3,
        "lr_warmup_ratio": 0.2,
        "label_smoothing": 0.1,
        "seed": 42,
        "log_every": 50,
    }
    values.update(overrides)
    return FineTuningTrainRequest(**values)


def test_training_strategy_preserves_submitted_parameter_overrides() -> None:
    preserved = _resolve_training_parameters(
        training_request(strategy="preserve_entity_matching")
    )
    custom = _resolve_training_parameters(training_request(strategy="custom"))

    assert preserved["encoder_lr"] == 0.5
    assert preserved["epochs"] == 99
    assert custom["encoder_lr"] == 0.5
    assert custom["epochs"] == 99


def test_fine_tuning_prompt_is_editable_and_passed_to_preparation(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "datasets" / "public-training"
    annotations = dataset_root / "annotations"
    subsets = dataset_root / "training-subsets" / "consensus"
    annotations.mkdir(parents=True)
    subsets.mkdir(parents=True)
    corpus = dataset_root / "corpus.jsonl"
    labels = annotations / "consensus.jsonl"
    subset_labels = subsets / "labels-00001.jsonl"
    corpus.write_text(
        json.dumps({
            "article_id": "one",
            "entity_name": "Acme",
            "article": "Acme was investigated.",
        }) + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps({"article_id": "one", "label": 2}) + "\n",
        encoding="utf-8",
    )
    subset_labels.write_text(labels.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = dataset_root / "dataset.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "id": "public-training",
        "name": "Public training",
        "purpose": "training-only",
        "prompt": {
            "question": "Original question for {entity_name}?",
            "criteria": [
                {"decision": "negative", "text": "Original negative"},
                {"decision": "positive", "text": "Original positive"},
            ],
        },
        "corpus": {"path": "corpus.jsonl", "records": 1},
        "annotations": [{
            "id": "consensus",
            "name": "Consensus",
            "path": "annotations/consensus.jsonl",
            "records": 1,
        }],
        "training_subsets": [{
            "annotation_set_id": "consensus",
            "path": "training-subsets/consensus",
            "manifest": "training-subsets/consensus/manifest.json",
            "sizes": [1],
        }],
    }), encoding="utf-8")
    (subsets / "manifest.json").write_text("{}", encoding="utf-8")
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "prepared"
    model_dir.mkdir()
    captured = {}

    def capture_start(self, command, phase, cwd, env=None):
        captured["command"] = command
        captured["phase"] = phase

    monkeypatch.setenv("LAYA_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("LAYA_FINE_TUNING_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("LAYA_ENABLE_FINE_TUNING", "true")
    monkeypatch.setattr("laya_adverse_media.app.FineTuningJob.start", capture_start)

    with TestClient(create_app(FakePredictor)) as client:
        defaults = client.get("/v1/fine-tuning").json()["defaults"]
        preset = next(
            item for item in defaults["dataset_presets"]
            if item["dataset_id"] == "public-training"
        )
        updated = client.post("/v1/fine-tuning/prompt", json={
            "dataset_id": "public-training",
            "question": "Updated question for {entity_name}?",
            "criteria": [
                {"decision": "negative", "text": "Updated negative"},
                {"decision": "positive", "text": "Updated positive"},
            ],
        })
        prepared = client.post("/v1/fine-tuning/prepare", json={
            "label_source": "external",
            "dataset_id": "public-training",
            "corpus": str(corpus),
            "labels": str(subset_labels),
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "seed": 17,
            "partition_strategy": "group_k_fold",
            "fold_count": 7,
            "fold_index": 3,
        })

    assert preset["prompt"]["question"] == "Original question for {entity_name}?"
    assert updated.status_code == 200
    assert json.loads(manifest.read_text(encoding="utf-8"))["prompt"]["question"] == (
        "Updated question for {entity_name}?"
    )
    assert prepared.status_code == 200, prepared.text
    assert captured["phase"] == "preparation"
    prompt_index = captured["command"].index("--prompt-file")
    assert Path(captured["command"][prompt_index + 1]) == manifest
    assert captured["command"][captured["command"].index("--partition-strategy") + 1] == "group_k_fold"
    assert captured["command"][captured["command"].index("--fold-count") + 1] == "7"
    assert captured["command"][captured["command"].index("--fold-index") + 1] == "3"


def test_model_options_are_sorted_newest_first(tmp_path) -> None:
    older = tmp_path / "laya-adverse-media-older"
    newer = tmp_path / "laya-adverse-media-newer"
    for checkpoint in (older, newer):
        for relative_path in (
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ):
            path = checkpoint / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    older_model = older / "model.safetensors"
    newer_model = newer / "model.safetensors"
    older_model.touch()
    time.sleep(0.01)
    newer_model.touch()
    custom_prompt = {
        "question": "Assess {entity_name} for a custom offense.",
        "criteria": [
            {"decision": "negative", "text": "Custom negative"},
            {"decision": "positive", "text": "Custom positive"},
        ],
    }
    (newer / "rl_agent_config.json").write_text(
        json.dumps({"prompt": custom_prompt}), encoding="utf-8"
    )

    discovered = _fine_tuned_model_paths(tmp_path)
    predictor = type("Predictor", (), {"models": {
        "crime-older": str(older),
        "crime-newer": str(newer),
    }})()

    assert list(discovered) == ["crime-newer", "crime-older"]
    options = _model_options(predictor)
    assert [option["id"] for option in options] == [
        "crime-layacrime-public",
        "crime-newer",
        "crime-older",
        "auto",
    ]
    assert options[1]["prompt"] == custom_prompt
    assert options[-1]["prompt"]["question"].startswith("How does this article")


def test_public_checkpoint_has_canonical_model_option(tmp_path) -> None:
    models_root = tmp_path / "models" / "fine-tuned"
    public = models_root / "layacrime-public"
    for relative_path in (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = public / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    predictor = type("Predictor", (), {"models": {
        name: str(path) for name, path in _fine_tuned_model_paths(models_root).items()
    }})()
    options = _model_options(predictor)

    assert options[0]["id"] == "crime-layacrime-public"
    assert options[0]["label"] == "LayaCrime.v0"
    assert options[0]["deletable"] is False
    assert not any("winner" in option["label"].casefold() for option in options)


def test_public_checkpoint_remains_installable_when_missing() -> None:
    predictor = type("Predictor", (), {"models": {}})()

    option = _model_options(predictor)[0]

    assert option["id"] == "crime-layacrime-public"
    assert option["label"] == "LayaCrime.v0"
    assert option["provider"] == "huggingface"
    assert option["repo_id"] == "leloss/LayaCrime-v0"
    assert option["installed"] is False
    assert option["installable"] is True


def test_public_checkpoint_install_downloads_validated_snapshot(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_start(self, command, model_id, models_dir, cwd, **kwargs):
        captured["command"] = command
        captured["model_id"] = model_id
        captured["models_dir"] = models_dir
        captured.update(kwargs)
        self.status = "running"
        self.phase = "preparing"
        self.model_id = model_id
        self.models_dir = models_dir

    monkeypatch.setenv("LAYA_FINE_TUNED_MODELS_DIR", str(tmp_path / "fine-tuned"))
    monkeypatch.setattr(
        "laya_adverse_media.app.ModelInstallationJob.start", fake_start
    )

    with TestClient(create_app(FakePredictor)) as client:
        response = client.post(
            "/v1/benchmark/models/install",
            json={"model_id": "crime-layacrime-public"},
        )

    assert response.status_code == 200, response.text
    assert captured["model_id"] == "crime-layacrime-public"
    assert "download_huggingface_snapshot.py" in " ".join(captured["command"])
    assert "leloss/LayaCrime-v0" in captured["command"]
    assert captured["command"].count("--required-file") == 5
    assert captured["models_dir"] == tmp_path / "fine-tuned" / "layacrime-public"


class _ActiveLanguageModel:
    active_model = "gpt-5-6-luna"
    is_healthy = True

    def predict(self, article, entity_name):
        return {
            "answers": {"criminal_association": {
                "choice": "A", "confidence": 1.0,
                "probabilities": {"A": 1.0, "B": 0.0},
            }},
            "routing": {"model": self.active_model, "provider": "azure"},
        }

    def close(self):
        pass


def test_individual_test_runs_the_active_language_model() -> None:
    app = create_app(FakePredictor)
    with TestClient(app) as client:
        app.state.generative_runtime = _ActiveLanguageModel()
        app.state.benchmark_active_model = "gpt-5-6-luna"
        response = client.post("/v1/adverse-media", json={
            "entity_name": "Acme Corp",
            "article": "Authorities charged Acme Corp with fraud.",
            "model_id": "gpt-5-6-luna",
        })

    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "negative"
    assert response.json()["routing"]["provider"] == "azure"


def test_individual_test_requires_language_model_activation() -> None:
    with TestClient(create_app(FakePredictor)) as client:
        response = client.post("/v1/adverse-media", json={
            "entity_name": "Acme Corp",
            "article": "Authorities charged Acme Corp with fraud.",
            "model_id": "gpt-5-6-luna",
        })

    assert response.status_code == 409
    assert "activate the selected model" in response.json()["detail"]


def test_academic_model_is_selectable_for_individual_and_benchmark_use() -> None:
    class Router:
        loaded = ["english"]
        models = {}

        def unload(self):
            self.loaded = []

        def preload(self, models):
            self.loaded = list(models)

    class AcademicRuntime:
        active_model = None

        def options(self):
            return [{
                "id": "academic-test-model",
                "label": "Academic test model",
                "family": "academic",
                "category": "decision",
                "deletable": False,
                "runtime_available": True,
                "prompt": None,
            }]

        def activate(self, model_id):
            self.active_model = model_id

        def predict(self, article, entity_name, model_id):
            return {"article": article, "entity": entity_name, "model": model_id}

    predictor = _SelectablePredictor(Router(), {}, lambda path: None, AcademicRuntime())

    assert _model_options(predictor)[-1]["id"] == "academic-test-model"
    assert predictor.predict(
        {"article": "Article", "entity_name": "Entity"},
        {},
        model="academic-test-model",
    ) == {
        "article": "Article",
        "entity": "Entity",
        "model": "academic-test-model",
    }
    assert predictor.active_model == "academic-test-model"


def test_fine_tuned_model_can_be_deleted(tmp_path, monkeypatch) -> None:
    models_root = tmp_path / "models"
    checkpoint = models_root / "laya-adverse-media-custom"
    for relative_path in (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = checkpoint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("LAYA_FINE_TUNED_MODELS_DIR", str(models_root))

    with TestClient(create_app(FakePredictor)) as client:
        catalog = client.get("/v1/fine-tuning").json()["defaults"]["model_presets"]
        deleted = client.post(
            "/v1/benchmark/models/delete", json={"model_id": "crime-custom"}
        )

    item = next(model for model in catalog if model["id"] == "crime-custom")
    assert item["deletable"] is True
    assert deleted.status_code == 200, deleted.text
    assert not checkpoint.exists()


def test_training_report_and_checkpoint_can_be_deleted(tmp_path, monkeypatch) -> None:
    models_root = tmp_path / "models"
    checkpoint = models_root / "laya-adverse-media-completed"
    for relative_path in (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = checkpoint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (checkpoint / "training_report.json").write_text(
        json.dumps({"history": [{"validation_loss": 0.2}], "test": {"accuracy": 0.9}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAYA_FINE_TUNED_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LAYA_ENABLE_FINE_TUNING", "true")

    with TestClient(create_app(FakePredictor)) as client:
        catalog = client.get("/v1/fine-tuning/reports").json()
        deleted = client.post(
            "/v1/fine-tuning/reports/delete",
            json={"run_id": "laya-adverse-media-completed"},
        )

    assert catalog["reports"][0]["deletable"] is True
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["reports"] == []
    assert all(preset["id"] != "crime-completed" for preset in deleted.json()["model_presets"])
    assert not checkpoint.exists()


def test_loaded_training_report_preserves_chart_metrics(tmp_path, monkeypatch) -> None:
    models_root = tmp_path / "models"
    checkpoint = models_root / "laya-adverse-media-completed"
    checkpoint.mkdir(parents=True)
    history = [
        {"epoch": 1, "train_loss": 0.8, "validation_loss": 0.7, "train_accuracy": 0.6, "validation_accuracy": 0.65},
        {"epoch": 2, "train_loss": 0.5, "validation_loss": 0.4, "train_accuracy": 0.75, "validation_accuracy": 0.8},
        {"epoch": 3, "train_loss": 0.3, "validation_loss": 0.45, "train_accuracy": 0.9, "validation_accuracy": 0.79},
    ]
    (checkpoint / "training_report.json").write_text(
        json.dumps({
            "history": history,
            "best_epoch": 2,
            "best_validation_loss": 0.4,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAYA_FINE_TUNED_MODELS_DIR", str(models_root))

    with TestClient(create_app(FakePredictor)) as client:
        loaded = client.post(
            "/v1/fine-tuning/reports/load",
            json={"run_id": "laya-adverse-media-completed"},
        )

    assert loaded.status_code == 200, loaded.text
    payload = loaded.json()
    assert payload["historical"] is True
    assert payload["loss_history"] == history
    assert payload["report"]["best_epoch"] == 2


def test_benchmark_models_include_local_and_credentialed_options(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LAYA_GGUF_MODELS_DIR", str(tmp_path / "gguf"))
    with TestClient(create_app(FakePredictor)) as client:
        catalog = client.get("/v1/benchmark/models")
        activation = client.post(
            "/v1/benchmark/models/activate",
            json={"model_id": "gpt-5-6-luna", "api_key": "do-not-echo"},
        )

    assert catalog.status_code == 200
    models = {model["id"]: model for model in catalog.json()["models"]}
    assert models["gpt-5-6-luna"]["credential_required"] is True
    assert models["gpt-5-6-luna"]["category"] == "language"
    assert models["auto"]["category"] == "decision"
    assert models["qwen3-8b-q6-k"]["installed"] is False
    assert activation.status_code == 422
    assert "do-not-echo" not in activation.text

def test_installed_gguf_can_be_deleted(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "gguf"
    models_dir.mkdir()
    model_path = models_dir / "Qwen3-8B-Q6_K.gguf"
    model_path.write_bytes(b"GGUFmodel")
    model_path.with_name(model_path.name + ".install.json").write_text(
        '{"size": 9}', encoding="utf-8"
    )
    monkeypatch.setenv("LAYA_GGUF_MODELS_DIR", str(models_dir))

    with TestClient(create_app(FakePredictor)) as client:
        before = client.get("/v1/benchmark/models").json()["models"]
        deleted = client.post(
            "/v1/benchmark/models/delete", json={"model_id": "qwen3-8b-q6-k"}
        )

    installed = next(item for item in before if item["id"] == "qwen3-8b-q6-k")
    assert installed["installed"] is True
    assert installed["deletable"] is True
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] == "qwen3-8b-q6-k"
    assert not model_path.exists()
    assert not model_path.with_name(model_path.name + ".install.json").exists()


def test_dead_generative_runtime_is_reset_to_auto(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "gguf"
    models_dir.mkdir()
    model_path = models_dir / "Qwen3-8B-Q6_K.gguf"
    model_path.write_bytes(b"GGUFmodel")
    model_path.with_name(model_path.name + ".install.json").write_text(
        '{"size": 9}', encoding="utf-8"
    )
    monkeypatch.setenv("LAYA_GGUF_MODELS_DIR", str(models_dir))

    class DeadRuntime:
        active_model = "qwen3-8b-q6-k"
        is_healthy = False

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    runtime = DeadRuntime()
    app = create_app(FakePredictor)
    with TestClient(app) as client:
        app.state.benchmark_active_model = "qwen3-8b-q6-k"
        app.state.generative_runtime = runtime
        response = client.get("/v1/benchmark/models")

    assert response.status_code == 200
    assert response.json()["active_model"] == "auto"
    assert app.state.benchmark_active_model == "auto"
    assert runtime.closed is True


def test_benchmark_model_download_bypasses_laya_offline_mode(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_start(self, command, model_id, models_dir, cwd, **kwargs):
        captured["command"] = command
        captured["model_id"] = model_id
        captured["models_dir"] = models_dir
        captured["cwd"] = cwd
        captured.update(kwargs)
        self.status = "running"
        self.phase = "preparing"
        self.model_id = model_id
        self.models_dir = models_dir

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LAYA_GGUF_MODELS_DIR", str(tmp_path / "gguf"))
    monkeypatch.setattr(
        "laya_adverse_media.app.ModelInstallationJob.start", fake_start
    )

    with TestClient(create_app(FakePredictor)) as client:
        response = client.post(
            "/v1/benchmark/models/install",
            json={
                "model_id": "qwen3-8b-q6-k",
                "huggingface_token": "transient-token",
            },
        )
        status = client.get("/v1/benchmark/models/install")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "running"
    assert status.json()["model_id"] == "qwen3-8b-q6-k"
    assert "HF_HUB_OFFLINE" not in captured["env"]
    assert json.loads(captured["stdin_data"])["token"] == "transient-token"
    assert "transient-token" not in " ".join(captured["command"])
    assert "transient-token" not in response.text


def test_custom_benchmark_model_is_registered_installed_and_removable(
    tmp_path, monkeypatch
) -> None:
    models_dir = tmp_path / "gguf"
    captured = {}

    def fake_start(self, command, model_id, models_root, cwd, **kwargs):
        captured["command"] = command
        self.status = "running"
        self.phase = "preparing"
        self.model_id = model_id
        self.models_dir = models_root

    monkeypatch.setenv("LAYA_GGUF_MODELS_DIR", str(models_dir))
    monkeypatch.setattr(
        "laya_adverse_media.app.ModelInstallationJob.start", fake_start
    )

    with TestClient(create_app(FakePredictor)) as client:
        created = client.post(
            "/v1/benchmark/models/custom",
            json={
                "name": "Future Gemma",
                "repo_id": "owner/future-gemma",
                "huggingface_path": "releases/future-gemma-Q8_0.gguf",
            },
        )
        model = created.json()["model"]
        installed = client.post(
            "/v1/benchmark/models/install",
            json={
                "model_id": model["id"],
                "repo_id": "owner/corrected-gemma",
                "huggingface_path": "weights/corrected-Q8_0.gguf",
            },
        )
        catalog = client.get("/v1/benchmark/models").json()["models"]
        deleted = client.post(
            "/v1/benchmark/models/delete", json={"model_id": model["id"]}
        )
        after = client.get("/v1/benchmark/models").json()["models"]

    assert created.status_code == 200, created.text
    assert installed.status_code == 200, installed.text
    assert "owner/corrected-gemma" in captured["command"]
    assert "weights/corrected-Q8_0.gguf" in captured["command"]
    assert "corrected-Q8_0.gguf" in captured["command"]
    current = next(item for item in catalog if item["id"] == model["id"])
    assert current["repo_id"] == "owner/corrected-gemma"
    assert current["huggingface_path"] == "weights/corrected-Q8_0.gguf"
    assert deleted.status_code == 200, deleted.text
    assert all(item["id"] != model["id"] for item in after)


def test_adverse_media_maps_laya_decision() -> None:
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post(
            "/v1/adverse-media",
            json={
                "entity_name": "Acme Corp",
                "article": "Authorities charged Acme Corp with fraud.",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "entity_name": "Acme Corp",
        "decision": "negative",
        "confidence": 0.91,
        "probabilities": {"negative": 0.91, "positive": 0.09},
        "needs_review": False,
        "routing": {"model": "english", "reason": "explicit model"},
    }
    assert predictor.calls[0][0]["entity_name"] == "Acme Corp"
    assert "criminal behavior or intent" in predictor.calls[0][1]["criminal_association"]["instructions"]
    assert predictor.calls[0][2] is None


def test_positive_low_confidence_decision_requires_review() -> None:
    with TestClient(create_app(PositivePredictor)) as client:
        response = client.post(
            "/v1/adverse-media",
            json={
                "entity_name": "Acme Corp",
                "article": "Acme Corp assisted investigators as the victim of a fraud.",
            },
        )

    assert response.status_code == 200
    assert response.json()["decision"] == "positive"
    assert response.json()["needs_review"] is True


def test_adverse_media_forwards_custom_question_template() -> None:
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post(
            "/v1/adverse-media",
            json={
                "entity_name": "Acme Corp",
                "article": "Acme Corp was investigated.",
                "question": "Assess whether {entity_name} planned or committed an offense.",
            },
        )

    assert response.status_code == 200
    instructions = predictor.calls[0][1]["criminal_association"]["instructions"]
    assert instructions == "Assess whether 'Acme Corp' planned or committed an offense."


def test_adverse_media_uses_selected_checkpoint_prompt_by_default(
    tmp_path, monkeypatch
) -> None:
    models_root = tmp_path / "models"
    checkpoint = models_root / "laya-adverse-media-custom"
    for relative_path in (
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = checkpoint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (checkpoint / "rl_agent_config.json").write_text(
        json.dumps({
            "prompt": {
                "question": "Does {entity_name} match the checkpoint task?",
                "criteria": [
                    {"decision": "negative", "text": "Checkpoint negative"},
                    {"decision": "positive", "text": "Checkpoint positive"},
                ],
            }
        }),
        encoding="utf-8",
    )
    predictor = FakePredictor()
    predictor.models = {}
    monkeypatch.setenv("LAYA_FINE_TUNED_MODELS_DIR", str(models_root))

    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post(
            "/v1/adverse-media",
            json={
                "entity_name": "Acme Corp",
                "article": "Acme Corp was investigated.",
                "model_id": "crime-custom",
            },
        )

    assert response.status_code == 200
    question = predictor.calls[0][1]["criminal_association"]
    assert question["instructions"] == "Does 'Acme Corp' match the checkpoint task?"
    assert question["criteria"] == {
        "A": "Checkpoint negative",
        "B": "Checkpoint positive",
    }


def test_explicit_multilingual_route_is_forwarded() -> None:
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post(
            "/v1/adverse-media",
            json={
                "entity_name": "Empresa Solaris",
                "article": "Artigo em português.",
                "routing_mode": "multilingual",
            },
        )

    assert response.status_code == 200
    assert predictor.calls[0][2] == "multilingual"


def test_health_reports_warm_model() -> None:
    with TestClient(create_app(FakePredictor)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "loaded": ["english"],
        "error": None,
    }


def test_workbench_and_assets_are_served() -> None:
    with TestClient(create_app(FakePredictor)) as client:
        individual = client.get("/")
        benchmark = client.get("/benchmark")
        fine_tuning = client.get("/fine-tuning")
        fine_tuning_status = client.get("/v1/fine-tuning")
        script = client.get("/static/benchmark.js")

    assert individual.status_code == 200
    assert "Individual Test" in individual.text
    assert benchmark.status_code == 200
    assert "Benchmark Run" in benchmark.text
    assert fine_tuning.status_code == 200
    assert "Fine-Tuning Console" in fine_tuning.text
    assert 'id="training-strategy"' in fine_tuning.text
    assert 'id="prompt-question"' in individual.text
    assert 'id="edit-prompt"' in individual.text
    assert 'id="prompt-question"' in benchmark.text
    assert 'id="edit-prompt"' in benchmark.text
    assert '<label>Model name<input id="azure-deployment"' in benchmark.text
    assert '<label>API endpoint<input id="azure-endpoint"' in benchmark.text
    assert '<label>API key<input id="azure-api-key"' in benchmark.text
    assert '"Cloud LLMs"' in script.text
    assert '"Self-hosted LLMs"' in script.text
    assert "new Option(endpointAction.action_label, endpointAction.id)" in script.text
    assert 'new Option("Add custom model…", customModelAction)' in script.text
    assert "if (action) group.append(action)" in script.text
    assert 'model.action_only ? ""' in script.text
    assert '["paused", "error"].includes(report.status)' in script.text
    assert 'interrupted ? "Resume run"' in script.text
    assert "interruptedModelNeedsActivation" in script.text
    defaults = fine_tuning_status.json()["defaults"]
    assert defaults["default_training_strategy"] == "balanced"
    presets = {item["id"]: item for item in defaults["dataset_presets"]}
    public_2000 = presets["adverse-media-public-tuning-2000-human-full"]
    assert not any("consensus" in preset_id for preset_id in presets)
    assert public_2000["samples"] == 2000
    assert public_2000["recommended_strategy"] == "public_natural_staged"
    assert [item["id"] for item in defaults["training_strategies"]] == [
        "preserve_entity_matching",
        "balanced",
        "maximum_task_adaptation",
        "relational_low_drift",
        "relational_adaptation",
        "relational_extended",
        "relational_rlcd",
        "public_natural_staged",
    ]
    assert script.status_code == 200
    assert "v1/benchmark" in script.text
    assert 'fetch("/health")' in script.text
    assert "if (running) timer = setTimeout(status, 200)" in script.text
    assert 'id="dataset"' not in benchmark.text
    assert "/v1/benchmark/label-sets/activate" in script.text


def test_diagnostics_reports_resident_predictor() -> None:
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.get("/diagnostics")

    assert response.status_code == 200
    assert response.json()["warm_start"] is True
    assert response.json()["predictor"].endswith("FakePredictor")
    assert response.json()["network_model_downloads"] == "disabled"


def test_startup_failure_keeps_health_available() -> None:
    def fail_to_load():
        raise RuntimeError("checkpoint download blocked")

    with TestClient(create_app(fail_to_load)) as client:
        health = client.get("/health")
        models = client.get("/v1/models")
        inference = client.post(
            "/v1/adverse-media",
            json={"entity_name": "Acme Corp", "article": "Article text"},
        )

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "loaded": [],
        "error": "checkpoint download blocked",
    }
    assert models.status_code == 503
    assert models.json()["detail"] == {
        "message": "Laya model is unavailable",
        "startup_error": "checkpoint download blocked",
    }
    assert inference.status_code == 503


def test_default_predictor_requires_local_checkpoint(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "english"
    model_path.mkdir()
    monkeypatch.setenv("LAYA_MODEL_PATH", str(model_path))

    try:
        _build_predictor()
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing local checkpoint was accepted")

    assert str(model_path) in message
    assert "model.safetensors" in message
    assert "Runtime network downloads are disabled" in message


def test_benchmark_scores_sequential_predictions_without_exposing_gold(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {"article_id": article_id("Acme", "Acme was charged."), "entity_name": "Acme", "article": "Acme was charged."},
        {"article_id": article_id("Beta", "Beta assisted investigators."), "entity_name": "Beta", "article": "Beta assisted investigators."},
    ]
    labels = [
        {"article_id": articles[0]["article_id"], "label": 2, "label_name": "bad_guy", "annotator": "reviewer"},
        {"article_id": articles[1]["article_id"], "label": 1, "label_name": "good_guy", "annotator": "reviewer"},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8")
    gold.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    (tmp_path / "source-one").write_text(
        "###entityName: Acme\n###disspositionReason: Hit\n###content: Acme was charged.\n",
        encoding="utf-8",
    )
    (tmp_path / "source-two").write_text(
        "###entityName: Beta\n###disspositionReason: False Positive\n###content: Beta assisted investigators.\n",
        encoding="utf-8",
    )
    predictor = FakePredictor()

    with TestClient(create_app(lambda: predictor, corpus, gold, tmp_path)) as client:
        started = client.post(
            "/v1/benchmark/start",
            json={
                "routing_mode": "english",
                "question": "Resolve {entity_name}, then assess criminal intent.",
            },
        )
        assert started.status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    assert report["status"] == "complete", report["error"]
    assert report["completed"] == 2
    assert report["question"] == "Resolve {entity_name}, then assess criminal intent."
    assert report["confusion_matrices"]["gold"]["values"] == [[0, 1], [0, 1]]
    assert report["confusion_matrices"]["human"]["values"] == [[0, 1], [0, 1]]
    assert report["metrics"]["gold"]["accuracy"] == 0.5
    assert report["probability_series"] == [
        {"index": 1, "negative": -91.0, "positive": 9.0},
        {"index": 2, "negative": -91.0, "positive": 9.0},
    ]
    saved = [json.loads(line) for line in (tmp_path / "laya-live.predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 2
    assert saved[0]["probabilities"] == {"negative": 0.91, "positive": 0.09}
    assert saved[0]["confidence"] == 0.91
    assert saved[0]["human_label"] == 2
    assert report["audit"]["annotators"] == {"reviewer": 2}
    assert all(set(state) == {"article", "entity_name"} for state, _, _ in predictor.calls)
    assert all("label" not in state and "annotator" not in state for state, _, _ in predictor.calls)
    assert [
        questions["criminal_association"]["instructions"]
        for _, questions, _ in predictor.calls
    ] == [
        "Resolve 'Acme', then assess criminal intent.",
        "Resolve 'Beta', then assess criminal intent.",
    ]


def test_benchmark_rejects_prompt_without_both_decisions(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    corpus.write_text(
        json.dumps({
            "article_id": "one",
            "entity_name": "Acme",
            "article": "Acme was charged.",
        }) + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps({"article_id": "one", "label": 2}) + "\n",
        encoding="utf-8",
    )

    with TestClient(create_app(FakePredictor, corpus, gold, tmp_path)) as client:
        response = client.post(
            "/v1/benchmark/start",
            json={
                "criteria": [
                    {"decision": "negative", "text": "First negative"},
                    {"decision": "negative", "text": "Second negative"},
                ]
            },
        )

    assert response.status_code == 422
    assert "negative and positive" in response.json()["detail"]


def test_benchmark_loads_active_evaluation_bundle(monkeypatch, tmp_path) -> None:
    datasets_root = tmp_path / "datasets"
    bundle_root = datasets_root / "holdout"
    annotations_dir = bundle_root / "annotations"
    annotations_dir.mkdir(parents=True)
    corpus = bundle_root / "corpus.jsonl"
    gold = annotations_dir / "human.jsonl"
    corpus.write_text(
        json.dumps(
            {"article_id": "one", "entity_name": "Acme", "article": "Acme was charged."}
        )
        + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "article_id": "one",
                "label": 2,
                "label_name": "negative",
                "annotator": "human",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_root / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "holdout",
                "name": "Holdout",
                "purpose": "evaluation-only",
                "corpus": {"path": "corpus.jsonl", "records": 1},
                "annotations": [
                    {"id": "human", "name": "Human", "path": "annotations/human.jsonl", "records": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    alternate_root = datasets_root / "alternate"
    alternate_annotations = alternate_root / "annotations"
    alternate_annotations.mkdir(parents=True)
    (alternate_root / "corpus.jsonl").write_text(
        json.dumps(
            {"article_id": "two", "entity_name": "Beta", "article": "Beta assisted investigators."}
        )
        + "\n",
        encoding="utf-8",
    )
    (alternate_annotations / "reviewers.jsonl").write_text(
        json.dumps(
            {
                "article_id": "two",
                "label": 1,
                "label_name": "positive",
                "annotator": "reviewer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (alternate_root / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "alternate",
                "name": "Alternate holdout",
                "purpose": "evaluation-only",
                "corpus": {"path": "corpus.jsonl", "records": 1},
                "annotations": [
                    {"id": "reviewers", "name": "Reviewers", "path": "annotations/reviewers.jsonl", "records": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    training_root = datasets_root / "training"
    training_annotations = training_root / "annotations"
    training_annotations.mkdir(parents=True)
    (training_root / "corpus.jsonl").write_text(corpus.read_text(encoding="utf-8"), encoding="utf-8")
    (training_annotations / "labels.jsonl").write_text(gold.read_text(encoding="utf-8"), encoding="utf-8")
    (training_root / "dataset.json").write_text(
        json.dumps({
            "schema_version": 1,
            "id": "training",
            "name": "Training only",
            "purpose": "training-only",
            "corpus": {"path": "corpus.jsonl", "records": 1},
            "annotations": [{
                "id": "labels",
                "name": "Training labels",
                "path": "annotations/labels.jsonl",
                "records": 1,
            }],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("LAYA_DATASETS_DIR", str(datasets_root))
    monkeypatch.setenv("LAYA_ACTIVE_DATASET", "holdout")
    monkeypatch.setenv("LAYA_BENCHMARK_LABEL_SET", "human")
    monkeypatch.setenv("LAYA_BENCHMARK_OUTPUT", str(output))
    monkeypatch.setenv("LAYA_BENCHMARK_SOURCE_DIR", str(tmp_path / "no-human-labels"))
    monkeypatch.delenv("LAYA_BENCHMARK_CORPUS", raising=False)
    monkeypatch.delenv("LAYA_BENCHMARK_GOLD", raising=False)
    monkeypatch.delenv("LAYA_BENCHMARK_LABELS", raising=False)

    with TestClient(create_app(FakePredictor)) as client:
        ground_truths = client.get("/v1/benchmark/label-sets").json()
        datasets = client.get("/v1/benchmark/datasets").json()
        blocked = client.post(
            "/v1/benchmark/datasets/activate",
            json={"dataset_id": "training"},
        )
        activated = client.post(
            "/v1/benchmark/label-sets/activate",
            json={"dataset_id": "alternate", "label_set_id": "reviewers"},
        )
        started = client.post("/v1/benchmark/start", json={})
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    assert {
        (row["dataset_id"], row["id"]) for row in ground_truths["label_sets"]
    } == {("holdout", "human"), ("alternate", "reviewers")}
    assert {row["id"] for row in datasets["datasets"]} == {"holdout", "alternate"}
    assert blocked.status_code == 422
    assert ground_truths["dataset"]["id"] == "holdout"
    assert activated.status_code == 200
    assert activated.json()["active_dataset"] == "alternate"
    assert activated.json()["active_label_set"] == "reviewers"
    assert started.status_code == 200
    assert report["status"] == "complete", report["error"]
    assert report["completed"] == 1
    assert report["audit"]["usable_gold_rows"] == 1
    assert report["audit"]["annotators"] == {"reviewer": 1}
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["article_id"] == "two"
    assert saved["gold_label"] == 1

def test_benchmark_resume_preserves_prediction_ledger(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {
            "article_id": article_id(f"Entity {index}", f"Article {index}"),
            "entity_name": f"Entity {index}",
            "article": f"Article {index}",
        }
        for index in range(8)
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8")
    gold.write_text("", encoding="utf-8")

    with TestClient(create_app(SlowPredictor, corpus, gold, tmp_path)) as client:
        assert client.post("/v1/benchmark/start", json={"routing_mode": "english"}).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["completed"] >= 1:
                break
            time.sleep(0.01)
        client.post("/v1/benchmark/pause")
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "paused":
                break
            time.sleep(0.01)

        paused_count = report["completed"]
        output = tmp_path / "laya-live.predictions.jsonl"
        assert len(output.read_text(encoding="utf-8").splitlines()) == paused_count
        metadata = json.loads(
            output.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        assert metadata["status"] == "paused"
        assert metadata["completed"] == paused_count

        assert client.post("/v1/benchmark/start", json={"routing_mode": "english"}).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "complete", report["error"]
    assert len(saved) == len(articles)
    assert len({row["article_id"] for row in saved}) == len(articles)


def test_failed_benchmark_resumes_without_reprocessing_completed_rows(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {
            "article_id": article_id(f"Entity {index}", f"Article {index}"),
            "entity_name": f"Entity {index}",
            "article": f"Article {index}",
        }
        for index in range(3)
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8"
    )
    gold.write_text("", encoding="utf-8")
    predictor = FailsOncePredictor()

    with TestClient(create_app(lambda: predictor, corpus, gold, tmp_path)) as client:
        assert client.post(
            "/v1/benchmark/start", json={"routing_mode": "english"}
        ).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "error":
                break
            time.sleep(0.01)

        output = tmp_path / "laya-live.predictions.jsonl"
        assert report["completed"] == 1
        assert len(output.read_text(encoding="utf-8").splitlines()) == 1

        assert client.post(
            "/v1/benchmark/start", json={"routing_mode": "english"}
        ).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    saved = output.read_text(encoding="utf-8").splitlines()
    assert report["status"] == "complete", report["error"]
    assert len(saved) == len(articles)
    assert len(predictor.calls) == len(articles)


def test_temporary_cloud_failure_pauses_and_resumes_paid_run(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {
            "article_id": article_id(f"Entity {index}", f"Article {index}"),
            "entity_name": f"Entity {index}",
            "article": f"Article {index}",
        }
        for index in range(3)
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8"
    )
    gold.write_text("", encoding="utf-8")

    class CloudRuntime:
        active_model = None
        is_healthy = True

        def __init__(self) -> None:
            self.calls = []
            self.failed = False

        def activate(self, model_id, credentials=None, spec=None):
            self.active_model = model_id
            return model_id

        def predict(self, article, entity_name):
            self.calls.append(article)
            if len(self.calls) == 2 and not self.failed:
                self.failed = True
                raise TemporaryCloudModelError("temporary no_capacity")
            return {
                "answers": {"criminal_association": {
                    "choice": "A", "confidence": 1.0,
                    "probabilities": {"A": 1.0, "B": 0.0},
                }},
                "routing": {"model": self.active_model, "provider": "azure"},
            }

        def close(self):
            self.active_model = None

    runtime = CloudRuntime()
    app = create_app(FakePredictor, corpus, gold, tmp_path)
    with TestClient(app) as client:
        app.state.generative_runtime = runtime
        activated = client.post("/v1/benchmark/models/activate", json={
            "model_id": "grok-4-1-fast-reasoning",
            "endpoint": "https://review.openai.azure.com/openai/v1/",
            "deployment": "grok-4-1-fast-reasoning",
            "api_key": "transient-key",
        })
        assert activated.status_code == 200, activated.text
        assert client.post("/v1/benchmark/start", json={
            "routing_mode": "grok-4-1-fast-reasoning"
        }).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "paused":
                break
            time.sleep(0.01)

        output = tmp_path / "laya-live.predictions.jsonl"
        assert report["completed"] == 1
        assert "no_capacity" in report["error"]
        assert len(output.read_text(encoding="utf-8").splitlines()) == 1
        changed = client.post("/v1/benchmark/models/activate", json={
            "model_id": "gpt-6-luna",
            "endpoint": "https://review.openai.azure.com/openai/v1/",
            "deployment": "gpt-6-luna",
            "api_key": "transient-key",
        })
        assert changed.status_code == 409
        reactivated = client.post("/v1/benchmark/models/activate", json={
            "model_id": "grok-4-1-fast-reasoning",
            "endpoint": "https://review.openai.azure.com/openai/v1/",
            "deployment": "grok-4-1-fast-reasoning",
            "api_key": "transient-key",
        })
        assert reactivated.status_code == 200, reactivated.text
        assert client.post("/v1/benchmark/start", json={
            "routing_mode": "grok-4-1-fast-reasoning"
        }).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    saved = output.read_text(encoding="utf-8").splitlines()
    assert report["status"] == "complete", report["error"]
    assert len(saved) == len(articles)
    assert len({json.loads(row)["article_id"] for row in saved}) == len(articles)
    assert runtime.calls == ["Article 0", "Article 1", "Article 1", "Article 2"]


def test_benchmark_resume_repairs_sparse_prediction_ledger(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {
            "article_id": article_id(f"Entity {index}", f"Article {index}"),
            "entity_name": f"Entity {index}",
            "article": f"Article {index}",
        }
        for index in range(4)
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8"
    )
    gold.write_text("", encoding="utf-8")
    predictor = FakePredictor()

    with TestClient(create_app(lambda: predictor, corpus, gold, tmp_path)) as client:
        assert client.post(
            "/v1/benchmark/start", json={"routing_mode": "english"}
        ).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

        output = tmp_path / "laya-live.predictions.jsonl"
        rows = output.read_text(encoding="utf-8").splitlines()
        output.write_text("\n".join(rows[2:]) + "\n", encoding="utf-8")
        client.app.state.benchmark.fail(RuntimeError("save failed"))

        resumed = client.post(
            "/v1/benchmark/start", json={"routing_mode": "english"}
        )
        assert resumed.status_code == 200, resumed.text
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "complete", report["error"]
    assert len(saved) == len(articles)
    assert {row["article_id"] for row in saved} == {
        article["article_id"] for article in articles
    }
    assert len(predictor.calls) == 6


def test_benchmark_runs_articles_without_gold_labels(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    corpus.write_text(
        json.dumps({"article_id": "one", "entity_name": "Acme", "article": "Text"}) + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps({"article_id": "one", "label": None, "annotator": None}) + "\n",
        encoding="utf-8",
    )

    with TestClient(create_app(FakePredictor, corpus, gold, tmp_path)) as client:
        report = client.get("/v1/benchmark").json()
        response = client.post("/v1/benchmark/start", json={})
        for _ in range(100):
            completed = client.get("/v1/benchmark").json()
            if completed["status"] == "complete":
                break
            time.sleep(0.01)

    assert report["audit"]["usable_gold_rows"] == 0
    assert response.status_code == 200
    assert completed["completed"] == 1
    assert completed["confusion_matrices"]["gold"]["compared"] == 0

def test_uploaded_label_set_can_be_deleted_with_fallback(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.jsonl"
    gold = tmp_path / "gold.jsonl"
    row = {"article_id": "one", "entity_name": "Acme", "article": "Text"}
    corpus.write_text(json.dumps(row) + "\n", encoding="utf-8")
    gold.write_text(json.dumps({"article_id": "one", "label": 1}) + "\n", encoding="utf-8")
    label_sets = tmp_path / "label-sets"
    monkeypatch.setenv("LAYA_LABEL_SETS_DIR", str(label_sets))

    with TestClient(create_app(FakePredictor, corpus, gold)) as client:
        uploaded = client.post(
            "/v1/benchmark/label-sets/upload",
            json={
                "name": "Temporary review",
                "content": json.dumps({"article_id": "one", "label": 2}),
            },
        )
        catalog = client.get("/v1/benchmark/label-sets").json()
        deleted = client.post(
            "/v1/benchmark/label-sets/delete",
            json={"label_set_id": "temporary-review"},
        )

    item = next(row for row in catalog["label_sets"] if row["id"] == "temporary-review")
    assert uploaded.status_code == 200, uploaded.text
    assert item["deletable"] is True
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["active_label_set"] == "default"
    assert not (label_sets / "temporary-review.jsonl").exists()


def test_benchmark_state_restores_persisted_predictions() -> None:
    article = {
        "article_id": "one",
        "entity_name": "Acme",
        "article": "Acme was charged.",
    }
    state = BenchmarkState()

    state.restore(
        [{
            "article_id": "one",
            "decision": "negative",
            "confidence": 0.91,
            "probabilities": {"negative": 0.91, "positive": 0.09},
            "elapsed_seconds": 1.25,
            "routing": {"model": "english"},
        }],
        [article],
        {},
        {"one": {"label": 2}},
        {"blind_articles": 1},
        total=1,
        elapsed_seconds=2.0,
    )

    report = state.snapshot()
    assert report["status"] == "complete"
    assert report["completed"] == 1
    assert report["current"]["article_id"] == "one"
    assert report["confusion_matrices"]["human"]["values"] == [[0, 0], [0, 1]]
    assert report["probability_series"] == [
        {"index": 1, "negative": -91.0, "positive": 9.0}
    ]


def test_benchmark_pending_item_includes_article_for_live_display() -> None:
    article = {
        "article_id": "one",
        "entity_name": "Acme",
        "article": "Acme was charged.",
    }
    state = BenchmarkState()
    state.start(1, {"blind_articles": 1})

    state.begin_item(article)

    assert state.snapshot()["pending"] == article




def test_benchmark_named_runs_can_be_saved_listed_and_loaded(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    corpus.write_text(
        json.dumps({
            "article_id": "one",
            "entity_name": "Acme",
            "article": "Acme was charged.",
        }) + "\n",
        encoding="utf-8",
    )
    gold.write_text("", encoding="utf-8")

    with TestClient(create_app(FakePredictor, corpus, gold, tmp_path)) as client:
        assert client.post("/v1/benchmark/start", json={}).status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

        first = client.post(
            "/v1/benchmark/runs/save", json={"name": "Base model"}
        ).json()["saved"]
        second = client.post(
            "/v1/benchmark/runs/save", json={"name": "Base model"}
        ).json()["saved"]
        assert first["run_id"] != second["run_id"]
        runs = client.get("/v1/benchmark/runs").json()["runs"]
        assert len(runs) == 3
        assert any("Laya · Router" in run["name"] for run in runs)

        deleted = client.post(
            "/v1/benchmark/runs/delete", json={"run_id": second["run_id"]}
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] == second["run_id"]
        assert len(deleted.json()["runs"]) == 2
        assert client.post(
            "/v1/benchmark/runs/load", json={"run_id": second["run_id"]}
        ).status_code == 404

        assert client.post("/v1/benchmark/reset").json()["completed"] == 0
        load_response = client.post(
            "/v1/benchmark/runs/load", json={"run_id": first["run_id"]}
        )
        assert load_response.status_code == 200, load_response.text
        loaded = load_response.json()

        corpus.write_text(
            json.dumps({
                "article_id": "changed",
                "entity_name": "Changed",
                "article": "The corpus changed after the run.",
            }) + "\n",
            encoding="utf-8",
        )
        mismatched = client.post(
            "/v1/benchmark/runs/load", json={"run_id": first["run_id"]}
        )

    assert loaded["status"] == "complete"
    assert loaded["completed"] == 1
    assert loaded["current"]["article_id"] == "one"
    assert len(first["corpus_sha256"]) == 64
    assert len(first["labels_sha256"]) == 64
    assert "metrics" in first
    assert "confusion_matrices" in first
    assert "question" in first
    assert "criteria" in first
    assert mismatched.status_code == 409
    assert "corpus fingerprint" in mismatched.json()["detail"]


def test_benchmark_save_rejects_ledger_from_another_explicit_model(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    corpus.write_text(
        json.dumps({
            "article_id": "one",
            "entity_name": "Acme",
            "article": "Acme was charged.",
        }) + "\n",
        encoding="utf-8",
    )
    gold.write_text("", encoding="utf-8")
    app = create_app(FakePredictor, corpus, gold, tmp_path)

    with TestClient(app) as client:
        assert client.post("/v1/benchmark/start", json={}).status_code == 200
        for _ in range(100):
            if client.get("/v1/benchmark").json()["status"] == "complete":
                break
            time.sleep(0.01)
        app.state.benchmark_routing_mode = "deepseek-r1-distill-qwen-14b-q4-k-m"

        response = client.post(
            "/v1/benchmark/runs/save", json={"name": "DeepSeek"}
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "prediction ledger does not match the selected model"
