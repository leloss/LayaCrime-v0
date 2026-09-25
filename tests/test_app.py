import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from laya_adverse_media.app import _build_predictor, create_app
from laya_adverse_media.benchmark import BenchmarkState, article_id


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

class SlowPredictor(FakePredictor):
    def predict(self, state, questions, *, model=None):
        time.sleep(0.03)
        return super().predict(state, questions, model=model)


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
        page = client.get("/")
        script = client.get("/static/benchmark.js")
        fine_tuning = client.get("/fine-tuning")

    assert page.status_code == 200
    assert "CrimeLaya Benchmark Evaluation" in page.text
    assert 'href="/fine-tuning"' in page.text
    assert script.status_code == 200
    assert "v1/benchmark" in script.text
    assert fine_tuning.status_code == 200
    assert "Fine-Tuning Console" in fine_tuning.text
    assert 'href="/"' in fine_tuning.text


def test_fine_tuning_prepare_runs_validated_local_command(tmp_path) -> None:
    training_script = tmp_path / "training_stub.py"
    training_script.write_text(
        "import sys\nprint('\\t'.join(sys.argv[1:]), flush=True)\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("{}\n", encoding="utf-8")
    prepared = tmp_path / "prepared"

    with TestClient(
        create_app(FakePredictor, fine_tuning_script=training_script)
    ) as client:
        response = client.post(
            "/v1/fine-tuning/prepare",
            json={
                "label_source": "azure",
                "corpus": str(corpus),
                "annotations": str(annotations),
                "model_dir": str(model_dir),
                "output_dir": str(prepared),
            },
        )
        assert response.status_code == 200, response.text
        assert str(tmp_path) not in " ".join(response.json()["command"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            report = client.get("/v1/fine-tuning").json()
            if report["status"] in {"complete", "failed"}:
                break
            time.sleep(0.01)

    assert report["status"] == "complete", report
    output = "\n".join(row["message"] for row in report["logs"])
    assert "--label-source" in output
    assert "<external>/corpus.jsonl" in output
    assert "<external>/annotations.jsonl" in output
    assert str(tmp_path) not in output
    assert report["exit_code"] == 0


def test_fine_tuning_rejects_missing_dataset_path(tmp_path) -> None:
    training_script = tmp_path / "training_stub.py"
    training_script.write_text("print('unused')\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with TestClient(
        create_app(FakePredictor, fine_tuning_script=training_script)
    ) as client:
        response = client.post(
            "/v1/fine-tuning/prepare",
            json={
                "model_dir": str(model_dir),
                "corpus": str(tmp_path / "missing.jsonl"),
                "annotations": str(tmp_path / "also-missing.jsonl"),
            },
        )

    assert response.status_code == 422
    assert "file does not exist" in response.json()["detail"]
    assert str(tmp_path) not in response.json()["detail"]


def test_fine_tuning_defaults_are_portable_paths() -> None:
    with TestClient(create_app(FakePredictor)) as client:
        defaults = client.get("/v1/fine-tuning").json()["defaults"]

    assert "project_root" not in defaults
    assert all(not Path(value).is_absolute() for value in defaults.values() if value)
    assert defaults["corpus"] == "artifacts/independent-annotations/blind_articles.jsonl"


def test_fine_tuning_is_disabled_by_default_on_public_bind(monkeypatch) -> None:
    monkeypatch.setenv("LAYA_HOST", "0.0.0.0")
    monkeypatch.delenv("LAYA_ENABLE_FINE_TUNING", raising=False)

    with TestClient(create_app(FakePredictor)) as client:
        page = client.get("/fine-tuning")
        api = client.get("/v1/fine-tuning")

    assert page.status_code == 403
    assert api.status_code == 403
    assert "LAYA_ENABLE_FINE_TUNING" in api.json()["detail"]


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
        inference = client.post(
            "/v1/adverse-media",
            json={"entity_name": "Acme Corp", "article": "Article text"},
        )

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "loaded": [],
        "error": "model startup failed; check the CrimeLaya server logs",
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

    assert "model.safetensors" in message
    assert str(model_path) not in message
    assert "Runtime network downloads are disabled" in message


def test_benchmark_scores_sequential_predictions_without_exposing_gold(tmp_path) -> None:
    corpus = tmp_path / "blind.jsonl"
    gold = tmp_path / "gold.jsonl"
    articles = [
        {"article_id": article_id("Acme", "Acme was charged."), "entity_name": "Acme", "article": "Acme was charged."},
        {"article_id": article_id("Beta", "Beta assisted investigators."), "entity_name": "Beta", "article": "Beta assisted investigators."},
    ]
    labels = [
        {"article_id": articles[0]["article_id"], "label": 2, "label_name": "bad_guy", "annotator": "gpt-5.6-terra"},
        {"article_id": articles[1]["article_id"], "label": 1, "label_name": "good_guy", "annotator": "gpt-5.6-terra"},
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
        started = client.post("/v1/benchmark/start", json={"routing_mode": "english"})
        assert started.status_code == 200
        for _ in range(100):
            report = client.get("/v1/benchmark").json()
            if report["status"] == "complete":
                break
            time.sleep(0.01)

    assert report["status"] == "complete", report["error"]
    assert report["completed"] == 2
    assert report["confusion_matrices"]["gpt"]["values"] == [[0, 1], [0, 1]]
    assert report["confusion_matrices"]["human"]["values"] == [[0, 1], [0, 1]]
    assert report["metrics"]["gpt"]["accuracy"] == 0.5
    assert report["probability_series"] == [
        {"index": 1, "negative": -91.0, "positive": 9.0},
        {"index": 2, "negative": -91.0, "positive": 9.0},
    ]
    saved = [json.loads(line) for line in (tmp_path / "laya-live.predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 2
    assert saved[0]["probabilities"] == {"negative": 0.91, "positive": 0.09}
    assert saved[0]["confidence"] == 0.91
    assert saved[0]["human_label"] == 2
    assert saved[0]["gpt_label"] == 2
    assert report["audit"]["annotators"] == {"gpt-5.6-terra": 2}
    assert all(set(state) == {"article", "entity_name"} for state, _, _ in predictor.calls)
    assert all("label" not in state and "annotator" not in state for state, _, _ in predictor.calls)

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


def test_benchmark_runs_articles_without_gpt_gold(tmp_path) -> None:
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
    assert completed["confusion_matrices"]["gpt"]["compared"] == 0


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
        assert len(client.get("/v1/benchmark/runs").json()["runs"]) == 2

        assert client.post("/v1/benchmark/reset").json()["completed"] == 0
        load_response = client.post(
            "/v1/benchmark/runs/load", json={"run_id": first["run_id"]}
        )
        assert load_response.status_code == 200, load_response.text
        loaded = load_response.json()

    assert loaded["status"] == "complete"
    assert loaded["completed"] == 1
    assert loaded["current"]["article_id"] == "one"
