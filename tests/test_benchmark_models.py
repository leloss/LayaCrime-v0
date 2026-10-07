import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_adverse_media.benchmark_models import (
    BENCHMARK_MODELS,
    DECISION_SCHEMA,
    GenerativeBenchmarkRuntime,
    TemporaryCloudModelError,
    _LlamaServerHTTPError,
    _LlamaServerResponseError,
    _LlamaServerTransportError,
    _matching_orphaned_server_pids,
    benchmark_model_options,
    model_installation_receipt_path,
    register_custom_model,
    registered_model_by_id,
    resolve_llama_server,
    usage_cost,
)


def test_matching_orphaned_server_pids_requires_exact_server_model_and_ppid_one(
    tmp_path: Path,
) -> None:
    server = tmp_path / "llama-server"
    models_dir = tmp_path / "models"
    model = models_dir / "model.gguf"
    proc_root = tmp_path / "proc"
    for process_id, parent_id, executable, model_path in (
        (101, 1, server, model),
        (102, 99, server, model),
        (103, 1, tmp_path / "other-server", model),
        (104, 1, server, tmp_path / "other-models" / "other.gguf"),
        (105, 1, server, models_dir / "other.gguf"),
    ):
        process = proc_root / str(process_id)
        process.mkdir(parents=True)
        (process / "stat").write_text(
            f"{process_id} (llama-server) S {parent_id} 0 0 0",
            encoding="utf-8",
        )
        (process / "cmdline").write_bytes(
            b"\0".join(
                (
                    str(executable).encode(),
                    b"--model",
                    str(model_path).encode(),
                    b"--port",
                    b"8000",
                    b"",
                )
            )
        )

    assert _matching_orphaned_server_pids(server, models_dir, proc_root) == [101, 105]


def install_test_model(path: Path, content: bytes = b"GGUFmodel") -> None:
    path.write_bytes(content)
    model_installation_receipt_path(path).write_text(
        f'{{"size": {len(content)}}}', encoding="utf-8"
    )


def test_benchmark_model_catalog_has_unique_valid_entries(tmp_path: Path) -> None:
    ids = [model.id for model in BENCHMARK_MODELS]
    options = benchmark_model_options(tmp_path, llama_server="llama-server")

    assert len(ids) == len(set(ids))
    assert all(identifier == identifier.casefold() for identifier in ids)
    assert all(model.repo_id and model.filename for model in BENCHMARK_MODELS if model.provider == "llama.cpp")
    assert all(not option["installed"] for option in options if option["provider"] == "llama.cpp")
    assert all(option["credential_required"] for option in options if option["provider"] == "azure")
    assert {
        model.id: model.azure_deployment
        for model in BENCHMARK_MODELS
        if model.provider == "azure"
    } == {
        "gpt-5-6-luna": "gpt-5.6-luna",
        "gpt-5-4-nano": "gpt-5.4-nano",
        "grok-4-1-fast-reasoning": "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning": "grok-4-1-fast-non-reasoning",
        "gpt-6-luna": "gpt-6-luna",
        "deepseek-v4-1-flash": "DeepSeek-V4.1-Flash",
        "azure-custom-endpoint": None,
    }
    endpoint_action = next(model for model in options if model["id"] == "azure-custom-endpoint")
    assert endpoint_action["action_only"] is True
    assert endpoint_action["action_label"] == "Add endpoint…"
    assert {
        model.id: model.source_repo_id
        for model in BENCHMARK_MODELS
        if model.provider == "llama.cpp"
    } == {
        "qwen3-8b-q6-k": "Qwen/Qwen3-8B",
        "qwen3-14b-q4-k-m": "Qwen/Qwen3-14B",
        "gemma-4-e4b-q6-k": "google/gemma-4-E4B",
        "gemma-4-12b-q4-k-m": "google/gemma-4-12B",
        "gpt-oss-20b-mxfp4": "openai/gpt-oss-20b",
        "deepseek-r1-distill-qwen-14b-q4-k-m": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        ),
        "qwen3-8-27b-iq4-xs": "Qwen/Qwen3.8-27B",
        "ternary-bonsai-27b-q2-g64": "Qwen/Qwen3.6-27B",
        "qwen3-6-35b-a3b-q3-k-m": "Qwen/Qwen3.6-35B-A3B",
        "qwen3-5-35b-a3b-q3-k-m": "Qwen/Qwen3.5-35B-A3B",
        "gpt-oss-20b-q5-k-m": "openai/gpt-oss-20b",
        "mistral-small-3-2-24b-q4-k-m": (
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        ),
        "devstral-small-2-24b-q4-k-m": (
            "mistralai/Devstral-Small-2-24B-Instruct-2512"
        ),
        "ministral-3-14b-reasoning-q5-k-m": (
            "mistralai/Ministral-3-14B-Reasoning-2512"
        ),
        "ministral-3-14b-instruct-q8-0": (
            "mistralai/Ministral-3-14B-Instruct-2512"
        ),
    }


def test_custom_model_persists_repository_path_and_appears_in_catalog(
    tmp_path: Path,
) -> None:
    created = register_custom_model(
        tmp_path,
        "My Gemma",
        "owner/gemma-gguf",
        "weights/gemma-Q8_0.gguf",
    )

    loaded = registered_model_by_id(tmp_path, created.id)
    option = next(item for item in benchmark_model_options(tmp_path) if item["id"] == created.id)

    assert loaded is not None
    assert loaded.repo_id == "owner/gemma-gguf"
    assert loaded.huggingface_path == "weights/gemma-Q8_0.gguf"
    assert loaded.filename == "gemma-Q8_0.gguf"
    assert option["custom"] is True
    assert option["deletable"] is True


@pytest.mark.parametrize("path", ["../model.gguf", "/model.gguf", "model.bin"])
def test_custom_model_rejects_unsafe_or_non_gguf_paths(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(ValueError, match="relative .gguf path"):
        register_custom_model(tmp_path, "Unsafe", "owner/repo", path)


def test_local_activation_unloads_laya_before_starting_server(tmp_path: Path) -> None:
    events: list[str] = []
    model = next(model for model in BENCHMARK_MODELS if model.provider == "llama.cpp")
    install_test_model(tmp_path / str(model.filename))
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"runtime")

    class Process:
        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: int) -> None:
            pass

        def poll(self):
            return None

    runtime = GenerativeBenchmarkRuntime(
        tmp_path,
        lambda: events.append("unload"),
        llama_server=str(executable),
        process_factory=lambda *args, **kwargs: events.append("start") or Process(),
    )
    runtime._wait_until_ready = lambda: events.append("ready")

    runtime.activate(model.id)
    assert runtime.is_healthy is True
    runtime.close()

    assert events[:3] == ["unload", "start", "ready"]
    assert runtime.is_healthy is False


def test_runtime_can_activate_registered_custom_model(tmp_path: Path) -> None:
    model = register_custom_model(
        tmp_path, "Custom Qwen", "owner/repo", "weights/custom.gguf"
    )
    install_test_model(tmp_path / str(model.filename))
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"runtime")
    commands = []

    class Process:
        def terminate(self) -> None:
            pass

        def wait(self, timeout: int) -> None:
            pass

        def poll(self):
            return None

    runtime = GenerativeBenchmarkRuntime(
        tmp_path,
        lambda: None,
        llama_server=str(executable),
        process_factory=lambda command, **kwargs: commands.append(command) or Process(),
    )
    runtime._wait_until_ready = lambda: None

    runtime.activate(model.id, spec=model)
    runtime.close()

    assert commands[0][commands[0].index("--model") + 1].endswith("custom.gguf")


def test_cuda_activation_uses_automatic_memory_fitting(tmp_path: Path) -> None:
    captured: list[str] = []
    model = next(model for model in BENCHMARK_MODELS if model.provider == "llama.cpp")
    install_test_model(tmp_path / str(model.filename))
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"runtime")

    class Process:
        pid = 123

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int) -> None:
            pass

        def poll(self):
            return None

    runtime = GenerativeBenchmarkRuntime(
        tmp_path,
        lambda: None,
        llama_server=str(executable),
        process_factory=lambda command, **kwargs: captured.extend(command) or Process(),
        require_gpu=True,
    )
    runtime._resolve_cuda_device = lambda _: "CUDA0"
    runtime._wait_until_ready = lambda: None
    runtime._assert_gpu_offload = lambda: None

    runtime.activate(model.id)
    runtime.close()

    assert captured[captured.index("--device") + 1] == "CUDA0"
    assert "--n-gpu-layers" not in captured
    assert captured[captured.index("--fit") + 1] == "on"
    assert captured[captured.index("--fit-target") + 1] == "1024"


def test_qwen38_keeps_full_context_with_automatic_memory_fitting(tmp_path: Path) -> None:
    captured: list[str] = []
    model = next(model for model in BENCHMARK_MODELS if model.id == "qwen3-8-27b-iq4-xs")
    install_test_model(tmp_path / str(model.filename))
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"runtime")

    class Process:
        pid = 123

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int) -> None:
            pass

        def poll(self):
            return None

    runtime = GenerativeBenchmarkRuntime(
        tmp_path,
        lambda: None,
        llama_server=str(executable),
        process_factory=lambda command, **kwargs: captured.extend(command) or Process(),
        require_gpu=True,
    )
    runtime._resolve_cuda_device = lambda _: "CUDA0"
    runtime._wait_until_ready = lambda: None
    runtime._assert_gpu_offload = lambda: None

    runtime.activate(model.id)
    runtime.close()

    assert captured[captured.index("--ctx-size") + 1] == "16384"
    assert "--n-gpu-layers" not in captured
    assert captured[captured.index("--fit-target") + 1] == "1024"


@pytest.mark.parametrize(
    ("content", "receipt", "state"),
    [
        (b"GGUFpartial", None, "unverified"),
        (b"GGUFpartial", '{"size": 99}', "incomplete"),
        (b"broken", '{"size": 6}', "corrupt"),
        (b"GGUFready", '{not-json', "corrupt"),
    ],
)
def test_catalog_resets_broken_local_installations(
    tmp_path: Path, content: bytes, receipt: str | None, state: str
) -> None:
    model = next(model for model in BENCHMARK_MODELS if model.provider == "llama.cpp")
    model_path = tmp_path / str(model.filename)
    model_path.write_bytes(content)
    if receipt is not None:
        model_installation_receipt_path(model_path).write_text(receipt, encoding="utf-8")

    option = next(item for item in benchmark_model_options(tmp_path) if item["id"] == model.id)

    assert option["installed"] is False
    assert option["installation_state"] == state
    assert option["installation_error"]
    assert option["deletable"] is True


def test_usage_cost_uses_model_token_rates() -> None:
    luna = next(model for model in BENCHMARK_MODELS if model.id == "gpt-5-6-luna")

    assert usage_cost(luna, 1_000_000, 100_000) == pytest.approx(0.32)


def test_generative_schema_does_not_request_self_reported_confidence() -> None:
    assert set(DECISION_SCHEMA["properties"]) == {"label"}
    assert DECISION_SCHEMA["required"] == ["label"]


def test_generative_prediction_is_one_hot() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    runtime._predict_local = lambda article, entity: (
        {"label": 2},
        {"input_tokens": 10, "output_tokens": 4},
    )

    result = runtime.predict("Article", "Entity")

    answer = result["answers"]["criminal_association"]
    assert answer["confidence"] == 1.0
    assert answer["probabilities"] == {"A": 1.0, "B": 0.0}


def test_azure_max_output_retries_with_low_reasoning_and_aggregates_usage() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "gpt-5-6-luna"
    )
    runtime._deployment = "luna-deployment"
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return type("Response", (), {
                    "status": "incomplete",
                    "incomplete_details": type(
                        "Details", (), {"reason": "max_output_tokens"}
                    )(),
                    "usage": type(
                        "Usage", (), {"input_tokens": 100, "output_tokens": 4096}
                    )(),
                })()
            return type("Response", (), {
                "status": "completed",
                "output_text": '{"label":2}',
                "usage": type(
                    "Usage", (), {"input_tokens": 100, "output_tokens": 32}
                )(),
            })()

    runtime._client = type("Client", (), {"responses": Responses()})()

    result, usage = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 2}
    assert [call["max_output_tokens"] for call in calls] == [4096, 8192]
    assert all(call["reasoning"] == {"effort": "low"} for call in calls)
    assert usage == {"input_tokens": 200, "output_tokens": 4128}


def test_grok_azure_uses_chat_completions() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model
        for model in BENCHMARK_MODELS
        if model.id == "grok-4-1-fast-non-reasoning"
    )
    runtime._deployment = str(runtime._active.azure_deployment)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="1"),
                    finish_reason="stop",
                )],
                usage=None,
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
        responses=SimpleNamespace(create=lambda **kwargs: pytest.fail("Responses API used")),
    )

    result, _ = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 1}
    assert calls[0]["model"] == "grok-4-1-fast-non-reasoning"
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][1] == {
        "role": "user",
        "content": '{"entity_name": "Entity", "article": "Article"}',
    }


@pytest.mark.parametrize(
    ("model_id", "input_price", "output_price"),
    [
        ("grok-4-1-fast-reasoning", 0.20, 0.50),
        ("grok-4-1-fast-non-reasoning", 0.20, 0.50),
        ("gpt-6-luna", 0.10, 0.50),
        ("deepseek-v4-1-flash", 0.30, 1.20),
    ],
)
def test_new_azure_model_prices(
    model_id: str, input_price: float, output_price: float
) -> None:
    model = next(model for model in BENCHMARK_MODELS if model.id == model_id)

    assert model.input_usd_per_million == input_price
    assert model.output_usd_per_million == output_price


def test_deepseek_azure_retries_empty_chat_content_and_accepts_plain_label() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "deepseek-v4-1-flash"
    )
    runtime._deployment = str(runtime._active.azure_deployment)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="" if len(calls) == 1 else "2"),
                    finish_reason="length" if len(calls) == 1 else "stop",
                )],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
        responses=SimpleNamespace(create=lambda **kwargs: pytest.fail("Responses API used")),
    )

    result, usage = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 2}
    assert [call["max_tokens"] for call in calls] == [512, 2048]
    assert calls[0]["messages"][1]["content"] == (
        '{"entity_name": "Entity", "article": "Article"}'
    )
    assert usage == {"input_tokens": 20, "output_tokens": 8}


def test_deepseek_azure_retries_empty_http_response_body() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "deepseek-v4-1-flash"
    )
    runtime._deployment = str(runtime._active.azure_deployment)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="1"),
                    finish_reason="stop",
                )],
                usage=None,
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result, _ = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 1}
    assert [call["max_tokens"] for call in calls] == [512, 2048]


@pytest.mark.parametrize(
    ("model_id", "content"),
    [
        ("grok-4-1-fast-non-reasoning", "Label: 2"),
        ("deepseek-v4-1-flash", '{"label":"2"}'),
    ],
)
def test_cloud_chat_accepts_unambiguous_label_variants(
    model_id: str, content: str
) -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(model for model in BENCHMARK_MODELS if model.id == model_id)
    runtime._deployment = str(runtime._active.azure_deployment)

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )],
                usage=None,
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result, _ = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 2}


def test_cloud_chat_retries_malformed_nonempty_content() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS
        if model.id == "grok-4-1-fast-non-reasoning"
    )
    runtime._deployment = str(runtime._active.azure_deployment)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content="No decision" if len(calls) == 1 else "2"
                    ),
                    finish_reason="stop",
                )],
                usage=None,
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result, _ = runtime._predict_azure("Article", "Entity")

    assert result == {"label": 2}
    assert [call["max_tokens"] for call in calls] == [512, 2048]


def test_deepseek_exhausted_empty_content_retries_shorter_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "deepseek-v4-1-flash"
    )
    runtime._deployment = str(runtime._active.azure_deployment)
    submitted_lengths = []

    class Completions:
        def create(self, **kwargs):
            submitted = json.loads(kwargs["messages"][1]["content"])["article"]
            submitted_lengths.append(len(submitted))
            content = "" if len(submitted) > 1_359 else '{"label":1}'
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )],
                usage=None,
            )

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result = runtime.predict("x" * 2_718, "Entity")

    assert submitted_lengths == [2_718, 2_718, 2_038, 2_038, 1_359]
    assert result["answers"]["criminal_association"]["choice"] == "B"
    assert result["routing"]["input"]["submitted_article_chars"] == 1_359


def test_azure_rate_limit_becomes_resumable_temporary_failure() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "grok-4-1-fast-reasoning"
    )
    runtime._deployment = str(runtime._active.azure_deployment)

    class RateLimitError(Exception):
        status_code = 429

    class Completions:
        def create(self, **kwargs):
            raise RateLimitError("no_capacity")

    runtime._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    with pytest.raises(TemporaryCloudModelError, match="resume the benchmark"):
        runtime._predict_azure("Article", "Entity")


def test_azure_bad_request_retries_with_distinct_shorter_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == "grok-4-1-fast-reasoning"
    )
    submitted_lengths = []

    class BadRequestError(Exception):
        status_code = 400

    def predict(article, entity_name):
        submitted_lengths.append(len(article))
        if len(article) > 1_359:
            raise BadRequestError("invalid request")
        return {"label": 1}, {"input_tokens": 10, "output_tokens": 2}

    runtime._predict_azure = predict

    result = runtime.predict("x" * 2_718, "Ravinder Singh")

    assert submitted_lengths == [2_718, 2_038, 1_359]
    assert result["routing"]["input"] == {
        "original_article_chars": 2_718,
        "submitted_article_chars": 1_359,
        "article_truncated": True,
    }


def test_custom_azure_endpoint_uses_entered_model_endpoint_and_key(
    tmp_path: Path, monkeypatch
) -> None:
    client_args = {}
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed", output_text='{"label":1}', usage=None
            )

    client = SimpleNamespace(responses=Responses(), close=lambda: None)

    def openai_client(**kwargs):
        client_args.update(kwargs)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_client))
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)

    runtime.activate(
        "azure-custom-endpoint",
        {
            "deployment": "review-model-v2",
            "endpoint": "https://review.openai.azure.com/openai/v1/",
            "api_key": "transient-key",
        },
    )
    result, _ = runtime._predict_azure("Article", "Entity")

    assert runtime.active_model == "azure-custom-endpoint"
    assert client_args["base_url"] == "https://review.openai.azure.com/openai/v1/"
    assert client_args["api_key"] == "transient-key"
    assert calls[0]["model"] == "review-model-v2"
    assert result == {"label": 1}


def test_local_bad_request_retries_with_smaller_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    submitted_lengths = []

    def predict(article: str, entity: str):
        submitted_lengths.append(len(article))
        if len(submitted_lengths) == 1:
            raise _LlamaServerHTTPError(400, "request exceeds context window")
        return {"label": 1}, {}

    runtime._predict_local = predict

    result = runtime.predict("x" * 60_000, "Entity")

    assert submitted_lengths == [50_000, 32_000]
    assert result["answers"]["criminal_association"]["probabilities"] == {
        "A": 0.0,
        "B": 1.0,
    }
    assert result["routing"]["input"]["submitted_article_chars"] == 32_000


def test_malformed_local_response_retries_with_smaller_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    submitted_lengths = []

    def predict(article: str, entity: str):
        submitted_lengths.append(len(article))
        if len(submitted_lengths) == 1:
            raise _LlamaServerResponseError("unterminated string")
        return {"label": 2}, {}

    runtime._predict_local = predict

    result = runtime.predict("x" * 60_000, "Entity")

    assert submitted_lengths == [50_000, 32_000]
    assert result["answers"]["criminal_association"]["probabilities"] == {
        "A": 1.0,
        "B": 0.0,
    }


def test_output_limit_response_is_not_retried_with_smaller_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    submitted_lengths = []

    def predict(article: str, entity: str):
        submitted_lengths.append(len(article))
        raise _LlamaServerResponseError(
            "output limit reached", finish_reason="length"
        )

    runtime._predict_local = predict

    with pytest.raises(_LlamaServerResponseError, match="output limit reached"):
        runtime.predict("x" * 60_000, "Entity")

    assert submitted_lengths == [50_000]


def test_transport_timeout_restarts_server_and_retries_shorter_article() -> None:
    runtime = GenerativeBenchmarkRuntime(Path("models"), lambda: None)
    spec = next(model for model in BENCHMARK_MODELS if model.provider == "llama.cpp")
    runtime._active = spec
    submitted_lengths = []
    activations = []

    def predict(article: str, entity: str):
        submitted_lengths.append(len(article))
        if len(submitted_lengths) == 1:
            raise _LlamaServerTransportError("request timed out")
        return {"label": 1}, {}

    def activate(model_id, credentials=None, selected_spec=None, **kwargs):
        activations.append(model_id)
        runtime._active = kwargs.get("spec") or selected_spec
        return model_id

    runtime._predict_local = predict
    runtime.activate = activate

    result = runtime.predict("x" * 60_000, "Entity")

    assert submitted_lengths == [50_000, 32_000]
    assert activations == [spec.id]
    assert result["answers"]["criminal_association"]["choice"] == "B"


def test_local_http_error_preserves_server_detail(tmp_path: Path, monkeypatch) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    runtime._base_url = "http://127.0.0.1:8000"

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            runtime._base_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"prompt exceeds context window"}}'),
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", reject
    )

    with pytest.raises(_LlamaServerHTTPError, match="prompt exceeds context window"):
        runtime._predict_local("Article", "Entity")


def test_local_unterminated_json_becomes_retryable_error(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    runtime._base_url = "http://127.0.0.1:8000"
    budgets = []

    def truncated(request, timeout):
        budgets.append(json.loads(request.data)["max_tokens"])
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"length","message":{"content":"{\\"label\\":\\""}}],'
            b'"usage":{}}'
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen",
        truncated,
    )

    with pytest.raises(
        _LlamaServerResponseError, match="finish_reason=length"
    ):
        runtime._predict_local("Article", "Entity")

    assert budgets == [128, 512]


def test_ministral_instruct_disables_reasoning_and_uses_larger_budget(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model
        for model in BENCHMARK_MODELS
        if model.id == "ministral-3-14b-instruct-q8-0"
    )
    runtime._base_url = "http://127.0.0.1:8000"
    captured = {}

    def respond(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"label\\":2}"}}],'
            b'"usage":{}}'
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", respond
    )

    result, _ = runtime._predict_local("Article", "Entity")

    assert result == {"label": 2}
    assert captured["max_tokens"] == 512
    assert captured["reasoning_format"] == "none"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_local_request_times_out_before_benchmark_appears_stuck(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.provider == "llama.cpp"
    )
    runtime._base_url = "http://127.0.0.1:8000"
    captured = {}

    def timeout(request, timeout):
        captured["timeout"] = timeout
        raise TimeoutError("model stalled")

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", timeout
    )

    with pytest.raises(RuntimeError, match="within 90 seconds"):
        runtime._predict_local("Article", "Entity")

    assert captured["timeout"] == 90


@pytest.mark.parametrize("model_id", ["gpt-oss-20b-mxfp4", "gpt-oss-20b-q5-k-m"])
def test_gpt_oss_uses_low_reasoning_effort_and_larger_output_budget(
    tmp_path: Path, monkeypatch, model_id: str
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == model_id
    )
    runtime._base_url = "http://127.0.0.1:8000"
    captured = {}

    def respond(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"label\\":2}"}}],'
            b'"usage":{}}'
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", respond
    )

    result, _ = runtime._predict_local("Article", "Entity")

    assert result == {"label": 2}
    assert captured["max_tokens"] == 2048
    assert captured["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_effort": "low",
    }


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek-r1-distill-qwen-14b-q4-k-m",
        "ministral-3-14b-reasoning-q5-k-m",
    ],
)
def test_reasoning_models_use_larger_output_budget_without_thinking(
    tmp_path: Path, monkeypatch, model_id: str
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model for model in BENCHMARK_MODELS if model.id == model_id
    )
    runtime._base_url = "http://127.0.0.1:8000"
    captured = {}

    def respond(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"label\\":2}"}}],'
            b'"usage":{}}'
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", respond
    )

    result, _ = runtime._predict_local("Article", "Entity")

    assert result == {"label": 2}
    assert captured["max_tokens"] == 4096
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    if model_id.startswith("ministral-3-"):
        assert captured["reasoning_format"] == "none"
    else:
        assert "reasoning_format" not in captured


def test_non_reasoning_model_keeps_short_non_thinking_request(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None)
    runtime._active = next(
        model
        for model in BENCHMARK_MODELS
        if model.provider == "llama.cpp"
        and not model.id.startswith("gpt-oss-20b-")
        and not model.id.startswith("deepseek-r1-distill-")
        and "reasoning" not in model.id
    )
    runtime._base_url = "http://127.0.0.1:8000"
    captured = {}

    def respond(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"label\\":1}"}}],'
            b'"usage":{}}'
        )

    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.urllib.request.urlopen", respond
    )

    runtime._predict_local("Article", "Entity")

    assert captured["max_tokens"] == 128
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_server_is_rediscovered_after_runtime_creation(tmp_path: Path) -> None:
    executable = tmp_path / "llama-server"

    assert resolve_llama_server(str(executable)) is None
    executable.write_bytes(b"runtime")
    assert resolve_llama_server(str(executable)) == str(executable.resolve())


def test_cuda_verification_accepts_gpu_layer_offload(tmp_path: Path) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None, require_gpu=True)
    runtime._log = (tmp_path / "llama-server.log").open("w+b")
    runtime._log.write(b"load_tensors: offloaded 41/41 layers to GPU\n")

    runtime._assert_gpu_offload()
    runtime.close()


def test_cuda_verification_rejects_cpu_only_runtime(tmp_path: Path) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None, require_gpu=True)
    runtime._log = (tmp_path / "llama-server.log").open("w+b")
    runtime._log.write(b"load_tensors: CPU model buffer size = 8123 MiB\n")

    with pytest.raises(RuntimeError, match="without confirmed CUDA layer offload"):
        runtime._assert_gpu_offload()

    assert runtime._log is None


def test_cuda_verification_explains_driver_runtime_mismatch(tmp_path: Path) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None, require_gpu=True)
    runtime._log = (tmp_path / "llama-server.log").open("w+b")
    runtime._log.write(
        b"ggml_cuda_init: failed to initialize CUDA: "
        b"CUDA driver version is insufficient for CUDA runtime version\n"
    )

    with pytest.raises(RuntimeError, match="toolkit newer than the installed NVIDIA driver"):
        runtime._assert_gpu_offload()

    assert runtime._log is None


def test_cuda_verification_accepts_process_with_model_in_vram(tmp_path: Path) -> None:
    runtime = GenerativeBenchmarkRuntime(tmp_path, lambda: None, require_gpu=True)
    runtime._log = (tmp_path / "llama-server.log").open("w+b")
    runtime._log.write(b"srv llama_server: model loaded\n")
    runtime._gpu_process_memory_mib = lambda: 6144

    runtime._assert_gpu_offload()
    runtime.close()


def test_cuda_device_resolution_uses_llama_server_device_list(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"runtime")
    monkeypatch.setattr(
        "laya_adverse_media.benchmark_models.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {
            "returncode": 0,
            "stdout": "Available devices:\n  CUDA0: Tesla T4\n",
            "stderr": "",
        })(),
    )

    assert GenerativeBenchmarkRuntime._resolve_cuda_device(str(executable)) == "CUDA0"