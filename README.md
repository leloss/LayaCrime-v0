# LayaCrime

LayaCrime is an open-source workbench for **Criminal Sentiment Analysis** with [Laya](https://github.com/NandhaKishorM/laya). The platform lets users define categories and prompts, import reviewed annotation sets, benchmark decision and language models, and fine-tune their own LayaCrime checkpoints.

This repository's public checkpoint and datasets form **LayaCrime.v0**: the smallest reproducible CSA release, limited to the foundational binary decision of adverse criminal association versus no such association. Version zero is a reference implementation and starting point, not the limit of the platform's taxonomy.

Given one article and one named entity, the task produces one of two decisions:

- **Negative**: the article credibly associates the entity with alleged, investigated, charged, convicted, sanctioned, or admitted criminal behavior or intent.
- **Positive**: it does not make that association, or mentions the entity only as a victim, witness, investigator, authority, or unrelated party.

The model evaluates the named entity, not the article's overall tone and not every person or organization in it. CSA is distinct from generic sentiment analysis, crime-topic detection, and legal judgment.

## What you can do

The server exposes three connected workflows:

| Workflow | URL | Purpose |
| --- | --- | --- |
| Individual Test | `/` | Inspect one article, confidence, probabilities, routing, and review status. |
| Benchmark | `/benchmark` | Compare Laya checkpoints, hosted models, and local GGUF models on one labeled corpus. |
| Fine-Tuning | `/fine-tuning` | Prepare reviewed labels, train Laya, inspect reports, and optionally publish an export to Hugging Face. |

OpenAPI documentation is available at `/docs`; health and runtime diagnostics are available at `/health` and `/diagnostics`.

## Publication boundary

This repository is intentionally small. This public repo contains the application and two cleared dataset bundles that should help users get started.

| Included in this repository | Acquired or created locally |
| --- | --- |
| FastAPI service and browser workbench | Base [Laya weights](https://huggingface.co/convaiinnovations/laya) |
| Benchmark and fine-tuning code | Published [LayaCrime.v0 checkpoint](https://huggingface.co/leloss/layacrime) |
| Academic comparison implementations and tests | Pinned upstream source checkouts and released checkpoints |
| Public tuning set with 2,000 rows | Local GGUF language models selected by the user |
| Independent public holdout with 1,000 rows | Custom datasets, prepared tensors, checkpoints, reports, and benchmark runs |
| Dataset manifests, hashes, licenses, and release checks | API credentials and Hugging Face tokens |

Model files are downloaded into ignored directories during setup. Private datasets, generated artifacts, saved runs, and all weights remain outside the Git release. See [models/README.md](models/README.md) for the model layout and [datasets/README.md](datasets/README.md) for the complete bundle contract.

The article manuscript, its build files, and generated PDFs are deliberately excluded from this repository. The public release contains the code and data needed to rerun its experiments, not the article source.

The entire `third_party/` directory is also ignored and is not published to GitHub. Setup recreates it from pinned upstream sources: Laya v0.3.10 is cloned into `third_party/laya`, and the Ubuntu setup clones and builds the tested llama.cpp revision under `third_party/llama.cpp`. Local checkouts and native build products therefore remain reproducible installation state rather than repository contents.

## Deploy on an Ubuntu 22.04 GPU host

The supported full setup targets Ubuntu 22.04 with an NVIDIA GPU, Python 3.10-3.13, and a working NVIDIA driver version 525 or newer. The default llama.cpp build targets CUDA architecture 75, used by the Tesla T4.

From the repository root:

```bash
./scripts/setup_environment.sh
./scripts/run_ui.sh
```

`setup_environment.sh` performs the complete host setup:

1. Installs Ubuntu build prerequisites with `sudo` when needed.
2. Creates an isolated Python environment under `~/.cache/laya-adverse-media/venv`.
3. Selects and installs a PyTorch CUDA build compatible with the installed driver.
4. Installs a compatible CUDA 12 toolkit without replacing the driver. CUDA 11 is rejected because current llama.cpp releases require a newer compiler toolchain.
5. Builds llama.cpp and runs a real CUDA initialization probe.
6. Downloads Laya and LayaCrime from Hugging Face into ignored model directories.
7. Verifies that PyTorch can see the GPU.

The launcher binds to `127.0.0.1:8000` by default. Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/benchmark
http://127.0.0.1:8000/fine-tuning
```

Setup does not install or replace the NVIDIA driver. On managed hosts, set `LAYA_AUTO_INSTALL_SYSTEM_DEPS=false` and provision the required system packages separately. To provision or repair only the local language-model runtime, run `./scripts/setup_llama_cpp.sh`. It adopts an existing pre-stamp build when its source revision, CMake cache, CUDA architecture, and device enumeration are valid; otherwise it exits immediately for an already verified runtime, resumes a compatible interrupted build incrementally, and cleans the build directory only when its revision, CUDA compiler, architecture, or CUDA configuration changed.

## Set up on Windows

Windows setup is useful for development, API testing, dataset work, and decision-model evaluation. The automated CUDA llama.cpp build is Linux-specific.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
& ..\.venv-laya\Scripts\laya-adverse-media.exe
```

The script creates `../.venv-laya`, clones the pinned Laya source under `third_party/laya`, installs the project, and downloads the published Laya and LayaCrime weights. Downloads use the Windows certificate store, including organization-managed TLS roots. It does not create `third_party/llama.cpp`; local GGUF installation and serving require the supported Ubuntu GPU setup or a separately supplied `LAYA_LLAMA_SERVER` executable.

## Classify one article

The Individual Test screen is the fastest way to inspect behavior. Choose a decision model, provide the entity and article, and review the decision, probability distribution, routing information, and review flag.

The same operation is available through the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/adverse-media \
  -H "Content-Type: application/json" \
  --data @examples/request.json
```

Request:

```json
{
  "entity_name": "Acme Corp",
  "article": "Authorities charged Acme Corp with falsifying invoices. The company denies the allegations.",
  "routing_mode": "auto"
}
```

Response fields include `decision`, `confidence`, positive and negative `probabilities`, `needs_review`, and routing metadata. The probabilities are temperature-scaled class probabilities. `confidence` is Laya's normalized entropy certainty, not the winning class probability. `needs_review` is true when that certainty falls below `LAYA_REVIEW_THRESHOLD`, which defaults to `0.80`; tune the threshold on a held-out calibration set for the desired review-rate/error-coverage tradeoff.

Inference runs offline after setup. The service validates local model files before importing Laya, sets `HF_HUB_OFFLINE=1`, and does not silently fetch missing weights at request time.

## Benchmark models

The Benchmark Console keeps task-specific classifiers and general-purpose language models distinct:

- **Decision Models** contains the Laya router, the published LayaCrime.v0 checkpoint, and compatible local fine-tuned exports.
- **Language Models** contains configured Azure deployments and installable local GGUF models served by llama.cpp.

The bundled language-model catalog includes Qwen3 8B/14B, Qwen3.8 27B, Qwen3.6 and Qwen3.5 35B-A3B, Gemma 4 E4B/12B, two gpt-oss 20B quantizations, DeepSeek-R1-Distill-Qwen 14B, Mistral Small 3.2 24B, Devstral Small 2 24B, and Ministral 3 14B reasoning/instruct configurations. Selecting an unavailable GGUF model opens an installation monitor. Repository and file fields remain editable before download. The catalog uses single-file GGUF artifacts verified against Hugging Face metadata; models larger than T4 VRAM rely on llama.cpp's automatic layer offload to CPU memory. A model is not marked ready until its download receipt, GGUF signature, llama.cpp CUDA device, running process, and substantial VRAM allocation have all been verified.

The project pins a tested llama.cpp revision rather than tracking its moving `master` branch. llama.cpp remains the local runtime because it directly supports the catalog's GGUF files, structured OpenAI-compatible responses, and efficient T4 offload. Ollama and llama-cpp-python use the same underlying inference implementation, while replacing it with vLLM would require a different model-distribution and memory contract. Override `LLAMA_CPP_REVISION` only when deliberately qualifying a newer revision.

Azure credentials are requested only when a hosted model is selected. They remain in process memory, are cleared from the browser field, and are not written to run metadata. Hosted responses use structured output with `store=false`; users remain responsible for their tenant's retention and processing configuration.

A typical benchmark run is:

1. Select a dataset and ground-truth annotation set.
2. Select a decision model or language model.
3. Review or edit the task prompt and criteria.
4. Choose a sample limit or run the complete compatible corpus.
5. Inspect aggregate metrics, timing, costs, the confusion matrix, and individual errors.
6. Reload or delete timestamped runs from the run library.

The selected model receives only the article, entity name, question, and criteria. The application retrieves the ground-truth label only after inference returns. Completed and partial runs are stored under ignored `artifacts/benchmark-runs/<dataset-id>/` directories with corpus and annotation fingerprints. A saved run cannot be loaded against modified inputs with the same IDs.

## Reproduce the academic comparisons

The release includes separate command-line implementations and tests for every academic comparison. Generated predictions and reports go under ignored `artifacts/` directories. Install the local reimplementation and Tartu dependencies with:

```bash
python -m pip install --editable ".[academic]"
python -m spacy download en_core_web_sm
```

Run the Khandpur entity-relevance component reimplementation on its public tuning split and the public holdout:

```bash
python scripts/benchmark_academic_baselines.py \
  --model khandpur \
  --training-corpus datasets/adverse-media-public-tuning-2000/corpus.jsonl \
  --training-labels datasets/adverse-media-public-tuning-2000/annotations/human.jsonl \
  --test-corpus datasets/adverse-media-public-holdout-1000/corpus.jsonl \
  --test-labels datasets/adverse-media-public-holdout-1000/annotations/human.jsonl \
  --output-root artifacts/academic-baselines
```

The Tartu transfer baseline uses its released training archives from a pinned local checkout:

```bash
git clone https://github.com/kristjanr/ut-ml-adverse-media third_party/ut-ml-adverse-media
git -C third_party/ut-ml-adverse-media checkout 12fa6ada0a6f46ce098a654e39aea0ebbf1d55f5
python scripts/benchmark_tartu_baseline.py \
  --test-corpus datasets/adverse-media-public-holdout-1000/corpus.jsonl \
  --test-labels datasets/adverse-media-public-holdout-1000/annotations/human.jsonl \
  --output-root artifacts/academic-baselines
```

NewsMTSC requires its own Python 3.10 environment because the pinned upstream package requires Python below 3.12, Transformers 4.17--4.24, and PyTorch below 2.1. Clone the exact tested source revision and install it in that environment:

```bash
python3.10 -m venv .venv-newsmtsc
source .venv-newsmtsc/bin/activate
git clone https://github.com/fhamborg/NewsMTSC third_party/NewsMTSC
git -C third_party/NewsMTSC checkout b9d9b79704ed1b35cecaf1d7c2343dc1bd734fb7
python -m pip install truststore
python -m pip install --editable third_party/NewsMTSC
python scripts/benchmark_newsmtsc_checkpoint.py \
  --test-corpus datasets/adverse-media-public-holdout-1000/corpus.jsonl \
  --test-labels datasets/adverse-media-public-holdout-1000/annotations/human.jsonl \
  --output-root artifacts/academic-baselines
```

Each runner supports `--limit-test` for an individual smoke test. The Khandpur runner additionally supports `--limit-training`. `scripts/compare_academic_baselines.py` consumes their prediction ledgers and produces paired McNemar and bootstrap comparisons against a LayaCrime ledger. Hosted article baselines are independently reproducible with `scripts/benchmark_public_holdout_llms.py`; credentials are read from the ignored `.env.local`, and outputs remain under `artifacts/llm-benchmark-runs/`.

## Public datasets

The release contains two natural, non-synthetic bundles:

| Dataset | Purpose | Rows | Class balance |
| --- | --- | ---: | --- |
| `adverse-media-public-tuning-2000` | Training and model selection | 2,000 | 1,000 positive / 1,000 negative |
| `adverse-media-public-holdout-1000` | Final independent evaluation only | 1,000 | 557 positive / 443 negative |

Both are human annotated, limit repeated entities, and contain original entity/article pairs without synthetic substitutions. The two bundles have zero article-ID overlap and zero normalized-article overlap.

Do not use the public holdout for prompt editing, strategy selection, checkpoint selection, calibration, or threshold tuning. Use the 2,000-row tuning bundle for those choices, then evaluate the frozen model and settings on the holdout.

Each bundle includes a manifest with its purpose, prompt, record counts, SHA-256 hashes, label schema, license, and derivation metadata. Both public bundles are marked `cleared-for-public-release` and distributed under Apache-2.0.

## Use your own data

A dataset bundle owns one corpus, one or more compatible annotation sets, and the prompt used for preparation and evaluation:

```text
datasets/<dataset-id>/
|-- dataset.json
|-- corpus.jsonl
`-- annotations/
    `-- <annotation-id>.jsonl
```

Corpus rows require a stable ID, the entity being judged, and article text:

```json
{"article_id":"article-001","entity_name":"Acme Corp","article":"Article text..."}
```

Annotation rows join by `article_id`. Integer label `1` means positive/no criminal association; label `2` means negative/criminal association:

```json
{"article_id":"article-001","label":2,"label_name":"negative","annotator":"human"}
```

Set the manifest purpose deliberately:

- `training-only` is available to Fine-Tuning and excluded from Benchmark.
- `training-and-evaluation` is available to both workflows.
- `evaluation-only` is available to Benchmark and excluded from Fine-Tuning.

Point `LAYA_DATASETS_DIR` to the parent bundle directory and restart the service. Benchmark can also import an additional annotation JSONL for an existing corpus. Keep independent holdouts in separate bundles and never place related entities or duplicate normalized article text across train and test partitions.

## Fine-tune Laya

Fine-tuning requires a CUDA GPU and reviewed labels. The web workflow is:

1. Open `/fine-tuning` and select a training-compatible dataset and annotation set.
2. Review the dataset prompt and choose a group-aware partition strategy.
3. Prepare tensors. Preparation validates rows, joins labels by stable ID, tokenizes the production prompt, and records a deterministic manifest.
4. Select a training strategy and inspect every optimization value.
5. Train, monitor calibration metrics, and load the completed report.
6. Evaluate the exported checkpoint in Individual Test and Benchmark.
7. Optionally create or update a Hugging Face model repository after training succeeds.

For the public tuning bundle, start with **Public natural staged**. The trainer keeps connected entity/article groups in one partition, balances A/B option order, evaluates epoch zero, selects the checkpoint with the lowest calibration loss, and applies post-training temperature calibration. It never trains on an `evaluation-only` bundle.

Exports are written under `models/fine-tuned/<model-name>/` and discovered automatically. A complete export includes model weights, tokenizer and encoder files, `rl_agent_config.json`, `training_report.json`, and `data_manifest.json`. The stored prompt follows the checkpoint into Individual Test and Benchmark.

For unattended Linux training, configure `.env.local`:

```dotenv
LAYA_ACTIVE_DATASET=adverse-media-public-tuning-2000
LAYA_BENCHMARK_LABEL_SET=human
```

Then run:

```bash
./scripts/train_remote.sh
```

The launcher prepares missing tensors and uses all visible GPUs by default. Set `NPROC_PER_NODE`, `OUTPUT_DIR`, or other documented trainer arguments when the host requires explicit control. See [docs/fine-tuning.md](docs/fine-tuning.md) for partition protocols, training parameters, output files, evaluation, and Hugging Face publication.

## Configuration

Copy `.env.example` to `.env.local`; the application entry point loads it automatically.

| Variable | Purpose |
| --- | --- |
| `LAYA_HOST`, `LAYA_PORT` | HTTP bind address and port. Defaults: `127.0.0.1:8000`. |
| `LAYA_DEVICE` | Laya inference/training device. Linux launcher default: `cuda`. |
| `LAYA_REQUIRE_CUDA` | Require verified GGUF GPU offload instead of CPU fallback. Linux launcher default: `true`. |
| `LAYA_DATASETS_DIR` | Parent directory containing dataset bundles. |
| `LAYA_ACTIVE_DATASET` | Dataset selected at startup. |
| `LAYA_BENCHMARK_LABEL_SET` | Annotation set selected at startup. |
| `LAYA_MODEL_BUNDLE` | Alternate local Laya Hugging Face cache bundle. |
| `LAYA_FINE_TUNED_MODELS_DIR` | Directory containing compatible exports. |
| `LAYA_ENABLE_FINE_TUNING` | Explicitly enable training when not bound to loopback. |
| `LAYA_FINE_TUNING_ALLOWED_ROOTS` | Filesystem roots that training endpoints may access. |
| `LAYA_REVIEW_THRESHOLD` | Confidence threshold below which API responses require review. |

Setup repository IDs and revisions can also be overridden with `LAYA_HF_REPO_ID`, `LAYA_HF_REVISION`, `LAYACRIME_HF_REPO_ID`, and `LAYACRIME_HF_REVISION`.

## Docker

The Docker image contains the application and LayaCrime.v0 public datasets, but no weights. It is intended for inference with externally mounted Laya and LayaCrime.v0 directories; fine-tuning is disabled by default.

```bash
docker build -t layacrime .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v /path/to/models--convaiinnovations--laya:/models/laya:ro \
  -v /path/to/fine-tuned:/models/fine-tuned:ro \
  layacrime
```

The container binds internally to `0.0.0.0:8000`, while the example publishes it only on host loopback. The image installs the pinned Laya Git dependency directly from `pyproject.toml`; it does not use `third_party/`. It also does not bundle local GGUF files or a CUDA llama.cpp runtime.

## Security and operations

- The application has no built-in authentication. Keep the default loopback bind, or place non-loopback deployments behind an authenticated TLS reverse proxy with request limits.
- Fine-tuning is enabled automatically only on loopback. Restrict `LAYA_FINE_TUNING_ALLOWED_ROOTS`; never set it to a filesystem root.
- Benchmark ledgers and training reports may contain entity names, predictions, rationales, and text-derived metadata. Treat `artifacts/` and custom datasets as sensitive.
- Keep API keys and Hugging Face tokens in environment variables or a secret manager. Do not commit `.env.local`.
- A model score is not a finding of guilt. Review source quality, named-entity resolution, jurisdiction, recency, and false-negative behavior before operational use.

## Reproducibility and release checks

Dataset manifests, prepared-data manifests, saved runs, and training reports preserve hashes and provenance at each stage. Public-release validation rejects models, private datasets, generated artifacts, saved runs, and weight files:

```bash
python scripts/check_release_contents.py
```

The release checker expects exactly the cleared public dataset boundary, rejects article files, and requires the academic and hosted-baseline runners and their focused tests. Public-bundle construction and its private source inputs remain in the separate annotation-production project and are not distributed here.

## License and notices

The project code and the two public dataset bundles are licensed under Apache-2.0. Third-party software and model artifacts retain their own licenses and terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or deployment.