# CrimeLaya

**Local Criminality Sentiment Analysis for adverse media monitoring and detection.**

CrimeLaya is an open-source CSA engine for Risk and Financial Crime Compliance teams. It classifies how an article portrays a named person or organization in relation to criminal activity, returning a negative or positive association with calibrated probabilities and a review signal.

Built on [Laya](https://github.com/NandhaKishorM/laya), CrimeLaya uses a non-generative System 1 decision model instead of producing tokens autoregressively. It is designed for high-volume adverse media workflows that need fast, repeatable classification without sending every article to a remote general-purpose LLM or paying a per-request model fee.

![CrimeLaya benchmark showing criminality sentiment metrics, confusion matrices, throughput, and probability history](docs/assets/crimelaya-benchmark.png)

## Why CrimeLaya

- **Purpose-built CSA output.** Classifies the named entity as `negative` when the article credibly associates it with alleged, investigated, charged, convicted, sanctioned, or admitted criminal behavior, and `positive` when it does not.
- **Local and private.** Articles and entity names can remain inside your environment during inference.
- **Fast System 1 decisions.** No autoregressive response generation, prompt-length billing, or remote-model round trip is required.
- **Operational probabilities.** Every result includes both class probabilities, confidence, and a configurable `needs_review` signal for downstream triage.
- **Drop-in integration.** A small FastAPI service exposes a stable REST contract for screening platforms, case-management systems, batch pipelines, and analyst tools.
- **Portable compute.** Run inference on CPU or CUDA GPU. Fine-tuning supports one GPU or multi-GPU Distributed Data Parallel.
- **Open source.** The application is Apache-2.0 licensed and uses publicly available Laya checkpoints.

## Where It Fits

CrimeLaya is the criminality-association decision layer in an adverse media pipeline:

```text
News and media ingestion
        |
Entity matching and article selection
        |
CrimeLaya CSA inference
        |
Confidence threshold and review policy
        |
Alert, suppress, or route to an analyst
```

It can replace expensive generative-model calls for the repeated classification step while preserving your existing ingestion, entity resolution, workflow, and human-review controls.

## Decision Contract

Send an article and the entity that must be judged. CrimeLaya evaluates only that named entity, not every person or organization mentioned in the article.

```http
POST /v1/adverse-media
Content-Type: application/json
```

```json
{
  "entity_name": "Acme Corp",
  "article": "Authorities charged Acme Corp and two executives with falsifying invoices. The company denies the allegations.",
  "routing_mode": "auto"
}
```

```json
{
  "entity_name": "Acme Corp",
  "decision": "negative",
  "confidence": 0.94,
  "probabilities": {
    "negative": 0.94,
    "positive": 0.06
  },
  "needs_review": false,
  "routing": {
    "model": "english"
  }
}
```

`LAYA_REVIEW_THRESHOLD` controls the review boundary and defaults to `0.80`. The API also accepts `english` or `multilingual` routing when automatic routing is not appropriate.

## Quick Start

CrimeLaya requires Python 3.10 through 3.13. The first setup downloads the public Laya source and checkpoint bundle from GitHub and Hugging Face.

### Windows

```powershell
git clone https://github.com/leloss/crime-laya.git
cd crime-laya
Set-ExecutionPolicy -Scope Process Bypass
./scripts/setup.ps1
./.venv/Scripts/laya-adverse-media.exe
```

### Ubuntu Linux

Ubuntu 22.04 and 24.04 provide supported Python versions. Install the operating-system prerequisites, clone CrimeLaya, and create an isolated environment:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv

git clone https://github.com/leloss/crime-laya.git
cd crime-laya

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="convaiinnovations/laya", cache_dir="models")
PY

laya-adverse-media
```

Open the service at `http://127.0.0.1:8000/`, the API documentation at `http://127.0.0.1:8000/docs`, and diagnostics at `http://127.0.0.1:8000/diagnostics`.

### Start on a Remote Ubuntu Host

Run the service from the repository with a network bind. CPU inference works without a CUDA installation:

```bash
cd crime-laya
source .venv/bin/activate
LAYA_HOST=0.0.0.0 LAYA_PORT=8000 LAYA_DEVICE=cpu laya-adverse-media
```

For an NVIDIA host with a compatible PyTorch installation, replace `LAYA_DEVICE=cpu` with `LAYA_DEVICE=cuda`. Keep the process running with your normal service manager, and place it behind authentication and TLS before exposing it beyond a trusted network. If direct port access is appropriate for your environment, allow it through Ubuntu's firewall:

```bash
sudo ufw allow 8000/tcp
```

The benchmark is available at `http://SERVER_IP:8000/`. Fine-tuning process controls are disabled on network binds by default. On a protected administrative host, enable them explicitly:

```bash
LAYA_HOST=0.0.0.0 LAYA_DEVICE=cuda LAYA_ENABLE_FINE_TUNING=true laya-adverse-media
```

## CPU and GPU Deployment

CPU is the default and requires no CUDA installation:

```bash
LAYA_DEVICE=cpu laya-adverse-media
```

Use a compatible NVIDIA GPU by selecting CUDA:

```bash
LAYA_DEVICE=cuda laya-adverse-media
```

Deployment settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAYA_HOST` | `127.0.0.1` | Bind address |
| `LAYA_PORT` | `8000` | HTTP port |
| `LAYA_DEVICE` | Laya default | `cpu`, `cuda`, or another supported Torch device |
| `LAYA_REVIEW_THRESHOLD` | `0.80` | Confidence below which `needs_review` is true |
| `LAYA_PROJECT_ROOT` | source checkout or current directory | Deployment root containing `models/`, `scripts/`, and `artifacts/` |
| `LAYA_MODEL_BUNDLE` | `models/models--convaiinnovations--laya` | Hugging Face cache-style model bundle |
| `LAYA_MODEL_PATH` | unset | Optional English checkpoint override |
| `LAYA_ENABLE_FINE_TUNING` | local binds only | Enable the administrative fine-tuning UI and process-control API |

For a network service, set `LAYA_HOST=0.0.0.0` and place CrimeLaya behind your normal authentication, TLS, observability, and request-control layer. Fine-tuning controls are disabled automatically on non-loopback binds; enable them only on a trusted administrative deployment with `LAYA_ENABLE_FINE_TUNING=true`.

## Benchmark and Validation

The web console runs a blind labeled set through CrimeLaya and reports:

- accuracy, negative precision, recall, and F1;
- source and independent-annotation confusion matrices;
- per-article probabilities and decision history;
- throughput and model-inference timing;
- immutable named result snapshots for comparisons between checkpoints.

Provide your own corpus and annotation ledger through `LAYA_BENCHMARK_CORPUS`, `LAYA_BENCHMARK_GOLD`, and optionally `LAYA_BENCHMARK_SOURCE_DIR`. Corpus rows contain `article_id`, `entity_name`, and `article`; annotation rows contain the matching `article_id` and label `1` for positive/no association or `2` for negative association.

## Fine-Tuning for Your Risk Taxonomy

The included RLCD pipeline adapts Laya to your labeled adverse media decisions while keeping entities isolated across train, calibration, and test splits. It provides mixed precision, activation checkpointing, calibrated probabilities, held-out evaluation, resumable checkpoints, and separate encoder/head learning rates.

Open [http://127.0.0.1:8000/fine-tuning](http://127.0.0.1:8000/fine-tuning) to prepare a caller-selected corpus and annotation ledger, launch single or multi-GPU training, and monitor epoch, batch, loss, exploration, process state, and logs. Paths are relative to the CrimeLaya repository unless explicitly configured otherwise; datasets are not uploaded through the browser. Non-loopback deployments must explicitly enable this administrative interface.

Prepare a labeled dataset locally, then launch the portable GPU workflow:

```bash
chmod +x scripts/train_remote.sh
./scripts/train_remote.sh
```

The launcher detects the installed NVIDIA driver, creates an isolated environment, downloads the base checkpoint, and uses every visible GPU by default. Limit or select the process count with `NPROC_PER_NODE`:

```bash
NPROC_PER_NODE=1 ./scripts/train_remote.sh
NPROC_PER_NODE=2 ./scripts/train_remote.sh
```

Fine-tuned checkpoints can be served by setting `LAYA_MODEL_PATH` to the exported model directory. See [Fine-tuning CrimeLaya](docs/fine-tuning.md) for the data contract, single-GPU and DDP commands, resume behavior, and evaluation outputs.

## Integration Notes

- The model receives only the article and requested entity during inference.
- `routing_mode=auto` selects the English or multilingual checkpoint; callers can make the route explicit.
- Batch pipelines can use [scripts/predict_blind_corpus.py](scripts/predict_blind_corpus.py) directly or call the HTTP endpoint.
- Model weights, source corpora, annotations, predictions, and fine-tuning artifacts are intentionally excluded from this repository.
- CrimeLaya is a classification component, not an autonomous compliance decision-maker. Organizations remain responsible for entity resolution, thresholds, review policy, validation, monitoring, and regulatory controls.

## Documentation

- [Fine-tuning and GPU deployment](docs/fine-tuning.md)
- [Annotation rubric](docs/annotation-rubric.md)
- [Local checkpoint layout](models/english/README.md)
- [OpenAPI interface](http://127.0.0.1:8000/docs) when the service is running

## License

CrimeLaya is available under the [Apache License 2.0](LICENSE).