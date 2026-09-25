# Fine-tuning CrimeLaya

The training pipeline follows Laya's RLCD recipe while using the Terra annotation ledger as supervision. Preparation is deterministic and keeps entities in exactly one of train, calibration, or test to prevent entity leakage.

## Data contract

The default inputs are:

- `artifacts/independent-annotations/blind_articles.jsonl`: the blind entity/article corpus.
- `artifacts/independent-annotations/batches/azure.annotations.jsonl`: annotations produced by `gpt-5.6-terra`.
- Corpus rows without an annotation are audited and excluded. No labels are fabricated.

Azure label `2` (`bad_guy`) maps to Laya choice index `0` (`negative`). Azure label `1` (`good_guy`) maps to choice index `1` (`positive`). Preparation recomputes every content-derived article ID and rejects mismatches, duplicate IDs, unknown IDs, and invalid labels.

The original human-label workflow remains available with `prepare --label-source human --data-dir PATH`. It is not the default and uses a caller-selected output directory so it cannot overwrite Azure artifacts accidentally.

## Fine-tuning console

With the CrimeLaya service running, open `http://127.0.0.1:8000/fine-tuning`. Console paths are relative to the repository root by default and cover the corpus, annotations, source dispositions, base checkpoint, prepared tensors, resume checkpoint, and model output. It can prepare either annotation source, launch one process per requested GPU, stop the active process, and stream bounded logs with epoch, batch, loss, and exploration progress.

The console launches the same `scripts/fine_tune_laya.py` commands documented below. It does not upload datasets from the browser. It is enabled automatically only on loopback binds. Set `LAYA_ENABLE_FINE_TUNING=true` only when a non-loopback deployment is protected for trusted administrators.

## Prepare locally

From the repository root:

```powershell
./.venv/Scripts/python.exe ./scripts/fine_tune_laya.py prepare
```

This creates the ignored directory `artifacts/adverse-media-data-azure` with `train.pt`, `calibration.pt`, `test.pt`, `manifest.json`, and `split-assignments.jsonl`. The manifest records source paths, the dataset SHA-256, label and annotator counts, missing annotation IDs, sequence configuration, seed, and split statistics.

## Set up a Linux GPU host

Use Python 3.10 through 3.13 and a recent NVIDIA driver. Copy or clone the repository, including these local assets:

```text
artifacts/adverse-media-data-azure/
models/models--convaiinnovations--laya/
third_party/laya/
```

The bootstrap script creates `.venv-gpu`, repairs a venv without `pip`, installs the training stack, clones Laya when it was not copied, downloads the public English checkpoint from Hugging Face, prepares tensors when only the raw Azure inputs are present, detects visible GPUs, and starts training. A system CUDA toolkit is not required because the PyTorch wheel includes its CUDA runtime. The host needs only a compatible NVIDIA driver.

```bash
chmod +x scripts/train_remote.sh
./scripts/train_remote.sh
```

It uses all visible GPUs by default. It selects CUDA 12.1 wheels for NVIDIA 525-549 drivers and CUDA 12.4 wheels for drivers 550 or newer. `TORCH_INDEX_URL` can override that choice. Environment variables customize deployment without editing the script:

Before running it, `nvidia-smi` must report the attached GPU and driver. Setting `TORCH_INDEX_URL` selects a PyTorch build; it does not install a host driver or make an unattached GPU visible.

The full repository layout is preferred. A minimal upload is also supported when the project root contains `scripts/fine_tune_laya.py` plus either the prepared Azure tensor directory or both raw annotation inputs. Without `pyproject.toml`, the bootstrap installs the standalone training dependencies directly. Set `PROJECT_ROOT` when the script is stored somewhere other than the repository's `scripts/` directory.

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
NPROC_PER_NODE=2 \
VENV_DIR="$HOME/venvs/crimelaya" \
OUTPUT_DIR="$PWD/models/laya-adverse-media-azure" \
./scripts/train_remote.sh --batch-size 2 --gradient-accumulation 16
```

Set `HF_TOKEN` in the shell if Hugging Face requires authentication. To resume, set `RESUME_FROM=models/laya-adverse-media-azure/checkpoint_latest`. The script refuses to replace a completed model unless `ALLOW_OVERWRITE=1` is explicitly set.

Create the environment and install a PyTorch build compatible with the host driver. This example uses the CUDA 12.4 wheel index; use the command recommended by PyTorch when the host differs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-deps --editable third_party/laya
python -m pip install --editable '.[train]'
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

The trainer prints device names, compute capability, memory, precision mode, world size, and effective global batch size at startup. It uses BF16 when supported and FP16 otherwise.

## Train

Single GPU:

```bash
python scripts/fine_tune_laya.py train
```

Defaults are four epochs, batch size 4, gradient accumulation 8, group size 4, encoder learning rate `2.5e-5`, and head learning rate `1e-4`. On a smaller GPU, lower both batches and increase accumulation to preserve the effective batch size:

```bash
python scripts/fine_tune_laya.py train \
  --batch-size 1 \
  --eval-batch-size 4 \
  --gradient-accumulation 32
```

For the two-T4 topology used by the original Laya recipe:

```bash
torchrun --standalone --nproc_per_node=2 scripts/fine_tune_laya.py train
```

Use one `torchrun` process per visible GPU. Do not launch plain `python` independently on each GPU.

## Resume

At the end of each epoch, rank zero writes model and trainer state to `models/laya-adverse-media-azure/checkpoint_latest`. Resume with the same prepared data and training arguments:

```bash
python scripts/fine_tune_laya.py train \
  --resume-from models/laya-adverse-media-azure/checkpoint_latest
```

For DDP, prepend the original `torchrun` command. Resume rejects a checkpoint whose dataset SHA-256 differs from the prepared data.

## Evaluate and export

The final model is written to `models/laya-adverse-media-azure`. The calibration split fits choice temperature after training; the untouched test split reports accuracy, balanced accuracy, Brier score, negative log likelihood, and a confusion matrix.

```bash
python scripts/fine_tune_laya.py evaluate
```

Copy the complete final model directory back to the deployment host. Preserve `training_report.json` and `data_manifest.json`; they bind the weights to their held-out evaluation and supervision provenance. Compare the base and fine-tuned models only on IDs assigned to `test` for an unbiased quality result.