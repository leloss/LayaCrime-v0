# Fine-tuning Laya for adverse media

The pipeline in `scripts/fine_tune_laya.py` specializes the English Laya checkpoint for the exact request schema used by this server. The default adverse-media baseline uses class-balanced cross-entropy, deterministic A/B option-order balancing, group-aware validation, best-checkpoint selection across every requested epoch, and post-training temperature calibration. The upstream RLCD loss remains available through `--rl-weight`, but is disabled by default so the first experiment has an interpretable objective.

The upstream `typed-decisions` checkpoint is a different specialist task and is not mixed into adverse-media training or evaluation.

## Data contract

The UI and remote launcher prepare dataset bundles from a corpus JSONL and a labels JSONL. Corpus rows require `article_id`, `entity_name`, and `article`; label rows require the same `article_id` and label `1` (positive/no association) or `2` (negative/criminal association).

The legacy human-file mode remains available. Each input file is expected to contain metadata followed by article content:

```text
###entityName: Acme Corp
###disspositionReason: Hit
###content: Article text...
```

Labels are mapped as follows:

| Source label | Model choice | Meaning |
| --- | --- | --- |
| `Hit` | `A` | Negative criminal association |
| `False Positive` | `B` | No criminal association for the named entity |

The model input contains only the entity and article plus the same adverse-media question used by production inference. Labels are never serialized into model input. Dataset-bundle preparation records annotation provenance and hashes in the prepared manifest.

The dataset prompt is tokenized during preparation, copied into the prepared manifest, and embedded in the exported `rl_agent_config.json`. Individual Test and Benchmark always use the prompt stored in the selected checkpoint, and saved benchmark runs record it.

## Prepare data

Prepare a dataset bundle:

```powershell
.\.venv-laya\Scripts\python.exe .\scripts\fine_tune_laya.py prepare `
  --label-source external_labels `
  --corpus .\datasets\adverse-media-public-tuning-2000\corpus.jsonl `
  --labels .\datasets\adverse-media-public-tuning-2000\annotations\human.jsonl `
  --prompt-file .\datasets\adverse-media-public-tuning-2000\dataset.json `
  --output-dir .\artifacts\prepared-data\adverse-media-public-tuning-2000\human
```

Preparation performs these operations:

- Parses and validates every file.
- Validates unique corpus and label identifiers and joins them without exposing labels to model input.
- Applies the selected group-aware partition over connected normalized-entity and shared-article components.
- Tokenizes with the local Laya English tokenizer and production question schema.
- Writes `train.pt`, `calibration.pt`, `split-assignments.jsonl`, and `manifest.json`.
- Records label provenance, class counts, and a deterministic fingerprint; training rejects stale or unsupported manifests.
- Reverses A/B option order for a deterministic half of training examples and remaps targets accordingly. Calibration retains production order.

Prepared-data schema 3 is required. Rerun **Prepare dataset** after deploying this trainer; older tensors include the retired internal-test contract.

As in the released recipe, preparation overrides the base checkpoint's sequence settings with `max_len=1024` and `head_max_len=256`. These values are recorded in the manifest and checked again before training. If the check reports a stale or missing `sequence_config`, rerun preparation rather than training older 512-token artifacts under the new model configuration.

## Partition approaches

- **Entity/article-disjoint holdout (80/20)** is the deterministic default: 80% training and 20% calibration.
- **70/30, 60/40, and 50/50 holdouts** use the same deterministic connected-group assignment with progressively larger calibration sets.
- **Grouped k-fold train/calibration** uses the selected fold for calibration and all remaining folds for training. Rotate through every fold index for grouped cross-validation.
- **Leave-one-entity/article-group-out** uses the selected connected group for calibration and all remaining groups for training. Rotate through every available index for exhaustive leave-one-group-out validation.

All approaches keep repeated entities and shared normalized article text in one partition. Row-level random folds are intentionally unavailable because they would leak related examples. No strategy creates an internal test partition; use an independent benchmark dataset for final evaluation.

## Hardware

The 421M-parameter model requires a CUDA GPU for practical training. The upstream recipe uses two 16 GB NVIDIA T4 GPUs; a single larger NVIDIA GPU can also run with an appropriate batch size.

Copy these directories to the GPU host without changing their relative layout:

- `artifacts/prepared-data/<dataset-id>/<annotation-id>`
- `models/models--convaiinnovations--laya`
- `third_party/laya`
- `scripts/fine_tune_laya.py`

Install the project dependencies in an isolated environment. Do not enable model network downloads.

## Train

Single GPU:

```bash
python scripts/fine_tune_laya.py train \
  --prepared-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human \
  --output-dir models/fine-tuned/laya-adverse-media-public
```

Two GPUs with Distributed Data Parallel:

```bash
torchrun --standalone --nproc_per_node=2 scripts/fine_tune_laya.py train \
  --prepared-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human \
  --output-dir models/fine-tuned/laya-adverse-media-public
```

The balanced baseline requests four epochs, uses encoder/head learning rates of `3e-6` and `5e-5`, and gives each class equal aggregate cross-entropy weight. RL weight defaults to zero. Before training, epoch zero is evaluated and saved as an eligible baseline checkpoint. After every epoch, the trainer computes unweighted NLL and accuracy on calibration data, emits `epoch_metrics`, and replaces `checkpoint_best` whenever calibration NLL reaches a new strict minimum. Standalone training runs every requested epoch unless adaptive controls are supplied through `--early-stopping-patience`, `--minimum-epochs`, `--max-epochs`, and `--extension-epochs`. If no epoch beats the base model, the export retains epoch zero instead of a degraded checkpoint. The final model is reloaded from `checkpoint_best`, not the last epoch.

`checkpoint_latest` is retained for diagnostics and `checkpoint_best` controls the final export. The final directory contains:

```text
models/fine-tuned/laya-adverse-media-public/
|-- model.safetensors
|-- rl_agent_config.json
|-- training_report.json
|-- data_manifest.json
|-- checkpoint_best/
|-- checkpoint_latest/
|-- encoder/
`-- tokenizer/
```

The calibration split selects the checkpoint and fits post-hoc temperature. The partition strategy, ratio, and selected fold or group are recorded in the prepared manifest, checkpoint config, and training report. Final quality metrics come from an independent benchmark holdout that must not be selected as training input.

The UI can optionally create or update a Hugging Face repository after training succeeds. The token is passed only through `HF_TOKEN` and is not written to the command or report. Upload includes the final model, tokenizer, config, report, and data manifest. Rolling `checkpoint_best` and `checkpoint_latest` directories are excluded to avoid duplicate model uploads.

The stable `article_id` values in `split-assignments.jsonl` match the benchmark console IDs. Do not use the calibration rows as a final benchmark after selecting checkpoints or fitting temperature on them. Compare base and fine-tuned models on a separate evaluation-only dataset.

For `adverse-media-public-tuning-2000`, use the **Public natural staged** strategy as the starting point. It trains the decision head for one epoch before unfreezing the encoder at `1e-6`, warms both learning-rate schedules, applies `0.05` label smoothing, and uses stronger weight decay. This is intended to reduce the rapid natural-data overfitting seen with full-encoder static-rate strategies. All values remain editable, and the final export selects the lowest validation-loss checkpoint from the epochs actually completed.

## Run the public experiment matrix

Stop the UI, llama-server, and any standalone training process before starting the matrix. The matrix is exploratory and uses only the 2,000-example tuning corpus. The default schedule rotates all five grouped cross-validation folds across three seeds for every named training strategy. Related entity/article components remain intact within each fold.

Every run starts with a 10-epoch budget. Training stops after two consecutive epochs without a calibration-NLL improvement of at least `0.001`, subject to a four-epoch minimum. If epoch 10 is a new improving minimum, the budget extends by two epochs at a time, up to a hard limit of 14. Exact checkpoint selection still uses the lowest calibration NLL, including improvements smaller than `0.001`; the threshold controls only stopping and extension.

The public 1,000-example benchmark is excluded from exploration by default. Strategies are compared using mean calibration NLL and its dispersion across folds and seeds. After the strategy and all hyperparameters are locked, train the final model and evaluate the public benchmark once. Historical matrix benchmark columns remain readable, but they do not determine exploratory ordering.

Preview the 120-run repeated-cross-validation schedule without allocating the GPU:

```bash
MATRIX_DRY_RUN=1 ./scripts/run_public_experiment_matrix.sh
```

Run the complete matrix:

```bash
./scripts/run_public_experiment_matrix.sh
```

The runner is resumable. Compatible prepared tensors and completed models are reused. Outputs are written under `artifacts/public-experiment-matrix/`:

- `results.tsv`: per-run results sorted by best calibration NLL.
- `aggregate.tsv`: calibration-size-weighted means and dispersion by strategy and partition approach.
- `logs/<run-id>.log`: preparation and training events for one cell.
- `runs/<run-id>/run.json`: status and path registry for one cell.
- `prepared/<partition>/`: independent tensors for each partition and seed.
- `models/<run-id>/`: exported checkpoint and training report.
- `benchmarks/`: optional diagnostic ledgers when `MATRIX_RUN_BENCHMARK=1` is deliberately enabled.

Run identifiers contain a short fingerprint of the strategy arguments and adaptive epoch policy. Changing a learning rate, patience, threshold, or epoch ceiling therefore creates a new run instead of silently reusing an incompatible checkpoint.

Useful controls:

```bash
# Run only the two highest-priority presets on all default folds and seeds.
MATRIX_STRATEGIES="public_natural_staged balanced" \
./scripts/run_public_experiment_matrix.sh

# Run the first four planned cells, then stop cleanly.
MATRIX_MAX_RUNS=4 ./scripts/run_public_experiment_matrix.sh

# Run selected grouped folds explicitly.
MATRIX_PARTITIONS="group_k_fold:5:0 group_k_fold:5:1" \
./scripts/run_public_experiment_matrix.sh

# Change the repeated-cross-validation seeds.
MATRIX_SEEDS="17 29 43" \
./scripts/run_public_experiment_matrix.sh

# Override the adaptive epoch policy.
MATRIX_PLANNED_EPOCHS=10 MATRIX_MAX_EPOCHS=16 \
MATRIX_MINIMUM_EPOCHS=5 MATRIX_EARLY_STOPPING_PATIENCE=3 \
./scripts/run_public_experiment_matrix.sh
```

Leave-one-group-out is supported with entries such as `leave_one_group_out:0:12`, but it is not part of the default matrix because exhaustively rotating hundreds of connected groups is prohibitively expensive.

No strategy is eliminated solely because another strategy has a better result on one fold. Calibration losses vary by fold and seed, so cross-strategy pruning before aggregate evidence is available can discard a robust configuration. Per-run early stopping provides the conservative pruning mechanism.

After reviewing `aggregate.tsv`, lock the strategy, partition protocol, seed policy, and optimization parameters before evaluating the independent benchmark. Run the benchmark directly for that single final checkpoint rather than enabling benchmark mode for the exploratory matrix:

```bash
python scripts/benchmark_laya_checkpoint.py \
  --run-id layacrime-locked-final \
  --model-dir models/fine-tuned/layacrime-locked-final \
  --corpus datasets/adverse-media-public-holdout-1000/corpus.jsonl \
  --labels datasets/adverse-media-public-holdout-1000/annotations/human.jsonl \
  --output artifacts/benchmark-runs/layacrime-locked-final.jsonl \
  --report artifacts/benchmark-runs/layacrime-locked-final.json
```

## Evaluate and deploy

Inspect an exported checkpoint on its calibration split:

```bash
python scripts/fine_tune_laya.py evaluate \
  --prepared-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human \
  --model-dir models/fine-tuned/laya-adverse-media-public \
  --split calibration
```

Deploy it as Router's English model while retaining the local multilingual base model:

```powershell
$env:LAYA_FINE_TUNED_MODELS_DIR = "$PWD\models\fine-tuned"
.\.venv-laya\Scripts\laya-adverse-media.exe
```

Before production use, review false negatives and false positives by source, jurisdiction, entity type, date, and article language. Accuracy alone is insufficient for adverse-media decisions.