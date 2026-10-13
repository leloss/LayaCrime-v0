# Fine-tuning Laya for Criminal Sentiment Analysis

This guide covers the training pipeline in `scripts/fine_tune_laya.py` and the scripts that reproduce the tuning, selection, and evaluation protocol of the LayaCrime.v0 paper. The browser workflow on `/fine-tuning` runs the same pipeline.

LayaCrime.v0 specializes the English typed-decisions checkpoint of Laya 0.3.10 (421.3 million parameters, ModernBERT-large backbone) without structural changes. Each training example is one binary-choice row per entity/article pair: the article and entity form the state, and the dataset prompt supplies the question and the two option criteria.

## Data contract

Training reads a dataset bundle: a corpus JSONL and an annotation JSONL (see [datasets/README.md](../datasets/README.md)). Corpus rows require `article_id`, `entity_name`, and `article`; annotation rows require the same `article_id` and label `1` (positive, no criminal association) or `2` (negative, criminal association). Labels are joined by ID and never serialized into the model input.

The prompt comes from the bundle's `dataset.json`. Preparation tokenizes it with every example, records it in the prepared manifest, and embeds it in the exported `rl_agent_config.json`, so Individual Test and Benchmark always use the prompt a checkpoint was trained with.

## Prepare data

```bash
python scripts/fine_tune_laya.py prepare \
  --corpus datasets/adverse-media-public-tuning-2000/corpus.jsonl \
  --labels datasets/adverse-media-public-tuning-2000/annotations/human.jsonl \
  --prompt-file datasets/adverse-media-public-tuning-2000/dataset.json \
  --partition-strategy group_k_fold --fold-count 5 --fold-index 0 \
  --seed 20260923 \
  --output-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human
```

Preparation:

- Validates every row and the uniqueness of corpus and label IDs.
- Splits the data into training and validation (calibration) partitions over connected components of normalized entities and shared articles, so related examples never cross partitions.
- Tokenizes with the local Laya English tokenizer, a maximum sequence length of 1,024, and a maximum question-head budget of 256. Articles that exceed the remaining budget are truncated after accounting for prompt length.
- Assigns each training example deterministically to the original or reversed option order and remaps its target, to reduce dependence on option position. Validation and inference keep the original order.
- Writes `train.pt`, `calibration.pt`, `split-assignments.jsonl`, and `manifest.json`, including a dataset fingerprint that training verifies.

### Partitions

- **Grouped k-fold** (default): five entity-disjoint folds, as in the paper. The selected fold is used for validation and the other four for training.
- **Entity/article-disjoint holdouts** (80/20, 70/30, 60/40, 50/50): one deterministic split.
- **Leave one group out**: the selected connected group is used for validation.

Every option keeps related entities and articles in one partition. None creates an internal test split; final evaluation always uses a separate bundle.

## Train

The paper's selected configuration, **Higher encoder learning rate**, on one GPU:

```bash
python scripts/fine_tune_laya.py train \
  --prepared-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human \
  --output-dir models/fine-tuned/my-layacrime \
  --strategy adaptation_encoder_lr_2e5 \
  --encoder-lr 2e-5 --head-lr 5e-5 --weight-decay 0.01 \
  --batch-size 4 --gradient-accumulation 8 \
  --epochs 4 --max-epochs 10 --minimum-epochs 4 \
  --early-stopping-patience 2 --early-stopping-min-delta 0.001 --extension-epochs 2 \
  --seed 20260923
```

`--strategy` only records the configuration name; the optimization values are passed explicitly. The table under [Tuning configurations](#tuning-configurations) lists all of them.

What the trainer does:

- Minimizes inverse-frequency-weighted cross-entropy over the option logits. The auxiliary RLCD objective is available through `--rl-weight` and is disabled by default.
- Uses AdamW with separate encoder and decision-head learning rates, cosine decay, gradient clipping at a maximum norm of 1, mixed precision (float16 on the T4; bfloat16 on GPUs that support it natively), and gradient checkpointing.
- Evaluates epoch zero (the base model) first, then computes validation negative log-likelihood (NLL) and accuracy after every epoch.
- Exports the epoch with the lowest validation NLL; validation accuracy breaks an exact tie. If no epoch beats the base model, epoch zero is exported.
- Fits a choice temperature on the validation fold and stores it with the checkpoint. Temperature scaling does not change the predicted class.

The paper's runs used one NVIDIA Tesla T4 (16 GB) on an Amazon EC2 `g4dn.xlarge` instance with Ubuntu 22.04, driver 535 and CUDA 12.2. With `torchrun --standalone --nproc_per_node=<n>`, the same command trains on several GPUs.

The export directory contains:

```text
models/fine-tuned/my-layacrime/
|-- model.safetensors
|-- rl_agent_config.json
|-- training_report.json
|-- data_manifest.json
|-- encoder/
|-- tokenizer/
|-- checkpoint_best/
`-- checkpoint_latest/
```

`training_report.json` holds the per-epoch history, the exported epoch, the temperature, and the training controls; `data_manifest.json` holds the partition, split sizes, prompt, and dataset fingerprint. The rolling `checkpoint_best/` and `checkpoint_latest/` directories are not needed for inference. Exports under `models/fine-tuned/` appear in the model selectors automatically.

## Reproduce the paper's tuning protocol

The protocol has three stages, each on its own dataset:

1. **Candidate generation** on the 2,000-pair tuning set: 14 configurations × 5 entity-disjoint folds × 3 seeds = 210 runs.
2. **Checkpoint selection** on the independent 1,000-pair selection set: rank every exported checkpoint by macro F1, breaking ties by selection accuracy, then validation NLL, then run ID.
3. **Evaluation** of the chosen checkpoint on the 1,000-pair benchmark set. Benchmark labels are never used for training or selection.

### Tuning configurations

All configurations use four-example exploration groups, inverse-frequency class weighting, an effective batch size of 32, and no label smoothing, warm-up, or RLCD unless stated.

| ID | Configuration (paper name) | Encoder LR | Head LR | Weight decay | Other settings |
| --- | --- | ---: | ---: | ---: | --- |
| `adaptation_encoder_lr_2e5` | Higher encoder learning rate (selected) | 2e-5 | 5e-5 | 0.01 | |
| `adaptation_weight_decay_2e2` | Fine-tuning with stronger weight decay | 1e-5 | 5e-5 | 0.02 | |
| `adaptation_head_lr_1e4` | Higher classifier learning rate | 1e-5 | 1e-4 | 0.01 | |
| `maximum_task_adaptation` | Tuning baseline | 1e-5 | 5e-5 | 0.01 | |
| `adaptation_weight_decay_0` | Fine-tuning without weight decay | 1e-5 | 5e-5 | 0 | |
| `adaptation_head_lr_2e5` | Lower classifier learning rate | 1e-5 | 2e-5 | 0.01 | |
| `adaptation_encoder_lr_5e6` | Lower encoder learning rate | 5e-6 | 5e-5 | 0.01 | |
| `relational_adaptation` | Lower encoder LR with regularization | 5e-6 | 5e-5 | 0.02 | |
| `relational_extended` | Balanced adaptation with regularization | 3e-6 | 5e-5 | 0.02 | |
| `balanced` | Balanced adaptation | 3e-6 | 5e-5 | 0.01 | |
| `public_natural_staged` | Gradual encoder adaptation | 1e-6 | 2e-5 | 0.05 | Encoder frozen for epoch 1, 0.1 warm-up ratio, label smoothing 0.05, sigma 0.3→0.05 |
| `relational_low_drift` | Minimal encoder adaptation | 1e-6 | 3e-5 | 0.02 | |
| `relational_rlcd` | Relation-preserving adaptation | 3e-6 | 5e-5 | 0.02 | RLCD weight 0.1, sigma 0.3→0.05 |
| `preserve_entity_matching` | Head-only training (frozen encoder) | 0 | 5e-5 | 0.01 | |

Exploration sigma decays from 0.4 to 0.1 unless stated. Every run requests four epochs and may extend to ten; training stops after two consecutive epochs without a validation-NLL improvement of at least 0.001, after a minimum of four epochs.

### Run the 210 training runs

Stop the web server and any other GPU process first. Preview the schedule without using the GPU:

```bash
MATRIX_DRY_RUN=1 ./scripts/run_public_experiment_matrix.sh
```

Then run or resume it:

```bash
./scripts/run_public_experiment_matrix.sh
```

The matrix trains on the tuning set only and never evaluates the benchmark set. It is resumable: completed models and compatible prepared tensors are reused, failed runs are retried (`MATRIX_RETRY_FAILED=0` skips them), and a stale lock from a stopped run is removed automatically. Outputs go to `artifacts/public-experiment-matrix/`:

- `results.tsv`: one row per run with its validation metrics, sorted by validation NLL.
- `aggregate.tsv`: validation-size-weighted means by configuration.
- `models/<run-id>/`: the exported checkpoint and its training report.
- `runs/<run-id>/run.json` and `logs/<run-id>.log`: status and training log of each run.

Run IDs include a fingerprint of the configuration, so outputs stay tied to their settings. To train a subset, set `MATRIX_STRATEGIES`, `MATRIX_PARTITIONS` (for example `group_k_fold:5:0`), `MATRIX_SEEDS`, or `MATRIX_MAX_RUNS`; use a separate `MATRIX_ROOT` when changing the protocol.

### Select the checkpoint and evaluate it

After all 210 runs finish, score every exported checkpoint on the selection set, freeze the winner, and evaluate only that checkpoint on the benchmark set:

```bash
PYTHON=~/.cache/laya-adverse-media/venv/bin/python
"$PYTHON" scripts/benchmark_all_laya_checkpoints.py --finalize-winner
```

The runner verifies checkpoint, dataset, and ledger hashes and resumes partial ledgers. Until all 210 runs have selection results, it reports the current leader without freezing a winner. Benchmark labels are loaded only after the winner is fixed.

To reproduce the paper's appendix table, which evaluates each configuration's selection-set leader on the benchmark set:

```bash
"$PYTHON" scripts/benchmark_all_laya_checkpoints.py --benchmark-strategy-winners
```

Reports and prediction ledgers are written under `artifacts/public-experiment-matrix/benchmarks/`.

## Evaluate a checkpoint on its validation fold

```bash
python scripts/fine_tune_laya.py evaluate \
  --prepared-dir artifacts/prepared-data/adverse-media-public-tuning-2000/human \
  --model-dir models/fine-tuned/my-layacrime \
  --split calibration
```

Validation rows select the epoch and fit the temperature, so they are not an unbiased test set. Use an `evaluation-only` bundle in the Benchmark page for that.

## Publish to Hugging Face

The `/fine-tuning` page can create or update a Hugging Face model repository after training succeeds. The token is passed only through `HF_TOKEN` and is not written to the command or the report. The upload contains the model, tokenizer, encoder configuration, training report, and data manifest, but not the rolling checkpoint directories.
