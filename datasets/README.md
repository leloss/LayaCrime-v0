# Dataset bundles

## LayaCrime.v0 public datasets

The paper's corpus consists of 4,000 entity/article pairs with historical human annotations, split into three bundles. These are the only datasets in this repository:

| Bundle | Paper name | Use | Pairs | Positive | Negative | Entities | Purpose |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `adverse-media-public-tuning-2000` | Tuning set | Training and entity-disjoint validation folds | 2,000 | 1,000 | 1,000 | 610 | `training-only` |
| `adverse-media-public-selection-1000` | Selection set | Ranking exported checkpoints | 1,000 | 500 | 500 | 554 | `evaluation-only` |
| `adverse-media-public-holdout-1000` | Benchmark set | Final comparison of the frozen checkpoint and all baselines | 1,000 | 557 | 443 | 479 | `evaluation-only` |

The articles were published from 2020 to 2024. Each bundle contains original entity/article pairs with human labels in `annotations/human.jsonl`, with no synthetic substitutions. The three bundles are article-disjoint under normalized-text comparison, and no entity/article pair appears in more than one bundle. Entity names can recur across bundles.

Use them in order: train and validate on the tuning set, select a checkpoint on the selection set, and evaluate the frozen checkpoint once on the benchmark set. Never use the benchmark set for training, prompt design, checkpoint selection, calibration, or threshold tuning. [docs/fine-tuning.md](../docs/fine-tuning.md) gives the commands.

The three bundles are distributed under the repository's Apache-2.0 license. Their manifests record `license: Apache-2.0`, `redistribution_status: cleared-for-public-release`, and the derivation statistics of each bundle.

## Bundle format

Each child directory of `datasets/` (or of `LAYA_DATASETS_DIR`) is a self-contained bundle: one immutable corpus plus one or more annotation sets for it.

```text
datasets/<dataset-id>/
|-- dataset.json
|-- corpus.jsonl
`-- annotations/
    `-- <annotation-id>.jsonl
```

Corpus rows require `article_id`, `entity_name`, and `article`. Annotation rows require the matching `article_id` and an integer `label`: `1` for positive (no criminal association) and `2` for negative (criminal association). `label_name`, `annotator`, and `status` are optional.

`dataset.json` uses `schema_version: 1` and contains:

- `id` and `name`: a stable lowercase ID and a display name.
- `purpose`: `training-only` (Fine-tuning only), `evaluation-only` (Benchmark only, for independent test sets), or `training-and-evaluation` (the default).
- `prompt`: the classification question, which may contain `{entity_name}`, and 2 to 20 `{decision, text}` criteria that must include both `negative` and `positive`. Preparation embeds it in every checkpoint trained on the bundle.
- `corpus` and `annotations`: relative paths, row counts, and SHA-256 hashes. The server verifies the hashes at startup and refuses a bundle whose files changed.

Saved benchmark runs record the corpus and annotation fingerprints, so a run cannot be reloaded against modified data with the same IDs.

Set `LAYA_DATASETS_DIR` to use another bundle directory, and `LAYA_ACTIVE_DATASET` and `LAYA_BENCHMARK_LABEL_SET` to choose the dataset and annotation set selected at startup. If a bundle has only one annotation set, its ID may be omitted. `LAYA_BENCHMARK_CORPUS` and `LAYA_BENCHMARK_LABELS` point the server at a loose corpus and label file outside any bundle.
