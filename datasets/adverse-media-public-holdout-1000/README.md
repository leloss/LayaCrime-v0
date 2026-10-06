# Adverse Media Public Holdout 1,000

Public article-disjoint benchmark with human annotations.

## Contract

- Rows: 1,000
- Intended use: final benchmarking only
- Labels: human annotated
- Positive labels: 557
- Negative labels: 443
- Augmentation: none; every row preserves its original entity and article
- Unique normalized articles: 1,000
- Maximum entity reuse: 8
- Dataset license: Apache-2.0
- Redistribution status: cleared for public release

The tuning and holdout bundles have zero article-ID and normalized-article-text overlap. The holdout must never be used for training, checkpoint selection, prompt selection, or hyperparameter selection. Dataset hashes and complete derivation statistics are recorded in `dataset.json`.

This public dataset bundle is distributed under the repository's Apache-2.0 license.
