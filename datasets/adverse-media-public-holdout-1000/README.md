# Adverse Media Public Holdout 1,000

The benchmark set of the LayaCrime.v0 paper: article-disjoint entity/article pairs with human annotations, used only to evaluate the frozen checkpoint and the baselines.

## Contract

- Rows: 1,000
- Intended use: comparison benchmarking of the frozen checkpoint and baselines
- Labels: human annotated
- Positive labels: 557
- Negative labels: 443
- Augmentation: none; every row preserves its original entity and article
- Unique entities: 479
- Unique normalized articles: 1,000
- Maximum entity reuse: 8
- Dataset license: Apache-2.0
- Redistribution status: cleared for public release

The tuning, selection, and benchmark bundles have zero article-ID and normalized-article-text overlap. The benchmark must never be used for training or checkpoint selection. Dataset hashes and complete derivation statistics are recorded in `dataset.json`.

This public dataset bundle is distributed under the repository's Apache-2.0 license.
