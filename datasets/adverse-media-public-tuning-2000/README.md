# Adverse Media Public Tuning 2,000

The tuning set of the LayaCrime.v0 paper: natural entity/article pairs with balanced human annotations.

## Contract

- Rows: 2,000
- Intended use: fine-tuning, with entity-disjoint validation folds
- Labels: human annotated
- Positive labels: 1,000
- Negative labels: 1,000
- Augmentation: none; every row preserves its original entity and article
- Unique entities: 610
- Unique normalized articles: 2,000
- Maximum entity reuse: 4
- Dataset license: Apache-2.0
- Redistribution status: cleared for public release

The tuning, selection, and benchmark bundles have zero article-ID and normalized-article-text overlap. Select checkpoints on the selection set; the benchmark set must never be used for training, checkpoint selection, prompt selection, or hyperparameter selection. Dataset hashes and complete derivation statistics are recorded in `dataset.json`.

This public dataset bundle is distributed under the repository's Apache-2.0 license.
