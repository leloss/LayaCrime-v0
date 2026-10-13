# Adverse Media Public Selection 1,000

The selection set of the LayaCrime.v0 paper: article-disjoint entity/article pairs with balanced human annotations, used to rank exported checkpoints.

## Contract

- Rows: 1,000
- Intended use: checkpoint selection
- Labels: human annotated
- Positive labels: 500
- Negative labels: 500
- Augmentation: none; every row preserves its original entity and article
- Unique entities: 554
- Unique normalized articles: 1,000
- Maximum entity reuse: 2
- Dataset license: Apache-2.0
- Redistribution status: cleared for public release

The tuning, selection, and benchmark bundles have zero article-ID and normalized-article-text overlap. The selection set must never be used for training; its macro F1 ranks the exported checkpoints. Dataset hashes and complete derivation statistics are recorded in `dataset.json`.

This public dataset bundle is distributed under the repository's Apache-2.0 license.
