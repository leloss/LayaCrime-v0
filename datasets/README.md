# Dataset bundles

## LayaCrime.v0 public release

Only these two bundle directories are unignored and intended for publication through this repository:

- `adverse-media-public-tuning-2000`: 2,000 natural entity/article pairs for fine-tuning, balanced 1,000 positive and 1,000 negative.
- `adverse-media-public-holdout-1000`: 1,000 human-annotated, article-disjoint natural pairs for final benchmarking only, with 443 negative and 557 positive labels.

Both use the released `human` annotation set, contain no synthetic entity substitutions or name transformations, limit repeated entities, and have zero article-ID and normalized-text overlap.

```powershell
..\.venv-laya\Scripts\python.exe scripts\build_public_datasets.py
```

Other local bundles remain ignored development inputs and diagnostics; they are not part of the public dataset release.

The tuning bundle stores its 2,000 labels under `annotations/human.jsonl`; the full annotation set is directly available as a fine-tuning preset. Generated tensors, checkpoints, and run outputs belong under ignored `artifacts/` and `models/` paths.

## License

The two public bundles are distributed under the repository's Apache-2.0 license. Their manifests record `license: Apache-2.0` and `redistribution_status: cleared-for-public-release`.

Each ignored child directory is a self-contained benchmark dataset. A bundle keeps one immutable corpus beside every compatible annotation set:

```text
datasets/<dataset-id>/
|-- dataset.json
|-- corpus.jsonl
|-- annotations/
|   |-- reviewers.jsonl
|   `-- model-generated.jsonl
`-- training-subsets/
    `-- <annotation-id>/
```

`dataset.json` uses `schema_version: 1`, gives the dataset a stable lowercase ID and display name, identifies the corpus path and hash, and lists annotation IDs, names, paths, row counts, and hashes. Paths must be relative and cannot leave the bundle.

The manifest also owns the classification `prompt`. Its `question` may contain `{entity_name}` and its `criteria` contains 2 to 20 `{decision, text}` entries. Both `negative` and `positive` decisions are required. Fine-tuning preparation tokenizes this prompt with every example, copies it into the prepared-data manifest, and embeds it in the exported checkpoint. The Fine-Tuning Console shows it read-only; change it in `dataset.json` when you build a bundle, and keep the prompt with its dataset.

Set `purpose` to `evaluation-only` for an independent holdout bundle. Its annotations remain available to the benchmark console but are excluded from fine-tuning presets. The default purpose is `training-and-evaluation`.

Set `purpose` to `training-only` for data that may feed Fine-Tuning but must never appear in the Benchmark dataset selector. Independent holdout bundles should use `evaluation-only`; do not combine nested training subsets and holdout annotations in the same benchmark corpus.

Use natural, article-disjoint data as the primary deployment-quality benchmark. Additional local datasets and synthetic diagnostics remain ignored and are not part of the LayaCrime.v0 public release.

Saved benchmark runs record SHA-256 fingerprints for the corpus and active annotation file. Loading a run fails if either file changed, preventing regenerated data from being evaluated under stale predictions with the same dataset ID.

Corpus rows require `article_id`, `entity_name`, and `article`. Annotation rows require the matching `article_id` and integer `label` (`1` for positive/no association, `2` for negative association); `label_name`, confidence, rationale, evidence, annotator, and status are optional.

Set `LAYA_DATASETS_DIR` to another bundle root or `LAYA_ACTIVE_DATASET` to select a specific bundle. The benchmark console resolves `LAYA_ACTIVE_DATASET` with `LAYA_BENCHMARK_LABEL_SET` at startup. If a bundle has one annotation set, its ID may be omitted.

Explicit `LAYA_BENCHMARK_CORPUS` and `LAYA_BENCHMARK_GOLD` override bundle paths. `LAYA_BENCHMARK_LABELS` remains supported as a legacy alias for the gold path.