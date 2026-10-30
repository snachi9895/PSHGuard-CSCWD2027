# Data dictionary

## Compact split manifest

`data/manifests/pshguard_split_manifest.part-01.csv` through `part-11.csv` contain 20,878 unique `sample_id` records. Parts are one logical table in original row order. Column names and per-part checksums are in `pshguard_split_manifest.summary.json`.

- `label`/`label_int`: inherited class annotation; the manifest is not an independent human relabeling.
- `split`: `train`, `val`, or `test`.
- `source_group`: recorded source-list grouping; do not treat it as an exclusive archive retrieval channel.
- `source_clone_component_id`: whole relatedness component used for partition isolation. The split grouping graph is distinct from the per-package behavior graph.
- Project/version/hash/fingerprint columns identify the package and grouping evidence, not executable payload content.
- `python_file_count`, parsed/skipped file counts, bytes, tokens, AST counts, and truncation fields describe fingerprint preprocessing. The graph budget analysis uses its recorded original-file-count definition; a sensitivity check used fingerprint file counts. These definitions must not be silently swapped.

## Metric units and missing values

Numeric `*_ratio` and ordinary saved metric fields use 0–1 unless explicitly named `percent`, `pp`, or documented otherwise. To express a ratio difference in percentage points, multiply by 100. Three-run SD is sample SD (`ddof=1`); it is not a confidence interval. Plot and manuscript rounding can conceal small full-precision differences.

In `results/current/table_01_existing_detectors.csv`, missing continuous AP is an empty CSV field. Learned rows have saved sample SD; deterministic tools have one frozen-policy point estimate. Display columns reproduce the manuscript's percentage rounding. The two internal branches are in `internal_branch_controls.csv`, separate from the external comparison table.

`table_02_graph_representation.csv` uses four `system_metric` identities: `single_raw_AP`, `single_OOF_F1`, `fusion_OOF_AP`, and `fusion_OOF_F1`.

For joint-size diagnosis, `definition=primary` is the original file-count definition; `sensitivity` is a prespecified alternative count. `raw_unweighted`, `retained_unweighted`, and `retained_overlap_weighted` are distinct evaluation populations/weights. `three_seed_mean` means the average of three system metrics, not an ensemble prediction. LR is `single_fixed`.

Empty joint-size CI fields retain the saved missing state. In particular, Datadog weighted CIs are unavailable because only 1,719/2,000 draws were valid, below the fixed 95% threshold. The 1,967 valid Guo draws met that threshold. The support count and population tables preserve excluded cells and weights; weights are not package counts.

The low-prevalence CSV is a previously saved hypothetical projection at fixed test recall/FPR, not deployment observations. Precision was averaged per seed; a ratio of rounded/mean alert counts need not equal that saved mean exactly.
