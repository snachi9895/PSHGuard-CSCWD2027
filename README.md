# PSHGuard — CSCWD2027 research artifact

Code, the compact split manifest, and saved aggregate results accompanying **PSHGuard: Detecting Malicious PyPI Packages With Complementary Behavioral Views**.
This snapshot was prepared on 30 October 2026 for the CSCWD2027 manuscript. It does not imply conference acceptance or publication.

PSHGuard represents package behavior with a budget-controlled typed graph and an independently extracted behavior sequence. One HGT and one ModernBERT are paired **per seed** through validation-fitted score fusion. Main results are means and sample standard deviations over three systems, not one ensemble of three HGT models.

## Start here

1. [Paper-to-file index](docs/PAPER_RESULT_INDEX.md): exact files for Tables I–III, Figures 3–4, and the reported diagnostics.
2. [Reproducibility guide](docs/REPRODUCIBILITY.md): implementation entry points, external inputs, and reporting identities.
3. [Data dictionary](docs/DATA_DICTIONARY.md): units, missing values, split metadata, and source provenance.
4. [Rights and attribution](docs/RIGHTS_AND_ATTRIBUTION.md).

## Layout

| Directory | Contents |
| --- | --- |
| `scripts/` | Frozen split, static graph, HGT, sequence, operating-point, and same-seed fusion implementation; import dependencies retain their original paths. |
| `baselines/ea4mp/src/ea4mp_repro/` | Behavior-text canonicalization dependency used by the sequence branch. |
| `data/manifests/` | Eleven CSV parts covering 20,878 package records and the original summary; no local archive paths. |
| `results/current/` | Convenience views of the saved results matching the current six-page manuscript. |
| `results/graph_representation/` | Single-HGT run summaries and same-seed fusion ablations, with their different evaluation identities. |
| `results/file_budget/` | Current same-seed dual-branch 4/8/16/32-file validation summaries. |
| `results/source_size/` | Saved joint-size support, balance, metrics, paired differences, and bootstrap validity. |
| `results/historical_20260911/` | Earlier aggregate audits retained for traceability; see the index before interpreting ensemble/budget results. |
| `research_state/`, `paper1015-Frontiers/` | Original relative paths for the later analysis code and frozen protocols. These names record provenance, not a publication decision. |
| `metadata/` | File checksums, source-copy mapping, manuscript identity, and numerical source maps. |

## Integrity check (no training, inference, or downloads)

```bash
python3 tools/verify_artifact.py
```

The verifier uses Python's standard library to check file hashes, manifest counts/component isolation, and the identity of the distributed summary tables. It does not import experiment scripts or inspect executable package samples.

## External inputs and scope

The 20,878 records comprise 16,613 training, 2,137 validation, and 2,128 final-test packages. Labels are inherited from the recorded collections. Source-list membership and archive retrieval channel are distinct fields/concepts.
Inputs are official PyPI source distributions/metadata, the Guo et al. public registry (<https://github.com/lxyeternal/pypi_malregistry>), the Datadog malicious-software-packages dataset (<https://github.com/DataDog/malicious-software-packages-dataset>), and `answerdotai/ModernBERT-base` at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`.

Raw archives, extracted package source, graph/tensor caches, checkpoints, per-package predictions, and bootstrap replicate dumps are not distributed. Full retraining requires reconstruction of these inputs and the recorded environment. This is an inspectable code/metadata/aggregate-result snapshot, not a self-contained training dataset. No third-party data or model is relicensed.

The manuscript PDF, figures, author correspondence, private Git history, and machine-specific configuration are outside this artifact. Repository availability depends on its visibility; no public release or archival DOI is claimed by this snapshot.
