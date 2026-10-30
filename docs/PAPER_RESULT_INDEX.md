# Current six-page manuscript: paper-to-file index

Manuscript identity and checksum: `metadata/MANUSCRIPT_IDENTITY.json`. This index refers to the 30 October author-contact version including the approved local text revisions.

| Paper item | Current distributed evidence | Reading note |
| --- | --- | --- |
| Table I: external comparison | `results/current/table_01_existing_detectors.csv`; `pypimaldet_r4_saved_summary.json` in the same directory | Seven external detectors plus PSHGuard. Saved S10-C alone predates the R4 row. |
| Internal branches, RQ2 | `results/current/internal_branch_controls.csv` | Same final-test population, three individual systems per method. |
| Fig.3: paired differences | `results/current/figure_03_paired_branch_differences.csv` | Saved point delta and matching 95% paired CI. F1 vs ModernBERT includes zero. |
| Table II A | `results/graph_representation/single_hgt_mean_sample_sd.csv`, `single_hgt_per_seed_metrics.csv`; compact view in `results/current/table_02_graph_representation.csv` | Raw AP and single-HGT validation OOF F1 have different scoring identities. |
| Table II B; Fig.4(a) | `results/graph_representation/fusion_saved_audit.json`, `fusion_per_seed_metrics.csv`; `results/current/figure_04a_fusion_ablation_differences.csv` | Same-seed HGT/ModernBERT fusion. Each interval stays attached to its original estimator. |
| Fig.4(b) | `results/file_budget/all_budgets_mean_sample_sd.csv` and `all_budgets_per_seed_metrics.csv` | Current dual-branch 4/8/16/32-file validation comparison. 16 and 32 F1 are near equal; SD error bars are not CIs. |
| Table III A/B | `results/current/table_03_common_support.csv`, `table_03_auroc_and_fpr.csv`; full saved CSVs in `results/source_size/` | Preserve raw, retained, and weighted populations; Datadog weighted CIs remain missing. |
| Source/size paired AUROC and balance, RQ4 | `results/source_size/paired_comparisons.csv`, `continuous_balance.csv`, `joint_cell_balance.csv`, `population_coverage.csv` | Conditional observational comparison with one fixed LR fit; no causal semantic-contribution percentage. |
| Component-disjoint data allocation | `data/manifests/`; `results/split_audit/saved_component_audit_summary.json`; split scripts | The grouping graph uses six relations; >50-package components are training-only. |
| RQ3 efficiency and calibration | `results/current/applicability_saved_values.json` plus historical S9-E1/S9-H aggregate files | Serial throughput is an estimate from separate prepared-input branch timings, excluding static preprocessing. ECE belongs to validation OOF. |
| RQ3 perturbation and difficult groups | Historical S9-F/S9-H JSON/CSV under `results/historical_20260911/` | Frozen perturbations and observational unmatched strata. These are not new attacks or matched causal groups. |
| Validation errors corrected/introduced by fusion | Historical S9-H `s9h_complementarity.csv` | Counts averaged across three systems; no new label adjudication. |
| Low-prevalence alert burden | `results/current/low_prevalence_saved_projection.csv` | At 1% prevalence: about 91 true/543 false alerts per 10,000, approximately 14% precision, conditional on unchanged test rates. |
| Partial malicious-source holdout | `results/current/applicability_saved_values.json` and historical S9-G2 summary | Prefix/list-defined partial shift; the archive channel overlaps and fusion F1 is below ModernBERT in this result. |

## Historical result identities

The files under `results/historical_20260911/` are retained unchanged. S9-B/S9-C/S9-C32 auxiliary classification fields may describe a **three-HGT ensemble**. They do not describe the current single-HGT Table II A or the current dual-branch budget panel. Raw-AP results can share sources while classification estimands differ. `results/graph_representation/saved_uncertainty_and_fusion_reference.json` labels the historical ensemble intervals explicitly.

No rerun or parameter selection was performed for this package. All convenience views select existing aggregate fields and convert display units only. The observed point deltas, sample SDs, paired confidence intervals, and missing states retain their original identities.

## Deliberately unbundled

The manuscript PDF/figure artwork, raw archives, checkpoints, baseline full repositories, per-package outputs, and bootstrap replicates are not present. Figure 1 is the author-approved framework artwork and Figure 2 is a static source example; this artifact does not redistribute the package source used in that example. The artifact supports inspection of code, split metadata, and aggregate evidence rather than claiming an executable end-to-end reproduction from these files alone.
