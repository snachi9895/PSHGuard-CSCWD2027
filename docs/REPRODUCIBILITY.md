# Reproducibility and reporting

## What can be inspected immediately

Run only `python3 tools/verify_artifact.py` for a no-download integrity check. Saved numeric summaries in `results/` and the 20,878-row split manifest can be inspected directly. The distribution does not include sample-level predictions or trained weights, so the new analysis runners require their original generated inputs before a fresh numerical replay.

## Frozen implementation entry points

- Final six-relation grouping and exact-V2 split: `scripts/final_source_clone_split_20260611/step09_build_final_source_clone_aware_graph_eligible_v2_strict40.py`; independent split audit: `step10_audit_frozen_graph_eligible_v2_strict40.py` in that directory.
- Graph representation: `scripts/final_graph_v10c_full_20260612/step02_v10c_full_export_pipeline.py` and its retained dependencies.
- HGT: `scripts/formal_training_v10c_20260612/train_v10c_formal_hgt.py`.
- Sequence extraction/training: `build_s1_base_modernbert_cache.py`, `run_s1_base_modernbert_screen.py`, and `run_s1_base_modernbert_confirmatory.py` in that directory.
- Same-seed fusion: `scripts/formal_training_v10c_20260612/audit_s8a_same_seed_paired_fusion.py`.
- Later descriptive aggregation: `research_state/single_hgt_ablation_reporting_20261021/scripts/summarize_saved_metrics.py`.
- Budget replay and verification: `research_state/file_budget_same_seed_fusion_replay_20261026/`.
- Joint-size analysis and verification: `paper1015-Frontiers/analysis/scripts/` with its frozen protocol in the adjacent `protocols/` directory.

These are archived scientific entry points, not setup commands to execute on the published snapshot. Their original inputs/outputs are workspace-relative and include generated files excluded here. The shortened `results/` paths are a distribution layout; `metadata/SOURCE_COPY_MAP.csv` maps them back to the saved workspace paths. Follow source code and specifications when reconstructing a separate full workspace.

## Scientific contracts

- Paired system seeds: 1001–1003. Each system contains one graph model and one same-seed sequence model.
- Primary graph-file limit: 16; token input: 512, including special tokens (at most 510 content tokens).
- Fusion: validation-only standardization and L2-regularized logistic regression. Validation OOF and frozen final-test deployment fits are different evaluation objects.
- Table II A: raw AP of single HGT models and their calibrated validation OOF classification F1; three-run mean and sample SD.
- Table II B and Fig.4(a): one HGT plus one same-seed ModernBERT; saved OOF ablation summaries and their matching saved intervals.
- Fig.4(b): current dual-branch budget results, three-run mean/sample SD. No new confidence intervals were computed for this budget comparison.
- Earlier three-HGT ensemble classifications in historical audits are auxiliary systems and do not replace current Table II A or dual-branch budgets.
- Table III: post hoc validation diagnosis conditional on stored scores/decisions. Fixed LR is one training fit. Joint-size weighting changes evaluation weights, not models or operating points. Datadog weighted intervals are missing under the fixed valid-draw rule.
- Final-test branch differences use saved point deltas and paired seed/component intervals; bootstrap means are not substituted for observed point deltas.

Historical README/spec files may contain prospective phrases such as “test locked” or “will be used.” Their timestamps and freeze status describe the stage when written. The completed reported experiments are preserved, and this distribution does not authorize reopening them.

## Environment and external artifacts

`docs/RECORDED_ENVIRONMENTS.md` records the historical versions. `requirements.txt` contains broad compatibility ranges, not an exact lockfile or a claim that an installation was tested in this packaging task. CUDA/PyTorch builds and the pinned ModernBERT revision require separate acquisition. The original implementation supports `PSHGUARD_ROOT` for workspace configuration where provided by a script; inspect each entry point before use.

No experiment, calibration/threshold fit, model inference, or bootstrap was run to create this distribution. The validation budget fits were completed in the separately frozen October analysis; this package copies their saved summaries.
