> Distribution note (2026-10-30): the current paper uses the completed exact-V2 strict40 split described in the final section and step09/step10. Early prospective sections below describe earlier diagnostics, including an earlier audit-only abstract-AST relation; they are not the final six-relation policy.

# Final Source-Clone-Aware Split Preparation

This directory prepares the model-independent source-clone grouping rules that
will be used before the final full-data split is locked.

The first diagnostic operates on the existing diverse 2,000-sample set. It
does not create or replace a split.

## Evidence policy

Strong relations may be forced into one future split:

- same normalized package name;
- same archive SHA-256;
- same complete raw Python-source corpus;
- same complete lexical Python-source corpus after formatting/comments are removed;
- same complete concrete AST corpus while names and constants are retained.

Audit-only relations are never automatically forced into one split:

- same abstract AST corpus after names and constants are removed;
- high overlap of non-common concrete-AST file fingerprints.

V10c graph structure, Event/Param counts, and graph size are not used to create
source-clone groups.

## Output

The 2,000-sample diagnostic is written under:

`manifests/final_source_clone_aware_split_202606/diagnostics_2000/`

Fingerprint cache files make the process resumable. Python members are read
directly from source archives, so no persistent unpacked source is created.

The first candidate split is written separately under:

`manifests/final_source_clone_aware_split_202606/candidate_2000_v1/`

It remains a candidate until its leakage, component, distribution, and
shortcut audits are accepted. It never overwrites an existing split.

## Final full-data split

The final filtered700k split script is:

`scripts/final_source_clone_split_20260611/step08_build_final_source_clone_aware_filtered700k_split.py`

It reads the already verified local-source manifest for all 20,885 filtered700k
samples, fingerprints archives directly without persistent unpacking, and writes
the locked full-data split under:

`manifests/final_source_clone_aware_split_202606/final_filtered700k_v1/`

## V2 strict40 graph-eligible split (2026-07-22)

The frozen remediation script and independent audit are:

- `step09_build_final_source_clone_aware_graph_eligible_v2_strict40.py`
- `step10_audit_frozen_graph_eligible_v2_strict40.py`

V2 is written to a new, non-overwriting directory:

`manifests/final_source_clone_aware_split_202606/final_filtered700k_graph_eligible_v2_strict40_20260722/`

It preserves the same 20,878 graph-eligible samples and exact split-by-label
counts, but lowers the complete abstract-AST relation threshold from 80 to the
pre-existing diagnostic value 40. A three-stage MILP first minimizes moved
rows, then old-bin L1 drift, then applies a deterministic SHA-based tie-break.
The optimum moves 2,142 rows; all enforced relations have zero cross-split
components. The independent re-read audit passes.

The derived V10c index is:

`experiment_outputs/train_final_v10c_filtered700k_20260612/split_v2_strict40_20260722/step01_training_pt_index_v10c_v2_strict40_graph_eligible.csv`

The old V1 directory and index remain immutable historical diagnostics. See
the V2 directory's `README_split_rules_zh.md` and `split_rule_manifest.json`
for the exact contract, transitions, hashes, and solver record.
