#!/usr/bin/env python3
"""Prepare the final code-graph-eligible task manifest after full export."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v10c_semantic_budget_filtered700k_20260612"
TASK_CSV = GRAPH_ROOT / "step01_final_source_clone_tasks_20883.csv"
META_DIR = GRAPH_ROOT / "meta"
OUT_ELIGIBLE = GRAPH_ROOT / "step05_graph_eligible_tasks_20878.csv"
OUT_EXCLUSIONS = GRAPH_ROOT / "step05_zero_ast_graph_exclusions_5.csv"
OUT_SUMMARY = GRAPH_ROOT / "step05_graph_eligible_distribution.csv"
OUT_AUDIT = GRAPH_ROOT / "step05_graph_eligible_audit.json"

GRAPH_VERSION = "v10c_formal_file16_source_clone_filtered700k_20260612"
EXPECTED_TASKS = 20883
EXPECTED_ELIGIBLE = 20878
EXPECTED_EXCLUSIONS = 5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect_meta(graph_id: str) -> dict:
    path = META_DIR / f"{graph_id}.meta.json"
    if not path.is_file():
        return {
            "graph_id": graph_id,
            "meta_exists": False,
            "graph_version": "",
            "num_nodes": -1,
            "num_edges": -1,
            "module_count": -1,
            "ast_count": -1,
            "param_count": -1,
            "event_count": -1,
            "original_py_files": -1,
            "parsed_py_files": -1,
            "kept_py_files": -1,
        }
    meta = json.loads(path.read_text(encoding="utf-8"))
    counts = meta.get("node_counts", {})
    return {
        "graph_id": graph_id,
        "meta_exists": True,
        "graph_version": meta.get("control_profile", ""),
        "num_nodes": int(meta.get("num_nodes", 0)),
        "num_edges": int(meta.get("num_edges", 0)),
        "module_count": int(counts.get("module", 0)),
        "ast_count": int(counts.get("ast", 0)),
        "param_count": int(counts.get("param", 0)),
        "event_count": int(counts.get("event", 0)),
        "original_py_files": int(meta.get("original_py_files", 0)),
        "parsed_py_files": int(meta.get("parsed_py_files", 0)),
        "kept_py_files": int(meta.get("kept_py_files", 0)),
    }


def exclusion_reason(row: pd.Series) -> str:
    if row["original_py_files"] == 0:
        return "no_python_source_files"
    return "no_retained_ast_nodes_under_event_centered_graph_rule"


def main() -> None:
    tasks = pd.read_csv(TASK_CSV, low_memory=False)
    require(len(tasks) == EXPECTED_TASKS, f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")
    require(tasks["graph_id"].nunique() == EXPECTED_TASKS, "task graph_id is not unique")

    metrics = pd.DataFrame(inspect_meta(str(graph_id)) for graph_id in tasks["graph_id"])
    require(metrics["meta_exists"].all(), "one or more task meta files are missing")
    require(
        set(metrics["graph_version"]) == {GRAPH_VERSION},
        f"unexpected graph versions: {metrics['graph_version'].value_counts().to_dict()}",
    )

    merged = tasks.merge(metrics, on="graph_id", how="inner", validate="one_to_one")
    eligible_mask = merged["ast_count"] > 0
    eligible = merged.loc[eligible_mask, tasks.columns].copy()
    exclusions = merged.loc[~eligible_mask].copy()
    exclusions["exclusion_rule"] = "ast_count_must_be_greater_than_zero"
    exclusions["exclusion_reason"] = exclusions.apply(exclusion_reason, axis=1)

    require(len(eligible) == EXPECTED_ELIGIBLE, f"expected {EXPECTED_ELIGIBLE} eligible, got {len(eligible)}")
    require(
        len(exclusions) == EXPECTED_EXCLUSIONS,
        f"expected {EXPECTED_EXCLUSIONS} exclusions, got {len(exclusions)}",
    )
    require(
        set(exclusions["split"]) == {"train"},
        f"zero-AST exclusions unexpectedly affect evaluation splits: {exclusions['split'].value_counts().to_dict()}",
    )

    summary = (
        eligible.groupby(["split", "label", "source_group"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    audit = {
        "status": "ready",
        "eligibility_rule": "ast_count > 0",
        "eligibility_rule_is_label_blind": True,
        "input_tasks": len(tasks),
        "eligible_tasks": len(eligible),
        "excluded_tasks": len(exclusions),
        "excluded_by_reason": exclusions["exclusion_reason"].value_counts().to_dict(),
        "excluded_by_split": exclusions["split"].value_counts().to_dict(),
        "excluded_by_label": exclusions["label"].value_counts().to_dict(),
        "graph_version": GRAPH_VERSION,
        "excluded_graph_ids": sorted(exclusions["graph_id"].astype(str)),
    }

    eligible.to_csv(OUT_ELIGIBLE, index=False, encoding="utf-8")
    exclusions.to_csv(OUT_EXCLUSIONS, index=False, encoding="utf-8")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("FINAL V10c GRAPH-ELIGIBLE TASKS READY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nEXCLUSIONS")
    print(
        exclusions[
            [
                "graph_id",
                "label",
                "source_group",
                "split",
                "original_py_files",
                "ast_count",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )
    print(f"\n[RESULT] eligible  : {len(eligible)}")
    print(f"[RESULT] excluded  : {len(exclusions)}")
    print(f"[RESULT] {OUT_ELIGIBLE}")
    print(f"[RESULT] {OUT_EXCLUSIONS}")
    print(f"[RESULT] {OUT_AUDIT}")


if __name__ == "__main__":
    main()
