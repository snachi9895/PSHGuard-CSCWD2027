#!/usr/bin/env python3
"""Prepare and validate the final source-clone-aware V10c graph tasks."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
SPLIT_ROOT = ROOT / "manifests/final_source_clone_aware_split_202606/final_filtered700k_v1"
IN_ALL = SPLIT_ROOT / "filtered700k_source_clone_aware_all.csv"
EXCLUSIONS = SPLIT_ROOT / "cross_label_clone_conflict_exclusions.csv"

GRAPH_ROOT = ROOT / "graph_artifacts/final_v10c_semantic_budget_filtered700k_20260612"
OUT_TASKS = GRAPH_ROOT / "step01_final_source_clone_tasks_20883.csv"
OUT_SMOKE = GRAPH_ROOT / "step01_balanced_smoke_tasks_30.csv"
OUT_SUMMARY = GRAPH_ROOT / "step01_task_distribution_summary.csv"
OUT_PREFLIGHT = GRAPH_ROOT / "step01_preflight_audit.json"

EXPECTED_ALL = 20885
EXPECTED_EXCLUSIONS = 2
EXPECTED_TASKS = 20883
SMOKE_PER_SOURCE = 10
ABSTRACT_AST_MIN_NODES = 80

REQUIRED_COLUMNS = {
    "graph_id",
    "sample_id",
    "label",
    "label_int",
    "split",
    "package_name",
    "version",
    "source_group",
    "group_key",
    "archive_path",
    "archive_sha256",
    "raw_corpus_digest",
    "lexical_corpus_digest",
    "concrete_ast_corpus_digest",
    "abstract_ast_corpus_digest",
    "source_clone_component_id",
    "python_file_count",
    "parsed_ast_file_count",
    "total_ast_nodes",
    "truncated_package",
    "status",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def max_split_count(frame: pd.DataFrame, column: str) -> int:
    eligible = frame[column].notna() & frame[column].astype(str).ne("")
    if not eligible.any():
        return 0
    return int(frame.loc[eligible].groupby(column)["split"].nunique().max())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def choose_smoke(tasks: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    split_quotas = {"train": 8, "val": 1, "test": 1}
    groups = [
        ("benign", "sdist_benign"),
        ("malicious", "sdist_malicious"),
        ("malicious", "external_datadog_samples"),
    ]
    for label, source_group in groups:
        group = tasks[
            (tasks["label"] == label) & (tasks["source_group"] == source_group)
        ].sort_values("graph_id")
        selected = []
        for split, quota in split_quotas.items():
            part = group[group["split"] == split].head(quota)
            require(
                len(part) == quota,
                f"smoke requires {quota} rows for {label}/{source_group}/{split}, got {len(part)}",
            )
            selected.append(part)
        sample = pd.concat(selected, ignore_index=True)
        require(len(sample) == SMOKE_PER_SOURCE, f"unexpected smoke size for {source_group}")
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    frame = pd.read_csv(IN_ALL, low_memory=False)
    exclusions = pd.read_csv(EXCLUSIONS, low_memory=False)

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    require(not missing_columns, f"missing required columns: {missing_columns}")
    require(len(frame) == EXPECTED_ALL, f"expected {EXPECTED_ALL} input rows, got {len(frame)}")
    require(frame["graph_id"].nunique() == EXPECTED_ALL, "input graph_id is not unique")
    require(set(frame["split"]) == {"train", "val", "test"}, "unexpected split values")
    require(set(frame["label"]) == {"benign", "malicious"}, "unexpected label values")
    require(set(frame["label_int"].astype(int)) == {0, 1}, "unexpected label_int values")
    require(set(frame["status"]) == {"ok"}, "source fingerprint manifest contains non-ok rows")

    excluded_ids = set(exclusions["graph_id"].astype(str))
    require(
        len(excluded_ids) == EXPECTED_EXCLUSIONS,
        f"expected {EXPECTED_EXCLUSIONS} exclusions, got {len(excluded_ids)}",
    )
    require(excluded_ids <= set(frame["graph_id"].astype(str)), "exclusion graph_id not found")
    tasks = frame[~frame["graph_id"].astype(str).isin(excluded_ids)].copy()
    require(len(tasks) == EXPECTED_TASKS, f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")
    require(tasks["graph_id"].nunique() == EXPECTED_TASKS, "task graph_id is not unique")

    missing_archives = [
        str(path) for path in tasks["archive_path"].astype(str) if not Path(path).is_file()
    ]
    require(not missing_archives, f"missing source archives: {len(missing_archives)}")

    abstract_eligible = (
        (tasks["parsed_ast_file_count"] > 0)
        & (tasks["parsed_ast_file_count"] == tasks["python_file_count"])
        & (~as_bool(tasks["truncated_package"]))
        & (tasks["total_ast_nodes"] >= ABSTRACT_AST_MIN_NODES)
    )
    leakage_checks = {
        "component_cross_split": max_split_count(tasks, "source_clone_component_id") <= 1,
        "group_key_cross_split": max_split_count(tasks, "group_key") <= 1,
        "archive_sha256_cross_split": max_split_count(tasks, "archive_sha256") <= 1,
        "raw_corpus_cross_split": max_split_count(tasks, "raw_corpus_digest") <= 1,
        "lexical_corpus_cross_split": max_split_count(tasks, "lexical_corpus_digest") <= 1,
        "concrete_ast_cross_split": max_split_count(tasks, "concrete_ast_corpus_digest") <= 1,
        "abstract_ast_80_cross_split": (
            max_split_count(tasks.loc[abstract_eligible], "abstract_ast_corpus_digest") <= 1
        ),
    }
    require(all(leakage_checks.values()), f"clone-aware leakage detected: {leakage_checks}")

    mixed_label_components = (
        tasks.groupby("source_clone_component_id")["label"].nunique().gt(1).sum()
    )
    require(mixed_label_components == 0, "cross-label clone component remains in formal tasks")

    smoke = choose_smoke(tasks)
    summary = (
        tasks.groupby(["split", "label", "source_group"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    disk_guard = Path("/mnt/d") if Path("/mnt/d").exists() else ROOT
    free_gb = shutil.disk_usage(disk_guard).free / (1024 ** 3)
    audit = {
        "status": "ready",
        "input_rows": len(frame),
        "excluded_rows": len(excluded_ids),
        "formal_task_rows": len(tasks),
        "smoke_task_rows": len(smoke),
        "disk_guard": str(disk_guard),
        "free_disk_gb": round(free_gb, 3),
        "leakage_checks": leakage_checks,
        "excluded_graph_ids": sorted(excluded_ids),
    }

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(OUT_TASKS, index=False, encoding="utf-8")
    smoke.to_csv(OUT_SMOKE, index=False, encoding="utf-8")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    OUT_PREFLIGHT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FINAL V10c SOURCE-CLONE-AWARE TASKS READY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\n[RESULT] formal tasks : {len(tasks)}")
    print(f"[RESULT] smoke tasks  : {len(smoke)}")
    print(f"[RESULT] free disk    : {free_gb:.1f} GB at {disk_guard}")
    print(f"[RESULT] {OUT_TASKS}")
    print(f"[RESULT] {OUT_SMOKE}")
    print(f"[RESULT] {OUT_PREFLIGHT}")


if __name__ == "__main__":
    main()
