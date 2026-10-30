#!/usr/bin/env python3
"""Protected export, diagnosis, audit, and conversion pipeline for final V10c graphs."""

from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
V10C_DIR = HERE.parent / "final_graph_v10c_20260610"
V10_DIR = HERE.parent / "final_graph_v10_20260610"
V9_DIR = HERE.parent / "final_graph_v9_20260608"
V3_DIR = HERE.parent / "final_dataset_v3"
for directory in (V10C_DIR, V10_DIR, V9_DIR, V3_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import step02_export_small_graphs_resumable as exporter
import step04_json_to_heterodata_small as converter
import step02_diagnose_v10_event_centered as diagnostics
import step04_audit_v10b_scale_and_templates as scale_audit
from step00_build_final_v10c_file16 import GRAPH_VERSION, FinalV10CFile16GraphBuilder


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v10c_semantic_budget_filtered700k_20260612"
TRAIN_ROOT = ROOT / "experiment_outputs/train_final_v10c_filtered700k_20260612"
FULL_TASK_CSV = GRAPH_ROOT / "step01_final_source_clone_tasks_20883.csv"
SMOKE_TASK_CSV = GRAPH_ROOT / "step01_balanced_smoke_tasks_30.csv"
GRAPH_ELIGIBLE_FULL_TASK_CSV = GRAPH_ROOT / "step05_graph_eligible_tasks_20878.csv"
EXPORT_SUMMARY = GRAPH_ROOT / "step02_v10c_full_export_summary.csv"
FAILURE_CSV = GRAPH_ROOT / "step02_v10c_full_failed_samples.csv"
PT_DIR = TRAIN_ROOT / "heterodata_pt_v10c_source_clone_filtered700k_graph_eligible_normalized"
PT_INDEX = TRAIN_ROOT / "step05_training_pt_index_v10c_source_clone_filtered700k_graph_eligible.csv"

EXPECTED_TASKS = {"smoke": 30, "full": 20883}
EXPECTED_CONVERT_TASKS = {"smoke": 30, "full": 20878}
ABSTRACT_AST_MIN_NODES = 80
PARAM_CONTEXT_FEATURES = [
    "use_count_log_norm", "callsite_count_log_norm", "file_count_log_norm",
    "category_diversity_norm", "source_kind_diversity_norm",
    "ast_link_count_log_norm", "event_link_count_log_norm",
    "reused_anchor", "cross_file_anchor", "event_linked_anchor",
]
AST_CONTEXT_FEATURES = [
    "context_role_event_anchor", "context_role_payload_anchor",
    "context_role_event_ancestor", "context_role_event_nearby",
    "context_role_payload_context", "context_role_neutral_fallback",
]
EVENT_PAYLOAD_FEATURES = [
    "payload_arg_count_norm", "payload_max_length_log", "payload_max_entropy_norm",
    "payload_has_encoded_blob", "payload_has_command", "payload_has_url_or_ip",
]
INDEX_METADATA_COLUMNS = [
    "graph_id", "source_clone_component_id", "source_clone_component_root",
    "oversized_holdout_excluded", "source_complexity_score", "source_size_bin",
    "source_complexity_bin", "python_file_count_bin", "archive_sha256",
    "raw_corpus_digest", "lexical_corpus_digest", "concrete_ast_corpus_digest",
    "abstract_ast_corpus_digest", "python_file_count", "parsed_ast_file_count",
    "skipped_python_file_count", "total_python_bytes", "total_tokens",
    "total_ast_nodes", "truncated_package", "previous_filtered700k_split",
]


def task_csv(task_set: str) -> Path:
    return SMOKE_TASK_CSV if task_set == "smoke" else FULL_TASK_CSV


def read_tasks(task_set: str) -> pd.DataFrame:
    path = task_csv(task_set)
    if not path.is_file():
        raise FileNotFoundError(
            f"task manifest not found: {path}\n"
            "Run step01_prepare_final_v10c_tasks.py in the IDE first."
        )
    frame = pd.read_csv(path, low_memory=False)
    expected = EXPECTED_TASKS[task_set]
    if len(frame) != expected or frame["graph_id"].nunique() != expected:
        raise RuntimeError(f"{task_set} tasks expected {expected} unique rows, got {len(frame)}")
    return frame


def read_conversion_tasks(task_set: str) -> pd.DataFrame:
    path = SMOKE_TASK_CSV if task_set == "smoke" else GRAPH_ELIGIBLE_FULL_TASK_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"conversion task manifest not found: {path}\n"
            "Run step05_prepare_graph_eligible_tasks.py before full conversion."
        )
    frame = pd.read_csv(path, low_memory=False)
    expected = EXPECTED_CONVERT_TASKS[task_set]
    if len(frame) != expected or frame["graph_id"].nunique() != expected:
        raise RuntimeError(f"{task_set} conversion tasks expected {expected} unique rows, got {len(frame)}")
    return frame


def max_split_count(frame: pd.DataFrame, column: str) -> int:
    eligible = frame[column].notna() & frame[column].astype(str).ne("")
    if not eligible.any():
        return 0
    return int(frame.loc[eligible].groupby(column)["split"].nunique().max())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def configure_export(task_set: str) -> None:
    exporter.TASK_CSV = task_csv(task_set)
    exporter.OUT_ROOT = GRAPH_ROOT
    exporter.TEMP_ROOT = GRAPH_ROOT / "_temporary_unpacked"
    exporter.GRAPH_DIR = GRAPH_ROOT / "graphs"
    exporter.META_DIR = GRAPH_ROOT / "meta"
    exporter.SUMMARY_CSV = EXPORT_SUMMARY
    exporter.FAILURE_CSV = FAILURE_CSV
    exporter.GRAPH_VERSION = GRAPH_VERSION
    exporter.GRAPH_BUILDER = FinalV10CFile16GraphBuilder


def build_export_view(task_set: str) -> Path:
    tasks = read_tasks(task_set)
    if not EXPORT_SUMMARY.is_file():
        raise FileNotFoundError(f"export summary not found: {EXPORT_SUMMARY}")
    exported = pd.read_csv(EXPORT_SUMMARY, low_memory=False)
    exported = exported.drop_duplicates(subset=["graph_id"], keep="last")
    view = tasks[["graph_id"]].merge(exported, on="graph_id", how="left", validate="one_to_one")
    missing = view["status"].isna().sum()
    if missing:
        raise RuntimeError(f"{task_set} export is incomplete: {missing} tasks have no result")
    view_path = GRAPH_ROOT / f"step03_{task_set}_export_view.csv"
    view.to_csv(view_path, index=False, encoding="utf-8")
    return view_path


def build_conversion_view(task_set: str) -> Path:
    tasks = read_conversion_tasks(task_set)
    if not EXPORT_SUMMARY.is_file():
        raise FileNotFoundError(f"export summary not found: {EXPORT_SUMMARY}")
    exported = pd.read_csv(EXPORT_SUMMARY, low_memory=False)
    exported = exported.drop_duplicates(subset=["graph_id"], keep="last")
    view = tasks[["graph_id"]].merge(exported, on="graph_id", how="left", validate="one_to_one")
    missing = view["status"].isna().sum()
    if missing:
        raise RuntimeError(f"{task_set} conversion view is incomplete: {missing} tasks have no result")
    if not view["status"].isin(["ok", "skipped_existing"]).all():
        raise RuntimeError(view["status"].value_counts(dropna=False).to_string())
    view_path = GRAPH_ROOT / f"step05_{task_set}_graph_eligible_export_view.csv"
    view.to_csv(view_path, index=False, encoding="utf-8")
    return view_path


def configure_diagnostics(task_set: str, view_path: Path) -> None:
    diagnostics.GRAPH_ROOT = GRAPH_ROOT
    diagnostics.IN_CSV = view_path
    diagnostics.DETAIL_CSV = GRAPH_ROOT / f"step03_{task_set}_sample_metrics.csv"
    diagnostics.LABEL_CSV = GRAPH_ROOT / f"step03_{task_set}_summary_by_label.csv"
    diagnostics.SOURCE_CSV = GRAPH_ROOT / f"step03_{task_set}_summary_by_source.csv"
    diagnostics.EDGE_CSV = GRAPH_ROOT / f"step03_{task_set}_edge_coverage_by_label.csv"


def configure_scale_audit(task_set: str, view_path: Path) -> None:
    scale_audit.GRAPH_ROOT = GRAPH_ROOT
    scale_audit.IN_CSV = view_path
    scale_audit.OUT_DETAIL = GRAPH_ROOT / f"step03_{task_set}_scale_template_audit_detail.csv"
    scale_audit.OUT_SOURCE = GRAPH_ROOT / f"step03_{task_set}_scale_summary_by_source.csv"
    scale_audit.OUT_TEMPLATES = GRAPH_ROOT / f"step03_{task_set}_repeated_template_summary.csv"
    scale_audit.OUT_OVERLAP = GRAPH_ROOT / f"step03_{task_set}_cross_split_template_overlap.csv"
    scale_audit.OUT_SHORTCUT = GRAPH_ROOT / f"step03_{task_set}_size_shortcut_baselines.csv"


def configure_converter(view_path: Path) -> None:
    converter.GRAPH_ROOT = GRAPH_ROOT
    converter.TRAIN_ROOT = TRAIN_ROOT
    converter.IN_CSV = view_path
    converter.OUT_PT_DIR = PT_DIR
    converter.OUT_INDEX_CSV = PT_INDEX
    converter.NORMALIZED_FEATURES = True
    for feature in AST_CONTEXT_FEATURES:
        if feature not in converter.FEATURE_KEYS["ast"]:
            converter.FEATURE_KEYS["ast"].append(feature)
    for feature in ["strict_payload", *PARAM_CONTEXT_FEATURES]:
        if feature not in converter.FEATURE_KEYS["param"]:
            converter.FEATURE_KEYS["param"].append(feature)
    for feature in EVENT_PAYLOAD_FEATURES:
        if feature not in converter.FEATURE_KEYS["event"]:
            converter.FEATURE_KEYS["event"].append(feature)


def enrich_and_validate_index(task_set: str) -> None:
    tasks = read_conversion_tasks(task_set)
    index = pd.read_csv(PT_INDEX, low_memory=False)
    extra_columns = [column for column in INDEX_METADATA_COLUMNS if column in tasks.columns]
    index = index.drop(columns=[column for column in extra_columns if column != "graph_id"], errors="ignore")
    index = index.merge(tasks[extra_columns], on="graph_id", how="inner", validate="one_to_one")
    expected = EXPECTED_CONVERT_TASKS[task_set]
    if len(index) != expected or index["graph_id"].nunique() != expected:
        raise RuntimeError(f"expected {expected} converted rows, got {len(index)}")
    if set(index["status"]) != {"ok"}:
        raise RuntimeError(index["status"].value_counts(dropna=False).to_string())
    missing = [path for path in index["pt_path_v9"] if not Path(str(path)).is_file()]
    if missing:
        raise RuntimeError(f"missing PT files: {len(missing)}")
    abstract_eligible = (
        (index["parsed_ast_file_count"] > 0)
        & (index["parsed_ast_file_count"] == index["python_file_count"])
        & (~as_bool(index["truncated_package"]))
        & (index["total_ast_nodes"] >= ABSTRACT_AST_MIN_NODES)
    )
    leakage_checks = {
        "component_cross_split": max_split_count(index, "source_clone_component_id") <= 1,
        "group_key_cross_split": max_split_count(index, "group_key") <= 1,
        "archive_sha256_cross_split": max_split_count(index, "archive_sha256") <= 1,
        "raw_corpus_cross_split": max_split_count(index, "raw_corpus_digest") <= 1,
        "lexical_corpus_cross_split": max_split_count(index, "lexical_corpus_digest") <= 1,
        "concrete_ast_cross_split": max_split_count(index, "concrete_ast_corpus_digest") <= 1,
        "abstract_ast_80_cross_split": (
            max_split_count(index.loc[abstract_eligible], "abstract_ast_corpus_digest") <= 1
        ),
    }
    if not all(leakage_checks.values()):
        raise RuntimeError(f"clone-aware leakage detected in PT index: {leakage_checks}")
    index["feature_profile"] = "v10c_file16_source_clone_graph_eligible_normalized_v1"
    index.to_csv(PT_INDEX, index=False, encoding="utf-8")


def preflight(task_set: str, min_free_gb: float) -> None:
    tasks = read_tasks(task_set)
    missing_archives = sum(not Path(str(path)).is_file() for path in tasks["archive_path"])
    disk_guard = Path("/mnt/d") if Path("/mnt/d").exists() else ROOT
    free_gb = shutil.disk_usage(disk_guard).free / (1024 ** 3)
    report = {
        "task_set": task_set,
        "tasks": len(tasks),
        "missing_archives": missing_archives,
        "disk_guard": str(disk_guard),
        "free_disk_gb": round(free_gb, 3),
        "required_free_disk_gb": min_free_gb,
        "graph_version": GRAPH_VERSION,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_archives:
        raise RuntimeError(f"missing source archives: {missing_archives}")
    if free_gb < min_free_gb:
        raise RuntimeError(f"free disk {free_gb:.1f} GB is below {min_free_gb:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["preflight", "export", "diagnose", "audit", "convert"],
        default="preflight",
    )
    parser.add_argument("--task-set", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--confirm-full-build", action="store_true")
    args = parser.parse_args()

    if args.stage == "export" and args.task_set == "full" and not args.confirm_full_build:
        raise RuntimeError(
            "Full export is protected. Re-run with --confirm-full-build only after smoke audit passes."
        )

    configure_export(args.task_set)
    if args.stage == "preflight":
        preflight(args.task_set, args.min_free_gb)
        return
    if args.stage == "export":
        sys.argv = [
            sys.argv[0],
            "--workers", str(args.workers),
            "--min-free-gb", str(args.min_free_gb),
        ]
        if args.force:
            sys.argv.append("--force")
        code = exporter.main()
        if code:
            raise SystemExit(code)
        return

    view_path = build_export_view(args.task_set)
    if args.stage == "diagnose":
        configure_diagnostics(args.task_set, view_path)
        diagnostics.main()
    elif args.stage == "audit":
        configure_scale_audit(args.task_set, view_path)
        scale_audit.main()
    elif args.stage == "convert":
        conversion_view_path = build_conversion_view(args.task_set)
        configure_converter(conversion_view_path)
        converter.main()
        enrich_and_validate_index(args.task_set)
        print(f"[RESULT] {PT_INDEX}")


if __name__ == "__main__":
    main()
