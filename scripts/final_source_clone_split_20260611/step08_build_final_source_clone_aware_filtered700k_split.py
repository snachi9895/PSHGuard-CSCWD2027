#!/usr/bin/env python3
"""Build the final source-clone-aware split for the filtered700k dataset.

This script is model-independent. It works on full source archives, not on
graph-derived fingerprints. The final split is designed to be locked before the
next full-data graph export.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import step01_diagnose_source_clones_2000 as fingerprint_base
import step02_build_source_clone_aware_candidate_2000 as component_base


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
BASE_MATCH_CSV = (
    ROOT
    / "experiment_outputs/train_final_v2_20260416/dataset_source_diagnostics_filtered700k"
    / "sample_source_match_detail.csv"
)
LEGACY_GROUP_CSV = ROOT / "manifests/final_dataset_splits_groupaware/final_all_labeled_groupaware.csv"

OUT_DIR = ROOT / "manifests/final_source_clone_aware_split_202606/final_filtered700k_v1"
BASE_ENRICHED_CSV = OUT_DIR / "filtered700k_source_clone_base_manifest.csv"
BASE_SUMMARY_CSV = OUT_DIR / "filtered700k_source_clone_base_summary.csv"

FINGERPRINT_DIR = OUT_DIR / "fingerprints"
FINGERPRINT_CACHE_DIR = FINGERPRINT_DIR / "fingerprint_cache"
FINGERPRINT_CSV = FINGERPRINT_DIR / "source_fingerprint_samples.csv"
FINGERPRINT_FAIL_CSV = FINGERPRINT_DIR / "fingerprint_failures.csv"
FINGERPRINT_SUMMARY_CSV = FINGERPRINT_DIR / "fingerprint_summary.csv"

OUT_ALL = OUT_DIR / "filtered700k_source_clone_aware_all.csv"
OUT_COMPONENTS = OUT_DIR / "source_clone_component_summary.csv"
OUT_DISTRIBUTION = OUT_DIR / "split_distribution_summary.csv"
OUT_LEAKAGE = OUT_DIR / "split_leakage_audit.csv"
OUT_RULES = OUT_DIR / "split_rule_manifest.json"
OUT_COMPONENT_SIZES = OUT_DIR / "holdout_component_size_summary.csv"

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 20260611
ABSTRACT_AST_MIN_NODES = 80
DEFAULT_COMPLEXITY_BINS = 5
DEFAULT_MAX_HOLDOUT_COMPONENT_SIZE = 50
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))


def normalize_package_name(text: str) -> str:
    return str(text or "").strip().lower()


def clean_optional_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text or "").strip()
    if value.lower() == "nan":
        return ""
    return value


def infer_source_family(label: str, source_group: str, legacy_source: str) -> str:
    legacy_source = clean_optional_text(legacy_source).lower()
    if legacy_source:
        return legacy_source
    label = str(label or "").strip().lower()
    source_group = str(source_group or "").strip().lower()
    if label == "benign":
        return "benign"
    if "datadog" in source_group:
        return "datadog"
    if "openssf" in source_group:
        return "openssf"
    return "guo"


def derive_group_key(label: str, package_name_norm: str, source_family: str, legacy_group_key: str) -> str:
    legacy_group_key = clean_optional_text(legacy_group_key).lower()
    if legacy_group_key:
        return legacy_group_key
    if str(label).lower() == "malicious":
        return f"{source_family}__{package_name_norm}"
    return package_name_norm


def quantile_bin(series: pd.Series, bins: int) -> pd.Series:
    if len(series) < bins:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=bins, labels=False, duplicates="drop").astype(int)


def prepare_base_manifest() -> pd.DataFrame:
    for path in (BASE_MATCH_CSV, LEGACY_GROUP_CSV):
        if not path.exists():
            raise FileNotFoundError(path)

    matched = pd.read_csv(BASE_MATCH_CSV, low_memory=False)
    legacy = pd.read_csv(LEGACY_GROUP_CSV, low_memory=False)[["sample_id", "group_key", "source"]]
    legacy = legacy.drop_duplicates("sample_id")

    frame = matched.merge(legacy, on="sample_id", how="left", suffixes=("", "_legacy"))
    frame["package_name_norm"] = frame["package_name"].map(normalize_package_name)
    frame["source"] = frame.apply(
        lambda row: infer_source_family(row["label"], row["source_group"], row.get("source", "")),
        axis=1,
    )
    frame["group_key"] = frame.apply(
        lambda row: derive_group_key(
            row["label"],
            row["package_name_norm"],
            row["source"],
            row.get("group_key", ""),
        ),
        axis=1,
    )
    frame["archive_path"] = frame["source_path"]
    frame["archive_name"] = frame["source_name"]
    frame["archive_group"] = frame["source_group"]
    frame["label_int"] = frame["label"].map({"benign": 0, "malicious": 1}).astype(int)
    frame["previous_filtered700k_split"] = frame["split"]
    frame["archive_exists"] = frame["archive_path"].map(lambda path: Path(str(path)).exists())

    if len(frame) != 20885:
        raise RuntimeError(f"expected 20885 filtered700k rows, got {len(frame)}")
    if (frame["match_status"] != "matched").any():
        bad = int((frame["match_status"] != "matched").sum())
        raise RuntimeError(f"base manifest has unmatched rows: {bad}")
    if (frame["candidate_kind"] != "archive").any():
        bad = int((frame["candidate_kind"] != "archive").sum())
        raise RuntimeError(f"base manifest has non-archive rows: {bad}")
    if not frame["archive_exists"].all():
        missing = frame.loc[~frame["archive_exists"], "archive_path"].head(10).tolist()
        raise RuntimeError(f"base manifest has missing archives: {missing}")
    if frame["graph_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate graph_id found in base manifest")
    if frame["group_key"].isna().any() or (frame["group_key"].astype(str).str.len() == 0).any():
        raise RuntimeError("empty group_key found in base manifest")

    frame = frame.sort_values(["label", "source_group", "split", "graph_id"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BASE_ENRICHED_CSV, index=False, encoding="utf-8")

    summary = (
        frame.groupby(["label", "split", "source_group"], dropna=False)
        .agg(
            samples=("graph_id", "count"),
            groups=("group_key", "nunique"),
            median_archive_bytes=("source_size", "median"),
        )
        .reset_index()
    )
    summary.to_csv(BASE_SUMMARY_CSV, index=False, encoding="utf-8")
    return frame


def load_tasks(max_samples: int = 0) -> pd.DataFrame:
    frame = prepare_base_manifest()
    if max_samples:
        return frame.head(max_samples).copy()
    return frame


def write_public_fingerprint_tables(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = []
    for graph_id in tasks["graph_id"].astype(str):
        path = fingerprint_base.cache_path(graph_id)
        if not fingerprint_base.valid_cache(path):
            missing.append(graph_id)
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if missing:
        raise RuntimeError(f"missing valid fingerprint cache files: {len(missing)}")

    public_rows = [{key: value for key, value in row.items() if key != "files"} for row in rows]
    frame = pd.DataFrame(public_rows)
    frame.to_csv(FINGERPRINT_CSV, index=False, encoding="utf-8")
    frame[frame["status"] != "ok"].to_csv(FINGERPRINT_FAIL_CSV, index=False, encoding="utf-8")
    summary = pd.DataFrame(
        [
            {"metric": "tasks", "value": len(tasks)},
            {"metric": "fingerprints_ok", "value": int((frame["status"] == "ok").sum())},
            {"metric": "fingerprints_failed", "value": int((frame["status"] != "ok").sum())},
            {"metric": "median_python_files", "value": float(frame["python_file_count"].median())},
            {"metric": "median_total_ast_nodes", "value": float(frame["total_ast_nodes"].median())},
        ]
    )
    summary.to_csv(FINGERPRINT_SUMMARY_CSV, index=False, encoding="utf-8")
    return frame


def run_fingerprints(tasks: pd.DataFrame, workers: int, force: bool) -> None:
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint_base.CACHE_DIR = FINGERPRINT_CACHE_DIR

    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fingerprint_base.fingerprint_one, record, force): record["graph_id"]
                for record in tasks.to_dict("records")
            }
            with tqdm(total=len(futures), unit="package", dynamic_ncols=True) as progress:
                try:
                    for future in as_completed(futures):
                        future.result()
                        progress.update(1)
                except KeyboardInterrupt:
                    interrupted = True
                    for future in futures:
                        future.cancel()
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if interrupted:
        raise KeyboardInterrupt
    write_public_fingerprint_tables(tasks)


def add_balance_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    out = frame.copy()
    out["source_complexity_score"] = (
        np.log1p(out["total_ast_nodes"].clip(lower=0))
        + 0.35 * np.log1p(out["python_file_count"].clip(lower=0))
        + 0.15 * np.log1p(out["total_python_bytes"].clip(lower=0))
    )
    out["source_size_bin"] = -1
    out["source_complexity_bin"] = -1
    out["python_file_count_bin"] = -1
    grouped = out.groupby(["label", "source_group"]).groups
    for _, indexes in grouped.items():
        indexes = list(indexes)
        out.loc[indexes, "source_size_bin"] = quantile_bin(out.loc[indexes, "source_size"], bins)
        out.loc[indexes, "source_complexity_bin"] = quantile_bin(
            out.loc[indexes, "source_complexity_score"],
            bins,
        )
        out.loc[indexes, "python_file_count_bin"] = quantile_bin(
            out.loc[indexes, "python_file_count"],
            bins,
        )
    for column in ("source_size_bin", "source_complexity_bin", "python_file_count_bin"):
        out[column] = out[column].astype(int)
    return out


def balance_tokens(row: pd.Series) -> list[str]:
    prefix = f"{row['label']}|{row['source_group']}"
    return [
        f"source|{prefix}",
        f"archive_size|{prefix}|{int(row['source_size_bin'])}",
        f"source_complexity|{prefix}|{int(row['source_complexity_bin'])}",
        f"python_files|{prefix}|{int(row['python_file_count_bin'])}",
    ]


def targets_for(frame: pd.DataFrame) -> dict[str, Counter]:
    totals: Counter = Counter()
    for tokens in frame["balance_tokens"]:
        totals.update(tokens)
    targets = {split: Counter() for split in SPLITS}
    for token, total in totals.items():
        train = int(round(total * RATIOS["train"]))
        val = int(round(total * RATIOS["val"]))
        test = total - train - val
        targets["train"][token] = train
        targets["val"][token] = val
        targets["test"][token] = test
    return targets


def stable_digest(text: str) -> str:
    return component_base.stable_digest(text)


def component_profiles(frame: pd.DataFrame) -> list[dict]:
    profiles = []
    for component_id, part in frame.groupby("source_clone_component_id", sort=True):
        profile: Counter = Counter()
        for tokens in part["balance_tokens"]:
            profile.update(tokens)
        profiles.append(
            {
                "component_id": component_id,
                "samples": len(part),
                "profile": profile,
                "stable_key": stable_digest("|".join(sorted(part["graph_id"].astype(str)))),
            }
        )
    return sorted(profiles, key=lambda item: (-item["samples"], item["stable_key"]))


def score_assignment(
    candidate: str,
    profile: Counter,
    counts: dict[str, Counter],
    targets: dict[str, Counter],
    component_counts: Counter,
    component_targets: dict[str, int],
) -> tuple[float, float, int]:
    sample_error = 0.0
    overflow = 0.0
    for split in SPLITS:
        for token in targets[split]:
            value = counts[split][token] + (profile[token] if split == candidate else 0)
            target = max(1, targets[split][token])
            difference = (value - target) / target
            sample_error += abs(difference)
            overflow += max(0.0, difference) ** 2
    component_error = 0.0
    for split in SPLITS:
        value = component_counts[split] + int(split == candidate)
        component_error += abs(value - component_targets[split]) / max(1, component_targets[split])
    split_order = {"train": 0, "val": 1, "test": 2}
    return sample_error + 5.0 * overflow + 0.5 * component_error, overflow, split_order[candidate]


def assign_components(frame: pd.DataFrame, max_holdout_component_size: int) -> pd.DataFrame:
    targets = targets_for(frame)
    profiles = component_profiles(frame)
    total_components = len(profiles)
    component_targets = {
        "train": int(round(total_components * RATIOS["train"])),
        "val": int(round(total_components * RATIOS["val"])),
    }
    component_targets["test"] = total_components - component_targets["train"] - component_targets["val"]
    counts = {split: Counter() for split in SPLITS}
    component_counts: Counter = Counter()
    assignments: dict[str, str] = {}

    oversized = []
    remaining = profiles
    if max_holdout_component_size > 0:
        oversized = [item for item in profiles if item["samples"] > max_holdout_component_size]
        remaining = [item for item in profiles if item["samples"] <= max_holdout_component_size]
        for item in oversized:
            assignments[item["component_id"]] = "train"
            counts["train"].update(item["profile"])
            component_counts["train"] += 1

    for item in remaining:
        chosen = min(
            SPLITS,
            key=lambda split: score_assignment(
                split,
                item["profile"],
                counts,
                targets,
                component_counts,
                component_targets,
            ),
        )
        assignments[item["component_id"]] = chosen
        counts[chosen].update(item["profile"])
        component_counts[chosen] += 1

    out = frame.copy()
    out["previous_filtered700k_split"] = out["split"]
    out["split"] = out["source_clone_component_id"].map(assignments)
    out["oversized_holdout_excluded"] = out["source_clone_component_id"].isin(
        {item["component_id"] for item in oversized}
    )
    return out


def component_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component_id, part in frame.groupby("source_clone_component_id", sort=True):
        rows.append(
            {
                "source_clone_component_id": component_id,
                "assigned_split": part["split"].iloc[0],
                "samples": len(part),
                "labels": ",".join(sorted(part["label"].unique())),
                "sources": ",".join(sorted(part["source_group"].unique())),
                "package_names": part["group_key"].nunique(),
                "median_archive_bytes": part["source_size"].median(),
                "median_python_files": part["python_file_count"].median(),
                "median_total_ast_nodes": part["total_ast_nodes"].median(),
                "oversized_holdout_excluded": int(part["oversized_holdout_excluded"].iloc[0]),
                "previous_splits": ",".join(sorted(part["previous_filtered700k_split"].unique())),
                "previous_split_count": part["previous_filtered700k_split"].nunique(),
                "example_graph_id": part["graph_id"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["samples", "source_clone_component_id"], ascending=[False, True])


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        part = frame[frame["split"] == split]
        for (label, source_group), group in part.groupby(["label", "source_group"], dropna=False):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "source_group": source_group,
                    "samples": len(group),
                    "components": group["source_clone_component_id"].nunique(),
                    "median_archive_bytes": group["source_size"].median(),
                    "median_python_files": group["python_file_count"].median(),
                    "median_total_ast_nodes": group["total_ast_nodes"].median(),
                    "median_source_complexity": group["source_complexity_score"].median(),
                }
            )
    return pd.DataFrame(rows)


def leakage_audit(frame: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    checks = {
        "rows_expected": len(frame) == expected_rows,
        "unique_graph_ids": frame["graph_id"].nunique() == len(frame),
        "all_splits_nonempty": set(frame["split"]) == set(SPLITS),
        "component_cross_split": frame.groupby("source_clone_component_id")["split"].nunique().max() == 1,
        "group_key_cross_split": frame.groupby("group_key")["split"].nunique().max() == 1,
        "archive_sha256_cross_split": frame.groupby("archive_sha256")["split"].nunique().max() == 1,
        "raw_corpus_cross_split": (
            frame[frame["python_file_count"] > 0].groupby("raw_corpus_digest")["split"].nunique().max() == 1
        ),
        "lexical_corpus_cross_split": (
            frame[(frame["python_file_count"] > 0) & (frame["total_tokens"] >= 50)]
            .groupby("lexical_corpus_digest")["split"]
            .nunique()
            .max()
            == 1
        ),
        "concrete_ast_cross_split": (
            frame[
                (frame["parsed_ast_file_count"] == frame["python_file_count"])
                & (frame["parsed_ast_file_count"] > 0)
                & (frame["total_ast_nodes"] >= 40)
            ]
            .groupby("concrete_ast_corpus_digest")["split"]
            .nunique()
            .max()
            == 1
        ),
        "abstract_ast_80_cross_split": (
            frame[component_base.eligible_abstract_ast(frame)]
            .groupby("abstract_ast_corpus_digest")["split"]
            .nunique()
            .max()
            == 1
        ),
    }
    return pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in checks.items()])


def build_final_split(complexity_bins: int, max_holdout_component_size: int) -> None:
    if not BASE_ENRICHED_CSV.exists():
        prepare_base_manifest()
    if not FINGERPRINT_CSV.exists():
        raise FileNotFoundError(FINGERPRINT_CSV)

    tasks = pd.read_csv(BASE_ENRICHED_CSV, low_memory=False)
    fingerprints = pd.read_csv(FINGERPRINT_CSV, low_memory=False)
    if len(tasks) != 20885:
        raise RuntimeError(f"expected 20885 task rows, got {len(tasks)}")
    if len(fingerprints) != len(tasks):
        raise RuntimeError(f"fingerprint row mismatch: tasks={len(tasks)} fingerprints={len(fingerprints)}")
    if (fingerprints["status"] != "ok").any():
        bad = int((fingerprints["status"] != "ok").sum())
        raise RuntimeError(f"cannot build final split with failed fingerprints: {bad}")

    fingerprint_columns = [
        column
        for column in fingerprints.columns
        if column
        not in {
            "graph_id",
            "sample_id",
            "package_name",
            "version",
            "group_key",
            "label",
            "source_group",
            "old_split",
            "archive_path",
        }
    ]
    frame = tasks.merge(
        fingerprints[["graph_id", *fingerprint_columns]],
        on="graph_id",
        how="inner",
        validate="one_to_one",
    )
    frame = component_base.build_components(frame)
    frame = add_balance_bins(frame, bins=complexity_bins)
    frame["balance_tokens"] = frame.apply(balance_tokens, axis=1)
    frame = assign_components(frame, max_holdout_component_size=max_holdout_component_size)
    frame = frame.sort_values(["split", "label", "source_group", "source_complexity_bin", "graph_id"]).reset_index(
        drop=True
    )

    audit = leakage_audit(frame, expected_rows=len(tasks))
    if not audit["passed"].all():
        raise RuntimeError(audit.to_string(index=False))

    components = component_summary(frame)
    distribution = distribution_summary(frame)
    holdout_sizes = (
        components[components["assigned_split"].isin(["val", "test"])]
        .groupby(["assigned_split", "samples"])
        .size()
        .reset_index(name="components")
        .sort_values(["assigned_split", "samples"])
    )

    frame.drop(columns=["balance_tokens"]).to_csv(OUT_ALL, index=False, encoding="utf-8")
    for split in SPLITS:
        frame[frame["split"] == split].drop(columns=["balance_tokens"]).to_csv(
            OUT_DIR / f"filtered700k_source_clone_aware_{split}.csv",
            index=False,
            encoding="utf-8",
        )
    components.to_csv(OUT_COMPONENTS, index=False, encoding="utf-8")
    distribution.to_csv(OUT_DISTRIBUTION, index=False, encoding="utf-8")
    audit.to_csv(OUT_LEAKAGE, index=False, encoding="utf-8")
    holdout_sizes.to_csv(OUT_COMPONENT_SIZES, index=False, encoding="utf-8")

    with OUT_RULES.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "final_candidate_ready_for_graph_export",
                "version": "final_filtered700k_v1",
                "seed": SEED,
                "ratios": RATIOS,
                "rows": len(tasks),
                "source_clone_component_rules": [
                    "same normalized package name",
                    "same archive SHA-256",
                    "same complete raw Python corpus",
                    "same complete lexical Python corpus with at least 50 tokens",
                    "same complete concrete AST corpus with at least 40 AST nodes",
                    f"same complete abstract AST corpus with at least {ABSTRACT_AST_MIN_NODES} AST nodes",
                ],
                "holdout_rules": [
                    f"components larger than {max_holdout_component_size} samples are assigned to train"
                    if max_holdout_component_size > 0
                    else "no oversized-holdout exclusion",
                    "balance label and source group",
                    "balance archive-size quantile",
                    "balance full-source complexity quantile",
                    "balance Python-file-count quantile",
                ],
                "does_not_use": [
                    "graph structure",
                    "Event or Param features",
                    "node or edge counts from built graphs",
                    "model predictions or model scores",
                ],
                "inputs": {
                    "matched_source_manifest": str(BASE_MATCH_CSV),
                    "legacy_group_manifest": str(LEGACY_GROUP_CSV),
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 100)
    print("FINAL FILTERED700K SOURCE-CLONE-AWARE SPLIT")
    print("=" * 100)
    print(frame.groupby(["split", "label", "source_group"]).size().to_string())
    print("\nDISTRIBUTION")
    print(distribution.to_string(index=False))
    print("\nLARGEST COMPONENTS")
    print(components.head(20).to_string(index=False))
    print("\nLEAKAGE AUDIT")
    print(audit.to_string(index=False))
    print(f"\n[RESULT] {OUT_ALL}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "fingerprint", "split", "all"], default="all")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--complexity-bins", type=int, default=DEFAULT_COMPLEXITY_BINS)
    parser.add_argument(
        "--max-holdout-component-size",
        type=int,
        default=DEFAULT_MAX_HOLDOUT_COMPONENT_SIZE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples and args.stage in {"split", "all"}:
        raise RuntimeError("--max-samples is only supported for prepare/fingerprint smoke runs")

    if args.stage in {"prepare", "fingerprint", "all"}:
        tasks = load_tasks(max_samples=args.max_samples)
        print("=" * 100)
        print("FINAL FILTERED700K SOURCE-CLONE BASE MANIFEST")
        print("=" * 100)
        print(f"[INFO] rows        : {len(tasks)}")
        print(f"[INFO] matched csv : {BASE_MATCH_CSV}")
        print(f"[INFO] output dir  : {OUT_DIR}")
        print(f"[INFO] full rows   : 20885")
        if args.stage in {"fingerprint", "all"}:
            print("-" * 100)
            print("SOURCE FINGERPRINT EXPORT")
            print(f"[INFO] workers     : {args.workers}")
            print(f"[INFO] archive I/O : read Python members directly; no persistent unpacked source")
            print(f"[INFO] cache dir   : {FINGERPRINT_CACHE_DIR}")
            try:
                run_fingerprints(tasks, workers=args.workers, force=args.force)
            except KeyboardInterrupt:
                print("[INFO] interrupted; completed fingerprint cache files are reusable")
                return 130

    if args.stage in {"split", "all"}:
        build_final_split(
            complexity_bins=args.complexity_bins,
            max_holdout_component_size=args.max_holdout_component_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
