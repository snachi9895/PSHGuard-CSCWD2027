#!/usr/bin/env python3
"""Build a model-independent source-clone-aware candidate split for 2K samples."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TASK_CSV = (
    ROOT
    / "graph_artifacts/final_v9b_strict_payload_medium_2000_20260609"
    / "step08_diverse_source_tasks_2000.csv"
)
DIAG_DIR = ROOT / "manifests/final_source_clone_aware_split_202606/diagnostics_2000"
FINGERPRINT_CSV = DIAG_DIR / "source_fingerprint_samples.csv"
OUT_DIR = ROOT / "manifests/final_source_clone_aware_split_202606/candidate_2000_v1"
OUT_ALL = OUT_DIR / "source_clone_aware_candidate_2000_all.csv"
OUT_COMPONENTS = OUT_DIR / "source_clone_component_summary.csv"
OUT_DISTRIBUTION = OUT_DIR / "split_distribution_summary.csv"
OUT_LEAKAGE = OUT_DIR / "split_leakage_audit.csv"
OUT_RULES = OUT_DIR / "split_rule_manifest.json"

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
ABSTRACT_AST_MIN_NODES = 80
SEED = 20260611


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def stable_digest(text: str) -> str:
    return hashlib.sha256(f"{SEED}|{text}".encode("utf-8")).hexdigest()


def stratum(row: pd.Series) -> str:
    return f"{row['label']}|{row['source_group']}|size_bin_{int(row['source_size_bin'])}"


def eligible_abstract_ast(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["status"] == "ok")
        & (frame["parsed_ast_file_count"] > 0)
        & (frame["parsed_ast_file_count"] == frame["python_file_count"])
        & (~frame["truncated_package"])
        & (frame["total_ast_nodes"] >= ABSTRACT_AST_MIN_NODES)
    )


def union_by_column(
    frame: pd.DataFrame,
    union_find: UnionFind,
    column: str,
    eligible: pd.Series,
) -> None:
    for value, part in frame[eligible].groupby(column, dropna=False):
        if pd.isna(value) or not str(value) or len(part) < 2:
            continue
        members = sorted(part["graph_id"].astype(str))
        for member in members[1:]:
            union_find.union(members[0], member)


def build_components(frame: pd.DataFrame) -> pd.DataFrame:
    union_find = UnionFind(frame["graph_id"].astype(str).tolist())
    ok = frame["status"] == "ok"
    nonempty_python = ok & (frame["python_file_count"] > 0) & (~frame["truncated_package"])
    complete_ast = (
        nonempty_python
        & (frame["parsed_ast_file_count"] == frame["python_file_count"])
        & (frame["total_ast_nodes"] >= 40)
    )

    # These relations are independent of labels and model-derived V10c graphs.
    union_by_column(frame, union_find, "group_key", ok)
    union_by_column(frame, union_find, "archive_sha256", ok)
    union_by_column(frame, union_find, "raw_corpus_digest", nonempty_python)
    union_by_column(frame, union_find, "lexical_corpus_digest", nonempty_python & (frame["total_tokens"] >= 50))
    union_by_column(frame, union_find, "concrete_ast_corpus_digest", complete_ast)
    union_by_column(frame, union_find, "abstract_ast_corpus_digest", eligible_abstract_ast(frame))

    out = frame.copy()
    out["source_clone_component_root"] = out["graph_id"].astype(str).map(union_find.find)
    roots = sorted(out["source_clone_component_root"].unique())
    component_ids = {root: f"source_clone_component_{index:05d}" for index, root in enumerate(roots)}
    out["source_clone_component_id"] = out["source_clone_component_root"].map(component_ids)
    return out


def targets_for(frame: pd.DataFrame) -> dict[str, Counter]:
    targets = {split: Counter() for split in SPLITS}
    totals = Counter(frame["split_stratum"])
    for name, total in totals.items():
        train = int(round(total * RATIOS["train"]))
        val = int(round(total * RATIOS["val"]))
        test = total - train - val
        targets["train"][name] = train
        targets["val"][name] = val
        targets["test"][name] = test
    return targets


def component_profiles(frame: pd.DataFrame) -> list[dict]:
    profiles = []
    for component_id, part in frame.groupby("source_clone_component_id", sort=True):
        profiles.append(
            {
                "component_id": component_id,
                "samples": len(part),
                "profile": Counter(part["split_stratum"]),
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
        for name in sorted(targets[split]):
            value = counts[split][name] + (profile[name] if split == candidate else 0)
            target = max(1, targets[split][name])
            difference = (value - target) / target
            sample_error += abs(difference)
            overflow += max(0.0, difference) ** 2
    component_error = 0.0
    for split in SPLITS:
        value = component_counts[split] + int(split == candidate)
        component_error += abs(value - component_targets[split]) / max(1, component_targets[split])
    split_order = {"train": 0, "val": 1, "test": 2}
    return sample_error + 4.0 * overflow + 0.15 * component_error, overflow, split_order[candidate]


def assign_components(frame: pd.DataFrame) -> pd.DataFrame:
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
    assignments = {}
    for item in profiles:
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
    out["previous_groupaware_split"] = out["split"]
    out["split"] = out["source_clone_component_id"].map(assignments)
    return out


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    targets = targets_for(frame)
    rows = []
    for split in SPLITS:
        part = frame[frame["split"] == split]
        for name, group in part.groupby("split_stratum"):
            rows.append(
                {
                    "split": split,
                    "stratum": name,
                    "samples": len(group),
                    "target_samples": targets[split][name],
                    "difference": len(group) - targets[split][name],
                    "components": group["source_clone_component_id"].nunique(),
                    "median_archive_bytes": group["source_size"].median(),
                }
            )
    return pd.DataFrame(rows)


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
                "size_bins": ",".join(sorted(part["source_size_bin"].astype(str).unique())),
                "previous_splits": ",".join(sorted(part["previous_groupaware_split"].unique())),
                "example_graph_id": part["graph_id"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["samples", "source_clone_component_id"], ascending=[False, True]
    )


def leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "rows_expected_2000": len(frame) == 2000,
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
            .groupby("lexical_corpus_digest")["split"].nunique().max()
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
            frame[eligible_abstract_ast(frame)]
            .groupby("abstract_ast_corpus_digest")["split"]
            .nunique()
            .max()
            == 1
        ),
    }
    return pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in checks.items()])


def main() -> None:
    for path in (TASK_CSV, FINGERPRINT_CSV):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(TASK_CSV, low_memory=False)
    fingerprints = pd.read_csv(FINGERPRINT_CSV, low_memory=False)
    fingerprint_columns = [
        column
        for column in fingerprints.columns
        if column not in {
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
    if len(frame) != len(tasks):
        raise RuntimeError(f"fingerprint merge lost rows: tasks={len(tasks)} merged={len(frame)}")
    if (frame["status"] != "ok").any():
        raise RuntimeError("candidate split requires all fingerprints to be valid")
    frame["split_stratum"] = frame.apply(stratum, axis=1)
    frame = build_components(frame)
    frame = assign_components(frame)
    frame = frame.sort_values(["split", "label", "source_group", "source_size_bin", "graph_id"]).reset_index(drop=True)

    audit = leakage_audit(frame)
    if not audit["passed"].all():
        raise RuntimeError(audit.to_string(index=False))
    distribution = distribution_summary(frame)
    components = component_summary(frame)

    frame.to_csv(OUT_ALL, index=False, encoding="utf-8")
    for split in SPLITS:
        frame[frame["split"] == split].to_csv(
            OUT_DIR / f"source_clone_aware_candidate_2000_{split}.csv",
            index=False,
            encoding="utf-8",
        )
    components.to_csv(OUT_COMPONENTS, index=False, encoding="utf-8")
    distribution.to_csv(OUT_DISTRIBUTION, index=False, encoding="utf-8")
    audit.to_csv(OUT_LEAKAGE, index=False, encoding="utf-8")
    rules = {
        "status": "candidate_not_locked",
        "seed": SEED,
        "ratios": RATIOS,
        "component_relations": [
            "same normalized package name",
            "same archive SHA-256",
            "same complete raw Python corpus",
            "same complete lexical Python corpus with at least 50 tokens",
            "same complete concrete AST corpus with at least 40 AST nodes",
            f"same complete abstract AST corpus with at least {ABSTRACT_AST_MIN_NODES} AST nodes",
        ],
        "component_relations_do_not_use": [
            "label",
            "V10c graph structure",
            "Event or Param features",
            "node or edge counts",
            "approximate single-file similarity",
        ],
        "assignment_strata": ["label", "source_group", "source_size_bin"],
    }
    with OUT_RULES.open("w", encoding="utf-8") as handle:
        json.dump(rules, handle, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("SOURCE-CLONE-AWARE CANDIDATE SPLIT: 2,000 SAMPLES")
    print("=" * 100)
    print(frame.groupby(["split", "label", "source_group"]).size().to_string())
    print("\nCOMPONENTS")
    print(
        components.groupby("assigned_split")
        .agg(components=("source_clone_component_id", "size"), samples=("samples", "sum"))
        .to_string()
    )
    print("\nLARGEST COMPONENTS")
    print(components.head(15).to_string(index=False))
    print("\nLEAKAGE AUDIT")
    print(audit.to_string(index=False))
    print(f"\n[RESULT] {OUT_ALL}")


if __name__ == "__main__":
    main()
