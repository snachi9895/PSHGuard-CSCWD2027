#!/usr/bin/env python3
"""Build the immutable graph-eligible v2 strict40 source-clone split.

The v2 split is a model-independent repair of the v1 split.  Source-clone
relations are rebuilt over the original 20,885-row fingerprint universe, then
induced onto the already frozen 20,878 graph-eligible population.  No sample,
label, graph, or PT artifact is added, removed, or regenerated.

Assignment is solved lexicographically with SciPy/HiGHS:

1. keep every strict40 component in one split, force components larger than 50
   samples into train, preserve exact split-by-label counts and a deterministic
   feasible source balance, and minimize the number of rows moved from v1;
2. at the minimum migration count, minimize L1 drift in the three pre-existing
   source-size/source-complexity/Python-file-count balance bins;
3. at both optima, use a fixed SHA-256 objective to select one stable solution.

Model predictions, Cerebro text, VirusTotal results, and test metrics are never
read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import step02_build_source_clone_aware_candidate_2000 as component_base


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
V1_DIR = ROOT / "manifests/final_source_clone_aware_split_202606/final_filtered700k_v1"
V1_ALL_CSV = V1_DIR / "filtered700k_source_clone_aware_all.csv"
V1_FINGERPRINT_CSV = V1_DIR / "fingerprints/source_fingerprint_samples.csv"
V1_FORMAL_INDEX = (
    ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "step05_training_pt_index_v10c_source_clone_filtered700k_graph_eligible.csv"
)
THRESHOLD_EVIDENCE_CSV = (
    ROOT
    / "manifests/final_source_clone_aware_split_202606/diagnostics_2000"
    / "abstract_ast_threshold_sensitivity.csv"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "manifests/final_source_clone_aware_split_202606"
    / "final_filtered700k_graph_eligible_v2_strict40_20260722"
)
DEFAULT_V10C_INDEX = (
    ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "split_v2_strict40_20260722"
    / "step01_training_pt_index_v10c_v2_strict40_graph_eligible.csv"
)

SPLITS = ("train", "val", "test")
SPLIT_TO_INDEX = {name: index for index, name in enumerate(SPLITS)}
ABSTRACT_AST_MIN_NODES = 40
MAX_HOLDOUT_COMPONENT_SIZE = 50
SEED = 20260722
EXPECTED_RELATION_ROWS = 20885
EXPECTED_FORMAL_ROWS = 20878
EXPECTED_FORMAL_COMPONENTS = 11813
EXPECTED_CROSS_V1_COMPONENTS = 13
EXPECTED_CROSS_V1_ROWS = 5445
EXPECTED_MINIMUM_MIGRATION = 2142

EXPECTED_SPLIT_LABEL_COUNTS = {
    ("train", "benign"): 8492,
    ("train", "malicious"): 8121,
    ("val", "benign"): 1082,
    ("val", "malicious"): 1055,
    ("test", "benign"): 1069,
    ("test", "malicious"): 1059,
}

# With every >50 component fixed in train, strict40 components already force
# 6,725 Guo rows into train.  This exceeds v1's 6,449 Guo-train target, so the
# old exact source distribution is infeasible.  The remaining sources are
# apportioned symmetrically between validation and test without model outputs.
EXPECTED_SOURCE_TARGETS = {
    ("train", "benign", "sdist_benign"): 8492,
    ("train", "malicious", "external_datadog_samples"): 1396,
    ("train", "malicious", "sdist_malicious"): 6725,
    ("val", "benign", "sdist_benign"): 1082,
    ("val", "malicious", "external_datadog_samples"): 362,
    ("val", "malicious", "sdist_malicious"): 693,
    ("test", "benign", "sdist_benign"): 1069,
    ("test", "malicious", "external_datadog_samples"): 363,
    ("test", "malicious", "sdist_malicious"): 696,
}

BALANCE_COLUMNS = (
    "source_size_bin",
    "source_complexity_bin",
    "python_file_count_bin",
)
FINGERPRINT_COLUMNS = (
    "archive_sha256",
    "raw_corpus_digest",
    "lexical_corpus_digest",
    "concrete_ast_corpus_digest",
    "abstract_ast_corpus_digest",
    "python_file_count",
    "parsed_ast_file_count",
    "total_python_bytes",
    "total_tokens",
    "total_ast_nodes",
    "truncated_package",
    "status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_abstract_eligible(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["status"] == "ok")
        & (frame["parsed_ast_file_count"] > 0)
        & (frame["parsed_ast_file_count"] == frame["python_file_count"])
        & (~frame["truncated_package"].astype(bool))
        & (frame["total_ast_nodes"] >= ABSTRACT_AST_MIN_NODES)
    )


def build_strict40_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Build source components with an explicit abstract-AST >=40 rule."""
    union_find = component_base.UnionFind(frame["graph_id"].astype(str).tolist())
    ok = frame["status"] == "ok"
    nonempty_python = ok & (frame["python_file_count"] > 0) & (~frame["truncated_package"].astype(bool))
    complete_ast = (
        nonempty_python
        & (frame["parsed_ast_file_count"] == frame["python_file_count"])
        & (frame["total_ast_nodes"] >= 40)
    )

    component_base.union_by_column(frame, union_find, "group_key", ok)
    component_base.union_by_column(frame, union_find, "archive_sha256", ok)
    component_base.union_by_column(frame, union_find, "raw_corpus_digest", nonempty_python)
    component_base.union_by_column(
        frame,
        union_find,
        "lexical_corpus_digest",
        nonempty_python & (frame["total_tokens"] >= 50),
    )
    component_base.union_by_column(
        frame,
        union_find,
        "concrete_ast_corpus_digest",
        complete_ast,
    )
    component_base.union_by_column(
        frame,
        union_find,
        "abstract_ast_corpus_digest",
        strict_abstract_eligible(frame),
    )

    out = frame.copy()
    out["source_clone_component_root"] = out["graph_id"].astype(str).map(union_find.find)
    roots = sorted(out["source_clone_component_root"].unique())
    ids = {root: f"source_clone_component_v2_{index:05d}" for index, root in enumerate(roots)}
    out["source_clone_component_id"] = out["source_clone_component_root"].map(ids)
    return out


def assert_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def normalized_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool).astype(str)
    return series.fillna("").astype(str)


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (V1_ALL_CSV, V1_FINGERPRINT_CSV, V1_FORMAL_INDEX, THRESHOLD_EVIDENCE_CSV):
        if not path.exists():
            raise FileNotFoundError(path)

    full = pd.read_csv(V1_ALL_CSV, low_memory=False)
    fingerprints = pd.read_csv(V1_FINGERPRINT_CSV, low_memory=False)
    formal = pd.read_csv(V1_FORMAL_INDEX, low_memory=False)
    assert_columns(full, ("graph_id", "split", "label", "source_group", "group_key", *FINGERPRINT_COLUMNS), "v1 all")
    assert_columns(fingerprints, ("graph_id", *FINGERPRINT_COLUMNS), "v1 fingerprints")
    assert_columns(
        formal,
        (
            "graph_id",
            "sample_id",
            "split",
            "label",
            "label_int",
            "source_group",
            "source_clone_component_id",
            "source_clone_component_root",
            "pt_path_v9",
            "graph_json_path_v9",
            *FINGERPRINT_COLUMNS,
            *BALANCE_COLUMNS,
        ),
        "v1 formal index",
    )

    if len(full) != EXPECTED_RELATION_ROWS or full["graph_id"].nunique() != len(full):
        raise RuntimeError(f"v1 relation universe contract failed: rows={len(full)}")
    if len(fingerprints) != EXPECTED_RELATION_ROWS or fingerprints["graph_id"].nunique() != len(fingerprints):
        raise RuntimeError(f"v1 fingerprint contract failed: rows={len(fingerprints)}")
    if len(formal) != EXPECTED_FORMAL_ROWS or formal["graph_id"].nunique() != len(formal):
        raise RuntimeError(f"formal population contract failed: rows={len(formal)}")
    if formal["sample_id"].nunique() != len(formal):
        raise RuntimeError("formal sample IDs are not unique")
    if not set(formal["graph_id"].astype(str)).issubset(set(full["graph_id"].astype(str))):
        raise RuntimeError("formal graph IDs are not a subset of the v1 relation universe")

    fingerprint_check = full[["graph_id", *FINGERPRINT_COLUMNS]].merge(
        fingerprints[["graph_id", *FINGERPRINT_COLUMNS]],
        on="graph_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_all", "_fingerprint"),
    )
    if (fingerprint_check["_merge"] != "both").any():
        raise RuntimeError("v1 all/fingerprint graph-ID mismatch")
    for column in FINGERPRINT_COLUMNS:
        left = normalized_values(fingerprint_check[f"{column}_all"])
        right = normalized_values(fingerprint_check[f"{column}_fingerprint"])
        if not left.equals(right):
            count = int((left != right).sum())
            raise RuntimeError(f"v1 fingerprint mismatch for {column}: {count}")

    formal_check_columns = (
        "sample_id",
        "group_key",
        "label",
        "label_int",
        "split",
        "source_group",
        *FINGERPRINT_COLUMNS,
    )
    formal_check = formal[["graph_id", *formal_check_columns]].merge(
        full[["graph_id", *formal_check_columns]],
        on="graph_id",
        how="left",
        validate="one_to_one",
        suffixes=("_formal", "_full"),
    )
    for column in formal_check_columns:
        left = normalized_values(formal_check[f"{column}_formal"])
        right = normalized_values(formal_check[f"{column}_full"])
        if not left.equals(right):
            count = int((left != right).sum())
            raise RuntimeError(f"formal/v1 mismatch for {column}: {count}")

    threshold_evidence = pd.read_csv(THRESHOLD_EVIDENCE_CSV)
    if 40 not in set(threshold_evidence["minimum_ast_nodes"].astype(int)):
        raise RuntimeError("the pre-existing threshold sensitivity audit does not contain threshold 40")

    observed_counts = formal.groupby(["split", "label"]).size().to_dict()
    if observed_counts != EXPECTED_SPLIT_LABEL_COUNTS:
        raise RuntimeError(f"unexpected v1 formal split/label counts: {observed_counts}")
    return full, fingerprints, formal


def component_profiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    parts: dict[str, pd.DataFrame] = {}
    for component_id, part in frame.groupby("source_clone_component_id", sort=True):
        labels = sorted(part["label"].astype(str).unique())
        if len(labels) != 1:
            raise RuntimeError(f"formal strict40 component crosses labels: {component_id} {labels}")
        old_counts = Counter(part["v1_split"].astype(str))
        rows.append(
            {
                "source_clone_component_id": component_id,
                "source_clone_component_root": str(part["source_clone_component_root"].iloc[0]),
                "samples": len(part),
                "label": labels[0],
                "train_v1": old_counts["train"],
                "val_v1": old_counts["val"],
                "test_v1": old_counts["test"],
                "v1_split_count": part["v1_split"].nunique(),
                "sources": ",".join(sorted(part["source_group"].astype(str).unique())),
                "source_count": part["source_group"].nunique(),
                "example_graph_id": str(part["graph_id"].iloc[0]),
            }
        )
        parts[str(component_id)] = part
    profiles = pd.DataFrame(rows).sort_values("source_clone_component_id").reset_index(drop=True)
    return profiles, parts


def add_constraint_row(
    row_index: int,
    entries: Iterable[tuple[int, float]],
    row_ids: list[int],
    col_ids: list[int],
    values: list[float],
) -> None:
    for column, value in entries:
        if value:
            row_ids.append(row_index)
            col_ids.append(column)
            values.append(float(value))


def build_hard_constraints(
    profiles: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
) -> tuple[csr_matrix, np.ndarray, np.ndarray, Bounds, np.ndarray]:
    component_ids = profiles["source_clone_component_id"].astype(str).tolist()
    component_index = {component_id: index for index, component_id in enumerate(component_ids)}
    nx = len(component_ids) * len(SPLITS)
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def variable(component: int, split: str) -> int:
        return component * len(SPLITS) + SPLIT_TO_INDEX[split]

    row = 0
    for component in range(len(component_ids)):
        add_constraint_row(
            row,
            ((variable(component, split), 1) for split in SPLITS),
            row_ids,
            col_ids,
            values,
        )
        lower.append(1)
        upper.append(1)
        row += 1

    for (split, label), target in sorted(EXPECTED_SPLIT_LABEL_COUNTS.items()):
        entries = []
        for component, profile in profiles.iterrows():
            if profile["label"] == label:
                entries.append((variable(component, split), int(profile["samples"])))
        add_constraint_row(row, entries, row_ids, col_ids, values)
        lower.append(target)
        upper.append(target)
        row += 1

    source_keys = sorted({(label, source) for _, label, source in EXPECTED_SOURCE_TARGETS})
    component_source_counts: dict[tuple[str, str], np.ndarray] = {}
    for label, source in source_keys:
        counts = np.zeros(len(component_ids), dtype=np.int64)
        for component_id, part in parts.items():
            component = component_index[component_id]
            counts[component] = int(((part["label"] == label) & (part["source_group"] == source)).sum())
        component_source_counts[(label, source)] = counts

    for (split, label, source), target in sorted(EXPECTED_SOURCE_TARGETS.items()):
        counts = component_source_counts[(label, source)]
        entries = [
            (variable(component, split), int(count))
            for component, count in enumerate(counts)
            if count
        ]
        add_constraint_row(row, entries, row_ids, col_ids, values)
        lower.append(target)
        upper.append(target)
        row += 1

    matrix = coo_matrix((values, (row_ids, col_ids)), shape=(row, nx)).tocsr()
    lb = np.asarray(lower, dtype=float)
    ub = np.asarray(upper, dtype=float)
    variable_lb = np.zeros(nx, dtype=float)
    variable_ub = np.ones(nx, dtype=float)
    for component, profile in profiles.iterrows():
        if int(profile["samples"]) > MAX_HOLDOUT_COMPONENT_SIZE:
            variable_ub[variable(component, "val")] = 0
            variable_ub[variable(component, "test")] = 0

    migration = np.zeros(nx, dtype=float)
    for component, profile in profiles.iterrows():
        size = int(profile["samples"])
        for split in SPLITS:
            migration[variable(component, split)] = size - int(profile[f"{split}_v1"])
    return matrix, lb, ub, Bounds(variable_lb, variable_ub), migration


def solve_milp(
    objective: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    matrix: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    stage: str,
):
    print(f"[MILP] {stage}: variables={len(objective)} constraints={matrix.shape[0]}")
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0, "presolve": True, "disp": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"{stage} failed: status={result.status} message={result.message}")
    print(f"[MILP] {stage}: optimal objective={result.fun}")
    return result


def build_balance_constraints(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    hard_matrix: csr_matrix,
    hard_lower: np.ndarray,
    hard_upper: np.ndarray,
    migration: np.ndarray,
    minimum_migration: int,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, Bounds, np.ndarray, np.ndarray, list[tuple]]:
    component_ids = profiles["source_clone_component_id"].astype(str).tolist()
    component_index = {component_id: index for index, component_id in enumerate(component_ids)}
    nx = len(component_ids) * len(SPLITS)

    tokens = sorted(
        {
            (str(row.label), str(row.source_group), column, int(getattr(row, column)))
            for row in frame.itertuples(index=False)
            for column in BALANCE_COLUMNS
        }
    )
    target_keys = [(split, *token) for split in SPLITS for token in tokens]
    nt = len(target_keys)
    nvars = nx + 2 * nt

    token_counts: dict[tuple, np.ndarray] = {}
    for token in tokens:
        label, source, column, value = token
        counts = np.zeros(len(component_ids), dtype=np.int64)
        for component_id, part in parts.items():
            counts[component_index[component_id]] = int(
                (
                    (part["label"].astype(str) == label)
                    & (part["source_group"].astype(str) == source)
                    & (part[column].astype(int) == value)
                ).sum()
            )
        token_counts[token] = counts

    target_counts = (
        frame.assign(_one=1)
        .melt(
            id_vars=["v1_split", "label", "source_group", "_one"],
            value_vars=list(BALANCE_COLUMNS),
            var_name="balance_column",
            value_name="balance_value",
        )
        .groupby(["v1_split", "label", "source_group", "balance_column", "balance_value"])["_one"]
        .sum()
        .to_dict()
    )

    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for target_index, (split, label, source, column, value) in enumerate(target_keys):
        split_index = SPLIT_TO_INDEX[split]
        counts = token_counts[(label, source, column, value)]
        for component, count in enumerate(counts):
            if count:
                row_ids.append(target_index)
                col_ids.append(component * len(SPLITS) + split_index)
                values.append(float(count))
        row_ids.extend((target_index, target_index))
        col_ids.extend((nx + target_index, nx + nt + target_index))
        values.extend((-1.0, 1.0))
        target = int(target_counts.get((split, label, source, column, value), 0))
        lower.append(target)
        upper.append(target)

    token_matrix = coo_matrix((values, (row_ids, col_ids)), shape=(nt, nvars)).tocsr()
    hard_extended = hstack(
        [hard_matrix, csr_matrix((hard_matrix.shape[0], 2 * nt))],
        format="csr",
    )
    migration_row = csr_matrix(np.concatenate([migration, np.zeros(2 * nt)])[None, :])
    matrix = vstack([hard_extended, migration_row, token_matrix], format="csr")
    lb = np.concatenate([hard_lower, [minimum_migration], np.asarray(lower, dtype=float)])
    ub = np.concatenate([hard_upper, [minimum_migration], np.asarray(upper, dtype=float)])

    base_lb = np.asarray(hard_matrix.shape[1] * [0.0], dtype=float)
    base_ub = np.asarray(hard_matrix.shape[1] * [1.0], dtype=float)
    for component, profile in profiles.iterrows():
        if int(profile["samples"]) > MAX_HOLDOUT_COMPONENT_SIZE:
            base_ub[component * len(SPLITS) + SPLIT_TO_INDEX["val"]] = 0
            base_ub[component * len(SPLITS) + SPLIT_TO_INDEX["test"]] = 0
    bounds = Bounds(
        np.concatenate([base_lb, np.zeros(2 * nt)]),
        np.concatenate([base_ub, np.full(2 * nt, np.inf)]),
    )
    integrality = np.concatenate([np.ones(nx), np.zeros(2 * nt)])
    objective = np.concatenate([np.zeros(nx), np.ones(2 * nt)])
    return matrix, lb, ub, bounds, integrality, objective, target_keys


def stable_objective(profiles: pd.DataFrame, variable_count: int) -> np.ndarray:
    objective = np.zeros(variable_count, dtype=float)
    for component, profile in profiles.iterrows():
        root = str(profile["source_clone_component_root"])
        for split in SPLITS:
            raw = hashlib.sha256(f"{SEED}|{root}|{split}".encode("utf-8")).digest()
            objective[component * len(SPLITS) + SPLIT_TO_INDEX[split]] = int.from_bytes(raw[:3], "big") + 1
    return objective


def extract_assignments(result_x: np.ndarray, profiles: pd.DataFrame) -> dict[str, str]:
    nx = len(profiles) * len(SPLITS)
    values = result_x[:nx].reshape(len(profiles), len(SPLITS))
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:
        raise RuntimeError("MILP returned a non-integral component assignment")
    if not (rounded.sum(axis=1) == 1).all():
        raise RuntimeError("MILP assignment is not one-hot")
    chosen = rounded.argmax(axis=1)
    return {
        str(profiles.iloc[index]["source_clone_component_id"]): SPLITS[int(split_index)]
        for index, split_index in enumerate(chosen)
    }


def cross_split_max(frame: pd.DataFrame, column: str, eligible: pd.Series | None = None) -> int:
    selected = frame if eligible is None else frame[eligible]
    selected = selected[selected[column].notna() & (selected[column].astype(str) != "")]
    if selected.empty:
        return 0
    return int(selected.groupby(column)["split"].nunique().max())


def leakage_audit(frame: pd.DataFrame, old_index: pd.DataFrame, migration: int) -> pd.DataFrame:
    old_by_id = old_index.set_index("graph_id")
    new_by_id = frame.set_index("graph_id")
    same_ids = set(old_by_id.index.astype(str)) == set(new_by_id.index.astype(str))
    immutable_columns = [
        "label",
        "label_int",
        "archive_sha256",
        "graph_json_path_v9",
        "pt_path_v9",
    ]
    immutable_ok = True
    if same_ids:
        old_aligned = old_by_id.loc[new_by_id.index]
        immutable_ok = all(
            normalized_values(old_aligned[column]).equals(normalized_values(new_by_id[column]))
            for column in immutable_columns
        )

    complete_python = (
        (frame["status"] == "ok")
        & (frame["python_file_count"] > 0)
        & (~frame["truncated_package"].astype(bool))
    )
    complete_ast = (
        complete_python
        & (frame["parsed_ast_file_count"] == frame["python_file_count"])
        & (frame["total_ast_nodes"] >= 40)
    )
    observed_split_label = frame.groupby(["split", "label"]).size().to_dict()
    observed_sources = frame.groupby(["split", "label", "source_group"]).size().to_dict()
    checks = {
        "formal_rows_20878": len(frame) == EXPECTED_FORMAL_ROWS,
        "formal_graph_ids_unique": frame["graph_id"].nunique() == len(frame),
        "formal_sample_ids_unique": frame["sample_id"].nunique() == len(frame),
        "formal_population_identical_to_v1": same_ids,
        "labels_hashes_graphs_and_pt_paths_unchanged": immutable_ok,
        "split_label_counts_exact": observed_split_label == EXPECTED_SPLIT_LABEL_COUNTS,
        "split_label_source_targets_exact": observed_sources == EXPECTED_SOURCE_TARGETS,
        "minimum_migration_exact": migration == EXPECTED_MINIMUM_MIGRATION,
        "strict40_component_cross_split_zero": cross_split_max(frame, "source_clone_component_id") == 1,
        "normalized_package_group_cross_split_zero": cross_split_max(frame, "group_key") == 1,
        "archive_sha256_cross_split_zero": cross_split_max(frame, "archive_sha256") == 1,
        "raw_corpus_cross_split_zero": cross_split_max(frame, "raw_corpus_digest", complete_python) == 1,
        "lexical_corpus_50_cross_split_zero": cross_split_max(
            frame,
            "lexical_corpus_digest",
            complete_python & (frame["total_tokens"] >= 50),
        )
        == 1,
        "concrete_ast_40_cross_split_zero": cross_split_max(
            frame,
            "concrete_ast_corpus_digest",
            complete_ast,
        )
        == 1,
        "abstract_ast_40_cross_split_zero": cross_split_max(
            frame,
            "abstract_ast_corpus_digest",
            strict_abstract_eligible(frame),
        )
        == 1,
        "holdout_component_size_at_most_50": int(
            frame[frame["split"].isin(["val", "test"])]
            .groupby("source_clone_component_id")
            .size()
            .max()
        )
        <= MAX_HOLDOUT_COMPONENT_SIZE,
    }
    return pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in checks.items()])


def component_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component_id, part in frame.groupby("source_clone_component_id", sort=True):
        old_counts = Counter(part["v1_split"])
        rows.append(
            {
                "source_clone_component_id": component_id,
                "source_clone_component_root": part["source_clone_component_root"].iloc[0],
                "assigned_split": part["split"].iloc[0],
                "samples": len(part),
                "label": part["label"].iloc[0],
                "sources": ",".join(sorted(part["source_group"].astype(str).unique())),
                "source_count": part["source_group"].nunique(),
                "v1_train": old_counts["train"],
                "v1_val": old_counts["val"],
                "v1_test": old_counts["test"],
                "v1_split_count": part["v1_split"].nunique(),
                "moved_samples": int(part["split_changed"].sum()),
                "oversized_holdout_excluded": int(len(part) > MAX_HOLDOUT_COMPONENT_SIZE),
                "median_archive_bytes": float(part["source_size"].median()),
                "median_python_files": float(part["python_file_count"].median()),
                "median_total_ast_nodes": float(part["total_ast_nodes"].median()),
                "example_graph_id": part["graph_id"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["samples", "source_clone_component_id"],
        ascending=[False, True],
    )


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "label", "source_group"], dropna=False)
        .agg(
            samples=("graph_id", "size"),
            components=("source_clone_component_id", "nunique"),
            median_archive_bytes=("source_size", "median"),
            median_python_files=("python_file_count", "median"),
            median_total_ast_nodes=("total_ast_nodes", "median"),
            median_source_complexity=("source_complexity_score", "median"),
        )
        .reset_index()
    )


def balance_bin_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        old_part = frame[frame["v1_split"] == split]
        new_part = frame[frame["split"] == split]
        for label in sorted(frame["label"].unique()):
            for source in sorted(frame.loc[frame["label"] == label, "source_group"].unique()):
                for column in BALANCE_COLUMNS:
                    values = sorted(set(frame[column].astype(int)))
                    for value in values:
                        old_count = int(
                            (
                                (old_part["label"] == label)
                                & (old_part["source_group"] == source)
                                & (old_part[column].astype(int) == value)
                            ).sum()
                        )
                        new_count = int(
                            (
                                (new_part["label"] == label)
                                & (new_part["source_group"] == source)
                                & (new_part[column].astype(int) == value)
                            ).sum()
                        )
                        rows.append(
                            {
                                "split": split,
                                "label": label,
                                "source_group": source,
                                "balance_column": column,
                                "balance_value": value,
                                "v1_samples": old_count,
                                "v2_samples": new_count,
                                "difference": new_count - old_count,
                                "absolute_difference": abs(new_count - old_count),
                            }
                        )
    return pd.DataFrame(rows)


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def build_readme(
    minimum_migration: int,
    balance_l1: int,
    transition: pd.DataFrame,
    output_hashes: dict[str, str],
    output_dir: Path,
    v10c_index: Path,
) -> str:
    test_old_kept = int(transition.loc[(transition["v1_split"] == "test") & (transition["split"] == "test"), "samples"].sum())
    test_out = int(transition.loc[(transition["v1_split"] == "test") & (transition["split"] != "test"), "samples"].sum())
    test_in = int(transition.loc[(transition["v1_split"] != "test") & (transition["split"] == "test"), "samples"].sum())
    return f"""# filtered700k graph-eligible V2 strict40 划分说明

## 定位

这是 V1 正式划分的**独立、不可覆盖、模型无关修复版**。它只修改 split 与源码克隆组件字段；正式样本集合、标签、归档 SHA、V10c 图和 PT 路径均保持不变。

- V2 目录：`{output_dir}`
- V1 源码划分：`{V1_DIR}`
- V1 正式 20,878 PT 索引：`{V1_FORMAL_INDEX}`
- V2 派生 PT 索引：`{v10c_index}`

V1 结果与预测继续保留为历史诊断 artifact，不覆盖、不删除；后续所有可训练方法必须共同使用 V2，不能只修改 Cerebro。

## 为什么需要 V2

V1 的完整抽象 AST 关系只在 `total_ast_nodes >= 80` 时合并。人工复核发现一个 5,319 行、恰为 72 AST 节点的恶意模板家族被拆进 train/val/test；该家族源码近乎逐行一致。阈值 40 并非根据模型结果新造：2026-06-11 的 `abstract_ast_threshold_sensitivity.csv` 已预先包含 40 节点诊断。

V2 将完整、可解析、未截断包的抽象 AST 门槛固定为 `>=40`，其他关系保持不变：规范化包名、archive SHA、完整 raw corpus、至少 50 token 的 lexical corpus、至少 40 节点的 concrete AST corpus。

## 两层数据契约

- 源码关系发现全集：20,885 行（复用 V1 源码指纹，不重新扫描源码）。
- 正式分配人口：固定 20,878 行 graph-eligible 样本。

历史排除的 7 行没有被 V2 再次删除或替换；V2 既不新增样本，也不删除正式样本。

## 分配规则

1. strict40 连通组件不可跨 split。
2. 组件大于 50 行时只允许进入 train，沿用 V1 的 holdout 规则。
3. 精确保留 V1 正式 split 的总量与良恶数量：train/val/test = 16,613/2,137/2,128。
4. 在这些硬约束下，MILP 第一阶段最小化相对 V1 改变 split 的样本数。
5. 锁定最小位移后，第二阶段最小化既有 archive-size/source-complexity/Python-file-count 分位桶的 L1 漂移。
6. 锁定前两阶段最优值后，以 `SHA256(20260722|component_root|split)` 的 24-bit 系数执行固定、确定性的 tie-break。

整个求解过程不读取模型预测、Cerebro 文本、VirusTotal 结果或 test 指标。

## 不可同时满足的旧来源比例

strict40 后，大于 50 的强制训练组件已经包含 6,725 个 Guo 恶意样本，高于 V1 train 的 6,449。因此，“大组件不进 holdout”与“旧 split×恶意来源计数完全不变”数学上不可同时满足。

V2 保持 split×label 精确不变，并固定可行且对称的来源目标：train 的 Datadog/Guo=1,396/6,725，val=362/693，test=363/696。该目标由组件约束和剩余来源按 holdout 容量对称分摊得到，不使用模型结果。

## 冻结结果

- strict40 正式组件：{EXPECTED_FORMAL_COMPONENTS:,}
- 相对 V1 新出现的跨 split 组件：{EXPECTED_CROSS_V1_COMPONENTS} 个，涉及 {EXPECTED_CROSS_V1_ROWS:,} 行
- 最小且最终 split 变更：{minimum_migration:,} 行（{minimum_migration / EXPECTED_FORMAL_ROWS:.2%}）
- 第二阶段 balance-bin L1：{balance_l1:,}
- 旧 test 保留：{test_old_kept:,}
- test 换出/换入：{test_out:,}/{test_in:,}

精确 transition、分布、组件与泄漏审计见本目录 CSV；规则、求解器版本及输入/输出 SHA 见 `split_rule_manifest.json`。

## 复用边界

- V10c：20,878 个既有 graph/PT 全部复用，只生成 V2 索引并重新训练。
- Cerebro：20,878 个包级特征/PyCG/文本缓存全部复用；后续必须用 metadata-rehydrating reaggregate 生成 V2 train/val/test 文本，不能直接运行旧 `--aggregate-only`。
- VirusTotal：analysis 由 archive SHA 内容寻址；旧结果可复用，只需处理 V2 新进入 test 且没有 completed analysis 的 SHA。不得覆盖 V1 manifest/results。

## 关键输出 SHA-256

```json
{json.dumps(output_hashes, ensure_ascii=False, indent=2)}
```
"""


def write_outputs(
    out_dir: Path,
    frame: pd.DataFrame,
    full_strict: pd.DataFrame,
    old_index: pd.DataFrame,
    profiles: pd.DataFrame,
    stage1_result,
    stage2_result,
    stage3_result,
    minimum_migration: int,
    balance_l1: int,
    target_keys: list[tuple],
    v10c_index: Path,
) -> tuple[Path, dict[str, str]]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing v2 output: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}.building_", dir=out_dir.parent) as tmp_name:
        tmp = Path(tmp_name)
        all_csv = tmp / "filtered700k_graph_eligible_v2_all.csv"
        frame.to_csv(all_csv, index=False, encoding="utf-8")
        for split in SPLITS:
            frame[frame["split"] == split].to_csv(
                tmp / f"filtered700k_graph_eligible_v2_{split}.csv",
                index=False,
                encoding="utf-8",
            )

        components = component_summary(frame)
        components.to_csv(tmp / "source_clone_component_summary.csv", index=False, encoding="utf-8")
        distribution = distribution_summary(frame)
        distribution.to_csv(tmp / "split_distribution_summary.csv", index=False, encoding="utf-8")
        transition = (
            frame.groupby(["v1_split", "split"]).size().reset_index(name="samples")
            .sort_values(["v1_split", "split"])
        )
        transition.to_csv(tmp / "v1_to_v2_transition_summary.csv", index=False, encoding="utf-8")
        frame[
            [
                "graph_id",
                "sample_id",
                "label",
                "source_group",
                "archive_sha256",
                "v1_split",
                "split",
                "split_changed",
                "source_clone_component_id_v1",
                "source_clone_component_id",
            ]
        ].to_csv(tmp / "v1_to_v2_transition.csv", index=False, encoding="utf-8")
        frame[frame["split_changed"]].to_csv(
            tmp / "v1_to_v2_changed_samples.csv",
            index=False,
            encoding="utf-8",
        )

        cross_ids = set(
            profiles.loc[profiles["v1_split_count"] > 1, "source_clone_component_id"].astype(str)
        )
        components[components["source_clone_component_id"].isin(cross_ids)].to_csv(
            tmp / "cross_v1_split_components_audit.csv",
            index=False,
            encoding="utf-8",
        )

        formal_ids = set(frame["graph_id"].astype(str))
        excluded = full_strict[~full_strict["graph_id"].astype(str).isin(formal_ids)].copy()
        excluded.to_csv(tmp / "historical_v1_only_exclusions_audit.csv", index=False, encoding="utf-8")

        balance = balance_bin_audit(frame)
        balance.to_csv(tmp / "balance_bin_l1_audit.csv", index=False, encoding="utf-8")
        observed_balance_l1 = int(balance["absolute_difference"].sum())
        if observed_balance_l1 != balance_l1:
            raise RuntimeError(
                f"balance-bin L1 mismatch: solver={balance_l1} observed={observed_balance_l1}"
            )
        audit = leakage_audit(frame, old_index, minimum_migration)
        audit.to_csv(tmp / "split_leakage_audit.csv", index=False, encoding="utf-8")
        if not audit["passed"].all():
            raise RuntimeError(f"v2 leakage audit failed:\n{audit.to_string(index=False)}")

        component_version = pd.DataFrame(
            [
                {
                    "version": "v1_abstract80",
                    "components": old_index["source_clone_component_id"].nunique(),
                    "multi_sample_components": int(
                        (old_index.groupby("source_clone_component_id").size() > 1).sum()
                    ),
                    "largest_component": int(old_index.groupby("source_clone_component_id").size().max()),
                    "components_over_50": int(
                        (old_index.groupby("source_clone_component_id").size() > 50).sum()
                    ),
                    "rows_in_components_over_50": int(
                        old_index[old_index["source_clone_component_id"].map(
                            old_index.groupby("source_clone_component_id").size()
                        ) > 50].shape[0]
                    ),
                },
                {
                    "version": "v2_abstract40",
                    "components": frame["source_clone_component_id"].nunique(),
                    "multi_sample_components": int(
                        (frame.groupby("source_clone_component_id").size() > 1).sum()
                    ),
                    "largest_component": int(frame.groupby("source_clone_component_id").size().max()),
                    "components_over_50": int(
                        (frame.groupby("source_clone_component_id").size() > 50).sum()
                    ),
                    "rows_in_components_over_50": int(
                        frame[frame["source_clone_component_id"].map(
                            frame.groupby("source_clone_component_id").size()
                        ) > 50].shape[0]
                    ),
                },
            ]
        )
        component_version.to_csv(tmp / "component_version_comparison.csv", index=False, encoding="utf-8")

        csv_paths = sorted(tmp.glob("*.csv"))
        output_hashes = {path.name: sha256_file(path) for path in csv_paths}
        formal_ids = set(frame["graph_id"].astype(str))
        full_component_sizes = full_strict.groupby("source_clone_component_id").size()
        formal_component_sizes = frame.groupby("source_clone_component_id").size()
        threshold_crossings = [
            component_id
            for component_id, formal_size in formal_component_sizes.items()
            if (int(full_component_sizes[component_id]) > MAX_HOLDOUT_COMPONENT_SIZE)
            != (int(formal_size) > MAX_HOLDOUT_COMPONENT_SIZE)
        ]
        if threshold_crossings:
            raise RuntimeError(f"full/formal holdout-size threshold crossings: {threshold_crossings}")
        full_mixed = full_strict.groupby("source_clone_component_id")["label"].nunique()
        full_mixed_ids = set(full_mixed[full_mixed > 1].index.astype(str))
        full_mixed_rows = full_strict[full_strict["source_clone_component_id"].isin(full_mixed_ids)]
        if len(full_mixed_ids) != 1 or len(full_mixed_rows) != 2:
            raise RuntimeError(
                f"unexpected full-universe mixed-label strict40 components: "
                f"components={len(full_mixed_ids)} rows={len(full_mixed_rows)}"
            )
        if full_mixed_rows["graph_id"].astype(str).isin(formal_ids).any():
            raise RuntimeError("a full-universe mixed-label strict40 row entered the formal population")
        rule_manifest = {
            "status": "frozen_v2_split_ready_for_shared_retraining_after_user_confirmation",
            "version": "final_filtered700k_graph_eligible_v2_strict40_20260722",
            "created_at_local_date": "2026-07-22",
            "post_hoc_remediation_disclosure": (
                "Created after detecting a v1 source-template family below the old abstract-AST threshold; "
                "all v1 results remain immutable diagnostic artifacts."
            ),
            "seed": SEED,
            "fingerprint_relation_universe_rows": EXPECTED_RELATION_ROWS,
            "formal_assignment_population_rows": EXPECTED_FORMAL_ROWS,
            "historical_exclusions_reused_not_recomputed": EXPECTED_RELATION_ROWS - EXPECTED_FORMAL_ROWS,
            "abstract_ast_min_nodes": ABSTRACT_AST_MIN_NODES,
            "max_holdout_component_size": MAX_HOLDOUT_COMPONENT_SIZE,
            "strict40_components": EXPECTED_FORMAL_COMPONENTS,
            "cross_v1_split_components": EXPECTED_CROSS_V1_COMPONENTS,
            "cross_v1_split_rows": EXPECTED_CROSS_V1_ROWS,
            "full_relation_components": int(full_strict["source_clone_component_id"].nunique()),
            "full_relation_mixed_label_components": len(full_mixed_ids),
            "full_relation_mixed_label_rows": len(full_mixed_rows),
            "full_formal_holdout_threshold_crossings": len(threshold_crossings),
            "minimum_moved_rows": minimum_migration,
            "balance_bin_l1": balance_l1,
            "split_label_targets": [
                {"split": split, "label": label, "samples": count}
                for (split, label), count in sorted(EXPECTED_SPLIT_LABEL_COUNTS.items())
            ],
            "split_label_source_targets": [
                {"split": split, "label": label, "source_group": source, "samples": count}
                for (split, label, source), count in sorted(EXPECTED_SOURCE_TARGETS.items())
            ],
            "component_relations": [
                "same normalized package group",
                "same archive SHA-256",
                "same complete raw Python corpus",
                "same complete lexical Python corpus with at least 50 tokens",
                "same complete concrete AST corpus with at least 40 AST nodes",
                "same complete abstract AST corpus with at least 40 AST nodes",
            ],
            "assignment_lexicographic_objectives": [
                "minimum moved rows from v1",
                "minimum L1 drift in pre-existing source_size/source_complexity/python_file_count bins",
                "stable 24-bit SHA256 tie-break objective",
            ],
            "does_not_read_or_use": [
                "model predictions or scores",
                "Cerebro behavior text",
                "VirusTotal results",
                "test metrics",
                "V10c graph structure or Event/Param features",
            ],
            "solver": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "scipy_version": scipy.__version__,
                "backend": "HiGHS via scipy.optimize.milp",
                "mip_rel_gap": 0.0,
                "stage1_status": int(stage1_result.status),
                "stage1_message": str(stage1_result.message),
                "stage1_objective": float(stage1_result.fun),
                "stage2_status": int(stage2_result.status),
                "stage2_message": str(stage2_result.message),
                "stage2_objective": float(stage2_result.fun),
                "stage3_status": int(stage3_result.status),
                "stage3_message": str(stage3_result.message),
                "stage3_objective": float(stage3_result.fun),
                "balance_target_count": len(target_keys),
            },
            "inputs": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    V1_ALL_CSV,
                    V1_FINGERPRINT_CSV,
                    V1_FORMAL_INDEX,
                    THRESHOLD_EVIDENCE_CSV,
                    Path(__file__).resolve(),
                    Path(component_base.__file__).resolve(),
                )
            },
            "derived_v10c_index": {
                "path": str(v10c_index),
                "expected_sha256": output_hashes["filtered700k_graph_eligible_v2_all.csv"],
            },
            "output_csv_sha256": output_hashes,
        }
        with (tmp / "split_rule_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(rule_manifest, handle, ensure_ascii=False, indent=2)
        readme = build_readme(
            minimum_migration,
            balance_l1,
            transition,
            output_hashes,
            out_dir,
            v10c_index,
        )
        (tmp / "README_split_rules_zh.md").write_text(readme, encoding="utf-8")

        os.replace(tmp, out_dir)

    canonical_index = out_dir / "filtered700k_graph_eligible_v2_all.csv"
    return canonical_index, output_hashes


def publish_v10c_index(canonical_index: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing V10c v2 index: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(canonical_index, temporary)
    os.replace(temporary, destination)
    source_hash = sha256_file(canonical_index)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError("published V10c v2 index hash mismatch")
    return destination_hash


def recover_index_publication(out_dir: Path, destination: Path) -> bool:
    """Finish the second half of a previously verified split publication."""
    if not out_dir.exists():
        return False
    canonical = out_dir / "filtered700k_graph_eligible_v2_all.csv"
    rules_path = out_dir / "split_rule_manifest.json"
    if not canonical.exists() or not rules_path.exists():
        raise RuntimeError(f"incomplete existing v2 directory cannot be recovered safely: {out_dir}")
    with rules_path.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    expected = rules.get("output_csv_sha256", {}).get(canonical.name, "")
    observed = sha256_file(canonical)
    if not expected or expected != observed:
        raise RuntimeError("existing canonical v2 index failed its recorded SHA-256")
    if destination.exists():
        raise FileExistsError(
            f"both v2 split and V10c index already exist; refusing to overwrite: {out_dir}, {destination}"
        )
    published = publish_v10c_index(canonical, destination)
    print(f"[RECOVERED] published verified V10c index: {destination}")
    print(f"[RECOVERED] SHA256: {published}")
    return True


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    v10c_index = Path(args.v10c_index)
    if recover_index_publication(out_dir, v10c_index):
        return
    if v10c_index.exists():
        raise FileExistsError(
            f"V10c v2 index exists while the split directory does not; refusing to overwrite: {v10c_index}"
        )

    full, _, old_index = validate_inputs()
    print(f"[INPUT] v1 relation universe: {len(full):,}")
    print(f"[INPUT] frozen formal population: {len(old_index):,}")

    relation = full.copy()
    relation["v1_split"] = relation["split"].astype(str)
    relation = relation.rename(
        columns={
            "source_clone_component_id": "source_clone_component_id_v1",
            "source_clone_component_root": "source_clone_component_root_v1",
        }
    )
    full_strict = build_strict40_components(relation)
    full_mixed = full_strict.groupby("source_clone_component_id")["label"].nunique()
    full_mixed_ids = set(full_mixed[full_mixed > 1].index.astype(str))
    full_mixed_rows = full_strict[full_strict["source_clone_component_id"].isin(full_mixed_ids)]
    formal_id_set = set(old_index["graph_id"].astype(str))
    if len(full_mixed_ids) != 1 or len(full_mixed_rows) != 2:
        raise RuntimeError(
            f"unexpected full strict40 mixed-label audit: components={len(full_mixed_ids)} "
            f"rows={len(full_mixed_rows)}"
        )
    if full_mixed_rows["graph_id"].astype(str).isin(formal_id_set).any():
        raise RuntimeError("full strict40 mixed-label component intersects the formal population")

    # The graph/PT training index intentionally omitted the raw archive byte
    # count.  Rehydrate it from the immutable v1 source manifest for the
    # distribution audit; it is descriptive only and is not a model input.
    strict_map = full_strict[
        [
            "graph_id",
            "source_clone_component_id",
            "source_clone_component_root",
            "source_size",
        ]
    ].copy()
    frame = old_index.copy()
    frame["v1_split"] = frame["split"].astype(str)
    frame["source_clone_component_id_v1"] = frame["source_clone_component_id"].astype(str)
    frame["source_clone_component_root_v1"] = frame["source_clone_component_root"].astype(str)
    frame = frame.drop(columns=["source_clone_component_id", "source_clone_component_root"])
    frame = frame.merge(strict_map, on="graph_id", how="left", validate="one_to_one")
    if frame["source_clone_component_id"].isna().any():
        raise RuntimeError("strict40 component mapping is incomplete")

    profiles, parts = component_profiles(frame)
    cross = profiles[profiles["v1_split_count"] > 1]
    if len(profiles) != EXPECTED_FORMAL_COMPONENTS:
        raise RuntimeError(f"unexpected strict40 component count: {len(profiles)}")
    if len(cross) != EXPECTED_CROSS_V1_COMPONENTS or int(cross["samples"].sum()) != EXPECTED_CROSS_V1_ROWS:
        raise RuntimeError(
            f"unexpected cross-v1 components: components={len(cross)} rows={int(cross['samples'].sum())}"
        )
    print(
        f"[STRICT40] components={len(profiles):,} cross_v1={len(cross)} "
        f"cross_rows={int(cross['samples'].sum()):,}"
    )

    hard_matrix, hard_lower, hard_upper, hard_bounds, migration = build_hard_constraints(profiles, parts)
    nx = len(profiles) * len(SPLITS)
    stage1 = solve_milp(
        migration,
        np.ones(nx),
        hard_bounds,
        hard_matrix,
        hard_lower,
        hard_upper,
        "stage 1 / minimum migration",
    )
    minimum_migration = int(round(float(stage1.fun)))
    if abs(float(stage1.fun) - minimum_migration) > 1e-6:
        raise RuntimeError(f"stage1 objective is not integral: {stage1.fun}")
    if minimum_migration != EXPECTED_MINIMUM_MIGRATION:
        raise RuntimeError(f"unexpected minimum migration: {minimum_migration}")

    (
        stage2_matrix,
        stage2_lower,
        stage2_upper,
        stage2_bounds,
        stage2_integrality,
        stage2_objective,
        target_keys,
    ) = build_balance_constraints(
        frame,
        profiles,
        parts,
        hard_matrix,
        hard_lower,
        hard_upper,
        migration,
        minimum_migration,
    )
    stage2 = solve_milp(
        stage2_objective,
        stage2_integrality,
        stage2_bounds,
        stage2_matrix,
        stage2_lower,
        stage2_upper,
        "stage 2 / minimum balance-bin L1",
    )
    balance_l1 = int(round(float(stage2.fun)))
    if abs(float(stage2.fun) - balance_l1) > 1e-6:
        raise RuntimeError(f"stage2 objective is not integral: {stage2.fun}")

    balance_row = csr_matrix(stage2_objective[None, :])
    stage3_matrix = vstack([stage2_matrix, balance_row], format="csr")
    stage3_lower = np.concatenate([stage2_lower, [balance_l1]])
    stage3_upper = np.concatenate([stage2_upper, [balance_l1]])
    stage3_objective = stable_objective(profiles, len(stage2_objective))
    stage3 = solve_milp(
        stage3_objective,
        stage2_integrality,
        stage2_bounds,
        stage3_matrix,
        stage3_lower,
        stage3_upper,
        "stage 3 / stable SHA-256 tie break",
    )

    assignments = extract_assignments(stage3.x, profiles)
    frame["split"] = frame["source_clone_component_id"].map(assignments)
    frame["split_changed"] = frame["split"] != frame["v1_split"]
    frame["split_transition"] = frame["v1_split"] + "->" + frame["split"]
    component_sizes = frame.groupby("source_clone_component_id").size()
    frame["oversized_holdout_excluded"] = (
        frame["source_clone_component_id"].map(component_sizes) > MAX_HOLDOUT_COMPONENT_SIZE
    )
    frame["split_version"] = "final_filtered700k_graph_eligible_v2_strict40_20260722"
    frame = frame.sort_values(
        ["split", "label", "source_group", "source_complexity_bin", "graph_id"]
    ).reset_index(drop=True)

    observed_migration = int(frame["split_changed"].sum())
    if observed_migration != minimum_migration:
        raise RuntimeError(
            f"assignment migration mismatch: solver={minimum_migration} observed={observed_migration}"
        )
    audit = leakage_audit(frame, old_index, observed_migration)
    if not audit["passed"].all():
        raise RuntimeError(f"pre-write leakage audit failed:\n{audit.to_string(index=False)}")

    canonical_index, _ = write_outputs(
        out_dir,
        frame,
        full_strict,
        old_index,
        profiles,
        stage1,
        stage2,
        stage3,
        minimum_migration,
        balance_l1,
        target_keys,
        v10c_index,
    )
    destination_hash = publish_v10c_index(canonical_index, v10c_index)

    transition = frame.groupby(["v1_split", "split"]).size().unstack(fill_value=0)
    print("=" * 100)
    print("FINAL GRAPH-ELIGIBLE V2 STRICT40 SPLIT")
    print("=" * 100)
    print(frame.groupby(["split", "label", "source_group"]).size().to_string())
    print("\nV1 -> V2 TRANSITION")
    print(transition.to_string())
    print("\nLEAKAGE AUDIT")
    print(audit.to_string(index=False))
    print(f"\n[RESULT] moved rows: {observed_migration:,}")
    print(f"[RESULT] balance-bin L1: {balance_l1:,}")
    print(f"[RESULT] split directory: {args.output_dir}")
    print(f"[RESULT] V10c index: {args.v10c_index}")
    print(f"[RESULT] V10c index SHA256: {destination_hash}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--v10c-index", default=str(DEFAULT_V10C_INDEX))
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
