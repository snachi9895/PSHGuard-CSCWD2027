#!/usr/bin/env python3
"""Independently re-read and audit the frozen v2 strict40 split artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
V1_INDEX = (
    ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "step05_training_pt_index_v10c_source_clone_filtered700k_graph_eligible.csv"
)
V2_DIR = (
    ROOT
    / "manifests/final_source_clone_aware_split_202606"
    / "final_filtered700k_graph_eligible_v2_strict40_20260722"
)
V2_ALL = V2_DIR / "filtered700k_graph_eligible_v2_all.csv"
V2_INDEX = (
    ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "split_v2_strict40_20260722"
    / "step01_training_pt_index_v10c_v2_strict40_graph_eligible.csv"
)
RULES = V2_DIR / "split_rule_manifest.json"

EXPECTED_SPLIT_LABEL = {
    ("train", "benign"): 8492,
    ("train", "malicious"): 8121,
    ("val", "benign"): 1082,
    ("val", "malicious"): 1055,
    ("test", "benign"): 1069,
    ("test", "malicious"): 1059,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def max_split_count(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> int:
    selected = frame if mask is None else frame[mask]
    selected = selected[selected[column].notna() & (selected[column].astype(str) != "")]
    return int(selected.groupby(column)["split"].nunique().max()) if len(selected) else 0


def main() -> None:
    for path in (V1_INDEX, V2_ALL, V2_INDEX, RULES):
        if not path.exists():
            raise FileNotFoundError(path)
    with RULES.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)

    for name, expected in rules["output_csv_sha256"].items():
        path = V2_DIR / name
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"recorded output hash failed: {path}")
    if sha256_file(V2_ALL) != sha256_file(V2_INDEX):
        raise RuntimeError("canonical v2 manifest and V10c derived index differ")

    old = pd.read_csv(V1_INDEX, low_memory=False).sort_values("graph_id").reset_index(drop=True)
    new = pd.read_csv(V2_ALL, low_memory=False).sort_values("graph_id").reset_index(drop=True)
    checks: dict[str, bool] = {
        "rows_20878": len(old) == len(new) == 20878,
        "graph_ids_unique": old["graph_id"].nunique() == new["graph_id"].nunique() == 20878,
        "sample_ids_unique": old["sample_id"].nunique() == new["sample_id"].nunique() == 20878,
        "graph_id_ordered_set_identical": norm(old["graph_id"]).equals(norm(new["graph_id"])),
    }
    for column in (
        "sample_id",
        "label",
        "label_int",
        "archive_sha256",
        "graph_json_path_v9",
        "graph_meta_path_v9",
        "pt_path_v9",
        "pt_path_v8",
        "module_count",
        "ast_count",
        "param_count",
        "event_count",
        "edge_type_count",
    ):
        checks[f"immutable_{column}"] = norm(old[column]).equals(norm(new[column]))

    checks["split_label_counts"] = new.groupby(["split", "label"]).size().to_dict() == EXPECTED_SPLIT_LABEL
    checks["moved_rows_2142"] = int((new["split"] != old["split"]).sum()) == 2142
    checks["strict_components_11813"] = new["source_clone_component_id"].nunique() == 11813
    checks["component_cross_split_zero"] = max_split_count(new, "source_clone_component_id") == 1
    checks["group_key_cross_split_zero"] = max_split_count(new, "group_key") == 1
    checks["archive_sha_cross_split_zero"] = max_split_count(new, "archive_sha256") == 1

    complete_python = (
        (new["status"] == "ok")
        & (new["python_file_count"] > 0)
        & (~new["truncated_package"].astype(bool))
    )
    complete_ast = (
        complete_python
        & (new["parsed_ast_file_count"] == new["python_file_count"])
        & (new["total_ast_nodes"] >= 40)
    )
    checks["raw_cross_split_zero"] = max_split_count(new, "raw_corpus_digest", complete_python) == 1
    checks["lexical50_cross_split_zero"] = (
        max_split_count(new, "lexical_corpus_digest", complete_python & (new["total_tokens"] >= 50)) == 1
    )
    checks["concrete40_cross_split_zero"] = max_split_count(new, "concrete_ast_corpus_digest", complete_ast) == 1
    checks["abstract40_cross_split_zero"] = max_split_count(new, "abstract_ast_corpus_digest", complete_ast) == 1
    sizes = new.groupby("source_clone_component_id").size()
    checks["holdout_component_max_50"] = int(
        new[new["split"].isin(["val", "test"])]
        .groupby("source_clone_component_id")
        .size()
        .max()
    ) <= 50
    checks["all_large_components_train"] = set(
        new.loc[new["source_clone_component_id"].isin(sizes[sizes > 50].index), "split"]
    ) == {"train"}

    failures = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"checks": checks, "failed": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"independent v2 audit failed: {failures}")

    print("INDEPENDENT V2 AUDIT: PASS")
    print(new.groupby(["split", "label", "source_group"]).size().to_string())
    print("transition")
    print(pd.crosstab(old["split"], new["split"]).to_string())
    print(f"canonical_sha256={sha256_file(V2_ALL)}")


if __name__ == "__main__":
    main()
