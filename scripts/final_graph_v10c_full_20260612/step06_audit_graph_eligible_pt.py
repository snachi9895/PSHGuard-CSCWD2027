#!/usr/bin/env python3
"""Audit every final graph-eligible PyG PT file and a representative batch."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v10c_semantic_budget_filtered700k_20260612"
TRAIN_ROOT = ROOT / "experiment_outputs/train_final_v10c_filtered700k_20260612"
INDEX_CSV = TRAIN_ROOT / "step05_training_pt_index_v10c_source_clone_filtered700k_graph_eligible.csv"
PT_DIR = TRAIN_ROOT / "heterodata_pt_v10c_source_clone_filtered700k_graph_eligible_normalized"
EXCLUSIONS_CSV = GRAPH_ROOT / "step05_zero_ast_graph_exclusions_5.csv"
OUT_AUDIT = TRAIN_ROOT / "step06_graph_eligible_pt_integrity_audit.json"
OUT_FAILURES = TRAIN_ROOT / "step06_graph_eligible_pt_integrity_failures.csv"

EXPECTED_ROWS = 20878
EXPECTED_NODE_DIMS = {"module": 4, "ast": 12, "param": 29, "event": 18}


def inspect_one(row: pd.Series) -> tuple[dict, object | None]:
    graph_id = str(row["graph_id"])
    path = Path(str(row["pt_path_v9"]))
    result = {"graph_id": graph_id, "status": "ok", "error_message": ""}
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        if set(data.node_types) != set(EXPECTED_NODE_DIMS):
            raise RuntimeError(f"unexpected_node_types={data.node_types}")
        for node_type, expected_dim in EXPECTED_NODE_DIMS.items():
            x = data[node_type].x
            if x.ndim != 2 or x.shape[1] != expected_dim:
                raise RuntimeError(
                    f"{node_type}_shape={tuple(x.shape)} expected_second_dim={expected_dim}"
                )
            if not torch.isfinite(x).all():
                raise RuntimeError(f"{node_type}_contains_non_finite_features")
        if data["ast"].x.shape[0] <= 0:
            raise RuntimeError("ast_count_must_be_greater_than_zero")
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise RuntimeError(f"invalid_edge_index_shape={edge_type}:{tuple(edge_index.shape)}")
            if edge_index.numel() == 0:
                continue
            src_type, _, dst_type = edge_type
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= data[src_type].x.shape[0]:
                raise RuntimeError(f"source_edge_index_out_of_bounds={edge_type}")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= data[dst_type].x.shape[0]:
                raise RuntimeError(f"target_edge_index_out_of_bounds={edge_type}")
        result.update(
            {
                "node_types": len(data.node_types),
                "edge_types": len(data.edge_types),
                **{f"{node_type}_count": int(data[node_type].x.shape[0]) for node_type in EXPECTED_NODE_DIMS},
            }
        )
        return result, data
    except Exception as exc:
        result["status"] = "failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        return result, None


def choose_batch_rows(index: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in index.groupby(["split", "label", "source_group"], dropna=False):
        parts.append(group.sort_values("graph_id").head(4))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    index = pd.read_csv(INDEX_CSV, low_memory=False)
    exclusions = pd.read_csv(EXCLUSIONS_CSV, low_memory=False)
    if len(index) != EXPECTED_ROWS or index["graph_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} unique rows, got {len(index)}")
    excluded_ids = set(exclusions["graph_id"].astype(str))
    overlap = excluded_ids & set(index["graph_id"].astype(str))
    if overlap:
        raise RuntimeError(f"excluded graph IDs remain in PT index: {sorted(overlap)}")

    rows = []
    edge_type_counts: Counter[str] = Counter()
    for _, row in tqdm(index.iterrows(), total=len(index), unit="pt", dynamic_ncols=True):
        result, data = inspect_one(row)
        rows.append(result)
        if data is not None:
            for edge_type in data.edge_types:
                edge_type_counts[str(edge_type)] += 1

    detail = pd.DataFrame(rows)
    failures = detail[detail["status"] != "ok"].copy()
    failures.to_csv(OUT_FAILURES, index=False, encoding="utf-8")
    if not failures.empty:
        raise RuntimeError(f"PT integrity failures: {len(failures)}; see {OUT_FAILURES}")

    batch_rows = choose_batch_rows(index)
    batch_data = [
        torch.load(Path(str(path)), map_location="cpu", weights_only=False)
        for path in batch_rows["pt_path_v9"]
    ]
    batches = list(DataLoader(batch_data, batch_size=8, shuffle=False))
    for batch in batches:
        for node_type, expected_dim in EXPECTED_NODE_DIMS.items():
            if batch[node_type].x.ndim != 2 or batch[node_type].x.shape[1] != expected_dim:
                raise RuntimeError(f"batched {node_type} feature shape invalid")

    pt_files = list(PT_DIR.glob("*.pt"))
    audit = {
        "status": "passed",
        "index_rows": len(index),
        "index_unique_graph_ids": int(index["graph_id"].nunique()),
        "pt_files_in_directory": len(pt_files),
        "excluded_graph_ids_in_index": sorted(overlap),
        "integrity_failures": len(failures),
        "expected_node_feature_dims": EXPECTED_NODE_DIMS,
        "node_count_summary": {
            node_type: {
                "min": int(detail[f"{node_type}_count"].min()),
                "median": float(detail[f"{node_type}_count"].median()),
                "max": int(detail[f"{node_type}_count"].max()),
            }
            for node_type in EXPECTED_NODE_DIMS
        },
        "edge_type_sample_coverage": dict(sorted(edge_type_counts.items())),
        "representative_batch_samples": len(batch_rows),
        "representative_batches": len(batches),
        "split_label_distribution": (
            index.groupby(["split", "label"]).size().rename("samples").reset_index().to_dict("records")
        ),
        "feature_profile_values": sorted(index["feature_profile"].astype(str).unique()),
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[RESULT] {OUT_AUDIT}")
    print(f"[RESULT] {OUT_FAILURES}")


if __name__ == "__main__":
    main()
