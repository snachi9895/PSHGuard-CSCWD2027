#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import math
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v9_payload_aware_bounded_small_20260608"
TRAIN_ROOT = ROOT / "experiment_outputs/train_final_v9_20260608"
IN_CSV = GRAPH_ROOT / "step02_export_summary_v9_small.csv"
OUT_PT_DIR = TRAIN_ROOT / "heterodata_pt_v9_small"
OUT_INDEX_CSV = TRAIN_ROOT / "step04_training_pt_index_v9_small.csv"
NORMALIZED_FEATURES = False

FEATURE_KEYS = {
    "module": ["is_python_file", "path_depth", "is_setup_py", "is_init_py"],
    "ast": ["relative_position", "is_call", "is_import", "is_function", "is_assign", "is_return"],
    "param": [
        "length", "entropy", "has_url", "has_ip", "has_base64_like_pattern", "has_powershell",
        "has_cmd", "has_path", "payload_like", "call_is_decoder", "call_is_network",
        "call_is_filewrite", "call_is_process", "call_is_exec_sink", "source_literal",
        "source_fstring", "source_concat_like", "source_unknown",
    ],
    "event": [
        "is_info_read", "is_decoder", "is_network", "is_filewrite", "is_process", "is_exec_sink",
        "phase_install", "phase_import", "phase_runtime", "has_payload_arg", "has_assignment", "arg_count",
    ],
}


def feature_value(node_type: str, name: str, raw_value) -> float:
    value = float(raw_value)
    if not NORMALIZED_FEATURES:
        return value
    if node_type == "module" and name == "path_depth":
        return min(max(value, 0.0), 16.0) / 16.0
    if node_type == "param" and name == "length":
        return math.log1p(min(max(value, 0.0), 1_000_000.0)) / math.log1p(1_000_000.0)
    if node_type == "param" and name == "entropy":
        return min(max(value, 0.0), 8.0) / 8.0
    if node_type == "event" and name == "arg_count":
        return min(max(value, 0.0), 16.0) / 16.0
    return value


def build_one(graph_path: Path, out_path: Path) -> dict:
    with open(graph_path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)

    data = HeteroData()
    nodes_by_type = {
        node_type: [node for node in graph.get("nodes", []) if node.get("type") == node_type]
        for node_type in FEATURE_KEYS
    }
    indexes = {
        node_type: {node["id"]: index for index, node in enumerate(nodes)}
        for node_type, nodes in nodes_by_type.items()
    }
    id_to_type = {
        node_id: node_type
        for node_type, mapping in indexes.items()
        for node_id in mapping
    }

    for node_type, feature_names in FEATURE_KEYS.items():
        rows = [
            [
                feature_value(node_type, name, node.get("features", {}).get(name, 0.0))
                for name in feature_names
            ]
            for node in nodes_by_type[node_type]
        ]
        data[node_type].x = (
            torch.tensor(rows, dtype=torch.float32)
            if rows else torch.zeros((0, len(feature_names)), dtype=torch.float32)
        )

    buckets: dict[tuple[str, str, str], list[list[int]]] = {}
    for edge in graph.get("edges", []):
        src_type = id_to_type.get(edge["src"])
        dst_type = id_to_type.get(edge["dst"])
        if not src_type or not dst_type:
            continue
        edge_type = (src_type, edge["type"], dst_type)
        buckets.setdefault(edge_type, [[], []])
        buckets[edge_type][0].append(indexes[src_type][edge["src"]])
        buckets[edge_type][1].append(indexes[dst_type][edge["dst"]])

    for edge_type, (srcs, dsts) in buckets.items():
        data[edge_type].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".pt.tmp")
    torch.save(data, temp_path)
    temp_path.replace(out_path)
    return {
        "pt_path_v9": str(out_path),
        "pt_path_v8": str(out_path),
        **{f"{node_type}_count": len(nodes) for node_type, nodes in nodes_by_type.items()},
        "edge_type_count": len(buckets),
        "status": "ok",
    }


def main():
    exported = pd.read_csv(IN_CSV, low_memory=False)
    exported = exported[exported["status"].isin(["ok", "skipped_existing"])].copy()
    OUT_PT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in tqdm(exported.iterrows(), total=len(exported), unit="graph", dynamic_ncols=True):
        graph_path = Path(str(row["graph_json_path_v9"]))
        out_path = OUT_PT_DIR / f"{row['graph_id']}.pt"
        info = {
            "graph_id": row.get("graph_id", ""),
            "sample_id": row.get("sample_id", row.get("graph_id", "")),
            "label": row.get("label", ""),
            "label_int": row.get("label_int", ""),
            "split": row.get("split", ""),
            "package_name": row.get("package_name", ""),
            "version": row.get("version", ""),
            "source_group": row.get("source_group", ""),
            "group_key": row.get("group_key", ""),
            "graph_json_path_v9": str(graph_path),
            "graph_meta_path_v9": row.get("graph_meta_path_v9", ""),
            "feature_profile": "normalized_bounded_v1" if NORMALIZED_FEATURES else "raw_v9",
        }
        try:
            info.update(build_one(graph_path, out_path))
        except Exception as exc:
            info.update({
                "pt_path_v9": str(out_path),
                "pt_path_v8": str(out_path),
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
        rows.append(info)

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_INDEX_CSV, index=False, encoding="utf-8")
    print(frame["status"].value_counts(dropna=False).to_string())
    print(f"[RESULT] {OUT_INDEX_CSV}")


if __name__ == "__main__":
    main()
