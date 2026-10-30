#!/usr/bin/env python3
"""Diagnostics for the V10 event-centered 2,000-sample graphs."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v10b_event_centered_medium_2000_20260610"
IN_CSV = GRAPH_ROOT / "step03_v10_export_summary.csv"
DETAIL_CSV = GRAPH_ROOT / "step03_v10_sample_metrics.csv"
LABEL_CSV = GRAPH_ROOT / "step03_v10_summary_by_label.csv"
SOURCE_CSV = GRAPH_ROOT / "step03_v10_summary_by_source.csv"
EDGE_CSV = GRAPH_ROOT / "step03_v10_edge_coverage_by_label.csv"

ROLE_NAMES = (
    "event_anchor", "payload_anchor", "event_ancestor",
    "event_nearby", "payload_context", "neutral_fallback",
)


def inspect_row(row: pd.Series) -> dict:
    with open(Path(str(row["graph_json_path_v9"])), "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})
    node_counts = Counter(node.get("type", "") for node in nodes)
    edge_counts = Counter(edge.get("type", "") for edge in edges)
    module_ids = {node["id"] for node in nodes if node.get("type") == "module"}
    linked_modules = {
        edge["src"]
        for edge in edges
        if edge.get("type") in {"E_AST", "E_ModuleEvent"}
    }
    roles = Counter()
    for node in nodes:
        if node.get("type") != "ast":
            continue
        features = node.get("features", {})
        for role in ROLE_NAMES:
            roles[role] += int(features.get(f"context_role_{role}", 0) > 0)
    event_anchors = roles["event_anchor"]
    event_count = node_counts["event"]
    return {
        "graph_id": row["graph_id"],
        "label": row["label"],
        "split": row["split"],
        "source_group": row["source_group"],
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "module_count": node_counts["module"],
        "empty_module_count": len(module_ids - linked_modules),
        "ast_count": node_counts["ast"],
        "event_count": event_count,
        "param_count": node_counts["param"],
        "original_py_files": meta.get("original_py_files", 0),
        "parsed_py_files": meta.get("parsed_py_files", 0),
        "kept_py_files": meta.get("kept_py_files", 0),
        "event_anchor_coverage": min(event_anchors, event_count) / event_count if event_count else 1.0,
        "has_event": int(event_count > 0),
        "has_param": int(node_counts["param"] > 0),
        "truncated_files": int(bool(meta.get("truncated_files", False))),
        "truncated_ast": int(bool(meta.get("truncated_ast", False))),
        "truncated_param": int(bool(meta.get("truncated_param", False))),
        "truncated_events": int(bool(meta.get("truncated_events", False))),
        **{f"role__{name}": roles[name] for name in ROLE_NAMES},
        **{f"edge__{name}": count for name, count in edge_counts.items()},
    }


def grouped_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    metrics = [
        "num_nodes", "num_edges", "module_count", "empty_module_count", "ast_count", "event_count", "param_count",
        "original_py_files", "parsed_py_files", "kept_py_files", "event_anchor_coverage",
        "has_event", "has_param", "truncated_files", "truncated_ast", "truncated_param",
        "truncated_events", *[f"role__{name}" for name in ROLE_NAMES],
    ]
    rows = []
    for name, part in frame.groupby(group, dropna=False):
        item = {group: name, "samples": len(part)}
        for metric in metrics:
            item[f"{metric}_mean"] = part[metric].mean()
            item[f"{metric}_median"] = part[metric].median()
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    exported = pd.read_csv(IN_CSV, low_memory=False)
    ok = exported[exported["status"].isin(["ok", "skipped_existing"])].copy()
    details = pd.DataFrame(
        [
            inspect_row(row)
            for _, row in tqdm(ok.iterrows(), total=len(ok), unit="graph", dynamic_ncols=True)
        ]
    )
    details.to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    by_label = grouped_summary(details, "label")
    by_label.to_csv(LABEL_CSV, index=False, encoding="utf-8")
    grouped_summary(details, "source_group").to_csv(SOURCE_CSV, index=False, encoding="utf-8")

    edge_rows = []
    edge_cols = [col for col in details.columns if col.startswith("edge__") and "REV_" not in col]
    for label, part in details.groupby("label"):
        for col in edge_cols:
            values = part[col].fillna(0)
            edge_rows.append(
                {
                    "label": label,
                    "edge_type": col.removeprefix("edge__"),
                    "coverage": (values > 0).mean(),
                    "mean_count": values.mean(),
                    "median_count": values.median(),
                }
            )
    pd.DataFrame(edge_rows).to_csv(EDGE_CSV, index=False, encoding="utf-8")

    print("=" * 100)
    print("V10 EVENT-CENTERED GRAPH DIAGNOSTICS")
    print("=" * 100)
    print(by_label.to_string(index=False))
    print(f"[RESULT] {DETAIL_CSV}")


if __name__ == "__main__":
    main()
