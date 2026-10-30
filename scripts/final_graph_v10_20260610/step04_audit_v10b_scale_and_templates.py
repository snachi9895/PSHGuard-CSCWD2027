#!/usr/bin/env python3
"""Audit V10b graph-size shortcuts and repeated semantic-structure templates."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
GRAPH_ROOT = ROOT / "graph_artifacts/final_v10b_event_centered_medium_2000_20260610"
IN_CSV = GRAPH_ROOT / "step03_v10_export_summary.csv"
OUT_DETAIL = GRAPH_ROOT / "step04_v10b_scale_template_audit_detail.csv"
OUT_SOURCE = GRAPH_ROOT / "step04_v10b_scale_summary_by_source.csv"
OUT_TEMPLATES = GRAPH_ROOT / "step04_v10b_repeated_template_summary.csv"
OUT_OVERLAP = GRAPH_ROOT / "step04_v10b_cross_split_template_overlap.csv"
OUT_SHORTCUT = GRAPH_ROOT / "step04_v10b_size_shortcut_baselines.csv"

SIZE_FEATURES = [
    "module_count",
    "ast_count",
    "event_count",
    "param_count",
    "edge_count",
    "parsed_py_files",
    "kept_py_files",
]


def digest(obj) -> str:
    raw = json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def node_descriptor(node: dict) -> tuple:
    features = tuple(
        sorted(
            key
            for key, value in node.get("features", {}).items()
            if isinstance(value, (int, float)) and float(value) > 0
        )
    )
    return (
        node.get("type", ""),
        node.get("sub_type", ""),
        node.get("category", ""),
        node.get("phase", ""),
        features,
    )


def inspect(row: pd.Series) -> dict:
    with open(Path(str(row["graph_json_path_v9"])), "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})
    node_by_id = {node["id"]: node_descriptor(node) for node in nodes}
    node_counts = Counter(node.get("type", "") for node in nodes)
    edge_counts = Counter(edge.get("type", "") for edge in edges)

    count_signature = digest(
        {
            "nodes": sorted(node_counts.items()),
            "edges": sorted(edge_counts.items()),
        }
    )
    # This fingerprint ignores concrete node IDs and literal text. It detects
    # repeated semantic-role and typed-edge patterns, but is not a formal graph
    # isomorphism proof.
    semantic_structure_signature = digest(
        {
            "nodes": sorted(Counter(node_by_id.values()).items(), key=repr),
            "edges": sorted(
                Counter(
                    (
                        node_by_id.get(edge["src"], ("missing",)),
                        edge.get("type", ""),
                        node_by_id.get(edge["dst"], ("missing",)),
                    )
                    for edge in edges
                ).items(),
                key=repr,
            ),
        }
    )
    return {
        "graph_id": row["graph_id"],
        "label": row["label"],
        "label_int": int(row["label_int"]),
        "split": row["split"],
        "source_group": row["source_group"],
        "group_key": row.get("group_key", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "module_count": node_counts["module"],
        "ast_count": node_counts["ast"],
        "event_count": node_counts["event"],
        "param_count": node_counts["param"],
        "parsed_py_files": int(meta.get("parsed_py_files", 0)),
        "kept_py_files": int(meta.get("kept_py_files", 0)),
        "count_signature": count_signature,
        "semantic_structure_signature": semantic_structure_signature,
    }


def evaluate_shortcut(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["split"] == "train"]
    tests = {
        "val": frame[frame["split"] == "val"],
        "test": frame[frame["split"] == "test"],
    }
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[SIZE_FEATURES])
    y_train = train["label_int"]
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=20260610),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260610,
        ),
    }
    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        for split, part in tests.items():
            prediction = model.predict(scaler.transform(part[SIZE_FEATURES]))
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "features": ",".join(SIZE_FEATURES),
                    "accuracy": accuracy_score(part["label_int"], prediction),
                    "precision": precision_score(part["label_int"], prediction, zero_division=0),
                    "recall": recall_score(part["label_int"], prediction, zero_division=0),
                    "f1": f1_score(part["label_int"], prediction, zero_division=0),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    exported = pd.read_csv(IN_CSV, low_memory=False)
    exported = exported[exported["status"].isin(["ok", "skipped_existing"])].copy()
    details = pd.DataFrame(
        [
            inspect(row)
            for _, row in tqdm(
                exported.iterrows(),
                total=len(exported),
                unit="graph",
                dynamic_ncols=True,
                desc="Audit V10b graphs",
            )
        ]
    )
    details.to_csv(OUT_DETAIL, index=False, encoding="utf-8")

    source_summary = (
        details.groupby(["label", "source_group"], dropna=False)[SIZE_FEATURES]
        .agg(["mean", "median", "max"])
        .round(3)
    )
    source_summary.to_csv(OUT_SOURCE, encoding="utf-8")

    templates = (
        details.groupby("semantic_structure_signature", dropna=False)
        .agg(
            samples=("graph_id", "count"),
            labels=("label", lambda values: ",".join(sorted(set(values)))),
            sources=("source_group", lambda values: ",".join(sorted(set(values)))),
            splits=("split", lambda values: ",".join(sorted(set(values)))),
            unique_groups=("group_key", "nunique"),
            example_graph_id=("graph_id", "first"),
        )
        .reset_index()
        .sort_values(["samples", "semantic_structure_signature"], ascending=[False, True])
    )
    templates.to_csv(OUT_TEMPLATES, index=False, encoding="utf-8")

    overlap = templates[
        templates["splits"].str.contains(",")
        | templates["labels"].str.contains(",")
    ].copy()
    overlap.to_csv(OUT_OVERLAP, index=False, encoding="utf-8")

    shortcuts = evaluate_shortcut(details)
    shortcuts.to_csv(OUT_SHORTCUT, index=False, encoding="utf-8")

    repeated = templates[templates["samples"] > 1]
    repeated_samples = int(repeated["samples"].sum())
    print("=" * 100)
    print("V10b SCALE AND TEMPLATE AUDIT")
    print("=" * 100)
    print(source_summary.to_string())
    print("\nSIZE-ONLY SHORTCUT BASELINES")
    print(shortcuts.to_string(index=False))
    print(
        f"\n[RESULT] repeated semantic-structure templates={len(repeated)} "
        f"covering sample occurrences={repeated_samples}/{len(details)}"
    )
    print(f"[RESULT] cross-split or cross-label repeated templates={len(overlap)}")
    print(f"[RESULT] {OUT_DETAIL}")


if __name__ == "__main__":
    main()
