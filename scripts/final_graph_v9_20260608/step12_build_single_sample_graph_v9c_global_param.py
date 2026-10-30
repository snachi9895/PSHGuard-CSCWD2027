#!/usr/bin/env python3
"""V9c: package-global Param anchors with reuse and connection context."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import step01_build_single_sample_graph_v9 as v9
from step01b_build_single_sample_graph_v9b_strict_payload import (
    StrictPayloadAwareBoundedBehaviorGraphBuilderV9B,
)


GRAPH_VERSION = "v9c_global_param_anchor_20260609"


class GlobalParamAnchorGraphBuilderV9C(StrictPayloadAwareBoundedBehaviorGraphBuilderV9B):
    """Deduplicate payload anchors package-wide and encode their reuse context."""

    def __init__(self, src_dir: Path):
        super().__init__(src_dir)
        self.global_param_nodes: dict[str, str] = {}
        self.param_usage: dict[str, dict[str, set | int]] = {}

    def _register_usage(
        self,
        param_id: str,
        rel_file: str,
        lineno: int,
        source_kind: str,
        category: str,
    ) -> None:
        usage = self.param_usage.setdefault(
            param_id,
            {
                "uses": 0,
                "files": set(),
                "callsites": set(),
                "categories": set(),
                "source_kinds": set(),
            },
        )
        usage["uses"] += 1
        usage["files"].add(rel_file)
        usage["callsites"].add((rel_file, int(lineno)))
        usage["categories"].add(category)
        usage["source_kinds"].add(source_kind)

    def _make_param_node(
        self,
        rel_file: str,
        lineno: int,
        text: str,
        source_kind: str,
        category: str,
    ) -> str:
        text = v9.v7.clean_text_for_json(text)
        if not self._should_keep_param(text, category):
            return ""

        occurrence_key = (rel_file, int(lineno), text)
        existing = self.global_param_nodes.get(text, "")
        if existing:
            self.param_nodes[occurrence_key] = existing
            self._register_usage(existing, rel_file, lineno, source_kind, category)
            return existing

        if len(self.global_param_nodes) >= v9.v7.MAX_PARAM_NODES_TOTAL:
            self.truncated_param = True
            return ""

        node_id = self._new_node_id("param")
        self._add_node(
            {
                "id": node_id,
                "type": "param",
                "text": text[:512],
                "file": v9.v7.clean_text_for_json(rel_file),
                "lineno": int(lineno),
                "features": self._build_param_features(text, source_kind, category),
            }
        )
        self.global_param_nodes[text] = node_id
        self.param_nodes[occurrence_key] = node_id
        self._register_usage(node_id, rel_file, lineno, source_kind, category)
        return node_id

    def _add_global_anchor_context(self) -> None:
        ast_links: dict[str, set[str]] = defaultdict(set)
        event_links: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            if edge["type"] == "E_Param":
                ast_links[edge["dst"]].add(edge["src"])
            elif edge["type"] == "E_EventParam":
                event_links[edge["dst"]].add(edge["src"])

        node_by_id = {node["id"]: node for node in self.nodes}
        for param_id, usage in self.param_usage.items():
            node = node_by_id[param_id]
            use_count = int(usage["uses"])
            file_count = len(usage["files"])
            callsite_count = len(usage["callsites"])
            category_count = len(usage["categories"] - {"other"})
            source_kind_count = len(usage["source_kinds"])
            ast_count = len(ast_links[param_id])
            event_count = len(event_links[param_id])
            node["features"].update(
                {
                    "use_count_log_norm": math.log1p(min(use_count, 64)) / math.log1p(64),
                    "callsite_count_log_norm": math.log1p(min(callsite_count, 64)) / math.log1p(64),
                    "file_count_log_norm": math.log1p(min(file_count, 16)) / math.log1p(16),
                    "category_diversity_norm": min(category_count, 6) / 6.0,
                    "source_kind_diversity_norm": min(source_kind_count, 4) / 4.0,
                    "ast_link_count_log_norm": math.log1p(min(ast_count, 64)) / math.log1p(64),
                    "event_link_count_log_norm": math.log1p(min(event_count, 64)) / math.log1p(64),
                    "reused_anchor": float(use_count > 1),
                    "cross_file_anchor": float(file_count > 1),
                    "event_linked_anchor": float(event_count > 0),
                }
            )

    def build(self) -> dict[str, Any]:
        graph = super().build()
        self._add_global_anchor_context()
        self._deduplicate_edges()

        node_counts: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            node_counts[node.get("type", "")] += 1
        graph.update(
            {
                "graph_version": GRAPH_VERSION,
                "nodes": self.nodes,
                "edges": self.edges,
            }
        )
        graph["meta"].update(
            {
                "control_profile": GRAPH_VERSION,
                "param_strategy": "strict_package_global_payload_anchors",
                "param_anchor_identity": "normalized_exact_payload_text_within_package",
                "param_context_features": "reuse_cross_file_category_and_graph_connections",
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "node_counts": dict(sorted(node_counts.items())),
                "edge_counts": self._edge_counts(),
                "unique_global_param_anchors": len(self.global_param_nodes),
                "param_occurrences": sum(int(usage["uses"]) for usage in self.param_usage.values()),
            }
        )
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    graph = GlobalParamAnchorGraphBuilderV9C(args.src_dir).build()
    print(json.dumps(graph["meta"], indent=2))
