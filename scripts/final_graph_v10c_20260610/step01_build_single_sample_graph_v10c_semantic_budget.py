#!/usr/bin/env python3
"""V10c: package-level semantic budgets for event-centered behavior graphs."""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V10_DIR = HERE.parent / "final_graph_v10_20260610"
V9_DIR = HERE.parent / "final_graph_v9_20260608"
V3_DIR = HERE.parent / "final_dataset_v3"
for directory in (V10_DIR, V9_DIR, V3_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import step01_build_single_sample_graph_v9 as v9
import step02e_build_single_sample_behavior_event_graph_v8b as precise_event_base
from step01_build_single_sample_graph_v10_event_centered import (
    EventCenteredGlobalParamGraphBuilderV10,
    location,
)


GRAPH_VERSION = "v10c_package_semantic_budget_global_param_20260610"


class PackageSemanticBudgetGraphBuilderV10C(EventCenteredGlobalParamGraphBuilderV10):
    """Bound package background and event categories without using labels."""

    max_selected_files = 8
    max_ast_per_file = 72
    max_ast_total = 384
    max_param_total = 96
    max_event_total = 48
    max_event_per_file = 24
    max_context_per_event = 6
    max_context_per_payload_call = 4
    max_neutral_per_file = 4
    max_neutral_total = 16
    context_line_window = 6
    max_events_per_category = 12

    def __init__(self, src_dir: Path):
        super().__init__(src_dir)
        self.selected_event_category_counts: Counter[str] = Counter()
        self.selected_event_total = 0
        self.selected_event_keys: set[tuple[str, int, int]] = set()

    def _precise_event_calls(self, tree: ast.AST) -> set[ast.Call]:
        calls = super()._precise_event_calls(tree)
        rel_file = getattr(self, "_active_rel_file", "")
        selected: set[ast.Call] = set()
        for node in sorted(calls, key=location):
            category = precise_event_base.precise_category(node, set())
            if category == "other":
                category = "filewrite"
            if self.selected_event_total >= self.max_event_total:
                break
            if self.selected_event_category_counts[category] >= self.max_events_per_category:
                continue
            selected.add(node)
            self.selected_event_total += 1
            self.selected_event_category_counts[category] += 1
            self.selected_event_keys.add(
                (
                    rel_file,
                    int(getattr(node, "lineno", 0)),
                    int(getattr(node, "col_offset", 0)),
                )
            )
        return selected

    def _visit_file(self, py_file: Path):
        self._active_rel_file = v9.v7.clean_text_for_json(str(py_file.relative_to(self.src_dir)))
        try:
            super()._visit_file(py_file)
        finally:
            self._active_rel_file = ""

    def _make_event_node(self, rel_file, node, category, scope, assignment_names):
        key = (
            rel_file,
            int(getattr(node, "lineno", 0)),
            int(getattr(node, "col_offset", 0)),
        )
        if key not in self.selected_event_keys:
            return ""
        if not self.ast_nodes.get((rel_file, key[1], "Call"), ""):
            return ""
        return super()._make_event_node(rel_file, node, category, scope, assignment_names)

    def build(self) -> dict[str, Any]:
        graph = super().build()
        graph["graph_version"] = GRAPH_VERSION
        graph["meta"].update(
            {
                "control_profile": GRAPH_VERSION,
                "budget_profile": "package_level_semantic_budget",
                "selection_strategy": "label_blind_event_category_quota_and_bounded_neutral_context",
                "max_events_per_category": self.max_events_per_category,
                "selected_event_category_counts": dict(sorted(self.selected_event_category_counts.items())),
                "selected_event_total": self.selected_event_total,
            }
        )
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(PackageSemanticBudgetGraphBuilderV10C(args.src_dir).build()["meta"], indent=2))
