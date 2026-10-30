#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import step02_build_single_sample_behavior_graph_v7 as v7
from step02c_build_single_sample_behavior_graph_v7c import BehaviorChainGraphBuilderV7C


EVENT_CATEGORIES = [
    "info_read",
    "decoder",
    "network",
    "filewrite",
    "process",
    "exec_sink",
]
MAX_EVENT_NODES_TOTAL = 256
MAX_EVENT_NODES_PER_FILE = 48

FLOW_RELATIONS = {
    ("info_read", "network"): "E_ReadToNetwork",
    ("decoder", "exec_sink"): "E_DecodeToExecute",
    ("decoder", "process"): "E_DecodeToExecute",
    ("network", "filewrite"): "E_DownloadToWrite",
    ("network", "process"): "E_DownloadToExecute",
    ("network", "exec_sink"): "E_DownloadToExecute",
    ("filewrite", "process"): "E_WriteToExecute",
    ("filewrite", "exec_sink"): "E_WriteToExecute",
}

INFO_READ_NAMES = {
    "os.getenv",
    "os.environ.get",
    "os.environ.__getitem__",
    "platform.node",
    "platform.platform",
    "socket.gethostname",
}


def normalized_call_category(node: ast.Call) -> str:
    name = v7.clean_text_for_json(v7.get_full_name(node.func))
    low = name.lower()

    if name in INFO_READ_NAMES or low.endswith(".getenv"):
        return "info_read"
    if name == "open" or low.endswith(".open"):
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            mode = node.args[1].value
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                mode = keyword.value.value
        return "filewrite" if any(flag in mode for flag in ("w", "a", "x", "+")) else "info_read"

    category = v7.call_category(name)
    if category != "other":
        return category

    if low.endswith((".b64decode", ".decompress", ".fromhex")):
        return "decoder"
    if low.endswith((".urlopen", ".request", ".get", ".post", ".put")) and any(
        term in low for term in ("request", "urllib", "http", "socket")
    ):
        return "network"
    if low.endswith((".write", ".write_text", ".write_bytes")):
        return "filewrite"
    if low.endswith((".popen", ".run", ".call", ".system", ".check_output", ".check_call")) and any(
        term in low for term in ("subprocess", "os.", "shell")
    ):
        return "process"
    return "other"


def assigned_names(target: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def referenced_names(nodes: list[ast.AST]) -> set[str]:
    names = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                names.add(sub.id)
    return names


class BehaviorEventGraphBuilderV8(BehaviorChainGraphBuilderV7C):
    """V8: one heterograph with explicit behavior events and high-confidence flows."""

    def __init__(self, src_dir: Path):
        super().__init__(src_dir)
        self.event_nodes: dict[str, dict[str, Any]] = {}
        self.event_records: list[dict[str, Any]] = []
        self.flow_evidence_counts: dict[str, int] = defaultdict(int)
        self.event_counts_by_file: dict[str, int] = defaultdict(int)
        self.truncated_events = False

    def _link_param_flow(self):
        # V7/V7c linked category co-occurrences inside a scope. V8 replaces
        # those speculative links with event-level nested-call and def-use flows.
        return

    @staticmethod
    def _phase(rel_file: str, scope: str) -> str:
        low_file = rel_file.lower()
        low_scope = scope.lower()
        if low_file.endswith("setup.py") or "install" in low_scope:
            return "install"
        if scope == "<module>":
            return "import"
        return "runtime"

    def _make_event_node(
        self,
        rel_file: str,
        node: ast.Call,
        category: str,
        scope: str,
        assignment_names: set[str],
    ) -> str:
        if len(self.event_nodes) >= MAX_EVENT_NODES_TOTAL or self.event_counts_by_file[rel_file] >= MAX_EVENT_NODES_PER_FILE:
            self.truncated_events = True
            return ""

        call_name = v7.clean_text_for_json(v7.get_full_name(node.func))
        lineno = int(getattr(node, "lineno", 0))
        col = int(getattr(node, "col_offset", 0))
        eid = self._new_node_id("event")
        phase = self._phase(rel_file, scope)
        arg_nodes = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
        literals = []
        for arg in arg_nodes:
            literals.extend(v7.get_string_literals(arg))
        has_payload_arg = any(v7.is_payload_like(text) for text in literals)

        features = {
            **{f"is_{name}": float(category == name) for name in EVENT_CATEGORIES},
            "phase_install": float(phase == "install"),
            "phase_import": float(phase == "import"),
            "phase_runtime": float(phase == "runtime"),
            "has_payload_arg": float(has_payload_arg),
            "has_assignment": float(bool(assignment_names)),
            "arg_count": float(len(arg_nodes)),
        }
        event = {
            "id": eid,
            "type": "event",
            "category": category,
            "call_name": call_name,
            "file": rel_file,
            "lineno": lineno,
            "col_offset": col,
            "scope": scope,
            "phase": phase,
            "features": features,
        }
        self._add_node(event)
        self.event_nodes[eid] = event
        self.event_counts_by_file[rel_file] += 1

        ast_id = self.ast_nodes.get((rel_file, lineno, "Call"), "")
        if ast_id:
            self._add_bidirectional(ast_id, "E_Event", eid, "REV_E_Event")
        module_id = self.module_nodes.get(rel_file, "")
        if module_id:
            self._add_bidirectional(module_id, "E_ModuleEvent", eid, "REV_E_ModuleEvent")

        for (p_file, p_line, _), param_id in self.param_nodes.items():
            if p_file == rel_file and p_line == lineno:
                self._add_bidirectional(eid, "E_EventParam", param_id, "REV_E_EventParam")

        record = {
            "id": eid,
            "category": category,
            "file": rel_file,
            "lineno": lineno,
            "col_offset": col,
            "scope": scope,
            "assigned_names": set(assignment_names),
            "referenced_names": referenced_names(arg_nodes),
        }
        self.event_records.append(record)
        return eid

    def _add_behavior_flow(self, src_id: str, dst_id: str, evidence: str):
        if not src_id or not dst_id or src_id == dst_id:
            return
        src = self.event_nodes[src_id]
        dst = self.event_nodes[dst_id]
        relation = FLOW_RELATIONS.get((src["category"], dst["category"]))
        if not relation:
            return
        self._add_bidirectional(
            src_id,
            relation,
            dst_id,
            f"REV_{relation}",
            evidence=evidence,
        )
        self.flow_evidence_counts[evidence] += 1

    def _extract_events_from_file(self, rel_file: str):
        py_file = self.src_dir / rel_file
        try:
            tree = ast.parse(v7.clean_text_for_json(py_file.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            return

        builder = self
        scope_stack: list[str] = []
        assignment_stack: list[set[str]] = []
        event_stack: list[str] = []

        class EventVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                scope_stack.append(v7.clean_text_for_json(node.name))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                scope_stack.append(v7.clean_text_for_json(node.name))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_ClassDef(self, node):
                scope_stack.append(v7.clean_text_for_json(node.name))
                self.generic_visit(node)
                scope_stack.pop()

            def visit_Assign(self, node):
                names = set()
                for target in node.targets:
                    names.update(assigned_names(target))
                assignment_stack.append(names)
                self.visit(node.value)
                assignment_stack.pop()

            def visit_AnnAssign(self, node):
                names = assigned_names(node.target)
                assignment_stack.append(names)
                if node.value is not None:
                    self.visit(node.value)
                assignment_stack.pop()

            def visit_Call(self, node):
                category = normalized_call_category(node)
                event_id = ""
                if category in EVENT_CATEGORIES:
                    scope = builder._get_scope_name(scope_stack)
                    assigned = assignment_stack[-1] if assignment_stack else set()
                    event_id = builder._make_event_node(rel_file, node, category, scope, assigned)
                    if event_id and event_stack:
                        # The child call produces input consumed by the outer call.
                        builder._add_behavior_flow(event_id, event_stack[-1], "nested_call")
                    if event_id:
                        event_stack.append(event_id)

                self.generic_visit(node)

                if event_id:
                    event_stack.pop()

        EventVisitor().visit(tree)

    def _link_def_use_flows(self):
        by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.event_records:
            by_scope[(record["file"], record["scope"])].append(record)

        for records in by_scope.values():
            records.sort(key=lambda r: (r["lineno"], r["col_offset"]))
            last_definition: dict[str, str] = {}
            for record in records:
                for name in sorted(record["referenced_names"]):
                    src_id = last_definition.get(name, "")
                    if src_id:
                        self._add_behavior_flow(src_id, record["id"], "def_use")
                for name in sorted(record["assigned_names"]):
                    last_definition[name] = record["id"]

    def build(self) -> dict[str, Any]:
        graph = super().build()

        # Use exactly the files retained by the bounded V7c selection profile.
        for rel_file in sorted(self.module_nodes):
            self._extract_events_from_file(rel_file)
        self._link_def_use_flows()
        self._deduplicate_edges()

        node_counts = defaultdict(int)
        event_category_counts = defaultdict(int)
        event_phase_counts = defaultdict(int)
        for node in self.nodes:
            node_counts[node.get("type", "")] += 1
            if node.get("type") == "event":
                event_category_counts[node.get("category", "")] += 1
                event_phase_counts[node.get("phase", "")] += 1

        graph.update({
            "graph_version": "v8_high_confidence_behavior_event",
            "nodes": self.nodes,
            "edges": self.edges,
        })
        graph["meta"].update({
            "control_profile": "v8_high_confidence_behavior_event_diverse_small",
            "event_strategy": "explicit_behavior_events_with_nested_and_def_use_flows",
            "disabled_relation": "E_ParamFlow",
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_counts": dict(sorted(node_counts.items())),
            "edge_counts": self._edge_counts(),
            "event_category_counts": dict(sorted(event_category_counts.items())),
            "event_phase_counts": dict(sorted(event_phase_counts.items())),
            "flow_evidence_counts": dict(sorted(self.flow_evidence_counts.items())),
            "truncated_events": self.truncated_events,
            "max_event_nodes_total": MAX_EVENT_NODES_TOTAL,
            "max_event_nodes_per_file": MAX_EVENT_NODES_PER_FILE,
        })
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(BehaviorEventGraphBuilderV8(args.src_dir).build()["meta"], indent=2))
