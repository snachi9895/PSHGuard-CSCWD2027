#!/usr/bin/env python3
"""V10: event-centered, payload-aware bounded behavior graph."""

from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V9_DIR = HERE.parent / "final_graph_v9_20260608"
V3_DIR = HERE.parent / "final_dataset_v3"
for directory in (V9_DIR, V3_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import step01_build_single_sample_graph_v9 as v9
import step02d_build_single_sample_behavior_event_graph_v8 as event_base
import step02e_build_single_sample_behavior_event_graph_v8b as precise_event_base
from step01b_build_single_sample_graph_v9b_strict_payload import strict_call_payload_count
from step12_build_single_sample_graph_v9c_global_param import GlobalParamAnchorGraphBuilderV9C


GRAPH_VERSION = "v10b_event_centered_nonempty_module_global_param_20260610"

MAX_SELECTED_FILES = 24
MAX_AST_PER_FILE = 96
MAX_AST_TOTAL = 960
MAX_PARAM_TOTAL = 256
MAX_EVENT_TOTAL = 256
MAX_EVENT_PER_FILE = 48

MAX_CONTEXT_PER_EVENT = 8
MAX_CONTEXT_PER_PAYLOAD_CALL = 5
MAX_NEUTRAL_PER_FILE = 6
MAX_NEUTRAL_TOTAL = 64
CONTEXT_LINE_WINDOW = 8

NEUTRAL_TYPES = {
    "Import", "ImportFrom", "FunctionDef", "AsyncFunctionDef", "ClassDef",
}
CONTEXT_TYPES = {
    "Assign", "AnnAssign", "AugAssign", "Return", "If", "For", "While",
    "With", "Try", "Expr", "Call", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Import", "ImportFrom",
}


def location(node: ast.AST) -> tuple[int, int, str]:
    return (
        int(getattr(node, "lineno", 0)),
        int(getattr(node, "col_offset", 0)),
        type(node).__name__,
    )


class EventCenteredGlobalParamGraphBuilderV10(GlobalParamAnchorGraphBuilderV9C):
    """Retain precise-event anchors and a bounded amount of local AST context."""

    max_selected_files = MAX_SELECTED_FILES
    max_ast_per_file = MAX_AST_PER_FILE
    max_ast_total = MAX_AST_TOTAL
    max_param_total = MAX_PARAM_TOTAL
    max_event_total = MAX_EVENT_TOTAL
    max_event_per_file = MAX_EVENT_PER_FILE
    max_context_per_event = MAX_CONTEXT_PER_EVENT
    max_context_per_payload_call = MAX_CONTEXT_PER_PAYLOAD_CALL
    max_neutral_per_file = MAX_NEUTRAL_PER_FILE
    max_neutral_total = MAX_NEUTRAL_TOTAL
    context_line_window = CONTEXT_LINE_WINDOW

    def __init__(self, src_dir: Path):
        v9.v7.MAX_PY_FILES = self.max_selected_files
        v9.v7.MAX_AST_NODES_PER_FILE = self.max_ast_per_file
        v9.v7.MAX_AST_NODES_TOTAL = self.max_ast_total
        v9.v7.MAX_PARAM_NODES_TOTAL = self.max_param_total
        event_base.MAX_EVENT_NODES_TOTAL = self.max_event_total
        event_base.MAX_EVENT_NODES_PER_FILE = self.max_event_per_file
        super().__init__(src_dir)
        self.neutral_ast_used = 0
        self.ast_role_counts: Counter[str] = Counter()
        self.file_context_stats: dict[str, dict[str, int]] = {}

    @staticmethod
    def _ancestor_candidates(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
        result = []
        current = parents.get(node)
        while current is not None:
            if type(current).__name__ in CONTEXT_TYPES:
                result.append(current)
            current = parents.get(current)
        return result

    def _nearby_candidates(self, anchor: ast.AST, candidates: list[ast.AST]) -> list[ast.AST]:
        anchor_line = int(getattr(anchor, "lineno", 0))

        def rank(node: ast.AST) -> tuple[int, int, int, int, str]:
            line = int(getattr(node, "lineno", 0))
            node_type = type(node).__name__
            type_rank = {
                "Assign": 0, "AnnAssign": 0, "AugAssign": 0, "Return": 1,
                "Call": 2, "If": 3, "For": 3, "While": 3, "With": 3, "Try": 3,
                "Import": 4, "ImportFrom": 4,
                "FunctionDef": 5, "AsyncFunctionDef": 5, "ClassDef": 5,
            }.get(node_type, 6)
            return (
                abs(line - anchor_line),
                type_rank,
                line,
                int(getattr(node, "col_offset", 0)),
                node_type,
            )

        return sorted(
            (
                node for node in candidates
                if node is not anchor
                and abs(int(getattr(node, "lineno", 0)) - anchor_line) <= self.context_line_window
            ),
            key=rank,
        )

    @staticmethod
    def _set_role(
        selected: dict[tuple[int, int, str], tuple[ast.AST, str, str]],
        node: ast.AST,
        scope: str,
        role: str,
    ) -> None:
        key = location(node)
        priorities = {
            "event_anchor": 0,
            "payload_anchor": 1,
            "event_ancestor": 2,
            "event_nearby": 3,
            "payload_context": 4,
            "neutral_fallback": 5,
        }
        previous = selected.get(key)
        if previous is None or priorities[role] < priorities[previous[2]]:
            selected[key] = (node, scope, role)

    def _mark_ast_role(self, ast_id: str, role: str) -> None:
        for node in reversed(self.nodes):
            if node.get("id") == ast_id:
                features = node.setdefault("features", {})
                for name in (
                    "event_anchor", "payload_anchor", "event_ancestor",
                    "event_nearby", "payload_context", "neutral_fallback",
                ):
                    features[f"context_role_{name}"] = float(name == role)
                break
        self.ast_role_counts[role] += 1

    @staticmethod
    def _precise_event_calls(tree: ast.AST) -> set[ast.Call]:
        """Use the same tracked-write-handle semantics as the V8b Event builder."""
        calls: set[ast.Call] = set()
        write_handle_stack: list[set[str]] = [set()]

        class EventAnchorVisitor(ast.NodeVisitor):
            def _visit_scoped(self, node):
                write_handle_stack.append(set())
                self.generic_visit(node)
                write_handle_stack.pop()

            def visit_FunctionDef(self, node):
                self._visit_scoped(node)

            def visit_AsyncFunctionDef(self, node):
                self._visit_scoped(node)

            def visit_ClassDef(self, node):
                self._visit_scoped(node)

            def visit_Assign(self, node):
                names = set()
                for target in node.targets:
                    names.update(event_base.assigned_names(target))
                if precise_event_base.is_write_open_call(node.value):
                    write_handle_stack[-1].update(names)
                self.generic_visit(node)

            def visit_AnnAssign(self, node):
                if precise_event_base.is_write_open_call(node.value):
                    write_handle_stack[-1].update(event_base.assigned_names(node.target))
                self.generic_visit(node)

            def visit_With(self, node):
                temporary_handles = set()
                for item in node.items:
                    if precise_event_base.is_write_open_call(item.context_expr) and item.optional_vars is not None:
                        temporary_handles.update(event_base.assigned_names(item.optional_vars))
                write_handle_stack[-1].update(temporary_handles)
                self.generic_visit(node)
                write_handle_stack[-1].difference_update(temporary_handles)

            def visit_AsyncWith(self, node):
                self.visit_With(node)

            def visit_Call(self, node):
                if precise_event_base.precise_category(node, write_handle_stack[-1]) != "other":
                    calls.add(node)
                self.generic_visit(node)

        EventAnchorVisitor().visit(tree)
        return calls

    def _visit_file(self, py_file: Path):
        rel_file = v9.v7.clean_text_for_json(str(py_file.relative_to(self.src_dir)))
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(v9.v7.clean_text_for_json(source))
        except Exception:
            return

        self._collect_imports(rel_file, tree)

        parents: dict[ast.AST, ast.AST] = {}
        scopes: dict[ast.AST, str] = {}
        all_candidates: list[ast.AST] = []
        event_calls: list[ast.Call] = []
        payload_calls: list[ast.Call] = []
        precise_calls = self._precise_event_calls(tree)
        scope_stack: list[str] = []

        class ContextVisitor(ast.NodeVisitor):
            def generic_visit(visitor_self, node):
                pushed = False
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope_stack.append(v9.v7.clean_text_for_json(getattr(node, "name", type(node).__name__)))
                    pushed = True
                scopes[node] = self._get_scope_name(scope_stack)
                if type(node).__name__ in v9.v7.IMPORTANT_AST_TYPES:
                    all_candidates.append(node)
                if isinstance(node, ast.Call):
                    if node in precise_calls:
                        event_calls.append(node)
                    elif strict_call_payload_count(node):
                        payload_calls.append(node)
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
                    visitor_self.visit(child)
                if pushed:
                    scope_stack.pop()

        ContextVisitor().visit(tree)
        selected: dict[tuple[int, int, str], tuple[ast.AST, str, str]] = {}

        for anchor in sorted(event_calls, key=location)[:self.max_event_per_file]:
            self._set_role(selected, anchor, scopes[anchor], "event_anchor")
            used = 1
            for node in self._ancestor_candidates(anchor, parents):
                if used >= self.max_context_per_event:
                    break
                self._set_role(selected, node, scopes.get(node, "<module>"), "event_ancestor")
                used += 1
            for node in self._nearby_candidates(anchor, all_candidates):
                if used >= self.max_context_per_event:
                    break
                self._set_role(selected, node, scopes.get(node, "<module>"), "event_nearby")
                used += 1

        for anchor in sorted(payload_calls, key=location):
            self._set_role(selected, anchor, scopes[anchor], "payload_anchor")
            used = 1
            for node in self._ancestor_candidates(anchor, parents) + self._nearby_candidates(anchor, all_candidates):
                if used >= self.max_context_per_payload_call:
                    break
                self._set_role(selected, node, scopes.get(node, "<module>"), "payload_context")
                used += 1

        neutral_candidates = sorted(
            (node for node in all_candidates if type(node).__name__ in NEUTRAL_TYPES),
            key=location,
        )
        neutral_added = 0
        for node in neutral_candidates:
            if neutral_added >= self.max_neutral_per_file or self.neutral_ast_used >= self.max_neutral_total:
                break
            before = len(selected)
            self._set_role(selected, node, scopes.get(node, "<module>"), "neutral_fallback")
            if len(selected) > before:
                neutral_added += 1
                self.neutral_ast_used += 1

        ranked = sorted(
            selected.values(),
            key=lambda item: (
                {
                    "event_anchor": 0, "payload_anchor": 1, "event_ancestor": 2,
                    "event_nearby": 3, "payload_context": 4, "neutral_fallback": 5,
                }[item[2]],
                *location(item[0]),
            ),
        )
        kept = ranked[:self.max_ast_per_file]
        if len(ranked) > len(kept):
            self.truncated_ast = True
        kept.sort(key=lambda item: location(item[0]))
        if not kept:
            self.file_context_stats[rel_file] = {
                "precise_event_calls_found": len(event_calls),
                "strict_payload_calls_found": len(payload_calls),
                "selected_ast_nodes": 0,
            }
            return

        mod_id = ""
        file_roles: Counter[str] = Counter()
        for node, scope, role in kept:
            sub_type = type(node).__name__
            lineno = int(getattr(node, "lineno", 0))
            ast_id = self._make_ast_node(rel_file, lineno, sub_type)
            if not ast_id:
                continue
            if not mod_id:
                mod_id = self._make_module_node(rel_file)
            self._mark_ast_role(ast_id, role)
            file_roles[role] += 1
            self._add_bidirectional(mod_id, "E_AST", ast_id, "REV_E_AST")
            self.file_ast_order[rel_file].append(ast_id)
            if not isinstance(node, ast.Call):
                continue
            category = v9.precise_category(node, set())
            args = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
            for arg in args[:v9.v7.MAX_CALL_ARGS]:
                source_kind, texts = v9.source_kind_and_texts(arg)
                for text in texts[:v9.v7.MAX_CALL_ARGS]:
                    param_id = self._make_param_node(rel_file, lineno, text, source_kind, category)
                    if param_id:
                        self._add_bidirectional(ast_id, "E_Param", param_id, "REV_E_Param")

        self.file_context_stats[rel_file] = {
            "precise_event_calls_found": len(event_calls),
            "strict_payload_calls_found": len(payload_calls),
            "selected_ast_nodes": sum(file_roles.values()),
            **{f"selected_{name}": count for name, count in sorted(file_roles.items())},
        }

    def build(self) -> dict[str, Any]:
        graph = super().build()
        graph["graph_version"] = GRAPH_VERSION
        graph["meta"].update(
            {
                "control_profile": GRAPH_VERSION,
                "selection_strategy": "label_blind_event_centered_bounded_context_with_neutral_fallback",
                "ast_selection_strategy": "precise_event_anchor_ancestors_nearby_context_and_neutral_fallback",
                "max_context_per_event": self.max_context_per_event,
                "max_context_per_payload_call": self.max_context_per_payload_call,
                "context_line_window": self.context_line_window,
                "max_neutral_ast_per_file": self.max_neutral_per_file,
                "max_neutral_ast_total": self.max_neutral_total,
                "ast_role_counts": dict(sorted(self.ast_role_counts.items())),
                "file_context_stats": self.file_context_stats,
                "budget_profile": "event_centered_medium_2000",
                "max_py_files": self.max_selected_files,
                "max_ast_nodes_per_file": self.max_ast_per_file,
                "max_ast_nodes_total": self.max_ast_total,
                "max_param_nodes_total": self.max_param_total,
                "max_event_nodes_total": self.max_event_total,
                "max_event_nodes_per_file": self.max_event_per_file,
            }
        )
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(EventCenteredGlobalParamGraphBuilderV10(args.src_dir).build()["meta"], indent=2))
