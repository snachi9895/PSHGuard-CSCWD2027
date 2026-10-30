#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent / "final_dataset_v3"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

import step02_build_single_sample_behavior_graph_v7 as v7
from step02e_build_single_sample_behavior_event_graph_v8b import (
    BehaviorEventGraphBuilderV8B,
    precise_category,
)


GRAPH_VERSION = "v9_payload_aware_bounded_behavior_20260608"

v7.IMPORTANT_AST_TYPES = {
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Call", "Assign",
    "AnnAssign", "AugAssign", "Import", "ImportFrom", "Return",
}
v7.MAX_PY_FILES = 16
v7.MAX_AST_NODES_PER_FILE = 36
v7.MAX_AST_NODES_TOTAL = 576
v7.MAX_PARAM_NODES_TOTAL = 192
v7.MAX_CALL_ARGS = 8
v7.MAX_PARAM_FLOW_LINKS_PER_SCOPE = 16

HIGH_RISK_PAIRS = {
    ("info_read", "network"),
    ("decoder", "exec_sink"),
    ("decoder", "process"),
    ("network", "filewrite"),
    ("network", "process"),
    ("network", "exec_sink"),
    ("filewrite", "process"),
    ("filewrite", "exec_sink"),
}


def call_payload_count(node: ast.Call) -> int:
    args = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
    return sum(
        1
        for arg in args
        for text in v7.get_string_literals(arg)
        if v7.is_payload_like(text)
    )


def source_kind_and_texts(arg: ast.AST) -> tuple[str, list[str]]:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return "literal", [arg.value]
    if isinstance(arg, ast.JoinedStr):
        return "fstring", v7.get_string_literals(arg)
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
        return "concat", v7.get_string_literals(arg)
    return "unknown", v7.get_string_literals(arg)


class PayloadAwareBoundedBehaviorGraphBuilderV9(BehaviorEventGraphBuilderV8B):
    """Paper-facing bounded semantic graph with deterministic extraction."""

    def __init__(self, src_dir: Path):
        super().__init__(src_dir)
        self.parsed_py_files = 0
        self.parse_failed_py_files = 0
        self.file_line_counts: dict[str, int] = {}
        self.file_semantic_profiles: dict[str, dict[str, Any]] = {}
        self.selected_file_profiles: list[dict[str, Any]] = []

    def _profile_file(self, path: Path) -> dict[str, Any] | None:
        rel = v7.clean_text_for_json(str(path.relative_to(self.src_dir)))
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(v7.clean_text_for_json(source))
        except Exception:
            self.parse_failed_py_files += 1
            return None

        self.parsed_py_files += 1
        self.file_line_counts[rel] = max(1, source.count("\n") + 1)
        categories = Counter()
        payload_count = 0
        import_count = 0
        function_count = 0
        important_count = 0

        for node in ast.walk(tree):
            sub_type = type(node).__name__
            important_count += int(sub_type in v7.IMPORTANT_AST_TYPES)
            import_count += int(isinstance(node, (ast.Import, ast.ImportFrom)))
            function_count += int(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            if isinstance(node, ast.Call):
                category = precise_category(node, set())
                if category != "other":
                    categories[category] += 1
                payload_count += call_payload_count(node)

        category_names = set(categories)
        strong_pair_count = sum(1 for pair in HIGH_RISK_PAIRS if pair[0] in category_names and pair[1] in category_names)
        low = rel.lower()
        is_setup = int(low.endswith("setup.py"))
        is_init = int(low.endswith("__init__.py"))
        semantic_score = (
            100 * is_setup
            + 30 * strong_pair_count
            + 14 * len(category_names)
            + 4 * min(sum(categories.values()), 12)
            + 8 * min(payload_count, 8)
            + 5 * is_init
            + min(import_count, 5)
            + min(function_count, 5)
        )
        profile = {
            "rel_file": rel,
            "path": path,
            "is_setup": is_setup,
            "is_init": is_init,
            "semantic_score": semantic_score,
            "precise_event_count": sum(categories.values()),
            "event_category_count": len(category_names),
            "event_categories": ",".join(sorted(category_names)),
            "strong_pair_count": strong_pair_count,
            "payload_arg_count": payload_count,
            "import_count": import_count,
            "function_count": function_count,
            "important_ast_count": important_count,
            "stable_hash": v7.stable_hash(rel, 32),
        }
        self.file_semantic_profiles[rel] = profile
        return profile

    def _iter_python_files(self) -> list[Path]:
        files = sorted(path for path in self.src_dir.rglob("*.py") if path.is_file())
        self.original_py_files = len(files)
        profiles = [profile for path in files if (profile := self._profile_file(path)) is not None]

        # One deterministic ranking algorithm. Semantic evidence drives the
        # ranking; stable hash only resolves ties and fills low-signal cases.
        profiles.sort(
            key=lambda p: (
                -p["is_setup"],
                -p["semantic_score"],
                -p["is_init"],
                -p["important_ast_count"],
                p["stable_hash"],
                p["rel_file"],
            )
        )
        selected = profiles[:v7.MAX_PY_FILES]
        self.selected_file_profiles = [
            {key: value for key, value in profile.items() if key != "path"}
            for profile in selected
        ]
        self.kept_py_files = len(selected)
        self.skipped_py_files = max(0, self.original_py_files - self.kept_py_files)
        self.truncated_files = self.original_py_files > self.kept_py_files
        return [profile["path"] for profile in selected]

    def _make_ast_node(self, rel_file: str, lineno: int, sub_type: str) -> str:
        node_id = super()._make_ast_node(rel_file, lineno, sub_type)
        if not node_id:
            return node_id
        total_lines = max(1, self.file_line_counts.get(rel_file, 1))
        for node in reversed(self.nodes):
            if node.get("id") == node_id:
                features = node.setdefault("features", {})
                features["relative_position"] = min(1.0, max(0.0, float(lineno) / total_lines))
                features.pop("lineno", None)
                break
        return node_id

    def _should_keep_param(self, text: str, category: str) -> bool:
        text = v7.clean_text_for_json(text)
        if not text:
            return False
        if category in {"decoder", "network", "filewrite", "process", "exec_sink"}:
            return True
        if v7.is_payload_like(text):
            return True
        return len(text) >= 32 and v7.safe_entropy(text[:2048]) >= 4.0

    @staticmethod
    def _ast_priority(node: ast.AST) -> tuple[int, int, int]:
        lineno = int(getattr(node, "lineno", 0))
        if isinstance(node, ast.Call):
            category = precise_category(node, set())
            if category != "other":
                return (0, lineno, int(getattr(node, "col_offset", 0)))
            if call_payload_count(node):
                return (1, lineno, int(getattr(node, "col_offset", 0)))
            return (2, lineno, int(getattr(node, "col_offset", 0)))
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return (3, lineno, int(getattr(node, "col_offset", 0)))
        return (4, lineno, int(getattr(node, "col_offset", 0)))

    def _visit_file(self, py_file: Path):
        rel_file = v7.clean_text_for_json(str(py_file.relative_to(self.src_dir)))
        mod_id = self._make_module_node(rel_file)
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(v7.clean_text_for_json(source))
        except Exception:
            return

        self._collect_imports(rel_file, tree)
        scope_stack: list[str] = []
        candidates: list[tuple[ast.AST, str]] = []

        class CandidateVisitor(ast.NodeVisitor):
            def generic_visit(visitor_self, node):
                pushed = False
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope_stack.append(v7.clean_text_for_json(getattr(node, "name", type(node).__name__)))
                    pushed = True
                if type(node).__name__ in v7.IMPORTANT_AST_TYPES:
                    candidates.append((node, self._get_scope_name(scope_stack)))
                super(CandidateVisitor, visitor_self).generic_visit(node)
                if pushed:
                    scope_stack.pop()

        CandidateVisitor().visit(tree)
        ranked = sorted(candidates, key=lambda item: self._ast_priority(item[0]))
        selected = ranked[:v7.MAX_AST_NODES_PER_FILE]
        if len(candidates) > len(selected):
            self.truncated_ast = True
        selected.sort(key=lambda item: (int(getattr(item[0], "lineno", 0)), int(getattr(item[0], "col_offset", 0))))

        for node, scope in selected:
            sub_type = type(node).__name__
            lineno = int(getattr(node, "lineno", 0))
            ast_id = self._make_ast_node(rel_file, lineno, sub_type)
            if not ast_id:
                continue
            self._add_bidirectional(mod_id, "E_AST", ast_id, "REV_E_AST")
            self.file_ast_order[rel_file].append(ast_id)
            if not isinstance(node, ast.Call):
                continue
            category = precise_category(node, set())
            args = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
            for arg in args[:v7.MAX_CALL_ARGS]:
                source_kind, texts = source_kind_and_texts(arg)
                for text in texts[:v7.MAX_CALL_ARGS]:
                    param_id = self._make_param_node(rel_file, lineno, text, source_kind, category)
                    if param_id:
                        self._add_bidirectional(ast_id, "E_Param", param_id, "REV_E_Param")

    def build(self) -> dict[str, Any]:
        graph = super().build()
        graph["graph_version"] = GRAPH_VERSION
        graph["meta"].update({
            "control_profile": GRAPH_VERSION,
            "selection_strategy": "deterministic_payload_behavior_semantic_ranking",
            "selection_tie_breaker": "stable_hash",
            "source_scan_strategy": "parse_all_python_files_then_select_bounded_semantic_subgraph",
            "ast_selection_strategy": "precise_event_then_payload_call_then_structural_context",
            "ast_position_feature": "relative_position",
            "param_strategy": "precise_event_arguments_or_payload_like_or_high_entropy",
            "parsed_py_files": self.parsed_py_files,
            "parse_failed_py_files": self.parse_failed_py_files,
            "max_py_files": v7.MAX_PY_FILES,
            "max_ast_nodes_per_file": v7.MAX_AST_NODES_PER_FILE,
            "max_ast_nodes_total": v7.MAX_AST_NODES_TOTAL,
            "max_param_nodes_total": v7.MAX_PARAM_NODES_TOTAL,
            "selected_file_profiles": self.selected_file_profiles,
        })
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(PayloadAwareBoundedBehaviorGraphBuilderV9(args.src_dir).build()["meta"], indent=2))

