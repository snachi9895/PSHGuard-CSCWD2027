#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import step02_build_single_sample_behavior_graph_v7 as v7
from step02d_build_single_sample_behavior_event_graph_v8 import (
    BehaviorEventGraphBuilderV8,
    EVENT_CATEGORIES,
    assigned_names,
)


NETWORK_EXACT = {
    "requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch",
    "requests.request", "requests.head", "urllib.request.urlopen", "httpx.get", "httpx.post",
    "httpx.put", "httpx.delete", "httpx.patch", "httpx.request", "socket.socket",
}
DECODER_EXACT = {
    "base64.b64decode", "base64.urlsafe_b64decode", "b64decode", "codecs.decode",
    "zlib.decompress", "gzip.decompress", "bz2.decompress", "lzma.decompress",
    "marshal.loads", "pickle.loads", "bytes.fromhex",
}
PROCESS_EXACT = {
    "os.system", "os.popen", "subprocess.Popen", "subprocess.run", "subprocess.call",
    "subprocess.check_output", "subprocess.check_call",
}
EXEC_EXACT = {"eval", "exec", "compile"}
INFO_READ_EXACT = {
    "os.getenv", "os.environ.get", "os.environ.__getitem__", "platform.node",
    "platform.platform", "socket.gethostname",
}
PATH_WRITE_SUFFIXES = (".write_text", ".write_bytes")
STRONG_DIRECT_RELATIONS = {
    "E_ReadToNetwork",
    "E_DecodeToExecute",
    "E_DownloadToExecute",
    "E_WriteToExecute",
}


def call_name(node: ast.Call) -> str:
    return v7.clean_text_for_json(v7.get_full_name(node.func))


def open_mode(node: ast.Call) -> str:
    mode = "r"
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
        mode = node.args[1].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            mode = keyword.value.value
    return mode


def is_open_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and call_name(node) in {"open", "io.open", "codecs.open"}


def is_write_open_call(node: ast.AST) -> bool:
    return is_open_call(node) and any(flag in open_mode(node) for flag in ("w", "a", "x", "+"))


def receiver_root(name: str) -> str:
    return name.split(".", 1)[0] if "." in name else ""


def precise_category(node: ast.Call, write_handles: set[str]) -> str:
    name = call_name(node)
    low = name.lower()

    if name in INFO_READ_EXACT:
        return "info_read"
    if name in NETWORK_EXACT:
        return "network"
    if name in DECODER_EXACT or low.endswith((".b64decode", ".decompress", ".fromhex")):
        return "decoder"
    if name in PROCESS_EXACT:
        return "process"
    if name in EXEC_EXACT:
        return "exec_sink"
    if is_write_open_call(node) or low.endswith(PATH_WRITE_SUFFIXES):
        return "filewrite"
    if low.endswith(".write") and receiver_root(name) in write_handles:
        return "filewrite"
    return "other"


class BehaviorEventGraphBuilderV8B(BehaviorEventGraphBuilderV8):
    """V8b: conservative event semantics and separately marked strong chains."""

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
        write_handle_stack: list[set[str]] = [set()]

        class EventVisitor(ast.NodeVisitor):
            def _visit_scoped(self, node):
                scope_stack.append(v7.clean_text_for_json(node.name))
                write_handle_stack.append(set())
                self.generic_visit(node)
                write_handle_stack.pop()
                scope_stack.pop()

            def visit_FunctionDef(self, node):
                self._visit_scoped(node)

            def visit_AsyncFunctionDef(self, node):
                self._visit_scoped(node)

            def visit_ClassDef(self, node):
                self._visit_scoped(node)

            def visit_Assign(self, node):
                names = set()
                for target in node.targets:
                    names.update(assigned_names(target))
                if is_write_open_call(node.value):
                    write_handle_stack[-1].update(names)
                assignment_stack.append(names)
                self.visit(node.value)
                assignment_stack.pop()

            def visit_AnnAssign(self, node):
                names = assigned_names(node.target)
                if is_write_open_call(node.value):
                    write_handle_stack[-1].update(names)
                assignment_stack.append(names)
                if node.value is not None:
                    self.visit(node.value)
                assignment_stack.pop()

            def visit_With(self, node):
                temporary_handles = set()
                for item in node.items:
                    if is_write_open_call(item.context_expr) and item.optional_vars is not None:
                        temporary_handles.update(assigned_names(item.optional_vars))
                    self.visit(item.context_expr)
                write_handle_stack[-1].update(temporary_handles)
                for statement in node.body:
                    self.visit(statement)
                write_handle_stack[-1].difference_update(temporary_handles)

            def visit_AsyncWith(self, node):
                self.visit_With(node)

            def visit_Call(self, node):
                category = precise_category(node, write_handle_stack[-1])
                event_id = ""
                if category in EVENT_CATEGORIES:
                    scope = builder._get_scope_name(scope_stack)
                    assigned = assignment_stack[-1] if assignment_stack else set()
                    event_id = builder._make_event_node(rel_file, node, category, scope, assigned)
                    if event_id and event_stack:
                        builder._add_behavior_flow(event_id, event_stack[-1], "nested_call")
                    if event_id:
                        event_stack.append(event_id)
                self.generic_visit(node)
                if event_id:
                    event_stack.pop()

        EventVisitor().visit(tree)

    def _mark_strong_chains(self):
        forward = defaultdict(list)
        existing = []
        for edge in self.edges:
            if edge["type"].startswith("REV_"):
                continue
            forward[edge["src"]].append(edge)
            if edge["type"] in STRONG_DIRECT_RELATIONS:
                existing.append((edge["src"], edge["dst"], edge["type"]))

        for src, dst, relation in existing:
            self._add_bidirectional(
                src, "E_StrongBehaviorChain", dst, "REV_E_StrongBehaviorChain",
                strong_reason=relation,
            )

        for src, first_edges in list(forward.items()):
            for first in first_edges:
                if first["type"] != "E_DownloadToWrite":
                    continue
                for second in forward.get(first["dst"], []):
                    if second["type"] == "E_WriteToExecute":
                        self._add_bidirectional(
                            src, "E_StrongBehaviorChain", second["dst"], "REV_E_StrongBehaviorChain",
                            strong_reason="download_write_execute",
                        )

    def build(self) -> dict[str, Any]:
        graph = super().build()
        self._mark_strong_chains()
        self._deduplicate_edges()

        node_counts = defaultdict(int)
        for node in self.nodes:
            node_counts[node.get("type", "")] += 1
        graph.update({
            "graph_version": "v8b_precise_behavior_event",
            "nodes": self.nodes,
            "edges": self.edges,
        })
        graph["meta"].update({
            "control_profile": "v8b_precise_behavior_event_diverse_small",
            "event_strategy": "conservative_api_allowlist_and_tracked_file_handles",
            "strong_chain_strategy": "direct_high_risk_or_download_write_execute",
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_counts": dict(sorted(node_counts.items())),
            "edge_counts": self._edge_counts(),
        })
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(BehaviorEventGraphBuilderV8B(args.src_dir).build()["meta"], indent=2))
