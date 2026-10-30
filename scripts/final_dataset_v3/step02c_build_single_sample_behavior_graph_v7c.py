#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import step02_build_single_sample_behavior_graph_v7 as v7


# V7c is the paper-facing graph extraction profile. It is not described as a
# size-only fix; the intent is behavior-relevant graph extraction with bounded
# complexity for stable training.
v7.IMPORTANT_AST_TYPES = {
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "Call",
    "Assign",
    "AnnAssign",
    "AugAssign",
    "Import",
    "ImportFrom",
    "Return",
}
v7.MAX_PY_FILES = 16
v7.MAX_AST_NODES_PER_FILE = 36
v7.MAX_AST_NODES_TOTAL = 576
v7.MAX_PARAM_NODES_TOTAL = 192
v7.MAX_CALL_ARGS = 8
v7.MAX_PARAM_FLOW_LINKS_PER_SCOPE = 16


BEHAVIOR_TERMS = {
    "process": [
        "subprocess",
        "os.system",
        "popen",
        "check_output",
        "check_call",
        "cmd.exe",
        "powershell",
        "pwsh",
        "/bin/sh",
        "bash",
    ],
    "network": [
        "requests.",
        "urlopen",
        "urllib.",
        "socket.",
        "http://",
        "https://",
        "wget",
        "curl",
    ],
    "decoder": [
        "base64",
        "b64decode",
        "zlib.",
        "marshal.",
        "pickle.",
        "fromhex",
    ],
    "exec_sink": [
        "eval(",
        "exec(",
        "compile(",
    ],
    "filewrite": [
        "write(",
        "write_text",
        "write_bytes",
        "open(",
        "chmod",
    ],
    "install": [
        "setup(",
        "setuptools",
        "distutils",
        "entry_points",
        "install_requires",
    ],
}


class BehaviorChainGraphBuilderV7C(v7.BehaviorChainGraphBuilderV7):
    def _score_python_file(self, p: Path) -> tuple[int, dict[str, int], str]:
        rel = str(p.relative_to(self.src_dir))
        low = rel.lower()
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")[:300000].lower()
        except Exception:
            text = ""

        category_hits = {}
        score = 0
        for category, terms in BEHAVIOR_TERMS.items():
            hits = sum(1 for term in terms if term in text)
            category_hits[category] = hits
            if hits:
                score += 12 + min(hits, 8) * 3

        if low.endswith("setup.py"):
            score += 120
        if low.endswith("__init__.py"):
            score += 18
        if "setup" in low or "install" in low:
            score += 24
        if "test" in low or "docs" in low or "example" in low:
            score -= 20
        if "__pycache__" in low:
            score -= 100

        return score, category_hits, rel

    def _iter_python_files(self) -> list[Path]:
        files = sorted([p for p in self.src_dir.rglob("*.py") if p.is_file()])
        self.original_py_files = len(files)

        scored = []
        setup_files = []
        init_files = []
        behavior_files = []
        fallback_files = []

        for p in files:
            score, category_hits, rel = self._score_python_file(p)
            item = (-score, len(rel), rel, p, score, category_hits)
            scored.append(item)
            low = rel.lower()
            if low.endswith("setup.py"):
                setup_files.append(item)
            elif low.endswith("__init__.py"):
                init_files.append(item)
            elif score > 0:
                behavior_files.append(item)
            else:
                fallback_files.append(item)

        for bucket in [setup_files, init_files, behavior_files, fallback_files]:
            bucket.sort()

        selected = []
        seen = set()

        def add_from(bucket, limit):
            added = 0
            for item in bucket:
                path = item[3]
                if path in seen:
                    continue
                selected.append(item)
                seen.add(path)
                added += 1
                if added >= limit or len(selected) >= v7.MAX_PY_FILES:
                    break

        add_from(setup_files, 2)
        add_from(init_files, 4)
        add_from(behavior_files, v7.MAX_PY_FILES)
        if len(selected) < min(v7.MAX_PY_FILES, len(files)):
            add_from(fallback_files, v7.MAX_PY_FILES - len(selected))

        selected = selected[:v7.MAX_PY_FILES]
        kept = [x[3] for x in selected]
        self.kept_py_files = len(kept)
        self.skipped_py_files = max(0, self.original_py_files - self.kept_py_files)
        self.truncated_files = self.original_py_files > self.kept_py_files
        return kept

    def build(self) -> dict[str, Any]:
        graph = super().build()
        graph["graph_version"] = "v7c_behavior_relevant_bounded"
        graph.setdefault("meta", {})
        graph["meta"].update({
            "control_profile": "v7c_behavior_relevant_bounded_small",
            "selection_strategy": "setup_init_behavior_priority_then_fallback",
            "ast_strategy": "event_centric_structural_ast",
            "max_py_files": v7.MAX_PY_FILES,
            "max_ast_nodes_per_file": v7.MAX_AST_NODES_PER_FILE,
            "max_ast_nodes_total": v7.MAX_AST_NODES_TOTAL,
            "max_param_nodes_total": v7.MAX_PARAM_NODES_TOTAL,
            "max_call_args": v7.MAX_CALL_ARGS,
            "max_param_flow_links_per_scope": v7.MAX_PARAM_FLOW_LINKS_PER_SCOPE,
            "important_ast_types": sorted(v7.IMPORTANT_AST_TYPES),
        })
        return graph
