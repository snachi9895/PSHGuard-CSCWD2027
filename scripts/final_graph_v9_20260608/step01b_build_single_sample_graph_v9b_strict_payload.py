#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import step01_build_single_sample_graph_v9 as v9


GRAPH_VERSION = "v9b_strict_payload_aware_bounded_behavior_20260608"
BASE64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def strict_base64_like(text: str) -> bool:
    text = v9.v7.clean_text_for_json(text)
    if len(text) < 32 or not BASE64_CHARS_RE.fullmatch(text):
        return False
    core = text.rstrip("=")
    groups = sum(any(ch in chars for ch in core) for chars in (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "+/",
    ))
    return groups >= 2 and len(set(core)) >= 12 and v9.v7.safe_entropy(core[:2048]) >= 4.0


def strict_payload_flags(text: str) -> dict[str, bool]:
    text = v9.v7.clean_text_for_json(text)
    entropy = v9.v7.safe_entropy(text[:2048])
    return {
        "has_url": bool(v9.v7.URL_RE.search(text)),
        "has_ip": bool(v9.v7.IP_RE.search(text)),
        "has_encoded_blob": strict_base64_like(text),
        "has_powershell": bool(v9.v7.POWERSHELL_RE.search(text)),
        "has_cmd": bool(v9.v7.CMD_RE.search(text)),
        "high_entropy_blob": len(text) >= 64 and entropy >= 4.3,
    }


def keep_strict_param(text: str, category: str) -> bool:
    flags = strict_payload_flags(text)
    if category in {"process", "exec_sink", "decoder"}:
        return bool(text)
    if flags["has_encoded_blob"] or flags["has_powershell"] or flags["has_cmd"] or flags["high_entropy_blob"]:
        return True
    if category == "network" and (flags["has_url"] or flags["has_ip"]):
        return True
    return False


def strict_call_payload_count(node: ast.Call) -> int:
    category = v9.precise_category(node, set())
    args = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
    return sum(
        1
        for arg in args
        for text in v9.v7.get_string_literals(arg)
        if keep_strict_param(text, category)
    )


class StrictPayloadAwareBoundedBehaviorGraphBuilderV9B(v9.PayloadAwareBoundedBehaviorGraphBuilderV9):
    """V9b removes broad payload-string shortcuts and enriches Event summaries."""

    def _profile_file(self, path: Path) -> dict[str, Any] | None:
        rel = v9.v7.clean_text_for_json(str(path.relative_to(self.src_dir)))
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(v9.v7.clean_text_for_json(source))
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
            important_count += int(sub_type in v9.v7.IMPORTANT_AST_TYPES)
            import_count += int(isinstance(node, (ast.Import, ast.ImportFrom)))
            function_count += int(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            if isinstance(node, ast.Call):
                category = v9.precise_category(node, set())
                if category != "other":
                    categories[category] += 1
                payload_count += strict_call_payload_count(node)

        category_names = set(categories)
        strong_pair_count = sum(
            1 for pair in v9.HIGH_RISK_PAIRS
            if pair[0] in category_names and pair[1] in category_names
        )
        low = rel.lower()
        is_setup = int(low.endswith("setup.py"))
        is_init = int(low.endswith("__init__.py"))
        semantic_score = (
            100 * is_setup
            + 30 * strong_pair_count
            + 14 * len(category_names)
            + 4 * min(sum(categories.values()), 12)
            + 10 * min(payload_count, 8)
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
            "strict_payload_arg_count": payload_count,
            "import_count": import_count,
            "function_count": function_count,
            "important_ast_count": important_count,
            "stable_hash": v9.v7.stable_hash(rel, 32),
        }
        self.file_semantic_profiles[rel] = profile
        return profile

    @staticmethod
    def _ast_priority(node: ast.AST) -> tuple[int, int, int]:
        lineno = int(getattr(node, "lineno", 0))
        col = int(getattr(node, "col_offset", 0))
        if isinstance(node, ast.Call):
            category = v9.precise_category(node, set())
            if category != "other":
                return (0, lineno, col)
            if strict_call_payload_count(node):
                return (1, lineno, col)
            return (2, lineno, col)
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return (3, lineno, col)
        return (4, lineno, col)

    def _should_keep_param(self, text: str, category: str) -> bool:
        return keep_strict_param(text, category)

    def _build_param_features(self, text: str, source_kind: str, category: str):
        features = super()._build_param_features(text, source_kind, category)
        flags = strict_payload_flags(text)
        features.update({
            "has_base64_like_pattern": float(flags["has_encoded_blob"]),
            "payload_like": float(any(flags.values())),
            "strict_payload": float(keep_strict_param(text, category)),
        })
        return features

    def _make_event_node(self, rel_file, node, category, scope, assignment_names):
        event_id = super()._make_event_node(rel_file, node, category, scope, assignment_names)
        if not event_id:
            return event_id
        args = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
        texts = [
            text
            for arg in args
            for text in v9.v7.get_string_literals(arg)
        ]
        kept = [text for text in texts if keep_strict_param(text, category)]
        flags = [strict_payload_flags(text) for text in kept]
        max_length = max((len(text) for text in kept), default=0)
        max_entropy = max((v9.v7.safe_entropy(text[:2048]) for text in kept), default=0.0)
        event = self.event_nodes[event_id]
        event["features"].update({
            "has_payload_arg": float(bool(kept)),
            "payload_arg_count_norm": min(len(kept), 8) / 8.0,
            "payload_max_length_log": math.log1p(min(max_length, 1_000_000)) / math.log1p(1_000_000),
            "payload_max_entropy_norm": min(max_entropy, 8.0) / 8.0,
            "payload_has_encoded_blob": float(any(item["has_encoded_blob"] for item in flags)),
            "payload_has_command": float(any(item["has_powershell"] or item["has_cmd"] for item in flags)),
            "payload_has_url_or_ip": float(any(item["has_url"] or item["has_ip"] for item in flags)),
        })
        return event_id

    def build(self) -> dict[str, Any]:
        graph = super().build()
        graph["graph_version"] = GRAPH_VERSION
        graph["meta"].update({
            "control_profile": GRAPH_VERSION,
            "selection_strategy": "deterministic_precise_event_and_strict_payload_ranking",
            "param_strategy": "strict_contextual_payload_arguments",
            "base64_strategy": "strict_length_charset_diversity_entropy",
            "event_payload_summary": "normalized_strict_payload_features",
        })
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    graph = StrictPayloadAwareBoundedBehaviorGraphBuilderV9B(args.src_dir).build()
    print(json.dumps(graph["meta"], indent=2))

