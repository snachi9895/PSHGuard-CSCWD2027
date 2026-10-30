#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


URL_RE = re.compile(r"(https?://|ftp://|www\.)", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BASE64_LIKE_RE = re.compile(r"^[A-Za-z0-9+/=]{16,}$")
POWERSHELL_RE = re.compile(r"(powershell|pwsh|-enc\b|-encodedcommand\b)", re.IGNORECASE)
CMD_RE = re.compile(r"(cmd\.exe|/bin/sh|bash|curl|wget|chmod|chown)", re.IGNORECASE)
PATH_RE = re.compile(r"([A-Za-z]:\\|/etc/|/tmp/|/var/|/home/|AppData|System32)", re.IGNORECASE)

DECODER_NAMES = {
    "base64.b64decode",
    "b64decode",
    "codecs.decode",
    "zlib.decompress",
    "marshal.loads",
    "pickle.loads",
    "json.loads",
    "bytes.fromhex",
}

NETWORK_NAMES = {
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.request",
    "urllib.request.urlopen",
    "socket.socket",
}

FILE_WRITE_NAMES = {
    "open",
    "Path.write_text",
    "Path.write_bytes",
}

PROCESS_NAMES = {
    "os.system",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_output",
    "subprocess.check_call",
}

EXEC_SINK_NAMES = {
    "eval",
    "exec",
    "compile",
}

IMPORTANT_AST_TYPES = {
    "Module",
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
    "If",
    "For",
    "While",
    "With",
    "Try",
    "Expr",
}

MAX_PY_FILES = 80
MAX_AST_NODES_PER_FILE = 80
MAX_AST_NODES_TOTAL = 3000
MAX_PARAM_NODES_TOTAL = 800
MAX_CALL_ARGS = 12
MAX_PARAM_FLOW_LINKS_PER_SCOPE = 32


def clean_text_for_json(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    s = s.replace("\x00", "")
    return s.strip()


def safe_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = defaultdict(int)
    for ch in s:
        freq[ch] += 1
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(clean_text_for_json(text).encode("utf-8", errors="ignore")).hexdigest()[:n]


def compact_module_name(text: str) -> str:
    text = str(text).replace("\\", "/")
    if text.endswith(".py"):
        text = text[:-3]
    parts = [p for p in Path(text).parts if p not in {".", ""}]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def get_full_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = get_full_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        return get_full_name(node.func)
    return ""


def get_string_literals(node: ast.AST) -> list[str]:
    vals = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            vals.append(sub.value)
        elif isinstance(sub, ast.JoinedStr):
            parts = []
            for v in sub.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
            if parts:
                vals.append("".join(parts))
    return vals


def call_category(call_name: str) -> str:
    if call_name in DECODER_NAMES:
        return "decoder"
    if call_name in NETWORK_NAMES:
        return "network"
    if call_name in FILE_WRITE_NAMES:
        return "filewrite"
    if call_name in PROCESS_NAMES:
        return "process"
    if call_name in EXEC_SINK_NAMES:
        return "exec_sink"
    return "other"


def is_payload_like(text: str) -> bool:
    text = clean_text_for_json(text)
    return any([
        bool(URL_RE.search(text)),
        bool(IP_RE.search(text)),
        bool(BASE64_LIKE_RE.match(text)),
        bool(POWERSHELL_RE.search(text)),
        bool(CMD_RE.search(text)),
        bool(PATH_RE.search(text)),
    ])


class BehaviorChainGraphBuilderV7:
    def __init__(self, src_dir: Path):
        self.src_dir = Path(src_dir)
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.node_id_counter = 0

        self.module_nodes: dict[str, str] = {}
        self.ast_nodes: dict[tuple[str, int, str], str] = {}
        self.param_nodes: dict[tuple[str, int, str], str] = {}
        self.file_ast_order: dict[str, list[str]] = defaultdict(list)
        self.module_imports: dict[str, set[str]] = defaultdict(set)
        self.param_ids_by_scope_category: dict[tuple[str, str, str], list[str]] = defaultdict(list)

        self.original_py_files = 0
        self.kept_py_files = 0
        self.skipped_py_files = 0
        self.truncated_files = False
        self.truncated_ast = False
        self.truncated_param = False

    def _new_node_id(self, prefix: str) -> str:
        self.node_id_counter += 1
        return f"{prefix}_{self.node_id_counter}"

    def _add_node(self, node: dict) -> str:
        self.nodes.append(node)
        return node["id"]

    def _add_edge(self, src: str, etype: str, dst: str, **extra):
        if src == dst:
            return
        self.edges.append({"src": src, "type": etype, "dst": dst, **extra})

    def _add_bidirectional(self, src: str, etype: str, dst: str, rev_etype: str | None = None, **extra):
        self._add_edge(src, etype, dst, **extra)
        self._add_edge(dst, rev_etype or f"REV_{etype}", src, **extra)

    def _iter_python_files(self) -> list[Path]:
        files = sorted([p for p in self.src_dir.rglob("*.py") if p.is_file()])
        self.original_py_files = len(files)

        scored = []
        for p in files:
            rel = str(p.relative_to(self.src_dir))
            score = 0
            low = rel.lower()
            if low.endswith("setup.py"):
                score += 100
            if "__init__.py" in low:
                score += 20
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:200000]
            except Exception:
                text = ""
            suspicious_terms = ["eval", "exec", "subprocess", "os.system", "requests", "urlopen", "base64", "powershell", "wget", "curl"]
            score += sum(10 for term in suspicious_terms if term in text)
            scored.append((-score, len(rel), rel, p))

        scored.sort()
        kept = [x[3] for x in scored[:MAX_PY_FILES]]
        self.kept_py_files = len(kept)
        self.skipped_py_files = max(0, self.original_py_files - self.kept_py_files)
        self.truncated_files = self.original_py_files > self.kept_py_files
        return kept

    def _make_module_node(self, rel_file: str) -> str:
        rel_file = clean_text_for_json(rel_file)
        if rel_file in self.module_nodes:
            return self.module_nodes[rel_file]
        nid = self._new_node_id("module")
        self._add_node({
            "id": nid,
            "type": "module",
            "name": rel_file,
            "features": {
                "is_python_file": 1.0,
                "path_depth": float(len(Path(rel_file).parts)),
                "is_setup_py": float(rel_file.endswith("setup.py")),
                "is_init_py": float(rel_file.endswith("__init__.py")),
            },
        })
        self.module_nodes[rel_file] = nid
        return nid

    def _make_ast_node(self, rel_file: str, lineno: int, sub_type: str) -> str:
        key = (rel_file, lineno, sub_type)
        if key in self.ast_nodes:
            return self.ast_nodes[key]
        if len(self.ast_nodes) >= MAX_AST_NODES_TOTAL:
            self.truncated_ast = True
            return ""

        nid = self._new_node_id("ast")
        self._add_node({
            "id": nid,
            "type": "ast",
            "sub_type": clean_text_for_json(sub_type),
            "file": clean_text_for_json(rel_file),
            "lineno": int(lineno),
            "features": {
                "lineno": float(lineno),
                "is_call": float(sub_type == "Call"),
                "is_import": float(sub_type in {"Import", "ImportFrom"}),
                "is_function": float(sub_type in {"FunctionDef", "AsyncFunctionDef"}),
                "is_assign": float(sub_type in {"Assign", "AnnAssign", "AugAssign"}),
                "is_return": float(sub_type == "Return"),
            },
        })
        self.ast_nodes[key] = nid
        return nid

    def _build_param_features(self, text: str, source_kind: str, category: str):
        text_norm = clean_text_for_json(text)
        has_url = bool(URL_RE.search(text_norm))
        has_ip = bool(IP_RE.search(text_norm))
        has_b64 = bool(BASE64_LIKE_RE.match(text_norm))
        has_ps = bool(POWERSHELL_RE.search(text_norm))
        has_cmd = bool(CMD_RE.search(text_norm))
        has_path = bool(PATH_RE.search(text_norm))
        payload_like = has_url or has_ip or has_b64 or has_ps or has_cmd or has_path
        return {
            "length": float(len(text_norm)),
            "entropy": float(safe_entropy(text_norm[:2048])),
            "has_url": float(has_url),
            "has_ip": float(has_ip),
            "has_base64_like_pattern": float(has_b64),
            "has_powershell": float(has_ps),
            "has_cmd": float(has_cmd),
            "has_path": float(has_path),
            "payload_like": float(payload_like),
            "call_is_decoder": float(category == "decoder"),
            "call_is_network": float(category == "network"),
            "call_is_filewrite": float(category == "filewrite"),
            "call_is_process": float(category == "process"),
            "call_is_exec_sink": float(category == "exec_sink"),
            "source_literal": float(source_kind == "literal"),
            "source_fstring": float(source_kind == "fstring"),
            "source_concat_like": float(source_kind == "concat"),
            "source_unknown": float(source_kind == "unknown"),
        }

    def _should_keep_param(self, text: str, category: str) -> bool:
        text = clean_text_for_json(text)
        if not text:
            return False
        if category in {"decoder", "network", "filewrite", "process", "exec_sink"}:
            return True
        if is_payload_like(text):
            return True
        return len(text) >= 16 and safe_entropy(text[:2048]) >= 3.5

    def _make_param_node(self, rel_file: str, lineno: int, text: str, source_kind: str, category: str) -> str:
        text = clean_text_for_json(text)
        if not self._should_keep_param(text, category):
            return ""
        if len(self.param_nodes) >= MAX_PARAM_NODES_TOTAL:
            self.truncated_param = True
            return ""
        key = (rel_file, lineno, text)
        if key in self.param_nodes:
            return self.param_nodes[key]
        nid = self._new_node_id("param")
        self._add_node({
            "id": nid,
            "type": "param",
            "text": text[:512],
            "file": clean_text_for_json(rel_file),
            "lineno": int(lineno),
            "features": self._build_param_features(text, source_kind, category),
        })
        self.param_nodes[key] = nid
        return nid

    def _get_scope_name(self, stack: list[str]) -> str:
        return "::".join(stack) if stack else "<module>"

    def _collect_imports(self, rel_file: str, tree: ast.AST):
        current_module = compact_module_name(rel_file)
        current_pkg = current_module.split(".")[:-1]
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    name = clean_text_for_json(alias.name)
                    if name:
                        self.module_imports[rel_file].add(name)
            elif isinstance(sub, ast.ImportFrom):
                module = clean_text_for_json(sub.module or "")
                candidates = set()
                if sub.level and sub.level > 0:
                    keep = max(0, len(current_pkg) - sub.level + 1)
                    base = current_pkg[:keep]
                    if module:
                        candidates.add(".".join(base + module.split(".")))
                    for alias in sub.names:
                        if alias.name != "*":
                            candidates.add(".".join(base + ([module] if module else []) + [alias.name]))
                elif module:
                    candidates.add(module)
                    for alias in sub.names:
                        if alias.name != "*":
                            candidates.add(f"{module}.{alias.name}")
                for c in candidates:
                    c = clean_text_for_json(c).strip(".")
                    if c:
                        self.module_imports[rel_file].add(c)

    def _visit_file(self, py_file: Path):
        rel_file = clean_text_for_json(str(py_file.relative_to(self.src_dir)))
        mod_id = self._make_module_node(rel_file)
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(clean_text_for_json(source))
        except Exception:
            return

        self._collect_imports(rel_file, tree)
        scope_stack: list[str] = []
        file_ast_added = 0

        class Visitor(ast.NodeVisitor):
            def __init__(self, outer: "BehaviorChainGraphBuilderV7"):
                self.outer = outer

            def generic_visit(self, node):
                nonlocal file_ast_added
                sub_type = type(node).__name__
                lineno = int(getattr(node, "lineno", 0))

                pushed = False
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope_stack.append(clean_text_for_json(getattr(node, "name", sub_type)))
                    pushed = True

                ast_id = ""
                if sub_type in IMPORTANT_AST_TYPES and file_ast_added < MAX_AST_NODES_PER_FILE:
                    ast_id = self.outer._make_ast_node(rel_file, lineno, sub_type)
                    if ast_id:
                        file_ast_added += 1
                        self.outer._add_bidirectional(mod_id, "E_AST", ast_id, "REV_E_AST")
                        self.outer.file_ast_order[rel_file].append(ast_id)

                if isinstance(node, ast.Call) and ast_id:
                    call_name = clean_text_for_json(get_full_name(node.func))
                    category = call_category(call_name)
                    arg_like_nodes = list(node.args) + [kw.value for kw in node.keywords if kw.value is not None]
                    curr_scope = self.outer._get_scope_name(scope_stack)

                    for arg in arg_like_nodes[:MAX_CALL_ARGS]:
                        source_kind = "unknown"
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            texts = [arg.value]
                            source_kind = "literal"
                        elif isinstance(arg, ast.JoinedStr):
                            texts = get_string_literals(arg)
                            source_kind = "fstring"
                        elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                            texts = get_string_literals(arg)
                            source_kind = "concat"
                        else:
                            texts = get_string_literals(arg)

                        for text in texts[:MAX_CALL_ARGS]:
                            pid = self.outer._make_param_node(rel_file, lineno, text, source_kind, category)
                            if not pid:
                                continue
                            self.outer._add_bidirectional(ast_id, "E_Param", pid, "REV_E_Param")
                            self.outer.param_ids_by_scope_category[(rel_file, curr_scope, category)].append(pid)

                super().generic_visit(node)
                if pushed:
                    scope_stack.pop()

        Visitor(self).visit(tree)
        if file_ast_added >= MAX_AST_NODES_PER_FILE:
            self.truncated_ast = True

    def _link_intra_seq(self):
        for rel_file, ids in self.file_ast_order.items():
            for left, right in zip(ids, ids[1:]):
                self._add_bidirectional(left, "E_IntraSeq", right, "REV_E_IntraSeq", file=rel_file)

    def _module_aliases(self) -> dict[str, str]:
        aliases = {}
        for rel_file, nid in self.module_nodes.items():
            mod = compact_module_name(rel_file)
            if mod:
                aliases[mod] = nid
                aliases[mod.split(".")[-1]] = nid
        return aliases

    def _link_internal_imports(self):
        aliases = self._module_aliases()
        for rel_file, imported in self.module_imports.items():
            src_id = self.module_nodes.get(rel_file)
            if not src_id:
                continue
            for name in sorted(imported):
                parts = name.split(".")
                candidates = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
                dst_id = ""
                matched = ""
                for c in candidates:
                    if c in aliases:
                        dst_id = aliases[c]
                        matched = c
                        break
                if dst_id and dst_id != src_id:
                    self._add_bidirectional(src_id, "E_InternalImport", dst_id, "REV_E_InternalImport", import_name=name, matched_module=matched)

    def _link_param_flow(self):
        flow_pairs = [
            ("decoder", "exec_sink"),
            ("decoder", "process"),
            ("network", "filewrite"),
            ("network", "process"),
            ("filewrite", "process"),
        ]
        scopes = set((rel, scope) for rel, scope, _ in self.param_ids_by_scope_category.keys())
        for rel, scope in scopes:
            link_count = 0
            for src_cat, dst_cat in flow_pairs:
                src_ids = list(dict.fromkeys(self.param_ids_by_scope_category.get((rel, scope, src_cat), [])))
                dst_ids = list(dict.fromkeys(self.param_ids_by_scope_category.get((rel, scope, dst_cat), [])))
                for s in src_ids:
                    for d in dst_ids:
                        if s != d:
                            self._add_bidirectional(s, "E_ParamFlow", d, "REV_E_ParamFlow", flow=f"{src_cat}->{dst_cat}", file=rel, scope=scope)
                            link_count += 1
                            if link_count >= MAX_PARAM_FLOW_LINKS_PER_SCOPE:
                                break
                    if link_count >= MAX_PARAM_FLOW_LINKS_PER_SCOPE:
                        break

    def _deduplicate_edges(self):
        uniq = {}
        for e in self.edges:
            key = (e["src"], e["type"], e["dst"])
            if key not in uniq:
                uniq[key] = e
        self.edges = list(uniq.values())

    def _edge_counts(self) -> dict[str, int]:
        counts = defaultdict(int)
        for e in self.edges:
            counts[e["type"]] += 1
        return dict(sorted(counts.items()))

    def build(self) -> dict[str, Any]:
        py_files = self._iter_python_files()
        for py_file in py_files:
            self._visit_file(py_file)

        self._link_intra_seq()
        self._link_internal_imports()
        self._link_param_flow()
        self._deduplicate_edges()

        node_counts = defaultdict(int)
        for n in self.nodes:
            node_counts[n.get("type", "")] += 1

        return {
            "graph_version": "v7_behavior_chain_controlled",
            "src_dir": str(self.src_dir),
            "nodes": self.nodes,
            "edges": self.edges,
            "meta": {
                "original_py_files": self.original_py_files,
                "kept_py_files": self.kept_py_files,
                "skipped_py_files": self.skipped_py_files,
                "truncated_files": self.truncated_files,
                "truncated_ast": self.truncated_ast,
                "truncated_param": self.truncated_param,
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "node_counts": dict(sorted(node_counts.items())),
                "edge_counts": self._edge_counts(),
            },
        }
