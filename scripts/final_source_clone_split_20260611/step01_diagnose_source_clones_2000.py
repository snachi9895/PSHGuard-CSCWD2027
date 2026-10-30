#!/usr/bin/env python3
"""Diagnose high-confidence package-level source clones on the V10c 2K set.

This stage deliberately does not create a new train/validation/test split.
It reads all Python files directly from each source archive and separates:

1. strong relations that are safe enough to force into one future split;
2. broader structural similarities that must only be audited.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import itertools
import json
import math
import os
import signal
import tarfile
import time
import tokenize
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import pandas as pd
from tqdm import tqdm


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TASK_CSV = (
    ROOT
    / "graph_artifacts/final_v9b_strict_payload_medium_2000_20260609"
    / "step08_diverse_source_tasks_2000.csv"
)
CURRENT_SPLIT_CSV = (
    ROOT
    / "manifests/v10c_template_aware_diverse_split_2000_20260610"
    / "v10c_template_aware_diverse_all.csv"
)
OUT_DIR = ROOT / "manifests/final_source_clone_aware_split_202606"
DIAG_DIR = OUT_DIR / "diagnostics_2000"
CACHE_DIR = DIAG_DIR / "fingerprint_cache"
SAMPLE_CSV = DIAG_DIR / "source_fingerprint_samples.csv"
STRONG_PAIR_CSV = DIAG_DIR / "strong_clone_pairs.csv"
CANDIDATE_PAIR_CSV = DIAG_DIR / "candidate_clone_pairs.csv"
COMPONENT_CSV = DIAG_DIR / "strong_component_summary.csv"
SUMMARY_CSV = DIAG_DIR / "diagnostic_summary.csv"
ERROR_CSV = DIAG_DIR / "fingerprint_failures.csv"
ABSTRACT_GROUP_CSV = DIAG_DIR / "abstract_ast_clone_group_audit.csv"
THRESHOLD_CSV = DIAG_DIR / "abstract_ast_threshold_sensitivity.csv"

MAX_PY_FILE_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_PY_BYTES = 80 * 1024 * 1024
MIN_STRONG_TOKENS = 50
MIN_STRONG_AST_NODES = 40
MIN_CANDIDATE_AST_NODES = 80
MIN_SHARED_WEIGHT = 100
MIN_OVERLAP_SIMILARITY = 0.80
MAX_COMMON_FILE_FREQUENCY = 20
ZIP_PASSWORD = b"infected"
FINGERPRINT_VERSION = "source_clone_fingerprint_v1"


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_multiset(values: list[str]) -> str:
    return digest_text("\n".join(sorted(values)))


def safe_member_name(name: str) -> str:
    normalized = str(PurePosixPath(name.replace("\\", "/")))
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(f"unsafe archive member: {name}")
    return normalized


def iter_python_members(archive_path: Path):
    lower = archive_path.name.lower()
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as handle:
            for member in handle.getmembers():
                name = safe_member_name(member.name)
                if not member.isfile() or not name.lower().endswith(".py"):
                    continue
                if member.size > MAX_PY_FILE_BYTES:
                    yield name, None, member.size, "file_too_large"
                    continue
                stream = handle.extractfile(member)
                if stream is not None:
                    yield name, stream.read(MAX_PY_FILE_BYTES + 1), member.size, ""
        return
    if lower.endswith((".zip", ".whl")):
        with zipfile.ZipFile(archive_path) as handle:
            for member in handle.infolist():
                name = safe_member_name(member.filename)
                if member.is_dir() or not name.lower().endswith(".py"):
                    continue
                if member.file_size > MAX_PY_FILE_BYTES:
                    yield name, None, member.file_size, "file_too_large"
                    continue
                try:
                    data = handle.read(member, pwd=ZIP_PASSWORD)
                except RuntimeError:
                    data = handle.read(member)
                yield name, data[: MAX_PY_FILE_BYTES + 1], member.file_size, ""
        return
    raise ValueError(f"unsupported archive: {archive_path}")


def decode_python(data: bytes) -> str:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def lexical_fingerprint(data: bytes) -> tuple[str, int]:
    kept: list[str] = []
    ignored = {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    try:
        for token in tokenize.tokenize(io.BytesIO(data).readline):
            if token.type not in ignored:
                kept.append(f"{token.type}:{token.string}")
    except Exception:
        text = decode_python(data)
        kept = [part for part in text.split() if part]
    return digest_text("\n".join(kept)), len(kept)


class AbstractSyntax(ast.NodeTransformer):
    """Remove concrete names and values for candidate generation only."""

    def visit_Name(self, node: ast.Name):
        return ast.copy_location(ast.Name(id="ID", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg):
        node.arg = "ARG"
        node.annotation = self.visit(node.annotation) if node.annotation else None
        return node

    def visit_Attribute(self, node: ast.Attribute):
        node.value = self.visit(node.value)
        node.attr = "ATTR"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = "FUNC"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node.name = "FUNC"
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = "CLASS"
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        value_type = type(node.value).__name__
        return ast.copy_location(ast.Constant(value=f"<{value_type}>"), node)


def ast_fingerprints(data: bytes) -> tuple[str, str, int]:
    try:
        tree = ast.parse(decode_python(data))
    except Exception:
        return "", "", 0
    concrete = ast.dump(tree, annotate_fields=True, include_attributes=False)
    abstract_tree = AbstractSyntax().visit(tree)
    ast.fix_missing_locations(abstract_tree)
    abstract = ast.dump(abstract_tree, annotate_fields=True, include_attributes=False)
    return digest_text(concrete), digest_text(abstract), sum(1 for _ in ast.walk(tree))


def archive_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cache_path(graph_id: str) -> Path:
    return CACHE_DIR / f"{graph_id}.json"


def valid_cache(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("fingerprint_version") == FINGERPRINT_VERSION
    except Exception:
        return False


def fingerprint_one(record: dict, force: bool = False) -> dict:
    started = time.time()
    graph_id = str(record["graph_id"])
    output_path = cache_path(graph_id)
    if not force and valid_cache(output_path):
        with output_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    archive_path = Path(str(record["archive_path"]))
    result = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "graph_id": graph_id,
        "sample_id": str(record.get("sample_id", graph_id)),
        "package_name": str(record.get("package_name", "")),
        "version": str(record.get("version", "")),
        "group_key": str(record.get("group_key", "")),
        "label": str(record.get("label", "")),
        "source_group": str(record.get("source_group", "")),
        "old_split": str(record.get("split", "")),
        "archive_path": str(archive_path),
        "status": "ok",
        "error_message": "",
    }
    try:
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        raw_hashes: list[str] = []
        lexical_hashes: list[str] = []
        concrete_ast_hashes: list[str] = []
        abstract_ast_hashes: list[str] = []
        file_records: list[dict] = []
        total_bytes = total_tokens = total_ast_nodes = parsed_ast_files = skipped_files = 0
        truncated_package = False
        for name, data, declared_size, member_error in iter_python_members(archive_path):
            if member_error or data is None or len(data) > MAX_PY_FILE_BYTES:
                skipped_files += 1
                continue
            if total_bytes + len(data) > MAX_PACKAGE_PY_BYTES:
                truncated_package = True
                skipped_files += 1
                continue
            raw_hash = digest_bytes(data)
            lexical_hash, token_count = lexical_fingerprint(data)
            concrete_ast_hash, abstract_ast_hash, ast_nodes = ast_fingerprints(data)
            total_bytes += len(data)
            total_tokens += token_count
            total_ast_nodes += ast_nodes
            parsed_ast_files += int(bool(concrete_ast_hash))
            raw_hashes.append(raw_hash)
            lexical_hashes.append(lexical_hash)
            if concrete_ast_hash:
                concrete_ast_hashes.append(concrete_ast_hash)
                abstract_ast_hashes.append(abstract_ast_hash)
            file_records.append(
                {
                    "name": name,
                    "bytes": len(data),
                    "declared_bytes": declared_size,
                    "tokens": token_count,
                    "ast_nodes": ast_nodes,
                    "raw_hash": raw_hash,
                    "lexical_hash": lexical_hash,
                    "concrete_ast_hash": concrete_ast_hash,
                    "abstract_ast_hash": abstract_ast_hash,
                }
            )

        result.update(
            {
                "archive_sha256": archive_sha256(archive_path),
                "python_file_count": len(file_records),
                "parsed_ast_file_count": parsed_ast_files,
                "skipped_python_file_count": skipped_files,
                "total_python_bytes": total_bytes,
                "total_tokens": total_tokens,
                "total_ast_nodes": total_ast_nodes,
                "truncated_package": truncated_package,
                "raw_corpus_digest": digest_multiset(raw_hashes),
                "lexical_corpus_digest": digest_multiset(lexical_hashes),
                "concrete_ast_corpus_digest": digest_multiset(concrete_ast_hashes),
                "abstract_ast_corpus_digest": digest_multiset(abstract_ast_hashes),
                "files": file_records,
            }
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.time() - started, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    temporary.replace(output_path)
    return result


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str):
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def grouped_pairs(frame: pd.DataFrame, column: str, reason: str, eligible=None) -> list[dict]:
    rows = []
    source = frame if eligible is None else frame[eligible(frame)]
    for value, part in source.groupby(column, dropna=False):
        if pd.isna(value) or not str(value) or len(part) < 2:
            continue
        members = part.sort_values("graph_id").to_dict("records")
        anchor = members[0]
        for member in members[1:]:
            rows.append(
                {
                    "left_graph_id": anchor["graph_id"],
                    "right_graph_id": member["graph_id"],
                    "reason": reason,
                    "evidence_value": value,
                }
            )
    return rows


def build_overlap_candidates(results: list[dict]) -> list[dict]:
    file_frequency: Counter[str] = Counter()
    samples: dict[str, dict] = {}
    inverted: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for result in results:
        if result["status"] != "ok":
            continue
        useful = {}
        for file_record in result["files"]:
            file_hash = file_record["concrete_ast_hash"]
            weight = max(file_record["tokens"], file_record["ast_nodes"])
            if file_hash and weight >= 20:
                useful[file_hash] = max(useful.get(file_hash, 0), weight)
        samples[result["graph_id"]] = {
            "weight": sum(useful.values()),
            "files": useful,
            "record": result,
        }
        file_frequency.update(useful)
    for graph_id, sample in samples.items():
        for file_hash, weight in sample["files"].items():
            if file_frequency[file_hash] <= MAX_COMMON_FILE_FREQUENCY:
                inverted[file_hash].append((graph_id, weight))

    shared: Counter[tuple[str, str]] = Counter()
    for members in inverted.values():
        for (left, left_weight), (right, right_weight) in itertools.combinations(members, 2):
            pair = tuple(sorted((left, right)))
            shared[pair] += min(left_weight, right_weight)

    candidates = []
    for (left, right), shared_weight in shared.items():
        denominator = max(samples[left]["weight"], samples[right]["weight"], 1)
        similarity = shared_weight / denominator
        if shared_weight < MIN_SHARED_WEIGHT or similarity < MIN_OVERLAP_SIMILARITY:
            continue
        left_record, right_record = samples[left]["record"], samples[right]["record"]
        candidates.append(
            {
                "left_graph_id": left,
                "right_graph_id": right,
                "reason": "high_concrete_ast_file_overlap_audit_only",
                "similarity": round(similarity, 6),
                "shared_weight": shared_weight,
                "left_weight": samples[left]["weight"],
                "right_weight": samples[right]["weight"],
                "left_label": left_record["label"],
                "right_label": right_record["label"],
                "left_source": left_record["source_group"],
                "right_source": right_record["source_group"],
            }
        )
    return sorted(candidates, key=lambda row: (-row["similarity"], -row["shared_weight"]))


def analyze(results: list[dict], tasks: pd.DataFrame) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(CURRENT_SPLIT_CSV, low_memory=False)[["graph_id", "split"]].rename(
        columns={"split": "current_v10c_split"}
    )
    public_rows = [{key: value for key, value in row.items() if key != "files"} for row in results]
    frame = pd.DataFrame(public_rows).merge(current, on="graph_id", how="left", validate="one_to_one")
    frame.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    frame[frame["status"] != "ok"].to_csv(ERROR_CSV, index=False, encoding="utf-8")
    ok = frame[frame["status"] == "ok"].copy()

    strong_pairs = []
    strong_pairs += grouped_pairs(ok, "group_key", "same_normalized_package_name")
    strong_pairs += grouped_pairs(ok, "archive_sha256", "same_archive_sha256")
    strong_pairs += grouped_pairs(
        ok,
        "raw_corpus_digest",
        "same_complete_raw_python_corpus",
        lambda part: (part["python_file_count"] > 0) & (~part["truncated_package"]),
    )
    strong_pairs += grouped_pairs(
        ok,
        "lexical_corpus_digest",
        "same_complete_lexical_python_corpus",
        lambda part: (
            (part["python_file_count"] > 0)
            & (part["total_tokens"] >= MIN_STRONG_TOKENS)
            & (~part["truncated_package"])
        ),
    )
    strong_pairs += grouped_pairs(
        ok,
        "concrete_ast_corpus_digest",
        "same_complete_concrete_ast_corpus",
        lambda part: (
            (part["parsed_ast_file_count"] > 0)
            & (part["parsed_ast_file_count"] == part["python_file_count"])
            & (part["total_ast_nodes"] >= MIN_STRONG_AST_NODES)
            & (~part["truncated_package"])
        ),
    )
    pair_columns = ["left_graph_id", "right_graph_id", "reason", "evidence_value"]
    strong = pd.DataFrame(strong_pairs, columns=pair_columns).drop_duplicates(
        ["left_graph_id", "right_graph_id", "reason"]
    )
    strong.to_csv(STRONG_PAIR_CSV, index=False, encoding="utf-8")

    audit_pairs = grouped_pairs(
        ok,
        "abstract_ast_corpus_digest",
        "same_abstract_ast_corpus_audit_only",
        lambda part: (
            (part["parsed_ast_file_count"] > 0)
            & (part["parsed_ast_file_count"] == part["python_file_count"])
            & (part["total_ast_nodes"] >= MIN_CANDIDATE_AST_NODES)
            & (~part["truncated_package"])
        ),
    )
    overlap_pairs = build_overlap_candidates(results)
    candidate = pd.concat(
        [
            pd.DataFrame(audit_pairs, columns=pair_columns),
            pd.DataFrame(overlap_pairs),
        ],
        ignore_index=True,
    )
    if not candidate.empty:
        candidate = candidate.drop_duplicates(["left_graph_id", "right_graph_id", "reason"])
    candidate.to_csv(CANDIDATE_PAIR_CSV, index=False, encoding="utf-8")

    union_find = UnionFind(ok["graph_id"].astype(str).tolist())
    for row in strong.to_dict("records"):
        union_find.union(row["left_graph_id"], row["right_graph_id"])
    ok["strong_component_root"] = ok["graph_id"].astype(str).map(union_find.find)
    root_order = {
        root: index
        for index, root in enumerate(sorted(ok["strong_component_root"].unique()))
    }
    ok["strong_component_id"] = ok["strong_component_root"].map(
        lambda root: f"source_component_{root_order[root]:05d}"
    )
    components = (
        ok.groupby("strong_component_id")
        .agg(
            samples=("graph_id", "size"),
            labels=("label", lambda values: ",".join(sorted(set(values)))),
            sources=("source_group", lambda values: ",".join(sorted(set(values)))),
            package_names=("group_key", "nunique"),
            old_splits=("old_split", lambda values: ",".join(sorted(set(values)))),
            old_split_count=("old_split", "nunique"),
            current_v10c_splits=("current_v10c_split", lambda values: ",".join(sorted(set(values)))),
            current_v10c_split_count=("current_v10c_split", "nunique"),
            example_graph_id=("graph_id", "first"),
        )
        .reset_index()
        .sort_values(["samples", "strong_component_id"], ascending=[False, True])
    )
    components.to_csv(COMPONENT_CSV, index=False, encoding="utf-8")

    abstract_eligible = ok[
        (ok["parsed_ast_file_count"] > 0)
        & (ok["parsed_ast_file_count"] == ok["python_file_count"])
        & (~ok["truncated_package"])
    ].copy()
    abstract_rows = []
    for signature, part in abstract_eligible[
        abstract_eligible["total_ast_nodes"] >= MIN_CANDIDATE_AST_NODES
    ].groupby("abstract_ast_corpus_digest"):
        if len(part) < 2:
            continue
        abstract_rows.append(
            {
                "abstract_ast_corpus_digest": signature,
                "samples": len(part),
                "labels": ",".join(sorted(part["label"].unique())),
                "sources": ",".join(sorted(part["source_group"].unique())),
                "package_names": part["group_key"].nunique(),
                "python_file_count": part["python_file_count"].iloc[0],
                "total_ast_nodes": part["total_ast_nodes"].iloc[0],
                "old_splits": ",".join(sorted(part["old_split"].unique())),
                "old_split_count": part["old_split"].nunique(),
                "current_v10c_splits": ",".join(sorted(part["current_v10c_split"].unique())),
                "current_v10c_split_count": part["current_v10c_split"].nunique(),
                "members": " | ".join(sorted(part["graph_id"].astype(str))),
            }
        )
    abstract_groups = pd.DataFrame(abstract_rows)
    if not abstract_groups.empty:
        abstract_groups = abstract_groups.sort_values(
            ["samples", "total_ast_nodes"], ascending=[False, False]
        )
    abstract_groups.to_csv(ABSTRACT_GROUP_CSV, index=False, encoding="utf-8")

    sensitivity_rows = []
    for threshold in (40, 80, 120, 200, 500, 1000):
        groups = [
            part
            for _, part in abstract_eligible[
                abstract_eligible["total_ast_nodes"] >= threshold
            ].groupby("abstract_ast_corpus_digest")
            if len(part) > 1
        ]
        sensitivity_rows.append(
            {
                "minimum_ast_nodes": threshold,
                "multi_sample_groups": len(groups),
                "samples_in_groups": sum(len(part) for part in groups),
                "largest_group": max((len(part) for part in groups), default=1),
                "mixed_label_groups": sum(part["label"].nunique() > 1 for part in groups),
                "groups_cross_old_split": sum(part["old_split"].nunique() > 1 for part in groups),
                "groups_cross_current_v10c_split": sum(
                    part["current_v10c_split"].nunique() > 1 for part in groups
                ),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(THRESHOLD_CSV, index=False, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {"metric": "tasks", "value": len(tasks)},
            {"metric": "fingerprints_ok", "value": len(ok)},
            {"metric": "fingerprints_failed", "value": int((frame["status"] != "ok").sum())},
            {"metric": "strong_relation_rows", "value": len(strong)},
            {"metric": "audit_candidate_rows", "value": len(candidate)},
            {"metric": "strong_components", "value": len(components)},
            {"metric": "multi_sample_strong_components", "value": int((components["samples"] > 1).sum())},
            {"metric": "largest_strong_component", "value": int(components["samples"].max())},
            {"metric": "strong_components_cross_old_split", "value": int((components["old_split_count"] > 1).sum())},
            {
                "metric": "strong_components_cross_current_v10c_split",
                "value": int((components["current_v10c_split_count"] > 1).sum()),
            },
            {
                "metric": "abstract_ast_groups_at_80_nodes",
                "value": len(abstract_groups),
            },
            {
                "metric": "samples_in_abstract_ast_groups_at_80_nodes",
                "value": int(abstract_groups["samples"].sum()) if not abstract_groups.empty else 0,
            },
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nLARGEST STRONG COMPONENTS")
    print(components.head(20).to_string(index=False))
    print(f"\n[RESULT] {DIAG_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fingerprint", "analyze", "all"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tasks = pd.read_csv(TASK_CSV, low_memory=False)
    if args.max_samples:
        tasks = tasks.head(args.max_samples)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"fingerprint", "all"}:
        print("=" * 100)
        print("SOURCE CLONE FINGERPRINT DIAGNOSTIC")
        print("=" * 100)
        print(f"[INFO] tasks       : {len(tasks)}")
        print(f"[INFO] workers     : {args.workers}")
        print(f"[INFO] archive I/O : read Python members directly; no persistent unpacked source")
        print(f"[INFO] output      : {DIAG_DIR}")
        interrupted = False
        original_handler = signal.getsignal(signal.SIGINT)
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(fingerprint_one, record, args.force): record
                    for record in tasks.to_dict("records")
                }
                with tqdm(total=len(futures), unit="package", dynamic_ncols=True) as progress:
                    try:
                        for future in as_completed(futures):
                            future.result()
                            progress.update(1)
                    except KeyboardInterrupt:
                        interrupted = True
                        for future in futures:
                            future.cancel()
        finally:
            signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            print("[INFO] interrupted; completed cache files are reusable")
            return 130

    if args.stage in {"analyze", "all"}:
        results = []
        missing = []
        for graph_id in tasks["graph_id"].astype(str):
            path = cache_path(graph_id)
            if not valid_cache(path):
                missing.append(graph_id)
                continue
            with path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        if missing:
            raise RuntimeError(f"missing valid fingerprint cache files: {len(missing)}")
        analyze(results, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
