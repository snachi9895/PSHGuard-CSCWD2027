#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from step02c_build_single_sample_behavior_graph_v7c import BehaviorChainGraphBuilderV7C


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
V7_ROOT = ROOT / "graph_artifacts" / "final_v7_behavior_chain_controlled_small"
V7C_ROOT = ROOT / "graph_artifacts" / "final_v7c_behavior_relevant_bounded_small"
IN_TASK_CSV = V7_ROOT / "step01_small_sample_tasks_v7.csv"

UNPACK_ROOT = V7C_ROOT / "unpacked"
GRAPH_DIR = V7C_ROOT / "graphs"
META_DIR = V7C_ROOT / "meta"
OUT_SUMMARY_CSV = V7C_ROOT / "step03c_graph_export_summary_v7c_small.csv"


def safe_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def safe_extract_tar(archive: Path, dest: Path):
    dest_resolved = dest.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"unsafe tar member: {member.name}")
        tf.extractall(dest)


def safe_extract_zip(archive: Path, dest: Path):
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"unsafe zip member: {member}")
        # Datadog malware samples use the conventional password "infected".
        # Passing a password is harmless for unencrypted ZIP members.
        zf.extractall(dest, pwd=b"infected")


def unpack_archive(archive_path: Path, out_dir: Path):
    complete_marker = out_dir / ".extract_complete"
    if complete_marker.exists():
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        safe_extract_tar(archive_path, out_dir)
    elif archive_path.name.endswith(".zip"):
        safe_extract_zip(archive_path, out_dir)
    else:
        raise ValueError(f"unsupported archive: {archive_path}")
    complete_marker.write_text("ok\n", encoding="ascii")


def export_one(row: pd.Series) -> dict:
    graph_id = str(row["graph_id"])
    archive_path = Path(str(row["archive_path"]))
    unpack_dir = UNPACK_ROOT / graph_id
    graph_path = GRAPH_DIR / f"{graph_id}.json"
    meta_path = META_DIR / f"{graph_id}.meta.json"

    result = {
        "graph_id": graph_id,
        "sample_id": row.get("sample_id", graph_id),
        "label": row.get("label", ""),
        "label_int": row.get("label_int", ""),
        "split": row.get("split", ""),
        "package_name": row.get("package_name", ""),
        "version": row.get("version", ""),
        "source_group": row.get("source_group", row.get("archive_group", "")),
        "group_key": row.get("group_key", ""),
        "archive_path": str(archive_path),
        "unpack_dir_v7c": str(unpack_dir),
        "graph_json_path_v7c": str(graph_path),
        "graph_meta_path_v7c": str(meta_path),
        "status": "ok",
        "error_message": "",
    }

    try:
        if not archive_path.exists():
            raise FileNotFoundError(f"archive_not_found: {archive_path}")
        unpack_archive(archive_path, unpack_dir)

        graph = BehaviorChainGraphBuilderV7C(unpack_dir).build()
        safe_write_json(graph_path, graph)

        meta = {
            **result,
            **graph.get("meta", {}),
        }
        safe_write_json(meta_path, meta)
        result.update({
            "node_count": graph["meta"]["num_nodes"],
            "edge_count": graph["meta"]["num_edges"],
            "module_count": graph["meta"]["node_counts"].get("module", 0),
            "ast_count": graph["meta"]["node_counts"].get("ast", 0),
            "param_count": graph["meta"]["node_counts"].get("param", 0),
            "original_py_files": graph["meta"]["original_py_files"],
            "kept_py_files": graph["meta"]["kept_py_files"],
            "skipped_py_files": graph["meta"]["skipped_py_files"],
            "truncated_files": graph["meta"]["truncated_files"],
            "truncated_ast": graph["meta"]["truncated_ast"],
            "truncated_param": graph["meta"]["truncated_param"],
        })
        for etype, count in graph["meta"]["edge_counts"].items():
            result[f"edge_count__{etype}"] = count
    except Exception as exc:
        result["status"] = "failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        try:
            if unpack_dir.exists() and not any(unpack_dir.iterdir()):
                shutil.rmtree(unpack_dir)
        except Exception:
            pass
    return result


def main():
    if not IN_TASK_CSV.exists():
        raise FileNotFoundError(f"missing task csv: {IN_TASK_CSV}")

    UNPACK_ROOT.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_TASK_CSV, low_memory=False)
    rows = []

    print("=" * 100)
    print("STEP03C EXPORT SMALL GRAPHS V7C BEHAVIOR-RELEVANT BOUNDED")
    print("=" * 100)
    print(f"[INFO] task csv  : {IN_TASK_CSV}")
    print(f"[INFO] output    : {V7C_ROOT}")
    print(f"[INFO] rows      : {len(df)}")
    print("-" * 100)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        result = export_one(row)
        rows.append(result)
        if i % 25 == 0 or i == len(df):
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"[PROGRESS] {i}/{len(df)} ok={ok} failed={len(rows)-ok}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")

    print("-" * 100)
    print(f"[RESULT] summary csv : {OUT_SUMMARY_CSV}")
    print(out_df["status"].value_counts(dropna=False).to_string())
    print("=" * 100)


if __name__ == "__main__":
    main()
