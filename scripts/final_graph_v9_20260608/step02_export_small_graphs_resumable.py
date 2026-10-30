#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import json
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent / "final_dataset_v3"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

import step03c_export_small_graphs_v7c as extract
from step01_build_single_sample_graph_v9 import GRAPH_VERSION, PayloadAwareBoundedBehaviorGraphBuilderV9


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TASK_CSV = ROOT / "graph_artifacts/final_v7c_behavior_relevant_bounded_diverse_small/step01d_diverse_source_tasks_v7c.csv"
OUT_ROOT = ROOT / "graph_artifacts/final_v9_payload_aware_bounded_small_20260608"
TEMP_ROOT = OUT_ROOT / "_temporary_unpacked"
GRAPH_DIR = OUT_ROOT / "graphs"
META_DIR = OUT_ROOT / "meta"
SUMMARY_CSV = OUT_ROOT / "step02_export_summary_v9_small.csv"
FAILURE_CSV = OUT_ROOT / "step02_failed_samples_v9_small.csv"
GRAPH_BUILDER = PayloadAwareBoundedBehaviorGraphBuilderV9


def safe_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, default=str)
    temp_path.replace(path)


def valid_existing(graph_path: Path, meta_path: Path) -> bool:
    try:
        with open(graph_path, "r", encoding="utf-8") as handle:
            graph = json.load(handle)
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return graph.get("graph_version") == GRAPH_VERSION and meta.get("control_profile") == GRAPH_VERSION
    except Exception:
        return False


def safe_cleanup_temp(unpack_dir: Path):
    resolved_root = TEMP_ROOT.resolve()
    resolved_target = unpack_dir.resolve()
    if resolved_target.parent != resolved_root:
        raise RuntimeError(f"refuse_cleanup_outside_temp_root: {resolved_target}")
    if unpack_dir.exists():
        shutil.rmtree(unpack_dir)


def export_one(record: dict, force: bool = False) -> dict:
    started = time.time()
    graph_id = str(record["graph_id"])
    archive_path = Path(str(record["archive_path"]))
    unpack_dir = TEMP_ROOT / graph_id
    graph_path = GRAPH_DIR / f"{graph_id}.json"
    meta_path = META_DIR / f"{graph_id}.meta.json"
    result = {
        "graph_id": graph_id,
        "sample_id": record.get("sample_id", graph_id),
        "label": record.get("label", ""),
        "label_int": record.get("label_int", ""),
        "split": record.get("split", ""),
        "package_name": record.get("package_name", ""),
        "version": record.get("version", ""),
        "source_group": record.get("source_group", ""),
        "group_key": record.get("group_key", ""),
        "archive_path": str(archive_path),
        "graph_json_path_v9": str(graph_path),
        "graph_meta_path_v9": str(meta_path),
        "temporary_unpack_removed": False,
        "status": "ok",
        "error_message": "",
    }
    try:
        if not force and valid_existing(graph_path, meta_path):
            result["status"] = "skipped_existing"
            return result
        if not archive_path.exists():
            raise FileNotFoundError(f"archive_not_found: {archive_path}")
        safe_cleanup_temp(unpack_dir)
        extract.unpack_archive(archive_path, unpack_dir)
        graph = GRAPH_BUILDER(unpack_dir).build()
        safe_write_json(graph_path, graph)
        meta = {**result, **graph["meta"]}
        safe_write_json(meta_path, meta)
        result.update({
            "node_count": graph["meta"]["num_nodes"],
            "edge_count": graph["meta"]["num_edges"],
            "module_count": graph["meta"]["node_counts"].get("module", 0),
            "ast_count": graph["meta"]["node_counts"].get("ast", 0),
            "param_count": graph["meta"]["node_counts"].get("param", 0),
            "event_count": graph["meta"]["node_counts"].get("event", 0),
            "parsed_py_files": graph["meta"].get("parsed_py_files", 0),
            "kept_py_files": graph["meta"].get("kept_py_files", 0),
        })
    except Exception as exc:
        result["status"] = "failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            safe_cleanup_temp(unpack_dir)
            result["temporary_unpack_removed"] = not unpack_dir.exists()
        except Exception as exc:
            result["status"] = "failed_cleanup"
            result["error_message"] = "; ".join(filter(None, [
                result["error_message"], f"cleanup_failed: {type(exc).__name__}: {exc}",
            ]))
        result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def latest_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["graph_id"], keep="last")


def write_progress(rows: list[dict]):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = latest_rows(rows)
    frame.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    failed = frame[frame["status"].astype(str).str.startswith("failed")] if not frame.empty else frame
    failed.to_csv(FAILURE_CSV, index=False, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    # The WSL virtual filesystem may report its expandable VHD capacity rather
    # than the Windows host drive's real free space. This project VHD lives on
    # D:, so protect /mnt/d when it is available.
    disk_guard_path = Path("/mnt/d") if Path("/mnt/d").exists() else ROOT
    free_gb = shutil.disk_usage(disk_guard_path).free / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"free disk {free_gb:.1f} GB is below --min-free-gb {args.min_free_gb:.1f}")

    for path in [OUT_ROOT, TEMP_ROOT, GRAPH_DIR, META_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(TASK_CSV, low_memory=False)
    if args.max_samples:
        frame = frame.head(args.max_samples)
    records = frame.to_dict("records")
    rows: list[dict] = (
        pd.read_csv(SUMMARY_CSV, low_memory=False).to_dict("records")
        if SUMMARY_CSV.exists() else []
    )
    interrupted = False

    print("=" * 100)
    print("V9 SMALL RESUMABLE GRAPH EXPORT")
    print("=" * 100)
    print(f"[INFO] tasks       : {len(records)}")
    print(f"[INFO] workers     : {args.workers}")
    print(f"[INFO] disk guard  : {disk_guard_path}")
    print(f"[INFO] free disk   : {free_gb:.1f} GB")
    print(f"[INFO] output      : {OUT_ROOT}")
    print("[INFO] cleanup     : remove only per-sample directories under _temporary_unpacked")
    print("[INFO] resume      : valid existing V9 graph/meta pairs are skipped")

    original_handler = signal.getsignal(signal.SIGINT)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(export_one, record, args.force): record for record in records}
            with tqdm(total=len(futures), unit="graph", dynamic_ncols=True) as progress:
                try:
                    for future in as_completed(futures):
                        rows.append(future.result())
                        progress.update(1)
                        if progress.n % 10 == 0:
                            write_progress(rows)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\n[INFO] interrupt received; writing completed results before exit...")
                    for future in futures:
                        future.cancel()
    finally:
        signal.signal(signal.SIGINT, original_handler)
        write_progress(rows)

    final_frame = latest_rows(rows)
    statuses = final_frame["status"].value_counts(dropna=False)
    print(statuses.to_string())
    print(f"[RESULT] {SUMMARY_CSV}")
    return 130 if interrupted else int(final_frame["status"].astype(str).str.startswith("failed").any())


if __name__ == "__main__":
    raise SystemExit(main())
