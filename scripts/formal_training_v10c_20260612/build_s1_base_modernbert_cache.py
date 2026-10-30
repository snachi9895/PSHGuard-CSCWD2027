#!/usr/bin/env python3
"""Build deterministic 512-token ModernBERT train/validation caches for S1-base."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SPEC_PATH = SCRIPT_PATH.with_name("s1_base_modernbert_screen_spec_v1.json")
EA4MP_SRC = PROJECT_ROOT / "baselines" / "ea4mp" / "src"
if str(EA4MP_SRC) not in sys.path:
    sys.path.insert(0, str(EA4MP_SRC))

from ea4mp_repro.behavior_canonicalization import canonicalize_behavior_text  # noqa: E402


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_seed1001_validation_20260815_v1"
)
CACHE_ROOT = OUTPUT_ROOT / "cache"
SUMMARY_PATH = CACHE_ROOT / "cache_audit.json"
LOCK_PATH = OUTPUT_ROOT / ".cache.lock"
MAX_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    marker = "/pypi_dataset_cutoff20260101/"
    normalized = value.replace("\\", "/")
    if marker not in normalized:
        raise RuntimeError(f"cannot safely rebase path: {value}")
    return (PROJECT_ROOT / normalized.split(marker, 1)[1]).resolve()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in result:
            raise RuntimeError("empty/duplicate sample_id")
        result[sample_id] = row
    return result


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_split(
    *,
    split: str,
    expected: int,
    tokenizer: Any,
    lengths: dict[str, dict[str, str]],
    s1_0_spec: dict[str, Any],
) -> dict[str, Any]:
    old_name = "validation" if split == "validation" else "train"
    input_spec = s1_0_spec["behavior_input"][old_name]
    sequence_rows = load_csv(resolve_project_path(input_spec["sequence_index"]))
    old_index = by_id(load_csv(resolve_project_path(input_spec["package_input_index"])))
    if len(sequence_rows) != expected:
        raise RuntimeError(f"{split} count drift")
    split_dir = CACHE_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    array_paths = {
        "input_ids": split_dir / "input_ids.npy",
        "attention_mask": split_dir / "attention_mask.npy",
        "labels": split_dir / "labels.npy",
    }
    temporary = {key: path.with_name(f".{path.name}.{os.getpid()}.tmp") for key, path in array_paths.items()}
    input_ids = np.lib.format.open_memmap(temporary["input_ids"], mode="w+", dtype=np.int32, shape=(expected, MAX_TOKENS))
    attention = np.lib.format.open_memmap(temporary["attention_mask"], mode="w+", dtype=np.uint8, shape=(expected, MAX_TOKENS))
    labels = np.lib.format.open_memmap(temporary["labels"], mode="w+", dtype=np.int8, shape=(expected,))
    meta_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(sequence_rows):
        sample_id = source["sample_id"]
        old = old_index[sample_id]
        length_row = lengths.get(sample_id)
        if length_row is None or length_row["split"] != split:
            raise RuntimeError(f"S1-0 length alignment failed: {sample_id}")
        behavior_missing = old["behavior_missing"] == "True"
        dfs_path = resolve_recorded_path(source["sample_directory"]) / "dfs_output.txt"
        if dfs_path.is_file():
            raw_text = dfs_path.read_text(encoding="utf-8")
        elif behavior_missing and old["dfs_sha256"] == "no_dfs_artifact":
            raw_text = ""
        else:
            raise RuntimeError(f"unexpected missing behavior artifact: {sample_id}")
        canonical, audit = canonicalize_behavior_text(raw_text, sample_id=sample_id)
        if audit.residual_sample_id_occurrences or audit.residual_raw_root_occurrences or audit.residual_source_tag_occurrences:
            raise RuntimeError(f"identity residual: {sample_id}")
        model_text = "" if behavior_missing else canonical
        encoded = tokenizer(
            model_text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_TOKENS,
            padding="max_length",
            return_attention_mask=True,
            verbose=False,
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        mask = np.asarray(encoded["attention_mask"], dtype=np.uint8)
        if ids.shape != (MAX_TOKENS,) or mask.shape != (MAX_TOKENS,):
            raise RuntimeError(f"token shape drift: {sample_id}")
        full_content = int(length_row["modern_content_tokens"])
        expected_nonpad = min(full_content + 2, MAX_TOKENS)
        if int(mask.sum()) != expected_nonpad:
            raise RuntimeError(f"token length replay failed: {sample_id}")
        if ids[0] != tokenizer.cls_token_id or ids[expected_nonpad - 1] != tokenizer.sep_token_id:
            raise RuntimeError(f"boundary token drift: {sample_id}")
        label = int(length_row["label_id"])
        input_ids[index] = ids
        attention[index] = mask
        labels[index] = label
        meta_rows.append(
            {
                "row_index": index,
                "sample_id": sample_id,
                "label_id": label,
                "source_group": length_row["source_group"],
                "component_id": length_row["component_id"],
                "full_modern_content_tokens": full_content,
                "nonpad_tokens": expected_nonpad,
                "truncated": int(full_content + 2 > MAX_TOKENS),
            }
        )
        if (index + 1) % 2000 == 0 or index + 1 == expected:
            elapsed = time.monotonic() - started
            print(f"[S1-base cache] {split} {index + 1:,}/{expected:,} rate={(index + 1) / max(elapsed, 1e-9):.1f} pkg/s", flush=True)
    input_ids.flush(); attention.flush(); labels.flush()
    del input_ids, attention, labels
    for key, path in array_paths.items():
        os.replace(temporary[key], path)
    meta_path = split_dir / "metadata.csv"
    meta_tmp = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
    with meta_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta_rows[0]))
        writer.writeheader(); writer.writerows(meta_rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(meta_tmp, meta_path)
    return {
        "packages": expected,
        "labels": dict(sorted({str(value): sum(int(row["label_id"]) == value for row in meta_rows) for value in (0, 1)}.items())),
        "truncated_packages": sum(int(row["truncated"]) for row in meta_rows),
        "files_sha256": {key + ".npy": sha256_file(path) for key, path in array_paths.items()} | {"metadata.csv": sha256_file(meta_path)},
    }


def run() -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.monotonic()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["artifact"]["status"] != "frozen_before_cache_build_and_gpu_training":
        raise RuntimeError("S1-base spec is not frozen")
    for key in ("s1_0_summary", "s1_0_lengths", "s1_0_spec", "outer_fold_manifest"):
        path = resolve_project_path(spec["inputs"][key])
        if sha256_file(path) != spec["inputs"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen input drift: {key}")
    s1_0_summary = json.loads(resolve_project_path(spec["inputs"]["s1_0_summary"]).read_text(encoding="utf-8"))
    if s1_0_summary["artifact"]["status"] != "pass" or s1_0_summary["scope"]["test_read"]:
        raise RuntimeError("S1-0 authorization invalid")
    s1_0_spec = json.loads(resolve_project_path(spec["inputs"]["s1_0_spec"]).read_text(encoding="utf-8"))
    lengths_rows = load_csv(resolve_project_path(spec["inputs"]["s1_0_lengths"]))
    lengths = by_id(lengths_rows)
    snapshot = resolve_project_path(spec["model"]["local_snapshot"])
    if sha256_file(snapshot / "model.safetensors") != spec["model"]["model_weight_sha256"]:
        raise RuntimeError("ModernBERT weights drift")
    if sha256_file(snapshot / "tokenizer.json") != spec["model"]["tokenizer_sha256"]:
        raise RuntimeError("ModernBERT tokenizer drift")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": build_split(split="train", expected=int(spec["scope"]["train_packages"]), tokenizer=tokenizer, lengths=lengths, s1_0_spec=s1_0_spec),
        "validation": build_split(split="validation", expected=int(spec["scope"]["validation_packages"]), tokenizer=tokenizer, lengths=lengths, s1_0_spec=s1_0_spec),
    }
    payload = {
        "artifact": {"name": "S1-base fixed-512 ModernBERT cache audit", "version": "1.0.0", "status": "pass"},
        "spec_sha256": sha256_file(SPEC_PATH),
        "splits": splits,
        "checks": {
            "S1_0_pass": True,
            "population_identity_exact": True,
            "full_lengths_replayed": True,
            "fixed_512_shapes": True,
            "labels_not_used_for_tokenization": True,
            "test_path_constructed": False,
            "test_read": False,
            "model_training": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "next_gate": "user_started_seed1001_validation_only_ModernBERT_training",
    }
    atomic_json(payload, SUMMARY_PATH)
    return payload


def status() -> None:
    if not SUMMARY_PATH.is_file():
        print("S1-base cache=PENDING test=LOCKED")
        return
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(
        f"S1-base cache={payload['artifact']['status'].upper()} "
        f"train={payload['splits']['train']['packages']:,} "
        f"validation={payload['splits']['validation']['packages']:,} "
        f"test=LOCKED result={SUMMARY_PATH}"
    )


def main() -> None:
    args = parse_args()
    if args.status:
        status(); return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
