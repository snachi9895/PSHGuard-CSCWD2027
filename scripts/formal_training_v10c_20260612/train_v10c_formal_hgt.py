#!/usr/bin/env python3
"""Formal full-dataset V10c HGT training with early stopping and epoch resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import warnings
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader


warnings.filterwarnings(
    "ignore",
    message=r"The usage of `scatter\(reduce='max'\)` can be accelerated via the 'torch-scatter' package.*",
    category=UserWarning,
    module=r"torch_geometric\.utils\._scatter",
)

HERE = Path(__file__).resolve().parent
V9_DIR = HERE.parent / "final_graph_v9_20260608"
if str(V9_DIR) not in sys.path:
    sys.path.insert(0, str(V9_DIR))

import train_v9_small_hgt_readout as architecture
from audit_binary_operating_points import (
    atomic_write_json as atomic_operating_point_json,
    audit_prediction_csv,
)


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = ROOT / "experiment_outputs/train_final_v10c_filtered700k_20260612"
V1_INDEX_CSV = (
    TRAIN_ROOT
    / "step05_training_pt_index_v10c_source_clone_filtered700k_graph_eligible.csv"
)
V2_INDEX_CSV = (
    TRAIN_ROOT
    / "split_v2_strict40_20260722"
    / "step01_training_pt_index_v10c_v2_strict40_graph_eligible.csv"
)
V2_FROZEN_SPLIT_DIR = (
    ROOT
    / "manifests"
    / "final_source_clone_aware_split_202606"
    / "final_filtered700k_graph_eligible_v2_strict40_20260722"
)
V2_FROZEN_TRAIN_CSV = V2_FROZEN_SPLIT_DIR / "filtered700k_graph_eligible_v2_train.csv"
V2_FROZEN_VALIDATION_CSV = V2_FROZEN_SPLIT_DIR / "filtered700k_graph_eligible_v2_val.csv"
V2_FROZEN_DEVELOPMENT_MANIFESTS = {
    "train": {
        "path": V2_FROZEN_TRAIN_CSV,
        "rows": 16_613,
        "sha256": "8635d5e35c28ab750d57b3cc9eedbbc390b6f5f024ba269a94e23856faf0d882",
    },
    "val": {
        "path": V2_FROZEN_VALIDATION_CSV,
        "rows": 2_137,
        "sha256": "80dfd90b01d0e19f04fdde640a87b472c9491b3f31c677ad336dc3bc18583d18",
    },
}
V2_FROZEN_SPLIT_VERSION = "final_filtered700k_graph_eligible_v2_strict40_20260722"
SPLIT_CONTRACTS = {
    "v1": {
        "index_csv": V1_INDEX_CSV,
        "output_root": TRAIN_ROOT / "formal_hgt_runs",
        "feature_profile": "v10c_file16_source_clone_graph_eligible_normalized_v1",
        "expected_index_sha256": "b70186b464f01d2e3a8d07521823b1dbf4328d0a5af2800ec81ad23715bffb2d",
    },
    "v2_strict40": {
        "index_csv": V2_INDEX_CSV,
        "output_root": TRAIN_ROOT / "split_v2_strict40_20260722" / "formal_hgt_runs",
        "feature_profile": "v10c_file16_source_clone_graph_eligible_strict40_v2",
        "expected_index_sha256": "f389a4651c3056627fd544a86451565e9a6d2bbb90509710ee12b4793f0c8e1d",
    },
}

EXPECTED_ROWS = 20_878
EXPECTED_SPLITS = {"train": 16_613, "val": 2_137, "test": 2_128}
CHECKPOINT_METRIC_CHOICES = ("f1", "ap")
IN_DIMS = {"module": 4, "ast": 12, "param": 29, "event": 18}
FORMAL_MODES = [
    "ast_only",
    "ast_structure",
    "ast_event_only",
    "ast_event",
    "ast_event_flow",
    "ast_event_param",
    "ast_event_param_context_no_structure",
    "ast_event_param_context",
    "full",
    "full_param_context",
]
COMPONENT_WEIGHTING_CHOICES = (
    "none",
    "inverse_sqrt",
    "inverse_sqrt_source_preserving",
)
OPTIMIZER_CHOICES = ("adam", "adamw")


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_architecture(seed: int, index_csv: Path) -> None:
    base = architecture.base
    base.TRAIN_ROOT = TRAIN_ROOT
    base.IN_CSV = index_csv
    base.IN_DIMS.update(IN_DIMS)
    base.SEED = seed
    base.HGTV8B = architecture.TypeBalancedHGTV9
    base.GraphDatasetV8B = architecture.ParamAblationDataset
    base.MODE_NODE_TYPES["full_param_context"] = list(base.MODE_NODE_TYPES["full"])
    base.MODE_EDGES["full_param_context"] = set(base.MODE_EDGES["full"])
    architecture.POOL_NODE_TYPES["full_param_context"] = ["module", "ast", "event"]
    base.MODE_NODE_TYPES["ast_event_param_context_no_structure"] = ["module", "ast", "param", "event"]
    base.MODE_EDGES["ast_event_param_context_no_structure"] = (
        set(base.AST_EDGES) | set(base.EVENT_LINK_EDGES) | set(architecture.PARAM_EDGES)
    )
    architecture.POOL_NODE_TYPES["ast_event_param_context_no_structure"] = ["module", "ast", "event"]
    architecture.PARAM_FEATURE_ABLATION = "none"
    architecture.PARAM_EDGE_ABLATION = "none"


def read_index(index_csv: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not index_csv.is_file():
        raise FileNotFoundError(index_csv)
    frame = pd.read_csv(index_csv, low_memory=False)
    if len(frame) != EXPECTED_ROWS or frame["graph_id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_ROWS} unique graph rows, got rows={len(frame)} "
            f"unique={frame['graph_id'].nunique()}"
        )
    if set(frame["status"]) != {"ok"}:
        raise RuntimeError(f"unexpected index statuses:\n{frame['status'].value_counts(dropna=False)}")
    split_counts = frame["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLITS:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    if frame["pt_path_v9"].isna().any():
        raise RuntimeError("the formal index contains missing PT paths")
    parts = {
        split: frame.loc[frame["split"] == split].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    return frame, parts


def read_frozen_v2_development_manifests() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read the P1 development population without opening a test-bearing index.

    This is deliberately separate from :func:`read_index`: AP checkpoint
    selection is a development-only protocol, so even transiently reading test
    rows or constructing a test dataset would violate its contract.
    """

    required_columns = {
        "graph_id",
        "sample_id",
        "label",
        "label_int",
        "split",
        "status",
        "pt_path_v9",
        "source_group",
        "source_clone_component_id",
        "split_version",
    }
    parts: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        specification = V2_FROZEN_DEVELOPMENT_MANIFESTS[split]
        path = Path(specification["path"])
        if "test" in path.name.lower():
            raise RuntimeError(f"P1 development-only guard refused test manifest: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        expected_sha256 = str(specification["sha256"])
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"frozen V2 {split} manifest SHA256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )

        part = pd.read_csv(path, low_memory=False)
        missing_columns = sorted(required_columns.difference(part.columns))
        if missing_columns:
            raise RuntimeError(f"{path} is missing required columns: {missing_columns}")
        expected_rows = int(specification["rows"])
        if len(part) != expected_rows:
            raise RuntimeError(
                f"frozen V2 {split} row-count mismatch: expected={expected_rows} actual={len(part)}"
            )
        observed_splits = sorted(part["split"].dropna().astype(str).unique().tolist())
        if observed_splits != [split]:
            raise RuntimeError(
                f"frozen V2 {split} manifest has split values {observed_splits}, expected {[split]}"
            )
        observed_versions = sorted(part["split_version"].dropna().astype(str).unique().tolist())
        if observed_versions != [V2_FROZEN_SPLIT_VERSION]:
            raise RuntimeError(
                f"frozen V2 {split} manifest has split_version values {observed_versions}, "
                f"expected {[V2_FROZEN_SPLIT_VERSION]}"
            )
        observed_statuses = sorted(part["status"].dropna().astype(str).unique().tolist())
        if observed_statuses != ["ok"] or part["status"].isna().any():
            raise RuntimeError(
                f"frozen V2 {split} manifest has statuses {observed_statuses}, expected ['ok']"
            )

        for identifier in ("sample_id", "graph_id"):
            values = part[identifier].astype("string")
            if values.isna().any() or values.str.strip().eq("").any():
                raise RuntimeError(f"frozen V2 {split} manifest contains empty {identifier}")
            if values.duplicated().any():
                raise RuntimeError(f"frozen V2 {split} manifest contains duplicate {identifier}")
        labels = sorted(part["label_int"].dropna().astype(int).unique().tolist())
        if labels != [0, 1] or part["label_int"].isna().any():
            raise RuntimeError(
                f"frozen V2 {split} manifest has label_int values {labels}, expected [0, 1]"
            )

        pt_paths = part["pt_path_v9"].astype("string")
        if pt_paths.isna().any() or pt_paths.str.strip().eq("").any():
            raise RuntimeError(f"frozen V2 {split} manifest contains empty PT paths")
        if pt_paths.duplicated().any():
            raise RuntimeError(f"frozen V2 {split} manifest contains duplicate PT paths")
        non_absolute = [value for value in pt_paths.astype(str) if not Path(value).is_absolute()]
        if non_absolute:
            raise RuntimeError(
                f"frozen V2 {split} manifest contains non-absolute PT paths; first={non_absolute[0]}"
            )
        missing_pt = [value for value in pt_paths.astype(str) if not Path(value).is_file()]
        if missing_pt:
            raise RuntimeError(
                f"frozen V2 {split} manifest contains {len(missing_pt)} missing PT files; "
                f"first={missing_pt[0]}"
            )
        parts[split] = part.copy().reset_index(drop=True)

    frame = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    for identifier in ("sample_id", "graph_id", "pt_path_v9"):
        if frame[identifier].astype(str).duplicated().any():
            raise RuntimeError(f"frozen V2 train/validation manifests overlap on {identifier}")
    return frame, parts


def attach_component_weights(
    parts: dict[str, pd.DataFrame],
    scheme: str,
    split_contract: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if scheme not in COMPONENT_WEIGHTING_CHOICES:
        raise ValueError(f"unsupported component weighting scheme: {scheme}")
    if scheme != "none" and split_contract != "v2_strict40":
        raise ValueError("component weighting is only authorized with --split-contract v2_strict40")

    weighted_parts = {name: part.copy() for name, part in parts.items()}
    for part in weighted_parts.values():
        part["_component_weight"] = 1.0

    train = weighted_parts["train"]
    if scheme == "none":
        return weighted_parts, {
            "scheme": scheme,
            "normalization": "identity",
            "train_weight_mean": 1.0,
            "train_weight_min": 1.0,
            "train_weight_max": 1.0,
        }

    component_column = "source_clone_component_id"
    required_columns = {component_column, "label"}
    if scheme == "inverse_sqrt_source_preserving":
        required_columns.add("source_group")
    missing_columns = required_columns - set(train.columns)
    if missing_columns:
        raise RuntimeError(f"{scheme} requires columns: {sorted(missing_columns)}")
    if train[list(required_columns)].isna().any().any():
        raise RuntimeError(f"{scheme} requires complete values for: {sorted(required_columns)}")

    if scheme == "inverse_sqrt":
        size_group_columns = [component_column]
        normalization_group_columns = ["label"]
        normalization = "mean_one_within_label"
        formula = "component_size^-0.5"
    else:
        # Keep the observed source mixture unchanged while balancing clone
        # families inside each source.  This avoids an unintended source shift
        # when one source has many more independent components than another.
        size_group_columns = ["label", "source_group", component_column]
        normalization_group_columns = ["label", "source_group"]
        normalization = "mean_one_within_label_and_source"
        formula = "within_source_component_size^-0.5"

    component_sizes = (
        train.groupby(size_group_columns)[component_column]
        .transform("size")
        .astype(float)
    )
    raw_weights = component_sizes.pow(-0.5)
    normalized = raw_weights.copy()
    for group_key, indexes in train.groupby(normalization_group_columns).groups.items():
        group_mean = float(raw_weights.loc[indexes].mean())
        if not np.isfinite(group_mean) or group_mean <= 0:
            raise RuntimeError(
                f"invalid component-weight mean for group={group_key}: {group_mean}"
            )
        normalized.loc[indexes] = raw_weights.loc[indexes] / group_mean

    train["_component_weight"] = normalized.astype(float)
    for group_key, indexes in train.groupby(normalization_group_columns).groups.items():
        observed_mean = float(train.loc[indexes, "_component_weight"].mean())
        if not np.isclose(observed_mean, 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"component weights must have mean 1 within group={group_key}, "
                f"got {observed_mean}"
            )

    label_stats = {}
    source_stats = {}
    for label, indexes in train.groupby("label").groups.items():
        label_part = train.loc[indexes]
        label_counts = (
            label_part.groupby(component_column).size().sort_values(ascending=False)
        )
        unweighted_component_mass = label_counts.astype(float)
        weighted_component_mass = (
            label_part.groupby(component_column)["_component_weight"]
            .sum()
            .sort_values(ascending=False)
        )
        label_stats[str(label)] = {
            "samples": int(len(label_part)),
            "components": int(len(label_counts)),
            "largest_component_samples": int(label_counts.iloc[0]),
            "unweighted_largest_component_mass": float(
                unweighted_component_mass.iloc[0] / unweighted_component_mass.sum()
            ),
            "weighted_largest_component_mass": float(
                weighted_component_mass.iloc[0] / weighted_component_mass.sum()
            ),
            "unweighted_top3_component_mass": float(
                unweighted_component_mass.iloc[:3].sum()
                / unweighted_component_mass.sum()
            ),
            "weighted_top3_component_mass": float(
                weighted_component_mass.iloc[:3].sum()
                / weighted_component_mass.sum()
            ),
        }
        # Preserve the historical field names for old analysis consumers.
        label_stats[str(label)]["inverse_sqrt_largest_component_mass"] = (
            label_stats[str(label)]["weighted_largest_component_mass"]
        )
        label_stats[str(label)]["inverse_sqrt_top3_component_mass"] = (
            label_stats[str(label)]["weighted_top3_component_mass"]
        )

        label_total_unweighted = float(len(label_part))
        label_total_weighted = float(label_part["_component_weight"].sum())
        label_source_stats = {}
        for source_group, source_indexes in label_part.groupby("source_group").groups.items():
            source_part = label_part.loc[source_indexes]
            source_component_mass = (
                source_part.groupby(component_column)["_component_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            label_source_stats[str(source_group)] = {
                "samples": int(len(source_part)),
                "components": int(source_part[component_column].nunique()),
                "unweighted_mass_share_within_label": float(
                    len(source_part) / label_total_unweighted
                ),
                "weighted_mass_share_within_label": float(
                    source_part["_component_weight"].sum() / label_total_weighted
                ),
                "weight_min": float(source_part["_component_weight"].min()),
                "weight_max": float(source_part["_component_weight"].max()),
                "weighted_largest_component_mass_within_source": float(
                    source_component_mass.iloc[0] / source_component_mass.sum()
                ),
                "weighted_top3_component_mass_within_source": float(
                    source_component_mass.iloc[:3].sum()
                    / source_component_mass.sum()
                ),
            }
        source_stats[str(label)] = label_source_stats

    return weighted_parts, {
        "scheme": scheme,
        "normalization": normalization,
        "formula": formula,
        "train_weight_mean": float(train["_component_weight"].mean()),
        "train_weight_min": float(train["_component_weight"].min()),
        "train_weight_max": float(train["_component_weight"].max()),
        "labels": label_stats,
        "sources": source_stats,
    }


class FormalDataset(architecture.ParamAblationDataset):
    """Attach deterministic per-sample training weights to each graph."""

    def __getitem__(self, index):
        data = super().__getitem__(index)
        row = self.df.iloc[index]
        value = float(row.get("_component_weight", 1.0))
        data.component_weight = torch.tensor([value], dtype=torch.float32)
        data.source_clone_component_id = str(row.get("source_clone_component_id", ""))
        return data


def make_loaders(
    parts: dict[str, pd.DataFrame],
    mode: str,
    batch_size: int,
    num_workers: int,
    train_generator: torch.Generator,
    device: torch.device,
) -> dict[str, DataLoader]:
    loaders = {}
    for split, part in parts.items():
        loaders[split] = DataLoader(
            FormalDataset(part, mode),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=(device.type == "cuda"),
            generator=train_generator if split == "train" else None,
        )
    return loaders


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return architecture.base.common.binary_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_pred_path: Path | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(-1)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * y.size(0)

        prob = F.softmax(out, dim=1)[:, 1]
        logit_margin = out[:, 1] - out[:, 0]
        pred = out.argmax(dim=1)
        y_cpu = y.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        prob_cpu = prob.detach().cpu().tolist()
        logit_margin_cpu = logit_margin.detach().cpu().tolist()
        y_true.extend(y_cpu)
        y_pred.extend(pred_cpu)
        y_score.extend(prob_cpu)

        if save_pred_path is not None:
            sample_ids = batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
            graph_ids = batch.graph_id if isinstance(batch.graph_id, list) else [batch.graph_id]
            package_names = batch.package_name if isinstance(batch.package_name, list) else [batch.package_name]
            versions = batch.version if isinstance(batch.version, list) else [batch.version]
            source_groups = batch.source_group if isinstance(batch.source_group, list) else [batch.source_group]
            component_ids = (
                batch.source_clone_component_id
                if isinstance(batch.source_clone_component_id, list)
                else [batch.source_clone_component_id]
            )
            for index in range(len(y_cpu)):
                rows.append(
                    {
                        "sample_id": sample_ids[index] if index < len(sample_ids) else "",
                        "graph_id": graph_ids[index] if index < len(graph_ids) else "",
                        "package_name": package_names[index] if index < len(package_names) else "",
                        "version": versions[index] if index < len(versions) else "",
                        "source_group": source_groups[index] if index < len(source_groups) else "",
                        "source_clone_component_id": (
                            component_ids[index] if index < len(component_ids) else ""
                        ),
                        "y_true": int(y_cpu[index]),
                        "y_pred": int(pred_cpu[index]),
                        "prob_malicious": float(prob_cpu[index]),
                        "logit_margin": float(logit_margin_cpu[index]),
                    }
                )

    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(y_true), 1)
    if len(set(y_true)) != 2:
        raise RuntimeError("ranking metrics require both binary classes in the evaluation loader")
    metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    if save_pred_path is not None:
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = save_pred_path.with_suffix(save_pred_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(save_pred_path)
    return metrics


def checkpoint_selection_key(metrics: dict, checkpoint_metric: str, epoch: int) -> tuple[float, ...]:
    """Return the frozen checkpoint-ordering key (larger is always better)."""

    if checkpoint_metric == "f1":
        # Historical behavior: strict F1 improvement only; ties retain the
        # already-saved (therefore earlier) checkpoint.
        return (float(metrics["f1"]),)
    if checkpoint_metric == "ap":
        return (
            float(metrics["average_precision"]),
            float(metrics["roc_auc"]),
            -float(metrics["loss"]),
            -float(epoch),
        )
    raise ValueError(f"unsupported checkpoint metric: {checkpoint_metric}")


def checkpoint_metric_value(metrics: dict, checkpoint_metric: str) -> float:
    if checkpoint_metric == "f1":
        return float(metrics["f1"])
    if checkpoint_metric == "ap":
        return float(metrics["average_precision"])
    raise ValueError(f"unsupported checkpoint metric: {checkpoint_metric}")


def capture_rng_state(train_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator": train_generator.get_state(),
    }


def restore_rng_state(state: dict, train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading a full checkpoint with map_location="cuda" also moves these
    # serialized RNG byte tensors to CUDA. PyTorch RNG setters require CPU
    # ByteTensors, so normalize them before restoring a resumed run.
    torch.set_rng_state(state["torch_cpu"].detach().cpu().to(torch.uint8).contiguous())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = [
            value.detach().cpu().to(torch.uint8).contiguous()
            for value in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    train_generator.set_state(
        state["train_generator"].detach().cpu().to(torch.uint8).contiguous()
    )


def config_from_args(
    args: argparse.Namespace,
    index_csv: Path,
    index_sha256: str,
    feature_profile: str,
    component_weight_stats: dict,
    input_rows: int,
    input_split_counts: dict[str, int],
) -> dict:
    development_only = args.checkpoint_metric == "ap"
    development_contract = {
        "enabled": development_only,
        "name": "p1_frozen_v2_train_plus_original_validation" if development_only else "historical",
        "combined_index_read": not development_only,
        "test_sample_manifest_read": False if development_only else None,
        "test_rows_read": False if development_only else None,
        "test_loader_created": False if development_only else None,
        "train_manifest": str(V2_FROZEN_TRAIN_CSV) if development_only else None,
        "train_manifest_sha256": (
            str(V2_FROZEN_DEVELOPMENT_MANIFESTS["train"]["sha256"])
            if development_only
            else None
        ),
        "validation_manifest": str(V2_FROZEN_VALIDATION_CSV) if development_only else None,
        "validation_manifest_sha256": (
            str(V2_FROZEN_DEVELOPMENT_MANIFESTS["val"]["sha256"])
            if development_only
            else None
        ),
        "allowed_splits": ["train", "val"] if development_only else ["train", "val", "test"],
        "test_evaluation_authorized": False if development_only else not args.validation_only,
    }
    return {
        "mode": args.mode,
        "run_label": args.run_label,
        "seed": args.seed,
        "determinism": args.determinism,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clip": args.gradient_clip,
        "benign_class_weight": 1.0,
        "malicious_class_weight": args.malicious_class_weight,
        "component_weighting": args.component_weighting,
        "component_weight_stats": component_weight_stats,
        "progress_every": args.progress_every,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_tie_break": (
            ["average_precision", "roc_auc", "lower_validation_loss", "earlier_epoch"]
            if args.checkpoint_metric == "ap"
            else ["f1", "earlier_epoch_on_tie"]
        ),
        "early_stopping_metric": args.checkpoint_metric,
        "development_only_contract": development_contract,
        "readout": "type_balanced",
        "split_contract": args.split_contract,
        "feature_profile": feature_profile,
        "index_csv": str(index_csv),
        "index_sha256": index_sha256,
        "input_rows": int(input_rows),
        "input_split_counts": {key: int(value) for key, value in input_split_counts.items()},
        "trainer_script_sha256": sha256_file(Path(__file__)),
    }


def manifest(config: dict, run_tag: str, device: torch.device) -> dict:
    return {
        "run_tag": run_tag,
        **config,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "formal_index_rows": config["input_rows"],
        "formal_split_counts": config["input_split_counts"],
    }


def validate_resume_config(saved: dict, current: dict) -> None:
    keys = [
        "mode",
        "seed",
        "determinism",
        "batch_size",
        "max_epochs",
        "min_epochs",
        "patience",
        "min_delta",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "hidden_dim",
        "gradient_accumulation_steps",
        "gradient_clip",
        "malicious_class_weight",
        "component_weighting",
        "checkpoint_metric",
        "readout",
        "feature_profile",
        "index_csv",
    ]
    if current.get("checkpoint_metric") == "ap":
        # No AP-mode checkpoint predates P1.  Fail closed if its implementation
        # changes between interrupted sessions; historical F1 resumes remain
        # compatible with checkpoints that did not record a script hash.
        keys.append("trainer_script_sha256")
    mismatches = {}
    for key in keys:
        saved_value = saved.get(key)
        current_value = current.get(key)
        if key == "component_weighting" and saved_value is None:
            saved_value = "none"
        if key == "optimizer" and saved_value is None:
            saved_value = "adam"
        if key == "weight_decay" and saved_value is None:
            saved_value = 0.0
        if key == "checkpoint_metric" and saved_value is None:
            saved_value = "f1"
        if saved_value != current_value:
            mismatches[key] = (saved_value, current_value)
    # V1 checkpoints created before the V2-safe CLI do not contain these two
    # fields. Preserve their resume compatibility while enforcing both fields
    # for every newly created V2 checkpoint.
    for key in ("split_contract", "index_sha256"):
        if key in saved and saved.get(key) != current.get(key):
            mismatches[key] = (saved.get(key), current.get(key))
    if mismatches:
        raise RuntimeError(f"resume configuration mismatch: {mismatches}")


def representative_rows(parts: dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    rows = []
    for split in (name for name in ("train", "val", "test") if name in parts):
        part = parts[split]
        if mode in {"full", "full_param_context"} or "param" in mode:
            candidates = part.loc[(part["event_count"] > 0) & (part["param_count"] > 0)]
        elif "event" in mode:
            candidates = part.loc[part["event_count"] > 0]
        else:
            candidates = part
        rows.append((candidates if not candidates.empty else part).iloc[[0]])
    return pd.concat(rows, ignore_index=True)


def make_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[optimizer_name]
    return optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_preflight(args: argparse.Namespace, parts: dict[str, pd.DataFrame], device: torch.device) -> None:
    missing_pt = sum(not Path(str(path)).is_file() for path in pd.concat(parts.values())["pt_path_v9"])
    if missing_pt:
        raise RuntimeError(f"formal index has {missing_pt} missing PT files")

    modes = [args.mode] if args.mode else ["ast_event", "full"]
    for mode in modes:
        sample_rows = representative_rows(parts, mode)
        loader = DataLoader(architecture.ParamAblationDataset(sample_rows, mode), batch_size=1, shuffle=False)
        schema = architecture.base.graph_schema(mode)
        model = architecture.TypeBalancedHGTV9(args.hidden_dim, schema, mode).to(device)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(batch.to(device))
                if out.shape != (1, 2) or not torch.isfinite(out).all():
                    raise RuntimeError(f"invalid preflight output for mode={mode}: shape={tuple(out.shape)}")
        print(f"[PREFLIGHT] mode={mode} representative forward passed ({len(sample_rows)} graphs)")

        gradient_loader = DataLoader(
            FormalDataset(sample_rows, mode),
            batch_size=len(sample_rows),
            shuffle=False,
        )
        model.train()
        optimizer = make_optimizer(
            model,
            args.optimizer,
            args.learning_rate,
            args.weight_decay,
        )
        batch = next(iter(gradient_loader)).to(device)
        logits = model(batch)
        labels = batch.y.view(-1)
        raw_loss = F.cross_entropy(logits, labels, reduction="none")
        component_weights = batch.component_weight.view(-1)
        class_weights = torch.tensor(
            [1.0, args.malicious_class_weight],
            dtype=torch.float32,
            device=device,
        )
        loss = (raw_loss * class_weights[labels] * component_weights).mean()
        loss.backward()
        finite_gradients = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not finite_gradients or not all(bool(value) for value in finite_gradients):
            raise RuntimeError(f"non-finite preflight gradients for mode={mode}")
        optimizer.step()
        print(
            f"[PREFLIGHT] mode={mode} {args.optimizer} backward/step passed "
            f"(weight_decay={args.weight_decay})"
        )
    print("[PREFLIGHT] formal index, PT paths, schemas, and representative forwards passed")


def train(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        raise ValueError("--run-label may contain only letters, digits, underscores, and hyphens")
    if args.min_epochs > args.max_epochs:
        raise ValueError("--min-epochs cannot exceed --max-epochs")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.checkpoint_metric == "ap":
        if args.split_contract != "v2_strict40":
            raise ValueError("--checkpoint-metric ap requires --split-contract v2_strict40")
        if not args.validation_only:
            raise ValueError("--checkpoint-metric ap requires --validation-only")

    contract = SPLIT_CONTRACTS[args.split_contract]
    output_root = Path(contract["output_root"])
    feature_profile = str(contract["feature_profile"])
    if args.checkpoint_metric == "ap":
        # Do not even open the combined V2 index in P1 mode: it contains test
        # rows.  The dedicated reader independently verifies both frozen
        # development manifests and their PT paths.
        index_csv = V2_FROZEN_TRAIN_CSV
        index_sha256 = str(V2_FROZEN_DEVELOPMENT_MANIFESTS["train"]["sha256"])
    else:
        index_csv = Path(contract["index_csv"])
        if not index_csv.is_file():
            raise FileNotFoundError(index_csv)
        index_sha256 = sha256_file(index_csv)
        expected_index_sha256 = str(contract["expected_index_sha256"])
        if index_sha256 != expected_index_sha256:
            raise RuntimeError(
                f"{args.split_contract} index SHA256 mismatch: "
                f"expected={expected_index_sha256} actual={index_sha256}"
            )

    device = choose_device(args.device)
    configure_architecture(args.seed, index_csv)
    architecture.configure_reproducibility(args.seed, args.determinism)
    if args.checkpoint_metric == "ap":
        frame, parts = read_frozen_v2_development_manifests()
    else:
        frame, parts = read_index(index_csv)
    parts, component_weight_stats = attach_component_weights(
        parts, args.component_weighting, args.split_contract
    )

    if args.preflight_only:
        run_preflight(args, parts, device)
        return
    if not args.mode:
        raise ValueError("--mode is required unless --preflight-only is used")

    contract_tag = "" if args.split_contract == "v1" else "_v2_strict40"
    checkpoint_tag = "_ckpt_ap" if args.checkpoint_metric == "ap" else ""
    weighting_tag = "" if args.component_weighting == "none" else f"_{args.component_weighting}"
    optimizer_tag = ""
    if args.optimizer != "adam" or args.weight_decay != 0.0:
        weight_decay_tag = f"{args.weight_decay:g}".replace("-", "m").replace(".", "p")
        optimizer_tag = f"_{args.optimizer}_wd{weight_decay_tag}"
    run_tag = (
        f"v10c_formal{contract_tag}_{args.mode}_type_balanced{checkpoint_tag}{weighting_tag}"
        f"{optimizer_tag}_"
        f"{args.run_label}_seed{args.seed}"
    )
    out_dir = output_root / run_tag
    checkpoint_dir = out_dir / "checkpoints"
    best_checkpoint = checkpoint_dir / "best_model.pt"
    last_checkpoint = checkpoint_dir / "last_state.pt"
    train_log_csv = out_dir / "train_log.csv"
    predictions_csv = out_dir / "test_predictions.csv"
    result_json = out_dir / "final_result.json"
    validation_result_json = out_dir / "validation_result.json"
    validation_predictions_csv = out_dir / "validation_predictions.csv"
    validation_operating_points_json = out_dir / "validation_operating_points.json"
    manifest_json = out_dir / "reproducibility_manifest.json"
    config = config_from_args(
        args,
        index_csv,
        index_sha256,
        feature_profile,
        component_weight_stats,
        len(frame),
        {name: len(part) for name, part in parts.items()},
    )

    if result_json.is_file():
        print(f"[COMPLETE] Final result already exists: {result_json}")
        print("[COMPLETE] Use a different --run-label for a new independent run.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(manifest(config, run_tag, device), manifest_json)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    loaders = make_loaders(parts, args.mode, args.batch_size, args.num_workers, train_generator, device)
    schema = architecture.base.graph_schema(args.mode)
    model = architecture.TypeBalancedHGTV9(args.hidden_dim, schema, args.mode).to(device)
    optimizer = make_optimizer(
        model,
        args.optimizer,
        args.learning_rate,
        args.weight_decay,
    )
    class_weights = torch.tensor([1.0, args.malicious_class_weight], dtype=torch.float32, device=device)

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_ap = -1.0
    best_val_auroc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_metrics: dict[str, float] = {}
    best_selection_key = (
        (-float("inf"),)
        if args.checkpoint_metric == "f1"
        else (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    )
    patience_reference_metric = -1.0
    no_improve_epochs = 0
    completed_elapsed_seconds = 0.0
    log_rows: list[dict] = []
    training_complete = False

    if last_checkpoint.is_file():
        if args.resume == "never":
            raise RuntimeError(f"last checkpoint exists but --resume=never: {last_checkpoint}")
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        validate_resume_config(state["config"], config)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_val_f1 = float(state["best_val_f1"])
        best_epoch = int(state["best_epoch"])
        best_checkpoint_metrics = dict(state.get("best_checkpoint_metrics", {}))
        best_val_ap = float(
            state.get("best_val_ap", best_checkpoint_metrics.get("average_precision", -1.0))
        )
        best_val_auroc = float(
            state.get("best_val_auroc", best_checkpoint_metrics.get("roc_auc", -1.0))
        )
        best_val_loss = float(state.get("best_val_loss", best_checkpoint_metrics.get("loss", float("inf"))))
        saved_selection_key = state.get("best_selection_key")
        if saved_selection_key is not None:
            best_selection_key = tuple(float(value) for value in saved_selection_key)
        elif args.checkpoint_metric == "f1":
            best_selection_key = (best_val_f1,)
        else:
            if not best_checkpoint_metrics:
                raise RuntimeError("AP resume checkpoint is missing best-checkpoint ranking metrics")
            best_selection_key = checkpoint_selection_key(
                best_checkpoint_metrics, args.checkpoint_metric, best_epoch
            )
        patience_reference_metric = float(
            state.get("patience_reference_metric", state.get("patience_reference_f1", -1.0))
        )
        no_improve_epochs = int(state["no_improve_epochs"])
        completed_elapsed_seconds = float(state["completed_elapsed_seconds"])
        log_rows = list(state["log_rows"])
        start_epoch = int(state["epoch"]) + 1
        training_complete = bool(state.get("training_complete", False))
        restore_rng_state(state["rng_state"], train_generator)
        if training_complete:
            print(f"[RESUME] Training already complete at epoch {start_epoch - 1}; no more epochs will run")
        else:
            print(f"[RESUME] Continuing from completed epoch {start_epoch - 1}")
    elif args.resume == "require":
        raise RuntimeError(f"--resume=require but no checkpoint exists: {last_checkpoint}")

    print("=" * 110)
    print(f"FORMAL V10c FULL-DATASET HGT - {args.mode.upper()}")
    print("=" * 110)
    print(f"[INFO] run tag        : {run_tag}")
    print(f"[INFO] split contract : {args.split_contract}")
    print(f"[INFO] input index    : {index_csv}")
    print(f"[INFO] index SHA256   : {index_sha256}")
    print(f"[INFO] output dir     : {out_dir}")
    print(f"[INFO] device         : {device}")
    print(f"[INFO] determinism    : {args.determinism}")
    print(f"[INFO] seed           : {args.seed}")
    if args.checkpoint_metric == "ap":
        print(f"[INFO] train/val      : {len(parts['train'])}/{len(parts['val'])}")
        print("[INFO] test data       : NOT READ; NO TEST LOADER")
    else:
        print(f"[INFO] train/val/test : {len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")
    print(f"[INFO] mode nodes     : {schema[0]}")
    print(f"[INFO] edge types     : {len(schema[1])}")
    print(
        f"[INFO] epoch policy   : min={args.min_epochs}, max={args.max_epochs}, "
        f"patience={args.patience}, min_delta={args.min_delta}"
    )
    print(
        f"[INFO] optimizer      : {args.optimizer} lr={args.learning_rate}, "
        f"weight_decay={args.weight_decay}, batch={args.batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, workers={args.num_workers}"
    )
    print(f"[INFO] component weight: {args.component_weighting}")
    print(f"[INFO] checkpoint gate : validation {args.checkpoint_metric.upper()}")
    print(f"[INFO] early stopping  : validation {args.checkpoint_metric.upper()}")
    print(f"[INFO] evaluation gate : {'validation-only' if args.validation_only else 'test-after-training'}")
    if args.checkpoint_metric == "ap":
        print("[INFO] P1 contract     : frozen V2 train + original validation only")
    if args.component_weighting != "none":
        malicious_stats = component_weight_stats["labels"]["malicious"]
        print(
            "[INFO] malicious mass : "
            f"largest {malicious_stats['unweighted_largest_component_mass'] * 100:.2f}%"
            f" -> {malicious_stats['weighted_largest_component_mass'] * 100:.2f}%, "
            f"top3 {malicious_stats['unweighted_top3_component_mass'] * 100:.2f}%"
            f" -> {malicious_stats['weighted_top3_component_mass'] * 100:.2f}%"
        )
        for source_group, source_stat in component_weight_stats["sources"][
            "malicious"
        ].items():
            print(
                f"[INFO] malicious source: {source_group} "
                f"{source_stat['unweighted_mass_share_within_label'] * 100:.2f}%"
                f" -> {source_stat['weighted_mass_share_within_label'] * 100:.2f}%"
            )
    print("[INFO] interrupt/resume: Ctrl+C is safe after a completed epoch; rerun the same command to resume")
    print("-" * 110)

    session_start = time.perf_counter()
    try:
        epoch_iterator = () if training_complete else range(start_epoch, args.max_epochs + 1)
        for epoch in epoch_iterator:
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            loss_sum = 0.0
            train_y_true: list[int] = []
            train_y_pred: list[int] = []
            train_loader = loaders["train"]
            total_steps = len(train_loader)

            for step, batch in enumerate(train_loader, 1):
                batch = batch.to(device, non_blocking=(device.type == "cuda"))
                out = model(batch)
                y = batch.y.view(-1)
                raw_loss = F.cross_entropy(out, y, reduction="none")
                component_weights = batch.component_weight.view(-1)
                loss = (
                    raw_loss * class_weights[y] * component_weights
                ).mean() / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0 or step == total_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                loss_sum += loss.item() * y.size(0) * args.gradient_accumulation_steps
                train_y_true.extend(y.detach().cpu().tolist())
                train_y_pred.extend(out.argmax(dim=1).detach().cpu().tolist())

                if step % args.progress_every == 0 or step == total_steps:
                    step_elapsed = time.perf_counter() - epoch_start
                    step_eta = step_elapsed / step * (total_steps - step)
                    partial = binary_metrics(train_y_true, train_y_pred)
                    print(
                        f"[TRAIN] epoch={epoch}/{args.max_epochs} step={step}/{total_steps} "
                        f"loss={loss_sum / max(len(train_y_true), 1):.4f} "
                        f"f1={partial['f1'] * 100:.2f}% "
                        f"elapsed={format_seconds(step_elapsed)} eta={format_seconds(step_eta)}"
                    )

            train_seconds = time.perf_counter() - epoch_start
            train_metrics = binary_metrics(train_y_true, train_y_pred)
            train_metrics["loss"] = loss_sum / max(len(train_y_true), 1)

            validation_start = time.perf_counter()
            val_metrics = evaluate(model, loaders["val"], device)
            validation_seconds = time.perf_counter() - validation_start
            epoch_seconds = time.perf_counter() - epoch_start
            completed_elapsed_seconds += epoch_seconds

            candidate_selection_key = checkpoint_selection_key(
                val_metrics, args.checkpoint_metric, epoch
            )
            is_new_best = candidate_selection_key > best_selection_key
            if is_new_best:
                best_selection_key = candidate_selection_key
                best_val_f1 = float(val_metrics["f1"])
                best_val_ap = float(val_metrics["average_precision"])
                best_val_auroc = float(val_metrics["roc_auc"])
                best_val_loss = float(val_metrics["loss"])
                best_epoch = epoch
                best_checkpoint_metrics = {
                    key: float(val_metrics[key])
                    for key in ("loss", "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc")
                }
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "best_val_f1": best_val_f1,
                        "best_val_ap": best_val_ap,
                        "best_val_auroc": best_val_auroc,
                        "best_val_loss": best_val_loss,
                        "best_checkpoint_metrics": best_checkpoint_metrics,
                        "best_selection_key": best_selection_key,
                        "checkpoint_metric": args.checkpoint_metric,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    best_checkpoint,
                )

            current_patience_metric = checkpoint_metric_value(val_metrics, args.checkpoint_metric)
            is_significant_improvement = (
                current_patience_metric > patience_reference_metric + args.min_delta
            )
            if is_significant_improvement:
                patience_reference_metric = current_patience_metric
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            average_epoch_seconds = completed_elapsed_seconds / epoch
            estimated_to_cap = average_epoch_seconds * (args.max_epochs - epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_average_precision": val_metrics["average_precision"],
                "val_roc_auc": val_metrics["roc_auc"],
                "best_val_f1": best_val_f1,
                "best_val_average_precision": best_val_ap,
                "best_val_roc_auc": best_val_auroc,
                "best_val_loss": best_val_loss,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_metric_value": current_patience_metric,
                "development_only_contract": args.checkpoint_metric == "ap",
                "best_epoch": best_epoch,
                "new_best": is_new_best,
                "significant_improvement": is_significant_improvement,
                "no_improve_epochs": no_improve_epochs,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "completed_elapsed_seconds": completed_elapsed_seconds,
                "estimated_seconds_to_max_epochs": estimated_to_cap,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(train_log_csv, index=False, encoding="utf-8")

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "best_val_ap": best_val_ap,
                    "best_val_auroc": best_val_auroc,
                    "best_val_loss": best_val_loss,
                    "best_checkpoint_metrics": best_checkpoint_metrics,
                    "best_selection_key": best_selection_key,
                    "checkpoint_metric": args.checkpoint_metric,
                    "best_epoch": best_epoch,
                    "patience_reference_metric": patience_reference_metric,
                    "patience_reference_f1": (
                        patience_reference_metric if args.checkpoint_metric == "f1" else -1.0
                    ),
                    "no_improve_epochs": no_improve_epochs,
                    "completed_elapsed_seconds": completed_elapsed_seconds,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                     "rng_state": capture_rng_state(train_generator),
                     "log_rows": log_rows,
                     "training_complete": False,
                     "config": config,
                },
                last_checkpoint,
            )

            best_display_value = best_val_ap if args.checkpoint_metric == "ap" else best_val_f1
            print(
                f"[EPOCH] {epoch}/{args.max_epochs} "
                f"train_f1={train_metrics['f1'] * 100:.2f}% "
                f"val_f1={val_metrics['f1'] * 100:.2f}% "
                f"val_ap={val_metrics['average_precision'] * 100:.2f}% "
                f"best_{args.checkpoint_metric}={best_display_value * 100:.2f}%@{best_epoch} "
                f"no_improve={no_improve_epochs}/{args.patience} "
                f"time={format_seconds(epoch_seconds)} "
                f"eta_to_cap={format_seconds(estimated_to_cap)}"
            )
            if is_new_best:
                print(f"[SAVE] best validation checkpoint updated at epoch {epoch}")

            if epoch >= args.min_epochs and no_improve_epochs >= args.patience:
                print(
                    f"[EARLY STOP] validation {args.checkpoint_metric.upper()} had no "
                    f">= {args.min_delta:.6f} improvement "
                    f"for {args.patience} epochs"
                )
                break
    except KeyboardInterrupt:
        print()
        print("[INTERRUPTED] Training stopped by user.")
        print(f"[INTERRUPTED] Rerun the same command to resume from: {last_checkpoint}")
        return

    if not log_rows:
        raise RuntimeError("training completed without any epoch log rows")
    final_epoch = int(log_rows[-1]["epoch"])
    if not training_complete:
        completion_state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        completion_state["training_complete"] = True
        atomic_torch_save(completion_state, last_checkpoint)
        training_complete = True

    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    best_validation_metrics = evaluate(
        model,
        loaders["val"],
        device,
        validation_predictions_csv,
    )
    selection_fields = (
        ("f1",)
        if args.checkpoint_metric == "f1"
        else ("average_precision", "roc_auc", "loss")
    )
    logged_selection_metrics = {
        "f1": best_val_f1,
        "average_precision": best_val_ap,
        "roc_auc": best_val_auroc,
        "loss": best_val_loss,
    }
    for field in selection_fields:
        if not np.isclose(
            best_validation_metrics[field],
            logged_selection_metrics[field],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"best validation checkpoint {field} mismatch: "
                f"logged={logged_selection_metrics[field]} "
                f"reevaluated={best_validation_metrics[field]}"
            )
    best_val_f1 = float(best_validation_metrics["f1"])
    best_val_ap = float(best_validation_metrics["average_precision"])
    best_val_auroc = float(best_validation_metrics["roc_auc"])
    best_val_loss = float(best_validation_metrics["loss"])
    validation_operating_points = audit_prediction_csv(
        validation_predictions_csv,
        label_column="y_true",
        score_column="prob_malicious",
        expected_rows=EXPECTED_SPLITS["val"],
        fixed_threshold=0.5,
    )
    atomic_operating_point_json(
        validation_operating_points,
        validation_operating_points_json,
    )

    validation_result = {
        "status": "validation_complete",
        "run_tag": run_tag,
        "mode": args.mode,
        "seed": args.seed,
        "run_label": args.run_label,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_average_precision": best_val_ap,
        "best_val_roc_auc": best_val_auroc,
        "best_val_loss": best_val_loss,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_metric_value": (
            best_val_ap if args.checkpoint_metric == "ap" else best_val_f1
        ),
        "checkpoint_tie_break": config["checkpoint_tie_break"],
        "development_only_contract": config["development_only_contract"],
        "best_validation_metrics": best_validation_metrics,
        "final_epoch": final_epoch,
        "stopped_early": final_epoch < args.max_epochs,
        "completed_training_seconds": completed_elapsed_seconds,
        "train_log_csv": str(train_log_csv),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "validation_predictions_csv": str(validation_predictions_csv),
        "validation_operating_points_json": str(validation_operating_points_json),
        "test_evaluated": False,
        "config": config,
    }
    atomic_json_save(validation_result, validation_result_json)
    if args.validation_only:
        print("=" * 110)
        print("VALIDATION-ONLY RESULT")
        print(f"[RESULT] mode           : {args.mode}")
        print(f"[RESULT] seed           : {args.seed}")
        print(f"[RESULT] best epoch     : {best_epoch}/{final_epoch}")
        print(f"[RESULT] checkpoint gate: validation {args.checkpoint_metric.upper()}")
        print(f"[RESULT] best val F1    : {best_val_f1 * 100:.2f}%")
        print(
            "[RESULT] val AP/AUROC   : "
            f"{validation_operating_points['average_precision'] * 100:.2f}%/"
            f"{validation_operating_points['roc_auc'] * 100:.2f}%"
        )
        print(
            "[RESULT] val-selected F1: "
            f"{validation_operating_points['validation_selected_operating_point']['f1'] * 100:.2f}% "
            "(diagnostic only)"
        )
        print(f"[RESULT] training time  : {format_seconds(completed_elapsed_seconds)}")
        print("[RESULT] test evaluated : NO")
        print(f"[RESULT] validation JSON: {validation_result_json}")
        print("=" * 110)
        return

    test_start = time.perf_counter()
    test_metrics = evaluate(model, loaders["test"], device, predictions_csv)
    test_seconds = time.perf_counter() - test_start
    stopped_early = final_epoch < args.max_epochs
    result = {
        "status": "complete",
        "run_tag": run_tag,
        "mode": args.mode,
        "seed": args.seed,
        "run_label": args.run_label,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_average_precision": best_val_ap,
        "best_val_roc_auc": best_val_auroc,
        "best_val_loss": best_val_loss,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_metric_value": (
            best_val_ap if args.checkpoint_metric == "ap" else best_val_f1
        ),
        "checkpoint_tie_break": config["checkpoint_tie_break"],
        "development_only_contract": config["development_only_contract"],
        "final_epoch": final_epoch,
        "stopped_early": stopped_early,
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "tp": test_metrics["tp"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "completed_training_seconds": completed_elapsed_seconds,
        "test_seconds": test_seconds,
        "session_seconds": time.perf_counter() - session_start,
        "train_log_csv": str(train_log_csv),
        "test_predictions_csv": str(predictions_csv),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "validation_predictions_csv": str(validation_predictions_csv),
        "validation_operating_points_json": str(validation_operating_points_json),
        "config": config,
    }
    atomic_json_save(result, result_json)

    print("=" * 110)
    print("FINAL RESULT")
    print(f"[RESULT] mode           : {args.mode}")
    print(f"[RESULT] seed           : {args.seed}")
    print(f"[RESULT] best epoch     : {best_epoch}/{final_epoch}")
    print(f"[RESULT] checkpoint gate: validation {args.checkpoint_metric.upper()}")
    print(f"[RESULT] best val F1    : {best_val_f1 * 100:.2f}%")
    print(f"[RESULT] test accuracy  : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"[RESULT] test precision : {test_metrics['precision'] * 100:.2f}%")
    print(f"[RESULT] test recall    : {test_metrics['recall'] * 100:.2f}%")
    print(f"[RESULT] test F1        : {test_metrics['f1'] * 100:.2f}%")
    print(f"[RESULT] TP/TN/FP/FN    : {test_metrics['tp']}/{test_metrics['tn']}/{test_metrics['fp']}/{test_metrics['fn']}")
    print(f"[RESULT] training time  : {format_seconds(completed_elapsed_seconds)}")
    print(f"[RESULT] result JSON    : {result_json}")
    print("=" * 110)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=FORMAL_MODES)
    parser.add_argument(
        "--split-contract",
        choices=sorted(SPLIT_CONTRACTS),
        default="v1",
        help=(
            "v1 preserves historical runs; v2_strict40 uses the frozen "
            "2026-07-22 split and an isolated output root."
        ),
    )
    parser.add_argument("--run-label", default="20260612_convergence_b4_s01of01")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--determinism", choices=["off", "strict"], default="strict")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="adam",
        help="adam preserves every historical run; adamw enables decoupled weight decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2/decoupled weight decay. Historical default is 0.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--malicious-class-weight", type=float, default=1.5)
    parser.add_argument(
        "--component-weighting",
        choices=COMPONENT_WEIGHTING_CHOICES,
        default="none",
        help=(
            "none preserves historical per-sample training; inverse_sqrt keeps all "
            "samples but normalizes component_size^-0.5 weights to mean one within "
            "each train label; inverse_sqrt_source_preserving normalizes the same "
            "family weighting inside each label/source stratum so the observed "
            "source mixture is unchanged. Authorized only for v2_strict40."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--resume", choices=["auto", "never", "require"], default="auto")
    parser.add_argument(
        "--checkpoint-metric",
        choices=CHECKPOINT_METRIC_CHOICES,
        default="f1",
        help=(
            "f1 preserves historical checkpoint/early-stopping behavior; ap enables the "
            "P1 development-only protocol (requires v2_strict40 and --validation-only) "
            "with AUROC, lower validation loss, and earlier epoch as frozen tie-breakers."
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "train and select the best checkpoint using validation only, write "
            "validation_result.json, and do not run test inference or write test "
            "predictions. "
            "Historical F1 runs may be rerun without this flag to test their frozen "
            "completed checkpoint. AP-mode runs remain development-only."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
