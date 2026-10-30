#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import warnings
from pathlib import Path

# cuBLAS needs this environment variable before CUDA is initialized.
if "--determinism" in sys.argv:
    determinism_index = sys.argv.index("--determinism")
    if determinism_index + 1 < len(sys.argv) and sys.argv[determinism_index + 1] == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_max_pool, global_mean_pool

warnings.filterwarnings(
    "ignore",
    message=r"The usage of `scatter\(reduce='max'\)` can be accelerated via the 'torch-scatter' package.*",
    category=UserWarning,
    module=r"torch_geometric\.utils\._scatter",
)

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent / "local_wsl_experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import train_v8b_diverse_small_hgt_smoke as base


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = ROOT / "experiment_outputs/train_final_v9_20260608"
base.TRAIN_ROOT = TRAIN_ROOT
base.IN_CSV = TRAIN_ROOT / "step04_training_pt_index_v9_small.csv"

PARAM_EDGES = {
    ("ast", "E_Param", "param"),
    ("param", "REV_E_Param", "ast"),
    ("event", "E_EventParam", "param"),
    ("param", "REV_E_EventParam", "event"),
}
EVENT_PARAM_EDGES = {
    ("event", "E_EventParam", "param"),
    ("param", "REV_E_EventParam", "event"),
}
PAYLOAD_TO_EVENT_EDGES = {
    ("param", "REV_E_EventParam", "event"),
}
EVENT_FLOW_NAMES = {
    "E_ReadToNetwork", "REV_E_ReadToNetwork",
    "E_DecodeToExecute", "REV_E_DecodeToExecute",
    "E_DownloadToWrite", "REV_E_DownloadToWrite",
    "E_DownloadToExecute", "REV_E_DownloadToExecute",
    "E_WriteToExecute", "REV_E_WriteToExecute",
    "E_StrongBehaviorChain", "REV_E_StrongBehaviorChain",
}
EVENT_FLOW_EDGES = {("event", name, "event") for name in EVENT_FLOW_NAMES}
STRONG_FLOW_EDGES = {
    ("event", "E_StrongBehaviorChain", "event"),
    ("event", "REV_E_StrongBehaviorChain", "event"),
}

base.MODE_NODE_TYPES.update({
    "ast_event_flow": ["module", "ast", "event"],
    "ast_event_param": ["module", "ast", "param", "event"],
    "ast_event_param_context": ["module", "ast", "param", "event"],
    "ast_event_payload_context": ["module", "ast", "param", "event"],
    "ast_event_payload_to_event": ["module", "ast", "param", "event"],
    "ast_event_param_strong": ["module", "ast", "param", "event"],
    "full": ["module", "ast", "param", "event"],
})
base.MODE_EDGES.update({
    "ast_event_flow": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | EVENT_FLOW_EDGES,
    "ast_event_param": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | PARAM_EDGES,
    "ast_event_param_context": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | PARAM_EDGES,
    "ast_event_payload_context": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | EVENT_PARAM_EDGES,
    "ast_event_payload_to_event": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | PAYLOAD_TO_EVENT_EDGES,
    "ast_event_param_strong": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | PARAM_EDGES | STRONG_FLOW_EDGES,
    "full": base.STRUCTURE_EDGES | base.EVENT_LINK_EDGES | PARAM_EDGES | EVENT_FLOW_EDGES,
})
POOL_NODE_TYPES = {
    "ast_event_param_context": ["module", "ast", "event"],
    "ast_event_payload_context": ["module", "ast", "event"],
    "ast_event_payload_to_event": ["module", "ast", "event"],
}
PARAM_BASE_FEATURE_DIM = 19
PARAM_FEATURE_ABLATION = "none"
PARAM_EDGE_ABLATION = "none"
OriginalGraphDatasetV8B = base.GraphDatasetV8B


def configure_reproducibility(seed: int, determinism: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if determinism == "strict":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def write_reproducibility_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    run_tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_tag": run_tag,
        "mode": args.mode,
        "readout": args.readout,
        "features": args.features,
        "param_feature_ablation": args.param_feature_ablation,
        "param_edge_ablation": args.param_edge_ablation,
        "seed": args.seed,
        "run_label": args.run_label,
        "determinism": args.determinism,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    with (out_dir / "reproducibility_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


class ParamAblationDataset(OriginalGraphDatasetV8B):
    """Apply controlled Param ablations while preserving graph-size signals."""

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if "param" not in data.node_types:
            return data

        if PARAM_EDGE_ABLATION == "shuffle_targets":
            sample_id = str(data.sample_id)
            self._shuffle_relation_targets(
                data,
                sample_id,
                ("ast", "E_Param", "param"),
                ("param", "REV_E_Param", "ast"),
            )
            self._shuffle_relation_targets(
                data,
                sample_id,
                ("event", "E_EventParam", "param"),
                ("param", "REV_E_EventParam", "event"),
            )

        if PARAM_FEATURE_ABLATION == "zero_all":
            data["param"].x = torch.zeros_like(data["param"].x)
        elif PARAM_FEATURE_ABLATION == "zero_context":
            if data["param"].x.size(1) <= PARAM_BASE_FEATURE_DIM:
                raise RuntimeError(
                    "zero_context requires V9c/V9d Param context features, "
                    f"but only {data['param'].x.size(1)} Param dimensions were found"
                )
            data["param"].x = data["param"].x.clone()
            data["param"].x[:, PARAM_BASE_FEATURE_DIM:] = 0.0
        return data

    @staticmethod
    def _shuffle_relation_targets(data, sample_id: str, forward_type, reverse_type) -> None:
        has_forward = forward_type in data.edge_types
        has_reverse = reverse_type in data.edge_types
        if not has_forward and not has_reverse:
            return

        edge_index = (
            data[forward_type].edge_index
            if has_forward
            else data[reverse_type].edge_index.flip(0)
        )
        edge_count = edge_index.size(1)
        if edge_count < 2:
            return

        digest = hashlib.sha256(f"v9c_param_shuffle_v1|{sample_id}|{forward_type}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        sources = edge_index[0].tolist()
        targets = edge_index[1].tolist()
        edge_pairs = set(zip(sources, targets))

        # Degree-preserving double-edge swaps change semantic pairings without
        # changing source/target degrees or introducing duplicate edges.
        for _ in range(max(32, edge_count * 20)):
            left = rng.randrange(edge_count)
            right = rng.randrange(edge_count)
            if left == right:
                continue
            src_left, src_right = sources[left], sources[right]
            dst_left, dst_right = targets[left], targets[right]
            if src_left == src_right or dst_left == dst_right:
                continue
            new_left = (src_left, dst_right)
            new_right = (src_right, dst_left)
            if new_left in edge_pairs or new_right in edge_pairs:
                continue
            edge_pairs.remove((src_left, dst_left))
            edge_pairs.remove((src_right, dst_right))
            edge_pairs.add(new_left)
            edge_pairs.add(new_right)
            targets[left], targets[right] = dst_right, dst_left

        shuffled = torch.tensor([sources, targets], dtype=edge_index.dtype)
        if has_forward:
            data[forward_type].edge_index = shuffled
        if has_reverse:
            data[reverse_type].edge_index = shuffled.flip(0)


class TypeBalancedHGTV9(base.HGTV8B):
    """Pool each node type independently before concatenation."""

    def __init__(self, hidden_dim: int, schema, mode: str):
        super().__init__(hidden_dim, schema, mode)
        self.node_types = list(POOL_NODE_TYPES.get(mode, base.MODE_NODE_TYPES[mode]))
        self.type_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim * 2)
            for node_type in self.node_types
        })
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * len(self.node_types), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](data[node_type].x))
            for node_type in data.node_types
            if node_type in self.lin_dict
        }
        # Some valid packages become edgeless after an ablation removes node
        # types. Keep their projected node features instead of rejecting them.
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
            if "edge_index" in data[edge_type]
        }
        if edge_index_dict:
            x1 = {key: F.relu(value) for key, value in self.conv1(x_dict, edge_index_dict).items()}
            x_dict = self.merge(x_dict, x1)
            x2 = {key: F.relu(value) for key, value in self.conv2(x_dict, edge_index_dict).items()}
            x_dict = self.merge(x_dict, x2)

        batch_size = int(data.y.view(-1).size(0))
        representations = []
        for node_type in self.node_types:
            x = x_dict.get(node_type)
            if x is None or x.size(0) == 0:
                pooled = torch.zeros((batch_size, self.type_norms[node_type].normalized_shape[0]), device=data.y.device)
            else:
                pooled = torch.cat([
                    global_mean_pool(x, data[node_type].batch, size=batch_size),
                    global_max_pool(x, data[node_type].batch, size=batch_size),
                ], dim=1)
            representations.append(self.type_norms[node_type](pooled))
        return self.classifier(torch.cat(representations, dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "ast_only", "ast_structure", "ast_event", "ast_event_flow",
            "ast_event_param", "ast_event_param_context", "ast_event_payload_context",
            "ast_event_payload_to_event", "ast_event_param_strong", "full",
        ],
        required=True,
    )
    parser.add_argument("--readout", choices=["unified", "type_balanced"], required=True)
    parser.add_argument(
        "--features",
        choices=[
            "raw", "normalized", "strict", "strict_medium", "strict_large",
            "strict_medium_2000", "v9c_global_param_medium_2000",
            "v9d_role_param_medium_2000",
        ],
        default="strict",
    )
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument(
        "--determinism",
        choices=["off", "strict"],
        default="off",
        help="Strict mode rejects CUDA/PyTorch operations that cannot run deterministically.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional suffix for repeated runs, for example repeat_a or repeat_b.",
    )
    parser.add_argument(
        "--param-feature-ablation",
        choices=["none", "zero_all", "zero_context"],
        default="none",
        help="Mask Param features at training time while preserving Param nodes and edges.",
    )
    parser.add_argument(
        "--param-edge-ablation",
        choices=["none", "shuffle_targets"],
        default="none",
        help="Shuffle Param relation targets while preserving node, edge, and degree counts.",
    )
    args = parser.parse_args()

    if args.run_label and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("--run-label may contain only letters, digits, underscores, and hyphens")
    if args.param_feature_ablation != "none" and args.features not in {
        "v9c_global_param_medium_2000",
        "v9d_role_param_medium_2000",
    }:
        parser.error("Param feature ablations require V9c or V9d 29-dimensional Param features")
    if args.param_feature_ablation != "none" and args.mode not in {
        "ast_event_param",
        "ast_event_param_context",
        "ast_event_payload_context",
        "ast_event_payload_to_event",
        "ast_event_param_strong",
        "full",
    }:
        parser.error("Param feature ablations require a mode that retains Param nodes")
    if args.param_edge_ablation != "none" and args.features != "v9c_global_param_medium_2000":
        parser.error("Param edge ablation currently requires the V9c graph")
    if args.param_edge_ablation != "none" and args.mode not in {
        "ast_event_param",
        "ast_event_param_context",
        "ast_event_payload_to_event",
    }:
        parser.error(
            "Param edge ablation currently requires ast_event_param, "
            "ast_event_param_context, or ast_event_payload_to_event"
        )

    if args.features == "normalized":
        base.IN_CSV = TRAIN_ROOT / "step04b_training_pt_index_v9_small_normalized.csv"
    elif args.features == "strict":
        base.IN_CSV = TRAIN_ROOT / "step04c_training_pt_index_v9b_strict_small_normalized.csv"
        base.IN_DIMS["param"] = 19
        base.IN_DIMS["event"] = 18
    elif args.features in {"strict_medium", "strict_large"}:
        profile = args.features.removeprefix("strict_")
        base.IN_CSV = TRAIN_ROOT / f"step06_training_pt_index_v9b_strict_{profile}_normalized.csv"
        base.IN_DIMS["param"] = 19
        base.IN_DIMS["event"] = 18
    elif args.features == "strict_medium_2000":
        base.IN_CSV = TRAIN_ROOT / "step09_training_pt_index_v9b_strict_medium_2000_normalized.csv"
        base.IN_DIMS["param"] = 19
        base.IN_DIMS["event"] = 18
    elif args.features == "v9c_global_param_medium_2000":
        base.IN_CSV = TRAIN_ROOT / "step13_training_pt_index_v9c_global_param_medium_2000_normalized.csv"
        base.IN_DIMS["param"] = 29
        base.IN_DIMS["event"] = 18
    elif args.features == "v9d_role_param_medium_2000":
        base.IN_CSV = TRAIN_ROOT / "step17_training_pt_index_v9d_role_param_medium_2000_normalized.csv"
        base.IN_DIMS["param"] = 29
        base.IN_DIMS["event"] = 18
    base.SEED = args.seed
    base.HGTV8B = TypeBalancedHGTV9 if args.readout == "type_balanced" else base.HGTV8B
    global PARAM_FEATURE_ABLATION, PARAM_EDGE_ABLATION
    PARAM_FEATURE_ABLATION = args.param_feature_ablation
    PARAM_EDGE_ABLATION = args.param_edge_ablation
    base.GraphDatasetV8B = ParamAblationDataset
    run_tag = f"v9_small_{args.features}_{args.mode}_{args.readout}_seed{args.seed}"
    if args.param_feature_ablation != "none":
        run_tag = f"{run_tag}_param_{args.param_feature_ablation}"
    if args.param_edge_ablation != "none":
        run_tag = f"{run_tag}_param_edges_{args.param_edge_ablation}"
    if args.run_label:
        run_tag = f"{run_tag}_{args.run_label}"
    configure_reproducibility(args.seed, args.determinism)
    write_reproducibility_manifest(TRAIN_ROOT / run_tag, args, run_tag)
    print(f"[INFO] determinism : {args.determinism}")
    print(f"[INFO] run label   : {args.run_label or '(none)'}")
    print(f"[INFO] Param mask  : {args.param_feature_ablation}")
    print(f"[INFO] Param edges : {args.param_edge_ablation}")
    base.train(args.mode, run_tag=run_tag)


if __name__ == "__main__":
    main()
