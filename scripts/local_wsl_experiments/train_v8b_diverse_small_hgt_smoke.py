#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv, global_max_pool, global_mean_pool

import train_v7c_small_hgt_smoke as common


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = ROOT / "experiment_outputs" / "train_final_v2_20260416"
IN_CSV = TRAIN_ROOT / "step05f_training_pt_index_v8b_diverse_small.csv"

BATCH_SIZE = 1
EPOCHS = 3
LR = 3e-4
HIDDEN_DIM = 64
NUM_WORKERS = 0
SEED = 20260605
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_WEIGHTS = torch.tensor([1.0, 1.5], dtype=torch.float32).to(DEVICE)

IN_DIMS = {"module": 4, "ast": 6, "param": 18, "event": 12}

AST_EDGES = {
    ("module", "E_AST", "ast"),
    ("ast", "REV_E_AST", "module"),
}
STRUCTURE_EDGES = AST_EDGES | {
    ("ast", "E_IntraSeq", "ast"),
    ("ast", "REV_E_IntraSeq", "ast"),
    ("module", "E_InternalImport", "module"),
    ("module", "REV_E_InternalImport", "module"),
}
EVENT_LINK_EDGES = {
    ("ast", "E_Event", "event"),
    ("event", "REV_E_Event", "ast"),
    ("module", "E_ModuleEvent", "event"),
    ("event", "REV_E_ModuleEvent", "module"),
}
STRONG_EDGES = {
    ("event", "E_StrongBehaviorChain", "event"),
    ("event", "REV_E_StrongBehaviorChain", "event"),
}

MODE_EDGES = {
    "ast_only": AST_EDGES,
    "ast_structure": STRUCTURE_EDGES,
    "ast_event_only": AST_EDGES | EVENT_LINK_EDGES,
    "ast_event": STRUCTURE_EDGES | EVENT_LINK_EDGES,
    "ast_strong": STRUCTURE_EDGES | EVENT_LINK_EDGES | STRONG_EDGES,
}
MODE_NODE_TYPES = {
    "ast_only": ["module", "ast"],
    "ast_structure": ["module", "ast"],
    "ast_event_only": ["module", "ast", "event"],
    "ast_event": ["module", "ast", "event"],
    "ast_strong": ["module", "ast", "event"],
}


def empty_or_all_indices(data, node_type: str) -> torch.Tensor:
    return torch.arange(data[node_type].num_nodes, dtype=torch.long)


def strong_event_indices(data) -> torch.Tensor:
    ids = []
    for edge_type in STRONG_EDGES:
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            ids.extend(edge_index[0].tolist())
            ids.extend(edge_index[1].tolist())
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


class GraphDatasetV8B(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df.reset_index(drop=True).copy()
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(Path(str(row["pt_path_v8"])), weights_only=False)

        if self.mode == "ast_strong":
            subset = {
                "module": empty_or_all_indices(data, "module"),
                "ast": empty_or_all_indices(data, "ast"),
                "event": strong_event_indices(data),
                "param": torch.empty(0, dtype=torch.long),
            }
            data = data.subgraph(subset)

        keep_edges = MODE_EDGES[self.mode]
        for edge_type in list(data.edge_types):
            if edge_type not in keep_edges:
                del data[edge_type]
        for node_type in list(data.node_types):
            if node_type not in MODE_NODE_TYPES[self.mode]:
                del data[node_type]

        data.sample_id = str(row["sample_id"])
        data.graph_id = str(row["graph_id"])
        data.label_name = str(row["label"])
        data.split = str(row["split"])
        data.package_name = str(row["package_name"])
        data.version = str(row["version"])
        data.source_group = str(row.get("source_group", ""))
        label_int = row.get("label_int", None)
        if pd.isna(label_int):
            label_int = {"benign": 0, "malicious": 1}[str(row["label"])]
        data.y = torch.tensor([int(label_int)], dtype=torch.long)
        return data


class HGTV8B(nn.Module):
    def __init__(self, hidden_dim: int, schema, mode: str):
        super().__init__()
        self.mode = mode
        self.lin_dict = nn.ModuleDict({
            node_type: nn.Linear(IN_DIMS[node_type], hidden_dim)
            for node_type in MODE_NODE_TYPES[mode]
        })
        self.conv1 = HGTConv(hidden_dim, hidden_dim, schema, heads=2)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, schema, heads=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def merge(old, new):
        return {node_type: new.get(node_type, value) for node_type, value in old.items()}

    def forward(self, data):
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](data[node_type].x))
            for node_type in data.node_types
            if node_type in self.lin_dict
        }
        x1 = {key: F.relu(value) for key, value in self.conv1(x_dict, data.edge_index_dict).items()}
        x_dict = self.merge(x_dict, x1)
        x2 = {key: F.relu(value) for key, value in self.conv2(x_dict, data.edge_index_dict).items()}
        x_dict = self.merge(x_dict, x2)

        all_x, all_batch = [], []
        for node_type, x in x_dict.items():
            if x.size(0) > 0:
                all_x.append(x)
                all_batch.append(data[node_type].batch)
        x = torch.cat(all_x, dim=0)
        batch = torch.cat(all_batch, dim=0)
        representation = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(representation)


def graph_schema(mode: str):
    return (MODE_NODE_TYPES[mode], sorted(MODE_EDGES[mode]))


def train(mode: str, run_tag: str | None = None):
    common.set_seed(SEED)
    out_dir = TRAIN_ROOT / (run_tag or f"v8b_diverse_small_{mode}_hgt_smoke_run1")
    ckpt_dir = out_dir / "checkpoints"
    log_csv = out_dir / "train_log.csv"
    best_ckpt = ckpt_dir / "best_model.pt"
    pred_csv = out_dir / "test_predictions.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, low_memory=False)
    df = df[df["status"] == "ok"].copy()
    parts = {split: df[df["split"] == split].copy() for split in ["train", "val", "test"]}
    loaders = {
        split: DataLoader(
            GraphDatasetV8B(part, mode),
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
        )
        for split, part in parts.items()
    }

    schema = graph_schema(mode)
    model = HGTV8B(HIDDEN_DIM, schema, mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_f1 = -1.0
    log_rows = []

    print("=" * 100)
    print(f"V8B DIVERSE SMALL HGT SMOKE TEST - {mode.upper()}")
    print("=" * 100)
    print(f"[INFO] input csv   : {IN_CSV}")
    print(f"[INFO] output dir  : {out_dir}")
    print(f"[INFO] device      : {DEVICE}")
    print(f"[INFO] train/val/test rows: {len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")
    print(f"[INFO] graph schema: {schema}")
    print("-" * 100)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss_sum, y_true, y_pred = 0.0, [], []
        for step, batch in enumerate(loaders["train"], 1):
            batch = batch.to(DEVICE)
            out = model(batch)
            y = batch.y.view(-1)
            raw_loss = F.cross_entropy(out, y, reduction="none")
            loss = (raw_loss * CLASS_WEIGHTS[y]).mean() / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item() * y.size(0) * GRADIENT_ACCUMULATION_STEPS
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(out.argmax(dim=1).detach().cpu().tolist())

        train_m = common.binary_metrics(y_true, y_pred)
        train_m["loss"] = loss_sum / max(len(y_true), 1)
        val_m = common.evaluate(model, loaders["val"])
        print(f"[EPOCH] {epoch} train_f1={train_m['f1']*100:.2f}% val_f1={val_m['f1']*100:.2f}%")
        log_rows.append({
            "epoch": epoch, "train_loss": train_m["loss"], "train_acc": train_m["accuracy"],
            "train_precision": train_m["precision"], "train_recall": train_m["recall"], "train_f1": train_m["f1"],
            "val_loss": val_m["loss"], "val_acc": val_m["accuracy"], "val_precision": val_m["precision"],
            "val_recall": val_m["recall"], "val_f1": val_m["f1"],
        })
        pd.DataFrame(log_rows).to_csv(log_csv, index=False, encoding="utf-8")
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({"model_state_dict": model.state_dict(), "best_val_f1": best_val_f1}, best_ckpt)

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=False)["model_state_dict"])
    test_m = common.evaluate(model, loaders["test"], save_pred_path=pred_csv)
    print("=" * 100)
    print("FINAL RESULT")
    print(f"[RESULT] mode          : {mode}")
    print(f"[RESULT] best_val_f1   : {best_val_f1*100:.2f}%")
    print(f"[RESULT] test_acc      : {test_m['accuracy']*100:.2f}%")
    print(f"[RESULT] test_precision: {test_m['precision']*100:.2f}%")
    print(f"[RESULT] test_recall   : {test_m['recall']*100:.2f}%")
    print(f"[RESULT] test_f1       : {test_m['f1']*100:.2f}%")
    print(f"[RESULT] tp/tn/fp/fn   : {test_m['tp']}/{test_m['tn']}/{test_m['fp']}/{test_m['fn']}")
    print(f"[RESULT] train_log_csv : {log_csv}")
    print(f"[RESULT] test_pred_csv : {pred_csv}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_EDGES), required=True)
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
