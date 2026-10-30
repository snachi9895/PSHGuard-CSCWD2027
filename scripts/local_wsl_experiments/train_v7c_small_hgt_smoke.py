#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv, global_max_pool

import torch_geometric.typing as pyg_typing
import torch_geometric.nn.dense.linear as pyg_linear

pyg_typing.WITH_SEGMM = False
pyg_linear.WITH_SEGMM = False


ROOT = Path(os.environ.get("PSHGUARD_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = ROOT / "experiment_outputs" / "train_final_v2_20260416"
IN_CSV = TRAIN_ROOT / "step05c_training_pt_index_v7c_small.csv"

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

IN_DIMS = {
    "module": 4,
    "ast": 6,
    "param": 18,
}

FULL_EDGE_TYPES = {
    ("module", "E_AST", "ast"),
    ("ast", "REV_E_AST", "module"),
    ("ast", "E_IntraSeq", "ast"),
    ("ast", "REV_E_IntraSeq", "ast"),
    ("module", "E_InternalImport", "module"),
    ("module", "REV_E_InternalImport", "module"),
    ("ast", "E_Param", "param"),
    ("param", "REV_E_Param", "ast"),
    ("param", "E_ParamFlow", "param"),
    ("param", "REV_E_ParamFlow", "param"),
}

AST_ONLY_EDGE_TYPES = {
    ("module", "E_AST", "ast"),
    ("ast", "REV_E_AST", "module"),
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df.reset_index(drop=True).copy()
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = torch.load(Path(str(row["pt_path_v7c"])), weights_only=False)

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

        keep_edges = AST_ONLY_EDGE_TYPES if self.mode == "ast_only" else FULL_EDGE_TYPES
        for etype in list(data.edge_types):
            if etype not in keep_edges:
                del data[etype]

        if self.mode == "ast_only" and "param" in data.node_types:
            del data["param"]

        return data


class HGTV7CSmoke(nn.Module):
    def __init__(self, hidden_dim: int, graph_schema, mode: str):
        super().__init__()
        self.mode = mode
        node_types = ["module", "ast"] if mode == "ast_only" else ["module", "ast", "param"]
        self.lin_dict = nn.ModuleDict({
            ntype: nn.Linear(IN_DIMS[ntype], hidden_dim)
            for ntype in node_types
        })

        self.conv1 = HGTConv(hidden_dim, hidden_dim, graph_schema, heads=2)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, graph_schema, heads=2)

        classifier_in = hidden_dim if mode == "ast_only" else hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _merge_with_residual(old_x_dict, new_x_dict):
        return {ntype: new_x_dict.get(ntype, x) for ntype, x in old_x_dict.items()}

    def forward(self, data):
        x_dict = {}
        for ntype in data.node_types:
            if ntype in self.lin_dict:
                x_dict[ntype] = F.relu(self.lin_dict[ntype](data[ntype].x))

        x1_dict = self.conv1(x_dict, data.edge_index_dict)
        x1_dict = {k: F.relu(v) for k, v in x1_dict.items()}
        x_dict = self._merge_with_residual(x_dict, x1_dict)

        x2_dict = self.conv2(x_dict, data.edge_index_dict)
        x2_dict = {k: F.relu(v) for k, v in x2_dict.items()}
        x_dict = self._merge_with_residual(x_dict, x2_dict)

        ast_pool = global_max_pool(x_dict["ast"], data["ast"].batch)
        if self.mode == "ast_only":
            return self.classifier(ast_pool)

        if "param" in x_dict and x_dict["param"].size(0) > 0:
            param_pool = global_max_pool(x_dict["param"], data["param"].batch)
            if param_pool.size(0) != ast_pool.size(0):
                fixed = torch.zeros(
                    (ast_pool.size(0), param_pool.size(1)),
                    dtype=param_pool.dtype,
                    device=param_pool.device,
                )
                present_graph_ids = torch.unique(data["param"].batch)
                fixed[present_graph_ids] = param_pool
                param_pool = fixed
        else:
            param_pool = torch.zeros_like(ast_pool)

        return self.classifier(torch.cat([ast_pool, param_pool], dim=1))


def binary_metrics(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model, loader, save_pred_path: Path | None = None):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    rows = []

    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        y = batch.y.view(-1)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * y.size(0)

        prob = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

        if save_pred_path is not None:
            sample_ids = batch.sample_id if isinstance(batch.sample_id, list) else [batch.sample_id]
            package_names = batch.package_name if isinstance(batch.package_name, list) else [batch.package_name]
            versions = batch.version if isinstance(batch.version, list) else [batch.version]
            source_groups = batch.source_group if isinstance(batch.source_group, list) else [batch.source_group]
            for i in range(y.size(0)):
                rows.append({
                    "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                    "package_name": package_names[i] if i < len(package_names) else "",
                    "version": versions[i] if i < len(versions) else "",
                    "source_group": source_groups[i] if i < len(source_groups) else "",
                    "y_true": int(y.detach().cpu()[i]),
                    "y_pred": int(pred.detach().cpu()[i]),
                    "prob_malicious": float(prob.detach().cpu()[i]),
                })

    m = binary_metrics(y_true, y_pred)
    m["loss"] = total_loss / len(y_true) if y_true else 0.0
    if save_pred_path is not None:
        pd.DataFrame(rows).to_csv(save_pred_path, index=False, encoding="utf-8")
    return m


def graph_schema_for_mode(mode: str):
    if mode == "ast_only":
        return (
            ["module", "ast"],
            [
                ("module", "E_AST", "ast"),
                ("ast", "REV_E_AST", "module"),
            ],
        )
    return (
        ["module", "ast", "param"],
        [
            ("module", "E_AST", "ast"),
            ("ast", "REV_E_AST", "module"),
            ("ast", "E_IntraSeq", "ast"),
            ("ast", "REV_E_IntraSeq", "ast"),
            ("module", "E_InternalImport", "module"),
            ("module", "REV_E_InternalImport", "module"),
            ("ast", "E_Param", "param"),
            ("param", "REV_E_Param", "ast"),
            ("param", "E_ParamFlow", "param"),
            ("param", "REV_E_ParamFlow", "param"),
        ],
    )


def train(mode: str, run_tag: str | None = None):
    set_seed(SEED)
    if not IN_CSV.exists():
        raise FileNotFoundError(f"missing V7c small pt index: {IN_CSV}")

    out_dir = TRAIN_ROOT / (run_tag or f"v7c_small_{mode}_hgt_smoke_run1")
    ckpt_dir = out_dir / "checkpoints"
    log_csv = out_dir / "train_log.csv"
    best_ckpt = ckpt_dir / "best_model.pt"
    pred_csv = out_dir / "test_predictions.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, low_memory=False)
    df = df[df["status"] == "ok"].copy()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_loader = DataLoader(GraphDataset(train_df, mode), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(GraphDataset(val_df, mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(GraphDataset(test_df, mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    graph_schema = graph_schema_for_mode(mode)
    model = HGTV7CSmoke(HIDDEN_DIM, graph_schema, mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_f1 = -1.0
    log_rows = []

    print("=" * 100)
    print(f"V7C SMALL HGT SMOKE TEST - {mode.upper()}")
    print("=" * 100)
    print(f"[INFO] input csv   : {IN_CSV}")
    print(f"[INFO] output dir  : {out_dir}")
    print(f"[INFO] device      : {DEVICE}")
    print(f"[INFO] epochs      : {EPOCHS}")
    print(f"[INFO] train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"[INFO] graph schema: {graph_schema}")
    print("-" * 100)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        y_true = []
        y_pred = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = batch.to(DEVICE)
            out = model(batch)
            y = batch.y.view(-1)

            loss_raw = F.cross_entropy(out, y, reduction="none")
            loss = (loss_raw * CLASS_WEIGHTS[y]).mean() / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * y.size(0) * GRADIENT_ACCUMULATION_STEPS
            pred = out.argmax(dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

            if step % 100 == 0:
                tmp_m = binary_metrics(y_true, y_pred)
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={train_loss_sum / max(len(y_true), 1):.4f} "
                    f"acc={tmp_m['accuracy']*100:.2f}% "
                    f"f1={tmp_m['f1']*100:.2f}%"
                )

        train_m = binary_metrics(y_true, y_pred)
        train_m["loss"] = train_loss_sum / max(len(y_true), 1)
        val_m = evaluate(model, val_loader)

        print(
            f"[EPOCH] {epoch} "
            f"| train_loss={train_m['loss']:.4f} train_f1={train_m['f1']*100:.2f}% "
            f"| val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']*100:.2f}% "
            f"val_p={val_m['precision']*100:.2f}% val_r={val_m['recall']*100:.2f}% "
            f"val_f1={val_m['f1']*100:.2f}%"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["accuracy"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
        })
        pd.DataFrame(log_rows).to_csv(log_csv, index=False, encoding="utf-8")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "graph_schema": graph_schema,
                "hidden_dim": HIDDEN_DIM,
                "in_dims": IN_DIMS,
                "mode": mode,
            }, best_ckpt)
            print(f"[SAVE] best checkpoint updated: epoch={epoch}, val_f1={val_m['f1']*100:.2f}%")

    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, save_pred_path=pred_csv)

    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)
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
    parser.add_argument("--mode", choices=["ast_only", "full"], required=True)
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
