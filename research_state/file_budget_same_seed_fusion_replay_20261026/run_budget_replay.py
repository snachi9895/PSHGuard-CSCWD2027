#!/usr/bin/env python3
"""Retrospective validation-only CPU replay; all writes stay in this directory.

Never imports or executes a historical audit runner. Only the locally copied,
AST-verified pure numeric S8-A functions are used. No neural-model dependency.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from legacy_fusion_math import crossfit, metrics, summarize


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (1001, 1002, 1003)
BUDGETS = (4, 8, 16, 32)
METRICS = ("average_precision", "auroc", "precision", "recall", "f1", "fpr")
META = ("sample_id", "graph_id", "source_group", "source_clone_component_id", "y_true", "outer_fold")
PURE_FUNCTIONS = ("confusion", "select_threshold", "make_model", "crossfit", "metrics", "summarize")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value, path):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(ready(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def frozen_input(relative, freeze):
    if relative not in freeze["input_sha256"]:
        raise ValueError(f"Input not frozen: {relative}")
    result = (ROOT / relative).resolve()
    if ROOT not in result.parents:
        raise ValueError("Input escapes workspace")
    return result


def functions_ast(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in PURE_FUNCTIONS}


def preflight():
    freeze = read_json(HERE / "protocol" / "PREFIT_FREEZE.json")
    protocol = read_json(HERE / "protocol" / "PROTOCOL_v1.json")
    if sha(HERE / "protocol" / "PROTOCOL_v1.json") != freeze["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze")
    for relative, expected in freeze["code_sha256"].items():
        if sha(HERE / relative) != expected:
            raise ValueError(f"Local code drift: {relative}")
    for relative, expected in freeze["input_sha256"].items():
        if sha(frozen_input(relative, freeze)) != expected:
            raise ValueError(f"Input drift: {relative}")
    original = frozen_input(freeze["paths"]["original_numeric_source"], freeze)
    if functions_ast(original) != functions_ast(HERE / "legacy_fusion_math.py"):
        raise ValueError("Copied numerical functions differ from original S8-A")
    if len(functions_ast(original)) != len(PURE_FUNCTIONS):
        raise ValueError("Missing expected pure numerical functions")
    if sklearn.__version__ != "1.7.2":
        raise ValueError(f"Unexpected sklearn version: {sklearn.__version__}")
    original_spec = read_json(frozen_input(freeze["paths"]["original_spec"], freeze))
    for key, expected in protocol["fusion_model"].items():
        if original_spec["fusion_model"][key] != expected:
            raise ValueError(f"Original fusion protocol mismatch: {key}")
    return freeze, protocol


def load_population(freeze):
    paths = freeze["paths"]
    p4 = pd.read_csv(frozen_input(paths["p4"], freeze), low_memory=False)
    if len(p4) != 2137 or p4["sample_id"].nunique() != 2137:
        raise ValueError("P4 population mismatch")
    base = p4[list(META)].copy()
    if base["y_true"].value_counts().to_dict() != {0: 1082, 1: 1055}:
        raise ValueError("Label totals mismatch")
    if base["source_clone_component_id"].nunique() != 1588:
        raise ValueError("Component count mismatch")
    if set(base["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Fold identity mismatch")
    if base.groupby("source_clone_component_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("Component crosses outer folds")
    for seed in SEEDS:
        base[f"graph_seed{seed}_margin"] = p4[f"seed{seed}_logit_margin"].to_numpy(dtype=float)
        sequence = pd.read_csv(frozen_input(paths["sequence"][str(seed)], freeze), low_memory=False)
        if len(sequence) != 2137 or sequence["sample_id"].nunique() != 2137:
            raise ValueError(f"Sequence population mismatch: {seed}")
        sequence = sequence.set_index("sample_id").loc[base["sample_id"]].reset_index()
        pairs = (("y_true", "label_id"), ("source_group", "source_group"),
                 ("source_clone_component_id", "component_id"), ("outer_fold", "outer_fold"))
        for left, right in pairs:
            if not np.array_equal(base[left].astype(str), sequence[right].astype(str)):
                raise ValueError(f"Sequence metadata mismatch: {seed}, {left}")
        base[f"modernbert_seed{seed}_margin"] = sequence["modernbert_logit_margin"].to_numpy(dtype=float)
    numeric = [c for c in base if c.endswith("_margin")]
    if not np.isfinite(base[numeric].to_numpy()).all():
        raise ValueError("Non-finite base margins")
    return base


def frame_for_budget(base, budget, freeze):
    frame = base.copy()
    if budget == 16:
        return frame
    for seed in SEEDS:
        relative = freeze["paths"]["graph"][str(budget)][str(seed)]
        graph = pd.read_csv(frozen_input(relative, freeze), low_memory=False)
        if len(graph) != 2137 or graph["sample_id"].nunique() != 2137:
            raise ValueError(f"Graph population mismatch: {budget}, {seed}")
        if set(graph["sample_id"]) != set(base["sample_id"]):
            raise ValueError("Graph IDs mismatch")
        graph = graph.set_index("sample_id").loc[base["sample_id"]].reset_index()
        for key in ("graph_id", "source_group", "source_clone_component_id", "y_true"):
            if not np.array_equal(graph[key].astype(str), base[key].astype(str)):
                raise ValueError(f"Graph metadata mismatch: {budget}, {seed}, {key}")
        frame[f"graph_seed{seed}_margin"] = graph["logit_margin"].to_numpy(dtype=float)
    numeric = [c for c in frame if c.endswith("_margin")]
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Non-finite budget margins")
    return frame


def evaluate_budget(base, budget, freeze, protocol):
    frame = frame_for_budget(base, budget, freeze)
    labels = frame["y_true"].to_numpy(dtype=int)
    output = frame[list(META)].copy()
    seed_rows, fold_rows, parameters = [], [], []
    for seed in SEEDS:
        feature_names = [f"graph_seed{seed}_margin", f"modernbert_seed{seed}_margin"]
        print(f"CPU OOF fit: graph budget={budget}, paired seed={seed}, folds=5", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = crossfit(frame, feature_names, protocol, "same_seed_pair", seed)
        if caught:
            detail = [f"{x.category.__name__}: {x.message}" for x in caught]
            raise RuntimeError(f"Warnings require investigation, no retuning: {detail}")
        row = metrics(labels, result["scores"], result["predictions"])
        row["fpr"] = row["fp"] / (row["fp"] + row["tn"])
        row.update({"budget": budget, "seed": seed, "system": "one_HGT_plus_same_seed_ModernBERT"})
        seed_rows.append(row)
        for item in result["fold_rows"]:
            item["budget"] = budget
            item["fpr"] = item["fp"] / (item["fp"] + item["tn"])
            fold_rows.append(item)
        for item in result["parameters"]:
            item["budget"] = budget
            parameters.append(item)
        output[f"seed{seed}_graph_raw_margin"] = frame[feature_names[0]].to_numpy()
        output[f"seed{seed}_sequence_raw_margin"] = frame[feature_names[1]].to_numpy()
        output[f"seed{seed}_same_seed_pair_score"] = result["scores"]
        output[f"seed{seed}_same_seed_pair_prediction"] = result["predictions"]
        threshold_map = {item["outer_fold"]: item["threshold"] for item in result["parameters"]}
        output[f"seed{seed}_fold_threshold"] = frame["outer_fold"].map(threshold_map)
    return pd.DataFrame(seed_rows), pd.DataFrame(fold_rows), parameters, output


def reference_gate(output, seed_metrics, freeze):
    golden = pd.read_csv(frozen_input(freeze["paths"]["golden"], freeze), low_memory=False)
    if len(golden) != 2137 or golden["sample_id"].nunique() != 2137:
        raise ValueError("Reference OOF population mismatch")
    if set(golden["sample_id"]) != set(output["sample_id"]):
        raise ValueError("Reference OOF IDs mismatch")
    golden = golden.set_index("sample_id").loc[output["sample_id"]].reset_index()
    for key in META:
        if not np.array_equal(golden[key].astype(str), output[key].astype(str)):
            raise ValueError(f"Reference OOF metadata mismatch: {key}")
    old_metrics = pd.read_csv(frozen_input(freeze["paths"]["golden_metrics"], freeze))
    old_metrics = old_metrics[old_metrics["system"] == "same_seed_pair"].set_index("seed")
    rows = []
    for seed in SEEDS:
        column = f"seed{seed}_same_seed_pair_score"
        diff = np.abs(output[column].to_numpy() - golden[column].to_numpy())
        pred = f"seed{seed}_same_seed_pair_prediction"
        mismatches = int(np.count_nonzero(output[pred].to_numpy() != golden[pred].to_numpy()))
        new = seed_metrics.set_index("seed").loc[seed]
        metric_differences = {m: float(abs(new[m] - old_metrics.loc[seed, m]))
                              for m in METRICS if m != "fpr"}
        rows.append({"seed": seed, "max_abs_score_difference": float(diff.max()),
                     "score_differences_above_1e_12": int(np.count_nonzero(diff > 1e-12)),
                     "decision_mismatches": mismatches,
                     "metric_absolute_differences": metric_differences,
                     "pass": bool(diff.max() <= 1e-12 and mismatches == 0
                                  and max(metric_differences.values()) <= 1e-12)})
    return {"status": "pass" if all(r["pass"] for r in rows) else "fail",
            "budget": 16, "rows": 2137, "seeds": rows,
            "absolute_score_tolerance": 1e-12, "decision_mismatches_allowed": 0}


def save_budget_result(result, budget, out):
    seed_metrics, fold_metrics, parameters, oof = result
    seed_metrics.to_csv(out / f"budget{budget}_per_seed_metrics.csv", index=False)
    fold_metrics.to_csv(out / f"budget{budget}_per_fold_metrics.csv", index=False)
    oof.to_csv(out / f"budget{budget}_oof_predictions.csv", index=False)
    write_json(parameters, out / f"budget{budget}_fold_parameters.json")


def summarize_all(results):
    rows = []
    for budget in BUDGETS:
        frame = results[results["budget"] == budget].sort_values("seed")
        if tuple(frame["seed"]) != SEEDS:
            raise ValueError("Missing/duplicate seed in summary")
        row = {"budget": budget, "runs": 3}
        for metric in METRICS:
            for key, value in summarize(frame[metric].tolist()).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def verify_inputs_unchanged(freeze):
    changed = [relative for relative, expected in freeze["input_sha256"].items()
               if sha(frozen_input(relative, freeze)) != expected]
    if changed:
        raise ValueError(f"Source inputs changed: {changed}")
    return {"status": "pass", "files_checked": len(freeze["input_sha256"]), "changed": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("reference", "remaining"))
    args = parser.parse_args()
    started = time.perf_counter()
    freeze, protocol = preflight()
    out = HERE / "outputs" / "v1" / args.stage
    if out.exists():
        raise FileExistsError(f"Stage already exists; do not overwrite: {out}")
    if args.stage == "remaining":
        gate_path = HERE / "outputs" / "v1" / "reference" / "REFERENCE_GATE.json"
        if not gate_path.is_file() or read_json(gate_path)["status"] != "pass":
            raise RuntimeError("Budget16 reference gate must pass before new-budget evaluation")
        if sha(gate_path) != read_json(HERE / "protocol" / "REFERENCE_PASS_FREEZE.json")["reference_gate_sha256"]:
            raise RuntimeError("Reference gate changed after acceptance")
    out.mkdir(parents=True, exist_ok=False)
    write_json({"stage": args.stage, "status": "preflight_pass_before_fitting",
                "protocol_sha256": freeze["protocol_sha256"], "input_files": len(freeze["input_sha256"]),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "component_isolation_required": True, "threadpool_limit": 1}, out / "PREFLIGHT.json")
    base = load_population(freeze)
    budgets = (16,) if args.stage == "reference" else (4, 8, 32)
    stage_rows = []
    with threadpool_limits(limits=1):
        for budget in budgets:
            result = evaluate_budget(base, budget, freeze, protocol)
            save_budget_result(result, budget, out)
            stage_rows.append(result[0])
            if budget == 16:
                gate = reference_gate(result[3], result[0], freeze)
                write_json(gate, out / "REFERENCE_GATE.json")
                print(json.dumps(gate, ensure_ascii=False), flush=True)
                if gate["status"] != "pass":
                    raise RuntimeError("Reference reproduction failed: stop before new budgets")
    stage_metrics = pd.concat(stage_rows, ignore_index=True)
    stage_metrics.to_csv(out / "per_seed_metrics.csv", index=False)
    if args.stage == "remaining":
        reference = pd.read_csv(HERE / "outputs" / "v1" / "reference" / "per_seed_metrics.csv")
        all_rows = pd.concat([reference, stage_metrics], ignore_index=True).sort_values(["budget", "seed"])
        all_rows.to_csv(out / "all_budgets_per_seed_metrics.csv", index=False)
        summary = summarize_all(all_rows)
        summary.to_csv(out / "all_budgets_mean_sample_sd.csv", index=False)
        ref = reference.set_index("seed")
        differences = []
        for _, row in all_rows.iterrows():
            differences.append({"budget": int(row["budget"]), "seed": int(row["seed"]),
                                **{f"{m}_difference_vs16_pp": float(100 * (row[m] - ref.loc[row["seed"], m]))
                                   for m in METRICS}})
        pd.DataFrame(differences).to_csv(out / "descriptive_per_seed_differences_vs16.csv", index=False)
        print(summary[["budget", "f1_mean", "f1_sample_standard_deviation", "average_precision_mean",
                       "recall_mean", "fpr_mean"]].to_string(index=False), flush=True)
    write_json({"status": "complete", "stage": args.stage, "budgets": budgets,
                "CPU_LR_fits": len(budgets) * 15, "elapsed_seconds": time.perf_counter() - started,
                "inputs_unchanged": verify_inputs_unchanged(freeze),
                "bootstrap_replicates": 0, "neural_training_or_forward": False,
                "official_model_or_threshold_changed": False, "final_test_read": False,
                "manuscript_or_PPT_changed_by_this_script": False}, out / "COMPLETION.json")


if __name__ == "__main__":
    main()
