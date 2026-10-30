#!/usr/bin/env python3
"""Validation-only CPU audit of three same-seed HGT+ModernBERT pairs.

The script never constructs or reads a test path. It aligns the already frozen
validation predictions, reuses the P5 component folds and fitting protocol,
and reports three genuinely paired two-model systems plus a descriptive
three-HGT hybrid reference.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import fcntl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "formal_training_v10c_20260612"
OUT = (
    ROOT
    / "experiment_outputs"
    / "train_final_v10c_filtered700k_20260612"
    / "s8a_same_seed_paired_fusion_validation_20260818_v1"
)
SPEC = SCRIPT_DIR / "s8a_same_seed_paired_fusion_spec_v1.json"
FREEZE = SCRIPT_DIR / "s8a_same_seed_paired_fusion_freeze_manifest_v1.json"
RUNNER = SCRIPT_DIR / "run_s8a_same_seed_paired_fusion.sh"
P5_SPEC = SCRIPT_DIR / "p5_topology_sequence_fusion_spec_v1.json"
P4 = (
    ROOT
    / "experiment_outputs"
    / "train_final_v10c_filtered700k_20260612"
    / "p4_calibrated_ensemble_validation_20260808_v1"
    / "p4_calibrated_ensemble_oof_predictions.csv"
)
M_PATHS = {
    1001: ROOT
    / "experiment_outputs"
    / "train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_seed1001_validation_20260815_v1"
    / "validation_predictions.csv",
    1002: ROOT
    / "experiment_outputs"
    / "train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_confirmatory_seeds1002_1003_validation_20260816_v1"
    / "seed1002"
    / "validation_predictions.csv",
    1003: ROOT
    / "experiment_outputs"
    / "train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_confirmatory_seeds1002_1003_validation_20260816_v1"
    / "seed1003"
    / "validation_predictions.csv",
}

PREFLIGHT = OUT / "s8a_preflight_audit.json"
RESULT = OUT / "s8a_same_seed_paired_fusion_audit.json"
OOF = OUT / "s8a_same_seed_paired_fusion_oof_predictions.csv"
PER_SEED = OUT / "s8a_per_seed_metrics.csv"
PER_FOLD = OUT / "s8a_per_fold_metrics.csv"
BOOTSTRAP = OUT / "s8a_component_bootstrap.csv"
HIER_BOOTSTRAP = OUT / "s8a_hierarchical_seed_component_bootstrap.csv"
INVENTORY = OUT / "s8a_artifact_inventory.csv"
LOCK = OUT / ".s8a.lock"

SEEDS = (1001, 1002, 1003)
EXPECTED_ROWS = 2137
EXPECTED_BENIGN = 1082
EXPECTED_MALICIOUS = 1055
EXPECTED_COMPONENTS = 1588


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


@contextlib.contextmanager
def audit_lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another S8-A audit owns {LOCK}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def frozen_paths() -> dict[str, Path]:
    paths = {
        "scripts/formal_training_v10c_20260612/s8a_same_seed_paired_fusion_spec_v1.json": SPEC,
        "scripts/formal_training_v10c_20260612/audit_s8a_same_seed_paired_fusion.py": Path(__file__).resolve(),
        "scripts/formal_training_v10c_20260612/run_s8a_same_seed_paired_fusion.sh": RUNNER,
        "scripts/formal_training_v10c_20260612/p5_topology_sequence_fusion_spec_v1.json": P5_SPEC,
        "experiment_outputs/train_final_v10c_filtered700k_20260612/p4_calibrated_ensemble_validation_20260808_v1/p4_calibrated_ensemble_oof_predictions.csv": P4,
    }
    for seed, path in M_PATHS.items():
        paths[str(path.relative_to(ROOT))] = path
    return paths


def preflight() -> tuple[dict[str, Any], dict[str, str]]:
    spec = read_json(SPEC)
    p5_spec = read_json(P5_SPEC)
    freeze = read_json(FREEZE)
    if spec["artifact"]["short_name"] != "S8-A-PAIR":
        raise ValueError("unexpected spec identity")
    if spec["population"]["seeds"] != list(SEEDS):
        raise ValueError("seed set drift")
    forbidden = ["model_training", "checkpoint_forward", "graph_rebuild", "pt_rewrite", "test_path_constructed", "test_read", "test_evaluated", "test_unlocked"]
    if any(bool(spec["scope"][name]) for name in forbidden):
        raise ValueError("forbidden S8-A scope enabled")
    expected_model = p5_spec["fusion_model"]
    for key in ("penalty", "C", "solver", "class_weight", "max_iter", "tol", "random_state"):
        if spec["fusion_model"][key] != expected_model[key]:
            raise ValueError(f"P5 fusion protocol drift: {key}")
    if int(spec["outer_evaluation"]["folds"]) != 5:
        raise ValueError("outer fold count drift")
    paths = frozen_paths()
    expected = freeze.get("sha256", {})
    if set(paths) != set(expected):
        raise ValueError(f"freeze inventory drift missing={sorted(set(paths)-set(expected))} extra={sorted(set(expected)-set(paths))}")
    observed: dict[str, str] = {}
    for relative, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256_file(path)
        if observed[relative] != expected[relative]:
            raise ValueError(f"SHA mismatch: {relative}")
    return spec, observed


def load_inputs() -> pd.DataFrame:
    p4 = pd.read_csv(P4, low_memory=False)
    required = {
        "sample_id", "graph_id", "source_group", "source_clone_component_id", "y_true", "outer_fold",
        "seed1001_logit_margin", "seed1002_logit_margin", "seed1003_logit_margin",
        "single_platt_seed1001_oof_score", "single_platt_seed1001_oof_prediction",
        "single_platt_seed1002_oof_score", "single_platt_seed1002_oof_prediction",
        "single_platt_seed1003_oof_score", "single_platt_seed1003_oof_prediction",
    }
    if required - set(p4):
        raise ValueError(f"P4 missing columns: {sorted(required-set(p4))}")
    if len(p4) != EXPECTED_ROWS or p4["sample_id"].duplicated().any():
        raise ValueError("P4 population is not exactly 2,137 unique rows")
    base = p4.copy()
    for seed in SEEDS:
        modern = pd.read_csv(M_PATHS[seed], low_memory=False)
        modern_required = {"sample_id", "label_id", "source_group", "component_id", "outer_fold", "modernbert_logit_margin", "modernbert_oof_score", "modernbert_oof_prediction"}
        if modern_required - set(modern):
            raise ValueError(f"ModernBERT seed{seed} missing columns: {sorted(modern_required-set(modern))}")
        if len(modern) != EXPECTED_ROWS or modern["sample_id"].duplicated().any():
            raise ValueError(f"ModernBERT seed{seed} population drift")
        modern = modern[list(modern_required)].rename(columns={
            "label_id": f"modern_label_{seed}",
            "source_group": f"modern_source_{seed}",
            "component_id": f"modern_component_{seed}",
            "outer_fold": f"modern_fold_{seed}",
            "modernbert_logit_margin": f"modernbert_seed{seed}_logit_margin",
            "modernbert_oof_score": f"modernbert_seed{seed}_oof_score",
            "modernbert_oof_prediction": f"modernbert_seed{seed}_oof_prediction",
        })
        base = base.merge(modern, on="sample_id", how="inner", validate="one_to_one")
        checks = {
            "label": np.array_equal(base["y_true"].to_numpy(), base[f"modern_label_{seed}"].to_numpy()),
            "source": np.array_equal(base["source_group"].astype(str).to_numpy(), base[f"modern_source_{seed}"].astype(str).to_numpy()),
            "component": np.array_equal(base["source_clone_component_id"].astype(str).to_numpy(), base[f"modern_component_{seed}"].astype(str).to_numpy()),
            "fold": np.array_equal(base["outer_fold"].astype(int).to_numpy(), base[f"modern_fold_{seed}"].astype(int).to_numpy()),
        }
        if not all(checks.values()):
            raise ValueError(f"ModernBERT seed{seed} alignment mismatch: {checks}")
    if len(base) != EXPECTED_ROWS or base["sample_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("joined population drift")
    labels = base["y_true"].to_numpy(dtype=int)
    if int(np.sum(labels == 0)) != EXPECTED_BENIGN or int(np.sum(labels == 1)) != EXPECTED_MALICIOUS:
        raise ValueError("label counts drift")
    if base["source_clone_component_id"].nunique() != EXPECTED_COMPONENTS:
        raise ValueError("component count drift")
    if set(base["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("fold identities drift")
    if int(base.groupby("source_clone_component_id")["outer_fold"].nunique().max()) != 1:
        raise ValueError("source-clone component crosses folds")
    numeric = [column for column in base if "margin" in column or "oof_score" in column]
    if not np.isfinite(base[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite score found")
    return base


def confusion(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    weights = np.ones(len(labels), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    tp = float(np.sum(weights[(labels == 1) & (predictions == 1)]))
    fp = float(np.sum(weights[(labels == 0) & (predictions == 1)]))
    tn = float(np.sum(weights[(labels == 0) & (predictions == 0)]))
    fn = float(np.sum(weights[(labels == 1) & (predictions == 0)]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / np.sum(weights), "precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    rows = []
    for threshold in np.unique(scores):
        row = confusion(labels, scores >= threshold)
        row["threshold"] = float(threshold)
        rows.append(row)
    return max(rows, key=lambda row: (row["f1"], row["precision"], -row["fp"], row["threshold"]))


def make_model(spec: dict[str, Any]) -> Pipeline:
    config = spec["fusion_model"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("fusion", LogisticRegression(C=float(config["C"]), penalty=str(config["penalty"]), solver=str(config["solver"]), class_weight=config["class_weight"], max_iter=int(config["max_iter"]), tol=float(config["tol"]), random_state=int(config["random_state"]))),
    ])


def crossfit(frame: pd.DataFrame, feature_names: list[str], spec: dict[str, Any], system: str, seed: int) -> dict[str, Any]:
    labels = frame["y_true"].to_numpy(dtype=int)
    folds = frame["outer_fold"].to_numpy(dtype=int)
    features = frame[feature_names].to_numpy(dtype=float)
    scores = np.full(len(frame), np.nan)
    predictions = np.full(len(frame), -1, dtype=int)
    fold_rows = []
    parameters = []
    for fold in range(1, 6):
        held = folds == fold
        train = ~held
        train_components = set(frame.loc[train, "source_clone_component_id"].astype(str))
        held_components = set(frame.loc[held, "source_clone_component_id"].astype(str))
        if train_components & held_components:
            raise ValueError(f"component leakage seed={seed} system={system} fold={fold}")
        model = make_model(spec)
        model.fit(features[train], labels[train])
        train_scores = model.predict_proba(features[train])[:, 1]
        threshold = float(select_threshold(labels[train], train_scores)["threshold"])
        held_scores = model.predict_proba(features[held])[:, 1]
        held_predictions = (held_scores >= threshold).astype(int)
        scores[held] = held_scores
        predictions[held] = held_predictions
        metric = confusion(labels[held], held_predictions)
        metric.update({
            "seed": seed, "system": system, "outer_fold": fold, "threshold": threshold,
            "average_precision": float(average_precision_score(labels[held], held_scores)),
            "auroc": float(roc_auc_score(labels[held], held_scores)),
            "heldout_rows": int(np.sum(held)), "heldout_components": len(held_components),
        })
        coefficients = model.named_steps["fusion"].coef_[0]
        for name, value in zip(feature_names, coefficients):
            metric[f"coefficient__{name}"] = float(value)
        fold_rows.append(metric)
        parameters.append({
            "seed": seed, "system": system, "outer_fold": fold, "feature_order": feature_names,
            "scaler_mean": model.named_steps["scaler"].mean_, "scaler_scale": model.named_steps["scaler"].scale_,
            "coefficients": coefficients, "intercept": float(model.named_steps["fusion"].intercept_[0]), "threshold": threshold,
        })
    if not np.isfinite(scores).all() or np.any(predictions < 0):
        raise ValueError(f"incomplete OOF output: {system} seed{seed}")
    return {"scores": scores, "predictions": predictions, "fold_rows": fold_rows, "parameters": parameters}


def metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = confusion(labels, predictions)
    result.update({"average_precision": float(average_precision_score(labels, scores)), "auroc": float(roc_auc_score(labels, scores))})
    return result


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "sample_standard_deviation": float(np.std(array, ddof=1)), "minimum": float(np.min(array)), "maximum": float(np.max(array))}


def bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "sample_standard_deviation": float(np.std(values, ddof=1)),
        "ci95_percentile_lower": float(np.quantile(values, 0.025)), "median": float(np.median(values)),
        "ci95_percentile_upper": float(np.quantile(values, 0.975)), "probability_gt_zero": float(np.mean(values > 0)),
    }


def run_audit() -> None:
    with audit_lock():
        spec, input_hashes = preflight()
        frame = load_inputs()
        labels = frame["y_true"].to_numpy(dtype=int)
        preflight_payload = {
            "artifact": {"name": "S8-A preflight", "status": "pass", "date": "2026-08-18"},
            "checks": {"frozen_inputs_verified": True, "rows_2137_unique": True, "component_fold_isolation": True, "all_inputs_aligned": True, "test_path_constructed": False, "test_read": False},
            "population": {"rows": len(frame), "benign": int(np.sum(labels == 0)), "malicious": int(np.sum(labels == 1)), "components": int(frame["source_clone_component_id"].nunique()), "folds": sorted(frame["outer_fold"].astype(int).unique().tolist())},
            "input_sha256": input_hashes,
        }
        atomic_json(preflight_payload, PREFLIGHT)

        per_seed_rows = []
        fold_rows = []
        output_columns = ["sample_id", "graph_id", "source_group", "source_clone_component_id", "y_true", "outer_fold"]
        output = frame[output_columns].copy()
        systems: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            graph_margin = f"seed{seed}_logit_margin"
            modern_margin = f"modernbert_seed{seed}_logit_margin"
            pair_features = [graph_margin, modern_margin]
            hybrid_features = [f"seed{s}_logit_margin" for s in SEEDS] + [modern_margin]
            pair = crossfit(frame, pair_features, spec, "same_seed_pair", seed)
            hybrid = crossfit(frame, hybrid_features, spec, "three_hgt_hybrid_reference", seed)
            graph_scores = frame[f"single_platt_seed{seed}_oof_score"].to_numpy(dtype=float)
            graph_predictions = frame[f"single_platt_seed{seed}_oof_prediction"].to_numpy(dtype=int)
            modern_scores = frame[f"modernbert_seed{seed}_oof_score"].to_numpy(dtype=float)
            modern_predictions = frame[f"modernbert_seed{seed}_oof_prediction"].to_numpy(dtype=int)
            seed_systems = {
                "graph": {"scores": graph_scores, "predictions": graph_predictions},
                "modernbert": {"scores": modern_scores, "predictions": modern_predictions},
                "same_seed_pair": pair,
                "three_hgt_hybrid_reference": hybrid,
            }
            for name, item in seed_systems.items():
                metric = metrics(labels, item["scores"], item["predictions"])
                metric.update({"seed": seed, "system": name})
                per_seed_rows.append(metric)
                output[f"seed{seed}_{name}_score"] = item["scores"]
                output[f"seed{seed}_{name}_prediction"] = item["predictions"]
            fold_rows.extend(pair["fold_rows"])
            fold_rows.extend(hybrid["fold_rows"])
            systems[seed] = seed_systems

        per_seed_frame = pd.DataFrame(per_seed_rows)
        pair_frame = per_seed_frame[per_seed_frame["system"] == "same_seed_pair"].sort_values("seed")
        summaries = {metric: summarize(pair_frame[metric].tolist()) for metric in ("average_precision", "auroc", "accuracy", "precision", "recall", "f1", "fp", "fn")}

        component_codes, components = pd.factorize(frame["source_clone_component_id"].astype(str), sort=True)
        component_count = len(components)
        rng = np.random.default_rng(int(spec["statistics"]["random_state"]))
        per_seed_replicates = int(spec["statistics"]["per_seed_component_bootstrap_replicates"])
        bootstrap_rows = []
        per_seed_boot_summary: dict[str, Any] = {}
        for seed in SEEDS:
            deltas = {name: np.empty(per_seed_replicates) for name in ("pair_minus_graph", "pair_minus_modernbert", "pair_minus_hybrid")}
            for replicate in range(per_seed_replicates):
                counts = rng.multinomial(component_count, np.full(component_count, 1.0 / component_count))
                weights = counts[component_codes].astype(float)
                f1 = {name: confusion(labels, item["predictions"], weights)["f1"] for name, item in systems[seed].items()}
                deltas["pair_minus_graph"][replicate] = f1["same_seed_pair"] - f1["graph"]
                deltas["pair_minus_modernbert"][replicate] = f1["same_seed_pair"] - f1["modernbert"]
                deltas["pair_minus_hybrid"][replicate] = f1["same_seed_pair"] - f1["three_hgt_hybrid_reference"]
            for replicate in range(per_seed_replicates):
                bootstrap_rows.append({"seed": seed, "replicate": replicate + 1, **{name: float(values[replicate]) for name, values in deltas.items()}})
            per_seed_boot_summary[str(seed)] = {name: bootstrap_summary(values) for name, values in deltas.items()}

        hierarchical_replicates = int(spec["statistics"]["hierarchical_seed_component_bootstrap_replicates"])
        hierarchical = {name: np.empty(hierarchical_replicates) for name in ("pair_minus_graph", "pair_minus_modernbert", "pair_minus_hybrid")}
        hierarchical_rows = []
        for replicate in range(hierarchical_replicates):
            sampled_seeds = rng.choice(np.array(SEEDS), size=len(SEEDS), replace=True)
            counts = rng.multinomial(component_count, np.full(component_count, 1.0 / component_count))
            weights = counts[component_codes].astype(float)
            replicate_metrics = {name: [] for name in systems[1001]}
            for seed in sampled_seeds:
                for name, item in systems[int(seed)].items():
                    replicate_metrics[name].append(confusion(labels, item["predictions"], weights)["f1"])
            means = {name: float(np.mean(values)) for name, values in replicate_metrics.items()}
            hierarchical["pair_minus_graph"][replicate] = means["same_seed_pair"] - means["graph"]
            hierarchical["pair_minus_modernbert"][replicate] = means["same_seed_pair"] - means["modernbert"]
            hierarchical["pair_minus_hybrid"][replicate] = means["same_seed_pair"] - means["three_hgt_hybrid_reference"]
            hierarchical_rows.append({"replicate": replicate + 1, "sampled_seeds": ",".join(str(int(seed)) for seed in sampled_seeds), **{name: float(values[replicate]) for name, values in hierarchical.items()}})
        hierarchical_summary = {name: bootstrap_summary(values) for name, values in hierarchical.items()}

        per_seed_lookup = {(int(row.seed), str(row.system)): row for row in per_seed_frame.itertuples(index=False)}
        deltas = {}
        for seed in SEEDS:
            pair = per_seed_lookup[(seed, "same_seed_pair")]
            deltas[str(seed)] = {}
            for comparator in ("graph", "modernbert", "three_hgt_hybrid_reference"):
                ref = per_seed_lookup[(seed, comparator)]
                deltas[str(seed)][f"pair_minus_{comparator}"] = {metric: float(getattr(pair, metric) - getattr(ref, metric)) for metric in ("average_precision", "auroc", "precision", "recall", "f1")}

        positive_coefficients = all(value > 0 for row in fold_rows for key, value in row.items() if key.startswith("coefficient__"))
        directional = {
            "pair_F1_gt_graph_count": sum(deltas[str(seed)]["pair_minus_graph"]["f1"] > 0 for seed in SEEDS),
            "pair_F1_gt_modernbert_count": sum(deltas[str(seed)]["pair_minus_modernbert"]["f1"] > 0 for seed in SEEDS),
            "pair_F1_gt_hybrid_count": sum(deltas[str(seed)]["pair_minus_three_hgt_hybrid_reference"]["f1"] > 0 for seed in SEEDS),
        }
        validity = {
            "preflight_pass": True, "three_same_seed_pairs": True, "all_coefficients_positive": positive_coefficients,
            "rows_2137": len(frame) == EXPECTED_ROWS, "components_1588": component_count == EXPECTED_COMPONENTS,
            "test_path_constructed": False, "test_read": False, "test_evaluated": False, "test_unlocked": False,
        }

        atomic_csv(output, OOF)
        atomic_csv(per_seed_frame, PER_SEED)
        atomic_csv(pd.DataFrame(fold_rows), PER_FOLD)
        atomic_csv(pd.DataFrame(bootstrap_rows), BOOTSTRAP)
        atomic_csv(pd.DataFrame(hierarchical_rows), HIER_BOOTSTRAP)
        artifacts = [PREFLIGHT, OOF, PER_SEED, PER_FOLD, BOOTSTRAP, HIER_BOOTSTRAP]
        inventory = pd.DataFrame([{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts])
        atomic_csv(inventory, INVENTORY)
        positive_validity = ("preflight_pass", "three_same_seed_pairs", "all_coefficients_positive", "rows_2137", "components_1588")
        negative_validity = ("test_path_constructed", "test_read", "test_evaluated", "test_unlocked")
        integrity_pass = all(bool(validity[name]) for name in positive_validity) and not any(bool(validity[name]) for name in negative_validity)
        result = {
            "artifact": {"name": "S8-A same-seed paired fusion audit", "status": "pass" if integrity_pass else "fail", "date": "2026-08-18"},
            "scope": spec["scope"], "population": preflight_payload["population"], "validity_checks": validity,
            "per_seed_metrics": json.loads(per_seed_frame.to_json(orient="records")),
            "same_seed_pair_summary_mean_sample_sd": summaries,
            "per_seed_deltas": deltas, "directional_consistency": directional,
            "per_seed_component_bootstrap": per_seed_boot_summary,
            "hierarchical_seed_component_bootstrap": hierarchical_summary,
            "interpretation_contract": spec["interpretation"],
            "descriptive_hybrid_reference_note": spec["descriptive_hybrid_reference"],
            "input_sha256": input_hashes,
            "output_artifact_inventory_sha256": sha256_file(INVENTORY),
        }
        atomic_json(result, RESULT)
        print(json.dumps({
            "status": result["artifact"]["status"], "pair_F1_mean": summaries["f1"]["mean"],
            "pair_F1_sample_sd": summaries["f1"]["sample_standard_deviation"], "directional": directional,
            "hierarchical_bootstrap": hierarchical_summary, "result": str(RESULT), "test": "LOCKED",
        }, indent=2))


def status() -> None:
    if not RESULT.is_file():
        print("S8-A status=PENDING test=LOCKED")
        return
    result = read_json(RESULT)
    summary = result["same_seed_pair_summary_mean_sample_sd"]
    print("S8-A SAME-SEED PAIRED FUSION STATUS (validation only; test locked)")
    print(f"status={result['artifact']['status']} pair_F1={summary['f1']['mean']:.6f}+/-{summary['f1']['sample_standard_deviation']:.6f} test=LOCKED")
    for row in result["per_seed_metrics"]:
        if row["system"] == "same_seed_pair":
            print(f"seed={int(row['seed'])} AP={row['average_precision']:.6f} P/R/F1={row['precision']:.6f}/{row['recall']:.6f}/{row['f1']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    else:
        run_audit()


if __name__ == "__main__":
    main()
