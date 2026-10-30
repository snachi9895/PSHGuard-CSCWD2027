"""Unmodified pure numerical functions copied from the frozen S8-A source.
No historical runner is imported or executed. See PREFIT_FREEZE.json for provenance.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
