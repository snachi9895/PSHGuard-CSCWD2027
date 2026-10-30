#!/usr/bin/env python3
"""Audit binary-classifier ranking, calibration, and validation thresholds.

This module is intentionally model-agnostic.  It reads only a prediction CSV
provided by the caller and never loads a train/test manifest.  The same
selection rule can therefore be applied to V10c and learned baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def expected_calibration_error(
    labels: np.ndarray,
    scores: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        frequency = float(np.mean(labels[mask]))
        error += float(np.sum(mask)) / total * abs(confidence - frequency)
    return float(error)


def audit_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    fixed_threshold: float = 0.5,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("prediction input is empty")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"labels must contain both 0 and 1, got {np.unique(labels).tolist()}")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite probabilities in [0, 1]")

    candidates = [
        metrics_at_threshold(labels, scores, float(threshold))
        for threshold in np.unique(scores)
    ]
    # Deterministic conservative tie-break: maximize F1, then precision, then
    # minimize false positives, then choose the higher threshold.
    optimal = max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -row["fp"],
            row["threshold"],
        ),
    )
    return {
        "status": "complete",
        "scope": "validation_predictions_only",
        "rows": int(len(labels)),
        "benign": int(np.sum(labels == 0)),
        "malicious": int(np.sum(labels == 1)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "ece_10_equal_width": expected_calibration_error(labels, scores, bins=10),
        "fixed_operating_point": metrics_at_threshold(
            labels, scores, float(fixed_threshold)
        ),
        "validation_selected_operating_point": optimal,
        "selection_policy": (
            "maximize validation F1; ties: maximize precision, minimize FP, "
            "then choose the higher threshold"
        ),
        "formal_use": (
            "diagnostic_only_until_the_same_validation-only rule is frozen "
            "for every learned comparison method"
        ),
    }


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_prediction_csv(
    prediction_csv: Path,
    label_column: str,
    score_column: str,
    expected_rows: int | None = None,
    fixed_threshold: float = 0.5,
) -> dict:
    frame = pd.read_csv(prediction_csv, low_memory=False)
    missing = {label_column, score_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{prediction_csv} is missing columns: {sorted(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} prediction rows, got {len(frame)}: {prediction_csv}"
        )
    audit = audit_operating_points(
        frame[label_column].to_numpy(),
        frame[score_column].to_numpy(),
        fixed_threshold=fixed_threshold,
    )
    audit.update(
        {
            "prediction_csv": str(prediction_csv.resolve()),
            "label_column": label_column,
            "score_column": score_column,
        }
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-column", default="y_true")
    parser.add_argument("--score-column", default="prob_malicious")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_prediction_csv(
        args.predictions,
        args.label_column,
        args.score_column,
        expected_rows=args.expected_rows,
        fixed_threshold=args.fixed_threshold,
    )
    atomic_write_json(audit, args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
