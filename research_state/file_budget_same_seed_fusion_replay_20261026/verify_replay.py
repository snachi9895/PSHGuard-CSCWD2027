#!/usr/bin/env python3
"""Independent, no-fit check of all saved OOF results and source preservation."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
METRICS = ("average_precision", "auroc", "precision", "recall", "f1", "fpr")
SEEDS = (1001, 1002, 1003)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_threshold(labels, scores):
    # Count all >= threshold outcomes without calling the replay's threshold code.
    unique, inverse = np.unique(scores, return_inverse=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(unique))
    negatives = np.bincount(inverse, weights=1 - labels, minlength=len(unique))
    tp = positives[::-1].cumsum()[::-1]
    fp = negatives[::-1].cumsum()[::-1]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = tp / labels.sum()
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    best = max(range(len(unique)), key=lambda i: (f1[i], precision[i], -fp[i], unique[i]))
    return float(unique[best])


def main():
    remaining = HERE / "outputs/v1/remaining"
    combined = pd.read_csv(remaining / "all_budgets_per_seed_metrics.csv")
    summary = pd.read_csv(remaining / "all_budgets_mean_sample_sd.csv")
    assert len(combined) == 12 and len(summary) == 4
    population = None
    folds_checked = 0
    max_score_error = 0.0
    max_threshold_error = 0.0
    min_coefficient = float("inf")
    rows = []
    for budget in (4, 8, 16, 32):
        directory = HERE / "outputs/v1" / ("reference" if budget == 16 else "remaining")
        oof = pd.read_csv(directory / f"budget{budget}_oof_predictions.csv")
        fold_metrics = pd.read_csv(directory / f"budget{budget}_per_fold_metrics.csv")
        params = json.loads((directory / f"budget{budget}_fold_parameters.json").read_text())
        assert len(oof) == 2137 and oof.sample_id.nunique() == 2137
        metadata = oof[["sample_id", "graph_id", "source_group", "source_clone_component_id", "y_true", "outer_fold"]]
        if population is None:
            population = metadata
        else:
            pd.testing.assert_frame_equal(metadata, population)
        assert metadata.groupby("source_clone_component_id").outer_fold.nunique().max() == 1
        assert metadata.source_clone_component_id.nunique() == 1588
        labels = oof.y_true.to_numpy(dtype=int)
        assert (int((labels == 0).sum()), int((labels == 1).sum())) == (1082, 1055)
        assert len(params) == 15 and len(fold_metrics) == 15
        for seed in SEEDS:
            score = oof[f"seed{seed}_same_seed_pair_score"].to_numpy()
            prediction = oof[f"seed{seed}_same_seed_pair_prediction"].to_numpy(dtype=int)
            threshold = oof[f"seed{seed}_fold_threshold"].to_numpy()
            np.testing.assert_array_equal(prediction, (score >= threshold).astype(int))
            tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
            measured = {"average_precision": average_precision_score(labels, score),
                        "auroc": roc_auc_score(labels, score),
                        "precision": precision_score(labels, prediction),
                        "recall": recall_score(labels, prediction),
                        "f1": f1_score(labels, prediction), "fpr": fp / (fp + tn)}
            saved = combined[(combined.budget == budget) & (combined.seed == seed)].iloc[0]
            for metric in METRICS:
                assert abs(measured[metric] - saved[metric]) < 1e-12, (budget, seed, metric)
            for key, value in zip(("tn", "fp", "fn", "tp"), (tn, fp, fn, tp)):
                assert saved[key] == value
            expected_features = [f"graph_seed{seed}_margin", f"modernbert_seed{seed}_margin"]
            features = oof[[f"seed{seed}_graph_raw_margin", f"seed{seed}_sequence_raw_margin"]].to_numpy()
            for fold in range(1, 6):
                item = [x for x in params if x["seed"] == seed and x["outer_fold"] == fold]
                assert len(item) == 1
                item = item[0]
                assert item["feature_order"] == expected_features
                held = oof.outer_fold.to_numpy() == fold
                train = ~held
                assert not (set(oof.loc[train, "source_clone_component_id"]) &
                            set(oof.loc[held, "source_clone_component_id"]))
                means = np.array(item["scaler_mean"])
                scales = np.array(item["scaler_scale"])
                np.testing.assert_allclose(means, features[train].mean(axis=0), atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(scales, features[train].std(axis=0), atol=1e-12, rtol=1e-12)
                coefficients = np.array(item["coefficients"])
                min_coefficient = min(min_coefficient, float(coefficients.min()))
                reconstructed = expit(((features - means) / scales) @ coefficients + item["intercept"])
                error = float(abs(reconstructed[held] - score[held]).max())
                assert error <= 1e-12
                max_score_error = max(max_score_error, error)
                np.testing.assert_array_equal(prediction[held], (reconstructed[held] >= item["threshold"]).astype(int))
                chosen = independent_threshold(labels[train], reconstructed[train])
                error = abs(chosen - item["threshold"])
                assert error <= 1e-12, (budget, seed, fold, error)
                max_threshold_error = max(max_threshold_error, error)
                fm = fold_metrics[(fold_metrics.seed == seed) & (fold_metrics.outer_fold == fold)].iloc[0]
                assert int(fm.heldout_rows) == int(held.sum())
                assert int(fm.heldout_components) == oof.loc[held, "source_clone_component_id"].nunique()
                assert abs(fm.threshold - item["threshold"]) <= 1e-12
                folds_checked += 1
            rows.append({"budget": budget, "seed": seed, "OOF_rows": 2137,
                         "all_metrics_and_counts_verified": True, "folds_verified_without_refitting": 5})
        group = combined[combined.budget == budget].sort_values("seed")
        assert tuple(group.seed) == SEEDS
        summary_row = summary[summary.budget == budget].iloc[0]
        for metric in METRICS:
            assert abs(group[metric].mean() - summary_row[f"{metric}_mean"]) < 1e-12
            assert abs(group[metric].std(ddof=1) - summary_row[f"{metric}_sample_standard_deviation"]) < 1e-12
    differences = pd.read_csv(remaining / "descriptive_per_seed_differences_vs16.csv")
    assert len(differences) == 12
    for _, row in differences.iterrows():
        current = combined[(combined.budget == row.budget) & (combined.seed == row.seed)].iloc[0]
        reference = combined[(combined.budget == 16) & (combined.seed == row.seed)].iloc[0]
        for metric in METRICS:
            assert abs(row[f"{metric}_difference_vs16_pp"] - 100 * (current[metric] - reference[metric])) < 1e-10
    protection = json.loads((HERE / "protocol/PROTECTED_BEFORE.json").read_text())
    changed = [x["path"] for x in protection if sha(x["path"]) != x["sha256"]]
    assert not changed, changed
    freeze = json.loads((HERE / "protocol/PREFIT_FREEZE.json").read_text())
    for name, expected in freeze["code_sha256"].items():
        assert sha(HERE / name) == expected
    assert sha(HERE / "protocol/PROTOCOL_v1.json") == freeze["protocol_sha256"]
    assert sha(HERE / "protocol/PROTOCOL_v1.md") == freeze["protocol_markdown_sha256"]
    for stage in ("reference", "remaining"):
        completion = json.loads((HERE / f"outputs/v1/{stage}/COMPLETION.json").read_text())
        assert completion["status"] == "complete" and not completion["final_test_read"]
    report = {"status": "pass", "budgets": [4, 8, 16, 32], "seeds": list(SEEDS),
              "paired_systems": 12, "folds_verified_without_new_fits": folds_checked,
              "per_seed_metric_sets_verified": rows,
              "all_summary_means_and_sample_SDs_verified": True,
              "all_per_seed_differences_verified": True,
              "max_reconstructed_score_error": max_score_error,
              "max_training_threshold_reconstruction_error": max_threshold_error,
              "minimum_observed_LR_coefficient": min_coefficient,
              "protected_files_checked": len(protection), "protected_files_changed": changed,
              "new_fits_in_verification": 0, "bootstrap_replicates": 0,
              "final_test_access": False, "old_results_or_PPT_modified": False}
    target = HERE / "verification"
    target.mkdir(exist_ok=False)
    with (target / "INDEPENDENT_VERIFICATION.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_seed_metric_sets_verified"}, indent=2))


if __name__ == "__main__":
    main()
