#!/usr/bin/env python3
"""Resumable seed-1001 validation-only ModernBERT S1-base screen."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SPEC_PATH = SCRIPT_PATH.with_name("s1_base_modernbert_screen_spec_v1.json")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_seed1001_validation_20260815_v1"
)
CACHE_ROOT = OUTPUT_ROOT / "cache"
CACHE_SUMMARY_PATH = CACHE_ROOT / "cache_audit.json"
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight_audit.json"
LAST_STATE_PATH = OUTPUT_ROOT / "last_epoch_state.pt"
BEST_MODEL_PATH = OUTPUT_ROOT / "best_validation_model.pt"
PREDICTIONS_PATH = OUTPUT_ROOT / "validation_predictions.csv"
RESULT_PATH = OUTPUT_ROOT / "validation_result.json"
LOCK_PATH = OUTPUT_ROOT / ".training.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
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


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    fp = float(np.sum((labels == 0) & (predictions == 1)))
    tn = float(np.sum((labels == 0) & (predictions == 0)))
    fn = float(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for threshold in np.unique(np.asarray(scores, dtype=np.float64)):
        metrics = confusion(labels, scores >= threshold)
        metrics["threshold"] = float(threshold)
        candidates.append(metrics)
    return max(
        candidates,
        key=lambda row: (row["f1"], row["precision"], -row["fp"], row["threshold"]),
    )


def raw_metrics(labels: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    result = confusion(labels, margins >= 0.0)
    result.update(
        {
            "average_precision": float(average_precision_score(labels, margins)),
            "auroc": float(roc_auc_score(labels, margins)),
            "native_margin_threshold": 0.0,
        }
    )
    return result


def crossfit_one_feature(
    *,
    labels: np.ndarray,
    margins: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        heldout = folds == fold
        training = ~heldout
        if set(components[heldout]) & set(components[training]):
            raise RuntimeError(f"component leakage in outer fold {fold}")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(margins[training, None])
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight=None,
            max_iter=1000,
            tol=1e-6,
            random_state=seed,
        )
        model.fit(train_x, labels[training])
        train_scores = model.predict_proba(train_x)[:, 1]
        selected = select_f1_threshold(labels[training], train_scores)
        heldout_scores = model.predict_proba(scaler.transform(margins[heldout, None]))[:, 1]
        heldout_predictions = (heldout_scores >= selected["threshold"]).astype(np.int64)
        scores[heldout] = heldout_scores
        predictions[heldout] = heldout_predictions
        metrics = confusion(labels[heldout], heldout_predictions)
        metrics.update(
            {
                "outer_fold": fold,
                "train_rows": int(training.sum()),
                "heldout_rows": int(heldout.sum()),
                "train_components": len(set(components[training])),
                "heldout_components": len(set(components[heldout])),
                "threshold": float(selected["threshold"]),
                "average_precision": float(average_precision_score(labels[heldout], heldout_scores)),
                "auroc": float(roc_auc_score(labels[heldout], heldout_scores)),
                "scaler_mean": float(scaler.mean_[0]),
                "scaler_scale": float(scaler.scale_[0]),
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )
        fold_rows.append(metrics)
    if not np.isfinite(scores).all() or np.any(predictions < 0):
        raise RuntimeError("incomplete OOF predictions")
    metrics = confusion(labels, predictions)
    metrics.update(
        {
            "average_precision": float(average_precision_score(labels, scores)),
            "auroc": float(roc_auc_score(labels, scores)),
            "folds": fold_rows,
        }
    )
    return {"scores": scores, "predictions": predictions, "metrics": metrics}


def stable_order(sample_ids: list[str], *, seed: int, epoch: int) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(sample_ids)),
            key=lambda index: hashlib.sha256(
                f"s1-base|{seed}|{epoch}|{sample_ids[index]}".encode("utf-8")
            ).digest(),
        ),
        dtype=np.int64,
    )


def cache_integrity(spec: dict[str, Any]) -> dict[str, Any]:
    if not CACHE_SUMMARY_PATH.is_file():
        raise RuntimeError("S1-base cache is missing; run the cache builder first")
    summary = json.loads(CACHE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["artifact"]["status"] != "pass" or summary["checks"]["test_read"]:
        raise RuntimeError("S1-base cache authorization invalid")
    if summary["spec_sha256"] != sha256_file(SPEC_PATH):
        raise RuntimeError("S1-base cache was built under a different specification")
    for split, expected in (("train", spec["scope"]["train_packages"]), ("validation", spec["scope"]["validation_packages"])):
        if int(summary["splits"][split]["packages"]) != int(expected):
            raise RuntimeError(f"{split} cache population drift")
        for name, digest in summary["splits"][split]["files_sha256"].items():
            if sha256_file(CACHE_ROOT / split / name) != digest:
                raise RuntimeError(f"{split} cache SHA256 drift: {name}")
    return summary


def build_preflight() -> dict[str, Any]:
    import sklearn
    import torch
    import transformers

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    cache = cache_integrity(spec)
    snapshot = resolve_project_path(spec["model"]["local_snapshot"])
    validation_cache = load_cache("validation")
    validation_meta = validation_cache["metadata"]
    fold_rows = read_csv(resolve_project_path(spec["inputs"]["outer_fold_manifest"]))
    fold_map = {row["sample_id"]: row for row in fold_rows}
    if len(fold_map) != len(validation_meta):
        raise RuntimeError("outer-fold preflight population drift")
    reference_labels = np.asarray(validation_cache["labels"], dtype=np.int64)
    reference_margins = np.asarray(
        [float(fold_map[row["sample_id"]]["ea4mp_behavior_bert_logit_margin"]) for row in validation_meta],
        dtype=np.float64,
    )
    reference_folds = np.asarray(
        [int(fold_map[row["sample_id"]]["outer_fold"]) for row in validation_meta],
        dtype=np.int64,
    )
    reference_components = np.asarray([row["component_id"] for row in validation_meta], dtype=object)
    reference_raw = raw_metrics(reference_labels, reference_margins)
    reference_oof = crossfit_one_feature(
        labels=reference_labels,
        margins=reference_margins,
        folds=reference_folds,
        components=reference_components,
        seed=int(spec["scope"]["seed"]),
    )
    checks = {
        "spec_frozen": spec["artifact"]["status"] == "frozen_before_cache_build_and_gpu_training",
        "cache_pass": cache["artifact"]["status"] == "pass",
        "model_weight_sha256": sha256_file(snapshot / "model.safetensors") == spec["model"]["model_weight_sha256"],
        "tokenizer_sha256": sha256_file(snapshot / "tokenizer.json") == spec["model"]["tokenizer_sha256"],
        "transformers_4_57_6": transformers.__version__ == "4.57.6",
        "torch_2_11_cuda128": torch.__version__ == "2.11.0+cu128",
        "sklearn_1_7_2": sklearn.__version__ == "1.7.2",
        "reference_raw_AP_exact": abs(
            reference_raw["average_precision"] - float(spec["evaluation"]["reference_raw_AP"])
        ) <= 1e-12,
        "reference_component_OOF_F1_exact": abs(
            reference_oof["metrics"]["f1"] - float(spec["evaluation"]["reference_component_OOF_F1"])
        ) <= 1e-12,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "test_path_constructed": False,
        "test_read": False,
    }
    positive = {key for key in checks if key not in {"test_path_constructed", "test_read"}}
    status = "pass" if all(checks[key] for key in positive) and not checks["test_path_constructed"] and not checks["test_read"] else "fail"
    payload = {
        "artifact": {"name": "S1-base ModernBERT GPU-training preflight", "version": "1.0.0", "status": status},
        "checks": checks,
        "spec_sha256": sha256_file(SPEC_PATH),
        "cache_summary_sha256": sha256_file(CACHE_SUMMARY_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "compute_executed": False,
        },
        "authorization": "ready_for_user_started_seed1001_validation_only_training" if status == "pass" else "blocked",
        "test": "locked",
    }
    atomic_json(payload, PREFLIGHT_PATH)
    return payload


def load_cache(split: str) -> dict[str, Any]:
    return {
        "input_ids": np.load(CACHE_ROOT / split / "input_ids.npy", mmap_mode="r"),
        "attention_mask": np.load(CACHE_ROOT / split / "attention_mask.npy", mmap_mode="r"),
        "labels": np.load(CACHE_ROOT / split / "labels.npy", mmap_mode="r"),
        "metadata": read_csv(CACHE_ROOT / split / "metadata.csv"),
    }


def set_determinism(torch: Any, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def move_batch(torch: Any, cache: dict[str, Any], indices: np.ndarray, device: Any) -> tuple[Any, Any, Any]:
    ids = torch.from_numpy(np.asarray(cache["input_ids"][indices], dtype=np.int64)).to(device, non_blocking=True)
    mask = torch.from_numpy(np.asarray(cache["attention_mask"][indices], dtype=np.int64)).to(device, non_blocking=True)
    labels = torch.from_numpy(np.asarray(cache["labels"][indices], dtype=np.int64)).to(device, non_blocking=True)
    return ids, mask, labels


def evaluate(torch: Any, model: Any, cache: dict[str, Any], *, device: Any, batch_size: int = 32) -> dict[str, Any]:
    model.eval()
    margins = np.empty(len(cache["labels"]), dtype=np.float64)
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(cache["labels"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(cache["labels"])), dtype=np.int64)
            ids, mask, labels = move_batch(torch, cache, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            logits = outputs.logits.detach().float().cpu().numpy()
            margins[indices] = logits[:, 1] - logits[:, 0]
            loss_sum += float(outputs.loss.detach().float()) * len(indices)
    labels_np = np.asarray(cache["labels"], dtype=np.int64)
    metrics = raw_metrics(labels_np, margins)
    metrics["loss"] = loss_sum / len(labels_np)
    model.train()
    return {"margins": margins, "metrics": metrics}


def selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(metrics["average_precision"]),
        float(metrics["auroc"]),
        -float(metrics["loss"]),
        -epoch,
    )


def finalize(
    *,
    torch: Any,
    model: Any,
    cache: dict[str, Any],
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    training_elapsed: float,
    device: Any,
) -> dict[str, Any]:
    best = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(best)
    evaluated = evaluate(torch, model, cache, device=device)
    margins = evaluated["margins"]
    labels = np.asarray(cache["labels"], dtype=np.int64)
    metadata = cache["metadata"]
    fold_rows = read_csv(resolve_project_path(spec["inputs"]["outer_fold_manifest"]))
    fold_map = {row["sample_id"]: row for row in fold_rows}
    if len(fold_map) != len(metadata):
        raise RuntimeError("outer-fold population drift")
    folds = np.empty(len(metadata), dtype=np.int64)
    components = np.empty(len(metadata), dtype=object)
    reference_margins = np.empty(len(metadata), dtype=np.float64)
    for index, meta in enumerate(metadata):
        row = fold_map.get(meta["sample_id"])
        if row is None:
            raise RuntimeError(f"missing outer fold: {meta['sample_id']}")
        if int(row["y_true"]) != int(meta["label_id"]) or row["source_clone_component_id"] != meta["component_id"]:
            raise RuntimeError(f"outer-fold metadata drift: {meta['sample_id']}")
        folds[index] = int(row["outer_fold"])
        components[index] = meta["component_id"]
        reference_margins[index] = float(row["ea4mp_behavior_bert_logit_margin"])
    candidate_oof = crossfit_one_feature(labels=labels, margins=margins, folds=folds, components=components, seed=int(spec["scope"]["seed"]))
    reference_oof = crossfit_one_feature(labels=labels, margins=reference_margins, folds=folds, components=components, seed=int(spec["scope"]["seed"]))
    reference_raw = raw_metrics(labels, reference_margins)
    if abs(reference_raw["average_precision"] - float(spec["evaluation"]["reference_raw_AP"])) > 1e-12:
        raise RuntimeError("reference raw AP did not reproduce")
    if abs(reference_oof["metrics"]["f1"] - float(spec["evaluation"]["reference_component_OOF_F1"])) > 1e-12:
        raise RuntimeError("reference OOF F1 did not reproduce")
    candidate_raw = evaluated["metrics"]
    ap_delta = float(candidate_raw["average_precision"] - reference_raw["average_precision"])
    f1_delta = float(candidate_oof["metrics"]["f1"] - reference_oof["metrics"]["f1"])
    gates = spec["evaluation"]["noninferiority_gate"]
    passed = ap_delta >= float(gates["raw_AP_delta_minimum"]) and f1_delta >= float(gates["component_OOF_F1_delta_minimum"])
    prediction_rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        prediction_rows.append(
            dict(meta)
            | {
                "outer_fold": int(folds[index]),
                "modernbert_logit_margin": float(margins[index]),
                "modernbert_oof_score": float(candidate_oof["scores"][index]),
                "modernbert_oof_prediction": int(candidate_oof["predictions"][index]),
                "reference_bert_oof_prediction": int(reference_oof["predictions"][index]),
            }
        )
    temporary = PREDICTIONS_PATH.with_name(f".{PREDICTIONS_PATH.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, PREDICTIONS_PATH)

    subgroup: dict[str, Any] = {}
    truncated = np.asarray([int(row["truncated"]) == 1 for row in metadata])
    for name, mask in (("full_at_512", ~truncated), ("truncated_at_512", truncated)):
        subgroup[name] = {
            "rows": int(mask.sum()),
            "malicious_rows": int(labels[mask].sum()),
            "candidate": confusion(labels[mask], candidate_oof["predictions"][mask]),
            "reference": confusion(labels[mask], reference_oof["predictions"][mask]),
        }
    payload = {
        "artifact": {"name": "S1-base ModernBERT seed1001 validation-only result", "version": "1.0.0", "status": "pass" if passed else "fail"},
        "model": {"id": spec["model"]["model_id"], "revision": spec["model"]["revision"], "seed": int(spec["scope"]["seed"]), "best_epoch": best_epoch},
        "candidate": {"raw": candidate_raw, "component_OOF": candidate_oof["metrics"]},
        "reference_EA4MP_BERT": {"raw": reference_raw, "component_OOF": reference_oof["metrics"]},
        "delta": {"raw_AP": ap_delta, "component_OOF_F1": f1_delta},
        "gate": {
            "raw_AP": ap_delta >= float(gates["raw_AP_delta_minimum"]),
            "component_OOF_F1": f1_delta >= float(gates["component_OOF_F1_delta_minimum"]),
            "overall": passed,
            "memory_encoder_authorized": passed,
        },
        "subgroups": subgroup,
        "history": history,
        "runtime": {"training_elapsed_this_invocation_seconds": training_elapsed, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)},
        "artifacts": {"best_model": str(BEST_MODEL_PATH), "best_model_sha256": sha256_file(BEST_MODEL_PATH), "validation_predictions": str(PREDICTIONS_PATH), "validation_predictions_sha256": sha256_file(PREDICTIONS_PATH)},
        "scope": {"train_packages": int(spec["scope"]["train_packages"]), "validation_packages": int(spec["scope"]["validation_packages"]), "test_read": False, "test_evaluated": False, "graph_or_PT_change": False},
        "next_gate": "freeze_behavior_unit_memory_encoder" if passed else "retain_P5_and_move_to_non_Mamba_S2",
    }
    atomic_json(payload, RESULT_PATH)
    return payload


def train() -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, ModernBertForSequenceClassification

    preflight = build_preflight()
    if preflight["artifact"]["status"] != "pass":
        raise RuntimeError("S1-base preflight failed")
    if RESULT_PATH.is_file():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    seed = int(spec["scope"]["seed"])
    set_determinism(torch, seed)
    device = torch.device("cuda:0")
    train_cache = load_cache("train")
    validation_cache = load_cache("validation")
    sample_ids = [row["sample_id"] for row in train_cache["metadata"]]
    snapshot = resolve_project_path(spec["model"]["local_snapshot"])
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.num_labels = 2
    config.reference_compile = False
    config._attn_implementation = "sdpa"
    config.use_cache = False
    model = ModernBertForSequenceClassification.from_pretrained(snapshot, config=config, local_files_only=True)
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    parameters = spec["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parameters["learning_rate"]), weight_decay=float(parameters["weight_decay"]))
    completed_epoch = 0
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    no_improve = 0
    if LAST_STATE_PATH.is_file():
        state = torch.load(LAST_STATE_PATH, map_location=device, weights_only=False)
        if state["spec_sha256"] != sha256_file(SPEC_PATH) or state["runner_sha256"] != sha256_file(SCRIPT_PATH):
            raise RuntimeError("resume configuration/script drift")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        completed_epoch = int(state["completed_epoch"])
        history = list(state["history"])
        best_epoch = int(state["best_epoch"])
        best_metrics = state["best_metrics"]
        no_improve = int(state["no_improve"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
        print(f"[S1-base RESUME] completed_epoch={completed_epoch} best_epoch={best_epoch}", flush=True)
    batch_size = int(parameters["batch_size"])
    max_epochs = int(parameters["maximum_epochs"])
    min_epochs = int(parameters["minimum_epochs"])
    patience = int(parameters["early_stopping_patience"])
    min_delta = float(parameters["minimum_AP_delta"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(completed_epoch + 1, max_epochs + 1):
        epoch_started = time.monotonic()
        order = stable_order(sample_ids, seed=seed, epoch=epoch)
        steps = (len(order) + batch_size - 1) // batch_size
        loss_sum = 0.0
        seen = 0
        for step, start in enumerate(range(0, len(order), batch_size), 1):
            indices = order[start : start + batch_size]
            ids, mask, labels = move_batch(torch, train_cache, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=ids, attention_mask=mask, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(parameters["gradient_clip_norm"]))
            optimizer.step()
            count = len(indices)
            loss_sum += float(output.loss.detach().float()) * count
            seen += count
            if step == 1 or step % 100 == 0 or step == steps:
                elapsed = time.monotonic() - epoch_started
                remaining = (steps - step) * elapsed / step
                print(f"[S1-base TRAIN] epoch={epoch}/{max_epochs} step={step}/{steps} loss={float(output.loss.detach()):.6f} elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m", flush=True)
        validation = evaluate(torch, model, validation_cache, device=device)
        metrics = validation["metrics"]
        epoch_row = {"epoch": epoch, "train_loss": loss_sum / seen, "validation": metrics, "elapsed_seconds": time.monotonic() - epoch_started}
        history.append(epoch_row)
        previous_ap = float(best_metrics["average_precision"]) if best_metrics is not None else -np.inf
        is_best = best_metrics is None or selection_key(metrics, epoch) > selection_key(best_metrics, best_epoch)
        significant = best_metrics is None or float(metrics["average_precision"]) >= previous_ap + min_delta
        if is_best:
            best_epoch = epoch
            best_metrics = dict(metrics)
            atomic_torch_save(torch, model.state_dict(), BEST_MODEL_PATH)
        no_improve = 0 if significant else no_improve + 1
        state = {
            "spec_sha256": sha256_file(SPEC_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "completed_epoch": epoch,
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "no_improve": no_improve,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        atomic_torch_save(torch, state, LAST_STATE_PATH)
        print(f"[S1-base EPOCH] {epoch}/{max_epochs} train_loss={epoch_row['train_loss']:.6f} val_AP={metrics['average_precision']:.6f} val_AUROC={metrics['auroc']:.6f} raw_F1={metrics['f1']:.6f} best={best_epoch} no_improve={no_improve}/{patience}", flush=True)
        if epoch >= min_epochs and no_improve >= patience:
            print(f"[S1-base EARLY STOP] validation AP no >= {min_delta:.6f} improvement for {patience} epochs", flush=True)
            break
    return finalize(torch=torch, model=model, cache=validation_cache, spec=spec, history=history, best_epoch=best_epoch, training_elapsed=time.monotonic() - started, device=device)


def status() -> None:
    print("S1-BASE MODERNBERT SEED-1001 STATUS (validation only; test locked)")
    print("=" * 78)
    cache = "PASS" if CACHE_SUMMARY_PATH.is_file() else "PENDING"
    preflight = "PASS" if PREFLIGHT_PATH.is_file() and json.loads(PREFLIGHT_PATH.read_text())["artifact"]["status"] == "pass" else "PENDING"
    if RESULT_PATH.is_file():
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        print(f"cache={cache} preflight={preflight} training=COMPLETE best_epoch={result['model']['best_epoch']} screen={result['artifact']['status'].upper()} test=LOCKED")
        print(f"candidate AP/OOF_F1={result['candidate']['raw']['average_precision']:.6f}/{result['candidate']['component_OOF']['f1']:.6f}")
        print(f"reference AP/OOF_F1={result['reference_EA4MP_BERT']['raw']['average_precision']:.6f}/{result['reference_EA4MP_BERT']['component_OOF']['f1']:.6f}")
        print(f"delta AP/F1={result['delta']['raw_AP']:+.6f}/{result['delta']['component_OOF_F1']:+.6f} memory_encoder={'AUTHORIZED' if result['gate']['memory_encoder_authorized'] else 'LOCKED'}")
        print(f"result={RESULT_PATH}")
        return
    if LAST_STATE_PATH.is_file():
        import torch
        state = torch.load(LAST_STATE_PATH, map_location="cpu", weights_only=False)
        print(f"cache={cache} preflight={preflight} training=RESUMABLE completed_epoch={state['completed_epoch']} best_epoch={state['best_epoch']} test=LOCKED")
    else:
        print(f"cache={cache} preflight={preflight} training=PENDING test=LOCKED")


def main() -> None:
    args = parse_args()
    if args.status:
        status(); return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        print(json.dumps(build_preflight(), ensure_ascii=False, indent=2, sort_keys=True)); return
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = train()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
