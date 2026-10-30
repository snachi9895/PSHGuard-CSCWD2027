#!/usr/bin/env python3
"""Resumable S1-base ModernBERT confirmatory training for seeds 1002/1003."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_s1_base_modernbert_screen as base


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SCRIPT_DIR = SCRIPT_PATH.parent
SPEC_PATH = SCRIPT_DIR / "s1_base_modernbert_confirmatory_spec_v1.json"
FREEZE_PATH = SCRIPT_DIR / "s1_base_modernbert_confirmatory_freeze_manifest_v1.json"
BASE_SPEC_PATH = SCRIPT_DIR / "s1_base_modernbert_screen_spec_v1.json"
BASE_RUNNER_PATH = SCRIPT_DIR / "run_s1_base_modernbert_screen.py"
MATRIX_ROOT = (
    PROJECT_ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_confirmatory_seeds1002_1003_validation_20260816_v1"
)
BASE_ROOT = (
    PROJECT_ROOT
    / "experiment_outputs/train_final_v10c_filtered700k_20260612"
    / "s1_base_modernbert_seed1001_validation_20260815_v1"
)
BASE_CACHE_ROOT = BASE_ROOT / "cache"
BASE_CACHE_AUDIT = BASE_CACHE_ROOT / "cache_audit.json"
SEED1001_RESULT = BASE_ROOT / "validation_result.json"
ALLOWED_SEEDS = {1002, 1003}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=sorted(ALLOWED_SEEDS), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def seed_paths(seed: int) -> dict[str, Path]:
    root = MATRIX_ROOT / f"seed{seed}"
    return {
        "root": root,
        "preflight": root / "preflight_audit.json",
        "last": root / "last_epoch_state.pt",
        "best": root / "best_validation_model.pt",
        "predictions": root / "validation_predictions.csv",
        "result": root / "validation_result.json",
        "lock": root / ".training.lock",
        "log": root / f"s1_base_modernbert_seed{seed}.run.log",
    }


def frozen_paths() -> dict[str, Path]:
    base_spec = load_json(BASE_SPEC_PATH)
    snapshot = base.resolve_project_path(base_spec["model"]["local_snapshot"])
    fold_manifest = base.resolve_project_path(base_spec["inputs"]["outer_fold_manifest"])
    paths = [
        SPEC_PATH,
        SCRIPT_PATH,
        SCRIPT_DIR / "run_s1_base_modernbert_confirmatory_matrix.sh",
        BASE_SPEC_PATH,
        BASE_RUNNER_PATH,
        BASE_CACHE_AUDIT,
        SEED1001_RESULT,
        fold_manifest,
        snapshot / "model.safetensors",
        snapshot / "tokenizer.json",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): path for path in paths}


def validate_freeze() -> tuple[dict[str, Any], dict[str, str]]:
    freeze = load_json(FREEZE_PATH)
    if freeze.get("artifact", {}).get("status") != "frozen_before_confirmatory_gpu_training":
        raise RuntimeError("confirmatory freeze manifest is not frozen")
    paths = frozen_paths()
    expected = freeze.get("sha256", {})
    if set(paths) != set(expected):
        raise RuntimeError(
            f"freeze path mismatch missing={sorted(set(paths)-set(expected))} "
            f"extra={sorted(set(expected)-set(paths))}"
        )
    observed: dict[str, str] = {}
    for relative, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256_file(path)
        if observed[relative] != expected[relative]:
            raise RuntimeError(
                f"frozen SHA mismatch: {relative}: expected={expected[relative]} "
                f"observed={observed[relative]}"
            )
    return freeze, observed


def effective_spec(seed: int) -> dict[str, Any]:
    confirm = load_json(SPEC_PATH)
    base_spec = load_json(BASE_SPEC_PATH)
    if confirm.get("artifact", {}).get("status") != "frozen_before_confirmatory_gpu_training":
        raise RuntimeError("confirmatory specification is not frozen")
    if seed not in {int(value) for value in confirm["scope"]["seeds"]}:
        raise RuntimeError(f"seed {seed} is not authorized")
    if sha256_file(BASE_SPEC_PATH) != confirm["frozen_reference"]["base_spec_sha256"]:
        raise RuntimeError("base specification drifted")
    if sha256_file(BASE_RUNNER_PATH) != confirm["frozen_reference"]["base_runner_sha256"]:
        raise RuntimeError("base runner drifted")
    if sha256_file(BASE_CACHE_AUDIT) != confirm["frozen_reference"]["cache_audit_sha256"]:
        raise RuntimeError("base cache audit drifted")
    if base_spec["training"] != confirm["training"]:
        raise RuntimeError("confirmatory training hyperparameters differ from S1-base")
    result1001 = load_json(SEED1001_RESULT)
    if result1001.get("artifact", {}).get("status") != "pass":
        raise RuntimeError("seed1001 context is not PASS")
    spec = copy.deepcopy(base_spec)
    spec["scope"]["seed"] = seed
    spec["scope"]["screen_only"] = False
    spec["decision"] = {
        "if_pass": "retain this seed for the separately frozen three-seed fusion confirmation audit",
        "if_fail": "retain the result as a confirmatory seed; no result may be silently discarded",
        "S1_base_itself_is_claimed_innovation": False,
    }
    return spec


def cache_integrity(base_spec: dict[str, Any]) -> dict[str, Any]:
    original_cache_root = base.CACHE_ROOT
    original_cache_summary = base.CACHE_SUMMARY_PATH
    original_spec_path = base.SPEC_PATH
    try:
        base.CACHE_ROOT = BASE_CACHE_ROOT
        base.CACHE_SUMMARY_PATH = BASE_CACHE_AUDIT
        base.SPEC_PATH = BASE_SPEC_PATH
        return base.cache_integrity(base_spec)
    finally:
        base.CACHE_ROOT = original_cache_root
        base.CACHE_SUMMARY_PATH = original_cache_summary
        base.SPEC_PATH = original_spec_path


def load_cache(split: str) -> dict[str, Any]:
    original = base.CACHE_ROOT
    try:
        base.CACHE_ROOT = BASE_CACHE_ROOT
        return base.load_cache(split)
    finally:
        base.CACHE_ROOT = original


def build_preflight(seed: int) -> dict[str, Any]:
    import sklearn
    import torch
    import transformers

    _, observed_hashes = validate_freeze()
    spec = effective_spec(seed)
    base_spec = load_json(BASE_SPEC_PATH)
    cache = cache_integrity(base_spec)
    snapshot = base.resolve_project_path(spec["model"]["local_snapshot"])
    validation_cache = load_cache("validation")
    fold_rows = base.read_csv(base.resolve_project_path(spec["inputs"]["outer_fold_manifest"]))
    fold_map = {row["sample_id"]: row for row in fold_rows}
    metadata = validation_cache["metadata"]
    labels = np.asarray(validation_cache["labels"], dtype=np.int64)
    margins = np.asarray(
        [float(fold_map[row["sample_id"]]["ea4mp_behavior_bert_logit_margin"]) for row in metadata],
        dtype=np.float64,
    )
    folds = np.asarray([int(fold_map[row["sample_id"]]["outer_fold"]) for row in metadata])
    components = np.asarray([row["component_id"] for row in metadata], dtype=object)
    reference_raw = base.raw_metrics(labels, margins)
    reference_oof = base.crossfit_one_feature(
        labels=labels,
        margins=margins,
        folds=folds,
        components=components,
        seed=seed,
    )
    paths = seed_paths(seed)
    checks = {
        "seed_authorized": seed in ALLOWED_SEEDS,
        "output_isolated_from_seed1001": BASE_ROOT not in paths["root"].parents
        and paths["root"] != BASE_ROOT,
        "cache_pass": cache["artifact"]["status"] == "pass",
        "train_population_exact": int(cache["splits"]["train"]["packages"]) == 16613,
        "validation_population_exact": int(cache["splits"]["validation"]["packages"]) == 2137,
        "model_weight_sha256": sha256_file(snapshot / "model.safetensors")
        == spec["model"]["model_weight_sha256"],
        "tokenizer_sha256": sha256_file(snapshot / "tokenizer.json")
        == spec["model"]["tokenizer_sha256"],
        "transformers_4_57_6": transformers.__version__ == "4.57.6",
        "torch_2_11_cuda128": torch.__version__ == "2.11.0+cu128",
        "sklearn_1_7_2": sklearn.__version__ == "1.7.2",
        "reference_raw_AP_exact": abs(
            reference_raw["average_precision"] - float(spec["evaluation"]["reference_raw_AP"])
        ) <= 1e-12,
        "reference_component_OOF_F1_exact": abs(
            reference_oof["metrics"]["f1"]
            - float(spec["evaluation"]["reference_component_OOF_F1"])
        ) <= 1e-12,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "test_path_constructed": False,
        "test_read": False,
    }
    positive = {key for key in checks if key not in {"test_path_constructed", "test_read"}}
    status = (
        "pass"
        if all(checks[key] for key in positive)
        and not checks["test_path_constructed"]
        and not checks["test_read"]
        else "fail"
    )
    payload = {
        "artifact": {
            "name": f"S1-base ModernBERT seed{seed} confirmatory preflight",
            "version": "1.0.0",
            "status": status,
        },
        "checks": checks,
        "seed": seed,
        "matrix_spec_sha256": sha256_file(SPEC_PATH),
        "base_spec_sha256": sha256_file(BASE_SPEC_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
        "input_sha256": observed_hashes,
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_total_bytes": torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
            "compute_executed": False,
        },
        "authorization": f"ready_for_user_started_seed{seed}_validation_only_training"
        if status == "pass"
        else "blocked",
        "test": "locked",
    }
    base.atomic_json(payload, paths["preflight"])
    return payload


def bind_base_output_paths(paths: dict[str, Path]) -> dict[str, Path]:
    originals = {
        "BEST_MODEL_PATH": base.BEST_MODEL_PATH,
        "PREDICTIONS_PATH": base.PREDICTIONS_PATH,
        "RESULT_PATH": base.RESULT_PATH,
    }
    base.BEST_MODEL_PATH = paths["best"]
    base.PREDICTIONS_PATH = paths["predictions"]
    base.RESULT_PATH = paths["result"]
    return originals


def restore_base_output_paths(originals: dict[str, Path]) -> None:
    for name, path in originals.items():
        setattr(base, name, path)


def finalize_seed(
    *,
    seed: int,
    torch: Any,
    model: Any,
    validation_cache: dict[str, Any],
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    training_elapsed: float,
    device: Any,
    paths: dict[str, Path],
) -> dict[str, Any]:
    originals = bind_base_output_paths(paths)
    try:
        payload = base.finalize(
            torch=torch,
            model=model,
            cache=validation_cache,
            spec=spec,
            history=history,
            best_epoch=best_epoch,
            training_elapsed=training_elapsed,
            device=device,
        )
    finally:
        restore_base_output_paths(originals)
    payload["artifact"]["name"] = (
        f"S1-base ModernBERT seed{seed} confirmatory validation-only result"
    )
    payload["matrix"] = {
        "seed": seed,
        "matrix_spec_sha256": sha256_file(SPEC_PATH),
        "base_spec_sha256": sha256_file(BASE_SPEC_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
        "test": "locked",
    }
    payload["next_gate"] = "wait_for_both_confirmatory_seeds_then_run_frozen_fusion_audit"
    base.atomic_json(payload, paths["result"])
    return payload


def train(seed: int) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, ModernBertForSequenceClassification

    paths = seed_paths(seed)
    paths["root"].mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(seed)
    if preflight["artifact"]["status"] != "pass":
        raise RuntimeError(f"seed{seed} preflight failed")
    if paths["result"].is_file():
        return load_json(paths["result"])
    spec = effective_spec(seed)
    base.set_determinism(torch, seed)
    device = torch.device("cuda:0")
    train_cache = load_cache("train")
    validation_cache = load_cache("validation")
    sample_ids = [row["sample_id"] for row in train_cache["metadata"]]
    snapshot = base.resolve_project_path(spec["model"]["local_snapshot"])
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.num_labels = 2
    config.reference_compile = False
    config._attn_implementation = "sdpa"
    config.use_cache = False
    model = ModernBertForSequenceClassification.from_pretrained(
        snapshot, config=config, local_files_only=True
    )
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    parameters = spec["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    completed_epoch = 0
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    no_improve = 0
    if paths["last"].is_file():
        state = torch.load(paths["last"], map_location=device, weights_only=False)
        expected = {
            "matrix_spec_sha256": sha256_file(SPEC_PATH),
            "base_spec_sha256": sha256_file(BASE_SPEC_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "seed": seed,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"seed{seed} resume configuration/script drift")
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
        print(
            f"[S1-CONFIRM RESUME] seed={seed} completed_epoch={completed_epoch} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    batch_size = int(parameters["batch_size"])
    max_epochs = int(parameters["maximum_epochs"])
    min_epochs = int(parameters["minimum_epochs"])
    patience = int(parameters["early_stopping_patience"])
    min_delta = float(parameters["minimum_AP_delta"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(completed_epoch + 1, max_epochs + 1):
        epoch_started = time.monotonic()
        order = base.stable_order(sample_ids, seed=seed, epoch=epoch)
        steps = (len(order) + batch_size - 1) // batch_size
        loss_sum = 0.0
        seen = 0
        for step, start in enumerate(range(0, len(order), batch_size), 1):
            indices = order[start : start + batch_size]
            ids, mask, labels = base.move_batch(torch, train_cache, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=ids, attention_mask=mask, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(parameters["gradient_clip_norm"])
            )
            optimizer.step()
            count = len(indices)
            loss_sum += float(output.loss.detach().float()) * count
            seen += count
            if step == 1 or step % 100 == 0 or step == steps:
                elapsed = time.monotonic() - epoch_started
                remaining = (steps - step) * elapsed / step
                print(
                    f"[S1-CONFIRM TRAIN] seed={seed} epoch={epoch}/{max_epochs} "
                    f"step={step}/{steps} loss={float(output.loss.detach()):.6f} "
                    f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                    flush=True,
                )
        validation = base.evaluate(
            torch, model, validation_cache, device=device
        )
        metrics = validation["metrics"]
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "validation": metrics,
            "elapsed_seconds": time.monotonic() - epoch_started,
        }
        history.append(epoch_row)
        previous_ap = float(best_metrics["average_precision"]) if best_metrics else -np.inf
        is_best = best_metrics is None or base.selection_key(
            metrics, epoch
        ) > base.selection_key(best_metrics, best_epoch)
        significant = best_metrics is None or float(metrics["average_precision"]) >= (
            previous_ap + min_delta
        )
        if is_best:
            best_epoch = epoch
            best_metrics = dict(metrics)
            base.atomic_torch_save(torch, model.state_dict(), paths["best"])
        no_improve = 0 if significant else no_improve + 1
        state = {
            "matrix_spec_sha256": sha256_file(SPEC_PATH),
            "base_spec_sha256": sha256_file(BASE_SPEC_PATH),
            "runner_sha256": sha256_file(SCRIPT_PATH),
            "seed": seed,
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
        base.atomic_torch_save(torch, state, paths["last"])
        print(
            f"[S1-CONFIRM EPOCH] seed={seed} {epoch}/{max_epochs} "
            f"train_loss={epoch_row['train_loss']:.6f} "
            f"val_AP={metrics['average_precision']:.6f} "
            f"val_AUROC={metrics['auroc']:.6f} raw_F1={metrics['f1']:.6f} "
            f"best={best_epoch} no_improve={no_improve}/{patience}",
            flush=True,
        )
        if epoch >= min_epochs and no_improve >= patience:
            print(
                f"[S1-CONFIRM EARLY STOP] seed={seed} validation AP no >= "
                f"{min_delta:.6f} improvement for {patience} epochs",
                flush=True,
            )
            break
    return finalize_seed(
        seed=seed,
        torch=torch,
        model=model,
        validation_cache=validation_cache,
        spec=spec,
        history=history,
        best_epoch=best_epoch,
        training_elapsed=time.monotonic() - started,
        device=device,
        paths=paths,
    )


def status(seed: int) -> None:
    paths = seed_paths(seed)
    print(
        f"S1-BASE MODERNBERT CONFIRMATORY SEED-{seed} STATUS "
        "(validation only; test locked)"
    )
    print("=" * 88)
    preflight = "PENDING"
    if paths["preflight"].is_file():
        preflight = str(load_json(paths["preflight"])["artifact"]["status"]).upper()
    if paths["result"].is_file():
        result = load_json(paths["result"])
        print(
            f"cache=REUSED_PASS preflight={preflight} training=COMPLETE "
            f"best_epoch={result['model']['best_epoch']} "
            f"screen={result['artifact']['status'].upper()} test=LOCKED"
        )
        print(
            f"candidate AP/OOF_F1={result['candidate']['raw']['average_precision']:.6f}/"
            f"{result['candidate']['component_OOF']['f1']:.6f}"
        )
        print(f"result={paths['result']}")
        return
    if paths["last"].is_file():
        import torch

        state = torch.load(paths["last"], map_location="cpu", weights_only=False)
        print(
            f"cache=REUSED_PASS preflight={preflight} training=RESUMABLE "
            f"completed_epoch={state['completed_epoch']} "
            f"best_epoch={state['best_epoch']} test=LOCKED"
        )
    else:
        print(
            f"cache=REUSED_PASS preflight={preflight} training=PENDING test=LOCKED"
        )


def main() -> None:
    args = parse_args()
    if args.status:
        status(args.seed)
        return
    paths = seed_paths(args.seed)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if args.preflight:
        print(
            json.dumps(
                build_preflight(args.seed), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return
    with paths["lock"].open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"seed{args.seed} training already owns the lock") from error
        result = train(args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
