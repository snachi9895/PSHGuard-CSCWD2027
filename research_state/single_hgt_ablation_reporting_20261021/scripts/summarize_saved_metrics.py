"""Describe saved validation metrics; never load models or package predictions."""

from pathlib import Path
from decimal import Decimal, localcontext
import csv
import hashlib
import json
import math
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
OUT = BASE / "outputs/v1"
PROTO_PATH = BASE / "protocol/PROTOCOL_v1.json"
PROTO = json.loads(PROTO_PATH.read_text(encoding="utf-8"))
CONDITIONS = PROTO["conditions"]
SEEDS = PROTO["seeds"]
METRICS = PROTO["metrics"]
PROTOCOLS = PROTO["metric_protocols"]
EXP = "experiment_outputs/train_final_v10c_filtered700k_20260612/"
GRAPH_AUDIT = EXP + "s9b_exact_v2_graph_representation_ablation_20260819_v1/s9b_graph_representation_result_audit.json"
FUSION_AUDIT = EXP + "s9b_ablation_fusion_replay_validation_20260819_v1/s9bf_ablation_fusion_replay_audit.json"
P4_METRICS = EXP + "p4_calibrated_ensemble_validation_20260808_v1/p4_calibrated_ensemble_candidate_metrics.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def read_csv(rel):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def close(a, b, tolerance=1e-12):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def main():
    assert not any(OUT.iterdir()), "Do not overwrite a previous output; use a new version directory."
    before = read_json("research_state/single_hgt_ablation_reporting_20261021/verification/protected_inputs_before.json")
    for rel, expected in before.items():
        assert sha(ROOT / rel) == expected["sha256"], rel
    source = PROTO["input_metrics"]["path"]
    assert sha(ROOT / source) == PROTO["input_metrics"]["sha256"]
    graph = read_json(GRAPH_AUDIT)
    fusion = read_json(FUSION_AUDIT)
    for key, rel in [
        ("spec_sha256", "scripts/formal_training_v10c_20260612/s9b_graph_representation_ablation_spec_v1.json"),
        ("freeze_manifest_sha256", "scripts/formal_training_v10c_20260612/s9b_graph_representation_ablation_freeze_manifest_v1.json"),
    ]:
        assert sha(ROOT / rel) == graph["integrity"][key]
    assert fusion["input_sha256"][GRAPH_AUDIT] == sha(ROOT / GRAPH_AUDIT)
    fusion_spec = "scripts/formal_training_v10c_20260612/s9b_ablation_fusion_replay_spec_v1.json"
    assert fusion["input_sha256"][fusion_spec] == sha(ROOT / fusion_spec)
    assert Path(graph["outputs"]["metrics_csv"]).as_posix().endswith(source)
    assert graph["scope"]["validation_rows"] == 2137
    assert graph["scope"]["source_clone_components"] == 1588

    all_rows = read_csv(source)
    assert len(all_rows) == 28
    single = {}
    legacy = {}
    extracted = []
    count_checks = 0
    p4 = {row["candidate"]: row for row in read_csv(P4_METRICS)}
    for record_number, row in enumerate(all_rows, 1):
        protocol, condition, candidate = (row[x] for x in ("metric_protocol", "condition", "candidate"))
        assert protocol in PROTOCOLS and condition in CONDITIONS
        values = {metric: float(row[metric]) for metric in METRICS}
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in values.values())
        counts = {name: int(float(row[name])) for name in ("tn", "fp", "fn", "tp")}
        assert all(float(row[name]) == value and value >= 0 for name, value in counts.items())
        tn, fp, fn, tp = (counts[name] for name in ("tn", "fp", "fn", "tp"))
        assert tn + fp == 1082 and tp + fn == 1055
        for name, expected in {
            "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "false_positive_rate": fp / (fp + tn),
        }.items():
            close(values[name], expected)
            count_checks += 1
        entry = {
            "metric_protocol": protocol, "condition": condition, "candidate": candidate,
            "seed": None, **values, **counts,
            "source_path": source, "source_data_record": record_number,
        }
        if candidate == "three_seed_ensemble":
            assert protocol == "component_isolated_oof" and condition not in legacy
            legacy[condition] = entry
            for key in METRICS + list(counts):
                close(entry[key], graph["component_isolated_oof_three_seed_ensemble"][condition][key])
            continue
        seed = int(float(row["seed"]))
        assert seed in SEEDS and candidate == f"seed{seed}"
        entry["seed"] = seed
        key = (protocol, condition, seed)
        assert key not in single
        single[key] = entry
        extracted.append(entry)
        if protocol == "component_isolated_oof" and condition == "practical":
            for name in METRICS + list(counts):
                reference = p4[f"single_platt_seed{seed}"]
                if name == "false_positive_rate":
                    p4_fp, p4_tn = float(reference["oof_fp"]), float(reference["oof_tn"])
                    close(entry[name], p4_fp / (p4_fp + p4_tn))
                else:
                    close(entry[name], reference[f"oof_{name}"])
    assert len(single) == 24 and set(legacy) == set(CONDITIONS)
    assert set(single) == {(p, c, s) for p in PROTOCOLS for c in CONDITIONS for s in SEEDS}

    summaries = []
    summary_map = {}
    decimal_checks = []
    for protocol in PROTOCOLS:
        for condition in CONDITIONS:
            for metric in METRICS:
                values = [single[(protocol, condition, seed)][metric] for seed in SEEDS]
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                with localcontext() as ctx:
                    ctx.prec = 50
                    dv = [Decimal(str(value)) for value in values]
                    dm = sum(dv) / Decimal(3)
                    ds = (sum((v - dm) ** 2 for v in dv) / Decimal(2)).sqrt()
                close(mean, float(dm))
                close(sd, float(ds))
                decimal_checks.append({"protocol": protocol, "condition": condition, "metric": metric,
                                       "mean_absolute_error": abs(mean-float(dm)), "sd_absolute_error": abs(sd-float(ds))})
                if protocol == "raw_argmax":
                    saved = graph["raw_three_seed_summary"][condition][metric]
                    close(mean, saved["mean"])
                    close(sd, saved["sample_standard_deviation"])
                record = {"metric_protocol": protocol, "condition": condition, "metric": metric, "runs": 3,
                          "mean_ratio": mean, "sample_sd_ratio": sd, "mean_percent": 100*mean,
                          "sample_sd_percentage_points": 100*sd,
                          **{f"seed{seed}_ratio": values[i] for i, seed in enumerate(SEEDS)}}
                summaries.append(record)
                summary_map[(protocol, condition, metric)] = record

    contrasts = []
    for protocol in PROTOCOLS:
        for condition in CONDITIONS[1:]:
            for metric in METRICS:
                differences = [single[(protocol, "practical", seed)][metric] - single[(protocol, condition, seed)][metric] for seed in SEEDS]
                mean = statistics.mean(differences)
                close(mean, summary_map[(protocol, "practical", metric)]["mean_ratio"]-summary_map[(protocol, condition, metric)]["mean_ratio"])
                contrasts.append({"metric_protocol": protocol, "ablation": condition, "metric": metric,
                                  "full_minus_ablation_mean_pp": 100*mean,
                                  **{f"seed{seed}_delta_pp": 100*differences[i] for i, seed in enumerate(SEEDS)},
                                  "positive_seed_count": sum(d > 1e-14 for d in differences),
                                  "negative_seed_count": sum(d < -1e-14 for d in differences),
                                  "tied_seed_count": sum(abs(d) <= 1e-14 for d in differences),
                                  "interpretation": "descriptive only; no corresponding single-HGT CI computed"})

    saved = {"status": "copied from saved audits; not recomputed inference",
             "graph_audit": GRAPH_AUDIT, "fusion_audit": FUSION_AUDIT,
             "original_raw_ap_intervals": {condition: graph["paired_bootstrap"][f"raw_ap/{condition}"] for condition in CONDITIONS[1:]},
             "legacy_three_hgt_ensemble_classification_intervals": {condition: graph["paired_bootstrap"][f"component_isolated_oof/{condition}"] for condition in CONDITIONS[1:]},
             "same_seed_one_graph_one_sequence_fusion": {
                 key: fusion[key] for key in ("three_seed_summary_mean_sample_sd", "full_minus_ablation_mean_deltas", "hierarchical_seed_component_bootstrap")}}
    for condition in CONDITIONS[1:]:
        interval = saved["original_raw_ap_intervals"][condition]
        assert interval["ci95_percentile_lower"] <= 0 <= interval["ci95_percentile_upper"]

    write_csv(OUT / "single_hgt_per_seed_metrics.csv", extracted)
    write_csv(OUT / "single_hgt_mean_sample_sd.csv", summaries)
    write_csv(OUT / "full_minus_ablation_descriptive.csv", contrasts)
    write_csv(OUT / "legacy_three_hgt_ensemble_metrics.csv", list(legacy.values()))
    write_json(OUT / "saved_uncertainty_and_fusion_reference.json", saved)
    write_json(OUT / "summary.json", {"protocol_sha256": sha(PROTO_PATH), "summaries": summaries,
                                     "descriptive_contrasts": contrasts,
                                     "single_model_confidence_intervals_computed": False})

    labels = {"practical": "完整图", "without_event": "去Event", "without_param_context": "去Param/context", "ast_module_only": "仅Module+AST"}
    names = {"average_precision": "AP", "auroc": "AUROC", "precision": "Precision", "recall": "Recall", "f1": "F1", "false_positive_rate": "FPR"}
    lines = ["# 四种图配置的单HGT结果", "", "每项为三个seed的均值±样本标准差；均值单位为%，标准差单位为百分点。FPR越低越好，其余指标越高越好。", ""]
    for protocol, title in [("raw_argmax", "原始分数与原工作点"), ("component_isolated_oof", "单HGT校准OOF评价")]:
        lines.extend([f"## {title}", "", "| 配置 | " + " | ".join(names[m] for m in METRICS) + " |", "|---|"+"---:|"*len(METRICS)])
        for condition in CONDITIONS:
            cells = []
            for metric in METRICS:
                v=summary_map[(protocol,condition,metric)]
                cells.append(f"{v['mean_percent']:.3f} ± {v['sample_sd_percentage_points']:.3f}")
            lines.append("| "+labels[condition]+" | "+" | ".join(cells)+" |")
        lines.append("")
    lines.extend(["## 描述性F1差值", "", "完整图减对应删除配置；正数表示完整图F1较高。没有为这些单HGT差值新算置信区间或显著性检验。", "", "| 对照 | OOF F1均值差（百分点） | 三个seed的差值（百分点） |", "|---|---:|---|"])
    for row in contrasts:
        if row["metric_protocol"] == "component_isolated_oof" and row["metric"] == "f1":
            lines.append(f"| {labels[row['ablation']]} | {row['full_minus_ablation_mean_pp']:+.3f} | "+", ".join(f"{row[f'seed{s}_delta_pp']:+.3f}" for s in SEEDS)+" |")
    lines.extend(["", "原始AP仍为主要指标，其三个既有配对区间均包含0。不能用OOF分类均值取代该不明确结果。旧三HGT集成及同seed双分支结果另存参考JSON，不能混作上表的单模型证据。", ""])
    (OUT / "RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")

    mismatches = [rel for rel, expected in before.items() if sha(ROOT / rel) != expected["sha256"]]
    assert not mismatches, mismatches
    verification = {"status": "passed", "single_model_rows": len(single), "legacy_ensemble_rows": len(legacy),
                    "population_consistency": True, "confusion_metric_checks": count_checks,
                    "full_graph_P4_single_seed_crosschecks": 3,
                    "raw_summary_mean_and_sample_sd_checks": 48,
                    "independent_decimal_mean_and_sd_checks": decimal_checks,
                    "source_lineage_spec_freeze_and_fusion_checks": True,
                    "protected_file_count": len(before), "protected_mismatches": mismatches,
                    "original_metrics_CSV_sha256": sha(ROOT / source),
                    "protocol_sha256": sha(PROTO_PATH), "script_sha256": sha(Path(__file__)),
                    "python_version": sys.version,
                    "predictions_read": False, "models_executed": False, "new_inference_statistics": False,
                    "manuscript_changed": False, "final_test_accessed": False}
    write_json(BASE / "verification/verification_result.json", verification)
    write_json(BASE / "verification/output_manifest.json", {
        str(path.relative_to(BASE)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(OUT.iterdir()) if path.is_file()})
    print((OUT / "RESULT_TABLES.md").read_text(encoding="utf-8"))
    print(f"Verified {len(before)} protected files unchanged; no CI, model execution or final-test access.")


if __name__ == "__main__":
    main()
