"""Independent sklearn replay of emitted tables; no imports from analysis runner."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT/'paper1015-Frontiers'
RUN = PROJECT/'analysis/runs/joint_size_20261015_v1/attempts/20261015T131833_740493Z'
OUT = PROJECT/'analysis/audits/20261015_joint_size_result_verification_v1'
MODELS = [(s, k) for s in ('graph', 'modernbert', 'same_seed_pair') for k in (1001, 1002, 1003)] + [('saved_fixed_LR', None)]
METRICS = ('auroc', 'average_precision', 'recall', 'fpr', 'precision', 'f1')
VIEWS = ('raw_unweighted', 'retained_unweighted', 'retained_overlap_weighted')


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    deps = {}
    def verify(rec):
        p = ROOT/rec['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], rec['path']
        deps[rec['path']] = rec['sha256']
    complete = json.loads((RUN/'complete.json').read_text())
    assert complete['status'] == 'COMPLETE'
    for rec in complete['outputs']:
        verify(rec)
    for rec in json.loads((RUN/'input_manifest.json').read_text()):
        verify(rec)
    data = read(PROJECT/'analysis/audits/20261015_validation_inputs_v3/validation_inputs_joined.csv')
    metrics = read(RUN/'metrics.csv')
    differences = read(RUN/'paired_comparisons.csv')
    coverage = read(RUN/'population_coverage.csv')
    cells_output = read(RUN/'joint_cell_support.csv')
    draws = np.load(RUN/'bootstrap_draw_metrics.npz')['values']
    groups = sorted({r['source_clone_component_id'] for r in data})
    group_map = {g: i for i, g in enumerate(groups)}
    checked, max_error = 0, 0.
    all_point_values = {}
    populations = {}
    for definition, field in [('primary', 'graph_original_py_files'), ('fingerprint_sensitivity', 'fingerprint_python_file_count')]:
        for source in ('guo', 'datadog'):
            rows = [r for r in data if r['upstream_source'] in ('benign', source)]
            y = np.array([int(r['y_true']) for r in rows])
            comp = np.array([r['source_clone_component_id'] for r in rows])
            # Explicit comparisons, independent of runner's searchsorted binning.
            f = np.array([int(r[field]) for r in rows])
            b = np.array([int(r['total_python_bytes']) for r in rows])
            c = np.where(f == 1, 0, np.where(f <= 4, 1, np.where(f <= 16, 2, 3)))*3
            c += np.where(b <= 4096, 0, np.where(b <= 32768, 1, 2))
            eligible = []
            for cell in range(12):
                ns = [int(((c == cell) & (y == cls)).sum()) for cls in (0, 1)]
                gs = [len(set(comp[(c == cell) & (y == cls)])) for cls in (0, 1)]
                ref = next(r for r in cells_output if r['definition'] == definition and r['contrast'] == source+'_vs_benign' and int(r['cell']) == cell)
                assert ns == [int(ref['benign_rows']), int(ref['malicious_rows'])]
                assert gs == [int(ref['benign_components']), int(ref['malicious_components'])]
                ok = min(ns) >= 5 and min(gs) >= 2
                assert ok == (ref['status'] == 'ELIGIBLE')
                if ok:
                    eligible.append(cell)
            def weights(mult):
                w = np.zeros(len(y))
                for cell in eligible:
                    mask = c == cell
                    sums = [mult[mask & (y == cls)].sum() for cls in (0, 1)]
                    if min(sums) == 0:
                        return None
                    for cls in (0, 1):
                        take = mask & (y == cls)
                        w[take] = mult[take]*min(sums)/sums[cls]
                return w
            wlist = [np.ones(len(y)), np.isin(c, eligible).astype(float), weights(np.ones(len(y)))]
            populations[source] = (rows, y, c, eligible, weights, comp) if definition == 'primary' else populations.get(source)
            for view, w in zip(VIEWS, wlist):
                for cls in (0, 1):
                    ref = next(r for r in coverage if r['definition'] == definition and r['contrast'] == source+'_vs_benign' and r['view'] == view and int(r['label']) == cls)
                    mask = (y == cls) & (w > 0)
                    assert int(ref['retained_rows']) == int(mask.sum())
                    assert int(ref['retained_components']) == len(set(comp[mask]))
                    assert abs(float(ref['weight_sum'])-w[mask].sum()) < 1e-10
                for system, seed in MODELS:
                    score = np.array([float(r[f'seed{seed}_{system}_score'] if seed else r['saved_fixed_LR_score']) for r in rows])
                    expected = [roc_auc_score(y, score, sample_weight=w), average_precision_score(y, score, sample_weight=w)]
                    if seed:
                        pred = np.array([int(r[f'seed{seed}_{system}_prediction']) for r in rows])
                        tp = w[(y == 1) & (pred == 1)].sum()
                        fp = w[(y == 0) & (pred == 1)].sum()
                        p, n = w[y == 1].sum(), w[y == 0].sum()
                        expected += [tp/p, fp/n, tp/(tp+fp) if tp+fp else np.nan, 2*tp/(p+tp+fp)]
                    for name, value in zip(METRICS, expected):
                        seed_name = str(seed) if seed else 'single_fixed'
                        ref = next(r for r in metrics if r['definition'] == definition and r['contrast'] == source+'_vs_benign' and r['view'] == view and r['system'] == system and r['seed'] == seed_name and r['metric'] == name)
                        error = abs(float(ref['point'])-value)
                        assert error < 1e-12, (definition, source, system, seed, name, error)
                        max_error = max(max_error, error)
                        checked += 1
                        all_point_values[definition, source, view, system, seed_name, name] = value
    # Recreate exact first 3 component draws, including sparse-cell invalidation.
    rng = np.random.Generator(np.random.PCG64(20261015))
    for rep in range(3):
        mult_all = np.bincount(rng.integers(0, 1588, size=1588), minlength=1588)
        for pi, source in enumerate(('guo', 'datadog')):
            rows, y, c, eligible, _, comp = populations[source]
            mult = mult_all[[group_map[g] for g in comp]].astype(float)
            w = np.zeros(len(y))
            valid = True
            for cell in eligible:
                mass = [mult[(c == cell) & (y == cls)].sum() for cls in (0, 1)]
                if not min(mass):
                    valid = False
                    break
                for cls in (0, 1):
                    mask = (c == cell) & (y == cls)
                    w[mask] = mult[mask]*min(mass)/mass[cls]
            if not valid:
                assert np.isnan(draws[rep, pi, 2]).all()
                continue
            for mi, (system, seed) in enumerate(MODELS):
                score = np.array([float(r[f'seed{seed}_{system}_score'] if seed else r['saved_fixed_LR_score']) for r in rows])
                expected = [roc_auc_score(y, score, sample_weight=w), average_precision_score(y, score, sample_weight=w)]
                np.testing.assert_allclose(draws[rep, pi, 2, mi, :2], expected, atol=1e-12, rtol=0)
    ci_checked = 0
    for row in [r for r in metrics if r['definition'] == 'primary']:
        pi = ('guo_vs_benign', 'datadog_vs_benign').index(row['contrast'])
        vi, mi = VIEWS.index(row['view']), METRICS.index(row['metric'])
        ids = [i for i, (s, k) in enumerate(MODELS) if s == row['system'] and (row['seed'] == 'three_seed_mean' or str(k or 'single_fixed') == row['seed'])]
        values = draws[:, pi, vi, ids, mi].mean(axis=-1)
        values = values[np.isfinite(values)]
        assert len(values) == int(row['bootstrap_valid'])
        if len(values) < 1900:
            assert row['status'] == 'BOOTSTRAP_UNSTABLE'
            assert not row['ci95_lower'] and not row['ci95_upper']
        else:
            expected = np.quantile(values, [.025, .975], method='linear')
            np.testing.assert_allclose([float(row['ci95_lower']), float(row['ci95_upper'])], expected, atol=1e-12, rtol=0)
        ci_checked += 1
    sensitivity = []
    for source in ('guo', 'datadog'):
        for metric in METRICS:
            changes = []
            for system in ('graph', 'modernbert', 'same_seed_pair', 'saved_fixed_LR'):
                seed_ids = (1001, 1002, 1003) if system != 'saved_fixed_LR' else ('single_fixed',)
                vals = []
                for seed in seed_ids:
                    k = (source, 'retained_overlap_weighted', system, str(seed), metric)
                    if ('primary', *k) in all_point_values:
                        vals.append(all_point_values[('fingerprint_sensitivity',)+k]-all_point_values[('primary',)+k])
                if vals:
                    changes.append(abs(np.mean(vals)))
            sensitivity.append({'contrast': source+'_vs_benign', 'metric': metric, 'maximum_absolute_point_change': max(changes)})
    result = {'status': 'PASS', 'completed_attempt': str(RUN.relative_to(ROOT)),
              'independent_metric_values_checked': checked, 'maximum_absolute_error': max_error,
              'CI_rows_checked': ci_checked, 'exact_component_bootstrap_draws_replayed': 3,
              'verified_dependency_hashes': len(deps), 'sensitivity_point_changes': sensitivity,
              'new_models_fitted': 0, 'test_predictions_read': 0}
    with (OUT/'verification.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with (OUT/'input_hashes.json').open('x', encoding='utf-8') as f:
        json.dump(deps, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
