"""Frozen validation-only joint-size diagnostic. No model runners or source samples.

Use --run explicitly. Every invocation creates a new immutable attempt directory.
Metrics use saved probabilities/decisions; bootstrap uses whole clone components.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'paper1015-Frontiers'
PROTOCOL = PROJECT / 'analysis/protocols'
SEEDS = (1001, 1002, 1003)
SYSTEMS = ('graph', 'modernbert', 'same_seed_pair', 'saved_fixed_LR')
MODELS = [(s, seed) for s in SYSTEMS[:3] for seed in SEEDS] + [('saved_fixed_LR', None)]
METRICS = ('auroc', 'average_precision', 'recall', 'fpr', 'precision', 'f1')
VIEWS = ('raw_unweighted', 'retained_unweighted', 'retained_overlap_weighted')
FILE_BINS = ('1', '2-4', '5-16', '>16')
BYTE_BINS = ('<=4KiB', '4-32KiB', '>32KiB')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            x = x.item()
        return None if isinstance(x, float) and not np.isfinite(x) else x
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    assert rows, path
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                             for k, v in row.items()})


class MetricEngine:
    def __init__(self, y, score, pred):
        self.y = y
        self.pred = pred
        self.orders = [np.argsort(-s, kind='stable') for s in score]
        self.ends = [np.r_[np.flatnonzero(np.diff(score[i, order]) != 0), len(y)-1]
                     for i, order in enumerate(self.orders)]

    def evaluate(self, w):
        out = np.full((len(self.orders), 6), np.nan)
        positive, negative = w * self.y, w * (1-self.y)
        p, n = positive.sum(), negative.sum()
        if not p or not n:
            return out
        for i, (order, ends) in enumerate(zip(self.orders, self.ends)):
            tp = np.cumsum(positive[order])[ends]
            fp = np.cumsum(negative[order])[ends]
            dp, dn = np.diff(np.r_[0., tp]), np.diff(np.r_[0., fp])
            out[i, 0] = np.sum(dn * (tp - 0.5*dp)) / (p*n)
            precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
            out[i, 1] = np.sum(dp * precision) / p
            if self.pred[i] is not None:
                t = np.dot(positive, self.pred[i])
                f = np.dot(negative, self.pred[i])
                out[i, 2:] = [t/p, f/n, t/(t+f) if t+f else np.nan,
                              2*t/(p+t+f)]
        return out


def self_check():
    # Ties, zero weights and repeated components: independent sklearn and hand checks.
    y = np.array([1, 0, 1, 0])
    score = np.array([[.8, .8, .3, .1]])
    pred = [np.array([1, 1, 0, 0])]
    engine = MetricEngine(y, score, pred)
    for w in (np.array([2., 1., 1., 2.]), np.array([2., 0., 1., 2.])):
        result = engine.evaluate(w)[0]
        assert abs(result[0] - roc_auc_score(y, score[0], sample_weight=w)) < 1e-12
        assert abs(result[1] - average_precision_score(y, score[0], sample_weight=w)) < 1e-12
    result = engine.evaluate(np.array([2., 1., 1., 2.]))[0]
    np.testing.assert_allclose(result[2:], [2/3, 1/3, 2/3, 2/3], atol=1e-12)
    # Two cells have counts (n0,n1)=(2,1),(1,2): each class/cell mass becomes 1.
    yy = np.array([0, 0, 1, 0, 1, 1])
    cc = np.array([0, 0, 0, 1, 1, 1])
    w = overlap_weights(yy, cc, [0, 1], np.ones(6))
    np.testing.assert_allclose(w, [.5, .5, 1, 1, .5, .5])
    # Component 0 drawn twice and 1 once repeats all their rows together.
    component = np.array([0, 0, 1, 2])
    multiplicity = np.bincount([0, 0, 1], minlength=3)[component]
    np.testing.assert_array_equal(multiplicity, [2, 2, 1, 0])
    return {'weighted_ties_and_zero_weights_vs_sklearn': 'PASS',
            'hand_checked_confusion_and_overlap_weights': 'PASS',
            'whole_component_multiplicity': 'PASS'}


def overlap_weights(y, cells, eligible, mult):
    w = np.zeros(len(y))
    if not len(eligible):
        return None
    for cell in eligible:
        a, b = (cells == cell) & (y == 0), (cells == cell) & (y == 1)
        n0, n1 = mult[a].sum(), mult[b].sum()
        if not n0 or not n1:
            return None
        mass = min(n0, n1)
        w[a], w[b] = mult[a]*mass/n0, mult[b]*mass/n1
    return w


def ess(w):
    return float(w.sum()**2 / np.dot(w, w)) if np.dot(w, w) else 0.


def make_population(data, source, column, definition):
    rows = [r for r in data if r['upstream_source'] in ('benign', source)]
    y = np.array([int(r['y_true']) for r in rows])
    files = np.array([int(r[column]) for r in rows])
    size = np.array([int(r['total_python_bytes']) for r in rows])
    cells = np.searchsorted([1, 4, 16], files, side='left') * 3 + np.searchsorted([4096, 32768], size, side='left')
    groups = np.array([r['source_clone_component_id'] for r in rows])
    score = np.array([[float(r[f'seed{seed}_{s}_score'] if seed else r['saved_fixed_LR_score'])
                       for r in rows] for s, seed in MODELS])
    pred = [np.array([int(r[f'seed{seed}_{s}_prediction']) for r in rows]) if seed else None for s, seed in MODELS]
    eligible, cell_rows = [], []
    for c in range(12):
        n = [int(((cells == c) & (y == j)).sum()) for j in (0, 1)]
        g = [len(set(groups[(cells == c) & (y == j)])) for j in (0, 1)]
        reason = ('EMPTY' if not sum(n) else 'ONE_CLASS_ONLY' if not min(n)
                  else 'SPARSE_ROWS' if min(n) < 5 else 'SPARSE_COMPONENTS' if min(g) < 2 else 'ELIGIBLE')
        if reason == 'ELIGIBLE':
            eligible.append(c)
        cell_rows.append({'definition': definition, 'contrast': source+'_vs_benign', 'cell': c,
                          'file_bin': FILE_BINS[c//3], 'byte_bin': BYTE_BINS[c%3],
                          'benign_rows': n[0], 'malicious_rows': n[1],
                          'benign_components': g[0], 'malicious_components': g[1],
                          'status': reason, 'target_mass_each_class': min(n) if reason == 'ELIGIBLE' else 0})
    retained = np.isin(cells, eligible)
    weights = [np.ones(len(y)), retained.astype(float), overlap_weights(y, cells, eligible, np.ones(len(y)))]
    return {'definition': definition, 'contrast': source+'_vs_benign', 'rows': rows, 'y': y, 'groups': groups,
            'files': files, 'bytes': size, 'cells': cells, 'eligible': eligible, 'retained': retained,
            'weights': weights, 'engine': MetricEngine(y, score, pred), 'cell_rows': cell_rows}


def support_status(pop, w):
    if w is None or not w.sum():
        return 'NO_ELIGIBLE_SUPPORT'
    for y in (0, 1):
        mask = (pop['y'] == y) & (w > 0)
        if len(set(pop['groups'][mask])) < 20 or ess(w[mask]) < 20:
            return 'LOW_SUPPORT'
    return 'OK'


def balance(pop):
    rows, counts = [], []
    y, groups = pop['y'], pop['groups']
    for view, w in zip(VIEWS, pop['weights']):
        for cls in (0, 1):
            mask = y == cls
            selected = mask & ((w > 0) if w is not None else False)
            ww = w[selected] if w is not None else np.array([])
            gm = Counter()
            for g, mass in zip(groups[selected], ww):
                gm[g] += mass
            counts.append({'definition': pop['definition'], 'contrast': pop['contrast'], 'view': view,
                           'label': cls, 'original_rows': int(mask.sum()), 'retained_rows': int(selected.sum()),
                           'excluded_rows': int(mask.sum()-selected.sum()), 'retention_fraction': float(selected.sum()/mask.sum()),
                           'original_components': len(set(groups[mask])), 'retained_components': len(gm),
                           'weight_sum': float(ww.sum()), 'class_ESS': ess(ww),
                           'max_component_weight_share': float(max(gm.values())/ww.sum()) if gm else None,
                           'status': support_status(pop, w)})
        for field in ('files', 'bytes'):
            x = np.log1p(pop[field])
            denom = np.sqrt((np.var(x[y == 0]) + np.var(x[y == 1])) / 2)
            stats = []
            for cls in (0, 1):
                m = y == cls
                if w is None or w[m].sum() == 0:
                    stats.append((np.nan, np.nan))
                else:
                    mu = np.average(x[m], weights=w[m])
                    stats.append((mu, np.sqrt(np.average((x[m]-mu)**2, weights=w[m]))))
            delta = abs(stats[0][0]-stats[1][0])
            smd = delta/denom if denom else (0. if delta == 0 else np.nan)
            ks = np.nan
            if w is not None and min(w[y == cls].sum() for cls in (0, 1)) > 0:
                order = np.argsort(x, kind='stable')
                ends = np.r_[np.flatnonzero(np.diff(x[order]) != 0), len(x)-1]
                ecdf = [np.cumsum((w*(y == cls))[order])[ends] / w[y == cls].sum() for cls in (0, 1)]
                ks = np.max(np.abs(ecdf[0]-ecdf[1]))
            rows.append({'definition': pop['definition'], 'contrast': pop['contrast'], 'view': view,
                         'log1p_feature': field, 'benign_mean': stats[0][0], 'benign_sd': stats[0][1],
                         'malicious_mean': stats[1][0], 'malicious_sd': stats[1][1],
                         'raw_fixed_SMD_denominator': denom, 'absolute_SMD': smd, 'weighted_KS': ks,
                         'residual_SMD_over_0_1': bool(smd > .1) if np.isfinite(smd) else None})
    return rows, counts


def validate_overall(data, track):
    y = np.array([int(r['y_true']) for r in data])
    scores = np.array([[float(r[f'seed{seed}_{s}_score'] if seed else r['saved_fixed_LR_score']) for r in data]
                       for s, seed in MODELS])
    preds = [np.array([int(r[f'seed{seed}_{s}_prediction']) for r in data]) if seed else None for s, seed in MODELS]
    actual = MetricEngine(y, scores, preds).evaluate(np.ones(len(y)))
    i3 = load(track('experiment_outputs/supplementary_review_20260906/I3_nuisance_context/complete.json'))
    rec = next(r for r in i3['outputs'] if r['path'].endswith('/validation_size_ranking_per_run.csv'))
    with track(rec['path'], rec['sha256']).open(newline='') as f:
        refs = list(csv.DictReader(f))
    for i, (system, seed) in enumerate(MODELS):
        rr = next(r for r in refs if r['stratum_type'] == 'overall' and r['system'] == system
                  and r['seed'] == (str(seed) if seed else 'single_fixed'))
        for j, metric in enumerate(METRICS[:2]):
            assert abs(actual[i, j]-float(rr[metric])) < 1e-12, (system, seed, metric)
    s9h_path = next(r['path'] for r in load(PROJECT/'analysis/audits/20261015_validation_inputs_v3/input_manifest.json')
                    if r['path'].endswith('s9h_calibration_fpr_error_analysis.json'))
    s9h = load(track(s9h_path))
    for system in SYSTEMS[:3]:
        ids = [i for i, (s, _) in enumerate(MODELS) if s == system]
        for j, metric in enumerate(METRICS):
            ref = s9h['overall_mean_sample_sd'][system][metric]
            assert abs(actual[ids, j].mean() - ref['mean']) < 1e-12
            assert abs(actual[ids, j].std(ddof=1) - ref['sample_standard_deviation']) < 1e-12
    return {'saved_I3_ranking_replay': 'PASS', 'saved_S9H_six_metric_means_and_SDs': 'PASS',
            'absolute_tolerance': 1e-12}


def aggregate(a, system):
    ids = [i for i, (s, _) in enumerate(MODELS) if s == system]
    vals = a[..., ids, :]
    return vals.mean(axis=-2)


def collect_metric_rows(pop, point, draws=None):
    result = []
    entities = [(s, str(seed) if seed else 'single_fixed', i) for i, (s, seed) in enumerate(MODELS)]
    entities += [(s, 'three_seed_mean', None) for s in SYSTEMS[:3]]
    for vi, view in enumerate(VIEWS):
        support = support_status(pop, pop['weights'][vi])
        for system, seed, idx in entities:
            p = point[vi, idx] if idx is not None else aggregate(point[vi], system)
            b = None if draws is None else (draws[:, vi, idx] if idx is not None else aggregate(draws[:, vi], system))
            ids = [i for i, (s, _) in enumerate(MODELS) if s == system]
            for mi, metric in enumerate(METRICS):
                if system == 'saved_fixed_LR' and mi >= 2:
                    continue
                values = b[:, mi] if b is not None else np.array([])
                valid = values[np.isfinite(values)]
                status = support
                if not np.isfinite(p[mi]) and status == 'OK':
                    status = 'UNDEFINED_METRIC'
                if b is None and status == 'OK':
                    status = 'SENSITIVITY_POINT_ONLY'
                elif b is not None and len(valid) < .95*len(b) and status == 'OK':
                    status = 'BOOTSTRAP_UNSTABLE'
                ci = np.quantile(valid, [.025, .975], method='linear') if status == 'OK' else [np.nan]*2
                result.append({'definition': pop['definition'], 'contrast': pop['contrast'], 'view': view,
                               'system': system, 'seed': seed, 'metric': metric, 'point': p[mi],
                               'sample_seed_sd': float(point[vi, ids, mi].std(ddof=1)) if idx is None else None,
                               'ci95_lower': ci[0], 'ci95_upper': ci[1], 'bootstrap_valid': len(valid) if b is not None else None,
                               'bootstrap_total': len(b) if b is not None else None, 'status': status})
    return result


def collect_comparisons(pop, point, draws=None):
    result = []
    pairs = [(s, 'saved_fixed_LR', [0, 1]) for s in SYSTEMS[:3]] + [('same_seed_pair', 'modernbert', [0, 1, 2, 3, 5])]
    for vi, view in enumerate(VIEWS):
        support = support_status(pop, pop['weights'][vi])
        for first, second, metrics in pairs:
            for seed in (*SEEDS, 'three_seed_mean'):
                def pick(a, s):
                    if seed == 'three_seed_mean':
                        return aggregate(a, s)
                    idx = MODELS.index((s, seed if s != 'saved_fixed_LR' else None))
                    return a[..., idx, :]
                delta = pick(point[vi], first)-pick(point[vi], second)
                b = None if draws is None else pick(draws[:, vi], first)-pick(draws[:, vi], second)
                for mi in metrics:
                    valid = b[:, mi][np.isfinite(b[:, mi])] if b is not None else np.array([])
                    status = support
                    if not np.isfinite(delta[mi]) and status == 'OK':
                        status = 'UNDEFINED_METRIC'
                    if b is None and status == 'OK':
                        status = 'SENSITIVITY_POINT_ONLY'
                    elif b is not None and len(valid) < .95*len(b) and status == 'OK':
                        status = 'BOOTSTRAP_UNSTABLE'
                    ci = np.quantile(valid, [.025, .975], method='linear') if status == 'OK' else [np.nan]*2
                    result.append({'definition': pop['definition'], 'contrast': pop['contrast'], 'view': view,
                                   'comparison': first+'_minus_'+second, 'seed': str(seed), 'metric': METRICS[mi],
                                   'point_delta': delta[mi], 'ci95_lower': ci[0], 'ci95_upper': ci[1],
                                   'bootstrap_valid': len(valid) if b is not None else None,
                                   'bootstrap_total': len(b) if b is not None else None, 'status': status})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Use --run for the frozen validation diagnostic')
    start, cpu_start = time.monotonic(), time.process_time()
    now = datetime.now(timezone.utc)
    config = load(PROTOCOL/'JOINT_SIZE_DIAGNOSTIC_PROTOCOL_20261015_v1.json')
    out = ROOT/config['output_parent']/now.strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    dependencies = {}

    def track(rel, expected=None):
        path = ROOT/rel
        h = sha(path)
        if expected:
            assert h == expected, rel
        if str(rel) in dependencies:
            assert dependencies[str(rel)]['sha256'] == h, rel
        dependencies[str(rel)] = {'path': str(rel), 'bytes': path.stat().st_size, 'sha256': h}
        return path

    try:
        freeze_path = PROTOCOL/'JOINT_SIZE_DIAGNOSTIC_FREEZE_20261015_v1.json'
        freeze = load(track(str(freeze_path.relative_to(ROOT))))
        for rec in freeze['files']:
            track(rec['path'], rec['sha256'])
        for rec in load(PROJECT/'analysis/audits/20261015_validation_inputs_v3/input_manifest.json'):
            track(rec['path'], rec['sha256'])
        track(str(Path(__file__).relative_to(ROOT)))
        # Preserve checks of the old manuscript dependencies and user template too.
        baseline = load(PROJECT/'planning/audits/20261015_baseline_comparison_v2/baseline_audit.json')
        for rec in baseline['inputs']:
            if not rec['path'].startswith('paper1015-Frontiers/'):
                track(rec['path'], rec['sha256'])
        registration = load(PROJECT/'planning/PROJECT_REGISTRATION_20261015.json')
        for rec in registration['template_files']:
            track('paper1015-Frontiers/Frontiers_LaTeX_Templates/'+rec['relative_path'], rec['sha256'])
        with track(config['input_csv']).open(newline='') as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2137 and len({r['sample_id'] for r in data}) == 2137
        all_groups = sorted({r['source_clone_component_id'] for r in data})
        assert len(all_groups) == 1588
        checks = self_check()
        checks.update(validate_overall(data, track))
        write_json(out/'verification_before_analysis.json', checks)
        print(json.dumps({'stage': 'historical_replay_and_metric_checks_PASS', 'attempt': str(out)}, ensure_ascii=False), flush=True)
        populations = []
        all_cells, all_coverage, all_balance, all_metrics, all_comparisons, confusions, bin_balance = [], [], [], [], [], [], []
        point_values = {}
        for definition, column in [('primary', config['primary_file_count_column']), ('fingerprint_sensitivity', config['sensitivity_file_count_column'])]:
            for source in ('guo', 'datadog'):
                pop = make_population(data, source, column, definition)
                populations.append(pop)
                all_cells.extend(pop['cell_rows'])
                br, coverage = balance(pop)
                all_balance.extend(br)
                all_coverage.extend(coverage)
                point = np.array([pop['engine'].evaluate(w) if w is not None else np.full((10, 6), np.nan)
                                  for w in pop['weights']])
                point_values[definition, source] = point
                for vi, (view, w) in enumerate(zip(VIEWS, pop['weights'])):
                    for c in range(12):
                        mass = [float(w[(pop['cells'] == c) & (pop['y'] == y)].sum()) if w is not None else 0 for y in (0, 1)]
                        totals = [float(w[pop['y'] == y].sum()) if w is not None else 0 for y in (0, 1)]
                        bin_balance.append({'definition': definition, 'contrast': pop['contrast'], 'view': view, 'cell': c,
                                            'benign_weight_mass': mass[0], 'malicious_weight_mass': mass[1],
                                            'benign_fraction': mass[0]/totals[0] if totals[0] else None,
                                            'malicious_fraction': mass[1]/totals[1] if totals[1] else None})
                        if vi == 2 and w is not None:
                            assert abs(mass[0]-mass[1]) < 1e-9
                    if w is None:
                        continue
                    if vi == 2:
                        assert abs(w[pop['y'] == 0].sum()-w[pop['y'] == 1].sum()) < 1e-9
                    for i, (system, seed) in enumerate(MODELS[:9]):
                        pred, y = pop['engine'].pred[i], pop['y']
                        selected = w > 0
                        row = {'definition': definition, 'contrast': pop['contrast'], 'view': view, 'system': system, 'seed': seed}
                        for name, mask in [('tp', (y == 1) & (pred == 1)), ('fp', (y == 0) & (pred == 1)),
                                           ('tn', (y == 0) & (pred == 0)), ('fn', (y == 1) & (pred == 0))]:
                            row[name+'_actual_rows'] = int((mask & selected).sum())
                            row[name+'_weighted_mass'] = float(w[mask].sum())
                        confusions.append(row)
        primary = populations[:2]  # creation order: primary Guo, primary Datadog.
        group_index = {g: i for i, g in enumerate(all_groups)}
        group_maps = [np.array([group_index[g] for g in p['groups']]) for p in primary]
        bootstrap = np.full((2000, 2, 3, 10, 6), np.nan)
        rng = np.random.Generator(np.random.PCG64(20261015))
        for b in range(2000):
            multiplicity = np.bincount(rng.integers(0, 1588, size=1588), minlength=1588)
            for pi, pop in enumerate(primary):
                mult = multiplicity[group_maps[pi]].astype(float)
                weights = [mult, mult*pop['retained'], overlap_weights(pop['y'], pop['cells'], pop['eligible'], mult)]
                for vi, w in enumerate(weights):
                    if w is not None:
                        bootstrap[b, pi, vi] = pop['engine'].evaluate(w)
            if (b+1) % 250 == 0:
                print(json.dumps({'stage': 'component_bootstrap', 'completed': b+1, 'total': 2000,
                                  'elapsed_seconds': round(time.monotonic()-start, 1)}), flush=True)
        validity = []
        for pop in populations:
            source = pop['contrast'].split('_vs_')[0]
            draws = bootstrap[:, ('guo', 'datadog').index(source)] if pop['definition'] == 'primary' else None
            point = point_values[pop['definition'], source]
            all_metrics.extend(collect_metric_rows(pop, point, draws))
            all_comparisons.extend(collect_comparisons(pop, point, draws))
            if draws is not None:
                for vi, view in enumerate(VIEWS):
                    valid = int(np.isfinite(draws[:, vi, 0, 0]).sum())
                    validity.append({'contrast': pop['contrast'], 'view': view, 'replicates': 2000,
                                     'valid': valid, 'valid_fraction': valid/2000,
                                     'support_status': support_status(pop, pop['weights'][vi])})
        for name, rows in [('joint_cell_support.csv', all_cells), ('population_coverage.csv', all_coverage),
                           ('continuous_balance.csv', all_balance), ('joint_cell_balance.csv', bin_balance),
                           ('metrics.csv', all_metrics), ('paired_comparisons.csv', all_comparisons),
                           ('confusion_counts_and_weighted_mass.csv', confusions), ('bootstrap_validity.csv', validity)]:
            write_csv(out/name, rows)
        np.savez_compressed(out/'bootstrap_draw_metrics.npz', values=bootstrap)
        write_json(out/'bootstrap_array_axes.json', {'shape': list(bootstrap.shape), 'axes': ['replicate', 'contrast', 'view', 'model', 'metric'],
                                                    'contrasts': [p['contrast'] for p in primary], 'views': VIEWS,
                                                    'models': MODELS, 'metrics': METRICS})
        write_json(out/'input_manifest.json', list(dependencies.values()))
        for rel, rec in dependencies.items():
            assert sha(ROOT/rel) == rec['sha256'], rel
        summary = {'status': 'COMPLETE', 'protocol_id': config['protocol_id'], 'protocol_freeze_sha256': sha(freeze_path),
                   'started_at_utc': now.isoformat(), 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                   'wall_seconds': time.monotonic()-start, 'cpu_seconds': time.process_time()-cpu_start,
                   'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                                   'numpy': np.__version__, 'sklearn': sklearn.__version__},
                   'input_dependencies_unchanged': len(dependencies), 'validation_rows': 2137, 'components': 1588,
                   'bootstrap_replicates': 2000, 'metric_checks': checks, 'models_fitted': 0, 'new_inferences': 0,
                   'thresholds_changed': 0, 'test_prediction_tables_read': 0, 'malware_files_opened': 0,
                   'manuscript_edited': False, 'protocol_changed': False,
                   'eligible_cells': {p['definition']+'__'+p['contrast']: p['eligible'] for p in populations}}
        write_json(out/'summary.json', summary)
        inventory = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.iterdir())]
        write_json(out/'complete.json', {'status': 'COMPLETE', 'summary': summary, 'outputs': inventory})
        print(json.dumps({'status': 'COMPLETE', 'attempt': str(out), 'wall_seconds': summary['wall_seconds'],
                          'eligible_cells': summary['eligible_cells'], 'bootstrap_validity': validity}, ensure_ascii=False), flush=True)
    except BaseException:
        write_json(out/'failure.json', {'status': 'FAILED', 'traceback': traceback.format_exc(),
                                      'elapsed_seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    main()
