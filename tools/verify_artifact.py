#!/usr/bin/env python3
"""Local-only audit of files and saved aggregate identities; standard library only."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read_rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    files = read_rows(ROOT / 'metadata/FILE_SHA256.csv')
    names = set()
    for r in files:
        rel = Path(r['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and r['path'] not in names
        names.add(r['path'])
        f = ROOT / rel
        assert f.is_file() and f.stat().st_size == int(r['bytes']), r['path']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == r['sha256'], r['path']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.relative_to(ROOT).parts
              and p.relative_to(ROOT).as_posix() != 'metadata/FILE_SHA256.csv'}
    assert actual == names, (actual-names, names-actual)
    split = Counter()
    seen = set()
    components = defaultdict(set)
    sizes = Counter()
    for p in sorted((ROOT / 'data/manifests').glob('*.part-*.csv')):
        for r in read_rows(p):
            assert r['sample_id'] and r['sample_id'] not in seen
            seen.add(r['sample_id'])
            split[r['split']] += 1
            components[r['source_clone_component_id']].add(r['split'])
            sizes[r['source_clone_component_id']] += 1
    assert len(seen)==20878 and split=={'train':16613,'val':2137,'test':2128}, split
    assert len(components)==11813 and all(len(v)==1 for v in components.values())
    assert max(sizes.values())==5319
    assert all(components[k]=={'train'} for k,v in sizes.items() if v>50)
    tab = read_rows(ROOT / 'results/current/table_01_existing_detectors.csv')
    assert len(tab)==8 and not any(r['method'].endswith('-only') for r in tab)
    assert {r['method'] for r in tab}=={'Bandit4Mal','GuardDog','OSSGadget','VirusTotal','Cerebro','EA4MP','PyPIMalDet','PSHGuard'}
    psh = next(r for r in tab if r['method']=='PSHGuard')
    assert psh['F1_percent_display']=='92.538' and psh['F1_sd_pp_display']=='1.238'
    assert len(read_rows(ROOT/'results/current/table_02_graph_representation.csv'))==16
    budget = read_rows(ROOT/'results/file_budget/all_budgets_mean_sample_sd.csv')
    assert {int(r['budget']) for r in budget}=={4,8,16,32}
    assert all(int(r['runs'])==3 for r in budget)
    paired = read_rows(ROOT/'results/current/figure_03_paired_branch_differences.csv')
    mb = next(r for r in paired if r['contrast']=='f1_fusion_minus_modernbert')
    assert float(mb['ci95_lower_ratio'])<0<float(mb['ci95_upper_ratio'])
    t3 = read_rows(ROOT/'results/current/table_03_auroc_and_fpr.csv')
    dd = [r for r in t3 if r['contrast']=='datadog_vs_benign' and r['view']=='retained_overlap_weighted']
    assert len(dd)==5 and all(r['ci95_lower']=='' and r['ci95_upper']=='' for r in dd)
    print(f'PASS: {len(files)} checksummed files; complete distribution inventory verified')
    print(f'PASS: {len(seen)} unique samples; splits={dict(split)}; {len(components)} isolated components')
    print('PASS: current table/figure identities, four budgets, and saved missing/uncertainty states')
    print('No training, prediction replay, calibration fit, bootstrap, or network access performed.')


if __name__ == '__main__':
    main()
