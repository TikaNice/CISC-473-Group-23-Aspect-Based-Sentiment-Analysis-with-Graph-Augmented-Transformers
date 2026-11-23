"""Summarize the dependency-vs-identity graph control.

The two arms live in different folders: identity runs are filed under
results/ablation/ (see src/result_paths.py), while the dependency arm is the
ordinary GCN run and is filed under results/custom/ with the 'aspectfix' tag.
Both are resolved through result_path so the pairing cannot silently break when
the folder layout changes.

Writes mean +/- sample SD per (domain, budget, graph) plus paired
dependency-minus-identity deltas to results/reports/ablation_summary.csv.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.result_paths import result_path

OUT = Path('results/reports/ablation_summary.csv')
GRAPH_LABEL = {'dependency': 'dependency edges', 'identity': 'identity (self-loops only)', 'full': 'fully connected'}


def parse_run(obj):
    exp = obj.get('experiment', '')
    domain = exp.split('_fraction')[0]
    fraction = float(re.search(r'fraction([0-9.]+)', exp).group(1))
    return domain, fraction, obj['seed'], obj.get('graph_type', 'dependency')


ARMS = {'dependency': 'aspectfix', 'identity': 'identity'}   # the two arms the grid runs
DOMAINS = ('restaurant', 'laptop')
FRACTIONS = (1.0, 0.1)
SEEDS = (42, 43, 44)


def load(path, graph, runs, missing):
    if not path.is_file():
        missing.append(str(path))
        return
    obj = json.loads(path.read_text(encoding='utf-8'))
    test = obj.get('final_test_metrics') or {}
    if 'test_accuracy' not in test:
        missing.append(str(path))
        return
    domain, fraction, seed, observed_graph = parse_run(obj)
    if graph != observed_graph:
        raise ValueError(f'{path}: expected {graph}, found {observed_graph}')
    runs[(domain, fraction, graph)].append({
        'seed': seed, 'acc': test['test_accuracy'], 'f1': test['test_macro_f1'], 'file': str(path),
    })


def collect():
    """Load both arms from their canonical locations, use only the reported grid."""
    runs, missing = defaultdict(list), []
    for domain in DOMAINS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                for graph, tag in ARMS.items():
                    load(result_path('gcn', domain, fraction, seed, tag), graph, runs, missing)
    if missing:
        print(f'{len(missing)} control record(s) absent, e.g. {missing[0]}')
    return runs


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else None


def main():
    runs = collect()
    rows = []
    for (domain, fraction, graph), records in sorted(runs.items()):
        records = sorted(records, key=lambda r: r['seed'])
        acc_m, acc_s = mean_sd([r['acc'] for r in records])
        f1_m, f1_s = mean_sd([r['f1'] for r in records])
        dep = runs.get((domain, fraction, 'dependency'), [])
        ident = runs.get((domain, fraction, 'identity'), [])
        paired = {r['seed']: r['f1'] for r in dep}
        diffs = [paired[r['seed']] - r['f1'] for r in ident if r['seed'] in paired]
        if graph == 'dependency' and diffs:
            d_m, d_s = mean_sd(diffs)
            delta = str(d_m)
            wins = sum(1 for d in diffs if d > 0)
            note = f'{wins}/{len(diffs)} seeds favor dependency'
        else:
            delta, d_s, note = '', None, ''
        rows.append({
            'domain': domain,
            'fraction': fraction,
            'graph': graph,
            'graph_label': GRAPH_LABEL.get(graph, graph),
            'n_seeds': len(records),
            'seeds': ' '.join(str(r['seed']) for r in records),
            'test_accuracy_mean': str(acc_m),
            'test_accuracy_std': '' if acc_s is None else str(acc_s),
            'test_macro_f1_mean': str(f1_m),
            'test_macro_f1_std': '' if f1_s is None else str(f1_s),
            'paired_dependency_minus_identity_mean': delta,
            'paired_sd': '' if d_s is None else str(d_s),
            'paired_note': note,
            'source_files': ' '.join(r['file'] for r in records),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        OUT.unlink(missing_ok=True)
        print(f'No ablation runs found for either arm; {OUT} not written.')
        return
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
