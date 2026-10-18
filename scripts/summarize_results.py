"""Validate the complete midterm matrix and regenerate report tables (no ML imports)."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from src.result_paths import result_path


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    print('HISTORICAL MATRIX: legacy GCN alignment. Use scripts.summarize_corrected for aspectfix results.')
    records = []
    for model in ('bert', 'gcn'):
        for domain in ('restaurant', 'laptop'):
            settings = [('baseline', 1, 3, '')]
            settings += [('low_resource', f, 3, '') for f in (.1, .25, .5)]
            settings += [('epoch_sweep', 1, e, f'ep{e}') for e in (1, 2, 4)]
            if domain == 'restaurant':
                settings += [('cross_domain', 1, 3, '')]
            for category, fraction, epochs, tag in settings:
                for seed in (42, 43, 44):
                    path = (Path('results/cross_domain/restaurant_to_laptop') / model /
                            f'{model}_restaurant_to_laptop_seed{seed}.json') if category == 'cross_domain' else result_path(model, domain, fraction, seed, tag)
                    obj = json.loads(path.read_text(encoding='utf-8'))
                    assert obj['seed'] == seed, f'Seed mismatch: {path}'
                    val = obj.get('validation_metrics', {})
                    test = obj if category == 'cross_domain' else obj['final_test_metrics']
                    row = dict(file=str(path), category=category,
                               domain='restaurant_to_laptop' if category == 'cross_domain' else domain,
                               model=model, fraction=fraction, epochs=epochs, seed=seed,
                               train_size=obj.get('train_size', ''), validation_size=obj.get('validation_size', ''),
                               test_size=obj['test_size'], validation_accuracy=val.get('eval_accuracy', ''),
                               validation_macro_f1=val.get('eval_macro_f1', ''),
                               test_accuracy=test['test_accuracy'], test_macro_f1=test['test_macro_f1'])
                    for key in ('test_accuracy', 'test_macro_f1'):
                        assert isinstance(row[key], (int, float)) and math.isfinite(row[key]) and 0 <= row[key] <= 1, f'Invalid {key}: {path}'
                    expected = 638 if category == 'cross_domain' or domain == 'laptop' else 1120
                    assert row['test_size'] == expected, f'Test size mismatch: {path}'
                    if category != 'cross_domain':
                        for key in ('validation_accuracy', 'validation_macro_f1'):
                            assert isinstance(row[key], (int, float)) and math.isfinite(row[key]) and 0 <= row[key] <= 1, f'Invalid {key}: {path}'
                    records.append(row)
    assert len(records) == 90
    # Keep smoke records visible, but exclude them from benchmark summaries.
    for model in ('bert', 'gcn'):
        path = result_path(model, 'restaurant', 1, 42, smoke=True)
        if not path.is_file():
            continue
        obj = json.loads(path.read_text())
        row = dict.fromkeys(records[0], '')
        row.update(file=str(path), category='smoke', domain='restaurant', model=model, fraction=1,
                   epochs=1, seed=42, train_size=obj['train_size'], validation_size=obj['validation_size'],
                   validation_accuracy=obj['validation_metrics']['eval_accuracy'],
                   validation_macro_f1=obj['validation_metrics']['eval_macro_f1'])
        records.append(row)
    groups = defaultdict(list)
    for row in records:
        if row['category'] != 'smoke':
            groups[tuple(row[k] for k in ('category', 'domain', 'model', 'fraction', 'epochs'))].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        assert sorted(r['seed'] for r in rows) == [42, 43, 44]
        row = dict(zip(('category', 'domain', 'model', 'fraction', 'epochs'), key))
        row['n_seeds'] = 3
        for metric in ('test_accuracy', 'test_macro_f1', 'validation_macro_f1'):
            values = [r[metric] for r in rows if isinstance(r[metric], (int, float))]
            row[metric + '_mean'] = statistics.mean(values) if values else ''
            row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        summaries.append(row)
    output = Path('results/reports')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'all_json_full.txt').open('w', encoding='utf-8') as handle:
        for row in records:
            path = Path(row['file'])
            handle.write(f'\n=== {path} ===\n' + path.read_text(encoding='utf-8') + '\n')
    write_csv(output / 'all_metrics.csv', records)
    write_csv(output / 'summary_by_group.csv', summaries)
    write_csv(output / 'baseline_summary.csv', [r for r in summaries if r['category'] == 'baseline'])
    write_csv(output / 'cross_domain_summary.csv', [r for r in summaries if r['category'] == 'cross_domain'])
    print(f'Validated 90 formal runs + {len(records) - 90} optional smoke runs; wrote 30 complete three-seed groups.')


if __name__ == '__main__':
    main()
