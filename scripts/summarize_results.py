"""Summarize valid BERT and corrected GCN records; record untrained runs explicitly."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from src.result_paths import result_path


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_records():
    records, status = [], []
    for model in ('bert', 'gcn'):
        for domain in ('restaurant', 'laptop'):
            settings = [('baseline', 1, 3, '')]
            settings += [('low_resource', f, 3, '') for f in (.1, .25, .5)]
            settings += [('epoch_sweep', 1, e, f'ep{e}') for e in (1, 2, 4)]
            if domain == 'restaurant':
                settings.append(('cross_domain', 1, 3, ''))
            for category, fraction, epochs, tag in settings:
                for seed in (42, 43, 44):
                    run_tag = ('aspectfix_' + tag if tag else 'aspectfix') if model == 'gcn' else tag
                    if category == 'cross_domain':
                        suffix = '_aspectfix' if model == 'gcn' else ''
                        path = Path('results/cross_domain/restaurant_to_laptop') / model / f'{model}_restaurant_to_laptop_seed{seed}{suffix}.json'
                    else:
                        path = result_path(model, domain, fraction, seed, run_tag)
                    item = dict(category=category, domain='restaurant_to_laptop' if category == 'cross_domain' else domain,
                                model=model, fraction=fraction, epochs=epochs, seed=seed,
                                revision='aspectfix_v1' if model == 'gcn' else 'bert_v1',
                                status='complete' if path.is_file() else 'pending', file=str(path))
                    status.append(item)
                    if not path.is_file():
                        continue
                    obj = json.loads(path.read_text(encoding='utf-8'))
                    assert obj['seed'] == seed, path
                    if model == 'gcn':
                        assert obj.get('aspect_alignment') == 'case_sensitive_ordinal_v1', path
                    val = obj.get('validation_metrics', {})
                    test = obj if category == 'cross_domain' else obj['final_test_metrics']
                    row = dict(file=str(path), category=category, domain=item['domain'], model=model,
                               fraction=fraction, epochs=epochs, seed=seed, revision=item['revision'],
                               train_size=obj.get('train_size', ''), validation_size=obj.get('validation_size', ''),
                               test_size=obj['test_size'], validation_accuracy=val.get('eval_accuracy', ''),
                               validation_macro_f1=val.get('eval_macro_f1', ''),
                               test_accuracy=test['test_accuracy'], test_macro_f1=test['test_macro_f1'])
                    keys = ('test_accuracy', 'test_macro_f1') + (() if category == 'cross_domain' else ('validation_accuracy', 'validation_macro_f1'))
                    for key in keys:
                        assert isinstance(row[key], (int, float)) and math.isfinite(row[key]) and 0 <= row[key] <= 1, (path, key)
                    expected = 638 if category == 'cross_domain' or domain == 'laptop' else 1120
                    assert row['test_size'] == expected, path
                    records.append(row)
    for model in ('bert', 'gcn'):
        path = result_path(model, 'restaurant', 1, 42, 'aspectfix' if model == 'gcn' else '', smoke=True)
        if not path.is_file():
            continue
        obj = json.loads(path.read_text())
        if model == 'gcn':
            assert obj.get('aspect_alignment') == 'case_sensitive_ordinal_v1'
        row = dict.fromkeys(records[0], '')
        row.update(file=str(path), category='smoke', domain='restaurant', model=model, fraction=1,
                   epochs=1, seed=42, revision='aspectfix_v1' if model == 'gcn' else 'bert_v1',
                   train_size=obj['train_size'], validation_size=obj['validation_size'],
                   validation_accuracy=obj['validation_metrics']['eval_accuracy'],
                   validation_macro_f1=obj['validation_metrics']['eval_macro_f1'])
        records.append(row)
    return records, status


def group_records(records):
    groups = defaultdict(list)
    for row in records:
        if row['category'] != 'smoke':
            groups[tuple(row[k] for k in ('category', 'domain', 'model', 'fraction', 'epochs', 'revision'))].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        row = dict(zip(('category', 'domain', 'model', 'fraction', 'epochs', 'revision'), key))
        row['n_seeds'] = len(rows)
        row['seeds'] = ','.join(str(r['seed']) for r in sorted(rows, key=lambda r:r['seed']))
        row['statistics_status'] = 'three_seeds' if len(rows) == 3 else 'single_seed' if len(rows) == 1 else 'partial_seeds'
        for metric in ('test_accuracy', 'test_macro_f1', 'validation_macro_f1'):
            values = [r[metric] for r in rows if isinstance(r[metric], (int, float))]
            row[metric + '_mean'] = statistics.mean(values) if values else ''
            row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        summaries.append(row)
    return summaries


def main():
    records, status = collect_records()
    summaries = group_records(records)
    output = Path('results/reports')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'all_json_full.txt').open('w', encoding='utf-8') as handle:
        for row in records:
            path = Path(row['file'])
            handle.write(f'\n=== {path} ===\n' + path.read_text(encoding='utf-8') + '\n')
    write_csv(output / 'all_metrics.csv', records)
    write_csv(output / 'experiment_status.csv', status)
    write_csv(output / 'summary_by_group.csv', summaries)
    write_csv(output / 'baseline_summary.csv', [r for r in summaries if r['category'] == 'baseline'])
    write_csv(output / 'cross_domain_summary.csv', [r for r in summaries if r['category'] == 'cross_domain'])
    completed = sum(r['status'] == 'complete' for r in status)
    print(f'Valid formal records: {completed}; pending: {len(status)-completed}; total saved records including optional smoke: {len(records)}.')
    print('Only corrected GCN records are included. One-seed groups have blank SD; see experiment_status.csv for pending runs.')


if __name__ == '__main__':
    main()
