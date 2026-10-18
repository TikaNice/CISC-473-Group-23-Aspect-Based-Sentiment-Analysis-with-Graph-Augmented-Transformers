"""Summarize epoch and low-resource JSON. Selection should use VALIDATION, not TEST."""
from src.result_paths import result_path

import csv
import json
import statistics
from pathlib import Path

ROOT = Path('results/reports')
ROOT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
DOMAINS = ('restaurant', 'laptop')
MODELS = ('bert', 'gcn')


def metrics(path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    val = d.get('validation_metrics', {})
    test = d.get('final_test_metrics', {})
    return {
        'val_acc': val.get('eval_accuracy'),
        'val_f1': val.get('eval_macro_f1'),
        'test_acc': test.get('test_accuracy'),
        'test_f1': test.get('test_macro_f1'),
    }


def summarize_one(file, settings, filename_fn, label):
    rows = []
    for domain in DOMAINS:
        for model in MODELS:
            for setting in settings:
                samples = []
                for seed in SEEDS:
                    path = filename_fn(model, domain, setting, seed)
                    m = metrics(path)
                    if m is not None and all(isinstance(m[k], (int, float)) for k in ('val_acc', 'val_f1', 'test_acc', 'test_f1')):
                        samples.append(m)
                if not samples:
                    continue
                record = {'domain': domain, 'model': model, label: setting, 'runs': len(samples)}
                for key in ('val_acc', 'val_f1', 'test_acc', 'test_f1'):
                    nums = [s[key] for s in samples]
                    record[key+'_mean'] = statistics.mean(nums)
                    record[key+'_std'] = statistics.stdev(nums) if len(nums) >= 2 else None
                rows.append(record)
                print(f'{domain:10} {model:4} {label}={str(setting):5} '
                      f'n={len(samples)}/3  val-F1={record["val_f1_mean"]:.4f}  '
                      f'test-acc={record["test_acc_mean"]:.4f}  test-F1={record["test_f1_mean"]:.4f}')
    if rows:
        with file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Saved:', file)
    if label == 'epochs':
        print('\nEpoch selection based ONLY on validation Macro-F1 (requires all three seeds):')
        for domain in DOMAINS:
            for model in MODELS:
                options = [r for r in rows if r['domain'] == domain and r['model'] == model and r['runs'] == 3]
                if options:
                    best = max(options, key=lambda x: x['val_f1_mean'])
                    print(f'{domain:10} {model:4} best among completed: {best["epochs"]} epochs '
                          f'(val Macro-F1 mean {best["val_f1_mean"]:.4f})')
                else:
                    print(f'{domain:10} {model:4} no complete 3-seed epoch setting')


print('HISTORICAL SWEEPS: legacy GCN alignment. Use scripts.summarize_corrected for aspectfix results.')
print('=== EPOCH SWEEP (3-epoch data reused) ===')
summarize_one(
    ROOT / 'epoch_sweep_summary.csv', (1, 2, 3, 4),
    lambda model, domain, ep, seed: result_path(model, domain, 1, seed, '' if ep == 3 else f'ep{ep}'),
    'epochs',
)
print('\n=== LOW-RESOURCE SWEEP ===')
summarize_one(
    ROOT / 'low_resource_summary.csv', ('0.1', '0.25', '0.5', '1'),
    lambda model, domain, frac, seed: result_path(model, domain, frac, seed),
    'fraction',
)
