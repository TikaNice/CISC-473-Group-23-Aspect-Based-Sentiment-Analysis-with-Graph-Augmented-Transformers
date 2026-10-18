"""Compare corrected GCN runs to matching existing BERT runs, one seed at a time."""
import argparse
import csv
import json
from pathlib import Path
from src.result_paths import result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = []
    for domain in ('restaurant', 'laptop'):
        for fraction in (1, .1, .25, .5):
            for model in ('bert', 'gcn'):
                path = result_path(model, domain, fraction, args.seed, 'aspectfix' if model == 'gcn' else '')
                rows.append(read_row(path, model, domain, fraction, args.seed, model == 'gcn'))
    for model in ('bert', 'gcn'):
        tag = '_aspectfix' if model == 'gcn' else ''
        path = Path('results/cross_domain/restaurant_to_laptop') / model / f'{model}_restaurant_to_laptop_seed{args.seed}{tag}.json'
        rows.append(read_row(path, model, 'restaurant_to_laptop', 1, args.seed, model == 'gcn'))
    target = Path(f'results/reports/corrected_seed{args.seed}.csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# Corrected BERT + GCN comparison — seed {args.seed}', '',
             'Single-seed preliminary results. No multi-seed mean, SD, or significance claim.',
             'BERT reuses the existing matching project configuration. GCN uses case-sensitive aspect alignment.',
             'Training settings remain the project protocol (3 epochs, batch 8, LR 2e-5, length 128), not the 2019 author protocol.',
             'Historical GCN scores are excluded. Epoch-budget experiments are deferred.', '',
             '| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |',
             '|---|---|---|---|---:|---:|']
    for row in rows:
        acc = f"{row['test_accuracy']:.4f}" if row['status'] == 'complete' else '—'
        f1 = f"{row['test_macro_f1']:.4f}" if row['status'] == 'complete' else '—'
        lines.append(f"| {row['domain']} | {row['fraction']} | {row['model']} | {row['status']} | {acc} | {f1} |")
    lines += ['', 'Matching data splits and sentence-budget sampling use seed 2026. Existing train/test overlaps remain disclosed in IMPLEMENTATION.md.',
              'Course reproducibility guidance requests at least three runs; this one-seed correction is an interim checkpoint for the group, with remaining seeds still pending.', '']
    target.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    done = sum(row['status'] == 'complete' for row in rows)
    print(f'Updated {target}: {done}/{len(rows)} comparison rows complete.')


def read_row(path, model, domain, fraction, seed, corrected):
    row = dict(domain=domain, fraction=fraction, model=model, seed=seed,
               revision='aspectfix_v1' if corrected else 'existing_bert', status='pending',
               validation_accuracy='', validation_macro_f1='', test_accuracy='', test_macro_f1='', file=str(path))
    if not path.is_file():
        return row
    obj = json.loads(path.read_text(encoding='utf-8'))
    if corrected:
        assert obj.get('aspect_alignment') == 'case_sensitive_ordinal_v1', path
    assert obj['seed'] == seed, path
    val = obj.get('validation_metrics', {})
    test = obj.get('final_test_metrics', obj)
    row.update(status='complete', validation_accuracy=val.get('eval_accuracy', ''),
               validation_macro_f1=val.get('eval_macro_f1', ''),
               test_accuracy=test['test_accuracy'], test_macro_f1=test['test_macro_f1'])
    return row


if __name__ == '__main__':
    main()
