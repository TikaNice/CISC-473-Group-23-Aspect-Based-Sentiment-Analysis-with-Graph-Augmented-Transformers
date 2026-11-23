"""Summarize the comparable three-epoch BERT and dependency-GCN budgets."""
from pathlib import Path
import statistics
from scripts.summarize_results import collect_records, group_records, write_csv


def main():
    records, _ = collect_records()
    output = Path('results/reports')
    output.mkdir(parents=True, exist_ok=True)
    selected = [r for r in records if r['category'] in ('baseline', 'low_resource')]
    rows = []
    for group in group_records(selected):
        row = dict(domain=group['domain'], model=group['model'], fraction=group['fraction'],
                   runs=group['n_seeds'], seeds=group['seeds'], revision=group['revision'],
                   statistics_status=group['statistics_status'])
        for short, metric in [('val_acc','validation_accuracy'), ('val_f1','validation_macro_f1'),
                              ('test_acc','test_accuracy'), ('test_f1','test_macro_f1')]:
            # Group summaries normally omit validation accuracy; derive it from source rows.
            subset = [r for r in selected if r['domain']==group['domain'] and r['model']==group['model']
                      and r['fraction']==group['fraction'] and r['epochs']==group['epochs']]
            values = [r[metric] for r in subset if isinstance(r[metric], (int,float))]
            row[short+'_mean'] = statistics.mean(values) if values else ''
            row[short+'_std'] = statistics.stdev(values) if len(values)>1 else ''
        rows.append(row)
    write_csv((output / 'low_resource_summary.csv'), rows)
    print('Saved:', (output / 'low_resource_summary.csv'), '(valid records only; sample counts shown)')


if __name__ == '__main__':
    main()
