"""Summarize available BERT/corrected-GCN sweeps; select epochs only with three seeds."""
from pathlib import Path
import statistics
from scripts.summarize_results import collect_records, group_records, write_csv


def main():
    records, _ = collect_records()
    output = Path('results/reports')
    output.mkdir(parents=True, exist_ok=True)
    for label, categories, filename in [
        ('epochs', ('baseline', 'epoch_sweep'), 'epoch_sweep_summary.csv'),
        ('fraction', ('baseline', 'low_resource'), 'low_resource_summary.csv'),
    ]:
        selected = [r for r in records if r['category'] in categories]
        rows = []
        for group in group_records(selected):
            row = dict(domain=group['domain'], model=group['model'], **{label:group[label]},
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
        write_csv(output/filename, rows)
        print('Saved:', output/filename, '(valid records only; sample counts shown)')
        if label == 'epochs':
            for domain in ('restaurant','laptop'):
                for model in ('bert','gcn'):
                    candidates=[r for r in rows if r['domain']==domain and r['model']==model and r['runs']==3]
                    if candidates:
                        best=max(candidates,key=lambda r:r['val_f1_mean'])
                        print(f'{domain} {model}: best completed three-seed epoch budget={best["epochs"]} (validation Macro-F1)')
                    else:
                        print(f'{domain} {model}: epoch-budget selection pending; no complete three-seed setting')


if __name__ == '__main__':
    main()
