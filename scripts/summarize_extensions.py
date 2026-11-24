"""Compute seeded extension summaries and paired test deltas."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

from scripts.summarize_results import write_csv
from src.evidence import validate_result
from src.result_paths import extension_grid

OUT = Path('results/reports/extension_summary')
METRICS = {'macro_f1': 'Macro-F1', 'accuracy': 'Accuracy'}
ARM_LABELS = {'pooling': 'BERT aspect pooling', 'constituency': 'Constituency GCN'}


@lru_cache(maxsize=None)
def scores(model, domain, fraction, seed, graph):
    return validate_result(model, domain, fraction, seed, graph)['final_test_metrics']


def main():
    groups = defaultdict(list)
    for model, domain, fraction, seed, graph in extension_grid():
        groups[domain, fraction, model, graph].append(seed)

    rows = []
    lines = [
        '# Extension results', '',
        'Means ± sample SD over seeds 42, 43 and 44. Scores are percentages. Deltas are percentage points.',
        'Each delta is the extension score minus the named reference score for the same seed.',
        '',
        '| Domain | Budget | Arm | Metric | Score | Δ vs BERT CLS | Δ vs dependency GCN |',
        '|---|---:|---|---|---:|---:|---:|',
    ]
    for (domain, fraction, model, graph), seeds in groups.items():
        for metric in METRICS:
            key = f'test_{metric}'
            observed = [scores(model, domain, fraction, seed, graph)[key] for seed in seeds]
            row = dict(domain=domain, fraction=fraction, arm=graph, metric=metric,
                       n_seeds=len(seeds), mean=mean(observed), sd=stdev(observed))
            paired = {}
            for ref_name, ref_model, ref_graph in (
                ('bert', 'bert', 'none'), ('dependency', 'gcn', 'dependency')
            ):
                deltas = [value - scores(ref_model, domain, fraction, seed, ref_graph)[key]
                          for seed, value in zip(seeds, observed, strict=True)]
                row[f'delta_vs_{ref_name}_mean'] = mean(deltas)
                row[f'delta_vs_{ref_name}_sd'] = stdev(deltas)
                paired[ref_name] = f'{100 * mean(deltas):+.2f} ± {100 * stdev(deltas):.2f}'
            rows.append(row)
            lines.append(f'| {domain.capitalize()} | {fraction:.0%} | {ARM_LABELS[graph]} | {METRICS[metric]} | '
                         f'{100 * row["mean"]:.2f} ± {100 * row["sd"]:.2f} | '
                         f'{paired["bert"]} | {paired["dependency"]} |')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUT.with_suffix('.csv'), rows)
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
