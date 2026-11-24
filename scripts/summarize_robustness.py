"""Summarize matched supplementary and canonical results over training seeds."""
import csv
from functools import cache
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile

from scripts.run_robustness import ARMS, MODELS, tagged_result, validate_run
from src.evidence import validate_result
from src.result_paths import DOMAINS, GRAPH_TAGS, SEEDS, result_path

METRICS = {
    'val_accuracy': ('validation_metrics', 'eval_accuracy'),
    'val_macro_f1': ('validation_metrics', 'eval_macro_f1'),
    'test_accuracy': ('final_test_metrics', 'test_accuracy'),
    'test_macro_f1': ('final_test_metrics', 'test_macro_f1'),
}
OUTPUT = Path('results/reports/robustness_summary.csv')


@cache
def read_run(arm, model, domain, fraction, seed):
    graph = MODELS[model]
    if arm == 'canonical':
        path = result_path(model, domain, fraction, seed, GRAPH_TAGS[graph])
        result = validate_result(model, domain, fraction, seed, graph)
    else:
        _, path = tagged_result(arm, model, domain, fraction, seed)
        result = validate_run(arm, model, domain, fraction, seed)
    return path, result


def comparison_rows(arm, domain, fraction, reference_fraction, models):
    trials = {}
    controls = {}
    for model in models:
        trials[model] = [read_run(arm, model, domain, fraction, seed) for seed in SEEDS]
        controls[model] = [read_run('canonical', model, domain, reference_fraction, seed)
                           for seed in SEEDS]
        for (path, trial), (_, control) in zip(trials[model], controls[model]):
            keys = ['validation_size', 'test_size', 'device', 'torch_version', 'dynamic_padding']
            if arm == 'updatematch10' and reference_fraction == fraction:
                keys.append('train_size')
            if any(trial.get(key) != control.get(key) for key in keys):
                raise ValueError(f'{path}: protocol metadata differs from the canonical counterpart')

    def scores(runs, group, key):
        values = [[record[group][key] for _, record in runs[model]] for model in models]
        if len(models) == 1:
            return values[0]
        bert, gcn = values
        return [g - b for b, g in zip(bert, gcn)]

    def files(runs):
        return '|'.join(str(path) for seed_records in zip(*runs.values())
                        for path, _ in seed_records)

    for metric, (group, key) in METRICS.items():
        supplementary = scores(trials, group, key)
        canonical = scores(controls, group, key)
        differences = [test - base for test, base in zip(supplementary, canonical)]
        yield {
            'arm': arm, 'domain': domain, 'model': models[0] if len(models) == 1 else 'gcn_minus_bert',
            'fraction': fraction, 'subset_seed': ARMS[arm][0], 'epochs': ARMS[arm][1],
            'canonical_fraction': reference_fraction, 'canonical_subset_seed': 2026, 'canonical_epochs': 3,
            'metric': metric, 'n': len(SEEDS), 'seeds': '|'.join(map(str, SEEDS)),
            'supplementary_mean': mean(supplementary), 'supplementary_sample_sd': stdev(supplementary),
            'canonical_mean': mean(canonical), 'canonical_sample_sd': stdev(canonical),
            'paired_delta_mean': mean(differences), 'paired_delta_sample_sd': stdev(differences),
            **{f'paired_delta_seed{seed}': delta for seed, delta in zip(SEEDS, differences)},
            'supplementary_positive_count': sum(value > 0 for value in supplementary),
            'canonical_positive_count': sum(value > 0 for value in canonical),
            'supplementary_sources': files(trials), 'canonical_sources': files(controls),
        }


def main():
    rows = []
    for arm, (_, _, fractions) in ARMS.items():
        for domain, fraction in product(DOMAINS, fractions):
            references = (fraction, 1.0) if arm == 'updatematch10' else (fraction,)
            for reference in references:
                for models in (('bert',), ('gcn',), ('bert', 'gcn')):
                    rows.extend(comparison_rows(arm, domain, fraction, reference, models))

    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                            dir=OUTPUT.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
            handle.close()
            temporary.replace(OUTPUT)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Wrote {len(rows)} comparisons to {OUTPUT}')


if __name__ == '__main__':
    main()
