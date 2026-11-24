"""Validate the canonical runs and their per-example evaluation evidence."""
import json
import math
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from src.predictions import LABELS, prediction_path
from src.result_paths import GRAPH_TAGS, SEEDS, checkpoint_dir, extension_grid, result_path, training_grid, transfer_path


@lru_cache(maxsize=2)
def gold_rows(domain):
    path = Path(f'data/processed/{domain}/test_gold.jsonl')
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row.get('span') and row['label'] in LABELS]


def read_predictions(model, domain, fraction, seed, graph, eval_domain, tag=None):
    tag = GRAPH_TAGS[graph] if tag is None else tag
    path = prediction_path(model, domain, fraction, seed, tag, eval_domain)
    expected = dict(model=model, train_domain=domain, eval_domain=eval_domain,
                    fraction=fraction, seed=seed, run_tag=tag, graph=graph)
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    gold = gold_rows(eval_domain)
    if len(rows) != len(gold) or not rows:
        raise ValueError(f'{path}: expected {len(gold)} Gold Test predictions, found {len(rows)}')
    for index, (row, source) in enumerate(zip(rows, gold)):
        if (row.get('index') != index or row.get('gold') != source['label']
                or row.get('pred') not in LABELS
                or any(row.get(key) != source.get(key) for key in ('text', 'span', 'start', 'end'))
                or any(row.get(key) != value for key, value in expected.items())):
            raise ValueError(f'{path}: mismatched prediction metadata or Gold Test record at index {index}')
    return rows


def load_predictions(model, domain, eval_domain, fraction, seeds=SEEDS):
    graph = 'dependency' if model == 'gcn' else 'none'
    tag = GRAPH_TAGS[graph]
    return {seed: read_predictions(model, domain, fraction, seed, graph, eval_domain)
            for seed in seeds
            if prediction_path(model, domain, fraction, seed, tag, eval_domain).is_file()}


def validate_result(model, domain, fraction, seed, graph, eval_domain=None, tag=None):
    tag = GRAPH_TAGS[graph] if tag is None else tag
    transfer = eval_domain is not None
    path = transfer_path(model, seed, tag) if transfer else result_path(model, domain, fraction, seed, tag)
    obj = json.loads(path.read_text(encoding='utf-8'))
    if transfer:
        expected = dict(seed=seed, model=model, run_tag=tag, source_training_domain=domain,
                        target_test_domain=eval_domain, experiment='restaurant_to_laptop_zero_shot')
        checkpoint = Path(obj.get('checkpoint', ''))
        root = checkpoint_dir(model, domain, fraction, seed, tag)
        if checkpoint.parent != root or not checkpoint.name.startswith('checkpoint-'):
            raise ValueError(f'{path}: invalid source checkpoint reference')
        source = result_path(model, domain, fraction, seed, tag)
        if obj.get('source_result_sha256') != sha256(source.read_bytes()).hexdigest():
            raise ValueError(f'{path}: transfer is not linked to the current source training result')
    else:
        suffix = f'{domain}_fraction{fraction:g}_seed{seed}' + (f'_{tag}' if tag else '')
        expected = dict(seed=seed, experiment=suffix)
        if model == 'gcn':
            expected['graph_type'] = graph
    if model == 'gcn':
        expected['aspect_alignment'] = 'case_sensitive_ordinal_v1'
    if graph in ('pooling', 'constituency'):
        expected['device'] = 'cuda:0'
    if graph == 'constituency':
        from src.constituency_graph import PARSER
        expected['constituency_parser'] = PARSER
    if any(obj.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path}: result metadata does not match its run')
    if not transfer:
        validation = obj.get('validation_metrics', {})
        for field in ('eval_accuracy', 'eval_macro_f1'):
            value = validation.get(field)
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f'{path}: invalid validation metric {field}')

    rows = read_predictions(model, domain, fraction, seed, graph, eval_domain or domain, tag)
    if obj.get('test_size') != len(rows) or any(row.get('device') != obj.get('device') for row in rows):
        raise ValueError(f'{path}: evaluation size or device disagrees with its predictions')
    counts = Counter((row['gold'], row['pred']) for row in rows)
    truth = Counter(row['gold'] for row in rows)
    guessed = Counter(row['pred'] for row in rows)
    accuracy = sum(counts[label, label] for label in LABELS) / len(rows)
    f1 = sum(2 * counts[label, label] / (truth[label] + guessed[label])
             for label in LABELS if truth[label] + guessed[label]) / len(LABELS)
    metrics = obj if transfer else obj['final_test_metrics']
    for field, value in (('test_accuracy', accuracy), ('test_macro_f1', f1)):
        recorded = metrics.get(field)
        if not isinstance(recorded, (int, float)) or not math.isclose(recorded, value, abs_tol=1e-10):
            raise ValueError(f'{path}: {field} disagrees with its predictions')
    return obj


INVALID_EVIDENCE = (OSError, ValueError, KeyError, TypeError, AttributeError)


def evidence_grid():
    yield from ((*run, None) for run in training_grid())
    yield from ((*run, None) for run in extension_grid())
    yield from ((model, 'restaurant', 1.0, seed, graph, 'laptop')
                for model, graph in (('bert', 'none'), ('gcn', 'dependency')) for seed in SEEDS)


def grid_errors():
    errors = []
    for run in evidence_grid():
        try:
            validate_result(*run)
        except INVALID_EVIDENCE as exc:
            errors.append(str(exc))
    return errors


if __name__ == '__main__':
    errors = grid_errors()
    for error in errors:
        print(error)
    total = sum(1 for _ in evidence_grid())
    if errors:
        raise SystemExit(f'{len(errors)} of {total} training/transfer records are missing or invalid.')
    print(f'Validated {total} training/transfer results against their Gold Test predictions.')
