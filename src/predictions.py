"""Per-example prediction records used for error analysis.

Every evaluation writes one JSONL file per (model, train domain, fraction,
seed, evaluation domain). Row order matches the evaluation dataset, and the
``index`` field is the position in that dataset, so predictions can be joined
back to the gold records without relying on filenames.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

LABELS = ['negative', 'neutral', 'positive']
PRED_DIR = Path('results/predictions')


def prediction_path(model, train_domain, fraction, seed, tag='', eval_domain=None):
    eval_domain = eval_domain or train_domain
    suffix = f'{model}_{train_domain}_fraction{float(fraction):g}_seed{seed}'
    if tag:
        suffix += '_' + tag
    return PRED_DIR / f'{suffix}_on_{eval_domain}.jsonl'


def write_predictions(path, rows, guesses, model, train_domain, eval_domain,
                      fraction, seed, tag='', graph=None, device=''):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) != len(guesses):
        raise ValueError(f'{path}: prediction count differs from row count')
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            for i, (row, guess) in enumerate(zip(rows, guesses)):
                f.write(json.dumps({
                    'index': i,
                    'text': row['text'],
                    'span': row['span'],
                    'start': row.get('start'),
                    'end': row.get('end'),
                    'gold': row['label'],
                    'pred': LABELS[int(guess)],
                    'model': model,
                    'train_domain': train_domain,
                    'eval_domain': eval_domain,
                    'fraction': float(fraction),
                    'seed': int(seed),
                    'run_tag': tag,
                    'graph': graph,
                    'device': device,
                }, ensure_ascii=False) + '\n')
            f.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path
