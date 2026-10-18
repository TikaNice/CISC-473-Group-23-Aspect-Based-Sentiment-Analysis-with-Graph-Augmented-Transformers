"""Shared training/evaluation helpers for the BERT and BERT+GCN experiments."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABEL_TO_ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
DOMAINS = ('restaurant', 'laptop')


def add_common_args(parser):
    parser.add_argument('--domain', choices=DOMAINS, default='restaurant')
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate for the pretrained BERT encoder')
    parser.add_argument('--head-lr', type=float, default=None,
                        help='learning rate for newly initialised layers (GCN + classifier); defaults to --lr')
    parser.add_argument('--eval-domains', nargs='*', choices=DOMAINS, default=list(DOMAINS),
                        help='gold test sets to evaluate on after training (in-domain and cross-domain)')
    parser.add_argument('--results-dir', default='results',
                        help='where result JSON, predictions and audits are written')
    parser.add_argument('--keep-checkpoints', action='store_true',
                        help='keep checkpoints/ after the run (deleted by default to save disk)')
    parser.add_argument('--smoke', action='store_true')
    return parser


def experiment_suffix(args):
    suffix = f'{args.domain}_fraction{args.train_fraction:g}_seed{args.seed}'
    return suffix + ('_smoke' if args.smoke else '')


def compute_metrics(pred):
    y_pred = np.asarray(pred.predictions).argmax(axis=-1)
    y_true = np.asarray(pred.label_ids)
    return {'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)}


def build_optimizer(model, lr, head_lr, weight_decay=0.01):
    """AdamW with separate learning rates for the pretrained encoder and the new head.

    Parameters under `model.bert` use `lr`; everything else (GCN layers, classifier) uses
    `head_lr`. Bias and LayerNorm weights get no weight decay, as in the HF Trainer default.
    """
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith('bert.')
        no_decay = name.endswith('bias') or 'LayerNorm' in name
        key = (is_encoder, no_decay)
        groups.setdefault(key, []).append(param)
    param_groups = [
        {'params': params, 'lr': lr if is_encoder else head_lr,
         'weight_decay': 0.0 if no_decay else weight_decay}
        for (is_encoder, no_decay), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def load_gold(domain, read_jsonl):
    gold = Path(f'data/processed/{domain}/test_gold.jsonl')
    if not gold.is_file():
        print(f'WARNING: {gold} not found; skipping {domain} test evaluation')
        return None
    return read_jsonl(gold)


def evaluate_and_dump(trainer, dataset, rows, out_path):
    """Evaluate on `dataset`, write per-example predictions to `out_path`, return metrics.

    `rows` must be aligned one-to-one with `dataset` (same order, same length).
    """
    assert len(rows) == len(dataset), f'{len(rows)} rows vs {len(dataset)} encoded examples'
    output = trainer.predict(dataset, metric_key_prefix='test')
    logits = np.asarray(output.predictions)
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    preds = logits.argmax(axis=-1)
    labels = np.asarray(output.label_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({
                'idx': i, 'text': row['text'], 'span': row['span'],
                'gold': ID_TO_LABEL[int(labels[i])], 'pred': ID_TO_LABEL[int(preds[i])],
                'probs': [round(float(p), 4) for p in probs[i]],
            }, ensure_ascii=False) + '\n')
    metrics = {k: float(v) for k, v in output.metrics.items()}
    metrics['n_evaluated'] = int(len(labels))
    metrics['confusion_matrix'] = confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist()
    metrics['predictions_file'] = str(out_path)
    return metrics


def evaluate_test_sets(trainer, args, model_tag, suffix, encode_rows, read_jsonl, result):
    """Evaluate on every gold test set in --eval-domains (in-domain and cross-domain)."""
    result['test'] = {}
    for eval_domain in args.eval_domains:
        rows = load_gold(eval_domain, read_jsonl)
        if rows is None:
            continue
        dataset, extra = encode_rows(rows)
        out = Path(args.results_dir) / 'preds' / f'{model_tag}_{suffix}_on_{eval_domain}.jsonl'
        metrics = evaluate_and_dump(trainer, dataset, rows, out)
        metrics.update(extra)
        metrics['setting'] = 'in_domain' if eval_domain == args.domain else 'cross_domain'
        metrics['test_size'] = len(rows)
        result['test'][eval_domain] = metrics
        print(f'[{model_tag}] {args.domain} -> {eval_domain}: '
              f'acc={metrics["test_accuracy"]:.4f} macro_f1={metrics["test_macro_f1"]:.4f} '
              f'n={metrics["n_evaluated"]}/{len(rows)}')


def save_result(result, trainer, model_tag, suffix, output_dir, args):
    result['log_history'] = trainer.state.log_history
    target = Path(args.results_dir) / f'{model_tag}_{suffix}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    print('Saved:', target)
    if not args.keep_checkpoints:
        shutil.rmtree(output_dir, ignore_errors=True)
