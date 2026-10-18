"""Rule-based error analysis of saved test predictions (BERT vs BERT+GCN).

Tags every test example with simple linguistic categories, then reports per-category
accuracy for each model (mean +/- std over seeds) and lists examples where the two
models consistently disagree.

Usage: python scripts/tag_errors.py --results-dir results/main --fraction 1
Writes <results-dir>/error_analysis.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot)\b|n't\b",
                      re.IGNORECASE)
CONTRAST = re.compile(r"\b(but|although|though|however|yet|whereas|except)\b", re.IGNORECASE)


def tag_examples(rows):
    """rows: list of dicts with text/span/gold, in test-set order. Returns list of tag sets."""
    by_sentence = defaultdict(list)
    for r in rows:
        by_sentence[r['text']].append(r['gold'])
    tags = []
    for r in rows:
        t = {'all'}
        if NEGATION.search(r['text']):
            t.add('negation')
        if CONTRAST.search(r['text']):
            t.add('contrast word')
        if len(by_sentence[r['text']]) > 1:
            t.add('multi-aspect sentence')
        if len(set(by_sentence[r['text']])) > 1:
            t.add('mixed polarity in sentence')
        if len(r['text'].split()) > 25:
            t.add('long sentence (>25 words)')
        t.add(f"gold = {r['gold']}")
        tags.append(t)
    return tags


def load_preds(pred_dir, model, domain, eval_domain, fraction):
    out = {}
    for path in sorted(pred_dir.glob(f'{model}_{domain}_fraction{fraction:g}_seed*_on_{eval_domain}.jsonl')):
        seed = int(re.search(r'seed(\d+)', path.name).group(1))
        out[seed] = [json.loads(line) for line in path.open(encoding='utf-8')]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='results/main')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--max-examples', type=int, default=8)
    args = p.parse_args()
    root = Path(args.results_dir)
    lines = ['# Error analysis (rule-based tags)', '',
             f'Training fraction {args.fraction:g}. Accuracy (%) is mean ± std over seeds; '
             'Δ is GCN − BERT. Tags are heuristic keyword/structure rules, not manual annotation.', '']
    for domain in ('restaurant', 'laptop'):
        for eval_domain in ('restaurant', 'laptop'):
            bert = load_preds(root / 'preds', 'bert', domain, eval_domain, args.fraction)
            gcn = load_preds(root / 'preds', 'gcn', domain, eval_domain, args.fraction)
            seeds = sorted(set(bert) & set(gcn))
            if not seeds:
                continue
            rows = bert[seeds[0]]
            tags = tag_examples(rows)
            # correct[model] has shape (n_seeds, n_examples)
            correct = {name: np.array([[x['pred'] == x['gold'] for x in preds[s]] for s in seeds])
                       for name, preds in (('bert', bert), ('gcn', gcn))}
            setting = 'in-domain' if domain == eval_domain else 'cross-domain'
            lines += [f'## Train {domain} → test {eval_domain} ({setting}), seeds {seeds}', '',
                      '| Category | n | BERT acc | GCN acc | Δ |', '|---|---|---|---|---|']
            all_tags = sorted({t for ts in tags for t in ts}, key=lambda t: (t != 'all', t))
            for tag in all_tags:
                idx = [i for i, ts in enumerate(tags) if tag in ts]
                b = 100 * correct['bert'][:, idx].mean(axis=1)
                g = 100 * correct['gcn'][:, idx].mean(axis=1)
                lines.append(f'| {tag} | {len(idx)} | {b.mean():.1f} ± {b.std(ddof=1) if len(seeds) > 1 else 0:.1f} | '
                             f'{g.mean():.1f} ± {g.std(ddof=1) if len(seeds) > 1 else 0:.1f} | {g.mean() - b.mean():+.1f} |')
            # Consistent disagreements: one model right on every seed, the other wrong on every seed.
            gcn_only = np.where(correct['gcn'].all(0) & ~correct['bert'].any(0))[0]
            bert_only = np.where(correct['bert'].all(0) & ~correct['gcn'].any(0))[0]
            lines += ['', f'GCN right on all seeds and BERT wrong on all seeds: **{len(gcn_only)}** examples; '
                          f'the reverse: **{len(bert_only)}**.', '']
            for title, idx, wrong_model in (('GCN right, BERT wrong', gcn_only, bert),
                                            ('BERT right, GCN wrong', bert_only, gcn)):
                if not len(idx):
                    continue
                lines += [f'**{title}** (first {min(len(idx), args.max_examples)}):', '',
                          '| Sentence | Aspect | Gold | Wrong prediction | Tags |', '|---|---|---|---|---|']
                for i in idx[:args.max_examples]:
                    r = rows[i]
                    wrong = wrong_model[seeds[0]][i]['pred']
                    t = ', '.join(sorted(tags[i] - {'all', f"gold = {r['gold']}"})) or '—'
                    text = r['text'].replace('|', '/')
                    lines.append(f"| {text} | {r['span']} | {r['gold']} | {wrong} | {t} |")
                lines.append('')
    out = root / 'error_analysis.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out.read_text())


if __name__ == '__main__':
    main()
