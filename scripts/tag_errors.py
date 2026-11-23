"""Rule-based error analysis over saved per-example predictions.

Reads results/predictions/*.jsonl (written by training and evaluation), tags
test examples with heuristic linguistic categories, reports per-category
accuracy mean +/- SD over seeds, and lists consistent disagreements between
BERT and BERT+GCN.

Usage: python -m scripts.tag_errors --fraction 1 --max-examples 8
Writes results/error_analysis.md
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evidence import load_predictions

NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot)\b|n't\b",
                      re.IGNORECASE)
CONTRAST = re.compile(r"\b(but|although|though|however|yet|whereas|except)\b", re.IGNORECASE)


def tag_examples(rows):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--max-examples', type=int, default=8)
    args = p.parse_args()
    lines = ['# Error analysis (rule-based tags)', '',
             f'Training fraction {args.fraction:g}. Accuracy (%) is mean +/- std over seeds; '
             'delta is GCN minus BERT. Tags are heuristic keyword/structure rules, not manual annotation.', '']
    for domain in ('restaurant', 'laptop'):
        for eval_domain in ('restaurant', 'laptop'):
            bert = load_predictions('bert', domain, eval_domain, args.fraction)
            gcn = load_predictions('gcn', domain, eval_domain, args.fraction)
            seeds = sorted(set(bert) & set(gcn))
            if not seeds:
                continue
            rows = bert[seeds[0]]
            tags = tag_examples(rows)
            correct = {name: np.array([[x['pred'] == x['gold'] for x in preds[s]] for s in seeds])
                       for name, preds in (('bert', bert), ('gcn', gcn))}
            setting = 'in-domain' if domain == eval_domain else 'cross-domain'
            lines += [f'## Train {domain} -> test {eval_domain} ({setting}), seeds {seeds}', '',
                      '| Category | n | BERT acc | GCN acc | delta |', '|---|---|---|---|---|']
            all_tags = sorted({t for ts in tags for t in ts}, key=lambda t: (t != 'all', t))
            for tag in all_tags:
                idx = [i for i, ts in enumerate(tags) if tag in ts]
                b = 100 * correct['bert'][:, idx].mean(axis=1)
                g = 100 * correct['gcn'][:, idx].mean(axis=1)
                b_cell = f'{b.mean():.1f}'
                g_cell = f'{g.mean():.1f}'
                if len(seeds) > 1:
                    b_cell += f' +/- {b.std(ddof=1):.1f}'
                    g_cell += f' +/- {g.std(ddof=1):.1f}'
                lines.append(f'| {tag} | {len(idx)} | {b_cell} | {g_cell} | {g.mean() - b.mean():+.1f} |')
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
                    t = ', '.join(sorted(tags[i] - {'all', f"gold = {r['gold']}"})) or '-'
                    text = r['text'].replace('|', '/')
                    lines.append(f"| {text} | {r['span']} | {r['gold']} | {wrong} | {t} |")
                lines.append('')
    out = Path('results/error_analysis.md')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out.read_text())


if __name__ == '__main__':
    main()
