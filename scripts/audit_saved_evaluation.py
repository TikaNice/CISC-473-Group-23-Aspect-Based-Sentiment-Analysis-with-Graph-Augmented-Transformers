"""Audit canonical saved predictions. Run from final/code with python -m scripts.audit_saved_evaluation."""
import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from src.evidence import gold_rows, read_predictions
from src.predictions import LABELS
from src.result_paths import GRAPH_TAGS, result_path, transfer_path

DOMAINS = ('restaurant', 'laptop')
BUDGETS = (.1, .25, .5, 1.)
SEEDS = (42, 43, 44)
ARMS = (('bert', 'none'), ('gcn', 'dependency'))
OUT = Path('results/reports')
VERIFIED = Path('results/verified_error_examples.json')
METRICS = ('accuracy', 'macro_f1')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def aspect(row):
    return row['span'], row.get('ordinal', 0)


def overlap_inventory():
    sources = defaultdict(list)
    for domain in DOMAINS:
        path = Path(f'data/raw/{domain}/train.jsonl')
        raw = [(line, json.loads(value)) for line, value in
               enumerate(path.read_text(encoding='utf-8').splitlines(), 1) if value.strip()]
        valid = [(line, row) for line, row in raw
                 if row.get('label') in LABELS and row.get('text') and row.get('span')]
        groups = [row['text'].casefold().strip() for _, row in valid]
        split = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=2026)
        train, validation = next(split.split(valid, groups=groups))
        splits = {valid[i][0]: name for name, indices in
                  [('train', train), ('validation', validation)] for i in indices}
        # Mirror the fixed sentence-group budget sampler used for both models.
        train_groups = sorted({groups[i] for i in train})
        random.Random(2026).shuffle(train_groups)
        group_rank = {group: rank for rank, group in enumerate(train_groups)}
        for line, row in raw:
            split_name = splits.get(line, 'filtered')
            rank = group_rank.get(row['text'].casefold().strip(), len(train_groups))
            budgets = [f'{budget:.0%}' for budget in BUDGETS
                       if split_name == 'train' and rank < math.ceil(len(train_groups) * budget)]
            sources[normalize(row['text'])].append((domain, line, row, split_name, budgets))
    matches = []
    for domain in DOMAINS:
        for index, gold in enumerate(gold_rows(domain)):
            for source_domain, line, row, split, budgets in sources[normalize(gold['text'])]:
                matches.append(dict(
                    test_domain=domain, test_index=index, source_domain=source_domain,
                    source_line=line, split=split, train_budgets=', '.join(budgets),
                    match='exact' if row['text'] == gold['text'] else 'normalized-only',
                    same_aspect=aspect(row) == aspect(gold), same_label=row['label'] == gold['label'],
                    test_span=gold['span'], test_ordinal=gold.get('ordinal', 0),
                    test_start=gold['start'], test_end=gold['end'], test_label=gold['label'],
                    test_text=gold['text'], source_span=row['span'],
                    source_ordinal=row.get('ordinal', 0), source_label=row['label'], source_text=row['text']))
    return matches


def metrics(rows):
    gold = [row['gold'] for row in rows]
    pred = [row['pred'] for row in rows]
    return dict(accuracy=accuracy_score(gold, pred),
                macro_f1=f1_score(gold, pred, labels=LABELS, average='macro', zero_division=0))


def runs():
    for domain in DOMAINS:
        for fraction in BUDGETS:
            arms = ARMS + ((('gcn', 'identity'),) if fraction in (.1, 1.) else ())
            for model, graph in arms:
                for seed in SEEDS:
                    yield model, domain, fraction, seed, graph, domain
    for model, graph in ARMS:
        for seed in SEEDS:
            yield model, 'restaurant', 1., seed, graph, 'laptop'


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare(excluded):
    per_run = []
    groups = defaultdict(list)
    full_data = {}
    for model, domain, fraction, seed, graph, target in runs():
        tag = GRAPH_TAGS[graph]
        rows = read_predictions(model, domain, fraction, seed, graph, target, tag)
        scored = [row for row in rows if row['index'] not in excluded[target]]
        before, after = metrics(rows), metrics(scored)
        transfer = domain != target
        path = transfer_path(model, seed, tag) if transfer else result_path(model, domain, fraction, seed, tag)
        result = json.loads(path.read_text(encoding='utf-8'))
        recorded = result if transfer else result['final_test_metrics']
        assert all(math.isclose(value, recorded['test_' + key], rel_tol=0, abs_tol=1e-12)
                   for key, value in before.items()), path
        item = dict(train_domain=domain, test_domain=target, fraction=fraction,
                    graph=graph, seed=seed, model=model, n_all=len(rows), n_excluded=len(rows)-len(scored))
        for key in METRICS:
            item.update({f'{key}_all': before[key], f'{key}_filtered': after[key],
                         f'{key}_change': after[key] - before[key]})
        per_run.append(item)
        groups[domain, target, fraction, graph].append(item)
        if not transfer and fraction == 1 and graph != 'identity':
            full_data[domain, model, seed] = rows
    assert len(per_run) == 66
    return per_run, groups, full_data


def sensitivity_table(groups):
    lines = [
        '| Train → test | Budget | Comparison | Metric | Left all → filtered | Right all → filtered | Right − left all → filtered | Gap change |',
        '|---|---:|---|---|---|---|---|---:|',
    ]
    comparisons = []
    changes = []
    for domain, target, budget, graph in groups:
        if graph != 'dependency':
            continue
        right = groups[domain, target, budget, graph]
        for control, label in [('none', 'BERT → GCN'), ('identity', 'Identity → GCN')]:
            left = groups.get((domain, target, budget, control))
            if left is None:
                continue
            assert [row['seed'] for row in left] == [row['seed'] for row in right] == list(SEEDS)
            for metric in METRICS:
                item = dict(train_domain=domain, test_domain=target, fraction=budget,
                            comparison=label, metric=metric)
                for variant in ('all', 'filtered'):
                    col = f'{metric}_{variant}'
                    for side, rows in [('left', left), ('right', right)]:
                        values = [row[col] for row in rows]
                        item[f'{side}_{variant}_mean'] = mean(values)
                        item[f'{side}_{variant}_sd'] = stdev(values)
                    item[f'gap_{variant}'] = mean(r[col] - l[col] for l, r in zip(left, right))
                original, filtered = item['gap_all'], item['gap_filtered']
                item['gap_change'] = filtered - original
                if not math.isclose(original, 0, abs_tol=1e-12) and original * filtered <= 0:
                    changes.append(f'{domain} → {target}, {budget:.0%}, {label}, {metric}')
                elif math.isclose(original, 0, abs_tol=1e-12) and not math.isclose(filtered, 0, abs_tol=1e-12):
                    changes.append(f'{domain} → {target}, {budget:.0%}, {label}, {metric}')
                comparisons.append(item)
                cells = [' → '.join(f'{100*item[f"{side}_{v}_mean"]:.2f} ± {100*item[f"{side}_{v}_sd"]:.2f}'
                                    for v in ('all', 'filtered')) for side in ('left', 'right')]
                lines.append(f'| {domain} → {target} | {budget:.0%} | {label} | {metric} | '
                             f'{cells[0]} | {cells[1]} | {100*original:+.2f} → {100*filtered:+.2f} | '
                             f'{100*(filtered-original):+.2f} |')
    return lines, comparisons, changes


def candidate_examples(rules):
    return {
        category: {(domain, index) for domain in DOMAINS
                   for index, row in enumerate(gold_rows(domain))
                   if re.search(rule['pattern'], row['text'], re.IGNORECASE)}
        for category, rule in rules.items()
    }


def category_analysis(full_data, rules, candidates):
    lines = ['## Manually verified error categories', '']
    results = []
    for domain in DOMAINS:
        rows = gold_rows(domain)
        invalid = [row for row in rows if not (0 <= row['start'] < row['end'] <= len(row['text'])
                   and row['text'][row['start']:row['end']] == row['span'])]
        assert not invalid, (domain, invalid)
        lines.append(f'- {domain.capitalize()} has a nonempty, exact character span on all {len(rows)} '
                     'scored rows. There are zero scored targets without explicit spans.')
    for category, rule in rules.items():
        verified = {(entry['domain'], entry['index']): entry['reason'] for entry in rule['verified']}
        assert len(verified) == len(rule['verified'])
        assert verified.keys() <= candidates[category] and all(verified.values())
        lines += ['', f'### {rule["title"]}', '', rule['definition'], '',
                  f'Candidate regex (case-insensitive) <code>{rule["pattern"].replace("|", "&#124;")}</code>. '
                  f'Manual review retained {len(verified)} of {len(candidates[category])} candidate aspect rows '
                  f'across {len({gold_rows(d)[i]["text"] for d, i in verified})} distinct sentences.', '',
                  '| Domain | Candidate rows | Verified rows | BERT errors by seed | GCN errors by seed |',
                  '|---|---:|---:|---|---|']
        for domain in DOMAINS:
            indices = [index for d, index in verified if d == domain]
            size = len(indices)
            nc = sum(d == domain for d, _ in candidates[category])
            cells = []
            for model in ('bert', 'gcn'):
                wrong = []
                for seed in SEEDS:
                    preds = full_data[domain, model, seed]
                    errors = sum(preds[i]['pred'] != preds[i]['gold'] for i in indices)
                    wrong.append(errors)
                    results.append(dict(category=category, domain=domain, candidates=nc, verified=size,
                                        model=model, seed=seed, errors=errors,
                                        error_rate=errors/size if size >= rule['min_rows_for_rate'] else ''))
                cell = ', '.join(map(str, wrong)) if size else '—'
                if size >= rule['min_rows_for_rate']:
                    cell += f' ({100*mean(wrong)/size:.1f}% ± {100*stdev(wrong)/size:.1f} pp)'
                cells.append(cell)
            lines.append(f'| {domain} | {nc} | {size} | {cells[0]} | {cells[1]} |')
        lines += ['', f'Error counts follow seeds {", ".join(map(str, SEEDS))}. Rates require at least '
                  f'{rule["min_rows_for_rate"]} verified rows in a domain. Rates are descriptive and share '
                  'the same examples across seeds.', '', 'Verified illustrations', '']
        for domain, index in list(verified)[:3]:
            row = gold_rows(domain)[index]
            guesses = [' / '.join(full_data[domain, model, seed][index]['pred'] for seed in SEEDS)
                       for model in ('bert', 'gcn')]
            lines.append(f'- {domain.capitalize()} {index}, **{row["span"]}** ({row["label"]}). '
                         f'“{row["text"]}” {verified[domain, index]} '
                         f'BERT {guesses[0]}. GCN {guesses[1]}.')
    return lines, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', action='store_true', help='Print regex candidates for manual review')
    args = parser.parse_args()
    rules = json.loads(VERIFIED.read_text(encoding='utf-8'))
    candidates = candidate_examples(rules)
    if args.candidates:
        for category, items in candidates.items():
            for domain, index in sorted(items):
                print(category, domain, index, gold_rows(domain)[index])
        return
    matches = overlap_inventory()
    assert normalize('  Ａ\tPhrase ') == normalize('a phrase')
    assert {(m['test_domain'], m['test_index'], m['source_domain'], m['split'], m['source_line'])
            for m in matches} == {('restaurant', 679, 'restaurant', 'train', 3010),
                                  ('laptop', 211, 'laptop', 'validation', 996)}
    assert len(matches) == 2 and all(m['match'] == 'exact' and m['same_aspect'] and m['same_label']
                                     for m in matches)
    assert [m['train_budgets'] for m in matches if m['split'] == 'train'] == ['100%']
    excluded = {domain: {m['test_index'] for m in matches if m['test_domain'] == domain} for domain in DOMAINS}
    per_run, groups, full_data = compare(excluded)
    table, summaries, changes = sensitivity_table(groups)
    max_change = 100 * max(abs(item['gap_change']) for item in summaries)
    analysis, error_rows = category_analysis(full_data, rules, candidates)
    lines = ['# Saved-prediction evaluation audit', '',
             'Run from `final/code` with `.venv/bin/python -m scripts.audit_saved_evaluation`.', '',
             '## Overlap inventory', '',
             'Gold indices are zero-based after the canonical Gold Test filter. Source line numbers '
             'are one-based in the raw training JSONL. Every Gold sentence was compared against both '
             'raw training files using Unicode NFKC, case folding and whitespace collapse. The audit '
             'distinguishes literal text equality from normalized-only equality. Aspect identity '
             'uses the exact span string and occurrence ordinal.', '']
    for item in matches:
        lines.append(f'- {item["test_domain"].capitalize()} Gold {item["test_index"]}, '
                     f'raw line {item["source_line"]} ({item["split"]}). “{item["test_text"]}” '
                     f'Exact text, same aspect **{item["test_span"]}**, same **{item["test_label"]}** label.')
    lines += ['', '## Overlap sensitivity', '',
              'Each run uses the same domain-specific Gold Test exclusion, including the Laptop exclusion '
              'in transfer for comparability. The source checkpoints stay fixed. Values are percentages '
              'with sample SD across three seeds. Gaps and gap changes are percentage points.', '',
              *table, '',
              'Ordering changes: ' + (', '.join(changes) if changes else 'none across these comparisons') + '.', '',
              f'The largest absolute change in a mean model gap was {max_change:.2f} percentage points. '
              'GCN retains its largest macro-F1 gains over BERT at 10% and 25%, keeps its smaller '
              'advantage over identity at 10%, and trails identity at full data in both domains. '
              'In transfer, GCN retains higher mean accuracy and lower mean macro-F1.', '',
              'The Restaurant duplicate occurs in the 100% training subset. The Laptop duplicate was '
              'in validation at every Laptop budget and could affect the validation macro-F1 ranking '
              'of epochs. Filtering Gold Test only quantifies scoring sensitivity for the fixed checkpoints. '
              'Counterfactual Laptop checkpoint rankings require validation predictions for each epoch '
              'with the duplicate removed, and scores for any newly selected checkpoint require its test '
              'predictions. A counterfactual without Restaurant training exposure requires retraining. '
              'Restaurant-to-Laptop transfer selected its checkpoints using Restaurant validation, '
              'so the Laptop validation duplicate could not influence that selection.', '', *analysis, '',
              'The candidate screen has unknown recall. The proxy is a small, manually verified '
              'sample of indirect sentiment, and several aspect rows can share a sentence. The two '
              'sarcastic sentences support illustrations and error counts only.']
    write_csv('saved_overlap_inventory.csv', matches)
    write_csv('saved_overlap_runs.csv', per_run)
    write_csv('saved_overlap_comparisons.csv', summaries)
    write_csv('saved_error_categories.csv', error_rows)
    (OUT / 'saved_evaluation_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Checked and scored all 66 canonical evaluations.')


if __name__ == '__main__':
    main()
