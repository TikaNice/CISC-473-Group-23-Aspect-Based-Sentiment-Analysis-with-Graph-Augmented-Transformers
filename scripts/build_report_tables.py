"""Emit the report's result tables as LaTeX from saved CSV summaries.

Writes report/tables/*.tex, so the report always matches results/reports/*.
Run after scripts/summarize_results.py and scripts/summarize_ablations.py.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

from src.evidence import load_predictions
from src.result_paths import GRAPH_TAGS, SEEDS, result_path, training_grid, transfer_path

OUT = Path('../report/tables')
LABEL = {'restaurant': 'Restaurant', 'laptop': 'Laptop'}


def read_rows(path):
    if not Path(path).is_file():
        print(f'warning: {path} missing; treating as empty')
        return []
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pm(mean, std):
    return f'{100*float(mean):.2f} $\\pm$ {100*float(std):.2f}'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding='utf-8')


def main_results():
    base = read_rows('results/reports/baseline_summary.csv')
    def cell(domain, model, key):
        rec = [r for r in base if r['domain'] == domain and r['model'] == model and float(r['fraction']) == 1]
        if not rec:
            return '--'
        return pm(rec[0][f'{key}_mean'], rec[0][f'{key}_std'])
    body = r'''\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Restaurant}} & \multicolumn{2}{c}{\textbf{Laptop}} \\
\textbf{Model} & \textbf{Accuracy} $\uparrow$ & \textbf{Macro-F1} $\uparrow$ & \textbf{Accuracy} $\uparrow$ & \textbf{Macro-F1} $\uparrow$ \\
\midrule
'''
    body += r'ASGCN-DG (Zhang et al., 2019, author protocol) & 80.77 & 72.02 & 75.55 & 71.05 \\' + '\n'
    body += r'Vanilla BERT (Xu et al., 2019, author protocol) & 81.54 & 71.94 & 75.29 & 71.91 \\' + '\n'
    body += r'\midrule' + '\n'
    body += f"BERT (this work) & {cell('restaurant','bert','test_accuracy')} & {cell('restaurant','bert','test_macro_f1')} & {cell('laptop','bert','test_accuracy')} & {cell('laptop','bert','test_macro_f1')} \\\\" + '\n'
    body += f"BERT+GCN (this work) & {cell('restaurant','gcn','test_accuracy')} & {cell('restaurant','gcn','test_macro_f1')} & {cell('laptop','gcn','test_accuracy')} & {cell('laptop','gcn','test_macro_f1')} \\\\" + '\n'
    pool = {(r['domain'], r['metric']): pm(r['mean'], r['sd'])
            for r in read_rows('results/reports/extension_summary.csv')
            if r['arm'] == 'pooling' and float(r['fraction']) == 1}
    if len(pool) == 4:
        body += ('BERT aspect pooling, no graph (this work) & ' + ' & '.join(
            pool[domain, metric] for domain in ('restaurant', 'laptop')
            for metric in ('accuracy', 'macro_f1')) + ' \\\\\n')
    body += r'''\bottomrule
\end{tabular}'''
    write('main_results.tex', body)


def data_budget():
    rows = read_rows('results/reports/low_resource_summary.csv')
    body = r'''\begin{tabular}{llccc}
\toprule
\textbf{Domain} & \textbf{Budget} & \textbf{BERT} & \textbf{BERT+GCN} & $\Delta$ (GCN $-$ BERT) \\
\midrule
'''
    for domain in ('restaurant', 'laptop'):
        for frac in (0.1, 0.25, 0.5, 1.0):
            recs = {r['model']: r for r in rows if r['domain'] == domain and float(r['fraction']) == frac}
            if len(recs) < 2:
                continue
            b = 100 * float(recs['bert']['test_f1_mean'])
            g = 100 * float(recs['gcn']['test_f1_mean'])
            bs = 100 * float(recs['bert']['test_f1_std'])
            gs = 100 * float(recs['gcn']['test_f1_std'])
            label = f'{int(round(frac*100))}\\%'
            dom = LABEL[domain] if frac == 0.1 else ''
            # Δ from the printed cells, so the column reconciles as it reads
            delta = round(g, 2) - round(b, 2)
            body += f'{dom} & {label} & {b:.2f} $\\pm$ {bs:.2f} & {g:.2f} $\\pm$ {gs:.2f} & ${delta:+.2f}$ \\\\\n'
        body += r'\midrule' + '\n'
    body = body.removesuffix('\\midrule\n')   # no dangling divider before \\bottomrule
    body += r'''\bottomrule
\end{tabular}'''
    write('data_budget.tex', body)


def transfer():
    rows = read_rows('results/reports/cross_domain_summary.csv')
    body = r'''\begin{tabular}{lcc}
\toprule
\textbf{Restaurant-trained model} & \textbf{Laptop accuracy} $\uparrow$ & \textbf{Laptop macro-F1} $\uparrow$ \\
\midrule
'''
    for model, name in (('bert', 'BERT'), ('gcn', 'BERT+GCN')):
        rec = [r for r in rows if r['model'] == model]
        if not rec:
            continue
        body += f"{name} & {pm(rec[0]['test_accuracy_mean'], rec[0]['test_accuracy_std'])} & {pm(rec[0]['test_macro_f1_mean'], rec[0]['test_macro_f1_std'])} \\\\" + '\n'
    body += r'''\bottomrule
\end{tabular}'''
    write('transfer.tex', body)


def ablation():
    rows = read_rows('results/reports/ablation_summary.csv')
    if not rows:
        write('ablation.tex', '\\textit{Graph-control runs still in progress.}')
        return
    body = r'''\begin{tabular}{llccc}
\toprule
\textbf{Domain} & \textbf{Budget} & \textbf{Dependency edges} & \textbf{Identity (self-loops)} & \textbf{Paired $\Delta$} \\
\midrule
'''
    for domain in ('restaurant', 'laptop'):
        for frac in (0.1, 1.0):
            recs = [r for r in rows if r['domain'] == domain and float(r['fraction']) == frac]
            dep = [r for r in recs if r['graph'] == 'dependency']
            ident = [r for r in recs if r['graph'] == 'identity']
            if not dep or not ident:
                continue
            d, i = dep[0], ident[0]
            label = f'{int(round(frac*100))}\\%'
            dom = LABEL[domain] if frac == 0.1 else ''
            delta = d['paired_dependency_minus_identity_mean']
            body += (f"{dom} & {label} & {pm(d['test_macro_f1_mean'], d['test_macro_f1_std'])} & "
                     f"{pm(i['test_macro_f1_mean'], i['test_macro_f1_std'])} & ${100*float(delta):+.2f}$ \\\\\n")
        body += r'\midrule' + '\n'
    body = body.removesuffix('\\midrule\n')   # no dangling divider before \\bottomrule
    body += r'''\bottomrule
\end{tabular}'''
    write('ablation.tex', body)


# ---------------------------------------------------------------- numbers
NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot)\b|n't\b", re.I)


def load_saved_runs():
    runs = {}
    for model, domain, fraction, seed, graph in training_grid():
        path = result_path(model, domain, fraction, seed, GRAPH_TAGS[graph])
        if path.is_file():
            obj = json.loads(path.read_text(encoding='utf-8'))
            runs[(domain, fraction, model, seed, graph)] = obj['final_test_metrics']
    return runs


def tex_escape(s):
    replacements = dict(zip(r'\&%$#_{}~^', (
        r'\textbackslash{}', r'\&', r'\%', r'\$', r'\#', r'\_', r'\{', r'\}',
        r'\textasciitilde{}', r'\textasciicircum{}',
    )))
    tokens = re.split(r'(\s+)', s)
    return ''.join(r'\allowbreak{}'.join(
        ''.join(replacements.get(char, char) for char in token[start:start + 11])
        for start in range(0, len(token), 11)
    ) for token in tokens)


def neutral_recall(model, domain, eval_domain, fraction=1.0):
    preds = load_predictions(model, domain, eval_domain, fraction)
    if set(preds) != set(SEEDS):
        return None
    correct = total = 0
    for seed, rows in preds.items():
        for r in rows:
            if r['gold'] == 'neutral':
                total += 1
                correct += (r['pred'] == 'neutral')
    return correct / total if total else None


def numbers():
    runs = load_saved_runs()
    def get(domain, frac, model, seed, graph=None):
        graph = graph or ('dependency' if model == 'gcn' else 'none')
        return runs.get((domain, frac, model, seed, graph))

    def stat(domain, frac, model, graph=None):
        vals = [get(domain, frac, model, s, graph) for s in (42, 43, 44)]
        vals = [v for v in vals if v]
        if len(vals) != len(SEEDS):
            return None
        f1 = np.array([v['test_macro_f1'] for v in vals])
        acc = np.array([v['test_accuracy'] for v in vals])
        sd = lambda a: float(a.std(ddof=1))
        return {'acc': float(acc.mean()), 'acc_sd': sd(acc), 'f1': float(f1.mean()), 'f1_sd': sd(f1), 'n': len(vals)}

    defaults = {}
    for tag in ('BertRest', 'GcnRest', 'BertLap', 'GcnLap'):
        defaults[f'{tag}Acc'] = defaults[f'{tag}AccSD'] = defaults[f'{tag}F'] = defaults[f'{tag}FSD'] = '--'
    for tag in ('TrBert', 'TrGcn'):
        for suffix in ('Acc', 'AccSD', 'F', 'FSD'):
            defaults[f'{tag}{suffix}'] = '--'
        defaults[f'{tag}NeutralRecall'] = '--'
    for suffix in ('Full', 'Ten'):
        for dom in ('Rest', 'Lap'):
            defaults[f'AblDelta{dom}{suffix}'] = '--'
            defaults[f'AblDelta{dom}{suffix}SD'] = '--'
    for name in ('BudgetDeltaMin', 'BudgetDeltaMax', 'BudgetWins', 'BudgetPairs',
                 'BudgetPositiveCells', 'TransferFOneWins', 'TransferAccWins',
                 'FullDeltaRest', 'FullDeltaRestSD', 'FullDeltaLap', 'FullDeltaLapSD'):
        defaults[name] = '--'
    for dom in ('Rest', 'Lap'):
        for suffix in ('Full', 'Ten'):
            defaults[f'AblWins{dom}{suffix}'] = '--'
    values = dict(defaults)

    def add(name, value):
        values[name] = value

    def fmt(v):
        return f'{100*v:.2f}'

    for domain, tag in (('restaurant', 'Rest'), ('laptop', 'Lap')):
        for model, tag2 in (('bert', 'Bert'), ('gcn', 'Gcn')):
            s = stat(domain, 1.0, model)
            if s:
                add(f'{tag2}{tag}Acc', fmt(s['acc']))
                add(f'{tag2}{tag}AccSD', fmt(s['acc_sd']))
                add(f'{tag2}{tag}F', fmt(s['f1']))
                add(f'{tag2}{tag}FSD', fmt(s['f1_sd']))

    # transfer
    tr = read_rows('results/reports/cross_domain_summary.csv')
    for model, tag in (('bert', 'TrBert'), ('gcn', 'TrGcn')):
        rec = [r for r in tr if r['model'] == model]
        if rec and int(rec[0]["n_seeds"]) == len(SEEDS):
            add(f'{tag}Acc', fmt(float(rec[0]['test_accuracy_mean'])))
            add(f'{tag}AccSD', fmt(float(rec[0]['test_accuracy_std'])))
            add(f'{tag}F', fmt(float(rec[0]['test_macro_f1_mean'])))
            add(f'{tag}FSD', fmt(float(rec[0]['test_macro_f1_std'])))

    for metric, name in (('test_macro_f1', 'TransferFOneWins'), ('test_accuracy', 'TransferAccWins')):
        paths = [(transfer_path('bert', seed), transfer_path('gcn', seed, 'aspectfix'))
                 for seed in SEEDS]
        if all(bert.is_file() and gcn.is_file() for bert, gcn in paths):
            rows = [(json.loads(bert.read_text()), json.loads(gcn.read_text())) for bert, gcn in paths]
            add(name, str(sum(gcn[metric] > bert[metric] for bert, gcn in rows)))

    # budget deltas (reduced budgets only; full data is reported separately).
    # The prose quotes the range of per-setting MEAN gains, so aggregate within
    # each domain/budget cell first and take the extrema of those six means; the
    # win count stays a count of individual paired runs.
    cell_means, wins, pairs = [], 0, 0
    for domain in ('restaurant', 'laptop'):
        for frac in (0.1, 0.25, 0.5):
            diffs = []
            for seed in (42, 43, 44):
                b, g = get(domain, frac, 'bert', seed), get(domain, frac, 'gcn', seed)
                if not b or not g:
                    continue
                d = g['test_macro_f1'] - b['test_macro_f1']
                diffs.append(d)
                pairs += 1
                wins += int(d > 0)
            if len(diffs) == len(SEEDS):
                cell_means.append(float(np.mean(diffs)))
    if len(cell_means) == 6:
        add('BudgetPositiveCells', str(sum(mean > 0 for mean in cell_means)))
        add('BudgetDeltaMin', f'{100*min(cell_means):+.2f}')
        add('BudgetDeltaMax', f'{100*max(cell_means):+.2f}')
        add('BudgetWins', str(wins))
        add('BudgetPairs', str(pairs))

    # full-data paired deltas per domain
    for domain, tag in (('restaurant', 'Rest'), ('laptop', 'Lap')):
        ds = []
        for seed in (42, 43, 44):
            b, g = get(domain, 1.0, 'bert', seed), get(domain, 1.0, 'gcn', seed)
            if b and g:
                ds.append(g['test_macro_f1'] - b['test_macro_f1'])
        if len(ds) == len(SEEDS):
            add(f'FullDelta{tag}', f'{100*np.mean(ds):+.2f}')
            add(f'FullDelta{tag}SD', f'{100*np.std(ds, ddof=1):.2f}')

    # ablation deltas
    rows = read_rows('results/reports/ablation_summary.csv')
    for domain, tag in (('restaurant', 'Rest'), ('laptop', 'Lap')):
        for frac, tag2 in ((1.0, 'Full'), (0.1, 'Ten')):
            rec = [r for r in rows if r['domain'] == domain and float(r['fraction']) == frac
                   and r['graph'] == 'dependency']
            if rec and rec[0]['paired_sd']:
                add(f'AblDelta{tag}{tag2}', f"{100*float(rec[0]['paired_dependency_minus_identity_mean']):+.2f}")
                add(f'AblDelta{tag}{tag2}SD', f"{100*float(rec[0]['paired_sd']):.2f}")
                add(f'AblWins{tag}{tag2}', rec[0]['paired_note'].split('/')[0])

    # neutral recall (transfer direction is the one the report and deck quote)
    for model, tag in (('bert', 'TrBert'), ('gcn', 'TrGcn')):
        r = neutral_recall(model, 'restaurant', 'laptop', 1.0)
        if r is not None:
            add(f'{tag}NeutralRecall', f'{100*r:.2f}')

    # extension arms: paired macro-F1 deltas against CLS BERT and the dependency GCN
    for r in read_rows('results/reports/extension_summary.csv'):
        tag = ({'pooling': 'Pool', 'constituency': 'Con'}[r['arm']] + {'restaurant': 'Rest', 'laptop': 'Lap'}[r['domain']]
               + ('Full' if float(r['fraction']) == 1 else 'Ten'))
        if r['metric'] == 'macro_f1':
            for ref, name in (('bert', 'VsBert'), ('dependency', 'VsGcn')):
                add(f'{tag}{name}', f"{100*float(r[f'delta_vs_{ref}_mean']):+.2f}")
                add(f'{tag}{name}SD', f"{100*float(r[f'delta_vs_{ref}_sd']):.2f}")
        if r['arm'] == 'pooling' and float(r['fraction']) == 1:
            add(tag.removesuffix('Full') + ('F' if r['metric'] == 'macro_f1' else 'Acc'), fmt(float(r['mean'])))

    # overlap-excluded sensitivity and verified error categories (scripts/audit_saved_evaluation.py)
    gaps = read_rows('results/reports/saved_overlap_comparisons.csv')
    if gaps:
        add('OverlapComparisons', str(len(gaps)))
        add('OverlapSignChanges', str(sum((float(r['gap_all']) > 0) != (float(r['gap_filtered']) > 0) for r in gaps)))
        add('OverlapMaxShift', f"{100*max(abs(float(r['gap_change'])) for r in gaps):.2f}")
    cats = read_rows('results/reports/saved_error_categories.csv')
    for category, tag in (('indirect_sentiment_proxy', 'Proxy'), ('sarcasm', 'Sarc')):
        for model, tag2 in (('bert', 'Bert'), ('gcn', 'Gcn')):
            rows_ = [r for r in cats if r['category'] == category and r['model'] == model]
            if rows_:
                add(f'{tag}Rows', str(sum(int(r['verified']) for r in rows_ if int(r['seed']) == SEEDS[0])))
                add(f'{tag}Preds', str(sum(int(r['verified']) for r in rows_)))
                add(f'{tag}{tag2}Err', str(sum(int(r['errors']) for r in rows_)))

    # Consistency gate. The tables in the report are built from results/reports/*.csv
    # while the macros above are built from the run JSONs; if those two views ever
    # disagree the deck and the report would quote different numbers for the same run.
    base = read_rows('results/reports/baseline_summary.csv')
    for domain, tag in (('restaurant', 'Rest'), ('laptop', 'Lap')):
        for model, tag2 in (('bert', 'Bert'), ('gcn', 'Gcn')):
            rec = [r for r in base
                   if r['domain'] == domain and r['model'] == model and float(r['fraction']) == 1]
            key = f'{tag2}{tag}F'
            if not rec or values.get(key, '--') == '--':
                continue
            csv_value = f"{100*float(rec[0]['test_macro_f1_mean']):.2f}"
            if values[key] != csv_value:
                raise ValueError(f'{key} macro={values[key]} but summary CSV={csv_value}')
    unfilled = sorted(k for k, v in values.items() if v == '--')
    if unfilled:
        print(f'WARNING: {len(unfilled)} macro(s) left unfilled: ' + ', '.join(unfilled))

    body = ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in values.items())
    write('numbers.tex', body)
    Path('../../mtp').mkdir(parents=True, exist_ok=True)
    Path('../../mtp/numbers.tex').write_text(body, encoding='utf-8')


def budget_slide():
    rows = read_rows('results/reports/low_resource_summary.csv')
    head = ('\\begin{tabular}{llcc}\n\\toprule\n'
            '\\textbf{Domain} & \\textbf{Budget} & \\textbf{BERT} & \\textbf{BERT+GCN} \\\\\n\\midrule\n')
    groups = []
    for domain in ('restaurant', 'laptop'):
        block = ''
        for frac in (0.1, 0.25, 0.5, 1.0):
            recs = {r['model']: r for r in rows if r['domain'] == domain and float(r['fraction']) == frac}
            if len(recs) < 2:
                continue
            b = 100 * float(recs['bert']['test_f1_mean'])
            g = 100 * float(recs['gcn']['test_f1_mean'])
            dom = LABEL[domain] if not block else ''
            block += f"{dom} & {int(round(frac*100))}\\% & {b:.2f} & {g:.2f} \\\\\n"
        if block:
            groups.append(block)
    body = '\\midrule\n'.join(groups) if groups else '\\multicolumn{4}{c}{\\textit{budget runs pending}} \\\\\n'
    Path('../../mtp/budget_slide.tex').write_text(head + body + '\\bottomrule\n\\end{tabular}\n', encoding='utf-8')



def failures():
    examples = failure_examples()
    for path, widths in ((OUT / 'failures.tex', 'rp{2.0cm}p{6.4cm}p{2.0cm}ll'),
                         (Path('../../mtp/failures_slide.tex'), 'rp{1.8cm}p{3.55cm}p{1.8cm}ll')):
        head = (f'\\begin{{tabular}}{{@{{}}{widths}@{{}}}}\n\\toprule\n'
                r'\textbf{Index} & \textbf{Pattern} & \textbf{Sentence excerpt} & '
                '\\textbf{Aspect} & \\textbf{Gold} & \\textbf{Both} \\\\\n\\midrule\n')
        body = ''
        for kind, row, pred in examples:
            text = row['text']
            if len(text) > 75:
                start = max(0, (row.get('start') or 0) - 22)
                text = ('...' if start else '') + text[start:start + 68].rstrip()
                if start + 68 < len(row['text']):
                    text += '...'
            fields = [str(row['index']), kind, text, row['span'], row['gold'], pred]
            body += ' & '.join(tex_escape(field) for field in fields) + ' \\\\\n'
        if not body:
            body = '\\multicolumn{6}{c}{\\textit{examples pending}} \\\\\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(head + body + '\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


def failure_examples():
    bert = load_predictions('bert', 'restaurant', 'restaurant', 1.0)
    gcn = load_predictions('gcn', 'restaurant', 'restaurant', 1.0)
    if set(bert) != set(SEEDS) or set(gcn) != set(SEEDS):
        return []
    rows = bert[SEEDS[0]]
    sentence_labels = {}
    for row in rows:
        sentence_labels.setdefault(row['text'], set()).add(row['gold'])
    stable = []
    for index, row in enumerate(rows):
        predictions = {runs[seed][index]['pred'] for runs in (bert, gcn) for seed in SEEDS}
        if len(predictions) == 1 and row['gold'] not in predictions:
            stable.append((row, predictions.pop()))

    chosen = []
    for kind, matches in (
        ('Negation cue', lambda row, pred: NEGATION.search(row['text'])),
        ('Neutral to positive', lambda row, pred: row['gold'] == 'neutral' and pred == 'positive'),
        ('Mixed aspects', lambda row, pred: pred in sentence_labels[row['text']]),
        ('Other', lambda row, pred: True),
    ):
        for row, pred in stable:
            if matches(row, pred) and all(row['index'] != item[1]['index'] for item in chosen):
                chosen.append((kind, row, pred))
                if len(chosen) == 3:
                    return chosen
                if kind != 'Other':
                    break
    return chosen


def controls():
    """Macro-F1 for every arm at the two control budgets: report table (with SD) and deck table."""
    arms = (('bert', 'none', 'BERT (CLS)'), ('gcn', 'pooling', 'Pooling, no graph'),
            ('gcn', 'identity', 'Identity'), ('gcn', 'dependency', 'Dependency'),
            ('gcn', 'constituency', 'Constituency'))

    def cell(domain, frac, model, graph, sd):
        f1 = [100 * json.loads(result_path(model, domain, frac, seed, GRAPH_TAGS[graph]).read_text(encoding='utf-8'))
              ['final_test_metrics']['test_macro_f1'] for seed in SEEDS]
        return f'{np.mean(f1):.2f}' + (f' $\\pm$ {np.std(f1, ddof=1):.2f}' if sd else '')

    for path, sd in ((OUT / 'controls.tex', True), (Path('../../mtp/ablation_slide.tex'), False)):
        body = ('\\begin{tabular}{ll' + 'c' * len(arms) + '}\n\\toprule\n'
                '& & & & \\multicolumn{3}{c}{\\textbf{GCN adjacency}} \\\\\n\\cmidrule(l){5-7}\n'
                '\\textbf{Domain} & \\textbf{Budget} & '
                + ' & '.join(f'\\textbf{{{name}}}' for *_, name in arms) + ' \\\\\n\\midrule\n')
        for domain in ('restaurant', 'laptop'):
            for frac in (0.1, 1.0):
                body += (f"{LABEL[domain] if frac == 0.1 else ''} & {frac:.0%}".replace('%', '\\%') + ' & '
                         + ' & '.join(cell(domain, frac, model, graph, sd) for model, graph, _ in arms) + ' \\\\\n')
            body += '\\midrule\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.removesuffix('\\midrule\n') + '\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


if __name__ == '__main__':
    main_results()
    data_budget()
    transfer()
    ablation()
    numbers()
    budget_slide()
    failures()
    controls()
    print('Wrote tables and macros.')
