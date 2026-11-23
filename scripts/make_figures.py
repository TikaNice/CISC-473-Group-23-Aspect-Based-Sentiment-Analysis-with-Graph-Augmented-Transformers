"""Build the report and slide figures from saved results.

Reads results/reports/*.csv and results/predictions/*.jsonl and writes:
  report/figures/data_efficiency.png
  report/figures/graph_control.png
  report/figures/confusion_transfer.png
and copies them into mtp/figures/.
"""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.evidence import load_predictions
from src.result_paths import SEEDS

NAVY, TEAL, ORANGE, INK, MUTED, GRID = '#122840', '#006B66', '#eb6834', '#0b0b0b', '#6f6d69', '#dcdad5'
LABELS = ['negative', 'neutral', 'positive']
OUT = Path('../report/figures')
MTP = Path('../../mtp/figures')
FRACTIONS = [0.1, 0.25, 0.5, 1.0]

plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif', 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def read_rows(path):
    if not Path(path).is_file():
        print(f'warning: {path} missing')
        return []
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def data_efficiency():
    rows = read_rows('results/reports/low_resource_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, domain in zip(axes, ('restaurant', 'laptop')):
        for model, colour, marker, name in (('bert', NAVY, 'o', 'BERT'),
                                            ('gcn', ORANGE, 's', 'BERT+GCN')):
            recs = [r for r in rows if r['domain'] == domain and r['model'] == model]
            recs.sort(key=lambda r: float(r['fraction']))
            x = [float(r['fraction']) for r in recs]
            m = [100 * float(r['test_f1_mean']) for r in recs]
            s = [100 * float(r['test_f1_std'] or 'nan') for r in recs]
            ax.plot(x, m, marker=marker, color=colour, label=name, linewidth=1.8)
            ax.fill_between(x, np.array(m) - np.array(s), np.array(m) + np.array(s), color=colour, alpha=0.16, lw=0)
        ax.set_xscale('log')
        ax.set_xticks(FRACTIONS)
        ax.set_xticklabels(['10%', '25%', '50%', '100%'])
        ax.set_xlabel('Training budget (share of sentences)')
        ax.set_title(domain.capitalize(), color=NAVY, fontsize=11)
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Test macro-F1 (%)')
    axes[0].legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'data_efficiency.png', dpi=220)
    plt.close(fig)


def graph_control():
    rows = read_rows('results/reports/ablation_summary.csv')
    if not rows:
        print('skip graph_control.png: no ablation runs yet')
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, domain in zip(axes, ('restaurant', 'laptop')):
        for i, frac in enumerate((0.1, 1.0)):
            recs = {r['graph']: r for r in rows if r['domain'] == domain and float(r['fraction']) == frac}
            for j, (graph, colour, name) in enumerate((('dependency', ORANGE, 'dependency edges'),
                                                       ('identity', NAVY, 'identity (self-loop only)'))):
                rec = recs.get(graph)
                if rec is None:
                    continue
                m = 100 * float(rec['test_macro_f1_mean'])
                s = 100 * float(rec['test_macro_f1_std'] or 'nan')
                x = i + (0.18 if j == 0 else -0.18)
                ax.bar(x, m, width=0.34, color=colour, label=name if i == 0 else None)
                ax.errorbar(x, m, yerr=s, color=INK, capsize=3, linewidth=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['10% data', '100% data'])
        ax.set_title(domain.capitalize(), color=NAVY, fontsize=11)
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Test macro-F1 (%)')
    axes[0].set_ylim(0, 90)
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'graph_control.png', dpi=220)
    plt.close(fig)


def confusion(domain, eval_domain, filename, title):
    preds = {}
    for model in ('bert', 'gcn'):
        predictions = load_predictions(model, domain, eval_domain, 1.0)
        if set(predictions) != set(SEEDS):
            continue
        matrix = np.zeros((3, 3), dtype=int)
        for rows in predictions.values():
            for row in rows:
                matrix[LABELS.index(row['gold']), LABELS.index(row['pred'])] += 1
        preds[model] = matrix
    if len(preds) < 2:
        print(f'skip {filename}: predictions missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, model in zip(axes, ('bert', 'gcn')):
        mat = preds[model]
        row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
        share = mat / row_sums
        ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{100*share[i,j]:.0f}%\n({mat[i,j]})', ha='center', va='center',
                        color='white' if share[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS)
        ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('BERT' if model == 'bert' else 'BERT+GCN', color=NAVY, fontsize=11)
        ax.tick_params(colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ('data_efficiency.png', 'graph_control.png', 'confusion_restaurant.png',
                  'confusion_laptop.png', 'confusion_transfer.png', 'less_data.png',
                  'confusion_cross_domain.png'):
        (OUT / stale).unlink(missing_ok=True)
        (MTP / stale).unlink(missing_ok=True)
    data_efficiency()
    graph_control()
    confusion('restaurant', 'laptop', 'confusion_transfer.png', 'Restaurant-trained models on Laptop Gold Test, pooled over three seeds')
    MTP.mkdir(parents=True, exist_ok=True)
    for name in ('data_efficiency.png', 'confusion_transfer.png'):
        if (OUT / name).is_file():
            shutil.copy2(OUT / name, MTP / name)
    print('Figures:', ', '.join(sorted(p.name for p in OUT.glob('*.png'))))


if __name__ == '__main__':
    main()
