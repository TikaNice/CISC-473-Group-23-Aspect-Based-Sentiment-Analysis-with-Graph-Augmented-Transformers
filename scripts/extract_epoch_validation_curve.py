"""Extract Validation accuracy/F1 after each epoch from saved Trainer state.
No training. Best-effort: only epochs represented in retained checkpoint state can be read.
Usage: python -m scripts.extract_epoch_validation_curve
"""
from src.result_paths import checkpoint_dir

import csv
import json
import tarfile
from pathlib import Path

rows = []
for domain in ('restaurant', 'laptop'):
    for model in ('bert', 'gcn'):
        for seed in (42, 43, 44):
            tag = 'aspectfix_ep4' if model == 'gcn' else 'ep4'
            base = checkpoint_dir(model, domain, 1, seed, tag)
            states = sorted(base.glob('checkpoint-*/trainer_state.json'),
                            key=lambda p: int(p.parent.name.split('-')[-1]))
            if states:
                state = json.loads(states[-1].read_text(encoding='utf-8'))
            else:
                archive = Path('checkpoints/training_metadata.tar.gz')
                if not archive.is_file():
                    continue
                with tarfile.open(archive, 'r:gz') as saved:
                    candidates = [m for m in saved.getmembers()
                                  if m.isfile() and Path(m.name).parent.parent == base
                                  and Path(m.name).name == 'trainer_state.json']
                    if not candidates:
                        continue
                    latest = max(candidates, key=lambda m: int(Path(m.name).parent.name.split('-')[-1]))
                    state = json.load(saved.extractfile(latest))
            found = {}
            for item in state.get('log_history', []):
                ep = item.get('epoch')
                if ep is None or 'eval_accuracy' not in item:
                    continue
                ep_rounded = int(round(float(ep)))
                if abs(ep_rounded - float(ep)) <= 0.01:
                    found[ep_rounded] = (item['eval_accuracy'], item.get('eval_macro_f1'))
            for epoch, (acc, f1) in sorted(found.items()):
                rows.append({'domain': domain, 'model': model, 'seed': seed,
                             'epoch': epoch, 'validation_accuracy': acc,
                             'validation_macro_f1': f1})
                print(f'{domain:10} {model:4} seed={seed} epoch={epoch} '
                      f'val-accuracy={acc:.4f} val-macro-f1={f1:.4f}')
if rows:
    target = Path('results/reports/epoch_validation_learning_curve.csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', target)
    print('Valid history only: legacy GCN states are excluded. Corrected GCN epoch-4 history remains pending until those runs are trained.')
else:
    print('No epoch-4 Trainer state found yet. Run the epoch sweep first.')
