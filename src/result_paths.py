"""Canonical result locations shared by training and reporting."""
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

DOMAINS = ('restaurant', 'laptop')
SEEDS = (42, 43, 44)
BUDGETS = (0.1, 0.25, 0.5, 1.0)
GRAPH_TAGS = {'none': '', 'dependency': 'aspectfix', 'identity': 'identity', 'full': 'fullgraph',
              'pooling': 'pooling', 'constituency': 'constituency'}


def result_path(model, domain, fraction=1, seed=42, tag='', smoke=False):
    fraction = float(fraction)
    suffix = f'{domain}_fraction{fraction:g}_seed{seed}'
    if smoke:
        suffix += '_smoke'
    if tag:
        suffix += '_' + tag
    base = Path('results')
    if smoke:
        folder = base / 'smoke' / domain / model
    elif tag.startswith('identity') or tag.startswith('fullgraph'):
        folder = base / 'ablation' / domain / model
    elif tag.startswith('ep') and tag[2:].isdigit() and fraction == 1:
        folder = base / 'epoch_sweep' / domain / model / f'epochs_{int(tag[2:])}'
    elif tag or fraction not in (0.1, 0.25, 0.5, 1):
        folder = base / 'custom' / domain / model
    elif fraction == 1:
        folder = base / 'baseline' / domain / model
    else:
        folder = base / 'low_resource' / domain / model / f'{fraction * 100:g}_percent'
    return folder / f'{model}_{suffix}.json'


def checkpoint_dir(model, domain, fraction=1, seed=42, tag='', smoke=False):
    """Store each training run alongside the same classification as its result."""
    relative = result_path(model, domain, fraction, seed, tag, smoke).relative_to('results')
    return Path('checkpoints') / relative.with_suffix('')


def transfer_path(model, seed, tag=''):
    suffix = f'_{tag}' if tag else ''
    return (Path('results/cross_domain/restaurant_to_laptop') / model
            / f'{model}_restaurant_to_laptop_seed{seed}{suffix}.json')


def training_grid():
    for domain in DOMAINS:
        for fraction in BUDGETS:
            arms = [('bert', 'none'), ('gcn', 'dependency')]
            if fraction in (0.1, 1.0):
                arms.append(('gcn', 'identity'))
            for model, graph in arms:
                for seed in SEEDS:
                    yield model, domain, fraction, seed, graph


def extension_grid():
    for domain in DOMAINS:
        for fraction in (0.1, 1.0):
            for graph in ("pooling", "constituency"):
                for seed in SEEDS:
                    yield "gcn", domain, fraction, seed, graph


def best_restaurant_checkpoint(model, seed, tag=''):
    root = checkpoint_dir(model, 'restaurant', 1, seed, tag)
    folders = [folder for folder in root.glob('checkpoint-*')
               if folder.is_dir() and folder.name.rsplit('-', 1)[-1].isdigit()]
    if not folders:
        raise FileNotFoundError(f'No trained checkpoints in {root}')
    latest = max(folders, key=lambda folder: int(folder.name.rsplit('-', 1)[-1]))
    state_file = root / 'trainer_state.json'
    if not state_file.is_file():
        state_file = latest / 'trainer_state.json'
    info = json.loads(state_file.read_text(encoding='utf-8'))
    best = info.get('best_model_checkpoint')
    if not best:
        raise ValueError(f'No best validation checkpoint recorded in {state_file}')
    candidate = root / Path(best).name
    if not any((candidate / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        raise FileNotFoundError(f'Missing weights for the recorded best checkpoint in {candidate}')
    return candidate


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=float)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('domain')
    parser.add_argument('fraction', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    print(result_path(args.model, args.domain, args.fraction, args.seed, args.tag))
