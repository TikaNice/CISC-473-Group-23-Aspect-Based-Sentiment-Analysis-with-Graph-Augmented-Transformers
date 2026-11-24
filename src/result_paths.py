"""Canonical result locations shared by training and reporting."""
from pathlib import Path


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
