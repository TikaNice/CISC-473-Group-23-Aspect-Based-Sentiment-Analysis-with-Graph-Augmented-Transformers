"""Run one training job unless its result and required artifacts validate.

Usage:
  python -m scripts.run_one bert restaurant 1 42 none
  python -m scripts.run_one gcn  restaurant 0.1 43 identity

"""
import subprocess
import sys
from fcntl import LOCK_EX, flock

from src.evidence import INVALID_EVIDENCE, validate_result
from src.result_paths import GRAPH_TAGS, best_restaurant_checkpoint, result_path


def main():
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    model, domain, fraction, seed, graph = sys.argv[1:6]
    fraction_f = float(fraction)
    if graph in ('none', '', 'dependency'):
        graph = 'dependency'
    graph = graph if model == 'gcn' else 'none'
    tag = GRAPH_TAGS[graph]
    out = result_path(model, domain, fraction_f, int(seed), tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.lock').open('a') as lock:
        flock(lock, LOCK_EX)
        return train_if_needed(model, domain, fraction_f, seed, graph, tag, out)


def train_if_needed(model, domain, fraction_f, seed, graph, tag, out):
    if out.is_file():
        try:
            validate_result(model, domain, fraction_f, int(seed), graph)
            if domain == 'restaurant' and fraction_f == 1 and graph in ('none', 'dependency'):
                best_restaurant_checkpoint(model, int(seed), tag)
        except INVALID_EVIDENCE as exc:
            print(f'rerun incomplete {out}: {exc}', flush=True)
        else:
            print(f'skip verified {out}')
            return 0
    cmd = [sys.executable, '-m', f'src.train_{model}', '--domain', domain,
           '--train-fraction', str(fraction_f), '--seed', seed, '--epochs', '3']
    if model == 'gcn' and graph != 'dependency':
        cmd += ['--graph', graph]
    print('run', ' '.join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
