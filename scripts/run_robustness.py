"""Run the tagged supplementary grids, skipping only verified results."""
import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_EX, flock
from itertools import product
from pathlib import Path

from src.evidence import INVALID_EVIDENCE, validate_result
from src.result_paths import DOMAINS, GRAPH_TAGS, SEEDS, result_path

ARMS = {
    'subset2027': (2027, 3, (0.1, 0.25, 0.5)),
    'updatematch10': (2026, 30, (0.1,)),
}
MODELS = {'bert': 'none', 'gcn': 'dependency'}


def grid(arm):
    for fraction, seed, domain, model in product(ARMS[arm][2], SEEDS, DOMAINS, MODELS):
        yield arm, model, domain, fraction, seed


def tagged_result(arm, model, domain, fraction, seed):
    base = GRAPH_TAGS[MODELS[model]]
    tag = f'{base}_{arm}' if base else arm
    return tag, result_path(model, domain, fraction, seed, tag)


def validate_run(arm, model, domain, fraction, seed):
    tag, path = tagged_result(arm, model, domain, fraction, seed)
    subset_seed, epochs, _ = ARMS[arm]
    result = validate_result(model, domain, fraction, seed, MODELS[model], tag=tag)
    expected = dict(subset_seed=subset_seed, epochs=epochs, batch_size=8, split_seed=2026,
                    optimizer_updates=math.ceil(result['train_size'] / 8) * epochs)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path}: training configuration or update count does not match its arm')
    return result


def run(job):
    arm, model, domain, fraction, seed = job
    tag, path = tagged_result(*job)
    subset_seed, epochs, _ = ARMS[arm]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        flock(lock, LOCK_EX)
        if path.is_file():
            try:
                validate_run(*job)
            except INVALID_EVIDENCE as exc:
                print(f'Rerun incomplete {path}: {exc}', flush=True)
            else:
                print(f'Skip verified {path}', flush=True)
                return True

        command = [sys.executable, '-u', '-m', f'src.train_{model}',
                   '--domain', domain, '--train-fraction', str(fraction),
                   '--seed', str(seed), '--subset-seed', str(subset_seed),
                   '--epochs', str(epochs), '--batch-size', '8', '--run-tag', tag]
        log = Path('logs') / path.with_suffix('.log').name
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'Run {" ".join(command)} > {log} 2>&1', flush=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write(f'\n{" ".join(command)}\n')
            handle.flush()
            status = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
        if status:
            print(f'Failed {path} (exit {status}); see {log}', flush=True)
            return False
        try:
            validate_run(*job)
        except INVALID_EVIDENCE as exc:
            print(f'Invalid result {path}: {exc}; see {log}', flush=True)
            return False
        print(f'Verified {path}', flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='+', choices=ARMS)
    args = parser.parse_args()
    jobs = [job for arm in dict.fromkeys(args.arms) for job in grid(arm)]
    with ThreadPoolExecutor(int(os.environ.get('JOBS', '3'))) as pool:
        completed = sum(pool.map(run, jobs))
    print(f'Verified {completed}/{len(jobs)} supplementary runs', flush=True)
    return int(completed != len(jobs))


if __name__ == '__main__':
    raise SystemExit(main())
