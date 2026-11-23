"""Drain the training grid on CUDA with a few concurrent workers, then run the transfers."""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from src.evidence import validate_result
from src.result_paths import extension_grid, training_grid


def run(job):
    model, domain, fraction, seed, graph = job
    command = [sys.executable, '-m', 'scripts.run_one', model, domain,
               str(fraction), str(seed), graph]
    for attempt in range(3):
        if subprocess.call(command) == 0:
            result = validate_result(model, domain, fraction, seed, graph)
            print(f'Validated {model} {domain} {fraction:g} seed {seed} {graph}: '
                  f'{result["device"]}', flush=True)
            return True
        time.sleep(15)
    print(f'CUDA worker gave up after three attempts: {command[3:]}', flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", choices=("pooling", "constituency"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA worker requires an available CUDA device')

    # Laptop first, largest runs first; run_one locks each run and skips finished ones.
    grid = (job for job in extension_grid() if job[-1] == args.extension) if args.extension else training_grid()
    jobs = sorted(grid, key=lambda job: (job[1] == 'restaurant', -job[2]))
    with ThreadPoolExecutor(int(os.environ.get('CUDA_WORKERS', '3'))) as pool:
        failed = [job for job, ok in zip(jobs, pool.map(run, jobs)) if not ok]
    if failed:
        raise SystemExit(f'CUDA worker stopped with {len(failed)} failed runs: {failed}')
    if args.extension:
        return
    subprocess.check_call([sys.executable, '-m', 'scripts.eval_cross_domain',
                           '--model', 'all', '--seeds', '42', '43', '44'])
    print('Grid complete.', flush=True)


if __name__ == '__main__':
    main()
