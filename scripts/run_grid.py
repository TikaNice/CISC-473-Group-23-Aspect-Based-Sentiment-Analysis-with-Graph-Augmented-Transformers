"""Run the BERT vs BERT+GCN experiment grid (model x train domain x label fraction x seed).

Every run also evaluates on both gold test sets, so the cross-domain results
(restaurant->laptop, laptop->restaurant) come for free with each in-domain run.

Runs are ordered by priority (all seeds at 100% data first, then 10%, 50%, 25%), so a
partially finished grid is still usable. Finished runs are skipped, so the script can be
stopped and restarted at any time.

Usage: python scripts/run_grid.py --results-dir results/main --head-lr 1e-3
       python scripts/run_grid.py --dry-run
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['bert', 'gcn'], choices=['bert', 'gcn'])
    p.add_argument('--domains', nargs='+', default=['restaurant', 'laptop'])
    p.add_argument('--fractions', nargs='+', type=float, default=[1.0, 0.1, 0.5, 0.25])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 999])
    p.add_argument('--head-lr', type=float, default=None)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--results-dir', default='results/main')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    runs = [(f, s, d, m) for f in args.fractions for s in args.seeds
            for d in args.domains for m in args.models]
    log_dir = Path(args.results_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, (fraction, seed, domain, model) in enumerate(runs, 1):
        name = f'{model}_{domain}_fraction{fraction:g}_seed{seed}'
        if (Path(args.results_dir) / f'{name}.json').is_file():
            print(f'[{i}/{len(runs)}] skip {name} (done)')
            continue
        cmd = [sys.executable, '-m', f'src.train_{model}', '--domain', domain,
               '--train-fraction', str(fraction), '--seed', str(seed),
               '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
               '--results-dir', args.results_dir]
        if args.head_lr is not None:
            cmd += ['--head-lr', str(args.head_lr)]
        print(f'[{i}/{len(runs)}] {name}: {" ".join(cmd)}', flush=True)
        if args.dry_run:
            continue
        start = time.time()
        with (log_dir / f'{name}.log').open('w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = 'ok' if code == 0 else f'FAILED (exit {code}, see {log_dir / name}.log)'
        print(f'    {status} in {(time.time() - start) / 60:.1f} min', flush=True)
        if code != 0:
            failures.append(name)
    if failures:
        print('Failed runs:', *failures, sep='\n  ')
        sys.exit(1)


if __name__ == '__main__':
    main()
