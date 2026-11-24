#!/usr/bin/env bash
# Full experiment grid, ordered so the headline evidence lands first.
#
#   phase A  full data, both domains, both models, three seeds   (main table, transfer,
#            checkpoints, per-example predictions for the failure analysis)
#   phase B  identity-graph control at 100% and 10%, three seeds (the ablation)
#   phase C  reduced budgets 50/25/10% for both models           (budget table, curve)
#
# Resumable: completed training and transfer evidence is checked before skipping.
# Pass --force to delete existing records and checkpoints first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=${PY:-python3}; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export ACCELERATE_USE_CPU=${ACCELERATE_USE_CPU:-true}
JOBS=${JOBS:-5}

if [[ "${1:-}" == "--force" ]]; then
  "$PY" - <<'PY'
import glob, os, shutil
for pattern in ('results/baseline/*/*/*.json', 'results/low_resource/*/*/*/*.json',
                'results/custom/*/*/*.json', 'results/ablation/*/*/*.json',
                'results/cross_domain/*/*/*.json', 'results/predictions/*.jsonl'):
    for path in glob.glob(pattern):
        os.remove(path)
shutil.rmtree('checkpoints', ignore_errors=True)
PY
fi

cmds=()

# phase A: full data
for domain in restaurant laptop; do
  for model in bert gcn; do
    for seed in 42 43 44; do
      cmds+=("$PY -m scripts.run_one $model $domain 1 $seed none")
    done
  done
done

# phase B: graph control (dependency vs identity) at full data and 10%
for domain in restaurant laptop; do
  for frac in 1 0.1; do
    for seed in 42 43 44; do
      cmds+=("$PY -m scripts.run_one gcn $domain $frac $seed identity")
    done
  done
done

# phase C: reduced budgets
for domain in restaurant laptop; do
  for model in bert gcn; do
    for frac in 0.5 0.25 0.1; do
      for seed in 42 43 44; do
        cmds+=("$PY -m scripts.run_one $model $domain $frac $seed none")
      done
    done
  done
done

echo "Running ${#cmds[@]} training jobs with JOBS=${JOBS}, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
printf '%s\n' "${cmds[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'

"$PY" -m scripts.eval_cross_domain --model all --seeds 42 43 44
if [[ "${1:-}" != "--runs-only" ]]; then
  bash scripts/reproduce.sh
fi
echo "Grid complete."
