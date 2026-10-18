#!/usr/bin/env bash
# Complete corrected GCN Midterm runs without changing the seed-42 protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
SEEDS="${SEEDS:-43 44}"
mkdir -p logs
if ! mkdir logs/midterm-training.lock 2>/dev/null; then
  echo 'Another Midterm runner may be active. Inspect it before removing logs/midterm-training.lock.' >&2
  exit 1
fi
trap 'rmdir logs/midterm-training.lock' EXIT
refresh() {
  python -m scripts.summarize_results
  python -m scripts.summarize_sweeps
  for seed in 42 43 44; do python -m scripts.summarize_corrected --seed "$seed"; done
}
refresh
# Verify matching BERT records before spending time on GCN training.
python - "$SEEDS" <<'PY'
import sys
from scripts.summarize_results import collect_records
_, status = collect_records()
seeds = [int(s) for s in sys.argv[1].split()]
assert seeds and all(s in (42,43,44) for s in seeds), seeds
missing = [r['file'] for r in status if r['model']=='bert' and r['seed'] in seeds and r['category']!='epoch_sweep' and r['status']!='complete']
assert not missing, f'Matching BERT results required: {missing}'
PY
run_gcn() {
  local domain="$1" fraction="$2" seed="$3" target logfile
  target=$(python -m src.result_paths gcn "$domain" "$fraction" "$seed" --tag aspectfix)
  logfile="logs/corrected_gcn_${domain}_fraction${fraction}_seed${seed}.log"
  if [[ -f "$target" ]]; then
    echo "REUSE corrected: $target"
  else
    echo "TRAIN corrected GCN: domain=$domain fraction=$fraction seed=$seed"
    if ! python -u -m src.train_gcn --domain "$domain" --train-fraction "$fraction" \
      --epochs 3 --batch-size 8 --seed "$seed" --run-tag aspectfix > "$logfile" 2>&1; then
      tail -n 40 "$logfile"
      return 1
    fi
    echo "SAVED $target"
  fi
  refresh
}
# Prioritize both domains' full-data results and transfer before reduced budgets.
for seed in $SEEDS; do
  run_gcn restaurant 1 "$seed"
  run_gcn laptop 1 "$seed"
  echo "EVALUATE corrected Restaurant -> Laptop, seed=$seed"
  logfile="logs/corrected_cross_domain_gcn_seed${seed}.log"
  if ! python -u -m scripts.eval_cross_domain --model gcn --seeds "$seed" --gcn-run-tag aspectfix > "$logfile" 2>&1; then
    tail -n 40 "$logfile"
    exit 1
  fi
  refresh
done
for fraction in 0.1 0.25 0.5; do
  for seed in $SEEDS; do
    for domain in restaurant laptop; do run_gcn "$domain" "$fraction" "$seed"; done
  done
done
echo 'CORRECTED MIDTERM RUNS COMPLETE'
