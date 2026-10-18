#!/usr/bin/env bash
# Correct GCN only; reuse matching BERT seed-42 records. Core results first.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
mkdir -p logs
run_gcn() {
  local domain="$1" fraction="$2" target logfile
  target=$(python -m src.result_paths gcn "$domain" "$fraction" 42 --tag aspectfix)
  logfile="logs/corrected_gcn_${domain}_fraction${fraction}_seed42.log"
  if [[ -f "$target" ]]; then
    echo "REUSE corrected: $target"
  else
    echo "TRAIN corrected GCN: domain=$domain fraction=$fraction seed=42"
    if ! python -u -m src.train_gcn --domain "$domain" --train-fraction "$fraction" \
      --epochs 3 --batch-size 8 --seed 42 --run-tag aspectfix > "$logfile" 2>&1; then
      tail -n 40 "$logfile"
      return 1
    fi
    echo "SAVED $target"
  fi
  python -m scripts.summarize_corrected --seed 42
}
run_gcn restaurant 1
run_gcn laptop 1
echo 'EVALUATE corrected Restaurant -> Laptop, seed 42'
if ! python -u -m scripts.eval_cross_domain --model gcn --seeds 42 --gcn-run-tag aspectfix \
  > logs/corrected_cross_domain_gcn_seed42.log 2>&1; then
  tail -n 40 logs/corrected_cross_domain_gcn_seed42.log
  exit 1
fi
python -m scripts.summarize_corrected --seed 42
for fraction in 0.1 0.25 0.5; do
  for domain in restaurant laptop; do
    run_gcn "$domain" "$fraction"
  done
done
echo 'CORRECTED SEED-42 MIDTERM EXPERIMENTS COMPLETE'
