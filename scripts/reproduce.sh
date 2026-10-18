#!/usr/bin/env bash
# Run with the project environment activated. Existing JSONs are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
for domain in restaurant laptop; do
  for input in "data/raw/$domain/train.jsonl" "data/processed/$domain/test_gold.jsonl"; do
    if [[ ! -f "$input" ]]; then
      echo "Missing $input. Follow QUICKSTART.md to prepare the datasets first." >&2
      exit 1
    fi
  done
done
bash scripts/run_low_resource.sh
bash scripts/run_epoch_sweep.sh
python -m scripts.eval_cross_domain --model all --seeds 42 43 44
python -m scripts.summarize_results
python -m scripts.summarize_sweeps
python -m scripts.extract_epoch_validation_curve

# Legacy summary tables above are historical. Keep corrected scores separate.
for seed in ${SEEDS:-42 43 44}; do
  python -m scripts.summarize_corrected --seed "$seed"
done
