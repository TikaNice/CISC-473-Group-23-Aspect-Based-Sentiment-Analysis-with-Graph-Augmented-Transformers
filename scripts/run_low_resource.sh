#!/usr/bin/env bash
# Run from group23 project root with .venv activated.
set -euo pipefail
mkdir -p logs
# By default reuse the completed 100% experiments; train 10%, 25%, 50%.
DOMAINS="${DOMAINS:-restaurant laptop}"
MODELS="${MODELS:-bert gcn}"
SEEDS="${SEEDS:-42 43 44}"
FRACTIONS="${FRACTIONS:-0.1 0.25 0.5 1}"

for frac in $FRACTIONS; do
  for seed in $SEEDS; do
    for domain in $DOMAINS; do
      for model in $MODELS; do
        tag=""
        [[ "$model" == "gcn" ]] && tag="aspectfix"
        out=$(python -m src.result_paths "$model" "$domain" "$frac" "$seed" --tag "$tag")
        if [[ -f "$out" ]]; then
          echo "SKIP existing: $out"
          continue
        fi
        echo "RUN model=$model domain=$domain fraction=$frac seed=$seed"
        python -u -m "src.train_${model}" \
          --domain "$domain" \
          --train-fraction "$frac" \
          --epochs 3 --batch-size 8 --seed "$seed" \
          2>&1 | tee "logs/lowres_${model}_${domain}_f${frac}_seed${seed}.log"
      done
    done
  done
done
echo 'LOW-RESOURCE SWEEP COMPLETE'
