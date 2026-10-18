#!/usr/bin/env bash
# Independent training schedules of 1, 2, 4 epochs. Reuses existing epoch 3 files.
set -euo pipefail
mkdir -p logs
DOMAINS="${DOMAINS:-restaurant laptop}"
MODELS="${MODELS:-bert gcn}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-1 2 4}"
for epochs in $EPOCHS; do
  for seed in $SEEDS; do
    for domain in $DOMAINS; do
      for model in $MODELS; do
        tag="ep${epochs}"
        [[ "$model" == "gcn" ]] && tag="aspectfix_ep${epochs}"
        out=$(python -m src.result_paths "$model" "$domain" 1 "$seed" --tag "$tag")
        if [[ -f "$out" ]]; then
          echo "SKIP existing: $out"
          continue
        fi
        echo "RUN epochs=$epochs model=$model domain=$domain seed=$seed"
        python -u -m "src.train_${model}" \
          --domain "$domain" --epochs "$epochs" \
          --batch-size 8 --seed "$seed" --run-tag "ep${epochs}" \
          2>&1 | tee "logs/epoch_${model}_${domain}_seed${seed}_ep${epochs}.log"
      done
    done
  done
done
echo 'EPOCH SWEEP COMPLETE'
