# Checkpoints and inference

## Where checkpoints are written

Every training run stores its checkpoints under `checkpoints/`, mirroring the
result path in `results/`:

| Run | Checkpoint folder |
| --- | --- |
| Restaurant, full data, seed 42, BERT | `checkpoints/baseline/restaurant/bert/bert_restaurant_fraction1_seed42/` |
| Restaurant, full data, seed 42, BERT+GCN | `checkpoints/custom/restaurant/gcn/gcn_restaurant_fraction1_seed42_aspectfix/` |
| Restaurant, 10% data, seed 42, BERT+GCN | `checkpoints/custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed42_aspectfix/` |
| Identity-graph control, full data, seed 42 | `checkpoints/ablation/restaurant/gcn/gcn_restaurant_fraction1_seed42_identity/` |

Each run directory holds Trainer checkpoints. Trainer selects the epoch with the
highest validation macro-F1 (`load_best_model_at_end=True`). New runs save model
weights and Trainer state. A weights-only checkpoint is roughly 0.44 GB. Earlier
BERT runs may also contain optimizer state.

## Restaurant -> Laptop transfer

`scripts/eval_cross_domain.py` needs the full-data Restaurant checkpoints for
seeds 42, 43 and 44, one per model. It resolves the best-validation checkpoint
from completed Trainer state and checks that the recorded weights exist. For runs
written by the older trainer, it reads state from the latest retained checkpoint.

```sh
python -m scripts.eval_cross_domain --model all --seeds 42 43 44
```

It only evaluates on the Laptop Gold Test set; no Laptop training or Laptop
checkpoint selection happens.

## If checkpoints are not available

Retrain the two Restaurant full-data models for the three seeds, then run the
transfer evaluation:

```sh
for seed in 42 43 44; do
  python -m src.train_bert --domain restaurant --train-fraction 1 --seed "$seed" --epochs 3
  python -m src.train_gcn  --domain restaurant --train-fraction 1 --seed "$seed" --epochs 3
done
python -m scripts.eval_cross_domain --model all --seeds 42 43 44 --overwrite
```

The `--overwrite` flag forces a fresh evaluation after retraining. Checkpoints
are kept locally, outside Git and the submission ZIP.
