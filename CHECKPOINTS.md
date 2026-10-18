# Retained checkpoints

`results/` contains scores and statistics. `checkpoints/` retains only the six Restaurant baseline models needed for Restaurant → Laptop inference: BERT and GCN, each with seeds 42, 43 and 44.

## Storage policy

Following the project owner's choice, all other trained model weights and all optimizer, scheduler and random-number states were deleted. These deleted files were not backed up. Recreating those models requires training again; resuming the original optimizer state is no longer possible.

The retained model parameters occupy about 2.64 GB (2.46 GiB), plus tokenizer/configuration files and small metadata. Previously the checkpoints occupied about 113.5 GB (105.7 GiB).

```text
checkpoints/
├── baseline/restaurant/
│   ├── bert/bert_restaurant_fraction1_seed{42,43,44}/checkpoint-<step>/
│   └── gcn/gcn_restaurant_fraction1_seed{42,43,44}/checkpoint-<step>/
├── training_metadata.tar.gz
├── INDEX.csv
└── README.md
```

`INDEX.csv` lists currently retained runs. The historical 86-run index is saved with the cleanup manifest outside the project.

## What remains possible

| Task | Availability |
|---|---|
| Rebuild result tables | All historical result JSONs remain unchanged |
| Recompute cross-domain scores | Six required Restaurant models remain |
| Predict using the retained Restaurant models | Available |
| Predict using deleted Laptop, low-resource or epoch models | Retraining required |
| Read saved validation curves | Existing CSVs remain unchanged |
| Re-extract retained Trainer validation history | Supported through training_metadata.tar.gz |
| Resume old training with optimizer state | Unavailable |

The metadata archive contains the original `trainer_state.json` files for all 86 runs. It preserves checkpoint selection records and available metric history, not trained models. The curve extractor uses it when checkpoint folders have been removed. It recovers 35 validation points from retained epoch-4 Trainer history; the complete 48-point historical curve is separately saved as `results/reports/epoch4_validation_from_logs.csv`.

## Future training

Training still saves checkpoints in the experiment hierarchy documented in QUICKSTART.md. New experiments can therefore grow this directory again. Batch scripts skip existing result JSONs; deleted model weights are not regenerated merely because their result files already exist. To recreate a deleted model, run its training command directly with the original settings. Use a distinct run tag when a new result should be kept separately.

Cross-domain evaluation resolves the six models through `src/result_paths.py`. Historical result JSONs retain their original checkpoint path strings as provenance. Archived Trainer records can likewise refer to deleted models; they are historical metadata, not loading instructions.

The directory is ignored by Git and has no tracked model files. Model weights are not needed to publish the source code, experiment configurations, result JSONs and summary tables on GitHub.
