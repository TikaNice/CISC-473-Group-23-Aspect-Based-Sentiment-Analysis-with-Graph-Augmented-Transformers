# Checkpoints and valid results

Scores are stored in results/. Model weights are stored locally in checkpoints/ and ignored by Git.

Current weights include three BERT Restaurant baselines, eight newly corrected seed-42 GCN runs, and three obsolete Restaurant GCN models retained from the earlier storage cleanup. Those three obsolete GCN models and their archived Trainer states are excluded from current evaluation and summary generation. Removing invalid result JSONs does not retrain or validate old model weights.

Cross-domain evaluation loads corrected GCN models with the aspectfix tag. Seeds 43/44 must be trained before corrected GCN evaluation. No old GCN scores are retained in results/.

The original 86-run Trainer metadata archive remains a historical trace. The validation-curve extractor uses historical metadata only for BERT; GCN must use corrected aspectfix_ep4 runs, which are not trained yet. Valid BERT curve CSVs remain available. Archived metadata is not a model and does not imply that the referenced historical result is valid.

New corrected training saves model weights and tokenizer/Trainer metadata without optimizer state. Currently all local checkpoints occupy roughly 6.2 GB, compared with about 113.5 GB before cleanup. Future training can increase storage again. Optimizer-based resume is unavailable for the pruned historical runs.

Use QUICKSTART.md for commands and results/reports/experiment_status.csv for completed/pending experiments. GitHub uploads should contain code and valid result records, not checkpoints, data, environments or logs.
