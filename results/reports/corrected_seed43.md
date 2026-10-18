# Per-seed model comparison - seed 43

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8420 | 0.7508 |
| restaurant | 1 | gcn | complete | 0.8348 | 0.7382 |
| restaurant | 0.1 | bert | complete | 0.6795 | 0.3921 |
| restaurant | 0.1 | gcn | complete | 0.7312 | 0.5109 |
| restaurant | 0.25 | bert | complete | 0.7661 | 0.5379 |
| restaurant | 0.25 | gcn | complete | 0.7982 | 0.6599 |
| restaurant | 0.5 | bert | complete | 0.8241 | 0.7241 |
| restaurant | 0.5 | gcn | complete | 0.8286 | 0.7218 |
| laptop | 1 | bert | complete | 0.7602 | 0.7063 |
| laptop | 1 | gcn | complete | 0.7555 | 0.6979 |
| laptop | 0.1 | bert | complete | 0.5627 | 0.4018 |
| laptop | 0.1 | gcn | complete | 0.6223 | 0.4887 |
| laptop | 0.25 | bert | complete | 0.7116 | 0.6052 |
| laptop | 0.25 | gcn | complete | 0.7273 | 0.6357 |
| laptop | 0.5 | bert | complete | 0.7179 | 0.6267 |
| laptop | 0.5 | gcn | complete | 0.7398 | 0.6750 |
| restaurant_to_laptop | 1 | bert | complete | 0.7320 | 0.6657 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7602 | 0.7071 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
