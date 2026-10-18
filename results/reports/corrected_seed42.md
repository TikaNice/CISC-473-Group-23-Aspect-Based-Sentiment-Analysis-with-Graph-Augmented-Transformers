# Per-seed model comparison - seed 42

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8348 | 0.7394 |
| restaurant | 1 | gcn | complete | 0.8295 | 0.7332 |
| restaurant | 0.1 | bert | complete | 0.6732 | 0.3771 |
| restaurant | 0.1 | gcn | complete | 0.7098 | 0.4421 |
| restaurant | 0.25 | bert | complete | 0.7777 | 0.5986 |
| restaurant | 0.25 | gcn | complete | 0.7937 | 0.6433 |
| restaurant | 0.5 | bert | complete | 0.7875 | 0.6499 |
| restaurant | 0.5 | gcn | complete | 0.8295 | 0.7324 |
| laptop | 1 | bert | complete | 0.7774 | 0.7286 |
| laptop | 1 | gcn | complete | 0.7712 | 0.7258 |
| laptop | 0.1 | bert | complete | 0.5674 | 0.3478 |
| laptop | 0.1 | gcn | complete | 0.5141 | 0.3768 |
| laptop | 0.25 | bert | complete | 0.6003 | 0.4852 |
| laptop | 0.25 | gcn | complete | 0.6944 | 0.5955 |
| laptop | 0.5 | bert | complete | 0.7351 | 0.6428 |
| laptop | 0.5 | gcn | complete | 0.7539 | 0.6970 |
| restaurant_to_laptop | 1 | bert | complete | 0.7304 | 0.6620 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7476 | 0.6893 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
