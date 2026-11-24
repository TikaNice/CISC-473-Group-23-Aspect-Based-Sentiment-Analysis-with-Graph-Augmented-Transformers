# Per-seed model comparison - seed 42

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8304 | 0.7377 |
| restaurant | 1 | gcn | complete | 0.8357 | 0.7452 |
| restaurant | 0.1 | bert | complete | 0.6920 | 0.4174 |
| restaurant | 0.1 | gcn | complete | 0.7161 | 0.4656 |
| restaurant | 0.25 | bert | complete | 0.7902 | 0.6307 |
| restaurant | 0.25 | gcn | complete | 0.7982 | 0.6580 |
| restaurant | 0.5 | bert | complete | 0.8161 | 0.7054 |
| restaurant | 0.5 | gcn | complete | 0.8313 | 0.7344 |
| laptop | 1 | bert | complete | 0.7853 | 0.7322 |
| laptop | 1 | gcn | complete | 0.7727 | 0.7213 |
| laptop | 0.1 | bert | complete | 0.5799 | 0.3831 |
| laptop | 0.1 | gcn | complete | 0.6176 | 0.4657 |
| laptop | 0.25 | bert | complete | 0.6505 | 0.5073 |
| laptop | 0.25 | gcn | complete | 0.6912 | 0.6113 |
| laptop | 0.5 | bert | complete | 0.7414 | 0.6789 |
| laptop | 0.5 | gcn | complete | 0.7335 | 0.6510 |
| restaurant_to_laptop | 1 | bert | complete | 0.7524 | 0.7001 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7539 | 0.6913 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
