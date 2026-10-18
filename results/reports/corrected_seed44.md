# Per-seed model comparison - seed 44

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8286 | 0.7329 |
| restaurant | 1 | gcn | complete | 0.8357 | 0.7428 |
| restaurant | 0.1 | bert | complete | 0.7027 | 0.4703 |
| restaurant | 0.1 | gcn | complete | 0.7027 | 0.4425 |
| restaurant | 0.25 | bert | complete | 0.7795 | 0.6168 |
| restaurant | 0.25 | gcn | complete | 0.7982 | 0.6624 |
| restaurant | 0.5 | bert | complete | 0.7973 | 0.6821 |
| restaurant | 0.5 | gcn | complete | 0.8179 | 0.7108 |
| laptop | 1 | bert | complete | 0.7539 | 0.7104 |
| laptop | 1 | gcn | complete | 0.7649 | 0.7213 |
| laptop | 0.1 | bert | complete | 0.6348 | 0.4989 |
| laptop | 0.1 | gcn | complete | 0.6364 | 0.5223 |
| laptop | 0.25 | bert | complete | 0.6959 | 0.6044 |
| laptop | 0.25 | gcn | complete | 0.7116 | 0.6405 |
| laptop | 0.5 | bert | complete | 0.7633 | 0.7122 |
| laptop | 0.5 | gcn | complete | 0.7351 | 0.6553 |
| restaurant_to_laptop | 1 | bert | complete | 0.7163 | 0.6506 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7571 | 0.7032 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
