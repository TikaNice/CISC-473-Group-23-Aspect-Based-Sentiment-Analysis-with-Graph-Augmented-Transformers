# Per-seed model comparison - seed 43

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8402 | 0.7489 |
| restaurant | 1 | gcn | complete | 0.8313 | 0.7364 |
| restaurant | 0.1 | bert | complete | 0.6929 | 0.4308 |
| restaurant | 0.1 | gcn | complete | 0.7214 | 0.4831 |
| restaurant | 0.25 | bert | complete | 0.7705 | 0.5666 |
| restaurant | 0.25 | gcn | complete | 0.7982 | 0.6679 |
| restaurant | 0.5 | bert | complete | 0.8045 | 0.6742 |
| restaurant | 0.5 | gcn | complete | 0.8223 | 0.7119 |
| laptop | 1 | bert | complete | 0.7649 | 0.7224 |
| laptop | 1 | gcn | complete | 0.7618 | 0.7118 |
| laptop | 0.1 | bert | complete | 0.6034 | 0.4278 |
| laptop | 0.1 | gcn | complete | 0.5972 | 0.4330 |
| laptop | 0.25 | bert | complete | 0.7163 | 0.6039 |
| laptop | 0.25 | gcn | complete | 0.6912 | 0.5950 |
| laptop | 0.5 | bert | complete | 0.7414 | 0.6680 |
| laptop | 0.5 | gcn | complete | 0.7476 | 0.6718 |
| restaurant_to_laptop | 1 | bert | complete | 0.7476 | 0.6916 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7602 | 0.6881 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
