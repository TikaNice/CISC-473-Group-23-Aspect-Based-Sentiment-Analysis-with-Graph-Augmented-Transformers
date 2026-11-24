# Per-seed model comparison - seed 44

Test results for BERT and the corrected dependency GCN under the project protocol.
Settings: 3 epochs, batch 8, LR 2e-5, length 128; fixed split and budget seed 2026.
Each row represents one run. Multi-seed statistics are stored in summary_by_group.csv.

| Domain | Fraction | Model | Status | Test accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|
| restaurant | 1 | bert | complete | 0.8214 | 0.7260 |
| restaurant | 1 | gcn | complete | 0.8402 | 0.7451 |
| restaurant | 0.1 | bert | complete | 0.6759 | 0.4269 |
| restaurant | 0.1 | gcn | complete | 0.6946 | 0.4335 |
| restaurant | 0.25 | bert | complete | 0.7795 | 0.6167 |
| restaurant | 0.25 | gcn | complete | 0.7893 | 0.6432 |
| restaurant | 0.5 | bert | complete | 0.8080 | 0.6891 |
| restaurant | 0.5 | gcn | complete | 0.8223 | 0.7183 |
| laptop | 1 | bert | complete | 0.7633 | 0.7112 |
| laptop | 1 | gcn | complete | 0.7633 | 0.7139 |
| laptop | 0.1 | bert | complete | 0.5862 | 0.3925 |
| laptop | 0.1 | gcn | complete | 0.6238 | 0.4914 |
| laptop | 0.25 | bert | complete | 0.7022 | 0.6162 |
| laptop | 0.25 | gcn | complete | 0.7288 | 0.6532 |
| laptop | 0.5 | bert | complete | 0.7304 | 0.6679 |
| laptop | 0.5 | gcn | complete | 0.7382 | 0.6624 |
| restaurant_to_laptop | 1 | bert | complete | 0.7367 | 0.6800 |
| restaurant_to_laptop | 1 | gcn | complete | 0.7351 | 0.6665 |

GCN revision: case_sensitive_ordinal_v1. Metrics are fractions on [0, 1].
Data overlaps and protocol limitations are documented in IMPLEMENTATION.md.
