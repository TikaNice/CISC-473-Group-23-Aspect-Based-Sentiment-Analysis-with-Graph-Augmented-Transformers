# Corrected BERT + GCN comparison — seed 42

Single-seed preliminary results. No multi-seed mean, SD, or significance claim.
BERT reuses the existing matching project configuration. GCN uses case-sensitive aspect alignment.
Training settings remain the project protocol (3 epochs, batch 8, LR 2e-5, length 128), not the 2019 author protocol.
Historical GCN scores are excluded. Epoch-budget experiments are deferred.

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

Matching data splits and sentence-budget sampling use seed 2026. Existing train/test overlaps remain disclosed in IMPLEMENTATION.md.
Course reproducibility guidance requests at least three runs; this one-seed correction is an interim checkpoint for the group, with remaining seeds still pending.
