> Historical experiment index. The GCN rows below precede the aspect-alignment correction. For corrected seed-42 results, read [reports/corrected_seed42.md](reports/corrected_seed42.md). Do not combine historical GCN seeds 43/44 with corrected seed 42 into a mean/SD.

# CISC 473 Group 23 — Experiment Results

Original experiment JSONs are organized by study, domain, model, and setting.
Scores below are **Gold Test** metrics, except smoke tests (validation only).
Values use sample standard deviation across available seeds (where n ≥ 2).

## File counts

- baseline: **12** JSON files
- low_resource: **36** JSON files
- epoch_sweep: **36** JSON files
- cross_domain: **6** JSON files
- smoke: **2** JSON files
- **Total: 92 JSON files**

## Metric summary

| Study | Domain | Model | Setting | Seeds | Test accuracy (mean ± SD) | Test Macro-F1 (mean ± SD) |
|---|---|---|---|---|---:|---:|
| baseline | laptop | bert | 100_percent_epochs_3 | 42, 43, 44 | 0.7638 ± 0.0122 | 0.7151 ± 0.0119 |
| baseline | laptop | gcn | 100_percent_epochs_3 | 42, 43, 44 | 0.7623 ± 0.0048 | 0.7138 ± 0.0118 |
| baseline | restaurant | bert | 100_percent_epochs_3 | 42, 43, 44 | 0.8351 ± 0.0067 | 0.7410 ± 0.0091 |
| baseline | restaurant | gcn | 100_percent_epochs_3 | 42, 43, 44 | 0.8327 ± 0.0029 | 0.7399 ± 0.0080 |
| low_resource | laptop | bert | 10_percent | 42, 43, 44 | 0.5883 ± 0.0403 | 0.4162 ± 0.0766 |
| low_resource | laptop | bert | 25_percent | 42, 43, 44 | 0.6693 ± 0.0602 | 0.5650 ± 0.0690 |
| low_resource | laptop | bert | 50_percent | 42, 43, 44 | 0.7388 ± 0.0229 | 0.6605 ± 0.0454 |
| low_resource | laptop | gcn | 10_percent | 42, 43, 44 | 0.5904 ± 0.0665 | 0.4619 ± 0.0758 |
| low_resource | laptop | gcn | 25_percent | 42, 43, 44 | 0.7090 ± 0.0180 | 0.6200 ± 0.0241 |
| low_resource | laptop | gcn | 50_percent | 42, 43, 44 | 0.7435 ± 0.0092 | 0.6777 ± 0.0195 |
| low_resource | restaurant | bert | 10_percent | 42, 43, 44 | 0.6851 ± 0.0155 | 0.4132 ± 0.0501 |
| low_resource | restaurant | bert | 25_percent | 42, 43, 44 | 0.7744 ± 0.0073 | 0.5844 ± 0.0414 |
| low_resource | restaurant | bert | 50_percent | 42, 43, 44 | 0.8030 ± 0.0189 | 0.6854 ± 0.0372 |
| low_resource | restaurant | gcn | 10_percent | 42, 43, 44 | 0.7143 ± 0.0159 | 0.4648 ± 0.0413 |
| low_resource | restaurant | gcn | 25_percent | 42, 43, 44 | 0.7961 ± 0.0029 | 0.6548 ± 0.0099 |
| low_resource | restaurant | gcn | 50_percent | 42, 43, 44 | 0.8259 ± 0.0064 | 0.7221 ± 0.0126 |
| epoch_sweep | laptop | bert | epochs_1 | 42, 43, 44 | 0.7377 ± 0.0018 | 0.6559 ± 0.0132 |
| epoch_sweep | laptop | bert | epochs_2 | 42, 43, 44 | 0.7701 ± 0.0024 | 0.7179 ± 0.0079 |
| epoch_sweep | laptop | bert | epochs_4 | 42, 43, 44 | 0.7602 ± 0.0063 | 0.7099 ± 0.0090 |
| epoch_sweep | laptop | gcn | epochs_1 | 42, 43, 44 | 0.7597 ± 0.0157 | 0.6968 ± 0.0215 |
| epoch_sweep | laptop | gcn | epochs_2 | 42, 43, 44 | 0.7654 ± 0.0045 | 0.7090 ± 0.0133 |
| epoch_sweep | laptop | gcn | epochs_4 | 42, 43, 44 | 0.7701 ± 0.0086 | 0.7164 ± 0.0121 |
| epoch_sweep | restaurant | bert | epochs_1 | 42, 43, 44 | 0.7917 ± 0.0063 | 0.6218 ± 0.0216 |
| epoch_sweep | restaurant | bert | epochs_2 | 42, 43, 44 | 0.8170 ± 0.0071 | 0.6980 ± 0.0160 |
| epoch_sweep | restaurant | bert | epochs_4 | 42, 43, 44 | 0.8408 ± 0.0014 | 0.7548 ± 0.0068 |
| epoch_sweep | restaurant | gcn | epochs_1 | 42, 43, 44 | 0.7967 ± 0.0081 | 0.6463 ± 0.0145 |
| epoch_sweep | restaurant | gcn | epochs_2 | 42, 43, 44 | 0.8315 ± 0.0059 | 0.7323 ± 0.0130 |
| epoch_sweep | restaurant | gcn | epochs_4 | 42, 43, 44 | 0.8399 ± 0.0106 | 0.7554 ± 0.0184 |
| cross_domain | restaurant_to_laptop | bert | zero_shot | 42, 43, 44 | 0.7262 ± 0.0086 | 0.6595 ± 0.0079 |
| cross_domain | restaurant_to_laptop | gcn | zero_shot | 42, 43, 44 | 0.7576 ± 0.0094 | 0.7025 ± 0.0123 |
| smoke | restaurant | bert | functional_test | 42 | — | — |
| smoke | restaurant | gcn | functional_test | 42 | — | — |

## Exact files

### baseline (12)

- `baseline/laptop/bert/bert_laptop_fraction1_seed42.json`
- `baseline/laptop/bert/bert_laptop_fraction1_seed43.json`
- `baseline/laptop/bert/bert_laptop_fraction1_seed44.json`
- `baseline/laptop/gcn/gcn_laptop_fraction1_seed42.json`
- `baseline/laptop/gcn/gcn_laptop_fraction1_seed43.json`
- `baseline/laptop/gcn/gcn_laptop_fraction1_seed44.json`
- `baseline/restaurant/bert/bert_restaurant_fraction1_seed42.json`
- `baseline/restaurant/bert/bert_restaurant_fraction1_seed43.json`
- `baseline/restaurant/bert/bert_restaurant_fraction1_seed44.json`
- `baseline/restaurant/gcn/gcn_restaurant_fraction1_seed42.json`
- `baseline/restaurant/gcn/gcn_restaurant_fraction1_seed43.json`
- `baseline/restaurant/gcn/gcn_restaurant_fraction1_seed44.json`

### low_resource (36)

- `low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed42.json`
- `low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed43.json`
- `low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed44.json`
- `low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed42.json`
- `low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed43.json`
- `low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed44.json`
- `low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed42.json`
- `low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed43.json`
- `low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed44.json`
- `low_resource/laptop/gcn/10_percent/gcn_laptop_fraction0.1_seed42.json`
- `low_resource/laptop/gcn/10_percent/gcn_laptop_fraction0.1_seed43.json`
- `low_resource/laptop/gcn/10_percent/gcn_laptop_fraction0.1_seed44.json`
- `low_resource/laptop/gcn/25_percent/gcn_laptop_fraction0.25_seed42.json`
- `low_resource/laptop/gcn/25_percent/gcn_laptop_fraction0.25_seed43.json`
- `low_resource/laptop/gcn/25_percent/gcn_laptop_fraction0.25_seed44.json`
- `low_resource/laptop/gcn/50_percent/gcn_laptop_fraction0.5_seed42.json`
- `low_resource/laptop/gcn/50_percent/gcn_laptop_fraction0.5_seed43.json`
- `low_resource/laptop/gcn/50_percent/gcn_laptop_fraction0.5_seed44.json`
- `low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed42.json`
- `low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed43.json`
- `low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed44.json`
- `low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed42.json`
- `low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed43.json`
- `low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed44.json`
- `low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed42.json`
- `low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed43.json`
- `low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed44.json`
- `low_resource/restaurant/gcn/10_percent/gcn_restaurant_fraction0.1_seed42.json`
- `low_resource/restaurant/gcn/10_percent/gcn_restaurant_fraction0.1_seed43.json`
- `low_resource/restaurant/gcn/10_percent/gcn_restaurant_fraction0.1_seed44.json`
- `low_resource/restaurant/gcn/25_percent/gcn_restaurant_fraction0.25_seed42.json`
- `low_resource/restaurant/gcn/25_percent/gcn_restaurant_fraction0.25_seed43.json`
- `low_resource/restaurant/gcn/25_percent/gcn_restaurant_fraction0.25_seed44.json`
- `low_resource/restaurant/gcn/50_percent/gcn_restaurant_fraction0.5_seed42.json`
- `low_resource/restaurant/gcn/50_percent/gcn_restaurant_fraction0.5_seed43.json`
- `low_resource/restaurant/gcn/50_percent/gcn_restaurant_fraction0.5_seed44.json`

### epoch_sweep (36)

- `epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed42_ep1.json`
- `epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed43_ep1.json`
- `epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed44_ep1.json`
- `epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed42_ep2.json`
- `epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed43_ep2.json`
- `epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed44_ep2.json`
- `epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed42_ep4.json`
- `epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed43_ep4.json`
- `epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed44_ep4.json`
- `epoch_sweep/laptop/gcn/epochs_1/gcn_laptop_fraction1_seed42_ep1.json`
- `epoch_sweep/laptop/gcn/epochs_1/gcn_laptop_fraction1_seed43_ep1.json`
- `epoch_sweep/laptop/gcn/epochs_1/gcn_laptop_fraction1_seed44_ep1.json`
- `epoch_sweep/laptop/gcn/epochs_2/gcn_laptop_fraction1_seed42_ep2.json`
- `epoch_sweep/laptop/gcn/epochs_2/gcn_laptop_fraction1_seed43_ep2.json`
- `epoch_sweep/laptop/gcn/epochs_2/gcn_laptop_fraction1_seed44_ep2.json`
- `epoch_sweep/laptop/gcn/epochs_4/gcn_laptop_fraction1_seed42_ep4.json`
- `epoch_sweep/laptop/gcn/epochs_4/gcn_laptop_fraction1_seed43_ep4.json`
- `epoch_sweep/laptop/gcn/epochs_4/gcn_laptop_fraction1_seed44_ep4.json`
- `epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed42_ep1.json`
- `epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed43_ep1.json`
- `epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed44_ep1.json`
- `epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed42_ep2.json`
- `epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed43_ep2.json`
- `epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed44_ep2.json`
- `epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed42_ep4.json`
- `epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed43_ep4.json`
- `epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed44_ep4.json`
- `epoch_sweep/restaurant/gcn/epochs_1/gcn_restaurant_fraction1_seed42_ep1.json`
- `epoch_sweep/restaurant/gcn/epochs_1/gcn_restaurant_fraction1_seed43_ep1.json`
- `epoch_sweep/restaurant/gcn/epochs_1/gcn_restaurant_fraction1_seed44_ep1.json`
- `epoch_sweep/restaurant/gcn/epochs_2/gcn_restaurant_fraction1_seed42_ep2.json`
- `epoch_sweep/restaurant/gcn/epochs_2/gcn_restaurant_fraction1_seed43_ep2.json`
- `epoch_sweep/restaurant/gcn/epochs_2/gcn_restaurant_fraction1_seed44_ep2.json`
- `epoch_sweep/restaurant/gcn/epochs_4/gcn_restaurant_fraction1_seed42_ep4.json`
- `epoch_sweep/restaurant/gcn/epochs_4/gcn_restaurant_fraction1_seed43_ep4.json`
- `epoch_sweep/restaurant/gcn/epochs_4/gcn_restaurant_fraction1_seed44_ep4.json`

### cross_domain (6)

- `cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed42.json`
- `cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed43.json`
- `cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed44.json`
- `cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed42.json`
- `cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed43.json`
- `cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed44.json`

### smoke (2)

- `smoke/restaurant/bert/bert_restaurant_fraction1_seed42_smoke.json`
- `smoke/restaurant/gcn/gcn_restaurant_fraction1_seed42_smoke.json`

## Compatibility notes

- Training and sweep runners now use src/result_paths.py.
  Training now writes directly to canonical folders; no organizer is needed.
- `python -m scripts.summarize_results` rebuilds the complete run and group tables.
- `bash scripts/reproduce.sh` runs missing experiments and rebuilds summaries.
- See QUICKSTART.md for experiment commands and CHECKPOINTS.md for model storage.
- The JSON payload of each experiment is unchanged by this organizer.
