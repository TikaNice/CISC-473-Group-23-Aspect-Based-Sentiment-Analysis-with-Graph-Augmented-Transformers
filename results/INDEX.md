# Experiment results

Only unaffected BERT results and corrected GCN results are retained. Invalid legacy GCN scores, smoke output and derived statistics were removed from the project.

## Availability

- BERT: 45 formal runs across seeds 42/43/44, plus one smoke run.
- Corrected BERT + GCN: 9 formal results at seed 42 (8 training/evaluation runs and 1 cross-domain inference).
- Corrected GCN seeds 43/44: not trained yet for the core/data-budget/cross-domain matrix.
- Corrected GCN epoch-budget extensions: not trained yet at any seed; deferred.
- Total retained experiment JSONs: 55 (54 formal + 1 BERT smoke).

## Which report to use

- [Matched seed-42 comparison](reports/corrected_seed42.md): compare both models using the same seed.
- [Experiment status](reports/experiment_status.csv): complete/pending entries for the full planned matrix.
- [All valid records](reports/all_metrics.csv): every retained run.
- [Grouped statistics](reports/summary_by_group.csv): includes seed counts; GCN single-seed scores have no SD.

Do not compare a BERT three-seed mean to a GCN single score without labeling the difference. No old GCN results are used to fill missing seeds.

## Valid grouped results

| Study | Domain | Model | Fraction | Epochs | Seeds | Test accuracy | Test Macro-F1 | SD status |
|---|---|---|---|---|---|---:|---:|---|
| baseline | laptop | bert | 1 | 3 | 42,43,44 | 0.7638 | 0.7151 | three_seeds |
| baseline | laptop | gcn | 1 | 3 | 42 | 0.7712 | 0.7258 | single_seed |
| baseline | restaurant | bert | 1 | 3 | 42,43,44 | 0.8351 | 0.7410 | three_seeds |
| baseline | restaurant | gcn | 1 | 3 | 42 | 0.8295 | 0.7332 | single_seed |
| cross_domain | restaurant_to_laptop | bert | 1 | 3 | 42,43,44 | 0.7262 | 0.6595 | three_seeds |
| cross_domain | restaurant_to_laptop | gcn | 1 | 3 | 42 | 0.7476 | 0.6893 | single_seed |
| epoch_sweep | laptop | bert | 1 | 1 | 42,43,44 | 0.7377 | 0.6559 | three_seeds |
| epoch_sweep | laptop | bert | 1 | 2 | 42,43,44 | 0.7701 | 0.7179 | three_seeds |
| epoch_sweep | laptop | bert | 1 | 4 | 42,43,44 | 0.7602 | 0.7099 | three_seeds |
| epoch_sweep | restaurant | bert | 1 | 1 | 42,43,44 | 0.7917 | 0.6218 | three_seeds |
| epoch_sweep | restaurant | bert | 1 | 2 | 42,43,44 | 0.8170 | 0.6980 | three_seeds |
| epoch_sweep | restaurant | bert | 1 | 4 | 42,43,44 | 0.8408 | 0.7548 | three_seeds |
| low_resource | laptop | bert | 0.1 | 3 | 42,43,44 | 0.5883 | 0.4162 | three_seeds |
| low_resource | laptop | bert | 0.25 | 3 | 42,43,44 | 0.6693 | 0.5650 | three_seeds |
| low_resource | laptop | bert | 0.5 | 3 | 42,43,44 | 0.7388 | 0.6605 | three_seeds |
| low_resource | laptop | gcn | 0.1 | 3 | 42 | 0.5141 | 0.3768 | single_seed |
| low_resource | laptop | gcn | 0.25 | 3 | 42 | 0.6944 | 0.5955 | single_seed |
| low_resource | laptop | gcn | 0.5 | 3 | 42 | 0.7539 | 0.6970 | single_seed |
| low_resource | restaurant | bert | 0.1 | 3 | 42,43,44 | 0.6851 | 0.4132 | three_seeds |
| low_resource | restaurant | bert | 0.25 | 3 | 42,43,44 | 0.7744 | 0.5844 | three_seeds |
| low_resource | restaurant | bert | 0.5 | 3 | 42,43,44 | 0.8030 | 0.6854 | three_seeds |
| low_resource | restaurant | gcn | 0.1 | 3 | 42 | 0.7098 | 0.4421 | single_seed |
| low_resource | restaurant | gcn | 0.25 | 3 | 42 | 0.7937 | 0.6433 | single_seed |
| low_resource | restaurant | gcn | 0.5 | 3 | 42 | 0.8295 | 0.7324 | single_seed |

## Exact retained files

- [bert_restaurant_fraction1_seed42.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed42.json)
- [bert_restaurant_fraction1_seed43.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed43.json)
- [bert_restaurant_fraction1_seed44.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed44.json)
- [bert_restaurant_fraction0.1_seed42.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed42.json)
- [bert_restaurant_fraction0.1_seed43.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed43.json)
- [bert_restaurant_fraction0.1_seed44.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed44.json)
- [bert_restaurant_fraction0.25_seed42.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed42.json)
- [bert_restaurant_fraction0.25_seed43.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed43.json)
- [bert_restaurant_fraction0.25_seed44.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed44.json)
- [bert_restaurant_fraction0.5_seed42.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed42.json)
- [bert_restaurant_fraction0.5_seed43.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed43.json)
- [bert_restaurant_fraction0.5_seed44.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed44.json)
- [bert_restaurant_fraction1_seed42_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed42_ep1.json)
- [bert_restaurant_fraction1_seed43_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed43_ep1.json)
- [bert_restaurant_fraction1_seed44_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed44_ep1.json)
- [bert_restaurant_fraction1_seed42_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed42_ep2.json)
- [bert_restaurant_fraction1_seed43_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed43_ep2.json)
- [bert_restaurant_fraction1_seed44_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed44_ep2.json)
- [bert_restaurant_fraction1_seed42_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed42_ep4.json)
- [bert_restaurant_fraction1_seed43_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed43_ep4.json)
- [bert_restaurant_fraction1_seed44_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed44_ep4.json)
- [bert_restaurant_to_laptop_seed42.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed42.json)
- [bert_restaurant_to_laptop_seed43.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed43.json)
- [bert_restaurant_to_laptop_seed44.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed44.json)
- [bert_laptop_fraction1_seed42.json](baseline/laptop/bert/bert_laptop_fraction1_seed42.json)
- [bert_laptop_fraction1_seed43.json](baseline/laptop/bert/bert_laptop_fraction1_seed43.json)
- [bert_laptop_fraction1_seed44.json](baseline/laptop/bert/bert_laptop_fraction1_seed44.json)
- [bert_laptop_fraction0.1_seed42.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed42.json)
- [bert_laptop_fraction0.1_seed43.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed43.json)
- [bert_laptop_fraction0.1_seed44.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed44.json)
- [bert_laptop_fraction0.25_seed42.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed42.json)
- [bert_laptop_fraction0.25_seed43.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed43.json)
- [bert_laptop_fraction0.25_seed44.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed44.json)
- [bert_laptop_fraction0.5_seed42.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed42.json)
- [bert_laptop_fraction0.5_seed43.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed43.json)
- [bert_laptop_fraction0.5_seed44.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed44.json)
- [bert_laptop_fraction1_seed42_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed42_ep1.json)
- [bert_laptop_fraction1_seed43_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed43_ep1.json)
- [bert_laptop_fraction1_seed44_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed44_ep1.json)
- [bert_laptop_fraction1_seed42_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed42_ep2.json)
- [bert_laptop_fraction1_seed43_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed43_ep2.json)
- [bert_laptop_fraction1_seed44_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed44_ep2.json)
- [bert_laptop_fraction1_seed42_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed42_ep4.json)
- [bert_laptop_fraction1_seed43_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed43_ep4.json)
- [bert_laptop_fraction1_seed44_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed44_ep4.json)
- [gcn_restaurant_fraction1_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction1_seed42_aspectfix.json)
- [gcn_restaurant_fraction0.1_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed42_aspectfix.json)
- [gcn_restaurant_fraction0.25_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed42_aspectfix.json)
- [gcn_restaurant_fraction0.5_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed42_aspectfix.json)
- [gcn_restaurant_to_laptop_seed42_aspectfix.json](cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed42_aspectfix.json)
- [gcn_laptop_fraction1_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction1_seed42_aspectfix.json)
- [gcn_laptop_fraction0.1_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.1_seed42_aspectfix.json)
- [gcn_laptop_fraction0.25_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.25_seed42_aspectfix.json)
- [gcn_laptop_fraction0.5_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.5_seed42_aspectfix.json)
- [bert_restaurant_fraction1_seed42_smoke.json](smoke/restaurant/bert/bert_restaurant_fraction1_seed42_smoke.json)

Regenerate with `python -m scripts.summarize_results`, `python -m scripts.summarize_sweeps` and `python -m scripts.summarize_corrected --seed 42` from the project root.
