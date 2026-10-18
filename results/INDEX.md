# Experiment results

The index includes valid BERT experiments and GCN experiments with case_sensitive_ordinal_v1 aspect alignment.

Available: 72/90 planned formal results; 73 saved records including optional smoke tests.

## Availability by model and experiment

| Model | Category | Complete | Pending |
|---|---|---:|---:|
| bert | baseline | 6 | 0 |
| bert | low_resource | 18 | 0 |
| bert | cross_domain | 3 | 0 |
| bert | epoch_sweep | 18 | 0 |
| gcn | baseline | 6 | 0 |
| gcn | low_resource | 18 | 0 |
| gcn | cross_domain | 3 | 0 |
| gcn | epoch_sweep | 0 | 18 |

The full-data, reduced-data and cross-domain matrix contains 54 planned results. GCN epoch-budget extensions are listed separately as pending.

## Reports

- [Every valid run](reports/all_metrics.csv)
- [Completed/pending run matrix](reports/experiment_status.csv)
- [Grouped means and sample SD](reports/summary_by_group.csv)
- [Full-data comparison](reports/baseline_summary.csv)
- [Data-budget comparison](reports/low_resource_summary.csv)
- [Cross-domain comparison](reports/cross_domain_summary.csv)

Grouped reports include exact seed counts, arithmetic means and sample standard deviations. A single-seed group has no standard deviation.

## Saved experiment records

- [baseline/laptop/bert/bert_laptop_fraction1_seed42.json](baseline/laptop/bert/bert_laptop_fraction1_seed42.json)
- [baseline/laptop/bert/bert_laptop_fraction1_seed43.json](baseline/laptop/bert/bert_laptop_fraction1_seed43.json)
- [baseline/laptop/bert/bert_laptop_fraction1_seed44.json](baseline/laptop/bert/bert_laptop_fraction1_seed44.json)
- [baseline/restaurant/bert/bert_restaurant_fraction1_seed42.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed42.json)
- [baseline/restaurant/bert/bert_restaurant_fraction1_seed43.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed43.json)
- [baseline/restaurant/bert/bert_restaurant_fraction1_seed44.json](baseline/restaurant/bert/bert_restaurant_fraction1_seed44.json)
- [cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed42.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed42.json)
- [cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed43.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed43.json)
- [cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed44.json](cross_domain/restaurant_to_laptop/bert/bert_restaurant_to_laptop_seed44.json)
- [cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed42_aspectfix.json](cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed42_aspectfix.json)
- [cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed43_aspectfix.json](cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed43_aspectfix.json)
- [cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed44_aspectfix.json](cross_domain/restaurant_to_laptop/gcn/gcn_restaurant_to_laptop_seed44_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.1_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.1_seed42_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.1_seed43_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.1_seed43_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.1_seed44_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.1_seed44_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.25_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.25_seed42_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.25_seed43_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.25_seed43_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.25_seed44_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.25_seed44_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.5_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.5_seed42_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.5_seed43_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.5_seed43_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction0.5_seed44_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction0.5_seed44_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction1_seed42_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction1_seed42_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction1_seed43_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction1_seed43_aspectfix.json)
- [custom/laptop/gcn/gcn_laptop_fraction1_seed44_aspectfix.json](custom/laptop/gcn/gcn_laptop_fraction1_seed44_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed42_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed43_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed43_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed44_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.1_seed44_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed42_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed43_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed43_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed44_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.25_seed44_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed42_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed43_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed43_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed44_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction0.5_seed44_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction1_seed42_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction1_seed42_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction1_seed43_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction1_seed43_aspectfix.json)
- [custom/restaurant/gcn/gcn_restaurant_fraction1_seed44_aspectfix.json](custom/restaurant/gcn/gcn_restaurant_fraction1_seed44_aspectfix.json)
- [epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed42_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed42_ep1.json)
- [epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed43_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed43_ep1.json)
- [epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed44_ep1.json](epoch_sweep/laptop/bert/epochs_1/bert_laptop_fraction1_seed44_ep1.json)
- [epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed42_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed42_ep2.json)
- [epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed43_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed43_ep2.json)
- [epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed44_ep2.json](epoch_sweep/laptop/bert/epochs_2/bert_laptop_fraction1_seed44_ep2.json)
- [epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed42_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed42_ep4.json)
- [epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed43_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed43_ep4.json)
- [epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed44_ep4.json](epoch_sweep/laptop/bert/epochs_4/bert_laptop_fraction1_seed44_ep4.json)
- [epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed42_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed42_ep1.json)
- [epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed43_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed43_ep1.json)
- [epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed44_ep1.json](epoch_sweep/restaurant/bert/epochs_1/bert_restaurant_fraction1_seed44_ep1.json)
- [epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed42_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed42_ep2.json)
- [epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed43_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed43_ep2.json)
- [epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed44_ep2.json](epoch_sweep/restaurant/bert/epochs_2/bert_restaurant_fraction1_seed44_ep2.json)
- [epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed42_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed42_ep4.json)
- [epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed43_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed43_ep4.json)
- [epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed44_ep4.json](epoch_sweep/restaurant/bert/epochs_4/bert_restaurant_fraction1_seed44_ep4.json)
- [low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed42.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed42.json)
- [low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed43.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed43.json)
- [low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed44.json](low_resource/laptop/bert/10_percent/bert_laptop_fraction0.1_seed44.json)
- [low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed42.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed42.json)
- [low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed43.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed43.json)
- [low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed44.json](low_resource/laptop/bert/25_percent/bert_laptop_fraction0.25_seed44.json)
- [low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed42.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed42.json)
- [low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed43.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed43.json)
- [low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed44.json](low_resource/laptop/bert/50_percent/bert_laptop_fraction0.5_seed44.json)
- [low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed42.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed42.json)
- [low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed43.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed43.json)
- [low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed44.json](low_resource/restaurant/bert/10_percent/bert_restaurant_fraction0.1_seed44.json)
- [low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed42.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed42.json)
- [low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed43.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed43.json)
- [low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed44.json](low_resource/restaurant/bert/25_percent/bert_restaurant_fraction0.25_seed44.json)
- [low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed42.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed42.json)
- [low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed43.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed43.json)
- [low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed44.json](low_resource/restaurant/bert/50_percent/bert_restaurant_fraction0.5_seed44.json)
- [smoke/restaurant/bert/bert_restaurant_fraction1_seed42_smoke.json](smoke/restaurant/bert/bert_restaurant_fraction1_seed42_smoke.json)
