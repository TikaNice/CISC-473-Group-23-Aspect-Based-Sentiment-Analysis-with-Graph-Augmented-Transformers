# CISC 473 Group 23 - ABSA

This project evaluates a BERT sentence-aspect classifier and a two-layer dependency GCN extension on SemEval-2014 Restaurant and Laptop. Experiments cover full-data classification, reduced training budgets, and Restaurant-to-Laptop transfer.

## Experiment status

Full-data, 10%/25%/50% data-budget, and cross-domain experiments are complete for both models with random seeds 42, 43 and 44: 54 formal results. The repository also contains 18 BERT epoch-budget results and one BERT smoke test. The 18 GCN epoch-budget extensions remain untrained.

| Setting | BERT Macro-F1 | BERT + GCN Macro-F1 |
|---|---:|---:|
| Restaurant, full data | 74.10 ± 0.91 | 73.81 ± 0.48 |
| Laptop, full data | 71.51 ± 1.19 | 71.50 ± 1.50 |
| Restaurant to Laptop | 65.95 ± 0.79 | 69.99 ± 0.93 |

Values are percentages, reported as mean ± sample standard deviation over three random seeds. GCN full-data performance is similar to BERT; its mean Macro-F1 is higher in all six reduced-data conditions and in cross-domain evaluation. The comparison does not isolate graph edges from other architectural differences, and no statistical significance is claimed.

## Documentation

- [QUICKSTART.md](QUICKSTART.md): environment, data preparation and experiment commands.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): architecture, training protocol, provenance and limitations.
- [CHECKPOINTS.md](CHECKPOINTS.md): local model storage and inference requirements.
- [results/INDEX.md](results/INDEX.md): experiment records and availability.

## Implementation scope

The training pipeline is a project-specific implementation using Hugging Face Transformers and PyTorch. Both models start independently from bert-base-uncased. The GCN uses one dependency graph, aspect pooling and CLS concatenation; it is distinct from the ASGCN, DualGCN and BERT-PT author implementations. Published baseline reproduction under matched preprocessing and training settings remains unverified.
