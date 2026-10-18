# CISC 473 Group 23 — ABSA

Current correction: historical GCN results used a case-insensitive aspect lookup. Corrected runs use the `aspectfix` tag and are reported separately. For the current single-seed Midterm run, use `bash scripts/run_midterm_seed42.sh`; read `results/reports/corrected_seed42.md` for progress and completed scores. Existing three-seed GCN tables are historical and must not be presented as corrected results. BERT-only records are reused.

A BERT sentence–aspect classifier and a two-layer dependency GCN extension, evaluated on SemEval-2014 Restaurant and Laptop with three random seeds, reduced training budgets and Restaurant-to-Laptop transfer.

- [QUICKSTART.md](QUICKSTART.md): environment, data preparation and experiment commands.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): code provenance, architecture, evaluation protocol and limitations.
- [CHECKPOINTS.md](CHECKPOINTS.md): retained models and storage policy.
- [results/INDEX.md](results/INDEX.md): saved experiment results.

The training pipeline and GCN are project-specific implementations built with Hugging Face Transformers and PyTorch, developed with ChatGPT/Codex assistance. They are not lightly modified copies of the ASGCN, DualGCN or BERT-PT author code. Those works motivate the comparison; their exact architectures and published scores have not been reproduced here.

The repository keeps code, dependency specifications and experiment records. Data, model checkpoints, environments and logs are ignored by Git. Follow QUICKSTART.md to prepare data; new inference with unavailable model weights requires retraining.
