# CISC 473 Group 23 — ABSA

Current results: only unaffected BERT records and corrected GCN (`aspectfix`) records are retained. Invalid legacy GCN JSONs and derived statistics were removed. BERT has seeds 42/43/44; corrected GCN has seed 42 for both domains, full/10%/25%/50% budgets and cross-domain evaluation. Corrected seeds 43/44 and GCN epoch-budget extensions remain pending. Read `results/reports/corrected_seed42.md` for the matched one-seed comparison and `results/reports/experiment_status.csv` for full availability. Run `bash scripts/run_midterm_seed42.sh` to reuse the completed one-seed experiments.

A BERT sentence–aspect classifier and a two-layer dependency GCN extension, evaluated on SemEval-2014 Restaurant and Laptop with three random seeds, reduced training budgets and Restaurant-to-Laptop transfer.

- [QUICKSTART.md](QUICKSTART.md): environment, data preparation and experiment commands.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): code provenance, architecture, evaluation protocol and limitations.
- [CHECKPOINTS.md](CHECKPOINTS.md): retained models and storage policy.
- [results/INDEX.md](results/INDEX.md): saved experiment results.

The training pipeline and GCN are project-specific implementations built with Hugging Face Transformers and PyTorch, developed with ChatGPT/Codex assistance. They are not lightly modified copies of the ASGCN, DualGCN or BERT-PT author code. Those works motivate the comparison; their exact architectures and published scores have not been reproduced here.

The repository keeps code, dependency specifications and experiment records. Data, model checkpoints, environments and logs are ignored by Git. Follow QUICKSTART.md to prepare data; new inference with unavailable model weights requires retraining.
