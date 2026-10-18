# CISC 473 quickstart

Parts of this code were written with AI assistance. Review and understand it, verify the metrics yourself, and credit/disclose AI assistance as required by the course.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_data.py   # SetFit train splits + official gold test XML
python scripts/prepare_gold.py    # -> data/processed/<domain>/test_gold.jsonl
```

If `import datasets` fails with "pyarrow requires NumPy 2.0", either upgrade NumPy or pin `pip install "datasets<4" "pyarrow<20"`.

## Single runs

```bash
python -m src.train_bert --domain restaurant --smoke     # 64-example check, no test eval
python -m src.train_gcn  --domain restaurant --smoke
python -m src.train_gcn  --domain restaurant --train-fraction 0.1 --seed 123 --head-lr 1e-3
```

Every non-smoke run trains on one domain, selects the checkpoint on validation macro-F1, then evaluates on **both** gold test sets (in-domain and cross-domain). It writes:

- `results/<model>_<domain>_fraction<f>_seed<s>.json`: metrics, confusion matrices, encoding stats, training log history
- `results/preds/<model>_..._on_<eval_domain>.jsonl`: per-example predictions and probabilities
- `results/audit/`: GCN train/validation records skipped because the aspect could not be located (test records are never skipped)

`--head-lr` sets the learning rate of newly initialised layers (GCN + classifier) separately from the BERT encoder (`--lr`, default 2e-5). Checkpoints are deleted after each run unless `--keep-checkpoints` is passed.

## Experiment grid (2 models x 2 domains x 4 label fractions x 3 seeds = 48 runs)

```bash
python scripts/run_grid.py --results-dir results/main --dry-run   # list the runs
python scripts/run_grid.py --results-dir results/main             # resumable; skips finished runs
python scripts/aggregate.py --results-dir results/main            # summary.md, tables.tex, figures/
python scripts/tag_errors.py --results-dir results/main           # error_analysis.md
```

Runs are ordered so all seeds at 100% data finish first, then 10%, 50%, 25%.
