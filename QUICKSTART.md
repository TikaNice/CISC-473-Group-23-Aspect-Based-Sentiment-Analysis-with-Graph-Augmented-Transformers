# Running the ABSA experiments

This project compares BERT and BERT + dependency GCN on SemEval-2014 Restaurant and Laptop. Commands below run from the project root. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture, sources and evaluation limitations.

## Current Midterm correction (seed 42)

```bash
bash scripts/run_midterm_seed42.sh
```

This runs corrected BERT + GCN on Restaurant and Laptop, evaluates Restaurant → Laptop, then runs 10%/25%/50% training budgets. Matching BERT-only seed-42 records are reused. It skips already completed `aspectfix` results and writes progress to `results/reports/corrected_seed42.md` and `.csv`. Epoch sweeps are deferred. Corrected checkpoints save model weights without optimizer state.

The course's Week-6 plan includes data-efficiency and cross-domain experiments. This one-seed run is preliminary; the course's three-run statistics remain a follow-up task. The existing historical GCN three-seed averages are not a substitute for corrected runs. The repair retains the project training protocol to match existing BERT records; it does not switch to the 2019 author protocol.

Rebuild only the corrected comparison:

```bash
python -m scripts.summarize_corrected --seed 42
```

## Environment

Activate the existing environment:

```bash
source .venv/bin/activate
```

For a fresh installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`requirements-current.txt` records the environment used during the project audit. It is provided as a version reference; `requirements.txt` contains the supported version ranges.

## Data

Skip this section if the data files are already present.

```bash
python scripts/download_data.py
```

Obtain the official SemEval-2014 Task 4 Gold Test XML files (available from the [SemEval2014Task4 dataset](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4)). Place `Restaurants_Test_Gold.xml` and `Laptops_Test_Gold.xml` in `data/gold/`, then convert them:

```bash
python scripts/prepare_gold.py
```

Training uses `data/raw/<domain>/train.jsonl`. Final testing uses `data/processed/<domain>/test_gold.jsonl`. The task has three sentiment classes; conflict labels are excluded. The downloaded raw test JSONL files have empty sentiment labels and are not used for scoring.

## Experiments

### Full-data baseline

Run both models on both domains with seeds 42, 43 and 44, using 3 epochs and batch size 8:

```bash
FRACTIONS=1 bash scripts/run_low_resource.sh
```

The same runner supports both full-data and reduced-data experiments. To run one model directly:

```bash
python -m src.train_bert --domain restaurant --epochs 3 --batch-size 8 --seed 42
python -m src.train_gcn --domain restaurant --epochs 3 --batch-size 8 --seed 42
```

For a short check of the training pipeline, add `--smoke` to either command. Smoke results are excluded from benchmark tables.

### Low-resource experiments

Train on 10%, 25% and 50% of the training sentence groups:

```bash
FRACTIONS="0.1 0.25 0.5" bash scripts/run_low_resource.sh
```

The default covers both models, both domains and all three seeds. To restrict the run:

```bash
DOMAINS=restaurant MODELS=gcn SEEDS=42 FRACTIONS=0.25 bash scripts/run_low_resource.sh
```

### Epoch experiments

Compare independent training runs with budgets of 1, 2 and 4 epochs:

```bash
bash scripts/run_epoch_sweep.sh
```

The 3-epoch comparison reuses the full-data baseline. To run a subset:

```bash
DOMAINS=laptop MODELS=bert SEEDS=42 EPOCHS="1 2" bash scripts/run_epoch_sweep.sh
```

Each run selects its best checkpoint by validation Macro-F1. The epoch budget does not necessarily equal the epoch of the selected checkpoint.

### Cross-domain evaluation

Evaluate Restaurant-trained baseline checkpoints on the Laptop test set:

```bash
python -m scripts.eval_cross_domain --model all --seeds 42 43 44
```

This performs inference only. It requires the six Restaurant baseline checkpoints. Add `--overwrite` to recompute existing cross-domain results.

### Complete experiment pipeline

After preparing the environment and data:

```bash
bash scripts/reproduce.sh
```

This runs full-data, low-resource and epoch experiments, evaluates cross-domain transfer, and rebuilds the summaries. Smoke tests are optional and are included in the run table only when present. The batch runners reuse existing result JSONs. Direct training commands rerun the requested experiment and can overwrite its result and checkpoint; use a distinct `--run-tag` for a separate run.

## Summaries

These commands read historical saved results without training (including legacy GCN results); use summarize_corrected above for the repaired comparison:

```bash
python -m scripts.summarize_results
python -m scripts.summarize_sweeps
```

| Output in `results/reports/` | Contents |
|---|---|
| `all_metrics.csv` | One row per run, including smoke tests |
| `summary_by_group.csv` | Mean and sample standard deviation for 30 formal three-seed groups |
| `baseline_summary.csv` | Full-data comparison |
| `cross_domain_summary.csv` | Restaurant → Laptop comparison |
| `low_resource_summary.csv` | Data-budget comparison, including the baseline |
| `epoch_sweep_summary.csv` | Epoch-budget comparison, including the baseline |
| `all_json_full.txt` | Full text of the experiment JSONs |

Extract validation history from retained epoch-4 checkpoints or the archived training metadata:

```bash
python -m scripts.extract_epoch_validation_curve
```

The resulting `epoch_validation_learning_curve.csv` can be incomplete because only selected checkpoints were retained. Optimizer state and nonessential model weights have been removed; archived Trainer metadata still supports this command. The historical `epoch4_validation_from_logs.csv` contains all 48 validation points recovered from the original logs.

## Output layout

New training results and checkpoints use the same experiment categories. Following storage cleanup, only six Restaurant baseline model checkpoints remain locally:

```text
results/                              checkpoints/
├── baseline/<domain>/<model>/         ├── baseline/<domain>/<model>/
├── low_resource/<domain>/<model>/     ├── low_resource/<domain>/<model>/
│   └── <percent>_percent/             │   └── <percent>_percent/
├── epoch_sweep/<domain>/<model>/      ├── epoch_sweep/<domain>/<model>/
│   └── epochs_<n>/                   │   └── epochs_<n>/
├── smoke/<domain>/<model>/            └── smoke/<domain>/<model>/
├── cross_domain/restaurant_to_laptop/
└── reports/
```

Each result is a JSON file. Each checkpoint run is a directory containing `checkpoint-<step>/`. Additional run tags or unsupported data fractions are placed under `custom/`.

See [CHECKPOINTS.md](CHECKPOINTS.md) for retained model files. Experimental limitations and sample overlaps are described in [IMPLEMENTATION.md](IMPLEMENTATION.md) and `results/reports/data-split-audit.json`. Historical checkpoints and their metadata archive are not included in a Git clone; saved CSVs remain available, and training creates new checkpoints.

Regression checks for the aspect correction: `python -m unittest discover -s tests -v`. Corrected GCN run tags are automatically prefixed with `aspectfix` so the old JSONs are preserved.
