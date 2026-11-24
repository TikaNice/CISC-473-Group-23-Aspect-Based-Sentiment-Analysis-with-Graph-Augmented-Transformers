# Quickstart

Run all commands from the project root. The project compares BERT with a BERT + dependency GCN model on SemEval-2014 Restaurant and Laptop.

## 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For an existing installation, activate it with `source .venv/bin/activate`. The first training run downloads `bert-base-uncased` if it is not already cached. `requirements-current.txt` records the versions used in the local experiment environment.

## 2. Prepare the data

Download the training data:

```bash
python scripts/download_data.py
```

Download the official Gold Test XML files from [SemEval2014Task4](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4), and save them as:

```text
data/gold/Restaurants_Test_Gold.xml
data/gold/Laptops_Test_Gold.xml
```

Convert the XML files:

```bash
python scripts/prepare_gold.py
```

Training reads `data/raw/<domain>/train.jsonl`; evaluation reads `data/processed/<domain>/test_gold.jsonl`. The task uses positive, neutral and negative labels, excluding conflict. The raw test files downloaded in the first step have empty sentiment labels and are not used for scoring.

## 3. Run experiments

Both models use a learning rate of `2e-5`, weight decay of `0.01`, batch size of `8`, maximum sequence length of `128`, and no warmup. Full-data and low-resource runs train for up to `3` epochs and select the best checkpoint by validation Macro-F1. Run seeds are `42`, `43` and `44`; the validation split and sentence-group sampling use seed `2026`.

### Full-data comparison

Run both models on both datasets with all three seeds:

```bash
FRACTIONS=1 bash scripts/run_low_resource.sh
```

To train a single model:

```bash
python -m src.train_bert --domain restaurant --epochs 3 --batch-size 8 --seed 42
python -m src.train_gcn --domain restaurant --epochs 3 --batch-size 8 --seed 42
```

### Low-resource comparison

Run both models with 10%, 25% and 50% of the training sentence groups:

```bash
FRACTIONS="0.1 0.25 0.5" bash scripts/run_low_resource.sh
```

Use the runner's variables to select a subset. For example:

```bash
DOMAINS=laptop MODELS=gcn SEEDS="43 44" FRACTIONS=0.25 bash scripts/run_low_resource.sh
```

### Cross-domain evaluation

Evaluate the Restaurant-trained models on the Laptop Gold Test set:

```bash
python -m scripts.eval_cross_domain --model all --seeds 42 43 44
```

This runs inference using each model's best Restaurant validation checkpoint. The checkpoints must be available locally; result JSONs alone are not enough. Use `--model bert` or `--model gcn` to select a model, and `--seeds 42` to evaluate one seed. Add `--overwrite` to replace existing cross-domain results.

### Epoch-budget comparison

Run separate experiments with budgets of 1, 2 and 4 epochs:

```bash
bash scripts/run_epoch_sweep.sh
```

The 3-epoch results come from the full-data comparison. To select a subset:

```bash
DOMAINS=laptop MODELS=bert SEEDS=42 EPOCHS="1 2" bash scripts/run_epoch_sweep.sh
```

Each run still selects its best checkpoint by validation Macro-F1, so the selected checkpoint may come from an earlier epoch.

### GCN experiment runner

This runner uses existing matching BERT records and runs GCN full-data, cross-domain and low-resource experiments, refreshing reports after each result:

```bash
SEEDS="43 44" bash scripts/run_midterm_seeds.sh
```

It activates `.venv` automatically. Set `SEEDS="42 43 44"` to cover all three seeds. Run only one training runner at a time.

### Run the complete experiment matrix

After preparing the environment and data:

```bash
bash scripts/reproduce.sh
```

This includes full-data, low-resource and epoch-budget training, cross-domain evaluation, and report generation. Batch runners skip existing result JSONs. Direct training commands rerun the requested experiment and can overwrite its files.

## 4. Read and rebuild reports

Start with [results/INDEX.md](results/INDEX.md) for available experiments. To rebuild the aggregate tables without training:

```bash
python -m scripts.summarize_results
python -m scripts.summarize_sweeps
```

Reports are saved in `results/reports/`:

| File | Contents |
|---|---|
| `baseline_summary.csv` | Full-data comparison |
| `low_resource_summary.csv` | Training-data budgets, including full data |
| `cross_domain_summary.csv` | Restaurant-to-Laptop evaluation |
| `epoch_sweep_summary.csv` | Epoch-budget comparison |
| `all_metrics.csv` | Individual run metrics |
| `summary_by_group.csv` | Group means, sample standard deviations and seed counts |
| `experiment_status.csv` | Completed and pending runs |

Missing runs are marked as pending. Groups with one completed seed have no standard deviation. Smoke tests are excluded from grouped benchmark statistics.

For a comparison at one seed:

```bash
python -m scripts.summarize_corrected --seed 42
```

This writes `corrected_seed42.csv` and `corrected_seed42.md`. Change the seed to `43` or `44` as needed.

## 5. Output files

- `results/`: experiment JSONs, reports and diagnostic records. Current GCN runs carry the `aspectfix` tag and are stored under `custom/<domain>/gcn/`; cross-domain results use `cross_domain/restaurant_to_laptop/`.
- `checkpoints/`: locally saved models and training metadata, excluded from Git. See [CHECKPOINTS.md](CHECKPOINTS.md) for storage details.
- `logs/`: training output, excluded from Git.

For model structure, data splits and evaluation limitations, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 6. Optional checks

Add `--smoke` to a direct training command for a short pipeline check. Run the aspect-alignment regression checks with:

```bash
python -m unittest discover -s tests -v
```

To extract validation history from retained epoch-4 checkpoints or the local Trainer metadata archive:

```bash
python -m scripts.extract_epoch_validation_curve
```

The extracted history may be incomplete if the required checkpoints or metadata are unavailable.
