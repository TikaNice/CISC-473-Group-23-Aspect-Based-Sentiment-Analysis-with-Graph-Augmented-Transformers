# Quickstart

Run commands from the repository root. In the code ZIP this directory is `final/code`. The project compares BERT with a BERT + dependency GCN model on SemEval-2014 Restaurant and Laptop.

## 1. Set up the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

For an existing installation, activate it with `source .venv/bin/activate`. The command above installs the CPU environment. The final results were recorded on CUDA. Training JSONs record the exact PyTorch build and device. Use `HF_HUB_OFFLINE=0` on a machine without the BERT files cached.

## 2. Prepare the data

Download the training data:

```bash
python scripts/download_data.py
```

Download the official Gold Test XML files from the pinned SemEval2014Task4 dataset revision:

```bash
mkdir -p data/gold
gold_base=https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/3fd43769112ef6517c263e303fbb7dc3002c35df/ABSA_Gold_TestData
curl -fL "$gold_base/Restaurants_Test_Gold.xml" -o data/gold/Restaurants_Test_Gold.xml
curl -fL "$gold_base/Laptops_Test_Gold.xml" -o data/gold/Laptops_Test_Gold.xml
```

Convert the XML files and verify the prepared inputs against the checksums from the recorded runs:

```bash
python scripts/prepare_gold.py
(cd data && sha256sum -c checksums.sha256)
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

This runs inference using each model's best Restaurant validation checkpoint. A new inference pass requires local checkpoints. Existing transfer results are reused when the prediction evidence validates. Use `--model bert` or `--model gcn` to select a model, and `--seeds 42` to evaluate one seed. Add `--overwrite` to replace existing cross-domain results.

### Epoch-budget comparison

Run separate experiments with budgets of 1, 2 and 4 epochs:

```bash
MODELS=bert bash scripts/run_epoch_sweep.sh
```

This exploratory arm is separate from the final comparisons. To select a subset:

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

### Run the main experiment matrix

After preparing the environment and data:

```bash
HF_HUB_OFFLINE=0 bash scripts/reproduce.sh --train
```

This resumes the grid of 60 training runs and six transfer evaluations, then rebuilds the results. On a fresh training workspace, complete the extension runs in Section 7 before this combined command: the final evidence check also requires those 24 records. The runner defaults to CPU. Set `ACCELERATE_USE_CPU=false` with a CUDA-enabled PyTorch installation to use a GPU. The grid checks saved result and prediction files before skipping training. Restaurant full-data BERT and dependency runs also require their best validation checkpoints. The epoch-budget arm is separate. Direct training commands rerun the requested experiment and can overwrite its files.

The trainers round each batch's longest input up to a multiple of eight while
retaining the 128-token limit. The result JSON records whether this is enabled.
Set `CISC473_DYNAMIC_PADDING=0` to use fixed 128-token batches for a run.

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
| `ablation_summary.csv` | Dependency-versus-identity results and paired source paths |
| `all_metrics.csv` | Individual run metrics |
| `summary_by_group.csv` | Group means, sample standard deviations and seed counts |
| `experiment_status.csv` | Completed and pending runs |

Missing runs are marked as pending in the summary tools. The full rebuild requires all canonical run records and predictions. Groups with one completed seed have no standard deviation. Smoke tests are excluded from grouped benchmark statistics.

For a comparison at one seed:

```bash
python -m scripts.summarize_corrected --seed 42
```

This writes `corrected_seed42.csv` and `corrected_seed42.md`. Change the seed to `43` or `44` as needed.

To rebuild the PDFs in the complete project workspace, install LaTeX with Beamer,
TikZ and `latexmk`, plus Poppler (`pdftotext` and `pdfinfo`). Put the report and
deck sources at `final/report/Group23_Final_Report.tex` and `mtp/Group23_Slides.tex`.
The PDFs also require the overlap and verified error analyses from that workspace.
From `final/code` run:

```bash
bash scripts/reproduce.sh --pdf
python -m scripts.qa_check
```

The PDF build uses the prepared data and saved predictions. It requires no model download.

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


## 7. Final experiment reproduction: seeds and complete execution order

### Recorded seeds and versions

| Setting | Value | Source |
|---|---|---|
| Training seeds | 42, 43, 44 | `src/result_paths.py`; per-run JSONs |
| Train/validation sentence-group split | 2026 | `DATA_SPLIT_SEED` in `src/train_bert.py` |
| Main training-subset sampling | 2026 | `DEFAULT_SUBSET_SEED` in `src/train_bert.py` |
| Alternative subset experiment | 2027; validation split stays 2026 | `scripts/run_robustness.py`, arm `subset2027` |
| Longer 10% training experiment | subset 2026; 30 epochs; seeds 42, 43, 44 | arm `updatematch10` |
| BERT revision | `86b5e0934494bd15c9632b12f734a8a67f723594` | `MODEL_REVISION` in `src/train_bert.py` |
| Restaurant training-data revision | `8885372fe73256f96bb60f65b550538ac5c26047` | `scripts/download_data.py` |
| Laptop training-data revision | `4d69e3de15751c40ac3829ccd31c3fa190f2068f` | `scripts/download_data.py` |

The trainers seed Python, NumPy and PyTorch with the training seed, and pass
`data_seed=2026` to Trainer. Training seeds do not resample the validation split
or labelled subsets. Report arithmetic means and sample standard deviations
across the three training seeds; these are not confidence intervals.
The `updatematch10` name denotes an approximate update-budget control: 30 epochs
at 10% give 1,260 Restaurant / 840 Laptop updates, versus 1,215 / 786 for
three full-data epochs. It is not an exact step-matched experiment.

The recorded final training environment used CUDA and PyTorch
`2.10.0a0+a36e1d39eb.nvinternal.main.41386253`. The CPU requirements in Section 1
are a separate environment, not an exact reconstruction of that NVIDIA build.
Exact historical environment reproduction requires the original container/image
from the training author; its image identifier is not recorded here.
Matching seeds alone does not guarantee identical scores across devices or builds.

### A. Verify and rebuild the committed results (no training)

Activate the environment and prepare/check the data as in Sections 1–2. On macOS,
use `(cd data && shasum -a 256 -c checksums.sha256)` if `sha256sum` is unavailable.
Then run:

```bash
python -m src.evidence
bash scripts/reproduce.sh
python -m scripts.summarize_robustness
python -m unittest discover -s tests -v
```

The evidence validator checks 90 canonical evaluations: 48 main in-domain,
12 identity, 12 pooling, 12 constituency and six transfer evaluations.
The robustness summarizer additionally validates its saved supplementary records.
These commands use saved predictions and need no model checkpoints or Stanza
model download. Rebuilding rewrites derived tables and analysis files, not training
results. PDF generation additionally requires the external report/slide sources
listed in Section 4; those sources are not included in this branch.

### B. Retrain the final experiments in a separate checkout

Use a separate checkout/output workspace for new training, preserving the published
records. Runners reuse validated artifacts, so running them over the submitted
results is a resume operation, not proof of fresh training. A clean experiment
workspace has prepared data but no previous run JSONs, predictions or checkpoints.
Do not use `--force` on the published-results workspace.

For the existing extension runners, use a CUDA-enabled environment: the accelerated
runner requires CUDA, and extension evidence currently expects `device=cuda:0`.
Do not install the CPU-pinned requirements over an existing CUDA environment.
Retain its PyTorch build and install the remaining dependencies from the requirements
file, plus `stanza==1.10.1`. For example, after activating a CUDA environment:

```bash
python -c "from pathlib import Path; Path('/tmp/group23-requirements-without-torch.txt').write_text(''.join(line for line in Path('requirements.txt').read_text().splitlines(keepends=True) if not line.startswith('torch==')))"
python -m pip install -r /tmp/group23-requirements-without-torch.txt
python -m pip install stanza==1.10.1
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA is required by the extension runner'"
export HF_HUB_OFFLINE=0
export ACCELERATE_USE_CPU=false
export CISC473_DYNAMIC_PADDING=1
export JOBS=1
export CUDA_WORKERS=1
```

One worker limits concurrent GPU memory usage; it does not change the configured
batch size or training epochs. Constituency preprocessing requires the frozen
parser resources used by the implementation:

```bash
export STANZA_RESOURCES_DIR="$PWD/stanza_resources"
python -c "import os, stanza; stanza.download('en', model_dir=os.environ['STANZA_RESOURCES_DIR'], processors={'tokenize':'combined','pos':'combined_charlm','constituency':'ptb3-revised_charlm'}, package=None, resources_version='1.10.0')"
python -m src.constituency_graph
```

This builds the training-text parse cache for both domains. The trainer builds
unlabelled test parses during final evaluation. Parser identity and phrase-graph
rules are defined in `src/constituency_graph.py`.

Run the following stages in order:

```bash
# 48 main in-domain + 12 identity training runs, followed by six transfers.
bash scripts/run_sweep.sh --runs-only

# Two domains x two budgets (10%, 100%) x three seeds per extension.
python -m scripts.run_accelerated_shard --extension pooling
python -m scripts.run_accelerated_shard --extension constituency

# 36 alternative-subset runs + 12 longer-10%-training runs.
python -m scripts.run_robustness subset2027 updatematch10

# Validate and rebuild summaries after all required runs exist.
python -m src.evidence
bash scripts/reproduce.sh
python -m scripts.summarize_robustness
```

This sequence covers 132 training runs and six transfer evaluations (138 total).
The optional BERT 1/2/4-epoch sweep is outside this final matrix.
`reproduce.sh --train` alone covers only the main/identity grid and transfer;
it does not launch the two extension grids or the robustness grids.

For inspection, one existing runner command takes the model, domain, fraction,
seed and graph type explicitly:

```bash
python -m scripts.run_one gcn restaurant 0.1 42 identity
```

The corresponding graph choices for the final comparison are `dependency`,
`identity`, `pooling` and `constituency`. Pooling removes the GCN layers;
identity retains the layers with self-loops only. All main and extension runs
use three epochs, batch size 8, learning rate 2e-5, weight decay 0.01,
maximum length 128 and validation-macro-F1 checkpoint selection.
