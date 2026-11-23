#!/usr/bin/env bash
# Reproduce the reported tables from saved run records (seconds, no model).
#
#   bash scripts/reproduce.sh              rebuild tables, figures, index
#   bash scripts/reproduce.sh --pdf        ... and compile the report and deck
#   bash scripts/reproduce.sh --train      rerun the experiments first (hours, CPU)
#
# Flags can be combined: --train --pdf reruns the grid and then rebuilds and
# recompiles everything from the new records.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN=0
PDF=0
for arg in "$@"; do
  case "$arg" in
    --train) TRAIN=1 ;;
    --pdf)   PDF=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=${PY:-python3}; fi

if [[ $TRAIN -eq 1 ]]; then
  bash scripts/run_sweep.sh --runs-only
fi

for f in data/raw/restaurant/train.jsonl data/raw/laptop/train.jsonl \
         data/processed/restaurant/test_gold.jsonl data/processed/laptop/test_gold.jsonl; do
  [[ -f "$f" ]] || { echo "Missing $f. Follow QUICKSTART.md to prepare the datasets." >&2; exit 1; }
done
"$PY" -m src.evidence
"$PY" -m scripts.summarize_results
"$PY" -m scripts.summarize_sweeps
"$PY" -m scripts.summarize_ablations
"$PY" -m scripts.summarize_extensions
"$PY" -m scripts.audit_saved_evaluation
"$PY" -m scripts.extract_epoch_validation_curve
"$PY" -m scripts.tag_errors --fraction 1 --max-examples 8
"$PY" -m scripts.build_report_tables
"$PY" -m scripts.write_readme_tables
"$PY" -m scripts.make_figures
for seed in 42 43 44; do "$PY" -m scripts.summarize_corrected --seed "$seed"; done
"$PY" -m scripts.write_index
echo "Tables rebuilt in results/reports and results/error_analysis.md."

if [[ $PDF -eq 1 ]]; then
  # Refuse to compile against a missing figure rather than shipping a broken PDF.
  for p in ../report/figures/data_efficiency.png ../report/figures/confusion_transfer.png \
           ../report/figures/graph_control.png; do
    [[ -f "$p" ]] || { echo "Missing figure $p; fix the results, do not fake it." >&2; exit 1; }
  done
  # Generated assets are rewritten even when their contents match the previous build.
  # Recompile so the PDF passes the submission freshness check on repeated runs.
  (cd ../report && latexmk -g -pdf -interaction=nonstopmode -halt-on-error Group23_Final_Report.tex)
  (cd ../../mtp && latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir=.build Group23_Slides.tex \
     && cp .build/Group23_Slides.pdf .)
  "$PY" -m scripts.qa_check --build-only
  echo "Compiled report/Group23_Final_Report.pdf and mtp/Group23_Slides.pdf."
fi
