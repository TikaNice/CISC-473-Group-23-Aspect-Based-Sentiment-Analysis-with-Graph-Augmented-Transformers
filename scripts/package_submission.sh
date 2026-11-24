#!/usr/bin/env bash
# Assemble a submission bundle from the checked PDFs and saved evidence.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FINAL="$ROOT/final"
CODE="$FINAL/code"
STAMP=$(date +%Y%m%d)
OUT="$FINAL/submission"
REPORT="$FINAL/report"
DECK="$ROOT/mtp"

(
  cd "$CODE"
  if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=${PY:-python3}; fi
  "$PY" -B scripts/qa_check.py
)

STAGE=$(mktemp -d "$FINAL/.submission.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
cp "$REPORT/Group23_Final_Report.pdf" "$DECK/Group23_Slides.pdf" "$STAGE/"
VIDEO="$DECK/recording/Group23_Presentation.mp4"
if [[ -f "$VIDEO" ]]; then cp "$VIDEO" "$STAGE/"; fi
mkdir -p "$STAGE/report/tables" "$STAGE/report/figures"
cp "$REPORT/Group23_Final_Report.tex" "$STAGE/report/"
cp "$REPORT"/tables/*.tex "$STAGE/report/tables/"
cp "$REPORT"/figures/*.png "$STAGE/report/figures/"
ZIP="$STAGE/Group23_Code_${STAMP}.zip"
git -C "$CODE" ls-files -z -- README.md CHECKPOINTS.md QUICKSTART.md IMPLEMENTATION.md \
  'requirements*.txt' src scripts tests data results |
(
  cd "$ROOT"
  mapfile -d '' -t code_files
  # Include the prepared inputs that live outside version control.
  zip -q -MM "$ZIP" "${code_files[@]/#/final/code/}" \
    final/code/data/gold/{Laptops,Restaurants}_Test_Gold.xml \
    final/code/data/raw/{laptop,restaurant}/{train,test}.jsonl \
    final/code/data/processed/{laptop,restaurant}/test_gold.jsonl \
    final/code/data/processed/constituency_stanza_1.10.1/{laptop,restaurant}.json \
    final/report/Group23_Final_Report.tex final/report/tables/*.tex final/report/figures/*.png \
    mtp/{Group23_Slides,numbers,ablation_slide,budget_slide,failures_slide}.tex mtp/VIDEO_SCRIPT.md mtp/figures/*.png \
    -x '*/.venv/*' '*/__pycache__/*' '*/checkpoints/*' '*/archive/*' \
    '*/.build/*' '*/.pytest_cache/*' '*.py[co]' '*.lock' '*.log' '*.aux' '*.fls' '*.fdb_latexmk' \
    '*.out' '*.synctex.gz' '*.nav' '*.snm' '*.toc' '*.zip' '*.tar' '*.tar.*'
)
zip -T "$ZIP"

{
  echo "Group 23 — CISC 473 final submission bundle"
  echo "Built: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Git base: $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unavailable)"
  if [[ -n "$(git -C "$CODE" status --porcelain --untracked-files=no)" ]]; then
    echo 'Tracked files: modified. The ZIP SHA-256 identifies the packaged snapshot.'
  fi
  echo
  echo "SHA-256:"
  (cd "$STAGE" && sha256sum ./*.pdf ./*.zip report/*.tex report/tables/*.tex report/figures/*.png)
  if [[ -f "$STAGE/Group23_Presentation.mp4" ]]; then
    (cd "$STAGE" && sha256sum ./Group23_Presentation.mp4)
  else
    echo
    echo 'Group23_Presentation.mp4 not present at build time.'
  fi
} > "$STAGE/MANIFEST.txt"

if [[ -e "$OUT" ]]; then
  OUT="$FINAL/submission-$(date +%Y%m%d-%H%M%S)"
fi
mv -T -- "$STAGE" "$OUT"
cat "$OUT/MANIFEST.txt"
echo "Bundle: $OUT"
