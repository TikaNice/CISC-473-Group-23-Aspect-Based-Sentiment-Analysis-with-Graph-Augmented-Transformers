# CISC 473 starter (BERT only; not the project extension)

This is a **reference starter**, not validated experimental output. Review and understand the code; implement, run, and verify the experiments yourself. Credit/disclose AI assistance as required by the course. Check actual metrics, datasets and model behavior on your Mac.

1. Place `scripts/prepare_gold.py` in `group23/scripts/`, and `src/train_bert.py` and `src/__init__.py` in `group23/src/` (merge with local edits; do not overwrite your own code).
2. Download official gold test XML files into `group23/data/gold/` from Hugging Face `alexcadillon/SemEval2014Task4`.
3. Run `python scripts/prepare_gold.py`; inspect output labels and count.
4. Run `python -m src.train_bert --domain restaurant --smoke` to verify 64-sample training on MPS. This is **not** a final reportable benchmark.
5. Run `python -m src.train_bert --domain restaurant --epochs 3 --batch-size 8` for the full BERT baseline. It may take longer than you expect on M3 Pro.
6. Read the resulting JSON and inspect mistakes. Do not invent metrics. The script reports test metrics only after training and checkpoint selection on a separate validation split.

Dataset: `data/raw/restaurant/train.jsonl`, `data/raw/laptop/train.jsonl`, and `data/processed/<domain>/test_gold.jsonl`.

The above code is a Hugging Face pair-classification baseline. It does NOT implement the dependency-graph GCN, cross-domain, or three-seed experiment matrix; those are future work. The `span` ordinal and offsets are not used as position features, which could be a limitation for multiple identically named target occurrences. No PyTorch Geometric dependency needed for this baseline.

On MPS failure or OOM, reduce `--batch-size` to 4 and rerun the smoke test. Confirm exact `transformers` version and traceback before adjusting code.
