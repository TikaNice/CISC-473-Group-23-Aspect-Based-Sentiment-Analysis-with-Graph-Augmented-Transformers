# Implementation and evaluation notes

## Scope and provenance

This is a project-specific BERT baseline and a simplified BERT + dependency GCN extension for aspect sentiment classification with given aspect terms. It is not aspect extraction, and it is not a lightly modified copy of a paper's author implementation. ChatGPT/Codex assisted with implementation, debugging and project documentation; contributors should describe the actual assistance in the report's AI Usage section.

| Component | Actual implementation | Relation to paper code |
|---|---|---|
| BERT baseline | Hugging Face `AutoModelForSequenceClassification`, sentence–aspect pairs, project-specific Trainer/data pipeline | Reuses the library model and pretrained weights, not the BERT-PT author training code; no review-domain post-training |
| GCN extension | BERT word features, one dependency graph, two GCN layers, aspect mean pooling and CLS concatenation | Inspired by graph ABSA methods, not an ASGCN or DualGCN reproduction |
| Preprocessing | SetFit-formatted train JSONL, official Gold XML conversion, sentence-group validation and sampling | Project-specific data pipeline |
| Experiments | Three seeds, data budgets, epoch budgets, Restaurant → Laptop inference | Project-specific evaluation pipeline |

Author repositories checked during the audit:

- [ASGCN](https://github.com/GeneZC/ASGCN/blob/master/models/asgcn.py): BiLSTM encoder, position weighting, aspect masking and attention-based readout. The current project replaces or omits these mechanisms.
- [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA/blob/main/DualGCN/models/dualgcn_bert.py): syntactic and semantic branches, mutual biaffine interaction and optional graph regularization. None of those dual-graph mechanisms is present here.
- [BERT-PT](https://github.com/howardhsu/BERT-for-RRC-ABSA): domain post-training and its own fine-tuning/data stack. Here BERT is initialized directly from `bert-base-uncased`.

The local reference repositories are not imported by the training code. Architecture-level differences establish that this is not a lightweight adaptation; they do not establish the authorship of every individual line. Retain relevant acknowledgements for any source material actually reused.

## Model and training protocol

The baseline tokenizes `(sentence, aspect)` as a pair, pads/truncates to 128 tokens and uses the Hugging Face BERT sequence-classification head, including its pooled CLS representation.

The GCN tokenizes a spaCy word sequence and the aspect as a pair. Sentence subwords are averaged into word nodes. Dependency arcs become undirected edges, with self-loops and symmetric degree normalization. Two linear graph-convolution/ReLU/dropout layers update word features. Aspect-node features are averaged and concatenated with the raw BERT CLS hidden state for classification.

Both models are initialized independently from pretrained BERT and fine-tuned end-to-end. The GCN run does not start from the already fine-tuned baseline checkpoint. Baseline and GCN also differ in tokenization and classification heads, so this comparison does not isolate a causal effect of dependency edges alone.

Runs use learning rate 2e-5, weight decay 0.01 and batch size 8. Baselines and low-resource runs have a three-epoch budget; independent epoch runs use one, two or four epochs. Each selects the best checkpoint by validation Macro-F1. Three run seeds are 42, 43 and 44. The sentence-group validation split and reduced-data sampling use seed 2026 and remain fixed across run seeds.

## Data

Training JSONL comes from the [SetFit SemEval Restaurant dataset](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) and corresponding Laptop dataset. Their raw test splits have empty sentiment labels. Evaluation therefore uses official Gold Test XML from [SemEval2014Task4](https://huggingface.co/datasets/alexcadillon/SemEval2014Task4), converted by `scripts/prepare_gold.py`.

Conflict labels are excluded during training/evaluation reads. Actual train/validation/test counts are Restaurant 3240/362/1120 and Laptop 2092/221/638. Reduced budgets sample whole sentence groups, not an exact percentage of aspect-label rows.

## Historical GCN aspect-alignment defect and correction

The historical `src/train_gcn.py::find_aspect` lowercased both sentence and aspect before applying the SetFit occurrence ordinal. This can select a different occurrence than the original annotation. For example, lowercase `bar` in `Bartender ... bar` can be aligned to the beginning of `Bartender`.

A case-sensitive occurrence lookup was compared with the original official training XML, downloaded for this audit. It matches the annotated offsets for every three-class training row. Relative to that lookup, the legacy code changes 7 Restaurant and 9 Laptop row offsets. Of these, 7 Restaurant and 8 Laptop offsets do not match any official offset for the same sentence/aspect/label; the remaining Laptop row selects another annotated occurrence. Thus zero skipped rows does not establish correct alignment.

The invalid legacy GCN result JSONs and derived score/curve rows have been removed from the active project. An external archive preserves the audit trail. Corrected case-sensitive lookup was verified against every training XML annotation; the seed-42 full-data, reduced-data and cross-domain runs have been recomputed with the aspectfix revision. BERT records remain unchanged. Corrected GCN seeds 43/44 and all corrected epoch-budget extensions remain pending. Gold evaluation uses explicit XML offsets; BERT does not use the faulty fallback. Historical model weights/Trainer metadata that remain locally are excluded from result and curve generation.

The detailed audit is in `results/reports/aspect-alignment-audit.json`.

## Other limitations

- Train and validation sentence groups are disjoint. One Restaurant sentence/aspect/label occurs in both actual training and Gold Test, and one Laptop example occurs in validation and Gold Test. See `results/reports/data-split-audit.json`. The original data and historical scores have not been changed.
- Aggregate GCN sample counts describe input records; future runs that skip encoding errors need to report actual evaluated counts as well. The saved formal logs reported no encoding exclusions, which is separate from the alignment defect above.
- Existing test results were inspected before later exploratory comparisons. Do not claim a completely untouched test set or select epoch budgets from test scores.
- The current matrix includes reduced-data in-domain runs and full-data cross-domain runs, not every combination of reduced data and cross-domain testing.
- No controlled identity/random/no-graph comparison, parser-noise study, MAMS evaluation or category-based error analysis is claimed here.
- Dependency versions, dataset revisions and pretrained model revisions are not fully locked. `requirements-current.txt` is an audit-time environment snapshot; new-machine installation and full retraining have not been verified.

## Validation performed

Syntax, result paths, JSON fields and three-seed group completeness were checked. Additional runtime checks cover Gold XML regeneration, all current three-class rows passing graph encoding, and a small saved-model forward/backward computation. These checks establish executable behavior, not equivalence to paper results or freedom from methodological defects.

The optional Smoke files are excluded from formal statistics. Summary generation uses available valid runs and records missing corrected runs as pending. Single-seed GCN groups do not have a standard deviation; BERT groups retain their three-seed statistics. `transformers>=4.46` is required because the scripts pass `processing_class` to Trainer.

The repository can be committed as an auditable preliminary experiment snapshot. Only valid BERT and corrected single-seed GCN results are retained for analysis. They use the project protocol rather than the 2019 author protocol; corrected multi-seed statistics remain pending.
