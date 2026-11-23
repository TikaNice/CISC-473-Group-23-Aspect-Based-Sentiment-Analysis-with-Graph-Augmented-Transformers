# Implementation and evaluation

## Task and model scope

The task is three-class sentiment classification for a supplied aspect term. Aspect extraction and end-to-end ABSA are outside the current implementation.

The BERT baseline uses Hugging Face AutoModelForSequenceClassification initialized from bert-base-uncased. Sentence-aspect pairs are padded or truncated to 128 tokens, and the pooled CLS representation feeds the sequence-classification head.

The GCN extension independently initializes AutoModel from bert-base-uncased. Sentence subwords are averaged into spaCy word nodes. Dependency arcs form an undirected graph with self-loops and symmetric degree normalization. Two graph-convolution, ReLU and dropout layers update the node features. Mean-pooled aspect nodes are concatenated with the raw BERT CLS state. Dropout applies to the concatenation before three-way classification. GCN dropout is 0.1.

Both models are fine-tuned end-to-end. The GCN does not load the fine-tuned BERT baseline. Different tokenization, pooling and classification heads accompany the graph extension, so the model comparison does not measure the isolated causal effect of dependency edges.

## Training and evaluation protocol

| Setting | Value |
|---|---|
| Optimizer | Hugging Face Trainer AdamW default |
| Learning rate | 2e-5, linear decay, no warmup |
| Weight decay | 0.01 |
| Batch size | 8 |
| Main training budget | 3 epochs |
| Maximum sequence length | 128 |
| Loss | Unweighted three-class cross-entropy |
| Checkpoint selection | Highest validation Macro-F1 |
| Training seeds | 42, 43, 44 |
| Split and sampling seed | 2026 |
| Metrics | Accuracy and unweighted Macro-F1 over negative, neutral and positive |

The same split and sampled sentence groups are used across models and training seeds. Three-seed summaries use the arithmetic mean and sample standard deviation. Model selection uses validation data; Restaurant-to-Laptop transfer uses the Restaurant validation-selected full-data checkpoint and no Laptop adaptation.

## Data and preprocessing

Training records come from the SetFit-formatted SemEval Restaurant and Laptop datasets. The raw test records have empty sentiment labels; evaluation uses official SemEval-2014 Gold Test XML converted by scripts/prepare_gold.py. Conflict labels are excluded.

| Domain | Training | Validation | Gold Test | 10% training | 25% training | 50% training |
|---|---:|---:|---:|---:|---:|---:|
| Restaurant | 3240 | 362 | 1120 | 335 | 794 | 1604 |
| Laptop | 2092 | 221 | 638 | 221 | 527 | 1068 |

Sentence groups use case-folded, trimmed text. Training and validation groups are disjoint. Reduced budgets select whole sentence groups, so aspect-record counts are approximate fractions of the full training pool. No data augmentation, class weighting or resampling is applied.

## Code provenance and published comparisons

The local reference repositories are not imported by the training pipeline. The implementation uses library BERT weights with project-specific preprocessing, training and reporting. It does not reproduce ASGCN's BiLSTM, position weighting and attention readout; DualGCN's syntactic/semantic branches, biaffine interaction and regularizers; or BERT-PT's review-domain post-training.

- [ASGCN](https://aclanthology.org/D19-1464/): aspect-specific dependency GCN with a BiLSTM encoder.
- [DualGCN](https://aclanthology.org/2021.acl-long.494/): complementary syntactic and semantic graph modeling.
- [BERT-PT](https://aclanthology.org/N19-1242/): review-domain and task-oriented post-training before downstream fine-tuning.

The published ASGCN-DG and vanilla BERT rows are contextual reference points. The final report describes the differences between the published protocols and the in-house comparison.

## Aspect-alignment correction

The original GCN fallback lowercased sentence and aspect text before interpreting the occurrence ordinal. Case-sensitive occurrence lookup matches all 5915 three-class official training annotations. The old lookup shifted 7 Restaurant and 9 Laptop offsets. Gold Test evaluation uses explicit XML offsets. For a record without offsets, the fallback trusts the source ordinal. A valid but incorrect ordinal can select a different occurrence. The encoder rejects truncation of any sentence-side aspect wordpiece.

In the archived MPS comparison, affected GCN experiments were retrained for seeds 42, 43 and 44 at full, 10%, 25% and 50% budgets; cross-domain inference was repeated. Current GCN records carry the aspectfix tag and case_sensitive_ordinal_v1 marker. Invalid historical GCN JSONs, score summaries and curve rows are excluded from the active results. Details are stored in [results/audit/aspect-alignment-audit.json](results/audit/aspect-alignment-audit.json).

## Completed experiments and validation

The final CUDA grid comprises 48 in-domain model runs, 12 identity controls and six Restaurant-to-Laptop evaluations. The archived MPS runs and optional 1/2/4-epoch experiments are separate. Smoke records are excluded from benchmark averages.

Validation covers aspect-alignment regression checks, valid JSON metrics, expected seed groups, Gold XML regeneration, graph encoding and saved-model forward/backward checks. The rebuild recomputes test accuracy and macro-F1 from stored predictions for each run and checks them against that run’s result JSON. These checks do not establish equivalence to an author implementation.

## Limitations

- One Restaurant sentence/aspect/label overlaps between actual training and Gold Test, and one Laptop example overlaps between validation and Gold Test. The original data are retained; details are in [results/audit/data-split-audit.json](results/audit/data-split-audit.json).
- Test results were viewed during exploratory analysis. Epoch-budget selection from test scores is not part of the protocol.
- Reduced-data in-domain experiments and full-data cross-domain experiments are separate; their full Cartesian product has not been evaluated.
- The identity control covers 10% and full data in-domain. A transfer-specific identity control, parser-noise analysis and MAMS evaluation are outside this comparison.
- Three seeds measure training variation on one fixed split and fixed budget sample. Reported SD is not a confidence interval or significance test.
- BERT initialization and the two SetFit training data downloads use pinned revisions. Seeds fix random inputs, with possible numerical variation across execution environments.
