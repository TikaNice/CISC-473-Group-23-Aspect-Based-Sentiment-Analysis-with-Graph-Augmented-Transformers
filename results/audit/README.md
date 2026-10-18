# Engineering diagnostics

This directory contains historical implementation and data checks. Benchmark metrics and experiment status are stored in ../reports/.

| File | Scope |
|---|---|
| aspect-alignment-audit.json | Official-XML verification of case-sensitive aspect lookup |
| data-split-audit.json | Split sizes, budget sizes and original-data overlaps |
| code-runtime-audit.json | Historical encoding and model runtime checks |
| reproducibility_hashes.csv | Historical audit-time file fingerprints |
| training_log_audit.csv | Retained BERT log inventory |

Runtime checks alone do not validate aspect alignment. Historical fingerprints are not current integrity hashes. Old GCN weights and archived Trainer states are excluded from active score and curve generation.
