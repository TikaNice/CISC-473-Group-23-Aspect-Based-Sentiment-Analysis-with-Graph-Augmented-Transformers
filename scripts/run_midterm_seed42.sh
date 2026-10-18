#!/usr/bin/env bash
# Preserve the seed-42 entry point; use the common runner and report refresh.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=42 bash scripts/run_midterm_seeds.sh
