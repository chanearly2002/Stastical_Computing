#!/usr/bin/env bash
# Re-run every experiment (≈ 1.5-2 h on 4 CPU cores). Results are resumable JSONL files in results/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-python3}"
./scripts/fetch_data.sh
$PY -m pytest -q tests
$PY src/run_ihdp_sweep.py --reps 0 50 --out results/ihdp_sweep.jsonl
$PY src/run_jobs.py --splits 0 10 --out results/jobs.jsonl
$PY src/run_synthetic.py --seeds 0 20 --out results/synthetic.jsonl
for q in 0.0 0.5 1.0; do
  $PY src/run_ihdp_sweep.py --reps 0 20 --q $q --alphas_wass 0.03 0.1 0.3 1 --alphas_mmd --out results/ihdp_imbalance.jsonl
done
$PY src/analyze_ihdp.py && $PY src/analyze_jobs.py && $PY src/analyze_synthetic.py && $PY src/analyze_imbalance.py
