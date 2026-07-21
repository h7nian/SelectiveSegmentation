#!/bin/bash -l
#SBATCH --job-name=selseg-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage: sbatch scripts/slurm/analyze.sbatch
# Aggregates the selective-eval JSONL files into the summary + figures.
# Must be submitted from the repository root (uses SLURM_SUBMIT_DIR), and
# outputs/slurm/ must exist first (mkdir -p outputs/slurm) — SLURM cannot
# start a job whose --output directory is missing. CPU-only.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python scripts/analyze_selective.py
python scripts/plot_risk_coverage.py
