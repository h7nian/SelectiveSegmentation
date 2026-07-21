#!/bin/bash -l
#SBATCH --job-name=selseg-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage: sbatch [slurm options] scripts/slurm/tests.sbatch
# Must be submitted from the repository root (uses SLURM_SUBMIT_DIR), and
# outputs/slurm/ must exist first (mkdir -p outputs/slurm) — SLURM cannot
# start a job whose --output directory is missing.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

pytest tests/ -v
