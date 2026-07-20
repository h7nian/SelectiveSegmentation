#!/bin/bash -l
#SBATCH --job-name=selseg-seed-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Analyze all three training seeds only after the strict planner has validated
# the 20 seed-extension assemblies and the locked seed-0 analysis.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m scripts.analyze_binary_seed_extension "$@"
