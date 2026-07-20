#!/bin/bash -l
#SBATCH --job-name=selseg-seed-train-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Train exactly one lock-listed (dataset, target architecture, seed) cell.
# The submitter supplies one validated private partition, account, and A100
# GRES tuple.  They are intentionally absent here so conflicting directives
# cannot silently override the balanced 10/10 plan.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.binary_seed_extension "$@"
