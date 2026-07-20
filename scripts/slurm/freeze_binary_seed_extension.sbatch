#!/bin/bash -l
#SBATCH --job-name=selseg-seed-freeze-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Freeze exactly one post-lock target-model seed checkpoint.  This wrapper is
# usable only after the complete 20-checkpoint lock exists.  The submitter
# supplies the validated private partition/account/A100 profile explicitly.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.binary_seed_extension "$@"
