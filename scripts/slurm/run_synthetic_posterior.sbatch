#!/bin/bash -l
#SBATCH --job-name=selseg-synthetic-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Exactly one known-posterior (coupling, sharpness, morphology, replicate)
# cell.  This workflow is CPU-only and never stores posterior mask samples.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.synthetic_posterior "$@"
