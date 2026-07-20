#!/bin/bash -l
#SBATCH --job-name=selseg-cardinality-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Compute the exact shared-threshold cardinality/PIT diagnostic for exactly one
# lock-listed frozen condition.  This is CPU-only and performs no model inference.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m selectseg.cardinality_diagnostics "$@"
