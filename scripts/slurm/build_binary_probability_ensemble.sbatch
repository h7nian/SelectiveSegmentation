#!/bin/bash -l
#SBATCH --job-name=selseg-ensemble-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Build exactly one dataset/model probability-ensemble artifact. The caller
# supplies the immutable source-lock hash and one condition; no job arrays or
# multiple experimental cells are accepted by this wrapper.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.binary_probability_ensemble "$@"
