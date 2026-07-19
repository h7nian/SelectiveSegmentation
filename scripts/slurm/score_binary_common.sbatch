#!/bin/bash -l
#SBATCH --job-name=selseg-common-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Score the M-independent analyzer fields for exactly one frozen artifact.
# The submitter supplies the immutable lock/artifact hashes and the one locked
# deployment threshold.  This job performs no inference and imports no Torch.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.score_binary_common \
  --score-workers "$SLURM_CPUS_PER_TASK" \
  --max-pending-scores "$((2 * SLURM_CPUS_PER_TASK))" \
  "$@"
