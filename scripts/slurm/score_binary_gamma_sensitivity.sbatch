#!/bin/bash -l
#SBATCH --job-name=selseg-gamma-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Score exactly one lock-listed (condition, gamma) experiment.  This is a CPU
# job: the planner assigns agsmall, amdsmall, or msismall and requests no GPU.
# The scorer streams the frozen artifact once and writes common quantities and
# all three M=32 indexed scores into one content-addressed auxiliary artifact.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.score_binary_gamma_sensitivity \
  --score-workers "$SLURM_CPUS_PER_TASK" \
  --max-pending-scores "$((2 * SLURM_CPUS_PER_TASK))" \
  "$@"
