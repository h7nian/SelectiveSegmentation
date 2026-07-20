#!/bin/bash -l
#SBATCH --job-name=selseg-m128-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Score one lock-listed condition at M=128 on CPUs.  The wrapper requests no
# GPU/GRES resource; the submission planner assigns exactly one of agsmall,
# amdsmall, or msismall to each independent condition job.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.score_binary_m128_auxiliary \
  --score-workers "$SLURM_CPUS_PER_TASK" \
  --max-pending-scores "$((2 * SLURM_CPUS_PER_TASK))" \
  "$@"
