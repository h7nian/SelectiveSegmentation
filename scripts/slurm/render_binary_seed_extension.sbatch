#!/bin/bash -l
#SBATCH --job-name=selseg-seed-render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m scripts.render_binary_seed_extension "$@"
