#!/bin/bash -l
#SBATCH --job-name=selseg-asym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage: sbatch --partition <cpu> --account ssafo scripts/slurm/asymmetric.sbatch
# Does the constraint t_lo + t_hi = 1 cost accuracy, and can extra NODES buy back
# what freeing the thresholds would, without the hyperparameters? Oracle grid
# search — far too heavy for a login node. CPU-only, no GPU.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python scripts/asymmetric_nodes.py
