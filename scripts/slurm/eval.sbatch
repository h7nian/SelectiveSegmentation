#!/bin/bash -l
#SBATCH --job-name=selseg-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage: sbatch [slurm options] scripts/slurm/eval.sbatch MODEL DATASET \
#            [extra selectseg.evaluate args, e.g. --checkpoint PATH]
# Must be submitted from the repository root (uses SLURM_SUBMIT_DIR), and
# outputs/slurm/ must exist first (mkdir -p outputs/slurm) — SLURM cannot
# start a job whose --output directory is missing.

set -euo pipefail

MODEL=$1
DATASET=$2
shift 2

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.evaluate --model "$MODEL" --dataset "$DATASET" "$@"
