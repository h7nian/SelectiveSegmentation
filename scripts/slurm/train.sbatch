#!/bin/bash -l
#SBATCH --job-name=selseg-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=saffo-a100
#SBATCH --account=ssafo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage: sbatch [slurm options] scripts/slurm/train.sbatch MODEL DATASET \
#            [extra selectseg.train args, e.g. --epochs 80]
# Must be submitted from the repository root (uses SLURM_SUBMIT_DIR), and
# outputs/slurm/ must exist first (mkdir -p outputs/slurm) — SLURM cannot
# start a job whose --output directory is missing.

set -euo pipefail

MODEL=$1
DATASET=$2
shift 2

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.train --model "$MODEL" --dataset "$DATASET" "$@"
