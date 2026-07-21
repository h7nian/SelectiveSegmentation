#!/bin/bash -l
#SBATCH --job-name=selseg-freeze-maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=saffo-a100,apollo_agate
#SBATCH --account=ssafo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Freeze one model/dataset condition as one independent Slurm job.
# Usage:
#   sbatch scripts/slurm/freeze_binary_maps.sbatch MODEL DATASET CHECKPOINT [args]
# Use "-" as CHECKPOINT for a pretrained (non-fine-tuned) condition.

set -euo pipefail

MODEL=$1
DATASET=$2
CHECKPOINT=$3
shift 3

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

ARGS=(
  --model "$MODEL"
  --dataset "$DATASET"
  --require-cuda
)
if [[ "$CHECKPOINT" != "-" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

python -m selectseg.freeze_binary_maps "${ARGS[@]}" "$@"
