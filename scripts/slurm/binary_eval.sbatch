#!/bin/bash -l
#SBATCH --job-name=selseg-binary-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=saffo-a100,apollo_agate
#SBATCH --account=ssafo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Usage:
#   sbatch scripts/slurm/binary_eval.sbatch MODEL DATASET CHECKPOINT [extra args]
# Use "-" as CHECKPOINT for the pretrained (non-fine-tuned) condition.

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
  --score-workers 8
  --max-pending-scores 16
)
if [[ "$CHECKPOINT" != "-" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

python -m selectseg.binary_eval "${ARGS[@]}" "$@"
