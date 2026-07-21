#!/bin/bash -l
#SBATCH --job-name=selseg-science-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# One dataset component per independent CPU job. The submitter supplies the
# complete CPU candidate partition list and the private account.

set -euo pipefail

DATASET=$1
OUTPUT=$2

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.scientific_inputs \
    build-dataset \
    --dataset "$DATASET" \
    --data-root data \
    --output "$OUTPUT"
