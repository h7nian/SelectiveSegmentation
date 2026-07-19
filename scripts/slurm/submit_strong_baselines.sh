#!/bin/bash
# Re-evaluate the ten established conditions with the inexpensive strong
# single-map baselines. M=2 supplies a common-score/risk join audit while
# avoiding the full M=32 boundary ladder. Each condition is one job.
#
# Usage: bash scripts/slurm/submit_strong_baselines.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

submit_condition() {
    local model=$1
    local dataset=$2
    local condition=$3
    local checkpoint=$4
    local batch_size=8
    if [[ "$dataset" == "fives" ]]; then
        batch_size=4
    fi

    sbatch --parsable \
        --job-name "selseg-base-$dataset-$condition" \
        scripts/slurm/binary_eval.sbatch \
        "$model" "$dataset" "$checkpoint" \
        --decision-threshold 0.5 --m-values 2 \
        --num-workers 4 --batch-size "$batch_size" \
        --output-dir outputs/binary_baselines
}

submit_condition clipseg pet clip-g -
submit_condition clipseg pet clip-t \
    outputs/binary_train/pet/clipseg/seed-0/checkpoint.pt
submit_condition deeplabv3 pet dl-e -
submit_condition deeplabv3 pet dl-t \
    outputs/binary_train/pet/deeplabv3/seed-0/checkpoint.pt

submit_condition clipseg kvasir clip-g -
submit_condition clipseg kvasir clip-t \
    outputs/binary_train/kvasir/clipseg/seed-0/checkpoint.pt
submit_condition deeplabv3 kvasir dl-t \
    outputs/binary_train/kvasir/deeplabv3/seed-0/checkpoint.pt

submit_condition clipseg fives clip-g -
submit_condition clipseg fives clip-t \
    outputs/binary_train/fives/clipseg/seed-0/checkpoint.pt
submit_condition deeplabv3 fives dl-t \
    outputs/binary_train/fives/deeplabv3/seed-0/checkpoint.pt
