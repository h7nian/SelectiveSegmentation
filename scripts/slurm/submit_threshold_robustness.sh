#!/bin/bash
# Evaluate the fixed model matrix at deployment thresholds 0.3 and 0.7,
# then run the dense M=128 reference at gamma=0.5. Gamma=0.5/M<=32 is the
# primary experiment in outputs/binary_final. Every condition is one job.
#
# Usage: bash scripts/slurm/submit_threshold_robustness.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/slurm

submit_condition() {
    local gamma=$1
    local tag=$2
    local model=$3
    local dataset=$4
    local condition=$5
    local checkpoint=$6
    local output_dir="outputs/binary_thresholds_matched/gamma$tag"
    local job_name="selseg-g${tag}m-$dataset-$condition"
    local batch_size=8
    if [[ "$dataset" == "fives" ]]; then
        batch_size=4
    fi

    sbatch --parsable \
        --job-name "$job_name" \
        scripts/slurm/binary_eval.sbatch \
        "$model" "$dataset" "$checkpoint" \
        --decision-threshold "$gamma" \
        --m-values 2 8 32 \
        --num-workers 4 \
        --batch-size "$batch_size" \
        --output-dir "$output_dir"
}

submit_m128() {
    local model=$1
    local dataset=$2
    local condition=$3
    local checkpoint=$4
    local batch_size=8
    if [[ "$dataset" == "fives" ]]; then
        batch_size=4
    fi

    sbatch --parsable \
        --job-name "selseg-m128m-$dataset-$condition" \
        scripts/slurm/binary_eval.sbatch \
        "$model" "$dataset" "$checkpoint" \
        --decision-threshold 0.5 \
        --m-values 128 \
        --num-workers 4 \
        --batch-size "$batch_size" \
        --output-dir outputs/binary_m128_matched
}

for threshold in "0.3 03" "0.7 07"; do
    read -r gamma tag <<< "$threshold"

    submit_condition "$gamma" "$tag" clipseg pet clip-g -
    submit_condition "$gamma" "$tag" clipseg pet clip-t \
        outputs/binary_train/pet/clipseg/seed-0/checkpoint.pt
    submit_condition "$gamma" "$tag" deeplabv3 pet dl-e -
    submit_condition "$gamma" "$tag" deeplabv3 pet dl-t \
        outputs/binary_train/pet/deeplabv3/seed-0/checkpoint.pt

    submit_condition "$gamma" "$tag" clipseg kvasir clip-g -
    submit_condition "$gamma" "$tag" clipseg kvasir clip-t \
        outputs/binary_train/kvasir/clipseg/seed-0/checkpoint.pt
    submit_condition "$gamma" "$tag" deeplabv3 kvasir dl-t \
        outputs/binary_train/kvasir/deeplabv3/seed-0/checkpoint.pt

    submit_condition "$gamma" "$tag" clipseg fives clip-g -
    submit_condition "$gamma" "$tag" clipseg fives clip-t \
        outputs/binary_train/fives/clipseg/seed-0/checkpoint.pt
    submit_condition "$gamma" "$tag" deeplabv3 fives dl-t \
        outputs/binary_train/fives/deeplabv3/seed-0/checkpoint.pt
done

submit_m128 clipseg pet clip-g -
submit_m128 clipseg pet clip-t \
    outputs/binary_train/pet/clipseg/seed-0/checkpoint.pt
submit_m128 deeplabv3 pet dl-e -
submit_m128 deeplabv3 pet dl-t \
    outputs/binary_train/pet/deeplabv3/seed-0/checkpoint.pt

submit_m128 clipseg kvasir clip-g -
submit_m128 clipseg kvasir clip-t \
    outputs/binary_train/kvasir/clipseg/seed-0/checkpoint.pt
submit_m128 deeplabv3 kvasir dl-t \
    outputs/binary_train/kvasir/deeplabv3/seed-0/checkpoint.pt

submit_m128 clipseg fives clip-g -
submit_m128 clipseg fives clip-t \
    outputs/binary_train/fives/clipseg/seed-0/checkpoint.pt
submit_m128 deeplabv3 fives dl-t \
    outputs/binary_train/fives/deeplabv3/seed-0/checkpoint.pt
