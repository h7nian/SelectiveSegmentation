#!/bin/bash
# Legacy bootstrap for training and independently validating the ISIC 2018
# and TN3K extensions. Each training condition and each (condition, M)
# evaluation is submitted as its own Slurm job. This is not the canonical
# freeze/score/assemble campaign launcher.
#
# Usage: bash scripts/slurm/submit_extended_binary.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

readonly GPU_PARTITIONS="saffo-a100,apollo_agate"
readonly ACCOUNT="ssafo"
readonly -a M_VALUES=(2 8 32)

submit_eval() {
    local name=$1
    local model=$2
    local dataset=$3
    local checkpoint=$4
    local m=$5
    local dependency=${6:-}

    if [[ -n "$dependency" ]]; then
        sbatch --parsable \
            --partition "$GPU_PARTITIONS" --account "$ACCOUNT" \
            --job-name "$name" --dependency "afterok:$dependency" \
            scripts/slurm/binary_eval.sbatch "$model" "$dataset" \
            "$checkpoint" --decision-threshold 0.5 --m-values "$m" \
            --num-workers 4 --batch-size 4 \
            --output-dir outputs/binary_extended
    else
        sbatch --parsable \
            --partition "$GPU_PARTITIONS" --account "$ACCOUNT" \
            --job-name "$name" \
            scripts/slurm/binary_eval.sbatch "$model" "$dataset" \
            "$checkpoint" --decision-threshold 0.5 --m-values "$m" \
            --num-workers 4 --batch-size 4 \
            --output-dir outputs/binary_extended
    fi
}

submit_dataset() {
    local dataset=$1
    local train_root="outputs/binary_train/$dataset"

    local clip_train
    local deeplab_train
    local m
    local -a clip_general_ids=()
    local -a clip_target_ids=()
    local -a deeplab_target_ids=()

    clip_train=$(sbatch --parsable \
        --partition "$GPU_PARTITIONS" --account "$ACCOUNT" \
        --job-name "selseg-train-$dataset-clip" \
        scripts/slurm/train.sbatch clipseg "$dataset" \
        --output-dir "$train_root/clipseg/seed-0" --epochs 40 --seed 0)
    deeplab_train=$(sbatch --parsable \
        --partition "$GPU_PARTITIONS" --account "$ACCOUNT" \
        --job-name "selseg-train-$dataset-dl" \
        scripts/slurm/train.sbatch deeplabv3 "$dataset" \
        --output-dir "$train_root/deeplabv3/seed-0" --epochs 40 --seed 0)

    for m in "${M_VALUES[@]}"; do
        clip_general_ids+=("$(submit_eval \
            "selseg-eval-$dataset-clip-g-m$m" clipseg "$dataset" - "$m")")
        clip_target_ids+=("$(submit_eval \
            "selseg-eval-$dataset-clip-t-m$m" clipseg "$dataset" \
            "$train_root/clipseg/seed-0/checkpoint.pt" "$m" "$clip_train")")
        deeplab_target_ids+=("$(submit_eval \
            "selseg-eval-$dataset-dl-t-m$m" deeplabv3 "$dataset" \
            "$train_root/deeplabv3/seed-0/checkpoint.pt" "$m" \
            "$deeplab_train")")
    done

    echo "$dataset: clip-train=$clip_train dl-train=$deeplab_train "\
         "clip-general=${clip_general_ids[*]} "\
         "clip-target=${clip_target_ids[*]} "\
         "dl-target=${deeplab_target_ids[*]}"
}

submit_dataset isic
submit_dataset tn3k
