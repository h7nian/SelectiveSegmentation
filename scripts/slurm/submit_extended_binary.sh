#!/bin/bash
# Train and evaluate the ISIC 2018 and TN3K extensions. Each train or
# evaluation condition is submitted as its own Slurm job.
#
# Usage: bash scripts/slurm/submit_extended_binary.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

submit_dataset() {
    local dataset=$1
    local train_root="outputs/binary_train/$dataset"
    local eval_root="outputs/binary_extended"

    local clip_train
    local deeplab_train
    local clip_general
    local clip_target
    local deeplab_target

    clip_train=$(sbatch --parsable \
        --job-name "selseg-train-$dataset-clip" \
        scripts/slurm/train.sbatch clipseg "$dataset" \
        --output-dir "$train_root/clipseg/seed-0" --epochs 40 --seed 0)
    deeplab_train=$(sbatch --parsable \
        --job-name "selseg-train-$dataset-dl" \
        scripts/slurm/train.sbatch deeplabv3 "$dataset" \
        --output-dir "$train_root/deeplabv3/seed-0" --epochs 40 --seed 0)

    clip_general=$(sbatch --parsable \
        --job-name "selseg-eval-$dataset-clip-g" \
        scripts/slurm/binary_eval.sbatch clipseg "$dataset" - \
        --decision-threshold 0.5 --m-values 2 8 32 \
        --num-workers 4 --batch-size 4 --output-dir "$eval_root")
    clip_target=$(sbatch --parsable \
        --job-name "selseg-eval-$dataset-clip-t" \
        --dependency "afterok:$clip_train" \
        scripts/slurm/binary_eval.sbatch clipseg "$dataset" \
        "$train_root/clipseg/seed-0/checkpoint.pt" \
        --decision-threshold 0.5 --m-values 2 8 32 \
        --num-workers 4 --batch-size 4 --output-dir "$eval_root")
    deeplab_target=$(sbatch --parsable \
        --job-name "selseg-eval-$dataset-dl-t" \
        --dependency "afterok:$deeplab_train" \
        scripts/slurm/binary_eval.sbatch deeplabv3 "$dataset" \
        "$train_root/deeplabv3/seed-0/checkpoint.pt" \
        --decision-threshold 0.5 --m-values 2 8 32 \
        --num-workers 4 --batch-size 4 --output-dir "$eval_root")

    echo "$dataset: clip-train=$clip_train dl-train=$deeplab_train "\
         "clip-general=$clip_general clip-target=$clip_target "\
         "dl-target=$deeplab_target"
}

submit_dataset isic
submit_dataset tn3k
