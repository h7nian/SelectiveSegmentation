#!/bin/bash
# Submit the full benchmark: for each model x dataset, one zero-shot
# evaluation plus one fine-tuning run whose evaluation starts on success.
#
# Usage: scripts/slurm/submit_all.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

# The private saffo-a100 partition starts immediately, while the public
# GPU queues are typically days deep; the full benchmark needs at most
# eight concurrent GPUs, well within the partition's sixteen.
EVAL_OPTS=(--partition saffo-a100 --account ssafo)
TRAIN_OPTS=(--partition saffo-a100 --account ssafo)

for model in clipseg deeplabv3; do
    for dataset in pet voc; do
        eval_id=$(sbatch --parsable "${EVAL_OPTS[@]}" \
            --job-name "selseg-eval-$model-$dataset" \
            scripts/slurm/eval.sbatch "$model" "$dataset")
        train_id=$(sbatch --parsable "${TRAIN_OPTS[@]}" \
            --job-name "selseg-train-$model-$dataset" \
            scripts/slurm/train.sbatch "$model" "$dataset")
        target_eval_id=$(sbatch --parsable "${TRAIN_OPTS[@]}" \
            --job-name "selseg-eval-$model-$dataset-target" \
            --dependency "afterok:$train_id" \
            scripts/slurm/eval.sbatch "$model" "$dataset" \
            --checkpoint "outputs/train/${model}_${dataset}/checkpoint.pt")
        echo "$model/$dataset: eval=$eval_id train=$train_id eval-target=$target_eval_id"
    done
done
