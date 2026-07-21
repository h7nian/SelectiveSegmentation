#!/bin/bash
# Config-B ablation plus Hausdorff back-fill for the original runs.
#
# Config B changes exactly one training hyperparameter per model relative to
# the defaults in selectseg/train.py ("config A"):
#   - deeplabv3: 80 epochs instead of 40 (tests undertraining on VOC)
#   - clipseg:   lr 3e-4 instead of 1e-4 (one step toward the paper's 1e-3)
#
# Submits per model x dataset:
#   1. re-evaluation of the config-A conditions (zero-shot and fine-tuned)
#      so their metrics JSONs gain HD95,
#   2. a config-B fine-tuning run (outputs/train_b/), and
#   3. a dependent evaluation of its checkpoint (outputs/eval_b/).
#
# Usage: scripts/slurm/submit_config_b.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

SBATCH_OPTS=(--partition saffo-a100 --account ssafo)

config_b_args() {
    if [[ $1 == deeplabv3 ]]; then
        echo "--epochs 80"
    else
        echo "--lr 3e-4"
    fi
}

for model in clipseg deeplabv3; do
    for dataset in pet voc; do
        zs_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" \
            --job-name "selseg-eval-$model-$dataset" \
            scripts/slurm/eval.sbatch "$model" "$dataset")
        a_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" \
            --job-name "selseg-eval-$model-$dataset-target" \
            scripts/slurm/eval.sbatch "$model" "$dataset" \
            --checkpoint "outputs/train/${model}_${dataset}/checkpoint.pt")
        # shellcheck disable=SC2046  # word-splitting of the extra args is intended
        train_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" \
            --job-name "selseg-train-$model-$dataset-b" \
            scripts/slurm/train.sbatch "$model" "$dataset" \
            --output-dir "outputs/train_b/${model}_${dataset}" \
            $(config_b_args "$model"))
        b_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" \
            --job-name "selseg-eval-$model-$dataset-b" \
            --dependency "afterok:$train_id" \
            scripts/slurm/eval.sbatch "$model" "$dataset" \
            --checkpoint "outputs/train_b/${model}_${dataset}/checkpoint.pt" \
            --output-dir outputs/eval_b)
        echo "$model/$dataset: evalA-zs=$zs_id evalA-target=$a_id trainB=$train_id evalB=$b_id"
    done
done
