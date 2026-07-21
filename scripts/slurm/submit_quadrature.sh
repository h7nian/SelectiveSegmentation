#!/bin/bash
# Regenerate every per-image JSONL with the quadrature ablation scores
# (r_2 and r_M, METHODS.md §6.2), then re-run the analysis.
#
# Same 12 evaluations as submit_selective.sh — the 8 paper conditions
# (2 models x 2 datasets x {zero-shot, fine-tuned}) plus the 4 config-B
# replicas — because analyze_selective.py reads both outputs/selective and
# outputs/selective_b, and the new scores must exist in every file it reads.
# Nothing but selectseg.selective is changed, so each JSONL is simply
# rewritten with the extra columns and the existing scores reproduced.
#
# The quadrature ladder adds ~60 distance transforms per present class per
# image, so the walltime is raised over selective.sbatch's default.
#
# Usage: scripts/slurm/submit_quadrature.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

PARTITION="${PARTITION:-saffo-a100}"
ACCOUNT="${ACCOUNT:-ssafo}"
# CPU partitions for the GPU-free analysis job (GPU partitions reject jobs
# without a --gres request).
CPU_PARTITIONS="${CPU_PARTITIONS:-saffo-2tb,amd2tb,ag2tb,msismall,amdsmall,amd512}"
SBATCH_OPTS=(--partition "$PARTITION" --account "$ACCOUNT" --time 06:00:00)

eval_ids=()
submit_selective() {
    # submit_selective JOB_NAME MODEL DATASET [extra selective_eval args...]
    local name=$1 model=$2 dataset=$3
    shift 3
    eval_ids+=($(sbatch --parsable "${SBATCH_OPTS[@]}" --job-name "$name" \
        scripts/slurm/selective.sbatch "$model" "$dataset" "$@"))
}

for model in clipseg deeplabv3; do
    for dataset in pet voc; do
        submit_selective "selseg-quad-$model-$dataset" "$model" "$dataset"
        submit_selective "selseg-quad-$model-$dataset-target" "$model" "$dataset" \
            --checkpoint "outputs/train/${model}_${dataset}/checkpoint.pt"
        submit_selective "selseg-quad-$model-$dataset-b" "$model" "$dataset" \
            --checkpoint "outputs/train_b/${model}_${dataset}/checkpoint.pt" \
            --output-dir outputs/selective_b
    done
done

dependency="afterok:$(IFS=:; echo "${eval_ids[*]}")"
analyze_id=$(sbatch --parsable --partition "$CPU_PARTITIONS" --account "$ACCOUNT" \
    --dependency "$dependency" scripts/slurm/analyze.sbatch)

echo "submitted ${#eval_ids[@]} eval jobs: ${eval_ids[*]}"
echo "analyze job (afterok all evals): $analyze_id"
