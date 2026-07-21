#!/bin/bash
# Submit the selective-prediction sweep: per-image confidence scores for
# every condition, then a dependent analysis + figures job.
#
# Every job requests all GPU partitions except interactive-gpu (SLURM
# schedules onto whichever frees up first) and is requeueable, so a
# preemption on preempt-gpu re-runs the job rather than losing it. Each
# eval writes its own JSONL fresh, so a requeue simply recomputes it.
#
# Usage: scripts/slurm/submit_selective.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p outputs/slurm

# All GPU partitions except interactive-gpu, schedulable under one account.
GPU_PARTITIONS="saffo-a100,apollo_agate,a100-4,a100-8,v100,msigpu,preempt-gpu"
# CPU partitions for the GPU-free analysis job (GPU partitions reject jobs
# without a --gres request). saffo-2tb / amd2tb / ag2tb usually start
# immediately; the msi/amd small partitions are fallbacks.
CPU_PARTITIONS="saffo-2tb,amd2tb,ag2tb,msismall,amdsmall,amd512"
ACCOUNT="hou00123"
SBATCH_OPTS=(--partition "$GPU_PARTITIONS" --account "$ACCOUNT")

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
        submit_selective "selseg-sel-$model-$dataset" "$model" "$dataset"
        submit_selective "selseg-sel-$model-$dataset-target" "$model" "$dataset" \
            --checkpoint "outputs/train/${model}_${dataset}/checkpoint.pt"
        submit_selective "selseg-sel-$model-$dataset-b" "$model" "$dataset" \
            --checkpoint "outputs/train_b/${model}_${dataset}/checkpoint.pt" \
            --output-dir outputs/selective_b
    done
done

dependency="afterok:$(IFS=:; echo "${eval_ids[*]}")"
analyze_id=$(sbatch --parsable --partition "$CPU_PARTITIONS" --account "$ACCOUNT" \
    --dependency "$dependency" scripts/slurm/analyze.sbatch)

echo "submitted ${#eval_ids[@]} eval jobs: ${eval_ids[*]}"
echo "analyze job (afterok all evals): $analyze_id"
