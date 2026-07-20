#!/bin/bash -l
#SBATCH --job-name=selseg-runtime-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Benchmark exactly one lock-listed condition.  Artifact decoding occurs before
# any timed region; model inference is absent.  Eight Python workers share
# the image panel while every native numerical library is pinned to one thread.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

python -m selectseg.benchmark_binary_runtime "$@"
