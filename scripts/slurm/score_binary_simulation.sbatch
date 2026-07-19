#!/bin/bash -l
#SBATCH --job-name=selseg-score-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Score exactly one M-specific (artifact, gamma, M, estimator, seed) shard. The
# job writes identity fields plus the three M-specific scores; it never repeats
# risks or common baselines and performs no inference/import of Torch. The
# caller must pass the scorer's full flag set; one sbatch invocation must never
# contain a list of gamma, M, or seed values.  Example:
#
# sbatch scripts/slurm/score_binary_simulation.sbatch \
#   --campaign-id CAMPAIGN_ID \
#   --campaign-lock outputs/campaign/campaign-lock.json \
#   --expected-campaign-lock-sha256 CAMPAIGN_SHA256 \
#   --artifact-manifest outputs/binary_artifacts/pet/clipseg-general/ID/manifest.json \
#   --expected-artifact-manifest-sha256 ARTIFACT_SHA256 \
#   --estimator-spec configs/estimators/midpoint-v1.json \
#   --expected-estimator-spec-sha256 SPEC_SHA256 \
#   --gamma 0.5 --m 32 --seed 0 \
#   --output-root outputs/binary_simulations

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m selectseg.score_binary_simulation \
  --score-workers "$SLURM_CPUS_PER_TASK" \
  --max-pending-scores "$((2 * SLURM_CPUS_PER_TASK))" \
  "$@"
