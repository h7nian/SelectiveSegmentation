#!/bin/bash -l
#SBATCH --job-name=selseg-diagnose-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Diagnose exactly one frozen artifact per independent Slurm job. The canonical
# campaign invokes this wrapper through
# ``scripts.submit_binary_simulations --phase diagnose``, which obtains the
# singular manifest and digest directly from the campaign lock. Standalone
# example:
#
# sbatch scripts/slurm/diagnose_binary_artifact.sbatch \
#   --artifact-manifest outputs/binary_maps/pet/clipseg-general/ID/manifest.json \
#   --expected-artifact-manifest-sha256 MANIFEST_SHA256 \
#   --output-root outputs/binary_diagnostics \
#   --write-descriptors

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python scripts/diagnose_binary_artifact.py "$@"
