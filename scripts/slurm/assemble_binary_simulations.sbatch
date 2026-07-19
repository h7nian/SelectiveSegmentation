#!/bin/bash -l
#SBATCH --job-name=selseg-assemble-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Assemble one condition after its common job and three M=2/8/32 simulations.
# The canonical campaign invokes this wrapper through
# ``scripts.submit_binary_simulations --phase assemble``, which derives all
# four manifest paths from the campaign lock without globbing. The standalone
# form remains useful for tests and auditing:
#
# sbatch scripts/slurm/assemble_binary_simulations.sbatch \
#   --campaign-lock outputs/binary_campaign/campaign.lock.json \
#   --common path/to/common/manifest.json \
#   --input path/to/m2/manifest.json \
#   --input path/to/m8/manifest.json \
#   --input path/to/m32/manifest.json \
#   --output-root outputs/binary_assembled

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/env.sh

python -m scripts.assemble_binary_simulations "$@"
