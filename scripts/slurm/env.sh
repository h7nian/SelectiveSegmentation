# Shared job environment; source from any node (login or compute).
# Usage: source scripts/slurm/env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Python multiprocessing cleanup is unreliable on the shared scratch TMPDIR.
# Give every job a private node-local temporary directory instead.
SELECTSEG_TMPDIR=$(mktemp -d "/tmp/selectseg-${SLURM_JOB_ID:-local}.XXXXXX")
export TMPDIR="$SELECTSEG_TMPDIR"

module load python3/3.12.4_anaconda2024.06-1_libmamba
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep model caches inside the repo so jobs never touch the network.
export HF_HOME="$REPO_ROOT/data/cache/huggingface"
export TORCH_HOME="$REPO_ROOT/data/cache/torch"
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
