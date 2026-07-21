# Selective Segmentation

Reference implementation and manuscript source for **Loss-Indexed Confidence for
Selective Binary Segmentation**.

The central idea is simple: confidence should predict the loss that matters at
deployment. Given a deployed hard mask \(\widehat Y\), a probability map \(p\),
and a segmentation loss \(L\), the code constructs a declared working posterior
over candidate masks and estimates

\[
C_L(x)=-\mathbb E_{Q_p}\!\left[L(Y,\widehat Y(x))\right].
\]

The current paper studies Dice, normalized Hausdorff distance (nHD), and
normalized HD95 (nHD95). Dice and nHD instantiate the same
Lipschitz--Wasserstein theory under different mask geometries; nHD95 is the
robust percentile-based extension covered by the general bounded-loss theory.

## Repository layout

```text
selectseg/          reusable library and command-line workers
  pipeline/         freeze, shared scoring, and loss-indexed scoring
  seed/             training-seed extension
  studies/          focused auxiliary studies
scripts/            thin analysis, rendering, submission, and release CLIs
  analyze/          one module per output schema
  render/           one module per manuscript artifact
  submit/           parameterized Slurm planners
  maintenance/      reproducibility and release utilities
  slurm/run.sbatch  the only generic Slurm worker wrapper
configs/            reviewed experiment specifications and immutable locks
results/            checked-in, portable, publication-facing result bundle
outputs/            local runtime artifacts and receipts; ignored by Git
docs/               canonical manuscript source
overleaf/           local Overleaf checkout; ignored by Git
tests/              unit, contract, provenance, and submission tests
```

There is deliberately no second GitHub checkout inside this repository.
`docs/` is the canonical paper tree; `overleaf/` is only a synchronization
target. Runtime files belong in `outputs/`, while reviewed portable evidence
belongs in `results/`. Published result bundles retain their internal layout
because their paths and hashes are part of the reproducibility contract.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[dev,plots]'
```

Large datasets, checkpoints, caches, frozen probability maps, and private
scheduler receipts are intentionally excluded from Git. Their logical IDs and
content hashes are retained in the locks and public manifests.

## Main experiment workflow

The main planner uses one `--phase` argument instead of separate scripts for
each setting:

```bash
python -m scripts.submit.main --config configs/binary_midpoint_main_v2.json --phase freeze
python -m scripts.submit.main --config configs/binary_midpoint_main_v2.json --phase common
python -m scripts.submit.main --config configs/binary_midpoint_main_v2.json --phase score
python -m scripts.submit.main --config configs/binary_midpoint_main_v2.json --phase assemble
python -m scripts.submit.main --config configs/binary_midpoint_main_v2.json --phase diagnose
```

All submission commands are dry runs unless `--submit` is provided. A real
submission also requires an append-only receipt, for example:

```bash
python -m scripts.submit.main \
  --config configs/binary_midpoint_main_v2.json \
  --phase freeze --submit \
  --receipt outputs/binary_midpoint_main_v2/receipts/freeze.jsonl
```

Before a schema-v2 replay, verify the complete scientific-input lock:

```bash
python -m selectseg.provenance verify \
  --lock configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json \
  --expected-sha256 8f2e492e959fe94727bfa535a578e1ac95e2e6a45230a7091faf7512e51c8bb3 \
  --mode full
```

The lock covers datasets and loader order, checkpoints, base-model files,
runtime environment, and the exact freeze source closure. Drift fails closed.

### Slurm policy

`scripts/slurm/run.sbatch` is the sole worker wrapper. Resource choices remain
visible in the generated `sbatch` command rather than being scattered among
task-specific wrapper files.

- Each scalar experiment is one independent Slurm job; arrays are not used.
- GPU jobs are eligible for both private queues: `saffo-a100,apollo_agate`.
- CPU planners can use `amdsmall,agsmall,msismall,saffo-2tb`.
- Candidate partition order is rotated across jobs so one busy queue does not
  serialize a campaign.
- Frozen probability maps and common scores are reused, so new confidence
  settings do not repeat model inference.
- Every phase records an append-only receipt and refuses blind duplicate
  submission.

The generic command emitted by planners has this shape:

```bash
sbatch --partition=saffo-a100,apollo_agate --gres=gpu:a100:1 \
  scripts/slurm/run.sbatch python -m selectseg.pipeline.freeze ...
```

Use `--scheduler-preflight-only` on the main schema-v2 planner to validate an
entire wave with `sbatch --test-only` before opening a receipt.

## Auxiliary studies

New parameter values should be added to an existing config and selected by an
argument, not implemented as a new worker or Slurm file. Separate modules are
kept only when the output schema or computational contract is genuinely
different.

```bash
# Region/block coupling and component-count variants
python -m scripts.submit.counts --config configs/auxiliary/dice_partition_ladder_v1.json

# Probability ensemble construction or ensemble baselines
python -m scripts.submit.ensemble --phase build
python -m scripts.submit.ensemble --phase baselines

# Runtime protocols through one CLI
python -m scripts.submit.runtime --mode basic
python -m scripts.submit.runtime --mode ladder

# Other focused studies
python -m scripts.submit.gamma
python -m scripts.submit.cardinality
python -m scripts.submit.m128
python -m scripts.submit.synthetic --phase pilot
```

The active runtime-ladder lock follows the current module layout. The exact
historical lock used by the published runtime result is retained beside that
result as `results/runtime_ladder_v2.lock.json`.

The Dice mechanism studies keep the deployed mask and every pixel marginal
fixed. For a fixed action, Dice depends only on overlap and outside counts, so
spatial partitions matter only through the induced two-dimensional count law;
its deviation from SDC is controlled by count dispersion. The predeclared
component--grid ladder does not show a stable
advantage over two-block, SDC, or foreground entropy; it is retained as a
negative mechanism result rather than a selected method.

## Analysis and paper artifacts

The main strict analysis and table renderer are:

```bash
python -m scripts.analyze.main --help
python -m scripts.render.paper --help
python -m scripts.analyze.counts --mode partition --help
python -m scripts.render.counts --mode partition --help
```

Raw JSON stores AURC in its natural scale. Manuscript tables multiply AURC and
AURC differences by 100 for readability. Generated tables should be rebuilt
from validated JSON rather than edited by hand.

The completed public campaign covers 16 model--dataset conditions across Pet,
Kvasir-SEG, FIVES, ISIC 2018, and TN3K. Its portable bundle in `results/`
contains manifests, per-image records, analyses, and redacted provenance. The
historical schema-v1 evidence is immutable; current code can reproduce or
analyze it but does not rewrite old receipts to claim newer scheduler policy.

## Manuscript

`docs/main.tex` is the canonical entry point. Sections, generated tables, and
figures live in `docs/Sections`, `docs/Tables`, and `docs/Figures`.

Build without placing auxiliary files in `docs/`:

```bash
mkdir -p /tmp/selectseg-paper
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/selectseg-paper docs/main.tex
```

Preview or apply the whitelist-based Overleaf synchronization:

```bash
python -m scripts.sync
python -m scripts.sync --apply
```

The sync command copies manuscript sources only; it does not create nested Git
clones or copy runtime outputs.

## Verification

```bash
ruff check . --exclude outputs,overleaf
python -m pytest -q
```

Model integration tests load large cached networks and are skipped by default.
Run them explicitly on a suitable node with:

```bash
SELECTSEG_RUN_MODEL_TESTS=1 python -m pytest -q tests/test_models.py
```

Useful fail-closed checks include:

```bash
python -m scripts.submit.main --help
python -m scripts.submit.main \
  --config configs/binary_midpoint_main_v2.json \
  --phase freeze --scheduler-preflight-only
python -m scripts.sync
```

## Reproducibility boundary

The working posterior is an explicit probe derived from one probability map;
it is not claimed to identify the true joint mask posterior. Marginal
calibration alone cannot validate its pixel coupling. The theory therefore
separates numerical integration error from loss-specific posterior discrepancy
and carries their score error into selective-risk and AURC regret. Exact Dice
removes the numerical term for the Dice instantiation, while nHD and nHD95 use
the declared quadrature protocol.

See `docs/README.md` for the manuscript-specific release record and
`docs/MECHANISM_STUDY.md` for the current coupling experiments.
