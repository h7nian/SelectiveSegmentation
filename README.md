# Selective Segmentation

Reference implementation and manuscript source for **Risk-Aligned Confidence for
Selective Segmentation**.

The central idea is simple: confidence should predict the loss that matters at
deployment, using a working posterior whose mask geometry matches that loss.
Given a deployed hard mask \(\widehat Y\), a probability map \(p\), and a
segmentation loss \(L\), the code constructs a declared posterior over candidate
masks and estimates

\[
C_L(x)=-\mathbb E_{Q_p}\!\left[L(Y,\widehat Y(x))\right].
\]

The flagship instance is full Hausdorff distance (HD) in unit-diagonal image
coordinates: shared
thresholds trace coherent nested boundary motion, and surface-Hausdorff geometry
gives a constant-one Wasserstein posterior bound. Normalized HD95 is the robust,
nonmetric boundary extension. Dice is retained as a regional contrast whose
count-law analysis explains why SDC remains competitive.

## Repository layout

```text
selectseg/          reusable library and command-line workers
  pipeline/         freeze, shared scoring, and risk-aligned scoring
  seed/             training-seed extension
  studies/          focused auxiliary studies
scripts/            thin analysis, rendering, submission, and release CLIs
  analyze/          one module per output schema
  render/           one module per manuscript artifact
  submit/           parameterized Slurm planners
  maintenance/      reproducibility and release utilities
  slurm/run.sbatch  the only generic Slurm worker wrapper
configs/            reviewed experiment specifications and immutable locks
results/            checked-in, portable, publication-facing result bundles
  extension/        seven-condition SegFormer/DUTS extension evidence
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

For a new target-model campaign, the same planner can first submit one training
job per target condition with `--phase train`. The architecture/domain extension
is declared in `configs/extension.json`; it adds SegFormer-B2 across the five
existing datasets and compares SegFormer-B2 with DeepLabV3 on DUTS without
altering the completed primary campaign.

The separately locked `configs/table_completion.json` design adds only
DUTS/CLIPSeg-Target and rebinds the seven frozen extension artifacts. It makes
the three target-model tables rectangular: CLIP-T, DL-T, and SF-T each use the
same six datasets, ten displayed methods, and three risk blocks. The original
seven-condition extension remains the sole source of its predeclared
28-comparison inference.

```bash
python scripts/download.py --datasets duts
python -m scripts.submit.main --config configs/extension.json --phase train
python -m scripts.submit.main --config configs/extension.json --phase freeze
# After locking, run common, score, assemble, and diagnose exactly as above.
python -m scripts.analyze.main --design extension --help
python -m scripts.analyze.main --design completion --help
python -m scripts.render.paper --design extension --help
python -m scripts.analyze.diagnostics --design extension --help
```

Completed training conditions can be frozen immediately without creating a
second config. Repeat `--condition DATASET/CONDITION` to submit a reviewed
subset, then rerun the same command without the filter: the shared receipt
skips earlier jobs and fills only the remaining conditions.

```bash
python -m scripts.submit.main --config configs/extension.json --phase freeze \
  --condition pet/segformer-target --condition kvasir/segformer-target
```

For training jobs that are still active, bind each freeze to its own scheduler
dependency instead of polling or serializing the wave. The dependency is part
of the receipt-bound command, so later invocations must preserve it.

```bash
python -m scripts.submit.main --config configs/extension.json --phase freeze \
  --afterok duts/segformer-target=TRAIN_JOB_ID \
  --afterok duts/deeplabv3-target=TRAIN_JOB_ID
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

# Marginal-preserving spatial copula: one condition × variant × repeat per job
python -m scripts.submit.counts --config configs/auxiliary/spatial_copula_v1.json

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
python -m scripts.analyze.counts --mode copula --help
python -m scripts.render.counts --mode copula --help
```

The spatial-copula planner writes 240 independent jobs for the locked
10-condition, 6-variant, 4-repeat design. Filters such as `--condition`,
`--variant-id`, and `--repeat-index` support pilots and exact retry cells
without adding worker scripts or changing the artifact schema.

Raw JSON stores AURC in its natural scale. Manuscript tables multiply AURC and
AURC differences by 100 for readability. Generated tables should be rebuilt
from validated JSON rather than edited by hand.

The completed public campaign covers 16 model--dataset conditions across Pet,
Kvasir-SEG, FIVES, ISIC 2018, and TN3K. Its portable bundle in `results/`
contains manifests, per-image records, analyses, and redacted provenance. The
historical schema-v1 evidence is immutable; current code can reproduce or
analyze it but does not rewrite old receipts to claim newer scheduler policy.
The separate architecture/domain extension is complete. Its portable bundle in
`results/extension/` contains the immutable lock, seven assembled
manifest/record pairs, the 10,000-resample analysis, the long-form result table,
and aggregate diagnostics. It remains separate from the primary bundle so that
the original 16-condition design is never silently redefined. Recompute its
analysis and paper tables with:

```bash
python -m scripts.analyze.main \
  --design extension \
  --inputs \
    results/extension/assembled/pet/segformer-target/937b390f22382cfd/records.jsonl \
    results/extension/assembled/kvasir/segformer-target/687c0bcc4dd35db7/records.jsonl \
    results/extension/assembled/fives/segformer-target/05979a7fc5f8eadc/records.jsonl \
    results/extension/assembled/isic/segformer-target/c93e814c00d6df3b/records.jsonl \
    results/extension/assembled/tn3k/segformer-target/10ae39efb93cd7d9/records.jsonl \
    results/extension/assembled/duts/segformer-target/c64ff75aa1c30d74/records.jsonl \
    results/extension/assembled/duts/deeplabv3-target/6c2ff62caf98d5f5/records.jsonl \
  --campaign-lock results/extension/campaign.lock.json \
  --bootstrap-samples 10000 \
  --bootstrap-workers 8 \
  --output-dir /tmp/selectseg-extension-analysis
python -m scripts.render.paper \
  --design extension \
  --analysis /tmp/selectseg-extension-analysis/analysis.json \
  --output-dir /tmp/selectseg-extension-tables
```

## Manuscript

`docs/main.tex` is the canonical entry point. Sections, generated tables, and
figures live in `docs/Sections`, `docs/Tables`, and `docs/Figures`.

Build without placing auxiliary files in `docs/`:

```bash
mkdir -p /tmp/selectseg-paper
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
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
removes the numerical term for the Dice instantiation, while HD and HD95 use
the declared quadrature protocol.

See `docs/README.md` for the manuscript-specific release record and
`docs/MECHANISM_STUDY.md` for the current coupling experiments.
