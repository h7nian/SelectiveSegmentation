# Manuscript source

`main.tex` is the entry point. Individual manuscript sections live in
`Sections/`, generated tables in `Tables/`, and vector figures in `Figures/`.
The proof appendix is included through `Sections/A_theory.tex`.

The source is tested with pdfTeX 1.40.28 from TeX Live 2025. Build it with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is available, the equivalent one-line command is
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

The tables are not hand-entered. The companion code repository uses normalized
penalized full Hausdorff distance (HD) in unit-diagonal image coordinates as the
flagship instance, HD95 as its robust nonmetric extension, and Dice as a regional
contrast. Each schema-v2 assembled row contains
all three risks, the three risk-aligned midpoint ladders at `M=2,8,32`, common
baselines, and the Exact Dice level-set oracle. The strict analysis JSON also
uses schema v2.

The main absolute-result display is a symmetric three-table grid: CLIP-T,
DL-T, and SF-T each span Oxford Pet, Kvasir-SEG, FIVES, ISIC 2018, TN3K, and
DUTS with the same ten methods and three risks. A separately locked
DUTS/CLIP-T condition completes this rectangular display without enlarging the
original seven-condition architecture/domain inference family.

The completed main campaign freezes model probability maps once for each of 16
conditions. It is an immutable schema-v1 campaign: its GPU jobs retain the
partition assignments recorded in their receipts, and its CPU phases retain
their deterministic rotation over the three declared CPU partitions. After
the campaign lock was written, 16 CPU jobs computed M-independent fields and
48 independent CPU jobs computed the Cartesian product of
`(artifact, gamma, M, seed, estimator)`. These historical locks and receipts
are evidence of what ran; they are not rewritten to claim the newer scheduler
policy.

Schema v2 implements two explicit policies. A `scheduler-preview-only` fixture
is permanently fail-closed and can only print commands or run
`sbatch --test-only`; it cannot create receipts, locks, recoveries, retries, or
real jobs. A `scientific-input-locked` campaign is executable only after an
exact path and SHA-256 bind a reviewed root lock covering all five evaluation
dataset bytes and loader order, checkpoints, base-model files, freeze sources,
and runtime environment. Before planning and again on each freeze compute node,
lock drift fails closed. The freeze worker uses consume verification: it fully
checks small inputs and verifies every image/mask byte on its unavoidable read.

Both policies retain `saffo-a100` and `apollo_agate` for every GPU experiment,
alternating which private queue appears first. Every schema-v2 CPU experiment
retains all four CPU candidates, rotates `amdsmall`, `agsmall`, and `msismall`
as its first choices, and keeps `saffo-2tb` last as the private fallback. Slurm
places the single job on one eligible partition. Every M-specific evaluation
receives one scalar M, and Slurm arrays are not used. The scientific-input
dataset components likewise use five independent jobs---one per dataset:

```bash
python -m scripts.submit.provenance \
  --scheduler-preflight-only
python -m scripts.submit.provenance --submit \
  --receipt outputs/binary_midpoint_main_v2/scientific_inputs/dataset-build-receipt.jsonl
```

Those no-overwrite components are combined with separately sealed source,
base-model, checkpoint, and environment components at
`configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json`. A full-byte
verification is required before the execution config can bind the lock:

```bash
SELSEG_SCIENCE_LOCK_SHA256=8f2e492e959fe94727bfa535a578e1ac95e2e6a45230a7091faf7512e51c8bb3
python -m selectseg.provenance verify \
  --lock configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json \
  --expected-sha256 "$SELSEG_SCIENCE_LOCK_SHA256" --mode full
python -m scripts.submit.main \
  --config configs/binary_midpoint_main_v2.json --phase freeze \
  --scheduler-preflight-only
python -m scripts.submit.main \
  --config configs/binary_midpoint_main_v2.json --phase freeze --submit \
  --receipt outputs/binary_midpoint_main_v2/receipts/freeze.jsonl
```

The locked freeze writes schema-v3 artifact manifests. Sixteen explicit
manifests then produce a schema-v2 post-freeze campaign lock, from which common,
score, assemble, and diagnostic inputs are derived without scans. Each phase
has one canonical append-only receipt; reconciliation appends observed Slurm
facts, interrupted successful submissions require identity-checked recovery,
and failed attempts require explicit per-job retry authorization. The sealed v1
config and receipts remain the original reproduction record for the manuscript
numbers. The following schema-v2 execution facts are a local release audit;
private receipts, job identifiers, node facts, and the completed v2 campaign
lock are not included in the anonymous artifact. The full-byte seal passed on
2026-07-21. The isolated one-image smoke
campaign then completed all seven jobs and terminal receipt gates. The full
schema-v2 replay subsequently completed 112/112 non-array jobs: 16 freezes, 16
common-score jobs, 48 scalar-M score jobs, 16 assemblies, and 16 diagnostics.
Its post-freeze campaign-lock SHA-256 is
`eb3d8f4078f482b541e1771ae43c4c87d12be8733904996d770eef19e09f704b`.
All 64 score inputs, 16 assembled records, and 16 diagnostic summaries passed
hash and row-count validation. CPU execution used all four declared partitions
(`saffo-2tb=64`, `amdsmall=11`, `agsmall=10`, `msismall=11`). Compared with the
immutable v1 assembled records, every one of the 816 method-by-risk AURCs is
exactly equal; the only non-provenance row differences are M=32 roundoff no
larger than `2.22e-16`.

The lock validates immutable manifest bytes and structure and records the
payload hashes declared there; it does not decompress every large payload on
the login node. Each common and M-specific score job stream-validates every
payload hash and array while consuming the frozen artifact.

The diagnostics report marginal Brier/ECE and level-set descriptors. Marginal
calibration cannot identify a joint mask posterior or validate the
comonotone/shared-threshold coupling. Their strict aggregate consumes one
campaign lock plus 16 explicit `diagnostics.json` paths, verifies every source
manifest digest, ordered cohort identity, condition and count, and emits
`diagnostics_analysis.json` plus the guarded
`Tables/binary_diagnostics.tex`. Held-out labels enter only the predeclared
descriptive aggregates and never confidence fitting or sample selection.
The two-condition development pilot in
`configs/binary_midpoint_dual_pilot.json` makes each freeze eligible on both
GPU partitions:
`expected_dataset_samples` validates the full split before `freeze_limit`
selects the first development-only images. Such limited artifacts are never
canonical results.

After validating all per-image manifests and records against the immutable
campaign lock, the analysis runs the declared tie-aware three-risk AURC
comparisons. It records only portable logical identifiers and content hashes.
The renderer produces exactly six canonical tables: `main_results.tex`, `full_target_results.tex`,
`complete_results.tex`, `cross_loss_results.tex`,
`quadrature_ablation.tex`, and `statistical_tests.tex`. It publishes the
`results_complete.tex` sentinel only after all six carry the same source
analysis SHA-256.

The 16 expected condition cohorts are Pet (3,669 images for each of four model
conditions), Kvasir-SEG (200 for each of three), FIVES (200 for each of three),
ISIC 2018 (1,000 for each of three), and TN3K (614 for each of three). The final
public artifact includes a portable campaign descriptor, all assembled
manifests, all per-image `records.jsonl` files, and redacted phase-completion
provenance; private filesystem, account, partition, and job identifiers are not
exported.
The training-seed extension follows the same rule: its portable analysis,
terminal scheduler summary, and write-last provenance guard are exported only
after all 162 one-job phase records, including the 20 training cells, satisfy
their locked grids. Raw AURC remains unscaled in JSON; manuscript displays
multiply AURC-derived quantities by 100.
The anonymous analysis artifact additionally contains 30 path-free
manifest/record pairs for the five-dataset, two-target-model, three-seed grid.
From an extracted artifact, `python -m scripts.maintenance.replay_seed`
recomputes the full seed JSON and both seed tables from those per-image records
and requires byte equality with all three released references. Original
execution-bearing assembly manifests are not included in this replay bundle.
Dataset archives, model caches, checkpoints, and frozen probability-map
payloads are not part of this LaTeX package; their immutable identifiers are
retained in the released manifests. The code release provides official dataset
acquisition instructions and archive locations for redistributable large
artifacts.

The checked-in ICLR 2026 style is the latest released working template as of
this source revision. It must be replaced with the official ICLR 2027 template
when that template is released.
