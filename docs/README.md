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

The tables are not hand-entered. The companion code repository instantiates
the framework with Dice, normalized penalized full Hausdorff distance (nHD),
and normalized penalized HD95 (nHD95). Each schema-v2 assembled row contains
all three risks, the three loss-indexed midpoint ladders at `M=2,8,32`, common
baselines, and the Exact Dice level-set oracle. The strict analysis JSON also
uses schema v2.

The completed main campaign freezes model probability maps once for each of 16
conditions. It is an immutable schema-v1 campaign: its GPU jobs retain the
partition assignments recorded in their receipts, and its CPU phases retain
their deterministic rotation over the three declared CPU partitions. After
the campaign lock was written, 16 CPU jobs computed M-independent fields and
48 independent CPU jobs computed the Cartesian product of
`(artifact, gamma, M, seed, estimator)`. These historical locks and receipts
are evidence of what ran; they are not rewritten to claim the newer scheduler
policy.

For every newly submitted generic `config_schema_version: 2` campaign, each
GPU job requests exactly the two-candidate list
`saffo-a100,apollo_agate`, and each CPU job requests exactly the
four-candidate list `saffo-2tb,agsmall,amdsmall,msismall`. Slurm chooses one
eligible partition from the applicable list. Every experiment remains one job,
each M-specific evaluation job receives one scalar M, and Slurm arrays are not
used. Sixteen strict assemblies each join one common shard with exactly
`M=2,8,32`; one read-only diagnostic job is also run per frozen artifact.
Assemble paths are derived from lock-bound content IDs rather than directory
scans, and diagnostic inputs are read directly from the lock. All five compute
phases use separate append-only receipts to prevent blind duplicate
submissions; selected compute resources are bound into each private execution
receipt but omitted from the anonymous manuscript package.

The companion code repository includes the isolated runnable example
`configs/binary_midpoint_main_v2.json`. It preserves the reported scientific
grid while writing only below `outputs/binary_midpoint_main_v2/`; the sealed
v1 config remains the reproduction record for the numbers in this manuscript.
Before any schema-v2 receipt write or real submission, every final command in
the wave must pass `sbatch --test-only`.

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
Dataset archives, model caches, checkpoints, and frozen probability-map
payloads are not part of this LaTeX package; their immutable identifiers are
retained in the released manifests. The code release provides official dataset
acquisition instructions and archive locations for redistributable large
artifacts.

The checked-in ICLR 2026 style is the latest released working template as of
this source revision. It must be replaced with the official ICLR 2027 template
when that template is released.
