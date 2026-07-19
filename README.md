# Selective Segmentation

Research code for **loss-indexed confidence in selective binary
segmentation**.  A foreground probability map `p` and the deployed mask
`Y_hat = {p >= gamma}` induce the midpoint confidence

```text
C[L, M] = -(1 / M) sum_m L({p >= (m - 1/2) / M}, Y_hat).
```

Changing `L` aligns confidence with a target risk; changing `M` changes only
the numerical approximation. The focused benchmark instantiates the same
framework with three losses: foreground Dice, image-diagonal-normalized
penalized full Hausdorff distance (nHD), and its robust pooled 95th-percentile
counterpart (nHD95). Rankings are evaluated against all three risks with
tie-aware AURC. The manuscript is in [`docs/main.tex`](docs/main.tex).

Every canonical assembled row uses schema v2 and contains all three risks,
the complete Dice/nHD/nHD95 midpoint ladder at `M in {2,8,32}`, common
probability-map baselines, and an Exact Dice level-set oracle. The strict
analysis output also uses schema v2.

## Install

Python 3.10 or newer is required. An editable install exposes the module
CLIs and two equivalent console entry points, `selectseg-train` and
`selectseg-binary-eval`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,plots]"
```

`requirements.txt` records the Python package versions used in the current
environment. On the MSI cluster, load the Python module and
source `scripts/slurm/env.sh` before running jobs.

## Build the manuscript

The paper source is tested with pdfTeX 1.40.28 from TeX Live 2025. From
`docs/`, build the bibliography and resolve cross-references with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

With `latexmk` installed, the equivalent command is
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. The checked-in
ICLR 2026 style is the latest released working template; it must be replaced
with the official ICLR 2027 template when that template becomes available.

## Data

The primary experiment uses one native binary task per image.  It never
selects a class from the ground truth.

| key | target | split policy | expected directory |
| --- | --- | --- | --- |
| `pet` | pet including trimap border | official `trainval` / `test` | `data/oxford-iiit-pet` |
| `kvasir` | polyp | deterministic SHA-256-ranked 80/20 split | `data/Kvasir-SEG/{images,masks}` |
| `fives` | retinal vessels | official `train` / `test` | `data/FIVES/{train,test}/{Original,Ground truth}` |
| `isic` | skin lesion | ISIC 2018 Task 1 official train / test | `data/ISIC2018/ISIC2018_Task1*` |
| `tn3k` | thyroid nodule | official `trainval` / `test` | `data/TN3K/extracted/Thyroid Dataset/tn3k` |

Masks are validated as total binary labels at evaluation time. Pet's border
is foreground by a predeclared policy. Kvasir-SEG thresholds the maximum JPEG
mask channel at 128 to reject compression residuals; FIVES maps every nonzero
mask value to foreground. ISIC uses its lossless 0/255 lesion masks. TN3K also
uses a threshold of 128 because its released masks are JPEG-compressed. Dataset
archives, caches, and checkpoints are local artifacts and are excluded from Git.

Download datasets and populate the repo-local model caches with:

```bash
python -m scripts.download_binary_assets
```

FIVES extraction additionally requires `unrar`; every downloaded archive is
checked against the SHA-256 recorded in the script.

Target checkpoints must exist before the main freeze-and-score campaign. The
ISIC/TN3K bootstrap launcher submits one training job per model; it also runs
the older fused evaluator for independent validation, not for the canonical
campaign artifacts:

```bash
bash scripts/slurm/submit_extended_binary.sh
```

## Reproduce the main campaign

Fine-tune a target model:

```bash
python -m selectseg.train \
  --model clipseg --dataset kvasir --epochs 40 --seed 0 \
  --output-dir outputs/binary_train/kvasir/clipseg/seed-0
```

Repeat this for each target-adapted model path declared by the campaign config.

The predeclared protocol is
[`configs/binary_midpoint_main.json`](configs/binary_midpoint_main.json). Its
cohorts are fixed before inference:

| dataset | images per condition | model conditions | assembled rows |
| --- | ---: | ---: | ---: |
| Pet | 3,669 | 4 | 14,676 |
| Kvasir-SEG | 200 | 3 | 600 |
| FIVES | 200 | 3 | 600 |
| ISIC 2018 | 1,000 | 3 | 3,000 |
| TN3K | 614 | 3 | 1,842 |
| **total** | **5,683 unique dataset images** | **16** | **20,718** |

Pet additionally includes the external DeepLabV3 condition. The other four
datasets use CLIPSeg-general, CLIPSeg-target, and DeepLabV3-target. With one
action threshold (`gamma=0.5`), three midpoint counts (`M=2,8,32`), one
deterministic seed (`0`), and one frozen estimator (`midpoint-v1`), the main
campaign contains 16 frozen conditions and exactly 48 simulations.

The workflow separates GPU inference, M-independent CPU work, and independent
quadrature experiments. Its canonical job graph is 16 GPU freezes, 16 common
CPU jobs, 48 M-specific CPU jobs, 16 strict assemblies, and 16 read-only
artifact diagnostics. Every independent experiment is one `sbatch` job; no
Slurm arrays or multi-experiment jobs are used.

1. **Freeze once.** Run the model once for each condition and write immutable,
   content-addressed foreground-probability/truth artifacts. Every job requests
   the combined private partition list `saffo-a100,apollo_agate` under account
   `ssafo`, allowing Slurm to use whichever eligible GPU is available first.
   Preview all 16 jobs, then submit the identical plan:

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase freeze
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase freeze --submit \
     --receipt outputs/binary_campaign/freeze-submissions.jsonl
   ```

2. **Lock the cohort.** After all freeze jobs finish, create one no-overwrite
   campaign lock. Supply exactly one explicit `manifest.json` path for every
   configured condition (16 `--artifact-manifest` flags for the main campaign):

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase lock \
     --artifact-manifest outputs/binary_artifacts/pet/clipseg-general/<artifact-id>/manifest.json \
     --artifact-manifest outputs/binary_artifacts/pet/clipseg-target/<artifact-id>/manifest.json \
     --artifact-manifest '<repeat for the other 14 configured conditions>' \
     --write-lock outputs/binary_campaign/campaign.lock.json
   ```

   Replace every placeholder with a concrete path; the launcher does not
   discover inputs. Lock creation checks the complete configured condition set,
   expected cohort sizes, ordered sample IDs, checkpoints, immutable manifest
   bytes and structure, and the estimator-spec bytes. It records the payload
   hashes declared by those manifests but deliberately does not decompress the
   large payloads on the login node. Each common and M-specific score job instead
   stream-validates every payload hash and array while consuming it.

3. **Compute common fields once.** Each frozen artifact gets one CPU job for
   the three deployment risks, M-independent baselines, and Exact Dice. This
   avoids recomputing shared floating-point fields on different nodes. Jobs
   rotate deterministically over `agsmall`, `amdsmall`, and `msismall` under
   account `ssafo`:

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase common \
     --campaign-lock outputs/binary_campaign/campaign.lock.json
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase common \
     --campaign-lock outputs/binary_campaign/campaign.lock.json --submit \
     --receipt outputs/binary_campaign/common-submissions.jsonl
   ```

4. **Score one simulation per job.** Preview the Cartesian expansion, then
   submit it. Each of the 48 CPU `sbatch` calls receives exactly one
   `(artifact, gamma, M, seed, estimator)` tuple; no Slurm array or fused list
   of `M` values is used. Dice, nHD, and nHD95 confidence are computed from the
   same tuple and shared candidate-boundary distances, preserving paired
   comparisons without repeating inference. These jobs use the same
   deterministic `agsmall`/`amdsmall`/`msismall` rotation.

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase score \
     --campaign-lock outputs/binary_campaign/campaign.lock.json
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase score \
     --campaign-lock outputs/binary_campaign/campaign.lock.json --submit \
     --receipt outputs/binary_campaign/score-submissions.jsonl
   ```

   Every compute phase uses a distinct
   append-only receipt. A receipt records intent before `sbatch`, records the
   returned job ID afterward, and skips tuples already marked submitted. An
   unresolved intent stops resubmission so the operator can inspect Slurm
   instead of creating a blind duplicate. The campaign lock and receipts are
   provenance records; neither is silently overwritten. Each planned CPU
   command includes its selected partition and account, and those exact bytes
   are bound into the corresponding receipt.

5. **Assemble exactly.** After all score jobs finish, preview and submit 16
   independent assembly jobs through the same planner. It derives the sole
   compatible common and `M=2,8,32` manifest paths from content IDs fixed by
   the campaign lock, estimator, axes, and scorer source bytes. It never scans
   an output directory or accepts operator-selected shard paths. Planning
   fails before `sbatch` if a shard is absent or if its manifest, records hash,
   row order, schema, or lock-bound provenance is inconsistent:

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase assemble \
     --campaign-lock outputs/binary_campaign/campaign.lock.json
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase assemble \
     --campaign-lock outputs/binary_campaign/campaign.lock.json --submit \
     --receipt outputs/binary_campaign/assemble-submissions.jsonl
   ```

   Each planned command still invokes the strict standalone assembler for one
   condition only. Assembly is deterministic bookkeeping, not a simulation
   replicate. The 16 jobs rotate over `agsmall`, `amdsmall`, and `msismall`.

6. **Diagnose each frozen artifact.** The diagnose phase reads exactly the 16
   artifact paths and expected manifest digests in the campaign lock and emits
   one independent read-only job per artifact. It also fixes the decision
   threshold from the locked protocol; no path is copied or discovered by a
   glob:

   ```bash
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase diagnose \
     --campaign-lock outputs/binary_campaign/campaign.lock.json \
     --diagnostic-output-root outputs/binary_diagnostics
   python -m scripts.submit_binary_simulations \
     --config configs/binary_midpoint_main.json --phase diagnose \
     --campaign-lock outputs/binary_campaign/campaign.lock.json \
     --diagnostic-output-root outputs/binary_diagnostics --submit \
     --receipt outputs/binary_campaign/diagnose-submissions.jsonl
   ```

   These 16 jobs use the same deterministic three-partition CPU rotation.

   Pixel-weighted Brier score and fixed-bin ECE test marginal foreground
   calibration only. They cannot identify a joint mask posterior or validate
   the comonotone/shared-threshold coupling. Label-dependent descriptors are
   held-out descriptive outputs, never inputs to a confidence score.

   After all diagnostic jobs finish, aggregate them with the same immutable
   lock and an explicit list of all 16 summaries. The command has no directory
   scan or glob fallback; replace every placeholder with one concrete path:

   ```bash
   python -m scripts.analyze_binary_diagnostics \
     --campaign-lock outputs/binary_campaign/campaign.lock.json \
     --inputs \
       'outputs/binary_diagnostics/pet/clipseg-general/<artifact-id>/<diagnostic-id>/diagnostics.json' \
       '<repeat for the other 15 locked conditions>' \
     --output-dir outputs/binary_final_v2_diagnostics \
     --paper-table docs/Tables/binary_diagnostics.tex
   ```

   The strict aggregate revalidates each diagnostic payload (including
   descriptors when present), source-artifact manifest digest, condition,
   ordered sample-ID digest, and cohort count against the lock. It reports
   Brier, ECE, truth/prediction empty rates, and predeclared M32 ladder-diversity
   summaries per condition without pooling or choosing example images.

7. **Analyze and render.** Once all 16 assembled condition artifacts exist,
   invoke the final analyzer with the immutable lock and all 16 paths listed
   explicitly. Directory discovery is available only for incomplete smoke
   analyses; it is rejected in final mode.

   ```bash
   python -m scripts.analyze_binary \
     --campaign-lock outputs/binary_campaign/campaign.lock.json \
     --inputs \
       'outputs/binary_assembled/pet/clipseg-general/<run-id>/records.jsonl' \
       '<repeat for the other 15 locked conditions>' \
     --bootstrap-samples 10000 --bootstrap-workers 4 \
     --output-dir outputs/binary_final_v2_analysis
   python -m scripts.render_paper_tables \
     --analysis outputs/binary_final_v2_analysis/analysis.json \
     --output-dir docs/Tables
   ```

   Publication risk-coverage figures likewise require the explicit 16
   assembled manifest or records paths. Use `--all-indexed` for the complete
   Dice-M32/nHD-M32/nHD95-M32 cross-loss overlay and `--png` only when a raster
   companion is needed:

   ```bash
   python -m scripts.plot_risk_coverage \
     --inputs '<16 explicit manifest.json or records.jsonl paths>' \
     --campaign-lock outputs/binary_campaign/campaign.lock.json \
     --output-dir docs/Figures --all-indexed
   ```

The analyzer consumes schema-v2 assembly rows and emits JSON, long-form CSV,
and a LaTeX table. Its JSON binds the campaign/config SHA-256, analyzer source
SHA-256, and every assembly manifest, records, sample-order, and cohort digest
without storing machine-absolute paths. It reports raw,
excess, and normalized AURC; paired image-level percentile-bootstrap intervals;
two-sided approximate bootstrap tail probabilities; and their within-family
Holm-transformed values. These are evidence summaries, not exact
null-calibrated p-values or a finite-sample error-control claim. Exact score
ties are averaged analytically rather than broken by input order.

The same analysis object records Dice quadrature fidelity to Dice-Exact for
every condition: per-image mean, median, 95th-percentile, and maximum absolute
confidence error; Spearman and Kendall tau-b rank agreement; and the exact
score-match fraction for `M=2,8,32`. The long-form CSV places these fields on
the corresponding Dice-risk/Dice-midpoint rows. They are descriptive numerical
checks and never enter the four contrasts or either Holm family.

The renderer produces exactly six canonical three-loss tables:
`main_results.tex`, `full_target_results.tex`, `complete_results.tex`,
`cross_loss_results.tex`, `quadrature_ablation.tex`, and
`statistical_tests.tex`. It publishes `results_complete.tex` only after all six
files carry the same analysis SHA-256, so a partial render cannot enter the
manuscript.

Export the final public provenance only after every phase receipt is resolved
and the locked analysis and diagnostics are complete. The exporter validates
those bindings, then writes a deterministic whitelist-only summary containing
logical IDs, hashes, and counts---never scheduler commands, job IDs, private
paths, timestamps, or environment identities:

```bash
python -m scripts.export_public_provenance \
  --campaign-lock outputs/binary_campaign/campaign.lock.json \
  --analysis outputs/binary_final_v2_analysis/analysis.json \
  --diagnostics-analysis outputs/binary_final_v2_diagnostics/diagnostics_analysis.json \
  --phase-receipt freeze outputs/binary_campaign/freeze.receipt.jsonl \
  --phase-receipt common outputs/binary_campaign/common.receipt.jsonl \
  --phase-receipt score outputs/binary_campaign/score.receipt.jsonl \
  --phase-receipt assemble outputs/binary_campaign/assemble.receipt.jsonl \
  --phase-receipt diagnose outputs/binary_campaign/diagnose.receipt.jsonl \
  --training-config '<logical-id>' '<train_config.json>' \
  --training-history '<logical-id>' '<history.json>' \
  --base-model clipseg CIDAS/clipseg-rd64-refined \
    999e0328d9e10b484360c477313983f9afdd7050 \
    d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55 \
  --base-model deeplabv3 \
    torchvision/DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 \
    torchvision-0.27.1 \
    cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06 \
  --output outputs/binary_final_v2_analysis/public_provenance.json
```

Repeat the training and base-model flags for every applicable artifact;
training histories are optional, but every target-trained condition requires
an explicitly identity-bound training config. Raw receipts remain private.

### Dual-GPU-partition development pilot

Before the main campaign, run the same commands above with
`configs/binary_midpoint_dual_pilot.json` and pilot-specific lock/receipt paths.
It intentionally contains FIVES and ISIC conditions; each freeze is eligible
on both private GPU partitions through the same combined request. The pilot is
followed by two common jobs, six independent M jobs, and two assemblies.

The pilot fields are deliberately explicit: `expected_dataset_samples`
validates the full evaluation split before subsampling, while `freeze_limit`
selects its first development-only images and must equal
`expected_num_samples`. A limited artifact is a pipeline check only and cannot
replace a full-cohort artifact in the main campaign lock or final analysis.

Threshold robustness at `gamma in {0.3, 0.5, 0.7}` and convergence to the
matched `M=128` reference are reproduced and analyzed with:

```bash
bash scripts/slurm/submit_threshold_robustness.sh
python -m scripts.analyze_auxiliary_experiments
```

The public mirror includes the campaign config and estimator spec, the
immutable lock at `results/campaign.lock.json`, the redacted receipt-derived
summary at `results/public_provenance.json`, the canonical analysis and CSV,
and all 16 assembly manifests with their per-image `records.jsonl` files under
`results/assembled/`. Raw receipts remain private because they contain
scheduler and machine metadata. Generated paper tables record the analysis
JSON SHA-256 in their source comments. Frozen probability maps, checkpoints,
and raw datasets are not committed to Git; their identities remain hash-bound
in the public manifests, and the source-dataset acquisition rules are given
above. The locked 48-simulation campaign and all 16 strict assemblies are
complete, covering 20,718 condition-specific rows.

DeepLabV3 is supported for target fine-tuning.  Its external COCO checkpoint
is only meaningful when the dataset vocabulary maps to checkpoint classes
(Pet in the binary benchmark); it is not a zero-shot baseline for polyps or
retinal vessels.

## Verify

```bash
python -m pytest -q tests/test_binary_framework.py \
  tests/test_binary_eval.py tests/test_binary_baselines.py \
  tests/test_binary_boundary.py tests/test_binary_diagnostics.py \
  tests/test_analyze_binary_diagnostics.py tests/test_plot_risk_coverage.py \
  tests/test_binary_artifacts.py tests/test_freeze_binary_maps.py \
  tests/test_score_binary_simulation.py \
  tests/test_submit_binary_simulations.py \
  tests/test_assemble_binary_simulations.py \
  tests/test_binary_theory.py \
  tests/test_merge_binary_auxiliary.py tests/test_analyze_binary.py \
  tests/test_render_paper_tables.py tests/test_export_public_provenance.py \
  tests/test_data.py
python -m ruff check .
```

The focused modules have no dependency on the legacy selective pipeline.  A
quick dependency check is:

```bash
rg "selectseg\.selective" selectseg/binary_framework.py \
  selectseg/binary_eval.py scripts/analyze_binary.py
```

No matches are expected.

## Repository map

```text
selectseg/binary_framework.py  losses, confidence, tie-aware AURC, bootstrap
selectseg/binary_baselines.py  exact Dice and strong single-map comparators
selectseg/binary_eval.py       strict one-row-per-image binary evaluator
selectseg/binary_artifacts.py  immutable probability/truth artifact I/O
selectseg/freeze_binary_maps.py  freeze-once GPU inference CLI
selectseg/binary_boundary.py  shared nHD/nHD95 digital-surface distances
selectseg/score_binary_common.py  risks, Exact Dice, and common baselines
selectseg/threshold_estimators.py  immutable quadrature specifications
selectseg/score_binary_simulation.py  exactly one locked simulation per run
selectseg/binary_diagnostics.py  streamed marginal/level-set diagnostics
selectseg/data.py              dataset specifications, validation, transforms
selectseg/models.py            CLIPSeg and DeepLabV3 adapters
selectseg/train.py             target fine-tuning CLI
scripts/submit_binary_simulations.py  lock-driven freeze/common/score/assemble/diagnose planner
scripts/assemble_binary_simulations.py  strict common+M=2/8/32 assembly
scripts/diagnose_binary_artifact.py  one frozen-artifact diagnostic CLI
scripts/analyze_binary_diagnostics.py  strict campaign-bound diagnostic aggregation
scripts/analyze_binary.py      strict statistical analysis and table export
scripts/export_public_provenance.py  deterministic redacted release provenance
scripts/plot_risk_coverage.py  deterministic tie-aware three-risk curves
scripts/analyze_auxiliary_experiments.py  threshold/M=128 audit
scripts/merge_binary_auxiliary.py  exact canonical/strong-score join
scripts/render_paper_tables.py generated submission tables with top-1 marking
tests/                         unit, schema, data, and pipeline tests
docs/                          ICLR manuscript source
```

The earlier multiclass/band implementation remains only in the private
development workspace for provenance.  It is intentionally omitted from the
focused public mirror and is not part of this workflow or its claims.
