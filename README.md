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
tie-aware AURC. Paper tables display raw AURC and AURC contrasts multiplied by
100 for readability; computations and normalized AURC remain unscaled. The
manuscript is in [`docs/main.tex`](docs/main.tex).

Every canonical assembled row uses schema v2 and contains all three risks,
the complete Dice/nHD/nHD95 midpoint ladder at `M in {2,8,32}`, common
probability-map baselines, and an Exact Dice level-set oracle. The strict
analysis output also uses schema v2.

## Install

Python 3.10 or newer is supported; Python 3.12.4 is the reference environment
recorded by the canonical artifacts. An editable install exposes the module
CLIs and two equivalent console entry points, `selectseg-train` and
`selectseg-binary-eval`. Install the exact recorded dependency set before the
editable package when reproducing reported results:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
```

For development against compatible newer dependencies, `pip install -e
".[dev,plots]"` remains available. On the MSI cluster, load the Python module
and source `scripts/slurm/env.sh` before running jobs.

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
is foreground by a fixed policy. Kvasir-SEG thresholds the maximum JPEG
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

Target checkpoints must exist before the main freeze-and-score campaign. All
shell launchers under `scripts/slurm/submit_*.sh` are retained as
legacy/noncanonical utilities; they do not generate the paper's locked
artifacts. The maintained `submit_extended_binary.sh` helper is only an
ISIC/TN3K checkpoint bootstrap and independent validator. It submits one
training condition per job and one scalar `M` in `{2,8,32}` per evaluation
job:

```bash
bash scripts/slurm/submit_extended_binary.sh
```

Do not use the other shell submitters (`submit_all.sh`, `submit_config_b.sh`,
`submit_selective.sh`, or `submit_quadrature.sh`) to reproduce reported
results. The config/lock-aware Python planners documented below are the
canonical interfaces.

## Reproduce the main campaign

Fine-tune a target model:

```bash
python -m selectseg.train \
  --model clipseg --dataset kvasir --epochs 40 --seed 0 \
  --output-dir outputs/binary_train/kvasir/clipseg/seed-0
```

Repeat this for each target-adapted model path declared by the campaign config.

The immutable protocol is
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

For every new campaign, the current scheduler rule is exact: each GPU training
or freeze job requests the candidate list `saffo-a100,apollo_agate`, and each
GPU-free job requests `saffo-2tb,agsmall,amdsmall,msismall`, always under
account `ssafo`. A comma-delimited candidate request lets Slurm place one job;
it does not create one job per partition. Every `sbatch` must still represent
exactly one experiment, and every quadrature score job must receive exactly
one scalar `M` (`2`, `8`, or `32`); Slurm arrays and fused `M` lists are not
allowed. Partition commands recorded in completed immutable receipts are
historical evidence and remain byte-for-byte unchanged even when they predate
this rule.

The generic planner activates this policy only for a new
`config_schema_version: 2` campaign containing both exact fields
`gpu_partition_candidates: ["saffo-a100", "apollo_agate"]` and
`cpu_partition_candidates: ["saffo-2tb", "agsmall", "amdsmall",
"msismall"]`.  The two fields are inseparable and order-sensitive.  Create a
new campaign ID and lock for such a wave; never edit a sealed v1 config or
receipt in place.

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
   avoids recomputing shared floating-point fields on different nodes. The
   completed seed-0 receipts preserve their historical deterministic rotation
   over `agsmall`, `amdsmall`, and `msismall` under account `ssafo`:

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
   comparisons without repeating inference. The completed receipts use the
   same historical `agsmall`/`amdsmall`/`msismall` rotation.

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
   Brier, ECE, truth/prediction empty rates, and fixed M32 ladder-diversity
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
     --output-dir outputs/binary_final_v3_analysis
   python -m scripts.render_paper_tables \
     --analysis outputs/binary_final_v3_analysis/analysis.json \
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

   The qualitative renderer keeps the immutable content-addressed package and
   can also publish verified stable filenames for the manuscript. It checks
   the selection, render manifest, and every panel digest before replacing the
   seven generated paper files (five panels, one TeX guard, and one manifest):

   ```bash
   python -m scripts.render_binary_qualitative_cases \
     --selection outputs/binary_qualitative_cases/<selection-id>/selection.json \
     --paper-output-dir docs/Figures
   ```

   To republish an existing immutable render without creating a new one, pass
   its manifest explicitly with `--render-manifest`. The generated risk-curve
   sentinel records the complete source-artifact-bundle digest, while the
   qualitative TeX records the full selection and campaign-lock digests.

The analyzer consumes schema-v2 assembly rows and emits JSON, long-form CSV,
and a LaTeX table. Its JSON binds the campaign/config SHA-256, analyzer source
SHA-256, and every assembly manifest, records, sample-order, and cohort digest
without storing machine-absolute paths. It reports raw,
excess, and normalized AURC plus paired image-level percentile-bootstrap
intervals. The manuscript renders all 64 fixed contrasts without filtering and
does not render or interpret the legacy compatibility tail-area fields as
significance tests.
Exact score ties are averaged analytically rather than broken by input order.

The same analysis object records Dice quadrature fidelity to Dice-Exact for
every condition: per-image mean, median, 95th-percentile, and maximum absolute
confidence error; Spearman and Kendall tau-b rank agreement; and the exact
score-match fraction for `M=2,8,32`. The long-form CSV places these fields on
the corresponding Dice-risk/Dice-midpoint rows. They are descriptive numerical
checks and never enter the four fixed contrasts.

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
  --analysis outputs/binary_final_v3_analysis/analysis.json \
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
  --output outputs/binary_final_v3_analysis/public_provenance.json
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

### Locked numerical and deployment sensitivities

The main campaign remains immutable. Two isolated auxiliary protocols reuse
its frozen maps without rerunning model inference. First, submit one M128 job
per locked condition:

```bash
python -m scripts.submit_m128_auxiliary \
  --campaign-lock outputs/binary_campaign/campaign.lock.json
python -m scripts.submit_m128_auxiliary \
  --campaign-lock outputs/binary_campaign/campaign.lock.json --submit \
  --receipt outputs/binary_m128_auxiliary_campaign/submit.receipt.jsonl
```

Each job computes Dice, nHD, and nHD95 jointly for the same condition. M128 is
a high-resolution numerical reference for the boundary losses, not an exact
integral. The strict analyzer requires all 16 auxiliary and all 16 canonical
record paths explicitly; `scripts/render_m128_auxiliary.py` then emits the
appendix table from its analysis JSON:

```bash
python -m scripts.render_m128_auxiliary \
  --analysis outputs/binary_m128_auxiliary_analysis/analysis.json
```

The canonical display-scaled artifact is written once to
`outputs/binary_m128_auxiliary_analysis/rendered_v2/m128_numerical_reference.tex`.
The pre-scaling file under `rendered/` is retained as a superseded provenance
artifact and must not be published as the manuscript table. The renderer never
overwrites either artifact.

Deployment-action sensitivity uses a separate immutable lock and exactly one
job per `(condition, gamma)` for `gamma in {0.3,0.7}`:

```bash
python -m scripts.submit_gamma_sensitivity \
  --auxiliary-lock configs/auxiliary/binary_gamma_sensitivity-v1.lock.json
python -m scripts.submit_gamma_sensitivity \
  --auxiliary-lock configs/auxiliary/binary_gamma_sensitivity-v1.lock.json \
  --submit \
  --receipt outputs/binary_gamma_sensitivity_campaign/submit.receipt.jsonl
```

Every gamma job computes all three risks and all three M32 indexed scores in
one pass. `scripts/analyze_gamma_sensitivity.py` requires the complete 32-run
grid, the 16 canonical assemblies, the auxiliary lock, and the locked main
analysis. This is a sensitivity analysis of the deployed action; it is not a
test-set search for a favorable threshold. Analyze the fixed grid and render
the table with:

```bash
python -m scripts.analyze_gamma_sensitivity \
  --auxiliary-lock configs/auxiliary/binary_gamma_sensitivity-v1.lock.json \
  --auxiliary-inputs <32-explicit-gamma-records.jsonl-paths> \
  --canonical-inputs <16-explicit-canonical-records.jsonl-paths> \
  --canonical-analysis outputs/binary_final_v3_analysis/analysis.json \
  --output-root outputs/binary_gamma_sensitivity_analysis
python -m scripts.render_gamma_sensitivity \
  --analysis outputs/binary_gamma_sensitivity_analysis/b0d4468443fcc46e/analysis.json
```

The display-scaled, write-once renderer output is
`outputs/binary_gamma_sensitivity_analysis/rendered_v3/f6915b8dce868dd7/gamma_sensitivity.tex`.
The earlier artifacts under `rendered/`, `rendered_overfull_fix/`, and
`rendered_v2/` are superseded provenance records and must not replace the v3
manuscript table. The content-addressed subdirectory and no-overwrite policy
remain unchanged.

The analysis-only grouped diagnostic is generated from the 16 explicit
canonical assembly paths with `scripts/analyze_working_risk_diagnostics.py`
and rendered by `scripts/render_working_risk_diagnostics.py`. It reports
action quality, primary-score ranking/accepted-set agreement, and matched
working-risk reliability. One reference mask per image cannot validate the
true conditional mask posterior.

The publication reliability plots use a separate strict, write-once workflow;
they do not modify the locked grouped-diagnostic artifact. The analyzer first
validates the exact 16-condition cohort against the immutable campaign lock,
then retains the ten target-adapted conditions. For each matched pair
Dice-Exact--Dice, nHD-M32--nHD, and nHD95-M32--nHD95, it orders images by
predicted working risk and `sample_id`, forms ten fixed equal-count bins, and
uses 2,000 within-bin image-bootstrap resamples with seed `20260720` for
pointwise 95% intervals on mean observed loss. The five rendered PDFs each
contain two model rows and three matched-loss columns:

```bash
python -m scripts.analyze_matched_risk_reliability \
  --campaign-lock outputs/binary_campaign/campaign.lock.json \
  --inputs '<16 explicit canonical records.jsonl paths>' \
  --output outputs/binary_matched_risk_reliability/analysis.json
python -m scripts.render_matched_risk_reliability \
  --analysis outputs/binary_matched_risk_reliability/analysis.json \
  --output-dir outputs/binary_matched_risk_reliability/rendered
```

These are single-label descriptive plots, not estimates of pointwise
conditional-risk calibration. Their intervals are pointwise rather than
simultaneous, and labels do not fit, tune, or select a score, threshold, bin
count, or image.

The shared-threshold cardinality implications are audited in a separate,
immutable auxiliary workflow, without changing the locked v1 marginal
diagnostic schema:

```bash
python -m scripts.submit_cardinality_diagnostics \
  --auxiliary-lock configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json
python -m scripts.submit_cardinality_diagnostics \
  --auxiliary-lock configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json \
  --submit \
  --receipt outputs/binary_cardinality_diagnostics_campaign/submissions.jsonl
```

Each CPU job processes exactly one frozen condition. For the observed truth
cardinality `k`, it computes the exact shared-threshold bounds
`F_-(k)=Q_p(K<k)` and `F(k)=Q_p(K<=k)` from two order statistics, together
with `E_Q[K]=sum_i p_i`, `Q_p(K=0)=1-max_i p_i`, and a predeclared
SHA-256 pseudo-randomized PIT realization (diagnostic seed `20260720`). After
all 16 jobs finish, pass their paths explicitly to
`scripts/analyze_cardinality_diagnostics.py`, then render with
`scripts/render_cardinality_diagnostics.py`. These pooled, single-label
quantities can falsify aggregate cardinality implications of `Q_p`; they
cannot establish pointwise posterior calibration or validate the coupling.

```bash
python -m scripts.analyze_cardinality_diagnostics \
  --auxiliary-lock configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json \
  --inputs <16-explicit-records.jsonl-paths> \
  --output outputs/binary_cardinality_diagnostics_analysis/analysis.json
python -m scripts.render_cardinality_diagnostics \
  --analysis outputs/binary_cardinality_diagnostics_analysis/analysis.json \
  --output-dir outputs/binary_cardinality_diagnostics_analysis/rendered
```

Scoring overhead is measured separately from inference and artifact I/O. The
completed locked v1 benchmark measured the production joint M32 workload and
Dice-Exact once per condition:

```bash
python -m scripts.submit_binary_runtime \
  --campaign-lock outputs/binary_campaign/campaign.lock.json
python -m scripts.submit_binary_runtime \
  --campaign-lock outputs/binary_campaign/campaign.lock.json --submit \
  --receipt outputs/binary_runtime_campaign/submit.receipt.jsonl
```

The benchmark compares the production joint Dice/nHD/nHD95-M32 workload with
Dice-Exact on the same deterministic 16-map panel. Its wall-clock and peak-RSS
summaries are hardware-dependent measurements, not claims about asymptotic
complexity.

The immutable v2 ladder completes the predeclared runtime experiment without
altering the v1 config or artifacts. It times joint Dice/nHD/nHD95 confidence
at each of `M in {2,8,32}` plus Dice-Exact, with one warm-up and four
Williams-balanced-order measured repetitions. Every timed method receives the same
preloaded deterministic 16-image panel; no confidence or boundary-distance
result is reused across methods. Eight Python workers and one native numerical
thread per worker remain fixed. Preview the exact 16-condition plan, then
submit one CPU job per condition:

```bash
python -m scripts.submit_binary_runtime_ladder
python -m scripts.submit_binary_runtime_ladder --submit \
  --receipt outputs/binary_runtime_ladder_v2_campaign/submit.receipt.jsonl
```

The v2 config and lock are
`configs/auxiliary/binary_runtime_ladder-v2.{json,lock.json}`; the lock SHA-256
is checked by the default submitter. Results are isolated under
`outputs/binary_runtime_ladder_v2/`. After all 16 jobs finish, pass the 16
`records.jsonl` paths explicitly to the compatible strict analyzer:

```bash
python -m scripts.analyze_binary_runtime \
  --campaign-lock outputs/binary_campaign/campaign.lock.json \
  --benchmark-lock configs/auxiliary/binary_runtime_ladder-v2.lock.json \
  --expected-benchmark-lock-sha256 3737c3751fd368f7abf55561493ea2eacbcd3ac788db72925a54cd3d7cdf9b33 \
  --inputs <16-explicit-v2-records.jsonl-paths> \
  --output outputs/binary_runtime_ladder_v2_analysis/analysis.json
python -m scripts.render_binary_runtime \
  --analysis outputs/binary_runtime_ladder_v2_analysis/analysis.json \
  --output-dir outputs/binary_runtime_ladder_v2_analysis/rendered
```

The locked v2 run is complete for all 16 conditions. Its strict report is
`outputs/binary_runtime_ladder_v2_analysis/analysis.json` (analysis ID
`de16d8594844134a`), and the content-addressed renderer emits
`outputs/binary_runtime_ladder_v2_analysis/rendered/binary_runtime.tex`. That
generated file is mirrored exactly at `docs/Tables/binary_runtime.tex`.

The renderer reports median milliseconds/image and images/second for all four
workloads. Peak RSS remains an explicitly non-method-attributable process
high-water mark. Hardware-dependent comparisons should be made within a
condition; cross-node absolute timings are descriptive.

After all sixteen jobs finish, aggregate only an explicit complete input list
and render the content-addressed manuscript table:

```bash
python -m scripts.analyze_binary_runtime \
  --campaign-lock outputs/binary_campaign/campaign.lock.json \
  --inputs <16-explicit-runtime-records.jsonl-paths> \
  --output outputs/binary_runtime_analysis/analysis.json
python -m scripts.render_binary_runtime \
  --analysis outputs/binary_runtime_analysis/analysis.json \
  --output-dir outputs/binary_runtime_analysis/rendered
```

Training-seed robustness is isolated from the seed-0 campaign. The locked
extension contains exactly five datasets by two target architectures by seeds
1 and 2, hence 20 one-GPU training jobs.

**Historical v1 receipt record (immutable).** The completed v1 extension
balanced jobs across single `saffo-a100` and `apollo_agate` assignments. Those
exact requests, runtime partitions, records, and receipts are already sealed:

```bash
python -m scripts.submit_binary_seed_extension --phase train
python -m scripts.submit_binary_seed_extension --phase train --submit \
  --receipt outputs/binary_seed_extension_campaign/train-submissions.jsonl
```

Always reuse that append-only training receipt: it is the duplicate-submission
guard for the twenty jobs already bound to this extension. The checkpoint and
downstream locks intentionally follow the immutable path recorded inside the
spec lock under `outputs/binary_seed_campaign/`.

Do not rewrite the completed v1 train or freeze commands to a comma-delimited
partition request: doing so would invalidate the exact receipt and scheduler
closure. The current `scripts.submit_binary_seed_extension` module is the v1
replay/finalization planner and deliberately preserves those historical
commands; it does **not** implement a future v2 candidate-GPU mode. Any new
seed campaign requires a separate v2 spec, lock, and planner update that uses
`saffo-a100,apollo_agate` for every train and freeze job. This is a forward
migration rule, not a retrospective change to v1 evidence.

After all 20 jobs are terminal, first apply the reviewed post-training record
validator hardening and run its focused tests. This ordering makes the stronger
runtime, GPU-profile, environment, and A100 checks part of the immutable
closure; never create the checkpoint lock with the pre-hardening validator.
Then preview and seal Slurm accounting. The finalizer binds the exact training
receipt, all strict successful-job records, and the existing hash-chained
scheduler-adjustment ledger. It publishes only a path-free summary. A terminal
failure appears in the dry-run preview but is never persisted; stop and design
an audited recovery receipt instead of changing receipt paths:

```bash
python -m scripts.finalize_seed_scheduler_ledger
python -m scripts.finalize_seed_scheduler_ledger --write
sha256sum outputs/public_seed/seed_scheduler_summary.json
```

Run `--write` only when the preview reports 20 successful `COMPLETED` jobs and
no failures. Record the displayed public-summary SHA-256. The checkpoint
planner revalidates the fixed private ledger, public summary, training receipt,
all 20 current records, and their identity-to-job bindings before it can write
the immutable checkpoint lock. It passes the closure's record-set digest into
the writer, which recomputes the same canonical aggregate immediately before
publication. Then submit exactly 20 freeze jobs, and create
the downstream lock only after all frozen artifacts validate:

```bash
python -m scripts.submit_binary_seed_extension --phase checkpoint-lock \
  --expected-scheduler-summary-sha256 <scheduler-summary-sha256> \
  --write-checkpoint-lock
python -m scripts.submit_binary_seed_extension --phase freeze \
  --expected-checkpoint-lock-sha256 <checkpoint-lock-sha256> --submit \
  --receipt outputs/binary_seed_campaign/freeze-submissions.jsonl
python -m scripts.submit_binary_seed_extension --phase downstream-lock \
  --expected-checkpoint-lock-sha256 <checkpoint-lock-sha256> \
  --write-downstream-lock
```

The downstream lock contains separate canonical-compatible seed-1 and seed-2
campaigns. After recording the SHA-256 of
`outputs/binary_seed_campaign/downstream.lock.json`, submit the post-freeze
waves with these fixed append-only receipts:

```bash
python -m scripts.submit_binary_seed_extension --phase common \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> --submit \
  --receipt outputs/binary_seed_campaign/common-submissions.jsonl
python -m scripts.submit_binary_seed_extension --phase score \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> --submit \
  --receipt outputs/binary_seed_campaign/score-submissions.jsonl
python -m scripts.submit_binary_seed_extension --phase diagnose \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> --submit \
  --receipt outputs/binary_seed_campaign/diagnose-submissions.jsonl
```

These are 20 common, 60 score, and 20 diagnostic jobs. Each experiment is an
independent job; arrays are never used. `common`, `score`, and `diagnose` are
separate waves and may overlap, but assembly must wait until all common and
score outputs validate:

**Historical v1 CPU receipt record (immutable).** The completed common, score,
and diagnostic waves retain their submitted single-partition commands. The
completed assembly, analysis, and render jobs requested the exact candidate
list `saffo-2tb,agsmall,amdsmall,msismall`. This order is receipt serialization,
not a priority declaration: Slurm may normalize the displayed list and places
the one experiment on one eligible partition. Before those submissions, the
planner ran `sbatch --test-only` on the combined request and would have aborted
the whole wave before touching its receipt if the candidate pool was
ineligible. These paragraphs document completed v1 provenance; all new CPU
campaign jobs follow the four-candidate rule stated above.

The already submitted common and score waves have one deliberately narrow
scheduler-only maintenance path. Their original wrappers requested 12 hours,
whereas the completed canonical campaign used at most 1,018 seconds for a
common job and 3,783 seconds for a score job. The fixed adjuster therefore
permits only `TimeLimit` 720 -> 180 minutes, leaving 7,017 seconds of headroom
over the larger historical maximum. It is hard-bound to downstream lock
`8aab3572...23fe73`, common receipt `b3c2659d...a3213e`, and score receipt
`cabec33d...633f67f`; together those receipts must resolve to 80 unique
top-level jobs. Preview first:

```bash
python -m scripts.adjust_seed_downstream_timelimits
```

The default is read-only. Use the following only if the preview reports the
exact 20 common plus 60 score jobs, all still `PENDING`, all with
`TimelimitRaw=720`, and unchanged identities, partitions, and commands:

```bash
python -m scripts.adjust_seed_downstream_timelimits --apply
```

`--apply` durably appends a hash-chained intent before issuing any scheduler
update. Its only mutation is `scontrol update JobId=<fixed-receipt-id>
TimeLimit=03:00:00`; after all 80 jobs report 180 minutes it appends a verified
applied event to
`outputs/binary_seed_campaign/downstream-timelimit-adjustments.jsonl`. If an
update fails or the process is interrupted, keep the immutable lock and both
receipts unchanged and rerun the same command after confirming all jobs that
still require adjustment remain pending. Never cancel, resubmit, delete a
receipt, or choose a new receipt path as recovery.

```bash
python -m scripts.submit_binary_seed_extension --phase assemble \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> --submit \
  --receipt outputs/binary_seed_campaign/assemble-submissions.jsonl
```

The dry run prints
`partition_candidates=saffo-2tb,agsmall,amdsmall,msismall`. The 20 printed
commands still contain exactly one strict assembler invocation and therefore
preserve one condition per Slurm job.

After all 20 assemblies finish, the analysis planner revalidates all 30
seed-0/1/2 assemblies and the locked seed-0 point estimates before submitting
its single job. Use the SHA-256 of the canonical seed-0 analysis, and retain
the fixed one-job receipt:

```bash
python -m scripts.submit_binary_seed_extension --phase analyze \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> \
  --canonical-analysis outputs/binary_final_v3_analysis/analysis.json \
  --expected-canonical-analysis-sha256 <canonical-analysis-sha256> --submit \
  --receipt outputs/binary_seed_campaign/analyze-submissions.jsonl
```

This singleton job uses the same four-partition candidate request; its fixed
receipt records the exact comma-delimited command selected before submission.

The completed analysis is fixed at
`outputs/binary_seed_analysis/analysis.json`. Hash it, then submit the one-job
renderer with its own append-only receipt:

```bash
python -m scripts.submit_binary_seed_extension --phase render \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> \
  --seed-analysis outputs/binary_seed_analysis/analysis.json \
  --expected-seed-analysis-sha256 <seed-analysis-sha256> --submit \
  --receipt outputs/binary_seed_campaign/render-submissions.jsonl
```

The renderer is likewise one job with the same candidate pool. Analysis and
render remain sequential because the latter is hash-bound to the completed
analysis bytes.

The renderer writes `outputs/binary_seed_analysis/seed_robustness.tex` and
prints its SHA-256. Publication is a separate local, non-Slurm gate. It checks
both supplied hashes, regenerates the expected TeX with the current renderer,
and only then creates the fixed manuscript file. It never replaces a different
existing table:

```bash
python -m scripts.publish_binary_seed_extension \
  --analysis outputs/binary_seed_analysis/analysis.json \
  --expected-analysis-sha256 <seed-analysis-sha256> \
  --table outputs/binary_seed_analysis/seed_robustness.tex \
  --expected-table-sha256 <seed-table-sha256>
```

Finally, export the portable seed analysis and its write-last provenance
guard. Supply exactly the seven downstream receipts and all 20 explicit
diagnostic summaries; never substitute directory discovery or private locks in
the public artifact. First let the read-only collector validate the fixed
diagnose receipt, its 20 unique submitted job IDs, the locked cell/artifact
bindings, and all descriptor payloads. It inspects only the 20 exact
artifact-ID directories derived from the downstream lock. The NUL-delimited
mode produces quoted-array-safe arguments without writing an intermediate
private path list:

```bash
readarray -d '' -t seed_diagnostic_args < <(
  python -m scripts.collect_binary_seed_diagnostics \
    --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
    --expected-downstream-lock-sha256 <downstream-lock-sha256> \
    --diagnose-receipt outputs/binary_seed_campaign/diagnose-submissions.jsonl \
    --format argv0
)
```

Then pass that exact validated closure directly to the exporter:

```bash
python -m scripts.export_binary_seed_provenance \
  --spec-lock configs/auxiliary/binary_seed_extension-v1.lock.json \
  --expected-spec-lock-sha256 <spec-lock-sha256> \
  --checkpoint-lock outputs/binary_seed_campaign/checkpoints.lock.json \
  --expected-checkpoint-lock-sha256 <checkpoint-lock-sha256> \
  --downstream-lock outputs/binary_seed_campaign/downstream.lock.json \
  --expected-downstream-lock-sha256 <downstream-lock-sha256> \
  --canonical-analysis outputs/binary_final_v3_analysis/analysis.json \
  --expected-canonical-analysis-sha256 <canonical-analysis-sha256> \
  --seed-analysis outputs/binary_seed_analysis/analysis.json \
  --expected-seed-analysis-sha256 <seed-analysis-sha256> \
  --table outputs/binary_seed_analysis/seed_robustness.tex \
  --expected-table-sha256 <seed-table-sha256> \
  --train-receipt outputs/binary_seed_extension_campaign/train-submissions.jsonl \
  --phase-receipt freeze outputs/binary_seed_campaign/freeze-submissions.jsonl \
  --phase-receipt common outputs/binary_seed_campaign/common-submissions.jsonl \
  --phase-receipt score outputs/binary_seed_campaign/score-submissions.jsonl \
  --phase-receipt diagnose outputs/binary_seed_campaign/diagnose-submissions.jsonl \
  --phase-receipt assemble outputs/binary_seed_campaign/assemble-submissions.jsonl \
  --phase-receipt analyze outputs/binary_seed_campaign/analyze-submissions.jsonl \
  --phase-receipt render outputs/binary_seed_campaign/render-submissions.jsonl \
  "${seed_diagnostic_args[@]}" \
  --private-scheduler-ledger outputs/binary_seed_extension_campaign/scheduler-adjustments.jsonl \
  --public-scheduler-summary outputs/public_seed/seed_scheduler_summary.json \
  --public-analysis-output outputs/public_seed/seed_robustness_analysis.json \
  --public-provenance-output outputs/public_seed/seed_provenance.json
```

The exporter requires 162 globally unique job identifiers across the eight
receipts, recomputes the analysis, verifies the rendered table byte-for-byte,
and rejects private paths, scheduler metadata, credentials, or unknown schema
fields before either public destination is written.

Every retry must reuse the exact receipt path shown for its phase; changing a
receipt path defeats duplicate-submission detection. The strict analysis joins
the same held-out cohort across seeds 0/1/2, reports raw AURC without image
pooling or seed-level hypothesis tests, and applies the predeclared Gate C. The
table displays the three seed values and mean $\pm$ sample SD [range], with
AURC contrasts multiplied by 100 only for display. Seed results remain a
descriptive model-stochasticity analysis and are never pooled as extra images
in the seed-0 bootstrap.

The known-posterior mechanism study has a 12-cell pilot and a gated 360-cell
full design. Each coupling--sharpness--morphology--replicate cell is one CPU
job; no sampled masks are persisted:

```bash
python -m scripts.submit_synthetic_posterior --phase pilot
python -m scripts.submit_synthetic_posterior --phase pilot --submit \
  --receipt outputs/synthetic_posterior_campaign/pilot-submissions.jsonl
```

After all 12 pilot manifests exist, create the strict pilot analysis and record
its SHA-256:

```bash
python -m scripts.analyze_synthetic_posterior \
  --lock configs/auxiliary/synthetic_posterior-v1.lock.json \
  --mode pilot \
  --output outputs/synthetic_posterior_analysis/pilot-analysis.json
sha256sum outputs/synthetic_posterior_analysis/pilot-analysis.json
```

Only a successful shared-threshold recovery/runtime gate permits the remaining
348 jobs. Both preview and submission require the pilot analysis and its
explicit expected digest. The submitter reloads all 12 pilot artifacts,
revalidates their locked identities and provenance, and recomputes the gate
before it constructs the full plan. Full submission must reuse the fixed
append-only receipt:

```bash
python -m scripts.submit_synthetic_posterior --phase full \
  --pilot-analysis outputs/synthetic_posterior_analysis/pilot-analysis.json \
  --expected-pilot-analysis-sha256 <pilot-analysis-sha256>
python -m scripts.submit_synthetic_posterior --phase full \
  --pilot-analysis outputs/synthetic_posterior_analysis/pilot-analysis.json \
  --expected-pilot-analysis-sha256 <pilot-analysis-sha256> --submit \
  --receipt outputs/synthetic_posterior_campaign/full-submissions.jsonl
```

The completed 360-cell union is strictly aggregated and rendered with:

```bash
python -m scripts.analyze_synthetic_posterior \
  --lock configs/auxiliary/synthetic_posterior-v1.lock.json \
  --mode complete \
  --output outputs/synthetic_posterior_analysis/complete_analysis.json
python -m scripts.render_synthetic_posterior \
  --analysis outputs/synthetic_posterior_analysis/complete_analysis.json \
  --output-dir outputs/synthetic_posterior_analysis/complete_rendered
```

The study compares known true-posterior risk with the working-posterior score;
it is mechanistic evidence, not a claim that any synthetic coupling represents
clinical annotation variability. AURC regret is multiplied by 100 only in the
rendered table and figure.

The public mirror includes the campaign config and estimator spec, the
immutable lock at `results/campaign.lock.json`, and an identical compatibility
copy at `outputs/binary_campaign/campaign.lock.json` so a clean clone can
validate those hash-bound auxiliary locks and reproduce their exact job plans.
It also includes the redacted
receipt-derived summary at `results/public_provenance.json`, canonical v3
analysis and CSV, all 16 assembly manifests with their per-image
`records.jsonl` files under `results/assembled/`, ten seed-0 training configs,
and the four retained seed-0 histories under `results/training/`. The v3
analysis is a provenance refresh: its statistics and CSV are unchanged from
v2, while its source digest binds the current analyzer after the display-only
AURC scaling update. The immutable prior analysis and provenance are retained
as `results/analysis_v2.json` and `results/public_provenance_v2.json`.

For executable lock validation, the same ten training configs and all 16
small frozen-artifact manifests are also present at the exact
`outputs/binary_train/` and `outputs/binary_artifacts/` paths named by the
immutable locks. The corresponding probability-map arrays and checkpoints are
not duplicated there.

Completed gamma, cardinality, matched-risk reliability, runtime-v1/v2, and
synthetic-posterior analyses are included as well. The portable pilot analysis
and its export sidecar preserve the exact successful Gate evidence that
preceded the 348-cell full submission. Runtime and synthetic
analyses are exported through `scripts/export_portable_analysis.py`: the
statistical payload is unchanged, repository-internal absolute paths are made
relative, and a deterministic sidecar binds the source and portable SHA-256
digests. Raw receipts remain private because they contain scheduler and
machine metadata. Generated paper tables record the analysis JSON SHA-256 in
their source comments. Frozen probability maps, checkpoints, and raw datasets
are not committed to Git; their identities remain hash-bound in the public
manifests, and the source-dataset acquisition rules are given above. Therefore
a clean checkout reproduces the locked analysis and paper tables from released
per-image records, but not model training, inference, or qualitative-image
generation without the separately acquired data and model artifacts. The
locked 48-simulation campaign and all 16 strict assemblies are complete,
covering 20,718 condition-specific rows.

DeepLabV3 is supported for target fine-tuning.  Its external COCO checkpoint
is only meaningful when the dataset vocabulary maps to checkpoint classes
(Pet in the binary benchmark); it is not a zero-shot baseline for polyps or
retinal vessels.

### Rebuild the locked analysis from a clean checkout

The shell enumeration below verifies that exactly 16 committed record files
exist, then passes every path explicitly to the strict analyzer; the analyzer
itself does not discover inputs. The two `cmp` calls certify exact canonical
JSON and CSV reproduction.

```bash
mapfile -t analysis_inputs < <(
  find results/assembled -type f -name records.jsonl -print | LC_ALL=C sort
)
test "${#analysis_inputs[@]}" -eq 16
python -m scripts.analyze_binary \
  --campaign-lock results/campaign.lock.json \
  --inputs "${analysis_inputs[@]}" \
  --bootstrap-samples 10000 --bootstrap-workers 4 \
  --output-dir rebuild/analysis
cmp rebuild/analysis/analysis.json results/analysis.json
cmp rebuild/analysis/main_table.csv results/main_table.csv
python -m scripts.render_paper_tables \
  --analysis rebuild/analysis/analysis.json \
  --output-dir rebuild/Tables
for table in main_results full_target_results complete_results \
  cross_loss_results quadrature_ablation statistical_tests results_complete; do
  cmp "rebuild/Tables/${table}.tex" "docs/Tables/${table}.tex"
done
```

### Build the anonymous analysis artifact v4

The deterministic builder publishes the byte-exact core needed for the
16-record analysis and seven canonical tables together with the mandatory
five-file portable seed release: the public seed analysis, aggregate scheduler
summary, provenance guard, full `seed_robustness.tex`, and compact Gate-C table.
The strict seed-release loader validates the three JSON schemas and their
cross-file joins; the full table must match its provenance hash and source
comment, while the compact table is independently rebuilt from the public seed
analysis. The 51 source files produce exactly 54 archive members. Git history,
external
URLs, identities, private paths, raw job identifiers, checkpoint bytes, NPZ
payloads, and submission-receipt contents remain excluded by explicit
allowlists and fail-closed scans. From this orchestration workspace use:

```bash
python -m scripts.build_anonymous_analysis_artifact build \
  --repo-root github \
  --output output/artifacts/selective-segmentation-analysis-v4.tar.gz
python -m scripts.build_anonymous_analysis_artifact verify \
  output/artifacts/selective-segmentation-analysis-v4.tar.gz
```

Inside a clean public clone, replace `--repo-root github` with
`--repo-root .`. The v4 filename is intentionally distinct from the retained
v3 artifact, and the builder refuses to overwrite either an existing v4 or any
other existing destination. The 16-condition core remains analysis
reproducible from its included per-image records. The seed addition is
verification-level: its public schemas, cross-file hashes, scheduler closure,
and rendered tables can be checked, but its per-image analysis cannot be rerun
because seed records, probability maps, receipts, and checkpoints are not in
the archive. The artifact therefore makes no claim that training or inference
can be rerun without the external datasets and model assets. During anonymous
review, distribute it through the review system without linking the
identity-bearing public repository.

## Verify

```bash
python -m pytest -q tests/test_binary_framework.py \
  tests/test_build_anonymous_analysis_artifact.py \
  tests/test_binary_eval.py tests/test_binary_baselines.py \
  tests/test_binary_boundary.py tests/test_binary_diagnostics.py \
  tests/test_analyze_binary_diagnostics.py tests/test_plot_risk_coverage.py \
  tests/test_binary_artifacts.py tests/test_freeze_binary_maps.py \
  tests/test_download_binary_assets.py \
  tests/test_score_binary_simulation.py \
  tests/test_score_binary_m128_auxiliary.py \
  tests/test_gamma_sensitivity.py tests/test_analyze_gamma_sensitivity.py \
  tests/test_analyze_m128_auxiliary.py \
  tests/test_binary_runtime.py \
  tests/test_cardinality_diagnostics.py \
  tests/test_binary_seed_extension.py \
  tests/test_adjust_seed_downstream_timelimits.py \
  tests/test_finalize_seed_scheduler_ledger.py \
  tests/test_export_binary_seed_provenance.py \
  tests/test_publish_binary_seed_extension.py tests/test_synthetic_posterior.py \
  tests/test_binary_qualitative_cases.py \
  tests/test_analyze_working_risk_diagnostics.py \
  tests/test_render_working_risk_diagnostics.py \
  tests/test_matched_risk_reliability.py \
  tests/test_submit_binary_simulations.py \
  tests/test_assemble_binary_simulations.py \
  tests/test_binary_theory.py \
  tests/test_merge_binary_auxiliary.py tests/test_analyze_binary.py \
  tests/test_render_paper_tables.py tests/test_export_public_provenance.py \
  tests/test_export_portable_analysis.py \
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
selectseg/binary_seed_extension.py  isolated seed-1/2 training and freeze contract
selectseg/synthetic_posterior.py  known-posterior coupling simulator
selectseg/binary_diagnostics.py  streamed marginal/level-set diagnostics
selectseg/cardinality_diagnostics.py  exact shared-threshold cardinality/PIT diagnostics
selectseg/data.py              dataset specifications, validation, transforms
selectseg/models.py            CLIPSeg and DeepLabV3 adapters
selectseg/train.py             target fine-tuning CLI
scripts/submit_binary_simulations.py  lock-driven freeze/common/score/assemble/diagnose planner
scripts/adjust_seed_downstream_timelimits.py  fixed 80-job pending-limit audit ledger
scripts/assemble_binary_simulations.py  strict common+M=2/8/32 assembly
scripts/diagnose_binary_artifact.py  one frozen-artifact diagnostic CLI
scripts/analyze_binary_diagnostics.py  strict campaign-bound diagnostic aggregation
scripts/analyze_binary.py      strict statistical analysis and table export
scripts/export_public_provenance.py  deterministic redacted release provenance
scripts/build_anonymous_analysis_artifact.py  deterministic anonymous core release
scripts/export_portable_analysis.py  hash-bound portable auxiliary analyses
scripts/plot_risk_coverage.py  deterministic tie-aware three-risk curves
scripts/submit_m128_auxiliary.py  one locked M128 condition per CPU job
scripts/analyze_m128_auxiliary.py  strict M32/M128/Exact numerical audit
scripts/submit_gamma_sensitivity.py  one locked condition/gamma per CPU job
scripts/analyze_gamma_sensitivity.py  strict deployment-action sensitivity
scripts/analyze_working_risk_diagnostics.py  grouped loss/ranking diagnostics
scripts/analyze_matched_risk_reliability.py  strict fixed-bin matched-risk analysis
scripts/render_matched_risk_reliability.py  five target-condition reliability figures
scripts/submit_cardinality_diagnostics.py  one exact cardinality job per condition
scripts/analyze_cardinality_diagnostics.py  strict cardinality/PIT aggregation
scripts/render_cardinality_diagnostics.py  compact target-condition diagnostic table
scripts/submit_binary_runtime.py  one fixed-protocol runtime job per condition
scripts/submit_binary_runtime_ladder.py  locked M2/M8/M32/Exact runtime ladder
scripts/analyze_binary_runtime.py  strict hardware-dependent timing summary
scripts/submit_binary_seed_extension.py  immutable v1 seed replay/finalization planner
scripts/publish_binary_seed_extension.py  hash-bound write-once seed-table publisher
scripts/submit_synthetic_posterior.py  gated one-cell-per-job synthetic planner
scripts/analyze_synthetic_posterior.py  strict known-posterior analysis
scripts/select_binary_qualitative_cases.py  mechanical post-analysis diagnostic case selection
scripts/render_binary_qualitative_cases.py  artifact-bound probability/mask panels
scripts/merge_binary_auxiliary.py  exact canonical/strong-score join
scripts/render_paper_tables.py generated submission tables with top-1 marking
tests/                         unit, schema, data, and pipeline tests
docs/                          ICLR manuscript source
```

The earlier multiclass/band implementation remains only in the private
development workspace for provenance.  It is intentionally omitted from the
focused public mirror and is not part of this workflow or its claims.
