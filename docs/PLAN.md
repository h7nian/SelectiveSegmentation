# Binary paper experiment plan — submission gate

Updated 2026-07-20. This document replaces the obsolete multiclass/band plan. It is the
execution contract for the current binary paper: five datasets, one fixed deployed mask,
three geometric losses, and loss-indexed confidence. It separates evidence that is already
locked from work that is still proposed. Proposed experiments must not be described in the
paper as completed until their content-addressed artifacts, analysis JSON, and rendered
tables exist.

The immutable seed-0 campaign remains the primary experiment. New work is a robustness or
mechanism extension; it must not retroactively select a different primary score, deployment
threshold, multiplicity family, model condition, or dataset because of a favorable result.

Status notation:

- **DONE**: locked artifacts and rendered results exist.
- **READY**: the scientific design is fixed, but implementation/audit gates may remain.
- **OPTIONAL**: useful evidence that is not required for the first submission-ready build.
- **BLOCKED**: data, identity metadata, or code support is not yet sufficient for a valid run.

---

## 1. Research questions and the claim each experiment may support

### RQ1 — Does deployment geometry change the selective ranking?

For the same frozen probability map and the same hard action
\(\widehat Y=\{i:p_i\geq0.5\}\), compare confidence indexed by:

1. Dice loss: region overlap;
2. normalized penalized Hausdorff loss (nHD): worst boundary displacement;
3. normalized penalized HD95 loss (nHD95): robust percentile boundary displacement.

The claim is deliberately directional but not universal: loss-indexed scores can induce
different rankings, and an indexed score is most naturally assessed under its corresponding
deployment loss. The theory does **not** imply that the matched score wins in every finite
condition.

**Primary evidence:** the two adjacent geometric comparisons, not an all-pairs method race:

- Dice-M32 versus nHD-M32 under Dice and nHD risk;
- nHD-M32 versus nHD95-M32 under nHD and nHD95 risk.

The nonadjacent Dice-versus-nHD95 comparison and all mismatched cells remain descriptive.

### RQ2 — Are the findings numerical artifacts of threshold quadrature?

Measure how \(M\in\{2,8,32\}\) changes score values, rankings, and AURC. Dice-Exact is the
probability-knot oracle and removes Dice quadrature error entirely. Re-score every frozen
condition at \(M=128\) as a high-resolution numerical reference for nHD and nHD95; M128 is
still a quadrature approximation and must never be called exact. For the boundary losses, the
M-ladder and M32--M128 comparison measure empirical numerical stability; neither validates the
working posterior.

The intended claim is limited: M32 is an accurate common-budget approximation in the tested
conditions. Any remaining gap to ideal conditional confidence may still be posterior-model
error.

### RQ3 — Are the conclusions stable to deployment and training choices?

The two highest-value robustness checks are:

- deployment threshold \(\gamma\in\{0.3,0.5,0.7\}\), with 0.5 retained as primary;
- target-model training seeds \(s\in\{0,1,2\}\), with seed 0 retained as the locked primary
  campaign and seeds 1--2 treated as a target-only descriptive checkpoint extension.

These checks answer different questions. Varying \(\gamma\) changes the deployed action while
holding the probability map fixed. Varying the training seed changes the probability map and
therefore tests model stochasticity. Neither check is permission to tune \(\gamma\), choose a
seed, or discard an unfavorable model on the test set. The three checkpoint values are not
pooled into the seed-0 image bootstrap, and three seeds do not estimate the full distribution
of training randomness.

### RQ4 — What can the data say about the shared-threshold working posterior?

Single-label test sets cannot identify the full conditional mask posterior. Diagnostics may:

- detect marginal probability miscalibration;
- compare predicted and observed matched losses at a population/condition level;
- test cardinality implications of \(Q_p\), including the empty-mask branch;
- expose failures in synthetic settings where the true posterior is known.

They may **not** establish that \(Q_p=P(Y\mid X)\), estimate pointwise posterior discrepancy,
or prove the reciprocal-cardinality condition from one annotation per image. Multi-rater data
is the proper future gate for a direct posterior study.

---

## 2. Evidence already complete and gaps still owed

### 2.1 Locked main campaign (**DONE**)

The current campaign contains five native-binary test cohorts and sixteen frozen
dataset--model conditions:

| Dataset | Unique test images | Frozen conditions |
|---|---:|---|
| Oxford-IIIT Pet | 3,669 | CLIPSeg-General, CLIPSeg-Target, DeepLabV3-Target, DeepLabV3-External |
| Kvasir-SEG | 200 | CLIPSeg-General, CLIPSeg-Target, DeepLabV3-Target |
| FIVES | 200 | CLIPSeg-General, CLIPSeg-Target, DeepLabV3-Target |
| ISIC 2018 | 1,000 | CLIPSeg-General, CLIPSeg-Target, DeepLabV3-Target |
| TN3K | 614 | CLIPSeg-General, CLIPSeg-Target, DeepLabV3-Target |

This is 5,683 unique test images and 20,718 condition rows. Ten target-adapted conditions
(five datasets times two architectures) contribute 11,366 condition rows. Reusing a test
image under multiple models or losses does not create additional independent samples.

The completed evidence includes:

- the fixed \(\gamma=0.5\) action and normalized penalized Dice/nHD/nHD95 risks;
- Dice-, nHD-, and nHD95-indexed midpoint scores at \(M=2,8,32\);
- Dice-Exact;
- seven single-map comparator families, giving seventeen fixed score rows in total;
- the complete \(3\times3\) indexed-score/risk matrix;
- exact tie-aware empirical AURC, oracle/random references, and normalized excess AURC;
- all risk--coverage curves;
- 10,000 paired image-bootstrap resamples for every fixed adjacent contrast;
- pointwise percentile intervals for all 64 condition-level contrasts, reported without
  significance calls or multiplicity-adjusted tail areas in the manuscript;
- pixel Brier/ECE, truth/prediction empty rates, and M32 level-set diversity diagnostics;
- Dice-Exact versus midpoint score/rank/AURC fidelity.

The current result supports the main story: the clearest separation is Dice versus nHD under
worst-boundary risk; nHD versus nHD95 changes rankings more modestly and condition-dependently.
It does not support a universal “same-name score always wins” statement.

### 2.2 Important remaining gaps

| Gap | Why it matters | Priority |
|---|---|---|
| Primary inference conditions on target-model seed 0 | The complete seeds 1--2 extension repeats the same protocol only for the ten target conditions; Gate C marks only nHD versus nHD95 under nHD95 risk as training-sensitive, and the checkpoint summaries never enter the seed-0 image bootstrap | P2, **DONE** |
| Deployment-threshold sensitivity | All 32 auxiliary runs and strict analysis are complete; Gate B fired for the nHD--nHD95 comparison | P1, **DONE** |
| Complete five-dataset M128 boundary reference | All 16 jobs and the strict analysis are complete; the result triggered the main-text sensitivity gate | P1, **DONE** |
| Matched-loss/cardinality diagnostics | Working-risk reliability and the exact cardinality/PIT auxiliary are complete; Gate E exposes strong FIVES/ISIC failures without claiming posterior calibration | P1, **DONE** |
| Scoring-time benchmark | Locked v1 is preserved; all 16 isolated M2/M8/M32/Exact v2 jobs, strict analysis, and rendering are complete | P1, **DONE** |
| Visual failure taxonomy | Five mechanically selected, artifact-bound panels are complete and explicitly post-analysis/descriptive | P1, **DONE** |
| Known-posterior stress test | All 360 cells and the strict analysis/render are complete; this isolates coupling error mechanistically but does not validate clinical posteriors | P3, **DONE** |
| No multi-rater or site-shift cohort | Direct posterior agreement and external deployment shift remain untested | P3, data-dependent |

The old three-dataset auxiliary threshold/M128 table is useful provenance but is not a
substitute for the complete protocol: it omits ISIC/TN3K, nHD, and the current locked analysis
structure. It must not be promoted to the main evidence without regeneration.

---

## 3. Estimands and reporting rules

### 3.1 Primary estimands

For condition \(h\), observed loss \(L\), and score \(S\), let

\[
  A_{h,L}(S)=\operatorname{AURC}_{h,L}(S),
\]

where AURC is the exact empirical expectation under uniform random ordering within exact
score ties. Lower is better. For each condition, the four primary contrasts are

\[
\begin{aligned}
  \Delta^{D}_{h} &= A_{h,D}(\text{Dice-M32})-A_{h,D}(\text{nHD-M32}),\\
  \Delta^{H}_{h,DH} &= A_{h,H}(\text{Dice-M32})-A_{h,H}(\text{nHD-M32}),\\
  \Delta^{H}_{h,H95} &= A_{h,H}(\text{nHD-M32})-A_{h,H}(\text{nHD95-M32}),\\
  \Delta^{H95}_{h} &= A_{h,H95}(\text{nHD-M32})-A_{h,H95}(\text{nHD95-M32}).
\end{aligned}
\]

Thus negative values favor the score named on the left. These four signed differences and
their pointwise paired-bootstrap intervals are the fixed primary method contrasts. Raw AURC,
rather than nAURC, is primary. Tables display raw AURC and AURC contrasts after multiplying by
100; this is a presentation-only transformation, and nAURC, estimands, analysis JSON, and all
computations remain on their original scales.

### 3.2 Scientific target versus control conditions

The primary scientific target is the ten target-adapted conditions:

\[
  5\ \text{datasets}\times
  \{\text{CLIPSeg-Target},\text{DeepLabV3-Target}\}.
\]

CLIPSeg-General and Pet DeepLabV3-External are fixed reference controls. They are useful for
revealing empty-action degeneracy, poor transfer, and dependence on segmentation quality, but
they are not additional training replicates. The existing analysis fixes and reports all 64
condition-level contrasts. Keep that full record and do not retroactively remove controls or
filter rows by interval direction. Submission-level conclusions should nevertheless be stated first for the ten target
conditions, with the six controls explicitly described as controls.

### 3.3 Secondary and descriptive estimands

The following are fixed secondary summaries and must not be mixed into the primary family:

- all nine cells of the \(3\times3\) indexed-score/risk matrix;
- raw AURC for all seventeen scores;
- normalized excess AURC when the oracle-to-random denominator is nonzero;
- risk at coverage \(c\in\{0.10,0.25,0.50,0.75,1.00\}\) and the full risk--coverage curve;
- Spearman \(\rho\), Kendall \(\tau_b\), and tie-aware accepted-set agreement between M32 and Dice-Exact;
- mean, median, 95th percentile, and maximum per-image absolute score error for the M-ladder;
- raw native-pixel HD/HD95 as descriptive scale checks only; normalized penalized risks remain
  the cross-image primary outcomes;
- method win counts as compact descriptions, never as binomial tests over “independent”
  conditions.

### 3.4 Estimands for the new robustness work

These definitions must be frozen in an auxiliary protocol before examining new outputs.

**Deployment-threshold robustness.** For \(\gamma\in\{0.3,0.7\}\), recompute the action,
risks, common baselines that depend on the action, and Dice/nHD/nHD95-M32 scores from the same
frozen probability maps. Report:

- each of the four signed contrasts at each \(\gamma\);
- sign agreement with \(\gamma=0.5\);
- the paired change \(\Delta_h(\gamma)-\Delta_h(0.5)\);
- rank correlation of each indexed score across deployment thresholds;
- mean risk and prediction-empty rate, because a “stable” AURC under a much worse action is not
  evidence of stable deployment quality.

Threshold robustness is a sensitivity analysis, not a search for the best \(\gamma\).

**Training-seed robustness.** For each target dataset--architecture pair, retain the three
condition-level estimates from seeds 0, 1, and 2. Report the three values, their mean, range,
and sample standard deviation for raw AURC and each primary contrast. The unit of seed
variation is a trained model, not an image. With only three seeds, use descriptive seed
variation; do not produce a pseudo-precise t-test over seeds or pool seed copies as independent
images. Treat this as checkpoint sensitivity under one fixed training protocol, not as an
estimate of the full distribution of optimization or training-sample randomness.

**Matched-loss reliability.** For each condition and matched loss, define predicted working
risk \(\widetilde r_{L,Q}=-C_{L,32}\) (Dice-Exact is an additional oracle row) and observed
loss \(\ell=L(Y,\widehat Y)\). Report:

- signed bias \(n^{-1}\sum_i(\widetilde r_i-\ell_i)\);
- MAE and RMSE;
- Spearman rank correlation;
- a ten-bin equal-count reliability curve with bootstrap intervals on bin-average observed
  loss;
- the same summaries stratified by empty versus nonempty deployed action when both strata have
  at least 20 images.

Call these **risk-reliability diagnostics**, not pointwise conditional calibration estimates.

**Cardinality diagnostics.** Since \(E_{Q_p}|Y|=\sum_i p_i\), report normalized predicted
foreground mass \(n^{-1}\sum_i p_i\) against observed foreground fraction \(|Y|/n\), their
bias/MAE, and an equal-count reliability plot. Also compute the observed-cardinality rank/PIT
under the discrete \(Q_p\) level-set cardinality distribution, using randomized tie handling
with a recorded diagnostic seed. Pooling these values can falsify aggregate implications of
\(Q_p\); it cannot identify the pointwise conditional cardinality law from one label per
image. Report the exact empty-action identity separately:

\[
  Q_p(Y=\varnothing)=1-\max_i p_i.
\]

**Runtime.** Measure scoring only, excluding model inference and artifact I/O where possible.
For every frozen condition, use one warm-up followed by four balanced timed repeats and report median
seconds/image, images/second, and peak resident memory for M2, M8, M32, and Dice-Exact. Record
CPU model, thread count, library versions, image dimensions, and whether distance transforms
were cached. Do not compare a cached method with an uncached one.

**Synthetic known-posterior stress test.** Under controlled Bernoulli mask marginals, compare
four couplings with the same \(p\): shared threshold (well specified), independent pixels,
local block-shared thresholds, and a spatial latent-copula/two-mode construction. Cross three
marginal sharpness levels and three foreground morphologies. For each generated input,
estimate the true conditional Dice/nHD/nHD95 risks with common-random-number posterior draws;
compare them with \(Q_p\)-indexed risk. Report absolute score error, rank correlation, selective
risk regret, AURC regret, Monte Carlo standard error, and performance stratified by coupling.
This experiment illustrates misspecification; it is not evidence that any synthetic coupling
is the true clinical posterior.

---

## 4. Statistical units and non-negotiable caveats

1. **Images are paired units within a frozen condition.** Every score sees the same probability
   map and every risk uses the same reference; all method intervals must preserve this pairing.
2. **Conditions are not independent replicates.** Models on the same dataset share test images,
   and three risks reuse each image. A 15/16 win count is descriptive, not a sample size of 16.
3. **Pixels and surface points are not inferential units.** Pixel ECE and surface quantiles are
   image-level diagnostics after aggregation; they do not justify pixel- or point-level
   confidence intervals for AURC.
4. **Datasets are the strongest external replication unit, but there are only five.** Do not
   fit a formal cross-dataset random-effects model and overinterpret its standard error.
5. **Seed variation and image variation are different.** Paired image bootstrap intervals
   condition on a trained checkpoint. Seed-1/2 results must be summarized across checkpoints,
   not injected as extra image rows into the seed-0 bootstrap.
6. **Patient clustering is preferred when identifiers exist.** TN3K contains multiple images
   for some patients, and other medical releases may contain frame/patient dependence. Before
   final inference, recover released patient/group identifiers where possible and rerun a
   group-cluster bootstrap. If identifiers cannot be reconstructed without guessing, retain
   the paired image-bootstrap analysis and state that its independence approximation can be
   optimistic.
7. **Intervals are pointwise.** They condition on one fitted checkpoint and are not
   simultaneous, patient-clustered, or family-wise-error guarantees. The manuscript makes no
   null-calibrated \(p\)-value or significance claim.
8. **Single annotation is not a posterior sample set.** Neither favorable Brier/ECE nor a flat
   pooled PIT validates \(Q_p\), Jaccard-Wasserstein agreement, or reciprocal-cardinality
   agreement pointwise.
9. **Empty actions are substantive, not missing data.** Keep them in AURC, apply the declared
   loss conventions, report their frequency, and show their reliability separately when the
   stratum is large enough.
10. **No test-set model selection.** The final epoch, \(\gamma=0.5\), M32 primary budget,
    score set, loss definitions, and comparator conventions remain fixed regardless of an
    auxiliary result.

---

## 5. Priority tiers and execution order

### P0 — Protect and audit the completed campaign (**DONE; recheck before submission**)

- Keep `configs/binary_midpoint_main.json`, its campaign lock, frozen artifact manifests, and
  generated analysis immutable.
- Verify every expected sample count, sample-order digest, checkpoint digest, source digest,
  and final table source hash.
- Run the complete test suite, repository sync checks, LaTeX build, unresolved-reference check,
  and a visual PDF inspection.
- Confirm every auto-generated table is generated from the final `analysis.json`; never hand
  edit numerical TeX.
- Archive Slurm receipts and export only portable provenance, not private absolute paths or
  checkpoints.

### P1 — Cheap five-dataset robustness and diagnostics (**DONE**)

1. Generalize the auxiliary scorer without weakening the strict validator for the primary
   campaign. The existing main submitter deliberately accepts only \(\gamma=0.5\),
   \(M=2,8,32\), seed 0; do not silently edit that lock contract.
2. **DONE:** re-score all sixteen frozen artifacts at \(M=128\), computing Dice, nHD, and
   nHD95 jointly in one CPU job per condition. The strict analysis is
   `outputs/binary_m128_auxiliary_analysis/analysis.json`. M128 remains a high-resolution
   reference; M32 remains the common-budget primary estimator.
3. **DONE:** re-score all sixteen frozen artifacts at \(\gamma=0.3\) and 0.7 with M32 for all three
   indexed losses and all action-dependent references. Each \((\text{condition},\gamma)\) is
   one job and computes all three losses in a single artifact pass.
4. **DONE:** keep the existing locked v1 diagnostic immutable and run the
   independent exact-cardinality auxiliary workflow once per artifact. It emits exact
   (F_-(k),F(k)), fixed-seed randomized-PIT, foreground-mass error, and the empty-mask identity.
   Matched-loss reliability remains in the separate grouped working-risk diagnostic.
5. **DONE:** the immutable v1 M32/Exact benchmark is preserved. A separate
   hash-locked v2 protocol times joint
   Dice/nHD/nHD95 at (M\in\{2,8,32\}) and Dice-Exact with one warm-up and
   four Williams-balanced-order repetitions on the identical preloaded panel.  It
   preserves the v1 config and artifacts, emitted exactly sixteen one-condition
   CPU jobs, and writes only to `outputs/binary_runtime_ladder_v2/`. All jobs,
   the strict analysis, and the rendered replacement table are complete.
6. **DONE:** select qualitative cases mechanically by fixed numerical score/risk-disagreement
   rules, not by visual appeal: one Dice-versus-nHD case, one nHD-versus-nHD95 case, one
   empty-action case, and one working-risk gap per dataset when available. These are explicitly
   post-analysis diagnostic examples, not representative samples.
7. Analyze and render only after all expected manifests are present and hashes match.

P1 requires no model inference and no GPU: use the already frozen native-resolution maps.

### P2 — Target-model seed extension (**DONE; 162/162 jobs verified**)

- **DONE:** train seeds 1 and 2 for CLIPSeg-Target and DeepLabV3-Target on each of the five
  datasets in twenty independent GPU jobs.
- **DONE:** freeze exactly one artifact and run the enhanced diagnostic for each new
  checkpoint in twenty independent jobs per phase.
- **DONE:** for each frozen artifact, run common scoring, M2/M8/M32 score shards, and
  assembly as separate CPU jobs.
- **DONE:** analyze the complete three-checkpoint target cohort in its isolated output tree
  and render the hash-bound full appendix table plus the compact Gate C main-text table.
- Use the identical data split, final-epoch rule, optimizer schedule, prompts, action threshold,
  loss conventions, and score definitions as seed 0.
- Analyze seed robustness in a separate output tree. Do not merge records into the primary
  seed-0 analysis or recompute its pointwise intervals.
- **DONE:** export 30 path-free seed-0/1/2 condition records with allowlisted manifests, a
  replay lock, and a write-last completion guard; the anonymous artifact recomputes the seed JSON and both seed tables
  from these records and verifies all three outputs byte-for-byte in one command.
- After writing the seed downstream lock, freeze the common scorer, simulation scorer,
  assembler, analyzer, and renderer sources until publication finishes.  Their source digests
  participate in content identities; changing them midway would make completed shards
  incompatible with later planner expectations even when the parent artifact lock is unchanged.

This is the highest-value GPU addition. A third architecture is lower priority than measuring
the stochastic variation of the two architectures already used.

### P3 — Mechanism and external-validity extensions (**SYNTHETIC DONE / data extensions BLOCKED**)

**Known-posterior synthetic study.** The 12-job pilot (four couplings by three
sharpness levels, one morphology, one replicate) passed Gate F; all 348 remaining cells,
the strict complete analysis, and the manuscript render are finished. The complete
360-job matrix:

\[
  4\ \text{couplings}\times3\ \text{sharpness levels}\times
  3\ \text{morphologies}\times10\ \text{replicates}=360.
\]

Each replicate is one independent Slurm job and writes only aggregate sufficient statistics
plus its simulation manifest; do not persist millions of sampled masks.

**Multi-rater data.** This remains blocked until a licensed, genuinely multi-annotation binary
cohort and reliable image/rater identities are available. Freeze a separate protocol before
looking at outcomes. The primary diagnostic should compare observed foreground-count spread
and loss distributions with the product and shared-threshold posterior predictions. Do not
substitute repeated augmentations or model seeds for human annotations.

**Site shift/OOD and extra-budget confidence.** A site-shift cohort, test-time augmentation,
MC dropout, ensembles, or a learned failure predictor may be added only as a separately labeled
information-budget extension. The matched primary table remains single-map. Do not delay the
core submission merely to make the baseline list longer.

---

## 6. Full job matrix

Every row below means one independent Slurm job per experimental unit. Arrays are intentionally
not used. Phase submission occurs in waves after the preceding manifests pass validation.

| Package / phase | Experimental unit | GPU jobs | CPU jobs | Dependency | Isolated output root |
|---|---|---:|---:|---|---|
| Main freeze (**DONE**) | 16 dataset--condition pairs | 16 | 0 | checkpoints/released models | `outputs/binary_artifacts/` |
| Main common (**DONE**) | 16 frozen artifacts | 0 | 16 | main lock + freeze | `outputs/binary_common_scores/` |
| Main M-score (**DONE**) | 16 artifacts × 3 M values | 0 | 48 | main lock + freeze | `outputs/binary_simulations/` |
| Main assemble (**DONE**) | 16 conditions | 0 | 16 | common + all 3 M shards | `outputs/binary_assembled/` |
| Main diagnose (**DONE**) | 16 frozen artifacts | 0 | 16 | main lock + freeze | `outputs/binary_diagnostics/` |
| **Main total** | 112 independent compute jobs | **16** | **96** | sequential phase gates | immutable current roots |
| P1 M128 reference (**DONE**) | 16 frozen artifacts; all three losses computed jointly | 0 | 16 | frozen maps + main lock | `outputs/binary_m128_auxiliary/` |
| P1 threshold sensitivity (**DONE**) | 16 artifacts × 2 new \(\gamma\) values; all three losses jointly | 0 | 32 | frozen maps + auxiliary lock | `outputs/binary_gamma_sensitivity/` |
| P1 exact cardinality diagnostics (**DONE**) | 16 frozen artifacts | 0 | 16 | frozen maps + diagnostic lock | `outputs/binary_cardinality_diagnostics/` |
| P1 runtime v1 (**DONE**) | 16 frozen artifacts | 0 | 16 | fixed CPU environment | `outputs/binary_runtime/` |
| P1 runtime ladder v2 (**DONE**) | 16 frozen artifacts; joint M2/M8/M32 plus Dice-Exact | 0 | 16 | immutable v2 lock + frozen maps | `outputs/binary_runtime_ladder_v2/` |
| P1 aggregate/render (**DONE locally**) | five strict analysis/render pairs | 0 | 0 | all 96 P1 compute manifests | isolated analysis roots |
| **P1 completed total** | 96 independent Slurm compute jobs + 10 local aggregation steps | **0** | **96** | no new inference | completed isolated roots |
| **P1 remaining total** | none | **0** | **0** | -- | -- |
| P2 train (**DONE; 20/20 verified**) | 5 datasets × 2 target architectures × 2 seeds | 20 | 0 | data audit passed | `outputs/binary_train_seed_extension/` |
| P2 freeze (**DONE; 20/20 verified**) | 20 new checkpoints | 20 | 0 | corresponding train job | `outputs/binary_seed_artifacts/` |
| P2 common (**DONE; 20/20 verified**) | 20 new frozen artifacts | 0 | 20 | seed-specific lock | `outputs/binary_seed_common/` |
| P2 M-score (**DONE; 60/60 verified**) | 20 artifacts × 3 M values | 0 | 60 | seed-specific lock | `outputs/binary_seed_simulations/` |
| P2 assemble (**DONE; 20/20 verified**) | 20 seed conditions | 0 | 20 | common + all M shards | `outputs/binary_seed_assembled/` |
| P2 diagnose (**DONE; 20/20 verified**) | 20 new frozen artifacts | 0 | 20 | enhanced diagnostic spec | `outputs/binary_seed_diagnostics/` |
| P2 analyze/render (**DONE; 2/2 verified**) | one analysis + one render | 0 | 2 | all P2 manifests | `outputs/binary_seed_analysis/` |
| **P2 completed total** | 162 independent Slurm jobs | **40** | **122** | train → freeze → score → analyze/render | completed seed-extension roots |
| P3 synthetic pilot (**DONE**) | 4 couplings × 3 sharpness levels | 0 | 12 | simulator tests | `outputs/synthetic_posterior_pilot/` |
| P3 synthetic full (**DONE**) | remaining cells of 360-job design | 0 | 348 | pilot gate passed | `outputs/synthetic_posterior_main/` |
| P3 synthetic analyze/render (**DONE**) | one analysis + one render | 0 | 2 | all 360 manifests | `outputs/synthetic_posterior_analysis/` |
| **P3 synthetic total** | 362 jobs including analysis/render | **0** | **362** | pilot before expansion | synthetic roots only |

The main, P1, and P2 totals do not count the non-Slurm lock-writing step. Qualitative panel
assembly may run inside the P1 analysis job; if memory makes that unsafe, split it into five
dataset jobs and record the increase in the receipt.

---

## 7. Slurm queues and resources

### GPU work

Use the private queues `saffo-a100` and `apollo_agate`.  Every new v2 training or freeze
experiment is its own job, retains both candidates, and deterministically alternates which
queue appears first; Slurm places that one job on one eligible queue.  Never bundle multiple
experiments into one allocation.

| Phase | GPUs | CPUs | Memory | Wall time | Queue policy |
|---|---:|---:|---:|---:|---|
| Target training | 1 | 16 | 64 GB | 24 h | `saffo-a100,apollo_agate`, account `ssafo` |
| Freeze probability maps | 1 | 8 | 48 GB | 12 h | `saffo-a100,apollo_agate`, account `ssafo` |

The maintained bootstrap wrappers and generic schema-v2 planner explicitly request one A100
(`--gres=gpu:a100:1`) with both candidate queues; scheduler acceptance for both private queues
was checked on 2026-07-19.  Completed v1 seed receipts retain their historical single-partition
assignments and the v1 seed replay/finalization planner does not masquerade as a v2 launcher.
Keep the realized partition recorded in the locked job specification rather than silently
migrating or duplicating a running campaign, and do not weaken `--require-cuda`.
Record partition, node, GPU model, driver, CUDA, and package versions in each manifest.  A
hardware difference may affect throughput; it must not change deterministic sample identity or
protocol settings.

### CPU work

Use the complete CPU candidate set `amdsmall,agsmall,msismall,saffo-2tb`.  Schema-v2 jobs
deterministically rotate the first three general queues by experiment index and always leave
the private `saffo-2tb` queue last as a fallback.  This avoids giving a delayed private queue
first preference across an entire wave while retaining all four candidates in every single-job
submission:

| Phase | CPUs | Memory | Wall time |
|---|---:|---:|---:|
| Common or M-specific score | 8 | 24 GB | 12 h |
| Enhanced diagnostic / threshold sensitivity | 8 | 24 GB | 12 h |
| Runtime benchmark | 8 | 24 GB | 4 h |
| Assembly | 1 | 4 GB | 30 min |
| Synthetic replicate | 4 | 16 GB | 4 h pilot cap |
| Analysis/render/tests | 4--8 | 16--32 GB | 2 h |

Use no blocking submission script that waits for hours. Submit a wave, retain the append-only
receipt, inspect `squeue`/terminal manifests, and submit the dependent wave only after every
expected predecessor is green. A failed job is resubmitted by exact identity; never rerun an
entire phase blindly.

Schema v2 has two mutually exclusive authorization policies. A dedicated
`scheduler-preview-only` fixture remains permanently fail-closed: dry-run and
`--scheduler-preflight-only` may print or validate commands but cannot open a receipt, create a
post-freeze lock, reconcile, recover, retry, or submit. The 16-condition execution config
`configs/binary_midpoint_main_v2.json` may use `scientific-input-locked` only after
`configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json` has been sealed and its exact
SHA-256 placed in the config. That root binds the real loader order and image/mask bytes for all
five datasets, every condition checkpoint, exact base-model files, freeze source closure, and
runtime environment. The science projection excludes only operational policy, paths, partition
candidates, and the root binding itself, avoiding a config/lock hash cycle without dropping a
scientific field.

Build the large dataset components as five independent CPU jobs---one experiment per job, all
requesting the complete four-partition candidate list---then build the small components and root
lock without overwrite:

```bash
python -m scripts.submit_scientific_input_components --scheduler-preflight-only
python -m scripts.submit_scientific_input_components --submit \
  --receipt outputs/binary_midpoint_main_v2/scientific_inputs/dataset-build-receipt.jsonl

SCIENCE_DIR=configs/scientific_inputs/binary-midpoint-main-v2
python -m selectseg.scientific_inputs build-base-models \
  --seed-extension-lock configs/auxiliary/binary_seed_extension-v1.lock.json \
  --output "$SCIENCE_DIR/base-models.json"
python -m selectseg.scientific_inputs build-checkpoints \
  --config configs/binary_midpoint_main_v2.json \
  --output "$SCIENCE_DIR/checkpoints.json"
python -m selectseg.scientific_inputs build-environment \
  --output "$SCIENCE_DIR/environment.json"
```

The source component must explicitly list the nine freeze-bearing files documented in the root
README; `build-root` then receives exactly five `--dataset-component` bindings plus the source,
base-model, checkpoint, and environment components. Before changing the execution policy, run
the authoritative audit rather than relying on metadata-only verification:

```bash
SELSEG_SCIENCE_LOCK_SHA256=17d30fc18b496c7062acfcec9a09ec8bd6f796339d132bd99f9a6cffad5b2cf0
python -m selectseg.scientific_inputs verify \
  --lock configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json \
  --expected-sha256 "$SELSEG_SCIENCE_LOCK_SHA256" --mode full
```

The planner uses fast verification before any planning, preflight, receipt, or submission. Each
freeze compute node uses consume verification: small bindings and cohort-selection files are
fully hashed before inference, and each large image/mask is verified on the same unavoidable
read that feeds the model. A successful locked freeze emits a schema-v3 artifact manifest; the
explicit 16-manifest `lock` phase then emits a schema-v2 post-freeze campaign lock. Downstream
jobs accept only that lock, validate frozen payloads while consuming them, and retain one scalar
M per score job.

Every real phase uses its sole canonical append-only receipt at
`outputs/binary_midpoint_main_v2/receipts/<phase>.jsonl`. Submission performs a whole-wave
`sbatch --test-only` before recording intent and issuing real jobs. Reconciliation only appends
observed terminal facts; a dangling intent requires explicit identity-checked
`--recover-submitted-job-id`; terminal failures require an exact `--retry-failed-job-id`, and
an actual `sbatch` failure requires `--retry-submission-failure`. Never change receipt paths or
blindly rerun the phase. The sealed v1 evidence remains the record for reported numbers until a
schema-v2 replay independently reaches its full terminal gates. A future seed-v2 campaign still
requires a separate auxiliary ID, seeds, paths, worker, runtime-attestation schema, and
no-overwrite lock; do not seal one merely to replay seeds 1 and 2 without scientific need.

---

## 8. Artifact and analysis isolation

1. **Never overwrite the main campaign.** The roots in
   `configs/binary_midpoint_main.json` are read-only inputs to extensions.
2. **One campaign ID per protocol.** Use distinct IDs such as
   `binary-gamma-g03-v1`, `binary-gamma-g07-v1`, `binary-seed1-v1`,
   `binary-seed2-v1`, and `binary-synthetic-coupling-v1`.
3. **One seed per campaign lock for the extension.** This avoids ambiguous duplicate condition
   keys while preserving the paper-facing names `CLIPSeg-Target` and `DeepLabV3-Target`.
4. **Content-address every artifact.** Manifest identity must include campaign/config digest,
   estimator digest, code/source digest, checkpoint digest, dataset/split identity, ordered
   sample digest, \(\gamma\), M, and seed where applicable.
5. **Locks are write-once and explicit.** Build them from the exact list of frozen manifests;
   assembly derives expected inputs from the lock and never discovers shards by glob.
6. **Receipts are phase-specific and append-only.** Preserve submitted, completed, failed, and
   replacement identities.
7. **Do not mix common and threshold-dependent fields.** A change in \(\gamma\) invalidates the
   action, observed losses, SDC/foreground-entropy-like references, and indexed score action;
   recompute all affected quantities together.
8. **Do not copy generated numbers by hand.** Analysis JSON is the single numerical source;
   table and figure scripts print its SHA-256 in every generated TeX/PDF artifact.
9. **No private material in the public artifact.** Export portable manifests and hashes, not
   access tokens, cluster paths, private checkpoints, environment secrets, or licensed data.
10. **Archive failure cases reproducibly.** Qualitative panels record sample ID, condition,
    score/risk values, selection rule, probability-map digest, and rendering script version.

Before any P1/P2 submission, add tests that prove the main strict protocol is unchanged, new
campaign identities cannot collide with main outputs, and the auxiliary assembler rejects
missing, duplicate, stale, or cross-campaign shards.

---

## 9. Paper placement

### Main paper

- Keep the framework schematic in the method section.
- Keep the primary result table with **methods as rows and datasets as columns**. Within each
  risk/model block, color every exactly best unrounded result dark blue; ties at the unrounded
  precision share the mark.
- Keep the two adjacent-geometry result paragraphs as the main narrative: region-to-worst
  boundary, then worst-to-robust boundary.
- Add at most one compact robustness sentence/table to the main paper after P1/P2. It should
  report threshold/seed direction retention, not another wall of AURCs.
- If space permits, add one compact qualitative panel showing the same action under Dice, nHD,
  and nHD95 risk. Otherwise place it first in the appendix.
- Do not add an equation number merely to support an experimental definition that is never
  referenced.

### Appendix

Retain the current complete panels and add generated artifacts in this order:

1. complete seventeen-score raw AURC tables;
2. full \(3\times3\) cross-loss matrix;
3. all 64 locked paired-bootstrap contrasts;
4. M2/M8/M32, Dice-Exact, and complete M128 boundary-loss fidelity;
5. five-dataset \(\gamma=0.3/0.5/0.7\) sensitivity (`threshold_robustness.tex`);
6. three-seed target-model robustness (`seed_robustness.tex`);
7. risk-reliability, cardinality/PIT, empty-action, and marginal diagnostics
   (`binary_diagnostics_v2.tex` plus reliability figures);
8. scoring runtime and memory (`binary_runtime.tex`);
9. mechanically selected post-analysis qualitative/failure panels;
10. known-posterior synthetic results if P3 passes its pilot;
11. all risk--coverage curves, grouped by dataset with the same method colors across figures.

The old `auxiliary_experiments.tex` remains noncanonical until regenerated under the complete
five-dataset protocol. Auto-generated filenames may differ, but the content order and status
labels above should remain.

---

## 10. Decision gates

### Gate A — Integrity before new compute

Proceed only if the full test suite is green, the main artifacts match their lock, all sixteen
assembled conditions have the expected sample count, generated tables reproduce exactly, and
the PDF compiles without unresolved references/citations. A provenance mismatch blocks all new
analysis until explained; it is not repaired by editing a manifest.

### Gate B — Threshold sensitivity

Keep \(\gamma=0.5\) primary regardless of the result. Promote threshold dependence from an
appendix sensitivity to a main-text limitation if either:

- the condition-macro mean of any of the four primary contrasts changes sign at 0.3 or 0.7; or
- at least three of the ten target-adapted conditions reverse the \(\gamma=0.5\) contrast
  direction for the same comparison.

Otherwise report the compact direction-retention count in the main text and keep full values in
the appendix. Always report concomitant mean-risk and empty-rate changes.

**Observed gate outcome (2026-07-20): fired.** Both nHD--nHD95 macro contrasts change sign at
\(\gamma=.3\), and the nHD-risk macro contrast also changes sign at \(\gamma=.7\). Under
nHD95 risk, three of ten target conditions reverse at \(\gamma=.3\). The Dice--nHD macro
directions persist at both auxiliary thresholds, with zero condition reversals under nHD risk.
Keep \(\gamma=.5\) primary, move the threshold warning into the main Results/Limitations, and
retain the complete action-quality, ranking, accepted-set, and contrast tables in the appendix.

### Gate C — Training-seed sensitivity

Do not call the result seed-robust merely because seed 0 lies near the three-seed mean. For each
dataset--architecture--contrast, inspect all three signs and the full range. If seed 0 is not
the majority direction, or if at least three of ten target conditions show a seed-dependent
direction reversal for one comparison, state that comparison as training-sensitive and move
the three-seed table into the main results. Otherwise summarize direction retention in the main
text and leave detailed seed values in the appendix.

**Observed gate outcome (2026-07-20): fired for nHD versus nHD95 under nHD95 risk.** Five
of ten target conditions reverse direction across the three checkpoints, and seed 0 is not in
the majority direction for three cells: ISIC/DeepLabV3-Target, Pet/DeepLabV3-Target, and
TN3K/CLIPSeg-Target. The corresponding reversal counts for Dice--nHD under Dice, Dice--nHD
under nHD, nHD--nHD95 under nHD, and nHD--nHD95 under nHD95 are respectively
\(1/10,0/10,0/10,5/10\). On the manuscript's \(\times100\) display scale, the fifteen values
in the five reversal cells range from \(-0.0856\) to \(+0.0236\). The largest cross-seed range
for the affected contrast is \(4.4756\) on FIVES/CLIPSeg-Target, whose three values remain
positive (\(+7.3446,+6.8177,+2.8690\)). Call only nHD versus nHD95 under nHD95 risk
training-sensitive; show its compact Gate C table in the main Results and retain the complete
three-seed table in the appendix. These three checkpoints remain descriptive and do not
support inference over training randomness.

### Gate D — Quadrature

M32 remains the common-budget primary method to keep all three losses symmetric. Dice-Exact is
the Dice numerical oracle; M128 is only a high-resolution reference for nHD/nHD95. If replacing
Dice-M32 by Dice-Exact reverses any primary Dice-versus-nHD conclusion or materially changes its
paired interval, report both versions next to the primary contrast. For each boundary score,
report M32--M128 per-image error, rank agreement, and matched-risk AURC gap. If any target
condition has Spearman \(\rho<0.98\), an absolute matched-risk AURC gap above \(10^{-3}\), or a
primary contrast reversal when M128 is substituted, move the sensitivity into the main text;
otherwise keep the full table in the appendix and report only the aggregate fidelity range.
Do not claim that Exact Dice or M128 removes posterior discrepancy.

**Observed gate outcome (2026-07-20): fired.** Across target conditions, nHD M32--M128
Spearman agreement is .973--.998 and the largest matched-risk AURC gap is .456 on the
manuscript's \(\times100\) display scale; nHD95 agreement is .951--.996 and the largest gap is
.324. Both largest gaps occur on FIVES. Keep M32 primary for equal budgets, add the compact
warning to the main Results, and retain the complete table in the appendix.

### Gate E — Diagnostics

There is no “diagnostic passed, therefore \(Q_p\) is correct” outcome. Gross matched-risk bias,
strong cardinality/PIT nonuniformity, or an empty-action failure must be highlighted as a
limitation/failure mode. Favorable aggregate diagnostics are reported as non-refutation only.

**Observed gate outcome (2026-07-20): unfavorable diagnostics highlighted.** FIVES target
conditions have randomized-PIT KS distances .644/.459 and FIVES CLIP-T has a +26.21
percentage-point empty-probability bias. ISIC CLIP-T assigns zero level-set mass to the
observed truth cardinality in 15.4\% of images. These are aggregate single-label
falsifications, not posterior-calibration estimates.

### Gate F — Synthetic expansion

Expand beyond the 12-job pilot only if:

- the shared-threshold well-specified case agrees with the direct posterior estimate within
  three Monte Carlo standard errors;
- common-random-number estimates reproduce under a duplicate test seed;
- all loss and empty-mask conventions match the real pipeline exactly; and
- the maximum pilot runtime fits the four-hour CPU cap with at least 25% headroom.

Synthetic alternatives are allowed to refute the method. Do not tune their coupling strength
to manufacture monotone degradation.

**Observed pilot outcome (2026-07-20): passed.** All 12 manifests validate against the lock;
shared-threshold posterior discrepancies are exactly zero, Exact Dice stays within three Monte
Carlo standard errors, the same seed reproduces byte-equivalent summaries across one and two
workers, and the maximum runtime is 28.4 seconds versus the 10,800-second gate. The remaining
348 cells were therefore submitted as independent jobs with an append-only receipt. All 360
cells subsequently passed strict aggregation; the complete analysis contains no missing or
duplicate cell and is rendered from a hash-bound JSON artifact.

### Gate G — Additional models or baselines

Add a third architecture or extra-information confidence method only if P1/P2 are complete and
the comparison answers a distinct question. Place TTA, MC dropout, ensemble, or learned-failure
scores in a separate information-budget table with inference/training cost. They must not be
mixed into the single-map primary table as if the budgets matched.

---

## 11. Submission-ready checklist

### Scientific scope

- [x] The abstract/introduction claim geometry-dependent rankings, not universal matched-score
      dominance.
- [x] Dice and nHD are described as Lipschitz--Wasserstein specializations under different
      geometries; nHD95 is a bounded robust extension, not a corollary of the nHD theorem.
- [x] The ten target-adapted conditions are the scientific target; six general/external rows are
      clearly identified as controls.
- [x] Single-label diagnostics are not presented as validation of the joint posterior.

### Primary experiment integrity

- [x] All five cohorts, sixteen conditions, sample counts, prompts, checkpoints, and split
      digests match the locked protocol.
- [x] Every score uses the same frozen probability map and hard action within a condition.
- [x] Surface extraction, pooled quantile, normalization, resizing, and empty-mask conventions
      match code, prose, and tests exactly.
- [x] All nine indexed score/risk cells, seventeen score rows, and all RC curves are present.
- [x] All 64 fixed contrast rows and 10,000 paired resamples reproduce from analysis JSON;
      legacy compatibility tail-area fields are not interpreted in the manuscript.
- [x] No win count is analyzed as if conditions were independent.

### Robustness and numerical evidence

- [x] Dice-Exact and the M2/M8/M32 ladder are complete on all sixteen conditions.
- [x] Five-dataset threshold sensitivity is complete for all three losses and Gate B is reflected
      in the main text.
- [x] Seeds 1--2 complete the full target-only pipeline; seed 0 remains the primary analysis,
      the extension stays checkpoint-descriptive, and its three values are not presented as a
      full estimate of training randomness.
- [x] Matched-risk reliability, cardinality/PIT, empty-action, and locked v1 runtime
      diagnostics are generated from locked artifacts.
- [x] The 16-condition M2/M8/M32/Exact runtime ladder v2 is complete, strictly
      analyzed, and replaces the narrower v1 runtime table.
- [x] Any threshold/seed decision gate that fired is reflected in the main text, not buried.

### Reproducibility

- [x] One experiment equals one Slurm job; all 162 P2 jobs appear in append-only phase receipts.
- [x] Independently record and full-byte verify the schema-v2 scientific-input root seal.
- [x] Complete and reconcile the declared schema-v2 smoke/full replay receipts. The smoke
      campaign completed 7/7 jobs and the full campaign completed 112/112 jobs with terminal
      receipts, strict output validation, all four CPU partitions used, and no failed or unknown
      state. The top-level diagnostic aggregator accepts the explicitly sealed v1/v2 campaign
      identities and rejects unsealed IDs; all 16 v2 summaries pass that canonical path. All 816
      replay-versus-v1 method-by-risk AURCs are exactly equal.
- [x] Main, threshold, seed, diagnostic, runtime, and synthetic output roots are disjoint.
- [x] Campaign locks, estimator specs, code/checkpoint/data digests, and ordered sample hashes are
      exported in portable provenance.
- [x] Tests cover identity collisions, empty masks, exact Dice, nHD/nHD95 conventions, tie-aware
      AURC, bootstrap pairing, and table regeneration.
- [x] A clean checkout can run tests, rebuild analysis from the public assembled records, and
      compile the paper using documented commands.
- [x] No access token, private path, private checkpoint, or licensed raw data is committed.

### Presentation

- [x] Main result tables use methods as rows, datasets as columns, and dark-blue exact top-1
      highlighting based on unrounded values.
- [x] Every analysis-derived table/figure set records a verifiable analysis or source-artifact
      digest and is referenced in the text; non-analytic schematics are explicitly exempt.
- [x] Main text stays compact; complete panels, diagnostics, sensitivity analyses, and curves are
      in the appendix.
- [x] Equations without cross-references are unnumbered.
- [x] Limitations and conclusion are merged and explicitly cover seed-0-conditioned primary
      inference, the target-only descriptive seed extension, the limits of three-seed evidence,
      single annotations, posterior non-identifiability, patient clustering, site shift, and
      the single-map information budget.
- [x] The final PDF is visually inspected for clipped tables, unreadable colors, missing legends,
      unresolved references, and page-limit compliance.

The paper is submission-ready when every applicable required checkbox is green, every omitted
P2/P3 item is described honestly as a limitation rather than implied as completed, and all
headline statements can be traced to a locked estimand and a reproducible artifact.
