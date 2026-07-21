# Pixel-Level Selective Segmentation — Survey and Roadmap

Date: 2026-07-06 | Multi-source, adversarially verified literature research
All recommendations build on `predict_probs()` — per-pixel, per-class
probability maps already exposed by every model wrapper; tier-1 methods need
no model changes.

## 0. Summary

Build the pixel-level risk–coverage framework first, with single-pass
confidence scores (MSP / margin / entropy) as the mandatory baseline; then
add TTA and Conformal Risk Control (~a day each). Ensembles are unusually
cheap for us (5–40 min per fine-tune) and come next. Learned failure
predictors and training-time methods come last. **Report calibration (ECE)
separately — never as a proxy for abstention quality.**

## 1. Tier 1 — directly on `predict_probs()` (near-zero cost)

### 1.1 Single-pass confidence scores + risk–coverage ⭐ first

Score each pixel, abstain in score order, plot risk–coverage:

| Score | Applies to | Formula (probability map p) |
|---|---|---|
| MSP (max softmax prob) | DeepLabV3 | `p.max(over classes)` |
| margin | DeepLabV3 | top1 − top2 |
| entropy | both | `−Σ p log p` (binary for CLIPSeg) |
| sigmoid confidence | CLIPSeg | per-prompt `\|p − 0.5\|`, or max merged class prob |

Thresholding a pretrained network's confidence is the dominant baseline
(SelectiveNet, ICML 2019); well-tuned thresholds often match training-time
methods (Feng et al., TMLR 2024); MSP is strong (Jaeger et al. 2023).

**Our key question:** overlay the four conditions' risk–coverage curves —
can zero-shot match fine-tuned once coverage tightens? Fine-tuning is
already non-monotone per class (RESULTS.md), which is the
selective-segmentation argument.

### 1.2 Test-time augmentation

Average hflip + multi-scale forwards; the variance is an uncertainty
source. "A light-weight alternative" to ensembles (ValUES, ICLR 2024 oral).
Eval-loop change only.

### 1.3 Conformal Risk Control

Calibrate one threshold λ on held-out data; the set
`C_λ(x) = {y : p_y ≥ 1−λ}` carries a distribution-free `E[loss] ≤ α`
guarantee for any bounded monotone loss (Angelopoulos et al., ICLR 2024).
Works for sigmoid and softmax maps, no retraining. The guarantee is a
marginal image-level expectation, not per-pixel.

### 1.4 Temperature scaling (control, not method)

Fit T on held-out NLL; report ECE separately. Temperature is monotone, so
it barely moves risk–coverage — and most calibration methods are useless or
harmful for failure prediction (Zhu et al., ECCV 2022 / TPAMI). Exception:
CLIPSeg's background is a fixed 0.5 threshold, so temperature *does* change
its predictions.

## 2. Tier 2 — moderate cost (1–2 weeks)

### 2.5 Deep ensembles ⭐ our special advantage

Most robust across five downstream tasks (ValUES). Usually expensive — but
our fine-tunes take 5–40 min, so a 5-seed ensemble is an overnight job.
Per-pixel variance / mutual information become scores. Zero-shot CLIPSeg
gets a free analogue: **prompt ensembles** ("cat" / "a photo of a cat" / …)
— rarely explored for selective segmentation, a potential contribution.

### 2.6 Pixel-score aggregation (a verified research gap)

"Aggregation of scores is a crucial but currently neglected component"
(ValUES); the best choice is regime-dependent. Comparing pixel / patch /
region / image-level abstention on our benchmark is a
literature-acknowledged gap.

### 2.7 MC-Dropout (DeepLabV3 only)

DeepLabHead's ASPP has Dropout(0.5) — keep active, sample N passes. CLIPSeg
has no natural dropout. Less robust than ensembles; include as comparison.

## 3. Tier 3 — internal features + extra training (later)

| Method | Source | Idea | Why later |
|---|---|---|---|
| ConfidNet / TCP | NeurIPS 2019 | auxiliary net regresses True Class Probability | feature access + multi-stage training; ~1–2 pt over MSP |
| ObsNet | ICCV 2021 | observer net over intermediate activations + adversarial training | skip connections essential; heavy |
| FSNet | RA-L 2022 | direct per-pixel error-map prediction | jointly retrains the segmenter; not post-hoc |
| SelectiveNet-style | ICML 2019 | coverage-constrained training | tuned thresholds often match it (TMLR 2024) |
| Soft Dice Confidence | arXiv:2402.10665 | Dice-optimal abstention score approximation | image-level only; superiority claim failed verification |

## 4. Evaluation protocol

- **Headline**: pixel-level risk–coverage + AURC / E-AURC + coverage at
  target risk (Zenk et al., Medical Image Analysis 2025, advocate exactly
  this).
- Per condition (4) × dataset (2) × score; stratify by class (chair/sofa vs
  easy).
- ECE / reliability diagrams in a separate section (§1.4).
- Stratify risk–coverage by distance-to-boundary (boundary uncertainty vs
  whole-object failure).
- Default risk: per-pixel 0/1 error; also report macro risk (background
  dominates otherwise).

## 5. Implementation order

| Step | Content | Effort | Extra GPU |
|---|---|---|---|
| 1 | `selective.py` scores + risk–coverage/AURC + 8 baseline curves | 1–2 days | ~15 min |
| 2 | TTA evaluation | 0.5 day | eval ×4–6 |
| 3 | CRC (FNR control, Pet first) | 1 day | negligible |
| 4 | Temperature-scaling control | 0.5 day | negligible |
| 5 | 5-seed ensembles + prompt ensemble | 1–2 days | ~8 GPU·h |
| 6 | Aggregation comparison (pixel/patch/region/image) | 2–3 days | reuses scores |

## 6. In-house proposal: Bi-Level Set Distance Field confidence

> ⚠️ **SUPERSEDED (2026-07-12, revised 2026-07-14) — THIS WHOLE SECTION.**
> **Read [`docs/FINDINGS.md`](docs/FINDINGS.md) first — it is the single source of
> truth**, and `docs/main.tex` is the spec. The method below is broadly what ships, but
> its *results*, its *α* analysis and several of its *theoretical claims* are invalid or
> retracted; the detailed banner is further down this section. Do not cite anything here
> without checking it against the paper.

*Recorded 2026-07-06; original idea, not yet validated.*

### Method

For a probability map $P$ and significance level $\alpha$ (e.g. 0.05):

1. **Bi-level sets**: conservative
   $Y_{\text{high}} = \{i \mid p_i \ge 1-\alpha\}$, aggressive
   $Y_{\text{low}} = \{i \mid p_i \ge \alpha\}$;
   $Y_{\text{high}} \subseteq \hat{Y} \subseteq Y_{\text{low}}$. The band
   between them is the "probability fog".
2. **Band width via distance transform** — a robust Hausdorff-style
   distance between the two level-set contours, $O(N)$, single pass. With
   $D_S$ the Euclidean distance transform of a surface $S$, three
   directional variants:

$$
\widehat{\mathrm{HD95}}^{\text{out}} = \mathrm{P}_{95}\big(\{ D_{\partial Y_{\text{high}}}(i) \mid i \in \partial Y_{\text{low}} \}\big),
\qquad
\widehat{\mathrm{HD95}}^{\text{in}} = \mathrm{P}_{95}\big(\{ D_{\partial Y_{\text{low}}}(i) \mid i \in \partial Y_{\text{high}} \}\big)
$$

   and the **symmetric (bidirectional Hausdorff) variant**, pooling both
   reading sets before the percentile — the same convention as the
   evaluation metric HD95:

$$
\widehat{\mathrm{HD95}}^{\text{sym}} = \mathrm{P}_{95}\big(
\{ D_{\partial Y_{\text{high}}}(i) \mid i \in \partial Y_{\text{low}} \}
\cup
\{ D_{\partial Y_{\text{low}}}(j) \mid j \in \partial Y_{\text{high}} \}
\big)
$$

   The sets are nested ($Y_{\text{high}} \subseteq Y_{\text{low}}$), so all
   three measure band thickness, but from different anchors: *outward*
   catches fog bulges and detached mid-confidence blobs, *inward* catches
   erosion/holes of the confident core (but is blind to detached blobs),
   *symmetric* combines both.
3. **Confidence**: $C = \exp(-\widehat{\mathrm{HD95}}_{\text{prob}} / \sigma)$,
   $\sigma$ = the application's tolerated deformation in pixels.

**A family of three discrepancy measures on one frame.** The same bi-level
sets admit two *region* measures of level-set agreement alongside the
*distance* one above. Because the sets are nested, the Dice/IoU between them
reduce to area ratios:

$$
\mathrm{IoU}(Y_{\text{high}}, Y_{\text{low}}) = \frac{|Y_{\text{high}}|}{|Y_{\text{low}}|},
\qquad
\mathrm{Dice}(Y_{\text{high}}, Y_{\text{low}}) = \frac{2\,|Y_{\text{high}}|}{|Y_{\text{high}}| + |Y_{\text{low}}|}
$$

Both lie in $[0,1]$ (higher = thinner band = more confident). The three
scores share the frame and the present-class aggregation and differ only in
*how* they read the band: **HD95 uses the distance transform (spatial —
where the fog is, how wide); Dice/IoU use only areas (how much fog,
discarding spatial arrangement).** This is a built-in ablation of whether
the spatial information helps.

### Assessment

**Strengths**: tier-1 cost (one forward + one EDT); works for sigmoid and
softmax; uncertainty in **geometric units (pixels)** — same scale as the
HD95 metric, unlike probability-unit scores; the band
$Y_{\text{low}} \setminus Y_{\text{high}}$ is exactly where pixel-level
abstention happens, and its width bounds the spatial cost of abstaining.

**Risks**:

- **Confidently-wrong blind spot** (main): it measures blur, not error — a
  sharp but misplaced prediction gets confidence ≈ 1. Fine-tuned models are
  much sharper than zero-shot ones; validate the correlation with true
  per-image HD95 before cross-condition use.
- $Y_{\text{high}} = \varnothing$ needs a one-sided convention; far-away
  low-confidence blobs widen the band (usually desirable, verify).
- Multi-class needs per-class computation + aggregation; the $\exp$ map is
  monotone — $\sigma$ only matters for absolute thresholds, not ranking.

### Related work

- Distance transform × probability map as an HD approximation exists as a
  *training loss* (Karimi & Salcudean, IEEE TMI 2019) — same machinery,
  new use here (inference-time confidence).
- Conformal Risk Control (§1.3) thresholds probability maps into exactly
  such nested level sets; conformally calibrating $\alpha$ would give the
  band a coverage guarantee — the most interesting extension.
- Sampling counterpart: contour variability across MC-dropout / ensemble
  samples; this is its closed-form single-pass analogue.
- Band-based evaluation precedent: Boundary IoU (Cheng et al., CVPR 2021).

### Validation experiments

1. Spearman correlation of $C$ with true per-image HD95 / image-level mIoU,
   all conditions × datasets;
2. risk–coverage as an image-level selective score vs tier-1 baselines
   (mean entropy, mean max-prob, band-area fraction), compared by AURC;
3. ablation: $\alpha \in \{0.01, 0.05, 0.1\}$, $\mathrm{P}_{95}$ vs $\max$.

### ⚠️ SUPERSEDED (2026-07-12) — read this before trusting §6 below

The results in this section are **invalid** and the paper (`docs/main.tex`) now
carries the corrected version. Three things were wrong:

1. **A convention bug in `selective.py` inverted the band's confidence on total
   detection failures.** When no foreground class is predicted, the
   present-class aggregation fell back to `0.0` — which, for a *negated*
   distance score, is the **maximum** confidence. Those images (mean
   `1 − mIoU` of 0.55–0.84, vs dataset means 0.04–0.40) were being ranked as
   the band's *best* predictions, while SDC's `0.0` fallback is its *minimum*.
   Same images, opposite ends of the ranking. **Fixed**: they now get `−diag`.
   Guarded by `test_detection_failure_is_least_confident_not_most`.

2. **The headline "risk-alignment split" was an artifact of (1).** With the
   convention corrected, band-beats-SDC under overlap risk goes from **2/8 to
   6/8** — the split disappears, and the corrected result is *stronger*: the
   band is better under **both** risks. The claim below that "on the six
   fine-tuned conditions the band width beats SDC on all three risks with
   bootstrap 1.00" was never true even before the fix (VOC overlap bootstrap
   was 0.00).

3. **Two spatially-aware baselines beat the band and were not reported.** BUC
   beats it on deeplabv3-target/Pet (2.409 vs 2.537) and boundary entropy on
   clipseg-general/Pet (8.540 vs 9.068). "SDC is the only real competitor" is
   false. (This *supports* the thesis — the competitive baselines are exactly
   the non-permutation-invariant ones.)

Also: **Band-IoU is the SAM stability score** (Kirillov et al., ICCV 2023) up to
reparameterization — SAM's default logit bracket maps to probability thresholds
0.7311/0.2689, summing to exactly 1. It is a published baseline, not our
ablation. And the theory below/in the first draft leaned on a false claim
("expected Dice is coupling-invariant"); the correct discriminating argument is
**permutation invariance** — see `docs/main.tex` Lemma 1.

The `α` sweep is degenerate at α=0.01 (**98.6–99.6%** of zero-shot CLIPSeg images
saturate, 92% tied) but only **0–17.7%** at the deployed α=0.1. ⚠️ An earlier version of
this banner said **27–67% of images at α=0.1** — that number is **RETRACTED**: it was
measured with a `width > 200px` proxy that over-counts genuinely wide bands, and it
over-stated the rate by ~3×. See `docs/main.tex` §"Thresholds" (`rem:retraction`), not
"§5" — the section numbering this banner cited has since drifted.

### Results (2026-07-10; SUPERSEDED — 3 level-set methods, 15 baselines, 3 risks, bootstrap)

All 12 condition × config combinations, 27 confidence scores × 3 α
(`selectseg.selective_eval` + `scripts/analyze_selective.py`; per-image
data in `outputs/selective*/`, summary in `outputs/selective/summary.json`,
curves in `figures/risk_coverage_{pet,voc}.png`). AURC lower is better.
Three risks: two **overlap** risks (1 − image mIoU, 1 − image mean Dice)
and the **boundary** risk image HD95 (on images where HD95 is defined).
Significance is a paired image-bootstrap (1000 resamples) of the default
band width vs. SDC. The confidence scores were independently code-audited;
core methods (band width, SDC, AURC) verified correct, and secondary
baselines corrected (MMMC sign, size-confounded total-uncertainty removed).

Baselines are the published single-pass, probability-map-only family: aMSP /
5th-pct MSP / worst-patch MSP, margin, Shannon entropy (global + foreground),
low-confidence fraction, tail-mean uncertainty, DOCTOR collision probability
(NeurIPS 2021), Generalized Entropy (CVPR 2023), boundary/interior entropy
and Boundary Uncertainty Concentration (IJCNN 2020 / MedIA 2024 / UNCV 2025),
and **Soft Dice Confidence** (arXiv:2402.10665). A **combined** score
(rank-average of band width + SDC) is evaluated as a two-signal candidate.

**Headline 1 — the three level-set measures specialize by risk type.**
Best-of-three among our own scores (HD95-distance / Dice-area / IoU-area),
wins per risk:

| Risk | HD95 (distance) | Dice (area) | IoU (area) |
|---|---|---|---|
| overlap (1 − mIoU) | 4/12 | **8/12** | 0/12 |
| overlap (1 − Dice) | 4/12 | **8/12** | 0/12 |
| boundary (HD95) | **12/12** | 0/12 | 0/12 |

The **distance-vs-area ablation is decisive**: on boundary risk the distance
transform beats both area measures **12/12**; on overlap risk the area
measure (Dice) beats the distance one 8/12 — spatial information helps
exactly when the risk is spatial. IoU-area is dominated by Dice-area (0/12).
Against all 15 baselines, HD95-distance beats on average 11.9/11.5/**14.5**
(iou/dice/hd95) — the strongest all-rounder, near-perfect on boundary risk;
the area measures beat ~9/15 on every risk.

**Headline 2 — vs. the strongest external baseline, SDC, and it is
significant.** Best-of-three (HD95 band width / SDC / band+SDC combined):

| Risk | band width | SDC | combined |
|---|---|---|---|
| overlap (1 − mIoU) | 4/12 | 1/12 | **7/12** |
| overlap (1 − Dice) | 4/12 | 1/12 | **7/12** |
| boundary (HD95) | **8/12** | 0/12 | 4/12 |

Band width vs. SDC head to head is **statistically decisive in both
directions**: on the six fine-tuned conditions the band width beats SDC on
all three risks with bootstrap 1.00; on zero-shot / VOC overlap risk SDC
wins with bootstrap 0.00; on boundary risk the band width wins with
bootstrap 0.71–1.00. (SDC ≈ Dice-area numerically — both region/Dice-based —
which is why the area level-set measure lands near SDC on overlap risk.)

**Takeaways:**

1. **Risk alignment is the story.** SDC estimates expected Dice → aligned
   with overlap risk; the band width is a boundary-geometry score → wins
   boundary risk (8/12 outright, and beats SDC 12/12 head-to-head there).
   The right estimator depends on whether the application penalizes region
   overlap or boundary error.
2. **The combined score is the best all-rounder**: 7/12 on both overlap
   risks (beating pure band *and* pure SDC by borrowing SDC's overlap
   alignment) and second on boundary — the two-signal fusion pays off.
3. **Fine-tuned + Pet: pure band width sweeps all three risks** (bootstrap
   1.00) — on the sharpest models on the binary task, boundary geometry is
   the dominant signal even for overlap risk.
4. **The distance-field geometry is the contribution**: the EDT-percentile
   width beats the crude area-per-boundary proxy 12/12 (often ~2×) and
   band-area fraction likewise.
5. **SDC is the only real competitor.** Against the whole baseline set the
   band width wins overlap in 8–11/12 and boundary in 10–12/12 — it loses
   overlap only to SDC. GEN (CVPR 2023 SOTA softmax OOD) is not competitive
   here (11/12 losses).
6. **Direction ablation: symmetric present-mean is the default** (wins 9/12
   over outward; inward alone never wins), mirroring the bidirectional HD95
   convention. **α is mild between 0.05–0.1**, catastrophic at 0.01 for
   zero-shot models.

**Next steps:** conformally calibrate $\alpha$ (§1.3); tune the combined
score's fusion weight; and extend from image-level to pixel/region
abstention using the band itself as the abstention region.

## 7. Key references

1. Kahl et al., *ValUES: A Framework for Systematic Validation of
   Uncertainty Estimation in Semantic Segmentation*, **ICLR 2024 (oral)**.
   arXiv:2401.08501
2. Angelopoulos et al., *Conformal Risk Control*, **ICLR 2024**.
   arXiv:2208.02814
3. Zenk et al., *Comparative Benchmarking of Failure Detection Methods in
   Medical Image Segmentation*, **Medical Image Analysis 2025**.
   arXiv:2406.03323
4. Zhu et al., *Rethinking Confidence Calibration for Failure Prediction*,
   **ECCV 2022** (TPAMI extension arXiv:2403.02886)
5. Geifman & El-Yaniv, *SelectiveNet*, **ICML 2019**. PMLR v97
6. Feng et al., *Towards Better Selective Classification*, **TMLR 2024**.
   arXiv:2304.03870
7. Corbière et al., *Addressing Failure Prediction by Learning Model
   Confidence (ConfidNet/TCP)*, **NeurIPS 2019**. arXiv:1910.04851
8. Besnier et al., *Triggering Failures: Out-Of-Distribution Detection by
   Learning from Local Adversarial Attacks in Semantic Segmentation
   (ObsNet)*, **ICCV 2021**. arXiv:2108.01634
9. Rahman et al., *FSNet: A Failure Detection Framework for Semantic
   Segmentation*, **IEEE RA-L 2022**. arXiv:2108.08748
10. Karimi & Salcudean, *Reducing the Hausdorff Distance in Medical Image
    Segmentation with Convolutional Neural Networks*, **IEEE TMI 2019**.
11. *Soft Dice Confidence: A Near-Optimal Confidence Estimator for
    Selective Prediction in Semantic Segmentation*, arXiv:2402.10665 — the
    strongest baseline (overlap risk); binary/medical, we extend it to
    multi-class via present-class aggregation.
12. Liu et al., *GEN: Pushing the Limits of Softmax-Based OOD Detection*,
    **CVPR 2023**.
13. Granese et al., *DOCTOR: A Simple Method for Detecting Misclassification
    Errors*, **NeurIPS 2021** (Gini/collision-probability score).
14. Rottmann et al., *Prediction Error Meta Classification … Aggregated
    Dispersion Measures of Softmax Probabilities (MetaSeg)*, **IJCNN 2020**
    (boundary/interior entropy and margin).
15. Zeevi et al., *Spatially-Aware Evaluation of Segmentation Uncertainty
    (Boundary Uncertainty Concentration)*, **UNCV @ CVPR 2025**.
