# Handoff — 2026-07-12 (archived historical snapshot)

> **ARCHIVED / NON-AUTHORITATIVE.** This handoff records an earlier development
> state. Do not use it for current paper claims, completion status, scheduler
> policy, or release numbers. Consult the
> [completion audit](output/COMPLETION_AUDIT.md), [active plan](docs/PLAN.md),
> and [repository README](README.md). The historical material below is retained
> for provenance.

> ⚠️ **PARTIALLY SUPERSEDED (revised 2026-07-14).** Read
> [`docs/FINDINGS.md`](docs/FINDINGS.md) only as another archived snapshot.
> Specifically: §2 and §3c predate the eight refutations of 2026-07-14 (§3c's
> order-statistic bullet is corrected in place below, but read the paper's eq:orderstat,
> prop:floor, obs:vshape and rem:vshapefails for the current theory), and **every M-ladder
> number anywhere in this file is stale — read them from `python scripts/show_results.py`,
> not here**. The contamination scare those numbers were once under is RESOLVED: the
> defect never fired at the deployed α, the M-ladder reproduced to the digit, and the
> embargo is lifted (`rem:mladdercontam`). §0a (the nesting-leak precondition and the
> `mband` repair) still stands.

## 0a. THE BEST FINDING OF THE NIGHT — a precondition nobody had stated

**The bi-level band requires a NORMALIZED (softmax) or binary probability map. On
per-prompt sigmoid maps it silently breaks — and it fails by producing FALSE
CONFIDENCE.**

I hypothesised this from the CLIPSeg failure pattern, built a label-free diagnostic
(`diag_nesting_leak@α`), and ran it. It's confirmed 8/8:

| condition | map | leak@0.05 | leak@0.1 | leak@0.2 | band vs SDC |
|---|---|---|---|---|---|
| clipseg-general, VOC | **sigmoid** | 2.2% | 6.8% | 13.2% | **LOSE** |
| clipseg-target, VOC | **sigmoid** | 2.7% | 4.2% | 6.1% | **LOSE** |
| deeplabv3-external, VOC | softmax | **0.0%** | **0.0%** | **0.0%** | win |
| deeplabv3-target, VOC | softmax | **0.0%** | **0.0%** | **0.0%** | win |

Nesting is violated on *exactly* the two conditions where the band loses, and is
*exactly zero* where it wins. **Mechanism:** CLIPSeg scores VOC with 20 independent
per-prompt sigmoids, background = 1 − max(fg). Semantically overlapping prompts
(cat/dog, cow/horse/sheep — the classic VOC confusions) co-fire: two clear 1−α while
only one wins the argmax, and the loser's conservative set escapes its own
prediction. A normalized softmax **cannot** do this — two classes can't both exceed
0.8 on the simplex. That's why the softmax leak is identically zero, and why the
leak grows as 1−α falls toward ½.

**The failure is worse than a broken invariant.** When leaked pixels lie in *both*
level sets, Y_lo and Y_hi coincide there and **the band vanishes — reporting maximal
confidence on exactly the images where two classes are fighting.** On a synthetic
co-firing map: plain band = 0.00 (perfectly confident), masked band = 18.38 px.

**The fix follows from the diagnosis, and it's free.** Intersect Y_hi with the
prediction (`neg_mband_*`). Provable no-op where the leak is zero — confirmed to
machine precision (47.876 → 47.876 on DeepLabV3) — and on CLIPSeg it closes
**42–77%** of the band's gap to SDC. It dominates, so **adopt it as the default**.

**But it flips no cell.** The band still loses 2/8. The residual gap is the *other*
mechanism — detection-failure blindness — which is **not repairable**: a
contour-displacement score cannot be made to see whether a class exists. Two
mechanisms, separable; only one is ours to fix. Don't let anyone paper over the
second.

This generalizes past this paper: **any open-vocabulary / prompt-based segmenter
(CLIPSeg, SAM-style) inherits the precondition.**

---

## 0. THE RESULT (the re-run landed)

**The 8/8 boundary win over SDC did NOT survive the honest convention.** Under
`hd95_penalized` (detection failures charged the image diagonal instead of being
deleted from the curve) it is **6/8**. I flagged this as the number that could
still hurt, and it did. It's reported as degraded, in the abstract.

**But the pattern is far better than the old story.** The band loses **exactly the
same two cells** under *both* risks — `clipseg-general_voc` and `clipseg-target_voc`.
A metric-alignment story cannot explain a score failing on the same conditions under
both an overlap *and* a boundary risk. A condition-level story can:

| condition | images with a one-sided class | pen/drop risk | band vs SDC |
|---|---|---|---|
| Pet (all 4) | **0.2–0.3%** | 1.1–1.3× | win |
| deeplabv3, VOC | 31% | 4.0× | win |
| **clipseg, VOC** | **53–67%** | **6.0×** | **LOSE** |

On CLIPSeg-VOC a **majority of images contain a hallucinated or missed class**, so
the penalized boundary risk there is dominated by *detection* error, not boundary
geometry. The band is a contour-displacement statistic — it measures how far the
boundary might move and **cannot see whether a class is there at all**. That's the
"measures blur, not error" limitation, made quantitative, and it locates the defect
in the *probability map's class structure*, not in the choice of metric.

Hypothesis for the mechanism (flagged in the paper as unmeasured): CLIPSeg scores
VOC with 20 **independent per-prompt sigmoids**, background = 1 − max(fg), so
semantically overlapping prompts (cat/dog, cow/horse/sheep — the classic VOC
confusions) co-fire. That both manufactures hallucinated classes *and* breaks the
nesting `Y_hi ⊆ Ŷ ⊆ Y_lo` the whole bi-level construction assumes. DeepLabV3's
normalized softmax can't do this (two classes can't both exceed 0.9 on the simplex);
binary Pet can't either. **A nesting-violation diagnostic would settle it — that's
the single highest-value next experiment.**

**AND THE CORE THESIS SURVIVES EVERYTHING.** The distance readout beats the band's
own area readouts under boundary risk **8/8 under BOTH conventions**. It is the one
result invariant to every convention we vary — including the two that flip the SDC
comparison. **Lead the paper with it.**

---


You asked me to develop the method in `docs/main.tex`, then to bring in more
agents, then to think about thresholds. All three are done. **One of them turned
up a real bug that was deciding the paper's headline result.** Read §1 first.

---

## 1. The thing you need to know: a convention bug was inverting the score

`selectseg/selective.py` aggregated the band width over *predicted-present*
classes. When an image has **no predicted foreground class at all** — a total
detection failure — the aggregation fell back to `0.0`:

```python
scores[f"neg_band_width_pmean{suffix}@{key}"] = -(
    sum(values) / len(values) if values else 0.0   # <-- BUG
)
```

Band widths are non-negative and the score is *negated*, so every real score is
≤ 0 and **`0.0` is the maximum attainable confidence.** The band was ranking its
total detection failures as its *most confident* predictions. SDC's `0.0`
fallback, by contrast, is SDC's *minimum*. The same images sat at opposite ends
of the two rankings.

Those images are catastrophic: mean `1 − mIoU` of **0.55–0.84**, against dataset
means of 0.04–0.40.

**Consequence.** I recomputed all overlap AURCs from the released JSONL with the
degenerate images ranked last (independent bootstrap, B=2000, seed 0):

| condition | band (orig) | band (fixed) | SDC | boot(fixed > SDC) |
|---|---|---|---|---|
| clipseg-general_pet | 0.0289 | **0.0176** | 0.0204 | 0.99 ✅ |
| clipseg-general_voc | 0.2986 | 0.2920 | **0.2375** | 0.00 ❌ |
| clipseg-target_pet | 0.0047 | **0.0047** | 0.0126 | 1.00 ✅ |
| clipseg-target_voc | 0.1843 | 0.1777 | **0.1627** | 0.00 ❌ |
| deeplabv3-external_pet | 0.0208 | **0.0118** | 0.0176 | 1.00 ✅ |
| deeplabv3-external_voc | 0.1288 | **0.1018** | 0.1072 | 0.99 ✅ |
| deeplabv3-target_pet | 0.0060 | **0.0060** | 0.0140 | 1.00 ✅ |
| deeplabv3-target_voc | 0.1126 | **0.0912** | 0.0950 | 0.96 ✅ |

**Band beats SDC on overlap risk 6/8 after the fix, versus 2/8 before.**

So the "risk-alignment split" — the story that SDC wins overlap and we win
boundary, which the whole first draft presented as the falsifiable confirmation
of the theory — **was substantially an artifact of this bug.** The corrected
result is *better*: the band wins under **both** risks. But the narrative had to
be rewritten, and I did.

Fixed in `selective.py` (detection failures → `−diag`, matching SDC's direction),
and guarded by a new regression test,
`tests/test_selective.py::test_detection_failure_is_least_confident_not_most`.

Note *where* it still loses: both CLIPSeg-on-VOC cells — which are exactly the
cells where 38–67% of images have a saturated band (§3). The bug and the
threshold problem point at the same place.

---

## 2. The theory had a false step; the fix is cleaner than the original

The first draft's "conceptual core" was: *the coupling is what separates us from
SDC — overlap metrics can't see the coupling, boundary metrics are made of it.*

**That does not follow, and it is false.** Prop. 3 proves only that the
*ratio of expectations* is coupling-invariant — trivially, since that object
never references the joint. But the framework's actual estimand
`E_Q[Dice(Y,Ŷ)]` is an expectation of a **ratio**, and it *is* coupling-dependent.
Exhaustively, for N=2, p=(½,½), Ŷ={1}: `E_indep[Dice]=5/12`, `E_Qp[Dice]=1/3`,
`SDC=1/2` — three different values.

Worse, the draft's escape hatch (SDC's <1% bound "once foreground mass > 17") is
**backwards**: among all couplings with marginals p, the comonotone `Q_p`
*maximizes* Var(Σᵢ Yᵢ), so it is the coupling under which the Dice denominator
concentrates *least*. And SDC's bound is proved under *conditional independence*
— the coupling we elsewhere reject.

That escape hatch fails for a **derived** reason, not a measured one, and the two
legs have different standing:

- **Under independence the gap is <0.4%** — a *consequence of concentration*, not a
  measurement: `Σᵢ Yᵢ` concentrates, so `E[ratio] → ratio-of-expectations` at
  `O(1/N)`. That is why SDC is nearly exact under its own assumption.
- **Against `Q_p` it does not shrink with mass** — this is the **theorem in the
  paragraph above**, not a synthetic sweep. `Q_p` is comonotone, so `Var(Σᵢ Yᵢ)`
  sits at the Fréchet–Hoeffding maximum and the delta-method step *never engages*,
  at any mass (`main.tex`, `rem:notcoupling`).

⚠️ **RETRACTED: the size of the `Q_p` gap.** This paragraph used to read "Measured:
independence gap <0.4%, `Q_p` gap **4.7% mean / 26% max**". The `Q_p` figure is
**cut** — no code anywhere computed it, and it is **unidentifiable**, not merely
undocumented: across defensible synthetic families the mean spans 0.21–11.37% and
the max 0.38–97.45%. The number *is* the generator. Do not try to recover it. This
line also dropped `main.tex`'s "measured on synthetic maps" qualifier, so it read as
if it came from the real evaluation; nothing on Pet/VOC ever produced it. The
argument is unaffected — it needs only that the bound *does not transport*, which
the comonotone step proves outright. See `docs/FINDINGS.md` §5b.

**The replacement is a one-line lemma that is true and predicts the same facts:**

> **Lemma (spatial blindness).** SDC, `E_Qp[Dice]`, and both area readouts of the
> band are invariant under any joint permutation of the pixel indices. The
> distance readout is not. A permutation-invariant functional of (p, ŷ) cannot
> resolve boundary geometry.

I verified both claims myself rather than taking the agents' word:

- **N=2 counterexample**, exhaustively enumerated: `E_indep[Dice] = 0.416667`
  (5/12), `E_Qp[Dice] = 0.333333` (1/3), `SDC = 0.5` — three distinct values. So
  expected Dice is *not* coupling-free.
- **Permutation invariance**, on a soft-boundary blob, one joint permutation σ of
  the pixel indices of (p, ŷ):

  | score | original | permuted | invariant? |
  |---|---|---|---|
  | SDC | 0.890158 | 0.890158 | yes |
  | Band-Dice | 0.713183 | 0.713183 | yes |
  | SAM stability | 0.554223 | 0.554223 | yes |
  | **Band-HD95** | **7.000** | **1.414** | **no** |
  | **r₂** | **3.803** | **1.000** | **no** |

  The three spatially blind scores are unchanged bit-for-bit; the two distance
  scores collapse.

This is strictly better: it's provable in one line, it's true, and it predicts
every experimental finding the coupling story was meant to explain. It also
correctly predicts something the coupling story got wrong — it does **not** say
SDC should *beat* us on overlap, only that it shouldn't lose *for lack of
geometry*. Which is exactly what the corrected data show.

---

## 3. Thresholds (your question) — I proposed a fix, tested it, and it FAILED

**Read this one carefully — my first answer to you was wrong on two counts, and I
retracted both.**

### 3a. My saturation numbers were wrong (over-counted ~3×)

I originally told you 27–67% of images stay degenerate even at α=0.1. **That was
measured with a `width > 200px` proxy, which over-counts genuinely wide bands.**
Recomputed against each image's *true* native diagonal:

| condition | α=0.01 | α=0.05 | α=0.1 |
|---|---|---|---|
| clipseg-general, Pet | **99.6%** | 37.9% | 14.7% |
| clipseg-general, VOC | **98.6%** | 46.3% | 17.7% |
| clipseg-target, Pet | 0.3% | 0.1% | 0.0% |
| clipseg-target, VOC | 17.3% | 5.1% | 1.4% |
| deeplabv3-external, VOC | 18.4% | 8.4% | 5.0% |
| deeplabv3-target, VOC | 12.0% | 5.4% | 2.7% |

The **α=0.01 collapse is real** (98.6–99.6% dead, 92% tied). But at the deployed
α=0.1 it's **0–17.7%, not 27–67%**.

**This kills my prediction to you.** I said fixing thresholds would close the
CLIPSeg-VOC overlap gap to SDC. But `clipseg-target_voc` — one of the two cells
where the band loses — is only **1.4% saturated.** Thresholds cannot explain that
loss. I've retracted the claim in the paper (Remark 6) rather than quietly
restating it. **We currently have no explanation for the two CLIPSeg-VOC losses**
and I declined to invent one.

### 3b. The fix I proposed collapses into the exact trap I warned you about

I implemented the "rank-anchored" threshold
`t_hi = min(1−α, P₉₅(p inside Ŷ_c))` and had four agents attack it. It **removes
the saturation** (tie fraction 0.90 → 0.04) and **makes the ranking worse in every
regime** — Kendall τ vs ground-truth risk falls 0.10–0.15, AURC rises, under both
the HD95 estimand *and* a convention-free `∫(1−Dice)dt` (so it isn't an artifact of
the diagonal convention).

**Why**, and this is the part worth remembering: *a percentile of the in-mask
probability values IS an area quantile of the mask.* Measured `|Y_hi|/|Ŷ|`: median
0.055, min 0.050 — i.e. `{p ≥ P₉₅(in-mask)} ∩ Ŷ` **is** the top-5%-by-area core.
So the "order-statistic anchor in probability" is literally the fixed-**area** rule
I told you (correctly) would destroy the signal. I set the trap and walked into it.
It also breaks nesting on 21-way softmax (an argmax winner need only clear 1/C, so
`t_hi` can drop below ½) and can *invert* the score.

**Verdict: shipped as an ablation, NOT the default.** `DEFAULT_BAND` untouched.

### 3c. What actually survives

- **The trimmed estimand** `r_α(x) = E[L(Y_T,Ŷ) | T ∈ (α,1−α)]` — makes α
  definitional, not a nuisance knob. Still right.
- **The order-statistic identity**, *with the CORRECTED indexing*:
  `∫₀¹H_x(t)dt = Σ_{k=1..m} (v_k − v_{k−1})·H_x(Y_{v_k}) + (1 − v_m)·diag`, with
  `v_1 < … < v_m` the distinct probability values and `v_0 := 0`. Each level carries the
  gap **BELOW** it (`Y_t = {p ≥ t}` is non-strict, so `Y_t` is constant on
  `(v_{k−1}, v_k]`), plus the atom `Q_p` puts on the **empty mask**.
  ⚠️ This bullet used to print the identity as `Σₖ(p₍ₖ₊₁₎−p₍ₖ₎)·H_x(p₍ₖ₎)` — each level
  paired with the gap **ABOVE** it, and no atom term — and certified it "still exactly
  true". **That form is WRONG** and it drops the empty-mask atom entirely (0.566 against
  a true 1.697 on the paper's worked example). See `docs/main.tex` eq:orderstat; pinned
  as false in
  `tests/test_theory.py::test_the_exact_identity_pairs_each_level_with_the_gap_BELOW_it`.
  What survives is the *lesson*: a true lemma doesn't license every estimator you can
  derive from it.
- **Label-free α selection against the saturation rate** — needs no labels, and
  the rate is now measured.
- **A 3-node rule** `(α, t_hi, 1−α)` that *samples* the saturating tail rather than
  eliding it, so a genuinely uncertain image stays at the bottom of the ranking
  instead of being replaced by a mask-shaped surrogate. **This is the one I'd try
  next.**
- **Conformal calibration of α** (CRC) — unchanged, still the most principled.

---

## 3-OLD. (superseded — kept for the record)

The band needs `Y_hi = {p ≥ 1−α} ≠ ∅`, i.e. `1−α ≤ maxᵢ pᵢ`. That is a condition
on **each image's own upper tail**; a global constant α has no reason to satisfy
it. Diffuse (zero-shot, sigmoid) maps rarely reach 0.9.

Fraction of images whose band **saturates to the image diagonal**:

| condition | α=0.01 | α=0.05 | α=0.1 |
|---|---|---|---|
| clipseg-general, Pet | **99.7%** | 57.4% | 27.5% |
| clipseg-general, VOC | **99.6%** | 82.7% | **67.4%** |
| clipseg-target, VOC | 67.4% | 49.6% | **38.4%** |
| deeplabv3-target, VOC | 50.0% | 34.6% | **29.1%** |
| deeplabv3-target, Pet | 1.6% | 0.3% | 0.2% |

At α=0.01 the zero-shot CLIPSeg score is **92% tied** (3669 images → 511 distinct
values). It isn't "insensitive", it's **dead**. And **even at the best α=0.1,
27–67% of images stay saturated on four of eight conditions** — so the VOC
numbers were computed on a score that is broken for a third to two-thirds of the
split. `METHODS.md`'s "α is mild between 0.05–0.1" hid this.

**My read: our own results understate the method, and VOC is where the headroom
is.** That's a falsifiable prediction and the re-run tests it.

What I put in the paper (§5 of `main.tex`):

1. **The estimand should be trimmed.** `H_x(t)` blows up as t→0 (`Y_t` → whole
   image) and saturates as t→1 (`Y_t` → ∅), so `∫₀¹ HD95(Y_t,Ŷ)dt` is dominated by
   degenerate level sets. The honest estimand is
   `r_α(x) = E[L(Y_T,Ŷ) | T ∈ (α, 1−α)]`, which makes **α definitional**, not a
   nuisance knob. Cost, stated honestly: the trimmed marginals are the winsorized
   `clip((pᵢ−α)/(1−2α), 0, 1)`.

2. **Thresholds belong in the image's own order statistics.** `H_x(t)` is
   *piecewise constant*, jumping only where t crosses one of the image's own
   probability values — so exactly
   `∫₀¹ H_x(t)dt = Σₖ (p₍ₖ₊₁₎ − p₍ₖ₎) · H_x(p₍ₖ₎)`.
   The natural nodes **are** the order statistics. Setting `t_hi` = the k-th
   largest probability makes `Y_hi ≠ ∅` *by construction* and kills the saturation.
   ⚠️ The subtlety that makes this safe: adapting the **nodes** per image is fine
   (we still estimate the *same* functional, so cross-image ranking survives);
   normalizing the **score** per image would destroy the signal. And the
   **weights must stay the gaps** — equal weights on quantile nodes silently
   swaps the posterior for one whose marginals are the image's own empirical CDF.
   Correct recipe: *adaptive nodes, gap weights, trimming.*

3. **Three label-free α rules**: degeneracy budget (`P̂(Y_hi=∅) ≤ δ`); quantile
   rule (fix `1−α` at a quantile of the pooled probability distribution, not a
   probability); **conformal calibration** (nested level sets are CRC's native
   object — this is the strongest option and it's already §1.3 of your roadmap).

Also worth knowing: **temperature scaling is a no-op for aMSP-style baselines but
NOT for us** — it moves the level sets. For this score, temperature and α are the
same knob.

---

## 4. What else the review found (73 agents, 46 findings survived verification)

Fixed in `main.tex` already:

- **Band-IoU *is* the SAM stability score** (Kirillov et al., ICCV 2023). SAM's
  default logit bracket maps through the sigmoid to probability thresholds
  0.7311 / 0.2689 — **summing to exactly 1**. Same nested construction, same
  symmetric bracket, same inference-time use. Re-labelled as a published
  baseline, not our ablation. (This *helps*: the area arm is now a deployed
  method, not a straw man.)
- **Props. 1–2 are textbook** — the coverage-function random set (Goodman,
  Molchanov) and the standard comonotone inverse-transform coupling. Demoted to a
  cited remark. The "independence destroys contour geometry" insight is 15 years
  old in the uncertain-isocontour literature (Pöthkow 2011; Bolin & Lindgren 2015).
- **Prop. 5 was proved for max-HD but everything we compute is HD95** — which is a
  *percentile*, **not a metric**, and does not obey the triangle inequality.
  Counterexample found: band width 13.41 px while `2·r₂ = 0`. The equality
  condition as stated was vacuous (0/398 on realistic triples), and the
  rank-equivalence conclusion is invalid. Restated honestly; the bridge is now an
  *empirical* question the re-run answers.
- **Every "/12" count** was over 8 config-A + 4 config-B cells, double-weighting
  the fine-tuned conditions. All counts restated out of **8**.
- **BUC and boundary entropy beat the band** in one cell each, and BUC beats SDC
  under boundary risk on **all four** Pet cells. "SDC is the only real competitor"
  was false by our own data. Both are now in Table 2. (Again: this *supports* the
  thesis — the competitive baselines are the non-permutation-invariant ones.)
- **α was selected on the test set** and the headline is not robust: at α=0.05 the
  boundary count drops 8/8 → 6/8. Disclosed; validation split is an open item.
- The "equal mIoU, 2.7 px HD95" example **did not exist** — it was a
  dataset-level config-A-vs-B comparison misremembered as a per-image anecdote.
  Replaced with real pairs from the JSONL (mIoU 0.333 → HD95 0.00 vs 26.03 px).
- Bibliography: MMMC → Holder & Shafique (not the SDC paper); `feng` pointed at
  ASPEST, not the paper cited; Karimi & Salcudean is TMI 2020; SDC had no authors.

Not yet fixed (needs the re-run): the **penalized boundary risk**. The current
HD95 risk *drops* images with no valid class — which are exactly the images the
bug mis-ranked. Under a diagonal penalty the 8/8 boundary win is **at risk**. Both
conventions are now implemented (`image_hd95` and `image_hd95_penalized`).

---

## 5. Code changes

| File | Change |
|---|---|
| `selectseg/selective.py` | **Bug fix**: detection failures → `−diag`, not `0.0`, on every negated score. New `diag_no_present_class` / `diag_saturated@α` label-free diagnostics. (`bilevel_r2`, `quadrature_risks`, M-ladder were added by the workflow.) |
| `selectseg/selective_eval.py` | New `image_hd95_penalized` (one-sided classes → image diagonal) so every risk is evaluated on the **same image set**. α grid widened to `0.01 0.05 0.1 0.15 0.2 0.3` — the old grid was unbracketed and still improving at the edge. |
| `scripts/analyze_selective.py` | New `hd95_penalized` risk; `diag_*` excluded from being ranked as scores and reported as diagnostics. |
| `tests/test_selective.py` | Regression test for the bug. |
| `docs/main.tex` | Rewritten (see §1–4). |
| `METHODS.md` | §6 marked **SUPERSEDED** with the reasons. |

## 6. Where things stand

- ✅ **Tests green: 83 passed** (job 13065312). Up from 68 — the threshold
  workflow added multi-class nesting coverage that did not exist before.
- 🚀 **Full re-run SUBMITTED**: `scripts/slurm/submit_quadrature.sh` →
  eval jobs `13065339–13065350`, dependent analyze `13065351`. Results land in
  `outputs/selective*/`, summary in `outputs/selective/summary.json`.

**The first number to look at when you're up:** `hd95_penalized` AURC, band vs
SDC. The current 8/8 boundary win is under the `drop` convention, which deletes
the images that detect nothing — exactly the ones the bug mis-ranked. Under the
diagonal penalty that win is genuinely at risk. If it degrades, report it
degraded; a paper that says "we win on the images where HD95 is defined, and here
is what happens when misses are penalized" is far more credible than one that
never asks.

Then: `spearman_band_vs_r2` (does the band actually stand in for the r₂ it claims
to instantiate? Prop. 5 does **not** establish this for HD95 — it's a percentile,
not a metric), `spearman_vs_dense` for the M-ladder, and the widened α grid
(0.01–0.3, previously unbracketed).

What the re-run settles, in priority order:

1. **The penalized boundary risk** — does the 8/8 boundary win survive counting
   detection failures as the worst boundary errors? *This is the one that can
   still hurt.*
2. **The surrogate ablation** (`spearman_band_vs_r2`) — is the band width actually
   rank-equivalent to the `r₂` it claims to instantiate? Prop. 5 does **not**
   establish this for HD95. Synthetic Kendall τ was 0.81, so it may well come out
   in our favour — but it must be *measured*.
3. **The M-ladder vs dense M=32** — expect the dense reference to disagree with
   M=2 on the zero-shot conditions, because of the max-softmax atom
   (`r_ls = ∫₀^{max p} H + (1−max p)·diag`). That's a **real property of the
   posterior, not a bug** — say so before a reviewer frames it otherwise.
4. **The widened α grid** — my prediction: the CLIPSeg-VOC overlap losses close.

`docs/main.tex` has no `pdflatex` on this box — compile it on Overleaf.
