# FINDINGS — archived historical snapshot

> **ARCHIVED / NON-AUTHORITATIVE.** This file preserves an earlier, superseded
> experimental narrative and must not be used for current paper claims,
> completion status, scheduler policy, or release numbers. The current
> authorities are the [completion audit](../output/COMPLETION_AUDIT.md),
> [active plan](PLAN.md), and [repository README](../README.md). The material
> below remains unchanged as historical context unless an explicit archival
> correction is required.

**To see the numbers:** `python scripts/show_results.py` (recomputes everything from the
per-image JSONL; there is no stale summary to trust).
**To check the theory:** `sbatch scripts/slurm/tests.sbatch` — 140 tests, of which
`tests/test_theory.py` pins every claim in `docs/main.tex`.

**Data state:** `outputs/selective/*.jsonl` are CLEAN and COMPLETE — 8 conditions, 20 472
images (3669/pet, 1449/voc), regenerated 2026-07-14 18:06–18:50 after the
`_present_readouts` fix. Everything on this page is recomputed from them.

---

## 1. What the paper actually shows

### The spine (holds under every convention we varied)

**The distance readout beats the area readout under boundary risk, 8/8 — under both
boundary conventions.** A *controlled* comparison: same posterior, same thresholds, same
aggregation, differing only in whether the band is read spatially or by area. Untouched by
everything else on this page.

| | overlap | boundary (drop) | boundary (penalized) |
|---|---|---|---|
| distance beats area | 7/8 | **8/8** | **8/8** |

### The estimator the theory derives beats the one we deployed — 24/24

**This is the cleanest result we have, and it answers P2.** At *matched* nodes
`(α, 1−α) = (0.1, 0.9)`, `r_2` (`neg_r2_pmean_sym@0.1`) beats the deployed band width
(`neg_band_width_pmean_sym@0.1`) on **every condition under every risk**:

| | overlap | boundary (drop) | boundary (penalized) |
|---|---|---|---|
| **r_2 beats the band** | **8/8** | **8/8** | **8/8** |

Same α, same two thresholds, same pooled-bidirectional HD95 convention, same present-class
mean, same `−diagonal` floors. **The only difference is the functional**: `HD95(Y_lo, Y_hi)`
vs `½[HD95(Y_lo,Ŷ) + HD95(Y_hi,Ŷ)]`. `selective.py:499-501` states the point exactly — the
ranking difference is "attributable to the estimator alone."

Free: no extra nodes, no extra cost, no new tuned parameter. It also **closes the paper's
one theory–practice gap**: Prop. 5's band↔`r_2` bridge is broken (HD95 is a percentile, not
a metric), but if we *deploy* the estimator the theory derives, there is nothing to bridge.

⚠️ **Honest bound on it.** The effect is significant (paired bootstrap, 10k reps, CI
excluding 0) on the **4 CLIPSeg conditions** (+7.07%, +9.23%, +3.23%, +2.32% of band AURC)
and **positive but null on the 4 DeepLabV3 conditions** (+0.27%, +1.09%, +1.93%, +2.23%).
So: 8/8 in sign, 4/8 significant. The 3 risks share the same images and are not 3
independent tests.

### Versus SDC (the closest prior work)

| | overlap | boundary (drop) | boundary (penalized) |
|---|---|---|---|
| band (deployed) beats SDC | 6/8 | 8/8 | 6/8 |
| r_2 @0.1 beats SDC | 6/8 | 8/8 | 6/8 |
| **dense M=32 beats SDC** | **7/8** | **8/8** | **7/8** |

The 8/8 holds only under the medical-imaging `drop` convention, which deletes images whose
classes appear on only one side — exactly the images the score is worst at. Charging them
the image diagonal (`penalized`) reduces it to 6/8. Dense recovers one cell
(`clipseg-target`/VOC). **This is now confirmed on clean data and no longer withheld.**

The two cells the band loses are **the same two under both risks** — `clipseg-*_voc`. A
metric-alignment story cannot explain that; a condition-level one can (§2).

---

## 2. Why the band fails where it fails — two separable mechanisms

### (a) A precondition nobody had stated: the map must be *normalized*

The bi-level construction assumes `Y_hi ⊆ Ŷ_c`. A softmax **cannot** break this. CLIPSeg's
**independent per-prompt sigmoids** can: two co-firing prompts (cow/horse/sheep — the
classic VOC confusions) both clear `1−α` while only one wins the argmax.

| condition | map | nesting leak @0.1 | band vs SDC |
|---|---|---|---|
| clipseg-general, VOC | **sigmoid** | **7.0%** | loses |
| clipseg-target, VOC | **sigmoid** | **4.3%** | loses |
| deeplabv3-*, VOC | softmax | **0.0%** | wins |

Leak is non-zero on *exactly* the two conditions that lose, and *exactly zero* on the
other six.

**The leak is the CLIPSeg × VOC *interaction*, and needs BOTH factors** — independent
per-prompt sigmoids **and** `C ≥ 3`. It is **not** explained by the argmax floor `φ`
(§3c): both CLIPSeg *Pet* conditions have the same `φ = ½` and leak **0.0%**, because
`amax` collapses cat/dog into one foreground channel, forcing `bg = 1 − fg₁` — a true
simplex, which cannot leak. So "VOC vs Pet" is a *necessary* factor here, not a
mis-scoping.

**The failure mode is false confidence.** Leaked pixels lie in *both* level sets, so the
contours coincide and **the band vanishes — reporting maximal confidence on images where
two classes are fighting.**

**The fix is free.** Intersect `Y_hi` with the prediction (`mband`). A *provable* no-op
where the leak is zero (verified to machine precision), closing **42–77%** of the gap to
SDC where it is not.

⚠️ **It is not the default, and that is an open author decision.**
`scripts/analyze_selective.py::DEFAULT_BAND` is still the *unrepaired*
`neg_band_width_pmean_sym@0.1`, and every table in `docs/main.tex` is reported against it —
deliberately, so all numbers refer to one fixed score. Flipping it invalidates every table
until they are regenerated, so it is a decision, not a cleanup. It flips no cell either way.

### (b) An intrinsic limit: it cannot see detection failures

`mband` closes most of the gap but **flips no cell**. The residual is structural: on
CLIPSeg-VOC **54–67% of images contain a hallucinated or missed class**, so the penalized
boundary risk there is ~6× the drop risk and is dominated by *detection* error. A
contour-displacement statistic measures how far a boundary might move; it **cannot see
whether a class is there at all.** Not repairable by a better threshold.

---

## 3. The theory — what survives

An independent adversarial review refuted most of what earlier drafts asserted. **Every
failure had the same shape: a clean derivation, asserted before it was tested.**

### Survives

- **The integral identity.** `E_{Q_p}[L] = ∫₀¹ L(Y_t, Ŷ) dt`, exactly — what makes the
  framework metric-agnostic.
- **The exact sum**, *with the corrected indexing*: each level carries the gap **below** it
  (`Y_t` is constant on `(v_{k−1}, v_k]`). Plus the empty-mask atom `(1 − max p)`.
- **Koksma's `O(1/M)` rate** (the *rate*; not the node prescription drawn from it).
- **Prop (sdc):** SDC is the *ratio-of-expectations* Dice instantiation and is
  coupling-invariant. `E_indep[Dice]=5/12`, `E_Qp[Dice]=1/3`, `SDC=1/2`, ratio `= 1/2`
  under **both** couplings.
- **Prop (floor):** `φ` is a **lower bound** on `m_c`. Confirmed per condition in §3c.
- **Lemma (perm):** SDC and every area readout are permutation-invariant; the distance
  readout is not. *With its reach honestly bounded* — the counterexample is a **scrambled**
  map, which no network emits, so it is an impossibility result over the full input space
  and only *suggestive* on the realistic manifold.

### Refuted — all pinned as FALSE in `tests/test_theory.py`

| claim | why it's wrong |
|---|---|
| the exact identity pairs each level with the gap **above** | drops the empty-mask atom; 0.566 vs a true 1.697 |
| `E_Q[Dice]` is coupling-invariant | it is an expectation of a **ratio**; three distinct values |
| `H(1/2) = 0` unconditionally | true only for binary |
| ...and even for binary | **only if no pixel ties at exactly ½** |
| the vertex of `H` is the argmax floor (`1/C`) | `φ` is a **lower bound** on `m_c`, not the vertex. No closed form |
| `H(t*) = 0 ⟺ C = 2` | false in both directions |
| `H` is unimodal | not guaranteed (`27.3 → 4.0 → 19.8 → 0.0`) |
| `H(1) = diag` | false when the map saturates to exactly 1.0 — float32 softmax routinely does |
| midpoint is right because it minimizes the Koksma bound | **non-sequitur** — a worse bound achieves a better error |
| "tuning buys nothing" | held on one synthetic generator; a second contradicts it |
| **SDC is coupling-free because Dice is permutation-invariant** | **permutation invariance ≠ coupling invariance.** SDC is coupling-free because it is a **ratio of expectations**: numerator and denominator are each *linear* in `Y`, so linearity of expectation collapses both onto the marginals. No coupling can move it |

**The root cause of four of these: `HD95` is a 95th percentile, not a metric.** No triangle
inequality, and **`HD95(A,B) = 0` does not imply `A = B`**.

### 3b. The contamination scare — RESOLVED, and it never bit

The dropped-present-class defect was **real** (a class can win the argmax while
`{p ≥ α}` is empty, and was silently dropped instead of saturated at the diagonal). It is
fixed and pinned by regression tests.

**But it never touched a headline number.** Natural experiment: `outputs/selective/summary.json`
(Jul 13 00:07) predates the fix (Jul 14 17:03) and the regen (18:06–18:50). Diffing AURC over
all 221 common scores, the scores that moved are **exactly 26 on `deeplabv3-external_voc` and
26 on `deeplabv3-target_voc`, ZERO on the other six — and every one is `@0.3`.**

- It could only ever fire where `φ <` the lowest threshold — i.e. only the **2** true
  21-way softmax conditions (§3c).
- **It never bit at the deployed `α = 0.1`**: no present class had max prob in `[0.048, 0.1)`.
- **The M-ladder was already clean by Jul 13** (all `r_M` deltas 0.0).

So `rem:mladdercontam`'s embargo ("do not cite") is **stale**, and the ladder table in
`docs/main.tex` reproduces **to the digit**. The audit was right to demand recomputation;
the recomputation vindicated the numbers.

### 3c. The argmax-floor taxonomy (already `prop:floor`(ii), now confirmed per condition)

**Only 2 of 8 conditions are a genuine 21-way softmax.**

| conditions | φ | why |
|---|---|---|
| `deeplabv3-{external,target}_voc` | **1/21 = 0.0476** | raw 21-way softmax |
| all 4 Pet | ½ | `C = 2` |
| both CLIPSeg VOC | ½ | `bg = 1 − max_c p_c`, so the winner is `max(M, 1−M) ≥ ½` |

Confirmed independently: `min p05_max_prob` dips below 0.5 on exactly those two softmax
conditions (0.359, 0.309) and never on the other six (≥ 0.501).

This rescopes **one** argument (where the dropped-class defect could fire — §3b) and
**nothing else**. In particular it does *not* explain the nesting leak (§2a).

---

## 4. Node placement — DECIDED, and mostly in the negative

**Node *placement* does not help. Node *count* is not what the headline measures.**

### The vertex-aware rule LOSES

`vtx2` (split at `m_c`, midpoint of each half — straddles the vertex by construction) was
built to test the straddle hypothesis. Against the deployed band, penalized boundary:

| rule | beats the band |
|---|---|
| dense M=32 | **8/8** |
| mid2 (0.25, 0.75) | 6/8 |
| **vtx2 (vertex-aware)** | **4/8 — worst** |

Head-to-head, `mid2` beats `vtx2` 6/8. On the only two conditions where the mechanism can
operate, `vtx2` is *worse than the band it was meant to repair*. This is the **fourth**
principled-looking adaptive node rule to fail empirically, joining rank-anchoring,
histogram importance sampling, and the `sqrt(V_L/V_R)` allocation.

### The straddle hypothesis is REFUTED — twice, independently

1. **The gain is largest exactly where the mechanism is switched off.** The band fails to
   straddle only on `deeplabv3-{external,target}_voc` (φ = 0.048), and those show the
   **smallest** VOC gains (+6.9% [1.9, 11.4], +5.6% [1.3, 10.0]) — significantly smaller
   (non-overlapping CIs) than `clipseg-{general,target}_voc` (+17.6% [15.0, 19.9], +14.7%
   [12.3, 17.2]), where φ = ½, the band already straddles, and straddling *provably cannot
   operate*.
2. **The discriminating test fails**: `vtx2` straddles by construction and loses (above).

### "5.6–17.6%" reproduces — but it is NOT a node-count measurement

The figure is **confirmed exactly** on clean data (17.56 / 14.67 / 6.90 / 5.63%, all four
CIs excluding zero). But its baseline is the **deployed band**, and "band → M=32" changes
**three** things: the readout family, the node placement, and the node count.

Decomposed at the **node-matched** rung `neg_r2_pmean_sym@0.1` (10k paired bootstrap,
% of band AURC, `*` = CI excludes 0):

| condition | READOUT | PLACEMENT | COUNT |
|---|---|---|---|
| clipseg-general_voc | +7.07* | **+9.26*** | +1.23 |
| clipseg-target_voc | +9.23* | +1.24 | +4.20* |
| deeplabv3-external_voc | +0.27 | −0.48 | **+7.10*** |
| deeplabv3-target_voc | +1.09 | −3.71 | **+8.24*** |
| clipseg-general_pet | +3.23* | +9.57* | **−10.45*** |
| clipseg-target_pet | +2.32* | +2.33 | **−4.35** |
| deeplabv3-external_pet | +1.93 | −0.79 | +5.34 |
| deeplabv3-target_pet | +2.23 | −3.91 | +3.08 |

**The COUNT leg is significantly positive 3×, significantly NEGATIVE 1×, and null 4×**
(a second cell, clipseg-target/Pet at −4.35%, is borderline — its bootstrap CI upper
bound sits just above zero across seeds).
On `clipseg-general_voc` — the **top of the 5.6–17.6% range** — the node-count component is
`+1.23 [−0.77, +3.40]`, indistinguishable from zero.

**So "more nodes help" is not what that range shows.** The honest reading: the total is a
*mixture*. On DeepLabV3-VOC it is essentially all node COUNT; on CLIPSeg-VOC it is
essentially all READOUT + PLACEMENT. The one clean, universal leg is READOUT (§1).

⚠️ **A 2-leg `band → gl2 → M32` split is NOT identified** — `gl2`'s nodes are
`(0.211, 0.789)` and the band's are `(0.1, 0.9)`, so that leg moves the readout *and* the
placement, and the attribution flips sign with the rung on all four DeepLabV3 conditions.
Use `neg_r2_pmean_sym@0.1` as the M=2 rung, or the decomposition means nothing. (This trap
was walked into once already.)

---

## 5. The posterior-assumption diagnostics — DEAD as published, and unfixable on this data

`levelset_auc` (0.985–0.999) and `residual_moran_i` (0.841–0.964) were going to be a
headline: *"our assumption nearly holds, SDC's is grossly violated everywhere."*

**Both statistics are implemented correctly. The headline is not licensed.** The failure is
**identification, not power.**

- **Both statistics test a CONJUNCTION**: `residual = (Y*−q) + (q−p)`, so each tests
  *"coupling AND `p = q`"*, and neither can attribute a violation to the coupling leg.
- **Data generated under SDC's OWN assumption reproduces the entire measured signature.**
  With `Y* ~ indep Bern(q)` for a deterministic mask `q`, and `p = blur(q)` (accurate but
  miscalibrated): `levelset_auc` 0.9995 — including **100% of images at AUC exactly 1.0**
  with a 6.3% non-degenerate band — `residual_moran_i` 0.84–0.95, `ρ(auc,miou) > 0`,
  `ρ(moran,miou) < 0`. Every row satisfies conditional independence *exactly*.
  So **"Moran 0.84–0.96 vs ~0 under independence" is not a refutation of independence** —
  the `~0` null silently assumes perfect calibration.
- **On Pet/VOC the question is vacuous.** The true marginal is near-deterministic (there is
  no real ambiguity about where the cat is), so **all couplings coincide**. What the two
  statistics actually measure is *model accuracy*: `ρ(levelset_auc, image_miou)` = +0.41 to
  +0.85 (**pooled +0.888**); `ρ(residual_moran_i, image_miou)` = −0.28 to −0.77 (pooled −0.49).
- **`Q_p` is itself refuted on 81% of images** (it predicts AUC = 1.0 *exactly*, with zero
  variance; measured AUC < 1 on 81%) — though the violation rate is tiny (median `1−AUC` = 2e-5).
- **"SDC is near-optimal on overlap anyway" is false**: it loses 6/8 to the band and is
  **1.2–2.7× worse** on all four Pet conditions (ratios 1.16 / 2.66 / 1.49 / 2.33). It is
  near-optimal on VOC only.
- And the framing invented a non-tension: **SDC's value is a function of `(p, ŷ)` alone**, so
  no coupling diagnostic can move it. It is not *surviving* a violated assumption; it is
  **invariant** to it.

### What would actually decide it

**Multiple annotations per image** (LIDC-IDRI, QUBIQ, multi-rater lesion sets). Estimate
`q̂` = mean over raters, then test the foreground-count spread: independence gives
`Var(|Y|) = Σ q(1−q)` (validated: MC sd 21.54 vs analytic 21.42); `Q_p` gives an sd
**23–75× larger** (22 px vs 830 px at matched sharpness). Decisive with a handful of
raters, closed-form null, and exactly what SDC's `<1%` bound needs.

**Pet/VOC have a single ground truth, so this cannot be run on the current data at all.**
That *is* the finding: **no reanalysis of these 8 JSONLs can license either half.** This
promotes the medical/multi-rater gap from "P3, a gap in coverage" to **a precondition for
the claim** (`docs/PLAN.md`).

Cheaper interim check (CPU-only, given cached `p`): **recalibrate** `p` (temperature or
per-pixel isotonic on a held-out split) and re-run both diagnostics. If Moran collapses
toward 0, the "gross violation" was miscalibration. Given `ρ(moran,miou) = −0.5..−0.77`,
a large collapse is predicted.

### 5b. Three numbers with no source — CUT

`"the independence gap is <0.4% while the Q_p gap is 4.7% mean / 26% max"` appears in **four
places that quote each other** — `docs/main.tex:400`, `HANDOFF.md:176`,
`scripts/show_results.py:207`, `selectseg/selective_eval.py:103` — and **no code anywhere
computes it.** Two of the four silently drop `main.tex`'s "measured on synthetic maps"
qualifier, so it reads as if it came from the real evaluation.

| number | verdict |
|---|---|
| **"4.7% mean / 26% max"** | **CUT. Do not try to recover the script.** It is *unidentifiable*: across defensible synthetic families the mean spans 0.21–11.37% and the max 0.38–97.45%. The number *is* the generator |
| "independence gap <0.4%" | **Keep, but re-source as a derivation**, not a measurement — under independence `Σ Y_i` concentrates, so `E[ratio] → ratio-of-expectations` at `O(1/N)` |
| "does not shrink with mass" | **Keep, stop calling it measured** — `main.tex` already *proves* it three lines earlier: `Q_p` is comonotone, so `Var(Σ Y_i)` sits at the Fréchet–Hoeffding maximum and the delta-method step never engages |

---

## 6. Where to look

| what | where |
|---|---|
| all numbers, recomputed live | `python scripts/show_results.py` |
| the paper | `docs/main.tex` |
| the plan and its gates | `docs/PLAN.md` |
| theory pinned as tests | `tests/test_theory.py` (140 pass) |
| the scripts that refuted claims | `scripts/refutations/` |
| per-image scores + true quality | `outputs/selective/*.jsonl` |

## 7. Standing lesson

**A derivation is a hypothesis until an independent check has tried to break it.** Every
error had the same shape: a clean derivation, asserted before it was tested. Twice the
disconfirming evidence was already in hand and was explained away.

Two additions from this round:
- **A decomposition must move one thing at a time — including the decomposition that
  accuses someone else of not doing so.** The `band → gl2` "readout leg" moved the nodes too.
- **A diagnostic that confirms your hypothesis has not confirmed it until you check what
  the rival hypothesis predicts for the same statistic.** Both couplings predicted the same
  AUC; the whole finding was an artifact of only ever computing our own side's null.
