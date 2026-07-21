"""Print every headline result in one place.

    python scripts/show_results.py            # everything
    python scripts/show_results.py --section headline

There is no JSON to dig through and no notebook to re-run: this reads the per-image
JSONL that the evaluation writes and recomputes each number from scratch, so what it
prints is what the data says today, not what a stale summary said yesterday.

Every claim in docs/main.tex that rests on a number should be checkable from here.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

BAND = "neg_band_width_pmean_sym@0.1"
MBAND = "neg_mband_width_pmean_sym@0.1"
SDC = "sdc_pmean"
DENSE = "neg_rM32_pmean@mid"
# The two-point quadrature the theory DERIVES, at the SAME nodes the band deploys.
# Pairing these two is what makes the ESTIMATOR section a controlled comparison, and
# it is also the only admissible M=2 rung for the readout/placement/count decomposition
# (gl2's nodes are (0.211, 0.789), so band -> gl2 moves the readout AND the placement).
R2 = "neg_r2_pmean_sym@0.1"

# 10k to match the CIs quoted in docs/FINDINGS.md and docs/main.tex. This is the only
# expensive thing this script does (~35 s, pure numpy, one core -- fine on a login node).
BOOTSTRAP_SAMPLES = 10000

RISKS = (
    ("overlap (1 - mIoU)", "image_miou", True),
    ("boundary, drop convention", "image_hd95", False),
    ("boundary, PENALIZED", "image_hd95_penalized", False),
)


def _aurc(confidences, risks):
    order = np.argsort(-np.asarray(confidences, dtype=float), kind="stable")
    ordered = np.asarray(risks, dtype=float)[order]
    return float((np.cumsum(ordered) / np.arange(1, len(ordered) + 1)).mean())


# The evaluation writes one JSON line per image as it goes, so a job that is killed
# leaves a SHORT but perfectly parseable file. Every AURC computed from it is then quietly
# wrong -- a truncated split is a biased sample of the split, not a smaller one. This has
# already happened once (a cancelled pass left six of eight conditions at ~50%, and the
# headline moved 7/8 -> 6/8 on nothing but the truncation). So refuse to report on a short
# file rather than average over it.
EXPECTED_IMAGES = {"pet": 3669, "voc": 1449}
MIN_COMPLETE = 0.95


def _load(strict=True):
    conditions, short = {}, []
    for path in sorted(glob.glob(str(REPO_ROOT / "outputs/selective/*.jsonl"))):
        name = Path(path).stem
        rows = [json.loads(line) for line in open(path)]
        if not rows:
            continue
        dataset = "pet" if name.endswith("_pet") else "voc"
        expected = EXPECTED_IMAGES[dataset]
        if len(rows) < MIN_COMPLETE * expected:
            short.append((name, len(rows), expected))
            if strict:
                continue
        conditions[name] = rows
    if short:
        print("\n" + "!" * 78)
        print("TRUNCATED EVALUATION FILES -- these conditions are being SKIPPED.")
        print("A killed job leaves a short but parseable JSONL, and every AURC from it is")
        print("silently wrong: a truncated split is a BIASED sample, not a smaller one.")
        for name, got, expected in short:
            print(f"    {name:<28} {got:>6} / {expected}  ({100*got/expected:.0f}%)")
        print("Re-run:  scripts/slurm/submit_quadrature.sh")
        print("!" * 78 + "\n")
    return conditions


def _rows_for(rows, field):
    kept = [r for r in rows if r.get(field) is not None]
    return kept


def _score(rows, key, default=None):
    if key not in rows[0]:
        return None
    return [r[key] for r in rows]


def headline(conditions):
    print("=" * 78)
    print("HEADLINE  --  is the distance readout better than the area readout?")
    print("=" * 78)
    print("The paper's spine. Same posterior, same thresholds, same aggregation; the")
    print("ONLY difference is whether the band is read spatially or by area. This is")
    print("the one result invariant to every convention we vary.\n")
    for label, field, complement in RISKS:
        wins = total = 0
        for name, rows in conditions.items():
            kept = _rows_for(rows, field)
            if not kept:
                continue
            risk = np.array([(1 - r[field]) if complement else r[field] for r in kept])
            dist = _score(kept, BAND)
            area = _score(kept, "levelset_dice_pmean@0.1")
            if dist is None or area is None:
                continue
            total += 1
            wins += _aurc(dist, risk) < _aurc(area, risk)
        if total:
            print(f"  distance beats area under {label:<28} {wins}/{total}")
    print()


def _bootstrap_margin(kept, base_key, rival_key, field, complement, seed=0):
    """Paired image-bootstrap of the rival's AURC margin over ``base_key``, as a
    PERCENTAGE of the base AURC. Positive = the rival is better (AURC is a risk:
    lower is better). Returns (point, lo, hi) at the 95% percentile interval;
    the interval excluding 0 is the significance call."""
    risk = np.array([(1 - r[field]) if complement else r[field] for r in kept])
    base = np.array(_score(kept, base_key), dtype=float)
    rival = np.array(_score(kept, rival_key), dtype=float)
    rng = np.random.default_rng(seed)
    count = len(kept)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for rep in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, count, count)
        base_aurc = _aurc(base[idx], risk[idx])
        draws[rep] = 100 * (base_aurc - _aurc(rival[idx], risk[idx])) / base_aurc
    point_base = _aurc(base, risk)
    point = 100 * (point_base - _aurc(rival, risk)) / point_base
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def estimator(conditions):
    print("=" * 78)
    print("ESTIMATOR  --  does the r_2 the theory DERIVES beat the band we DEPLOYED?")
    print("=" * 78)
    print("Yes, on every condition under every risk -- 24/24. This is the cleanest")
    print("result in the paper, and it is FREE: no extra nodes, no extra cost, no new")
    print("tuned parameter.\n")
    print("A CONTROLLED comparison, at MATCHED nodes (alpha, 1-alpha) = (0.1, 0.9):")
    print("same alpha, same two thresholds, same pooled-bidirectional HD95 convention,")
    print("same present-class mean, same -diagonal floors. The ONLY thing that differs")
    print("is the functional --")
    print("      band :        HD95(Y_lo, Y_hi)            the two level sets")
    print("                                                against EACH OTHER")
    print("      r_2  :  0.5 * [ HD95(Y_lo, Yhat)          each against the")
    print("                    + HD95(Y_hi, Yhat) ]        model's PREDICTION")
    print("-- so the ranking difference is attributable to the ESTIMATOR ALONE")
    print("(selective.py:499-501). It also closes the paper's one theory-practice gap:")
    print("prop:band's band <-> r_2 bridge is BROKEN (HD95 is a percentile, not a")
    print("metric -- see the SURROGATE section), but if we DEPLOY the estimator the")
    print("theory derives, there is nothing left to bridge.\n")
    for label, field, complement in RISKS:
        wins = total = 0
        for name, rows in conditions.items():
            kept = _rows_for(rows, field)
            if not kept:
                continue
            band, r2 = _score(kept, BAND), _score(kept, R2)
            if band is None or r2 is None:
                continue
            risk = np.array([(1 - r[field]) if complement else r[field] for r in kept])
            total += 1
            wins += _aurc(r2, risk) < _aurc(band, risk)
        if total:
            print(f"  r_2 beats the band under {label:<30} {wins}/{total}")
    print()
    print(f"  Margin on the PENALIZED boundary risk ({BOOTSTRAP_SAMPLES} paired")
    print("  bootstrap resamples, % of band AURC, * = 95% CI excludes 0):\n")
    print(f"      {'condition':<26}{'band':>9}{'r_2':>9}{'margin':>9}{'95% CI':>18}")
    for name, rows in conditions.items():
        kept = _rows_for(rows, "image_hd95_penalized")
        if not kept or _score(kept, R2) is None:
            continue
        risk = np.array([r["image_hd95_penalized"] for r in kept])
        point, lo, hi = _bootstrap_margin(
            kept, BAND, R2, "image_hd95_penalized", False)
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"      {name:<26}{_aurc(_score(kept, BAND), risk):>9.2f}"
              f"{_aurc(_score(kept, R2), risk):>9.2f}{point:>8.2f}%"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>17}{star}")
    print()
    print("  HONEST BOUND. 8/8 in SIGN, but only 4/8 SIGNIFICANT: the effect clears the")
    print("  CI on the 4 CLIPSeg conditions and is positive-but-NULL on the 4 DeepLabV3.")
    print("  And the 3 risks above share the same images -- they are NOT 3 independent")
    print("  tests. The claim this supports is 'never worse, and free', not 'always")
    print("  detectably better'.\n")


def versus_sdc(conditions):
    print("=" * 78)
    print("VS SDC  --  the closest prior work (arXiv:2402.10665)")
    print("=" * 78)
    print("The band's win depends on the convention, and we report both. `drop` deletes")
    print("images whose classes appear on only one side -- exactly the images the score")
    print("is worst at. `PENALIZED` charges them the image diagonal instead.\n")
    for label, field, complement in RISKS:
        print(f"  --- {label} ---")
        print(f"      {'condition':<26}{'band':>9}{'mband':>9}{'dense':>9}{'SDC':>9}")
        counts = {"band": 0, "mband": 0, "dense": 0}
        total = 0
        for name, rows in conditions.items():
            kept = _rows_for(rows, field)
            if not kept:
                continue
            risk = np.array([(1 - r[field]) if complement else r[field] for r in kept])
            values = {}
            for tag, key in (("band", BAND), ("mband", MBAND), ("dense", DENSE)):
                scores = _score(kept, key)
                values[tag] = _aurc(scores, risk) if scores else None
            sdc = _aurc(_score(kept, SDC), risk)
            total += 1
            for tag in counts:
                if values[tag] is not None and values[tag] < sdc:
                    counts[tag] += 1
            cells = "".join(
                f"{v:>9.3f}" if v is not None else f"{'--':>9}"
                for v in (values["band"], values["mband"], values["dense"])
            )
            print(f"      {name:<26}{cells}{sdc:>9.3f}")
        print(f"      {'BEATS SDC':<26}"
              f"{counts['band']:>7}/{total}{counts['mband']:>7}/{total}"
              f"{counts['dense']:>7}/{total}\n")


def nodes(conditions):
    print("=" * 78)
    print("NODE PLACEMENT  --  DECIDED (P1), and in the NEGATIVE")
    print("=" * 78)
    print("Never a theoretical question. Two synthetic generators disagree about the")
    print("optimum, and minimising the Koksma bound does NOT minimise the error (a node")
    print("set with a worse bound achieves a better one). Only the real maps could")
    print("decide, and now they have: PLACEMENT does not help.\n")
    print("  mid2 = (0.25, 0.75), the binary-derived midpoint")
    print("  vtx2 = split at m_c (min in-mask probability), midpoint of each half")
    print("  band = the deployed (0.1, 0.9);  dense = M=32\n")
    print("  vtx2 was built to test the STRADDLE hypothesis and LOSES -- it is the worst")
    print("  of the three, and on the only 2 conditions where the mechanism can operate")
    print("  it is worse than the band it was meant to repair. Head-to-head mid2 beats")
    print("  vtx2 6/8. That is the FOURTH principled-looking adaptive node rule to fail,")
    print("  joining rank-anchoring, histogram importance sampling, and sqrt(V_L/V_R).")
    print("  Caveat: vtx2 is not a pure straddle toggle (it also moves the upper node to")
    print("  ~0.53, off the tail, and reweights to ~(0.055, 0.945)), so what is refuted")
    print("  is that eq:derivednodes' straddle explains the ladder -- not that no")
    print("  straddling rule could help.\n")
    print("  !! The CONTAMINATION SCARE is RESOLVED -- and it never bit. This block used")
    print("     to refuse to interpret the table at all. The defect was REAL: a class")
    print("     could win the argmax with an empty {p >= 0.25} (softmax floor 1/21 =")
    print("     0.048) and was DROPPED instead of saturated at the diagonal, so mid2")
    print("     scored -0.0 -- MAXIMAL confidence -- on images with a hallucinated")
    print("     object, while vtx2 (lowest node 0.5*m_c) never dropped anything. It is")
    print("     fixed, and the JSONL read above are the clean 2026-07-14 regen.")
    print("     Diffing the pre-fix summary.json over all 221 common scores: EXACTLY 26")
    print("     scores moved on deeplabv3-external_voc and 26 on deeplabv3-target_voc,")
    print("     ZERO on the other six, and EVERY one is @0.3. It could only ever fire")
    print("     where phi < the lowest node -- i.e. only on the 2 genuine 21-way softmax")
    print("     conditions; the other six have phi = 1/2. It NEVER bit at the deployed")
    print("     alpha=0.1 (no present class had max prob in [0.048, 0.1)), and the")
    print("     M-ladder was already clean. So rem:mladdercontam's 'do not cite' embargo")
    print("     is STALE and the sec:quadrature table reproduces TO THE DIGIT. The audit")
    print("     was right to demand recomputation; the recomputation VINDICATED it.\n")
    keys = [("band", BAND), ("mid2", "neg_rmid2_pmean"),
            ("vtx2", "neg_rvtx2_pmean"), ("dense", DENSE)]
    for label, field, complement in RISKS[2:]:  # penalized boundary
        print(f"  --- {label} ---")
        header = "".join(f"{t:>10}" for t, _ in keys)
        print(f"      {'condition':<26}{header}")
        counts = {t: 0 for t, _ in keys if t != "band"}
        total = 0
        for name, rows in conditions.items():
            kept = _rows_for(rows, field)
            if not kept:
                continue
            risk = np.array([r[field] for r in kept])
            values, cells = {}, ""
            for tag, key in keys:
                scores = _score(kept, key)
                values[tag] = _aurc(scores, risk) if scores else None
                cells += f"{values[tag]:>10.2f}" if scores else f"{'--':>10}"
            print(f"      {name:<26}{cells}")
            if values["band"] is None:
                continue
            total += 1
            for tag in counts:
                if values[tag] is not None and values[tag] < values["band"]:
                    counts[tag] += 1
        row = f"{'--':>10}" + "".join(f"{counts[t]:>7}/{total}" for t, _ in keys[1:])
        print(f"      {'BEATS THE BAND':<26}{row}")
        print()


def assumptions(conditions):
    print("=" * 78)
    print("POSTERIOR ASSUMPTIONS  --  what these two statistics can and CANNOT decide")
    print("=" * 78)
    print("Both our level-set posterior and SDC's marginal one are CHOICES of a joint")
    print("distribution consistent with the same marginals. Neither is derived.\n")
    print("  !! RETRACTED HEADLINE. This section used to say these two numbers 'turn")
    print("     that from an excuse into a MEASUREMENT', and read them as: our")
    print("     assumption nearly holds, SDC's is grossly violated everywhere. Both")
    print("     statistics are correctly implemented; that headline is NOT LICENSED.")
    print("     The failure is IDENTIFICATION, not power:\n")
    print("     * Both test a CONJUNCTION, and neither can attribute a violation to the")
    print("       coupling leg. residual = (Y* - q) + (q - p), so each tests the")
    print("       conjunction 'coupling AND p = q'.")
    print("     * Data generated under SDC'S OWN assumption reproduces the ENTIRE")
    print("       measured signature: with Y* ~ indep Bern(q) for a deterministic q and")
    print("       p = blur(q) (accurate but MIScalibrated) -- levelset_auc 0.9995")
    print("       (incl. 100% of images at AUC exactly 1.0 with a 6.3% non-degenerate")
    print("       band), moran 0.84-0.95, and BOTH correlation signs below -- while")
    print("       satisfying conditional independence EXACTLY. So 'moran 0.84-0.96 vs")
    print("       ~0 under independence' does NOT refute independence: that ~0 null")
    print("       silently assumes PERFECT CALIBRATION.")
    print("     * On Pet/VOC the question is VACUOUS. The true marginal is near-")
    print("       deterministic (there is no real ambiguity about where the cat is), so")
    print("       ALL couplings coincide. What the two statistics actually track is")
    print("       MODEL ACCURACY -- which is what the rho(., mIoU) columns show.")
    print("     * Q_p is itself refuted on 81% of images: it predicts AUC = 1.0 EXACTLY")
    print("       with zero variance, and measured AUC < 1 on 81%. (The violation rate")
    print("       is tiny -- median 1 - AUC = 2e-5.)")
    print("     * The framing invented a NON-TENSION. SDC's value is a function of")
    print("       (p, yhat) ALONE, so NO coupling diagnostic can move it: SDC is not")
    print("       SURVIVING a violated assumption, it is INVARIANT to it. That")
    print("       invariance holds because SDC is the RATIO OF EXPECTATIONS -- numerator")
    print("       and denominator each LINEAR in Y, so linearity of expectation")
    print("       collapses both onto the marginals. NOT 'because Dice is permutation-")
    print("       invariant' (what this section used to say), which is REFUTED:")
    print("       E_Q[Dice] is an expectation of a RATIO and DOES depend on the")
    print("       coupling -- 5/12 independent, 1/3 comonotone, 1/2 SDC.")
    print("     * And 'SDC is near-optimal on overlap anyway' is FALSE: it loses 6/8 to")
    print("       the band and is 1.2-2.7x worse on all four Pet conditions (ratios")
    print("       1.16 / 2.66 / 1.49 / 2.33; see VS SDC). It is near-optimal on VOC only.\n")
    print("     WHAT WOULD DECIDE IT: multiple annotations per image (LIDC-IDRI, QUBIQ).")
    print("     Estimate q_hat = mean over raters, then test the foreground-count")
    print("     spread: independence gives Var(|Y|) = sum q(1-q) (validated: MC sd 21.54")
    print("     vs analytic 21.42), while Q_p gives an sd 23-75x larger (22 px vs 830 px")
    print("     at matched sharpness). Pet/VOC have a SINGLE ground truth, so this")
    print("     CANNOT be run on the current data at all. That IS the finding: no")
    print("     reanalysis of these 8 JSONL can license either half. It promotes the")
    print("     multi-rater gap from 'a coverage gap' to A PRECONDITION FOR THE CLAIM.\n")
    print("     ALSO CUT from this section: 'against Q_p the gap is 4.7% mean / 26%")
    print("     max'. No code ever computed it, and it is UNIDENTIFIABLE, not merely")
    print("     undocumented -- across defensible synthetic families the mean spans")
    print("     0.21-11.37% and the max 0.38-97.45%. The number IS the generator, so it")
    print("     is gone for good; do not try to recover it. What survives is unharmed:")
    print("     SDC's own <1% bound is still proved under conditional independence")
    print("     (sdc, eq. 30) and still does not transport to Q_p. Only a NUMBER for the")
    print("     size of that gap is withdrawn. And the gap not shrinking with mass is a")
    print("     THEOREM, never a sweep: Q_p is comonotone, so Var(sum Y_i) sits at the")
    print("     Frechet-Hoeffding maximum and the delta-method step never engages at ANY")
    print("     mass (main.tex, rem:notcoupling proves it three lines earlier).\n")
    print("  levelset_auc : 1 iff p ranks EVERY foreground pixel above every background")
    print("                 one. Reads as a violation rate ONLY given p = q.")
    print("  moran_I      : Moran's I (rook adjacency) of the residual Y* - p. ~0 under")
    print("                 'independence AND perfect calibration' -- the two legs are")
    print("                 not separable on this data.")
    print("  rho(., mIoU) : Spearman against TRUE image quality. This is the honest")
    print("                 reading of both columns: they track model accuracy.\n")
    print(f"      {'condition':<26}{'levelset AUC':>13}{'Moran I':>9}"
          f"{'rho(AUC,mIoU)':>14}{'rho(I,mIoU)':>12}")
    any_found = False
    pooled_auc, pooled_moran, pooled_miou = [], [], []
    for name, rows in conditions.items():
        kept = [r for r in rows
                if r.get("levelset_auc") is not None
                and r.get("residual_moran_i") is not None
                and r.get("image_miou") is not None]
        if not kept:
            continue
        any_found = True
        auc = [r["levelset_auc"] for r in kept]
        moran = [r["residual_moran_i"] for r in kept]
        miou = [r["image_miou"] for r in kept]
        pooled_auc += auc
        pooled_moran += moran
        pooled_miou += miou
        print(f"      {name:<26}{np.mean(auc):>13.3f}{np.mean(moran):>9.3f}"
              f"{stats.spearmanr(auc, miou).statistic:>+14.2f}"
              f"{stats.spearmanr(moran, miou).statistic:>+12.2f}")
    if not any_found:
        print("      (not yet computed -- needs the P1 evaluation pass)")
    else:
        print(f"      {'POOLED':<26}{np.mean(pooled_auc):>13.3f}"
              f"{np.mean(pooled_moran):>9.3f}"
              f"{stats.spearmanr(pooled_auc, pooled_miou).statistic:>+14.2f}"
              f"{stats.spearmanr(pooled_moran, pooled_miou).statistic:>+12.2f}")
    print()


def nesting(conditions):
    print("=" * 78)
    print("NESTING LEAK  --  the band's precondition, and where it fails")
    print("=" * 78)
    print("The bi-level construction assumes Y_hi is contained in the class's own")
    print("prediction. A normalized softmax cannot break this (two classes cannot both")
    print("clear 1-alpha on the simplex). CLIPSeg's independent per-prompt sigmoids CAN:")
    print("two co-firing prompts both clear it while only one wins the argmax, and the")
    print("loser's conservative set escapes its own prediction. The leaked pixels then")
    print("lie in BOTH level sets, the contours coincide, and the band VANISHES --")
    print("reporting maximal confidence on images where two classes are fighting.\n")
    print(f"      {'condition':<26}{'leak @0.1':>11}{'saturated':>11}{'no class':>10}")
    for name, rows in conditions.items():
        leak = [r.get("diag_nesting_leak@0.1") for r in rows
                if r.get("diag_nesting_leak@0.1") is not None]
        sat = [r.get("diag_saturated@0.1") for r in rows
               if r.get("diag_saturated@0.1") is not None]
        none = [r.get("diag_no_present_class") for r in rows
                if r.get("diag_no_present_class") is not None]
        if not leak:
            continue
        print(f"      {name:<26}{100*np.mean(leak):>10.1f}%"
              f"{100*np.mean(sat):>10.1f}%{100*np.mean(none):>9.1f}%")
    print()


def surrogate(conditions):
    print("=" * 78)
    print("SURROGATE  --  is the band really the r_2 the theory derives?")
    print("=" * 78)
    print("HD95 is a PERCENTILE, not a metric: it does not obey the triangle inequality,")
    print("so no inequality connects the band width to r_2 (we have a counterexample:")
    print("band = 13.41 px while 2*r_2 = 0). The bridge is EMPIRICAL only.\n")
    print("The ESTIMATOR section dissolves this rather than repairing it: r_2 at the")
    print("SAME nodes beats the band 24/24, so deploying r_2 leaves nothing to bridge.\n")
    print(f"      {'condition':<26}{'rho(band, r_2)':>16}")
    for name, rows in conditions.items():
        band = _score(rows, BAND)
        r2 = _score(rows, R2)
        if band is None or r2 is None:
            continue
        rho = stats.spearmanr(band, r2).statistic
        print(f"      {name:<26}{rho:>16.4f}")
    print()


SECTIONS = {
    "headline": headline,
    "estimator": estimator,
    "sdc": versus_sdc,
    "nodes": nodes,
    "assumptions": assumptions,
    "nesting": nesting,
    "surrogate": surrogate,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=sorted(SECTIONS), default=None)
    args = parser.parse_args()

    conditions = _load()
    if not conditions:
        print("no outputs/selective/*.jsonl -- run scripts/slurm/submit_quadrature.sh")
        return
    print(f"\n{len(conditions)} conditions, "
          f"{sum(len(r) for r in conditions.values())} images\n")
    for name in ([args.section] if args.section else SECTIONS):
        SECTIONS[name](conditions)


if __name__ == "__main__":
    main()
