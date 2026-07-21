"""Aggregate the selective-prediction experiment (METHODS.md §6).

Reads the per-image JSONL files written by selectseg.selective_eval and,
per condition and confidence score, reports AURC under three risks — the
overlap risks 1 - image mIoU and 1 - image mean Dice, and the boundary
risk image HD95 (on the images where HD95 is defined) — plus Spearman
correlations and a paired bootstrap of the default band width against SDC.
The rank-anchored band and r_2 are summarized alongside the constant-threshold
ones they generalize, as an ABLATION: the deployed score remains the
constant-threshold band width (DEFAULT_BAND).
Writes a machine-readable summary JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.selective import aurc, image_confidence_scores  # noqa: E402

# Risk name -> (per-image field, whether risk = 1 - field). HD95 is a raw
# distance (not a complement). ``hd95`` follows the medical-imaging convention
# of scoring only the classes present in both prediction and ground truth, so
# it is undefined on the images that fail to detect anything and drops them --
# which deletes the worst images from the boundary curve. ``hd95_penalized``
# charges those the image diagonal and is therefore defined on the same image
# set as the overlap risks; both are reported so the truncation is visible.
RISKS = (
    ("iou", "image_miou", True),
    ("dice", "image_mdice", True),
    ("hd95", "image_hd95", False),
    ("hd95_penalized", "image_hd95_penalized", False),
)

# !! CONTAMINATION WARNING -- THE QUADRATURE LADDER (M-ablation) OF THIS SCRIPT.
# Any ladder number aggregated from JSONL written BEFORE the _present_readouts fix is
# CONTAMINATED and must not be cited (docs/main.tex rem:mladdercontam,
# docs/FINDINGS.md §3b). A class can WIN the argmax while its aggressive set
# {p >= alpha} is empty -- a C-way softmax winner needs only p >= 1/C = 0.048 on VOC,
# below alpha = 0.1 -- and the aggregation silently DROPPED such a class instead of
# saturating it at the diagonal. The ladder's rules do not all fire the same way: the
# floor 0.048 sits BELOW the lowest nodes of mid16 (0.031) and mid32 (0.016) but ABOVE
# those of gl2 (0.211), mid4 (0.125) and mid8 (0.062), so the dense rules saw those
# classes and the cheap ones did not. The ladder was comparing rules computed over
# DIFFERENT SETS OF CLASSES ON THE SAME IMAGE, and the dense rule's advantage may be an
# artifact: it was not integrating better, it was the only rule looking. The bug is
# fixed in selectseg.selective (_present_readouts, and now the mid2/vtx2 candidate
# block too), but THE FIX IS IN THE SCORER, NOT HERE -- this script only aggregates
# what the JSONL already contains. Regenerate outputs/selective/*.jsonl before citing
# any ladder or node-placement number.
#
# The distance-vs-area headline is unaffected: both readouts share the aggregation.

# The band-width variant every table in the paper is reported against, and its
# strongest rival, paired for the head-to-head bootstrap.
#
# THIS IS THE *REPORTING* DEFAULT, NOT AN ENDORSEMENT. docs/main.tex (eq:mband) and
# docs/FINDINGS.md §2a both RECOMMEND the mask-intersected band (neg_mband_width_*) as
# the DEPLOYED form on any non-normalized map: it is a provable no-op where the nesting
# leak is zero (47.876 -> 47.876 on deeplabv3-external, machine precision) and closes
# 42-77% of the band's AURC gap to SDC where it is not. The unrepaired band is kept
# here only so that every number in the paper refers to one fixed score; it flips no
# cell either way. Switching this constant is an author decision that invalidates every
# table until they are regenerated -- see docs/PLAN.md P1.3.
DEFAULT_BAND = "neg_band_width_pmean_sym@0.1"
SDC = "sdc_pmean"
BOOTSTRAP_SAMPLES = 1000

# The two ablations of METHODS.md §6.2, both read as rank correlations because
# AURC depends on a score only through its ranking. The surrogate ablation
# pairs the deployed band width with the two-point quadrature r_2 it stands in
# for; the quadrature ablation ranks the cheap rules (and both M=2 scores)
# against the dense M=32 reference.
DEFAULT_R2 = "neg_r2_pmean_sym@0.1"
# The rank-anchored ('q') band and its r_2 -- AN ABLATION, NOT A CANDIDATE
# DEFAULT. They clip the conservative threshold to the 95th percentile of each
# class's probabilities inside its own predicted mask.
#
# They coincide with the constant-threshold pair on every class whose clip is
# inactive or reverted, which diag_clipped@ counts -- NOT on every image where
# 1 - alpha is achievable. The clip fires whenever P95(in-mask) < 1 - alpha,
# which strictly contains the saturated set {max p < 1 - alpha}: a ranking
# difference between the two is therefore NOT attributable to the degenerate
# images alone, and on the clipped-but-healthy images the fix is a
# reparameterization of a score that was working. Compare diag_clipped@ with
# diag_saturated@ to size that; a synthetic study has the ranking degrading
# there, so read spearman_vs_dense (below) and the AURC table before believing
# the ties it breaks were worth breaking.
RANK_ANCHORED_BAND = "neg_qband_width_pmean_sym@0.1"
RANK_ANCHORED_R2 = "neg_qr2_pmean_sym@0.1"
DENSE_QUADRATURE = "neg_rM32_pmean@mid"
QUADRATURE_LADDER = (
    DEFAULT_BAND,
    RANK_ANCHORED_BAND,
    DEFAULT_R2,
    RANK_ANCHORED_R2,
    "neg_rM2_pmean@gl",
    "neg_rM4_pmean@mid",
    "neg_rM8_pmean@mid",
    "neg_rM16_pmean@mid",
    DENSE_QUADRATURE,
)

# Rank correlations reported alongside the ladder. The first is the surrogate
# ablation.
#
# NOTE: NO THEORY PREDICTS rho = 1 HERE. An earlier version of this comment said
# "Prop. 4 predicts rho = 1 under coherent nesting" -- wrong twice over. (i) The
# propositions renumbered: Prop. 4 is now prop:floor (the argmax floor), which says
# nothing about band-vs-r_2; the one meant is prop:band. (ii) prop:band never
# predicted a rank correlation of 1. It bounds HD(dY_lo, dY_hi) <= 2 r_2 for the MAX
# Hausdorff distance only, and rem:nobridge withdraws all three ingredients of the old
# claim: HD95 is a percentile, not a metric, so the bound does not transfer (band =
# 13.41 px where 2 r_2 = 0, an unbounded violation); there is NO equality condition
# ("we claim none"; measured 0/398); and a one-sided inequality is not benign for a
# pure rank statistic (Kendall tau = 0.81, 9.5% discordant pairs). So the shortfall
# below does NOT measure "how often nesting is incoherent" -- nesting is guaranteed by
# construction for alpha < 1/2. The measured rho IS the empirical bridge (0.974-0.992
# across the eight conditions) and is the only thing connecting the deployed band width
# to the r_2 it stands in for.
#
# The second pair is the same ablation for the rank-anchored pair; the third measures
# how far the clip actually reorders the split. It is bounded away from 1 by the
# CLIPPED images, not the saturated ones: a condition with no saturation at all can
# still have rho < 1, and only a condition with diag_clipped@ == 0 must read exactly 1.
SPEARMAN_PAIRS = (
    ("spearman_band_vs_r2", "band vs r2 (surrogate)", DEFAULT_BAND, DEFAULT_R2),
    (
        "spearman_qband_vs_qr2",
        "qband vs qr2 (surrogate)",
        RANK_ANCHORED_BAND,
        RANK_ANCHORED_R2,
    ),
    (
        "spearman_band_vs_qband",
        "band vs qband (the fix)",
        DEFAULT_BAND,
        RANK_ANCHORED_BAND,
    ),
)

# Fields that are not confidence scores and must not be ranked as such: the
# true quality metrics, and the ``diag_`` diagnostics (degeneracy rates) that
# tell us whether a given alpha even produces a usable band.
#
# score_keys is built by EXCLUSION, so anything not named here is ranked, given
# an AURC and a bootstrap CI, and printed in the table. That makes this list a
# label-leak boundary: a new ground-truth field that someone forgets to add here
# does not crash, it lands in the results as a miraculous score that closes 100%
# of the oracle gap (verified: injecting image_boundary_iou = image_miou yields
# aurc == oracle_aurc, gap_closure 1.0, spearman 1.0). Two belts: every
# label-seeing field carries the ``image_`` prefix or is named below, and
# _assert_no_label_leak checks the leftover key set against the score registry.
METRIC_FIELDS = (
    "index",
    "image_miou",
    "image_mdice",
    "image_hd95",
    "image_hd95_penalized",
    # ground-truth diagnostics of the two rival posterior ASSUMPTIONS -- not
    # confidence scores (they see the label), and must never be ranked as such
    "levelset_auc",
    "residual_moran_i",
)
DIAGNOSTIC_PREFIX = "diag_"
# every field computed against the ground truth is named image_*; no confidence
# score is (they are mean_max_prob, neg_band_width_*, sdc_*, buc, ...)
METRIC_PREFIX = "image_"

# Scores that are NOT per-image functionals of x: their value depends on the other
# images in the split, so they cannot be evaluated at deployment time on a single
# image and they violate the ranking contract eq:rank. They are still reported --
# they are informative -- but they are marked, here and in summary.json, so a
# reader cannot mistake one for a deployable score.
TRANSDUCTIVE_SCORES = ("combined_band_sdc",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["outputs/selective", "outputs/selective_b"],
    )
    parser.add_argument("--output", default="outputs/selective/summary.json")
    return parser.parse_args()


def _finite_or_none(value):
    """None for NaN/inf (e.g. Spearman of a constant score)."""
    return float(value) if np.isfinite(value) else None


def _risk_values(rows, field, complement):
    values = np.array([row[field] for row in rows], dtype=float)
    # numpy turns a None into a silent NaN, and a single NaN risk poisons every
    # AURC under that risk (and makes the bootstrap read low rather than NaN,
    # because NaN comparisons are always False). Fail loudly instead.
    if not np.isfinite(values).all():
        raise ValueError(
            f"{field} has {int((~np.isfinite(values)).sum())} non-finite value(s); "
            "a None/NaN risk silently NaNs every AURC under this risk"
        )
    return 1 - values if complement else values


def _aurc_np(confidences, risks):
    """Vectorized AURC (accept in descending-confidence order)."""
    order = np.argsort(-confidences, kind="stable")
    ordered = risks[order]
    return float((np.cumsum(ordered) / np.arange(1, len(ordered) + 1)).mean())


def _bootstrap_band_beats_sdc(rows, field, complement, seed=0):
    """Fraction of paired image-bootstrap resamples where the default band
    width has a strictly lower AURC than SDC (a significance proxy)."""
    risks = _risk_values(rows, field, complement)
    band = np.array([row[DEFAULT_BAND] for row in rows])
    sdc = np.array([row[SDC] for row in rows])
    rng = np.random.default_rng(seed)
    count = len(rows)
    wins = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, count, count)
        wins += _aurc_np(band[idx], risks[idx]) < _aurc_np(sdc[idx], risks[idx])
    return wins / BOOTSTRAP_SAMPLES


def _add_combined_score(rows):
    """Rank-average of the default band width and SDC — a two-signal score
    combining boundary geometry with the overlap-aligned SDC.

    TRANSDUCTIVE, and listed in :data:`TRANSDUCTIVE_SCORES` for that reason. The
    ranks are taken across the whole split, so this is NOT a per-image functional
    g: X -> R: its value (and, for ~1% of pairs, its ordering) depends on which
    other images are in the batch, and scored one image at a time it collapses to
    the constant 1.0. It therefore violates the ranking contract eq:rank and
    cannot be deployed as written. It is reported because it is informative, and
    marked because it is not deployable; the honest deployable version freezes the
    two rank transforms (empirical CDFs) on a held-out split, which measures the
    same or better.
    """
    band_rank = stats.rankdata([row[DEFAULT_BAND] for row in rows])
    sdc_rank = stats.rankdata([row[SDC] for row in rows])
    for row, combined in zip(rows, (band_rank + sdc_rank) / 2):
        row["combined_band_sdc"] = float(combined)


def _spearman(rows, left, right):
    return _finite_or_none(
        stats.spearmanr(
            [row[left] for row in rows], [row[right] for row in rows]
        ).statistic
    )


def _quadrature_agreement(rows):
    """Rank agreement of the cheap estimators with the dense reference.

    ``spearman_vs_dense`` is the quadrature ablation: how much ranking each
    rule loses against dense M=32 quadrature — reported for the rank-anchored
    band and r_2 alongside the constant-threshold ones. This is the decisive
    column for the clip: moving the upper node down to the map's own 95th
    in-mask percentile stops the estimator probing the levels the map never
    reaches, which is where its uncertainty lives, so if the clip degrades the
    estimator, qband/qr2 will sit *below* band/r2 against the dense reference.

    The pairs of :data:`SPEARMAN_PAIRS` add the surrogate ablation and the
    band-vs-qband correlation, which is how much of the ranking the clip rewrites.
    NO theory predicts rho = 1 for the surrogate ablation: an earlier version of this
    docstring claimed "Prop. 4 predicts a rank correlation of 1 under coherent nesting,
    so the shortfall measures how often nesting is incoherent", and that is REFUTED.
    prop:band bounds the MAX Hausdorff distance only, HD95 is a percentile and not a
    metric, so the bound does not transfer (rem:nobridge); there is no equality
    condition; and a one-sided inequality permutes a rank statistic anyway. The
    measured rho is the EMPIRICAL bridge, not a shortfall against a predicted 1. See
    the comment on :data:`SPEARMAN_PAIRS`.

    Keys missing from the JSONL (written before a score existed) are skipped.
    """
    if DENSE_QUADRATURE not in rows[0]:
        return None
    agreement = {
        key: _spearman(rows, key, DENSE_QUADRATURE)
        for key in QUADRATURE_LADDER
        if key in rows[0]
    }
    result = {"spearman_vs_dense": agreement}
    for name, _, left, right in SPEARMAN_PAIRS:
        if left in rows[0] and right in rows[0]:
            result[name] = _spearman(rows, left, right)
    return result


def _require(rows, keys):
    """Fail loudly, and by name, on a JSONL that cannot answer the question.

    The risk loop and the quadrature ladder skip keys they do not find, because
    those really are optional (a JSONL written before a score existed). The band
    and SDC are NOT optional: they are the head-to-head this script exists to run,
    and DEFAULT_BAND is pinned to alpha=0.1 while ``--alphas`` is a user-settable
    sweep on selective_eval -- so a grid that omits 0.1 produces a JSONL with no
    band key at all. Silently reporting a summary with the headline comparison
    missing would be worse than stopping.
    """
    missing = [key for key in keys if key not in rows[0]]
    if missing:
        raise ValueError(
            f"the JSONL is missing required score key(s) {missing}; "
            "selective_eval must be run with an --alphas grid that includes 0.1"
        )


def _score_registry(row):
    """The set of keys :func:`image_confidence_scores` actually emits.

    Recovered by running it on a tiny synthetic map with the alphas the JSONL
    itself was written with (read back off its own ``diag_saturated@`` keys).
    Turns the deny-list into an allow-list: see :func:`_assert_no_label_leak`.
    """
    alphas = sorted(
        float(key.split("@", 1)[1])
        for key in row
        if key.startswith(f"{DIAGNOSTIC_PREFIX}saturated@")
    )
    probe = torch.rand(3, 8, 8)
    return set(image_confidence_scores(probe, alphas or [0.1]))


def _assert_no_label_leak(score_keys, row):
    """No field computed against the ground truth may be ranked as a score.

    score_keys is built by EXCLUSION, so a label-seeing field that nobody
    remembered to add to METRIC_FIELDS gets ranked -- and scores *perfectly*,
    because it is the risk. Silent and flattering is the worst combination a
    failure can have, so this is an assertion rather than a comment: every ranked
    key must be one the score library actually emits, or an explicitly declared
    transductive score. Anything else is unclassified and stops the run.
    """
    unclassified = sorted(
        set(score_keys) - _score_registry(row) - set(TRANSDUCTIVE_SCORES)
    )
    if unclassified:
        raise ValueError(
            f"key(s) {unclassified} are not emitted by image_confidence_scores "
            "and are not declared transductive, yet would be ranked as confidence "
            "scores. If they see the ground truth, add them to METRIC_FIELDS."
        )


def analyze(rows):
    _require(rows, (DEFAULT_BAND, SDC))
    _add_combined_score(rows)
    score_keys = sorted(
        key
        for key in rows[0]
        if key not in METRIC_FIELDS
        and not key.startswith(DIAGNOSTIC_PREFIX)
        and not key.startswith(METRIC_PREFIX)
    )
    _assert_no_label_leak(score_keys, rows[0])
    hd_rows = [row for row in rows if row.get("image_hd95") is not None]
    result = {
        "num_images": len(rows),
        "num_images_hd95": len(hd_rows),
        # scores whose value depends on the rest of the split, so they are not
        # deployable per-image functionals (eq:rank). Reported, but marked.
        "transductive_scores": [
            key for key in TRANSDUCTIVE_SCORES if key in score_keys
        ],
        # label-free degeneracy diagnostics: the share of images with no
        # predicted foreground class at all, and (per alpha) the share of present
        # classes whose conservative level set is empty so the band saturates
        # (diag_saturated), whose rank-anchored clip is active so the two bands
        # differ (diag_clipped), and whose clip had to be abandoned because it
        # would have collapsed the band -- a flat in-mask distribution, or
        # t_hi <= alpha (diag_clip_reverted). These are what alpha should be
        # selected against -- a saturated band carries no ranking information,
        # and needs no ground truth to detect. diag_clipped >= diag_saturated is
        # the price of the fix: the gap is the share of *healthy* classes whose
        # score it rewrites.
        "diagnostics": {
            key: float(np.mean([row[key] for row in rows]))
            for key in sorted(rows[0])
            if key.startswith(DIAGNOSTIC_PREFIX)
        },
        "risks": {},
        "scores": {key: {} for key in score_keys},
    }

    for name, field, complement in RISKS:
        # a JSONL written before a risk existed (the released outputs predate
        # image_hd95_penalized) simply drops that risk rather than crashing
        if field not in rows[0]:
            continue
        # Filter per RISK, not only for hd95. image_hd95 is None wherever no class
        # is present on both sides (the medical convention), and a *stale* JSONL
        # can carry a None in image_hd95_penalized too -- which numpy would coerce
        # to a silent NaN that poisons every AURC under that risk. hd95_penalized
        # is written as 0.0 now (see selective_eval.image_quality), so this is
        # belt-and-braces; num_images makes any truncation visible either way.
        subset = [row for row in rows if row.get(field) is not None]
        if not subset:
            continue
        risks = _risk_values(subset, field, complement)
        random_aurc = float(risks.mean())
        oracle_aurc = _aurc_np(-risks, risks)
        gap = random_aurc - oracle_aurc
        result["risks"][name] = {
            "num_images": len(subset),
            "random_aurc": random_aurc,
            "oracle_aurc": oracle_aurc,
            "band_beats_sdc_bootstrap": _bootstrap_band_beats_sdc(
                subset, field, complement
            ),
        }
        for key in score_keys:
            score_aurc = _aurc_np(np.array([row[key] for row in subset]), risks)
            result["scores"][key][name] = {
                "aurc": score_aurc,
                "e_aurc": score_aurc - oracle_aurc,
                "gap_closure": (random_aurc - score_aurc) / gap if gap else None,
            }

    # Correlate against qualities (higher = better), so a good confidence
    # score is positive for both: image mIoU, and negative HD95 (boundary
    # quality). A constant score yields NaN, reported as None.
    miou = [row["image_miou"] for row in rows]
    hd_quality = [-row["image_hd95"] for row in hd_rows]
    for key in score_keys:
        result["scores"][key]["spearman_iou"] = _finite_or_none(
            stats.spearmanr([row[key] for row in rows], miou).statistic
        )
        result["scores"][key]["spearman_hd95_quality"] = _finite_or_none(
            stats.spearmanr([row[key] for row in hd_rows], hd_quality).statistic
        )
    result["quadrature"] = _quadrature_agreement(rows)
    return result


def main():
    args = parse_args()
    summary = {}
    for directory in args.input_dirs:
        config = "B" if directory.rstrip("/").endswith("_b") else "A"
        for path in sorted(Path(directory).glob("*.jsonl")):
            rows = [json.loads(line) for line in path.open()]
            if rows:
                summary[f"{path.stem} ({config})"] = analyze(rows)

    for name, result in summary.items():
        print(f"\n=== {name}: {result['num_images']} images ===")
        for risk_name, risk in result["risks"].items():
            print(
                f"  [{risk_name}] n={risk['num_images']} "
                f"random {risk['random_aurc']:.4f} oracle {risk['oracle_aurc']:.4f} "
                f"band>SDC bootstrap {risk['band_beats_sdc_bootstrap']:.2f}"
            )
        print(f"  {'score':<32}" + "".join(f"{n:>10}" for n, _, _ in RISKS))
        for key in sorted(result["scores"]):
            cells = "".join(
                f"{result['scores'][key][n]['aurc']:>10.4f}"
                if n in result["scores"][key]
                else f"{'-':>10}"
                for n, _, _ in RISKS
            )
            print(f"  {key:<32}{cells}")
        quadrature = result["quadrature"]
        if quadrature:
            print(f"  {'spearman vs dense (M=32)':<32}")
            for key, rho in quadrature["spearman_vs_dense"].items():
                cell = f"{rho:>10.4f}" if rho is not None else f"{'-':>10}"
                print(f"    {key:<30}{cell}")
            for name, label, _, _ in SPEARMAN_PAIRS:
                if name not in quadrature:
                    continue
                rho = quadrature[name]
                cell = f"{rho:>10.4f}" if rho is not None else f"{'-':>10}"
                print(f"    {label:<30}{cell}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
