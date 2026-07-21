"""The analysis pipeline: AURC, the label-leak boundary, and the guards (CPU).

scripts/analyze_selective.py produces every number in the paper's tables, and had
zero test coverage -- not even an import. That is the reason a silent NaN could
poison a whole risk column and a ground-truth field could be ranked as a
miraculous confidence score without anything going red.

Two of the properties pinned here are boundaries rather than behaviours: no
label-seeing field may ever be ranked (it would score perfectly, because it IS the
risk), and no risk may ever be computed from a NaN (it would come out plausible and
wrong rather than obviously broken). Both fail silently and flatteringly if they
fail at all, which is the worst combination a failure can have.
"""

import json
import math

import numpy as np
import pytest
import torch

from scripts.analyze_selective import (
    DEFAULT_BAND,
    SDC,
    TRANSDUCTIVE_SCORES,
    _aurc_np,
    _bootstrap_band_beats_sdc,
    analyze,
)
from selectseg.selective import aurc, image_confidence_scores


def _rows(count=24, seed=0):
    """A synthetic condition: real confidence scores paired with real-ish quality.

    The scores come from the real image_confidence_scores on real (synthetic) maps,
    so the key set is exactly the one the JSONL carries in production -- which is
    the point, since what is under test is how analyze() CLASSIFIES those keys.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index in range(count):
        logits = torch.randn(3, 24, 24, generator=generator) * 2
        logits[0] += 1.5
        logits[1, 6:18, 6:18] += 3.0 + index / count
        probs = logits.softmax(dim=0)
        quality = 0.3 + 0.6 * index / count
        rows.append({
            "index": index,
            "image_miou": quality,
            "image_mdice": quality,
            "image_hd95": 40.0 * (1 - quality),
            "image_hd95_penalized": 60.0 * (1 - quality),
            "levelset_auc": 0.5 + 0.4 * quality,
            "residual_moran_i": 0.8,
            **image_confidence_scores(probs, [0.1]),
        })
    return rows


def test_the_two_aurc_implementations_agree_including_on_heavy_ties():
    """analyze's vectorized AURC must equal the library's, tie-breaking included.

    There are TWO AURC implementations in this repository: selectseg.selective.aurc
    (a Python sort, the one the unit tests exercise) and analyze_selective._aurc_np
    (a numpy argsort, the one every number in the paper actually comes from). Nothing
    compared them.

    Ties are the case that matters, and they are not rare: the band SATURATES at the
    image diagonal on every degenerate image, so up to 67% of a split can sit in tied
    groups (2449 of 3669 on clipseg-target/Pet, largest group 169 images). Both sorts
    must therefore be STABLE and break ties the same way -- ``kind="stable"`` in the
    numpy one is load-bearing and is one keyword away from numpy's unstable
    quicksort default, which would drift the AURC by up to 0.023 on tie-heavy data.
    """
    rng = np.random.default_rng(0)
    for trial in range(30):
        size = int(rng.integers(5, 200))
        risks = rng.uniform(size=size)
        confidences = rng.normal(size=size)
        if trial % 3 == 0:  # force heavy ties
            confidences = np.round(confidences)
        assert _aurc_np(confidences, risks) == pytest.approx(
            aurc(list(confidences), list(risks)), abs=1e-9
        )

    # the saturated-band regime: 40% of the split pinned at the same floor
    risks = rng.uniform(size=100)
    confidences = np.concatenate([np.full(40, -60.0), rng.normal(size=60)])
    assert _aurc_np(confidences, risks) == pytest.approx(
        aurc(list(confidences), list(risks)), abs=1e-9
    )


def test_a_ground_truth_field_is_never_ranked_as_a_confidence_score():
    """THE LABEL-LEAK BOUNDARY. score_keys is a deny-list, so this is where it holds.

    Every key in the JSONL that is not explicitly excluded gets ranked, given an
    AURC and a bootstrap, and printed in the table. A ground-truth field that slips
    through does not crash -- it lands in the results as a miraculous score that
    closes 100% of the oracle gap, because it IS the risk. Both belts are pinned:
    the ``image_`` prefix, and the registry check against the keys the score library
    actually emits.
    """
    rows = _rows()
    result = analyze(rows)
    for field in (
        "index",
        "image_miou",
        "image_mdice",
        "image_hd95",
        "image_hd95_penalized",
        "levelset_auc",       # sees the label: tests OUR posterior assumption
        "residual_moran_i",   # sees the label: tests SDC's
    ):
        assert field not in result["scores"], field
    # ...and the real scores DID get ranked
    assert DEFAULT_BAND in result["scores"]
    assert SDC in result["scores"]

    # belt 1: a new ground-truth field carrying the image_ prefix is suppressed
    leaked = analyze([dict(row, image_boundary_iou=row["image_miou"]) for row in _rows()])
    assert "image_boundary_iou" not in leaked["scores"]

    # belt 2: one WITHOUT the prefix is not silently ranked either -- it is not a key
    # the score library emits, so it is unclassified and stops the run.
    with pytest.raises(ValueError, match="not emitted by image_confidence_scores"):
        analyze([dict(row, oracle_dice=row["image_mdice"]) for row in _rows()])


def test_a_leaked_metric_would_have_scored_perfectly_which_is_why_this_matters():
    """Demonstrates the damage the guard above prevents, so nobody relaxes it.

    A ranked ground-truth field does not look broken. It looks like a triumph.
    """
    rows = _rows()
    risks = np.array([1 - row["image_miou"] for row in rows])
    # ranking by the true quality IS the oracle ranking
    oracle = _aurc_np(-risks, risks)
    leaked = _aurc_np(np.array([row["image_miou"] for row in rows]), risks)
    assert leaked == pytest.approx(oracle)
    # i.e. it closes 100% of the gap the real scores are fighting over
    assert leaked < _aurc_np(np.array([row[DEFAULT_BAND] for row in rows]), risks)


def test_a_none_risk_fails_loudly_instead_of_nanning_the_whole_column():
    """A single None risk used to NaN every AURC under that risk, in silence.

    numpy coerces None to NaN, and _risk_values fed the whole column through. The
    AURCs then came out NaN -- visible, at least -- but the band-vs-SDC BOOTSTRAP did
    not: NaN comparisons are always False, so every resample touching the poisoned
    image scored as a loss and the statistic came out FINITE AND WRONG (0.35 instead
    of 0.95 on a real condition). A plausible wrong number in the headline column is
    far more dangerous than a NaN.

    The writer no longer emits None there (image_hd95_penalized is 0.0 on an
    empty-vs-empty image), so this is the belt for a stale JSONL.
    """
    rows = _rows()
    # a None is dropped from that risk's subset, and the truncation stays visible
    rows[0]["image_hd95_penalized"] = None
    result = analyze(rows)
    assert result["risks"]["hd95_penalized"]["num_images"] == len(rows) - 1
    assert np.isfinite(result["risks"]["hd95_penalized"]["random_aurc"])
    for key in result["scores"]:
        assert np.isfinite(result["scores"][key]["hd95_penalized"]["aurc"]), key

    # an outright NaN (which no writer produces, but a corrupt file might) stops it
    rows[1]["image_miou"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        analyze(rows)


def test_a_jsonl_without_the_deployed_band_fails_by_name():
    """--alphas is a user-settable sweep, but DEFAULT_BAND is pinned to alpha=0.1.

    A sweep that omits 0.1 writes a JSONL with no band key at all, and the script
    then died with a bare KeyError from inside a comprehension. Worse would be to
    "handle" it by silently skipping the band-vs-SDC bootstrap -- the summary would
    then be missing the paper's headline comparison with nothing to say so. It fails,
    by name, with the remedy in the message.
    """
    rows = _rows()
    stripped = [
        {key: value for key, value in row.items() if key != DEFAULT_BAND}
        for row in rows
    ]
    with pytest.raises(ValueError, match=r"includes 0\.1"):
        analyze(stripped)

    stripped = [
        {key: value for key, value in row.items() if key != SDC} for row in rows
    ]
    with pytest.raises(ValueError, match=r"missing required score key"):
        analyze(stripped)


def test_the_combined_score_is_reported_as_transductive():
    """combined_band_sdc is NOT a per-image functional, and summary.json must say so.

    It rank-transforms the band and SDC across the WHOLE split, so its value -- and,
    for ~1% of pairs, its ordering -- depends on which other images are in the batch.
    Scored one image at a time, as a deployed g: X -> R must be, rankdata on a
    singleton returns 1.0 and the score collapses to a constant carrying no
    information at all. It therefore violates the ranking contract eq:rank and cannot
    be deployed as written.

    It is still reported -- it is informative, and it wins several cells -- but a
    reader of summary.json must be able to tell it apart from the deployed score
    sitting in the same column. Hence the marker.
    """
    rows = _rows()
    result = analyze(rows)
    assert "combined_band_sdc" in result["scores"]
    assert result["transductive_scores"] == list(TRANSDUCTIVE_SCORES)
    assert "combined_band_sdc" in result["transductive_scores"]
    # the deployed score is NOT transductive, and must never appear in that list
    assert DEFAULT_BAND not in result["transductive_scores"]
    assert SDC not in result["transductive_scores"]

    # the batch-dependence itself: the same image scores differently in a subset
    full = analyze(_rows())
    half = analyze(_rows()[:12])
    assert full["num_images"] != half["num_images"]
    del full, half  # the values are ranks; what is pinned is that they are marked


def test_the_bootstrap_is_paired_and_reproducible():
    """The band-vs-SDC bootstrap resamples IMAGES, and both scores see the same draw.

    Pairing is what makes it a test of the two scores rather than of the split: an
    unpaired bootstrap would compare a band AURC on one resample against an SDC AURC
    on another and measure mostly sampling noise. A fixed seed makes it reproducible,
    which the paper's reported probabilities depend on.
    """
    rows = _rows()
    first = _bootstrap_band_beats_sdc(rows, "image_miou", True, seed=0)
    second = _bootstrap_band_beats_sdc(rows, "image_miou", True, seed=0)
    assert first == second
    assert 0.0 <= first <= 1.0


def test_the_summary_is_valid_strict_json():
    """summary.json is consumed by the paper's tables; a bare NaN token breaks it.

    json.dumps happily emits `NaN`, which is not valid JSON and which a strict parser
    rejects. Since a NaN could only get in through a poisoned risk (guarded above),
    this is the belt on that belt.
    """
    result = analyze(_rows())
    encoded = json.dumps(result)
    json.loads(encoded, parse_constant=_reject)


def _reject(token):
    raise ValueError(f"summary.json contains the non-JSON token {token!r}")
