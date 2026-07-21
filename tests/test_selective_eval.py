"""The per-image JSONL writer: quality metrics and posterior diagnostics (CPU).

This module had ZERO coverage, and that is not incidental -- it is why a defect in
the one number the diagnostic exists to produce could ship green. Everything here
runs on tiny synthetic tensors in milliseconds; there was never a cost reason.

The two diagnostics are the paper's only measurements of its own posterior
ASSUMPTIONS, which is the one term of the risk decomposition no amount of
quadrature can reduce:

* ``levelset_auc`` tests OUR assumption (the truth is a superlevel set of p, so p
  ranks pixels correctly by foregroundness -- AUC is 1 iff that holds exactly).
* ``residual_moran_i`` tests SDC's (conditional independence implies the residual
  Y* - p is spatially uncorrelated).

Both see the ground truth, so neither may ever be ranked as a confidence score --
which tests/test_analyze_selective.py pins from the other side.
"""

import math

import numpy as np
import pytest
import torch

from selectseg.data import SPECS
from selectseg.selective_eval import image_quality, posterior_assumption_diagnostics

IGNORE = 255


def _binary_spec():
    return SPECS["pet"]  # C = 2: background + pet


def test_levelset_auc_is_one_for_a_perfect_separator_and_half_for_a_tie():
    """levelset_auc = 1 iff the probability map ranks every foreground pixel above
    every background pixel -- the exact content of the level-set assumption.

    The tie case is the one that matters in practice and the one a naive
    implementation gets wrong. Probability maps are full of plateaus (saturated 0s
    and 1s), so the ranks MUST be tie-averaged: breaking ties by array position
    would score an all-tied map as AUC 0 or 1 -- a perfect or perfectly-inverted
    ranking -- when the truthful answer is that it ranks nothing at all, 0.5.
    """
    spec = _binary_spec()
    mask = torch.zeros(16, 16, dtype=torch.long)
    mask[4:12, 4:12] = 1

    perfect = torch.zeros(2, 16, 16)
    perfect[1] = 0.1
    perfect[1][mask == 1] = 0.9
    perfect[0] = 1 - perfect[1]
    assert posterior_assumption_diagnostics(perfect, mask, spec)["levelset_auc"] == (
        pytest.approx(1.0)
    )

    inverted = perfect.flip(0)
    assert posterior_assumption_diagnostics(inverted, mask, spec)["levelset_auc"] == (
        pytest.approx(0.0)
    )

    tied = torch.full((2, 16, 16), 0.5)
    assert posterior_assumption_diagnostics(tied, mask, spec)["levelset_auc"] == (
        pytest.approx(0.5)
    )


def test_levelset_auc_is_none_when_no_class_has_both_labels():
    """AUC needs both classes present; an all-background image has no positives."""
    spec = _binary_spec()
    mask = torch.zeros(16, 16, dtype=torch.long)
    probs = torch.full((2, 16, 16), 0.5)
    assert posterior_assumption_diagnostics(probs, mask, spec) == {
        "levelset_auc": None,
        "residual_moran_i": None,
    }


def test_morans_i_separates_a_coherent_residual_from_an_iid_one():
    """Moran's I is ~0 under spatial independence and strongly positive under the
    coherence real segmentation residuals have. That gap is the whole diagnostic.

    SDC's derivation assumes conditional independence given the input, which implies
    the residual Y* - p is spatially uncorrelated. Segmentation masks are manifestly
    not like that -- their errors come in contiguous patches along boundaries -- and
    this is the number that says so without hand-waving.
    """
    spec = _binary_spec()
    size = 40
    mask = torch.zeros(size, size, dtype=torch.long)
    mask[8:32, 8:32] = 1

    # a COHERENT residual: the model is uniformly wrong about the whole object
    coherent = torch.zeros(2, size, size)
    coherent[1] = 0.5
    coherent[0] = 0.5
    high = posterior_assumption_diagnostics(coherent, mask, spec)["residual_moran_i"]
    assert high > 0.8

    # an IID residual: the same marginals, scattered at random
    generator = torch.Generator().manual_seed(0)
    noisy = torch.zeros(2, size, size)
    noisy[1] = torch.rand(size, size, generator=generator)
    noisy[0] = 1 - noisy[1]
    scattered = mask.clone()
    scattered[:] = (torch.rand(size, size, generator=generator) < 0.5).long()
    low = posterior_assumption_diagnostics(noisy, scattered, spec)["residual_moran_i"]
    assert abs(low) < 0.15


def test_morans_i_excludes_ignored_pixels():
    """The IGNORE ring must not enter the residual -- it is not a false positive.

    ``(mask == index) & valid`` only zeroes the TRUTH at an ignored pixel; it leaves
    the pixel in the array with its probability intact, so it enters the statistic as
    a fabricated (0 - p) residual. That is not scattered noise: VOC's void label and
    Pet's trimap border form a CONTIGUOUS RING around every object boundary --
    exactly where p is most uncertain and the fabricated residual is largest -- so
    they inject a large coherent structure into the very statistic that measures the
    residual's coherence. The error grows with model quality, because a sharp model's
    genuine residual is ~0 on the valid pixels and the fiction dominates what is
    left.

    Constructed here so the answer is known: on the valid pixels the model is exactly
    right, so the true residual is identically zero, Moran's I is undefined
    (zero variance) and must be reported as None -- not as some number manufactured
    out of the void ring.
    """
    spec = _binary_spec()
    size = 32
    truth = torch.zeros(size, size, dtype=torch.long)
    truth[8:24, 8:24] = 1

    probs = torch.zeros(2, size, size)
    probs[1] = truth.float()          # exactly right, everywhere
    probs[0] = 1 - probs[1]

    # a 1-pixel void ring around the object, where p is high but the label is IGNORE
    ringed = truth.clone()
    ringed[7:25, 7:25] = IGNORE
    ringed[8:24, 8:24] = 1
    void = ringed == IGNORE
    assert void.sum() == 18 * 18 - 16 * 16  # a contiguous ring, as in VOC

    probs_ringed = probs.clone()
    # ONLY the void pixels move: p is high there, but the label is IGNORE, so a
    # residual of (0 - 0.9) at every one of them is pure fabrication.
    probs_ringed[1][void] = 0.9
    probs_ringed[0] = 1 - probs_ringed[1]

    result = posterior_assumption_diagnostics(probs_ringed, ringed, spec)
    # the residual is identically 0 on every VALID pixel, so there is no variance to
    # correlate and the honest answer is None. Counting the void ring would instead
    # report a large, entirely fabricated, spatial correlation.
    assert result["residual_moran_i"] is None
    # ...and the AUC, which was already valid-restricted, agrees the model is perfect
    assert result["levelset_auc"] == pytest.approx(1.0)


def test_image_quality_penalized_hd95_is_never_none():
    """image_hd95_penalized must be a number on EVERY image, including empty-vs-empty.

    It exists precisely so that the boundary risk is defined on the same image set as
    the overlap risks -- ``image_hd95`` follows the medical convention of scoring only
    the classes present on both sides, which costs a detection failure nothing and
    deletes the worst images from the curve. A ``None`` here defeats its own purpose,
    and worse: numpy coerces it to a silent NaN, every AURC under that risk becomes
    NaN, and the band-vs-SDC bootstrap (whose NaN comparisons are all False, so the
    poisoned resamples count as losses) reports a plausible but WRONG number rather
    than announcing itself.

    An image with no foreground on either side has no boundary to get wrong -- it
    scores mIoU = mDice = 1.0 -- so its penalized boundary error is 0.0.
    """
    spec = _binary_spec()
    empty = torch.zeros(16, 16, dtype=torch.long)
    quality = image_quality(empty, empty, spec)
    assert quality["image_miou"] == pytest.approx(1.0)
    assert quality["image_mdice"] == pytest.approx(1.0)
    assert quality["image_hd95"] is None          # undefined: the medical convention
    assert quality["image_hd95_penalized"] == 0.0  # defined: zero boundary error


def test_image_quality_penalized_hd95_charges_a_detection_failure_the_diagonal():
    """A wholly missed class is the worst boundary error the image admits.

    This is the counterpart of the test above and the reason the penalized convention
    exists at all: HD95 is undefined for a class present on only one side, and the
    unpenalized average simply DROPS it -- charging a detection failure nothing.
    """
    spec = _binary_spec()
    truth = torch.zeros(24, 24, dtype=torch.long)
    truth[8:16, 8:16] = 1
    missed = torch.zeros(24, 24, dtype=torch.long)  # predicted nothing

    quality = image_quality(missed, truth, spec)
    assert quality["image_hd95"] is None  # undefined, and silently dropped...
    assert quality["image_hd95_penalized"] == pytest.approx(math.hypot(24, 24))


def test_image_quality_is_perfect_on_an_exact_prediction():
    spec = _binary_spec()
    truth = torch.zeros(24, 24, dtype=torch.long)
    truth[8:16, 8:16] = 1
    quality = image_quality(truth.clone(), truth, spec)
    assert quality["image_miou"] == pytest.approx(1.0)
    assert quality["image_mdice"] == pytest.approx(1.0)
    assert quality["image_hd95"] == pytest.approx(0.0)
    assert quality["image_hd95_penalized"] == pytest.approx(0.0)


def test_multiclass_diagnostics_average_over_the_classes_in_the_ground_truth():
    """Both diagnostics are means over the foreground classes actually present.

    A 21-way VOC mask usually contains one or two objects; averaging over all 20
    foreground classes would drown the signal in classes the image does not contain.
    """
    spec = SPECS["voc"]
    mask = torch.zeros(24, 24, dtype=torch.long)
    mask[2:10, 2:10] = 3
    mask[14:22, 14:22] = 7

    probs = torch.full((spec.num_classes, 24, 24), 0.01)
    probs[0] = 0.9
    for index in (3, 7):
        probs[index][mask == index] = 0.95
    result = posterior_assumption_diagnostics(probs, mask, spec)
    # both present classes are perfectly separated, so the mean AUC is 1
    assert result["levelset_auc"] == pytest.approx(1.0)
    assert result["residual_moran_i"] is not None
    assert -1.0 <= result["residual_moran_i"] <= 1.0
