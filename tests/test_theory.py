"""The theory of docs/main.tex, pinned directly against the code (CPU).

tests/test_selective.py pins the *scores*: their conventions, their degeneracies,
their orientation. This module pins the *claims* they are derived from. Every test
below corresponds to a numbered result in the paper, and each exists because the
claim is load-bearing somewhere a bug would be silent:

* P1/P2 (rem:classical) -- the two properties that make Q_p the right posterior.
  If the level-set construction stopped being the comonotone coupling, every
  downstream expectation would still *compute*, just against the wrong law.
* prop:integral / eq:quad -- the exactness that licenses estimating an expectation
  over 2^N masks by a one-dimensional quadrature.
* prop:floor / obs:vshape -- the integrand's shape. Two facts about it are PROVABLE
  (Y_t contains Yhat_c iff t <= m_c; m_c >= the argmax floor phi; and H(1) = diag only
  when max_i p_ci < 1) and one regularity is only EMPIRICAL (H is V-shaped with its
  minimum NEAR m_c). Node placement is an OPEN empirical question: the minimax argument
  licenses nothing (rem:minimaxfallacy) and there is NO closed form for the vertex.
  An earlier version of this list cited a now-DEAD label, prop:vshape, and said the
  integrand's shape "is what makes the derived nodes derived rather than tuned" -- both
  the label and the claim are retracted.
* prop:sdc / rem:notcoupling -- the precise sense in which SDC is coupling-free,
  and the precise sense in which expected Dice is not. Conflating the two is the
  single most tempting error in this area, and the paper's own N=2 counterexample
  is pinned here so nobody re-conflates them.
* lem:perm -- "the discriminating claim". Findings 1 and 4 rest on it, and a
  refactor that made SDC boundary-aware, or reintroduced an area term into the
  band, would destroy the paper's central dichotomy with the rest of the suite
  green.
* eq:rank -- the reason a score may be negated, rescaled, or rank-averaged at all.

These are theory tests: where a claim is exact, they assert exactness.
"""

import itertools
import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from selectseg.selective import (
    _level_set_losses,
    aurc,
    bilevel_band_widths,
    bilevel_overlap,
    image_confidence_scores,
    importance_nodes,
    quadrature_nodes,
    quadrature_risks,
)

# ---------------------------------------------------------------------------
# fixtures: the probability maps the theory is stated over
# ---------------------------------------------------------------------------


def _binary_ramp(size=64):
    """A soft-boundary blob whose probabilities never hit 1/2 exactly.

    The exclusion matters for prop:floor(i), and for REFUTED CLAIM 4. ``torch.argmax``
    breaks a tie toward index 0 (background) while the closed level set {p >= 1/2}
    keeps it, so Yhat = {p > 1/2} STRICTLY while Y_{1/2} = {p >= 1/2}: a pixel at
    exactly p = 1/2 sits in Y_{1/2} and outside Yhat, and H(1/2) would be nonzero for a
    tie-breaking reason rather than a theoretical one (pinned in
    test_H_at_one_half_is_zero_only_when_no_pixel_ties_at_one_half). Real float maps do
    not attain 1/2 exactly; this fixture does not either.

    (This docstring used to cite prop:vshape, a DEAD label, which also implied the
    fixture was about the VERTEX. It is not -- it is about the strictness mismatch.)
    """
    axis = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
    return torch.clamp(0.98 - (radius - 9.0) / 15.0, 0.02, 0.98)


def _smooth_field(seed=7, size=48):
    """A smoothed random field: many distinct probability values, coherent mask.

    The setting the paper's quadrature numbers are measured on. Returned as numpy,
    because the exact-integral reference is computed over ``np.unique``.
    """
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.random((size, size)), 4.0)
    field = (field - field.min()) / (field.max() - field.min())
    return (0.02 + 0.96 * field).astype(np.float32)


def _softmax_blob(classes=21, size=64):
    """A C-way softmax map with one foreground class winning a blob."""
    logits = torch.zeros(classes, size, size)
    logits[0] = 1.0
    axis = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    logits[1] = 2.2 - radius / 8
    return logits.softmax(dim=0)


# ---------------------------------------------------------------------------
# the level-set posterior Q_p (def:levelset, rem:classical)
# ---------------------------------------------------------------------------


def _levelset_law(probabilities):
    """The exact law of Q_p on a tiny pixel set: (mask, mass) pairs.

    T ~ U(0,1) and Y^(T) = {i : p_i >= T}, so the level set only changes as T
    crosses one of the p_i. Sorting them descending, the mask {} carries the mass
    above the largest probability and the k-th nested mask carries the gap to the
    next one. Enumerating that is exact -- no sampling -- which is what lets the
    tests below assert equalities rather than tolerances.
    """
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(-values)
    sorted_values = values[order]
    law = [(frozenset(), 1.0 - sorted_values[0])]
    for position in range(len(sorted_values)):
        following = (
            sorted_values[position + 1]
            if position + 1 < len(sorted_values)
            else 0.0
        )
        law.append(
            (
                frozenset(order[: position + 1].tolist()),
                sorted_values[position] - following,
            )
        )
    return [(mask, mass) for mask, mass in law if mass > 0]


def _independent_law(probabilities):
    """The exact law of the independent coupling with the same marginals."""
    count = len(probabilities)
    law = []
    for bits in itertools.product([0, 1], repeat=count):
        mass = math.prod(
            probabilities[i] if bits[i] else 1 - probabilities[i]
            for i in range(count)
        )
        if mass > 0:
            law.append(
                (frozenset(i for i in range(count) if bits[i]), mass)
            )
    return law


def _dice(sample, predicted):
    """Dice(Y, Yhat) under the bounded convention rem:bounded: 0 on the empty Y.

    Without eq:bounded the expectation does not exist -- Q_p puts mass
    1 - max_i p_i on the empty mask -- so this convention is not cosmetic, it is
    what makes every expectation in this file finite.
    """
    if not sample:
        return 0.0
    return 2 * len(sample & predicted) / (len(sample) + len(predicted))


def test_p1_marginal_consistency_of_the_level_set_posterior():
    """P1: P_{Q_p}(Y_i = 1) = p_i, the coverage-function property.

    This is what makes Q_p a *posterior for these marginals* at all: the network
    says pixel i is foreground with probability p_i, and a joint law that did not
    reproduce that would be answering a different question. Checked two ways --
    exactly, on the enumerated law, and by Monte Carlo over T ~ U(0,1), which is
    the construction as the code actually samples it (a thresholding).
    """
    probabilities = [0.05, 0.31, 0.5, 0.72, 0.96]
    for index, probability in enumerate(probabilities):
        exact = sum(
            mass for mask, mass in _levelset_law(probabilities) if index in mask
        )
        assert exact == pytest.approx(probability, abs=1e-12)

    generator = torch.Generator().manual_seed(0)
    thresholds = torch.rand(200_000, generator=generator)
    probs = torch.tensor(probabilities)
    # Y^(T) = {i : p_i >= T}, one row per draw
    samples = probs[None, :] >= thresholds[:, None]
    empirical = samples.double().mean(dim=0)
    assert empirical.numpy() == pytest.approx(probabilities, abs=5e-3)


def test_p2_comonotone_joint_is_the_frechet_hoeffding_bound():
    """P2: P_{Q_p}(Y_i = 1, Y_j = 1) = min(p_i, p_j).

    The single global threshold makes the samples a NESTED chain, so two pixels
    co-fire exactly when the threshold clears the larger of the two probabilities
    -- the Frechet-Hoeffding upper bound, i.e. the maximally positively dependent
    coupling of the marginals. This is the whole reason the posterior produces a
    coherent contour displacement instead of salt-and-pepper speckle, and it is
    what a boundary loss needs in order to mean anything. An independent coupling
    with the same marginals gives p_i * p_j instead, and is measurably different.
    """
    probabilities = [0.05, 0.31, 0.5, 0.72, 0.96]
    law = _levelset_law(probabilities)
    for i, j in itertools.product(range(len(probabilities)), repeat=2):
        joint = sum(mass for mask, mass in law if i in mask and j in mask)
        assert joint == pytest.approx(
            min(probabilities[i], probabilities[j]), abs=1e-12
        )
    # nested samples: the chain is totally ordered by inclusion
    masks = sorted((mask for mask, _ in law), key=len)
    for smaller, larger in zip(masks, masks[1:]):
        assert smaller <= larger
    # and it is NOT the independent coupling, whose joint is the product
    independent = _independent_law(probabilities)
    product_joint = sum(mass for mask, mass in independent if 0 in mask and 4 in mask)
    assert product_joint == pytest.approx(0.05 * 0.96)
    assert product_joint != pytest.approx(min(0.05, 0.96))


# ---------------------------------------------------------------------------
# prop:sdc and rem:notcoupling
# ---------------------------------------------------------------------------


def _soft_dice_confidence(probabilities, predicted):
    """SDC (eq:sdc), straight from the formula, on a tiny pixel set."""
    return (
        2 * sum(probabilities[i] for i in predicted)
        / (sum(probabilities) + len(predicted))
    )


def _ratio_of_expectations(law, probabilities, predicted):
    """E[2 sum_i Y_i yhat_i] / E[sum_i Y_i + sum_i yhat_i] under a given law."""
    numerator = sum(mass * 2 * len(mask & predicted) for mask, mass in law)
    denominator = sum(
        mass * (len(mask) + len(predicted)) for mask, mass in law
    )
    return numerator / denominator


def test_prop_sdc_is_the_ratio_of_expectations_under_any_coupling():
    """prop:sdc: the ratio-of-expectations Dice equals SDC for ANY coupling.

    Numerator and denominator are both LINEAR in Y, so linearity of expectation
    collapses them onto the marginals and the higher-order structure of the joint
    never enters. Brute-forced here against the two couplings that actually differ
    -- the independent one and the comonotone Q_p -- on pixel sets small enough to
    enumerate all 2^N masks exactly.

    This is the honest part of SDC's derivation, and it is why SDC is a legitimate
    baseline rather than a straw man: it is not making a *wrong* choice of
    coupling, it is making NO choice. That is also precisely its limitation --
    see the next two tests.
    """
    predicted = frozenset({0, 2, 3})
    rng = np.random.default_rng(2)
    for _ in range(5):
        probabilities = list(rng.uniform(0.05, 0.95, 5))
        target = _soft_dice_confidence(probabilities, predicted)
        for law in (_independent_law(probabilities), _levelset_law(probabilities)):
            assert _ratio_of_expectations(law, probabilities, predicted) == (
                pytest.approx(target, abs=1e-12)
            )


def test_expected_dice_is_not_coupling_invariant_the_n2_counterexample():
    """rem:notcoupling: E_Q[Dice] DOES depend on the coupling. The paper's own case.

    It is tempting -- and wrong -- to read prop:sdc as "expected Dice is
    coupling-free". Expected Dice is the expectation of a RATIO, a nonlinear
    function of the two linear statistics, and the joint law of that *pair* very
    much depends on the coupling. The paper settles it with an exhaustive N = 2
    example, p = (1/2, 1/2), Yhat = {1}, and gets three different numbers:

        E_indep[Dice] = 5/12,   E_{Q_p}[Dice] = 1/3,   SDC = 1/2.

    Pinned here so the distinction cannot quietly rot. Note the ordering, too:
    SDC is not a bound on either expectation, it is a third quantity.
    """
    probabilities = [0.5, 0.5]
    predicted = frozenset({0})

    independent = sum(
        mass * _dice(mask, predicted)
        for mask, mass in _independent_law(probabilities)
    )
    comonotone = sum(
        mass * _dice(mask, predicted)
        for mask, mass in _levelset_law(probabilities)
    )
    sdc = _soft_dice_confidence(probabilities, predicted)

    assert independent == pytest.approx(5 / 12)
    assert comonotone == pytest.approx(1 / 3)
    assert sdc == pytest.approx(1 / 2)
    # three genuinely different values, and SDC is not between them by accident:
    # it is the ratio of expectations, not an expectation of a ratio.
    assert comonotone < independent < sdc
    # ...while the ratio-of-expectations IS the same under both (prop:sdc)
    for law in (_independent_law(probabilities), _levelset_law(probabilities)):
        assert _ratio_of_expectations(law, probabilities, predicted) == (
            pytest.approx(sdc)
        )


# ---------------------------------------------------------------------------
# prop:integral -- the threshold-integral identity and its quadrature
# ---------------------------------------------------------------------------


def _exact_threshold_integral(prob, predicted):
    """int_0^1 H(t) dt, computed EXACTLY by the piecewise-constant identity.

    H(t) = L(Y_t, Yhat) is piecewise constant in t, jumping only where t crosses
    one of the image's own probability values, so the integral is a finite sum:

        int_0^1 H = sum_k (edge_{k+1} - edge_k) H(midpoint of the bin)

    over the edges [0, p_(1), ..., p_(K), 1]. The edges are anchored at 0 and 1,
    not at the map's extremes: the top bin carries the atom Q_p puts on the EMPTY
    mask, weight 1 - max_i p_i, which under rem:bounded is worth a full image
    diagonal and is the single largest term for a diffuse map. Returns the HD95
    integral and the 1 - Dice integral, plus each integrand's total variation
    (needed for the Koksma bound).
    """
    values = np.unique(prob)
    edges = np.concatenate([[0.0], values, [1.0]])
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    gaps = np.diff(edges)
    distances, dice_losses, _ = _level_set_losses(prob, predicted, list(midpoints))
    distances, dice_losses = np.array(distances), np.array(dice_losses)
    return {
        "hd95": float(gaps @ distances),
        "dice": float(gaps @ dice_losses),
        "variation_hd95": float(np.abs(np.diff(distances)).sum()),
        "variation_dice": float(np.abs(np.diff(dice_losses)).sum()),
    }


def test_quadrature_converges_to_the_exact_threshold_integral():
    """prop:integral + eq:quad: r_M -> int_0^1 H(t) dt, the exact sum over the
    image's own probability values.

    E_{Q_p}[L(Y, Yhat)] is an expectation over 2^N masks. prop:integral says it is
    EXACTLY a one-dimensional integral, because a sample from Q_p is just a
    thresholding, and eq:quad estimates that integral with M level-set evaluations.
    This test closes the loop: the estimator the code ships must actually converge
    to the quantity the proposition says it equals. If it did not -- if, say, a
    node rule never reached the empty level set -- r_M would converge to something
    else entirely and every score would still look perfectly reasonable.

    The reference is the exact piecewise-constant sum, not a denser quadrature, so
    this is a test against the theory and not against ourselves.
    """
    prob = _smooth_field()
    predicted = prob >= 0.5
    exact = _exact_threshold_integral(prob, predicted)
    tensor, mask = torch.tensor(prob), torch.tensor(predicted)

    coarse = quadrature_risks(tensor, mask, quadrature_nodes("mid", 2))
    dense = quadrature_risks(tensor, mask, quadrature_nodes("mid", 256))
    for measure in ("hd95", "dice"):
        assert abs(dense[measure] - exact[measure]) < abs(
            coarse[measure] - exact[measure]
        ), measure
    # the dense rule is close in absolute terms, not merely closer
    assert dense["hd95"] == pytest.approx(exact["hd95"], abs=0.05)
    assert dense["dice"] == pytest.approx(exact["dice"], abs=1e-3)


def test_koksma_bound_holds_at_every_m():
    """eq:koksma: |r_M - int_0^1 H| <= V(H) / (2M) for the midpoint rule.

    CAUTION: this bounds the error, it does not PRESCRIBE the nodes. Minimizing the
    Koksma bound is not minimizing the error -- see
    test_minimizing_the_koksma_bound_does_not_minimize_the_error, which pins that
    non-sequitur. The O(1/M) RATE is sound; the node prescription drawn from it was not.

    The star-discrepancy of the M-point midpoint rule is 1/(2M), and H is of bounded
    variation -- because it is PIECEWISE CONSTANT (it jumps only where t crosses one of
    the image's own probability values) and BOUNDED by rem:bounded, which is all Koksma
    needs. So the quadrature error is bounded a priori, with no assumption of
    smoothness, which matters because HD95 is a percentile and H has a kink.

    This does NOT rest on obs:vshape. An earlier version of this docstring derived the
    BV hypothesis from "obs:vshape splits it at the vertex into two monotone halves" --
    i.e. from UNIMODALITY, which is REFUTED (H can fall, rise and fall again; see
    test_H_is_not_guaranteed_unimodal). If H is not unimodal the "two monotone halves"
    do not exist and that justification collapses. The theorem is fine; the reason given
    for it was not, and BV never required it. Note _exact_threshold_integral already
    computes V(H) the correct way, as the total variation of a piecewise-constant
    sampling -- the code was right while the docstring above it was wrong.

    Note what is and is NOT asserted. The BOUND is the theorem, and it shrinks like
    1/M. The error itself is *not* monotone in M and the test does not pretend it
    is: HD95 jumps as level sets gain and lose pixels, so a lucky coarse rule can
    beat an unlucky finer one on a single map. Asserting monotone decrease would be
    asserting something false and would eventually fail for a good reason, which is
    the worst kind of flake.
    """
    prob = _smooth_field()
    predicted = prob >= 0.5
    exact = _exact_threshold_integral(prob, predicted)
    tensor, mask = torch.tensor(prob), torch.tensor(predicted)
    for count in (2, 4, 8, 16, 32, 64, 128):
        risk = quadrature_risks(tensor, mask, quadrature_nodes("mid", count))
        for measure, variation in (
            ("hd95", exact["variation_hd95"]),
            ("dice", exact["variation_dice"]),
        ):
            error = abs(risk[measure] - exact[measure])
            assert error <= variation / (2 * count) + 1e-9, (measure, count, error)


def test_the_empty_mask_atom_is_worth_a_diagonal():
    """rem:bounded: Q_p puts mass 1 - max_i p_i on the EMPTY mask, worth diag.

    This is the term that makes E_{Q_p}[HD95] exist at all, and the term a diffuse
    map's risk is mostly *made of*. A rule whose nodes never exceed max_i p_i
    silently replaces it with H({p >= max p}) ~ 0 -- which is not an inaccuracy but
    an inconsistency, an error that does not vanish as M grows. Pinned as an
    identity: for a map whose probabilities all sit below a level, the exact
    integral above that level is exactly (1 - max p) * diagonal.
    """
    prob = np.full((16, 16), 0.3, dtype=np.float32)
    prob[4:12, 4:12] = 0.6
    predicted = prob >= 0.5
    diagonal = math.hypot(16, 16)
    exact = _exact_threshold_integral(prob, predicted)
    # H(t) = diag for every t > max p = 0.6, an interval of length 0.4
    above = _level_set_losses(prob, predicted, [0.8])[0][0]
    assert above == pytest.approx(diagonal)
    # and that atom alone accounts for 0.4 * diag of the exact integral
    assert exact["hd95"] >= 0.4 * diagonal
    assert exact["dice"] >= 0.4 * 1.0  # the Dice loss of an empty set is 1


# ---------------------------------------------------------------------------
# prop:floor / obs:vshape -- the integrand's shape
# ---------------------------------------------------------------------------


def test_h_at_one_half_is_zero_for_binary_maps_only():
    """prop:floor(i): for C = 2 and NO pixel tied at exactly 1/2, Y_{1/2} = Yhat, so
    H(1/2) = 0. This is NOT a statement about the vertex, and NOT an iff.

    CAUTION -- this test's name and docstring originally taught TWO claims that have
    since been REFUTED, and both are spelled out so they cannot creep back:

    * "The vertex of H sits at the argmax floor" (1/2 binary, 1/C softmax). It does
      not: phi is a LOWER BOUND on m_c, and a given map's winner need never descend to
      it. There is no closed form for the vertex. See
      test_the_argmax_floor_is_not_the_vertex.
    * "H(t*) = 0 iff Yhat_c is itself a level set, i.e. iff C = 2" (REFUTED CLAIM 6).
      False in BOTH directions. Forward: see below, C > 2 breaks it. Reverse: this
      suite's own C=3 fixture in test_the_argmax_floor_is_not_the_vertex has
      Y_{0.35} = Yhat_c exactly and H(0.35) = 0 with C = 3. And H = 0 does not even
      imply Y_t = Yhat_c, because HD95 is a 95th percentile that truncates a one-pixel
      disagreement away (test_hd95_zero_does_not_imply_set_equality).

    What survives, and what this test pins: for C = 2 with no ties, the level set at
    1/2 IS the prediction ({p >= 1/2} = Yhat), so H(1/2) = 0 exactly. Note the test
    evaluates t = 1/2, which is NOT the vertex in general -- for the C=21 fixture below
    the minimum sits near m_c ~ 0.05.

    It is FALSE for C > 2, and the paper says so only because this was measured rather
    than assumed. A C-way argmax winner needs just p >= 1/C, so {p >= 1/2} is a strict
    SUBSET of Yhat_c (it can even be empty, at which point H(1/2) is a whole diagonal).
    An earlier draft asserted H(1/2) = 0 unconditionally and was validated only on
    binary maps -- confirming a claim in exactly the one case where it holds.
    """
    foreground = _binary_ramp()
    assert not (foreground == 0.5).any()  # see _binary_ramp: no argmax ties
    probs = torch.stack([1 - foreground, foreground])
    predicted = (probs.argmax(dim=0) == 1).numpy()
    # for C = 2 the argmax mask and the half level set are the SAME set
    assert (predicted == (foreground.numpy() >= 0.5)).all()
    distances, dice_losses, _ = _level_set_losses(
        foreground.numpy(), predicted, [0.5]
    )
    assert distances[0] == 0.0
    assert dice_losses[0] == 0.0

    # C = 21: the argmax winner's floor is 1/21 = 0.048, so {p >= 1/2} is a strict
    # subset of Yhat (here: empty), and H(1/2) is a whole image diagonal.
    probs = _softmax_blob(classes=21)
    predicted = (probs.argmax(dim=0) == 1).numpy()
    foreground = probs[1].numpy()
    assert predicted.sum() > 0
    assert (foreground >= 0.5).sum() < predicted.sum()  # strict subset
    distances, _, _ = _level_set_losses(foreground, predicted, [0.5])
    assert distances[0] > 0.0
    assert distances[0] == pytest.approx(math.hypot(64, 64))


def test_the_integrand_grows_toward_both_tails():
    """obs:vshape -- H is EMPIRICALLY V-shaped on realistic maps: small near m_c, large
    at 0 and 1. AN OBSERVATION, NOT A THEOREM.

    rem:vshapefails exhibits an H that falls, rises and falls again (27.3, 4.0, 19.8,
    0.0 on a map whose prediction is a single pixel), so unimodality is NOT guaranteed
    and there is not always "the" vertex. This test therefore MEASURES the regularity on
    one realistic binary field; it does not assert it in general. See
    test_H_is_not_guaranteed_unimodal for the counterexample, pinned.

    What the observation buys is the explanation of the two failed ablations
    (rank_anchored_threshold, importance_nodes): each pulls a node toward the
    probability mass, which is toward the minimum, where the integrand contributes
    least, while the integral's mass lives in the TAILS.

    It does NOT settle node placement. An earlier version of this docstring said the
    V-shape "is the fact that makes the deployed nodes (alpha, 1 - alpha) = (0.1, 0.9)
    the right ones" -- that is RETRACTED (rem:minimaxfallacy: "we therefore withdraw any
    claim that node placement is settled by theory"). Node placement is an OPEN
    empirical question, and (0.1, 0.9) is retro-justified as lying in the tails, not
    derived as optimal. Indeed rem:straddle suspects (0.1, 0.9) of being badly placed on
    multi-class maps, where the minimum sits near 0.06 and both nodes fall on one side.
    """
    prob = _smooth_field()
    predicted = prob >= 0.5
    heights = {
        t: _level_set_losses(prob, predicted, [t])[0][0]
        for t in (0.02, 0.25, 0.5, 0.75, 0.98)
    }
    # binary, untied map: m_c = 1/2, so here (and only here) the minimum is at 1/2 and
    # H vanishes there -- prop:floor(i), not a general fact about the vertex.
    assert heights[0.5] == 0.0
    # strictly increasing away from the vertex, in both directions
    assert heights[0.5] < heights[0.25] < heights[0.02]
    assert heights[0.5] < heights[0.75] < heights[0.98]
    # ...while the probability values concentrate in the middle, which is the trap
    assert 0.25 < float(np.median(prob)) < 0.75


# ---------------------------------------------------------------------------
# lem:perm -- the discriminating claim
# ---------------------------------------------------------------------------


def _joint_permutation(probs, seed=0):
    """Relabel the pixels of every channel with one common permutation.

    sigma acts JOINTLY on (p, yhat) -- the same permutation on every class -- so
    the argmax mask of the permuted map is the permutation of the argmax mask.
    Every area, every marginal, every sum over pixels is preserved exactly; only
    the Euclidean positions of the surface pixels change.
    """
    classes, height, width = probs.shape
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(height * width, generator=generator)
    return probs.reshape(classes, -1)[:, order].reshape(classes, height, width)


# SDC and the pixel baselines are sums of float32 over the pixels, and float
# addition is not associative, so relabelling the pixels changes the accumulation
# ORDER and can move the last bit or two. The functional is exactly invariant; its
# float32 realization is invariant to ~1e-7. The area readouts are exactly
# invariant because they are ratios of integer counts.
_FLOAT_SUM_TOLERANCE = 1e-6


def test_lem_perm_area_and_sdc_are_invariant_the_distance_band_is_not():
    """lem:perm: the claim Findings 1 and 4 rest on, and the suite's only test of it.

    Every quantity SDC and the area readouts are built from -- sum_i p_i,
    sum_i yhat_i, sum_i p_i yhat_i, |Y_lo|, |Y_hi| -- is a SUM OVER PIXELS, and a
    sum over pixels cannot tell where the pixels are. So a joint relabelling of the
    pixel indices leaves them all unchanged, while the band's HD95 collapses,
    because scattering the pixels destroys the coherent contour whose displacement
    it measures. A permutation-invariant functional is constant on the orbit of
    (p, yhat), and that orbit contains both a coherent mask and a scattered one
    with the same boundary CARDINALITY and arbitrarily different boundary GEOMETRY
    -- hence no permutation-invariant score can resolve boundary geometry, no
    matter how it is engineered.

    This is a structural claim about the score class, so the test is structural:
    a refactor that made SDC boundary-aware, or that reintroduced an area term
    into the band, would break the paper's central dichotomy while leaving every
    other test in this repository green.
    """
    foreground = _binary_ramp()
    probs = torch.stack([1 - foreground, foreground])
    permuted = _joint_permutation(probs)

    original = image_confidence_scores(probs, [0.1, 0.3])
    scrambled = image_confidence_scores(permuted, [0.1, 0.3])

    # EXACTLY invariant: ratios of integer pixel counts.
    for key in (
        "levelset_iou_pmean@0.1",
        "levelset_dice_pmean@0.1",
        "levelset_iou_pmean@0.3",
        "levelset_dice_pmean@0.3",
        # E_{Q_p}[Dice] itself -- the area arm of the quadrature, at every rule
        "neg_rM32_dice_pmean@mid",
        "neg_rM2_dice_pmean@gl",
    ):
        assert scrambled[key] == original[key], key

    # invariant as a functional; float32 summation order moves the last bits
    for key in ("sdc_pmean", "sdc_pmin"):
        assert scrambled[key] == pytest.approx(
            original[key], abs=_FLOAT_SUM_TOLERANCE
        ), key

    # NOT invariant: the distance readouts, which is the entire point
    for key in (
        "neg_band_width_pmean_sym@0.1",
        "neg_mband_width_pmean_sym@0.1",
        "neg_qband_width_pmean_sym@0.1",
        "neg_r2_pmean_sym@0.1",
        "neg_rM32_pmean@mid",
        "neg_band_per_boundary@0.1",
    ):
        assert abs(scrambled[key] - original[key]) > 1.0, key


def test_lem_perm_at_the_readout_level():
    """lem:perm again, one level down, where the two readouts are built.

    The score-level test above could in principle be satisfied by an aggregation
    accident. This pins the primitives themselves: bilevel_overlap is a ratio of
    two integer areas and is bit-for-bit invariant; bilevel_band_widths reads a
    Euclidean distance transform and is not. Same map, same alpha, same two level
    sets -- the ONLY difference between them is that one asks *how many* and the
    other asks *how far*.
    """
    foreground = _binary_ramp()
    probs = torch.stack([1 - foreground, foreground])
    scrambled = _joint_permutation(probs)[1]

    assert bilevel_overlap(scrambled, 0.3) == bilevel_overlap(foreground, 0.3)

    original_width = bilevel_band_widths(foreground, 0.3)
    permuted_width = bilevel_band_widths(scrambled, 0.3)
    assert original_width["symmetric"] != pytest.approx(
        permuted_width["symmetric"]
    )

    # and the same split inside one call: quadrature_risks' dice arm is invariant,
    # its hd95 arm is not, on identical nodes.
    nodes = quadrature_nodes("mid", 32)
    original_risk = quadrature_risks(
        foreground, torch.stack([1 - foreground, foreground]).argmax(0) == 1, nodes
    )
    permuted_probs = _joint_permutation(probs)
    permuted_risk = quadrature_risks(
        scrambled, permuted_probs.argmax(0) == 1, nodes
    )
    assert permuted_risk["dice"] == pytest.approx(original_risk["dice"], abs=1e-9)
    assert abs(permuted_risk["hd95"] - original_risk["hd95"]) > 1.0


# ---------------------------------------------------------------------------
# eq:rank -- AURC sees a score only through its ranking
# ---------------------------------------------------------------------------


def test_aurc_depends_on_the_score_only_through_its_ranking():
    """eq:rank: AURC is invariant under ANY strictly increasing transform of g.

    This is the licence for a great deal of what the paper does: negating a
    distance so that higher means more confident, comparing a width in pixels
    against a dimensionless Dice, rank-averaging two incommensurable scores, and
    reading every ablation as a rank correlation. If AURC depended on the score's
    scale, none of those would be legitimate -- and the failure would be silent,
    because the numbers would still come out.

    Also pinned: a strictly DECREASING transform must CHANGE it. That is the direct
    guard on a sign error, which in this codebase is not hypothetical -- an
    accidental sign flip is exactly what makes a detection failure the most
    confident image of the split.
    """
    rng = np.random.default_rng(1)
    confidences = list(rng.normal(size=50))
    risks = list(rng.uniform(size=50))
    baseline = aurc(confidences, risks)

    for transform in (
        lambda x: math.exp(3 * x),
        lambda x: 2 * x - 7,
        lambda x: math.atan(x),
        lambda x: x ** 3,  # strictly increasing on all of R
        lambda x: math.log(x + 10),  # the confidences sit well above -10
    ):
        assert aurc([transform(c) for c in confidences], risks) == pytest.approx(
            baseline
        )

    # a strictly decreasing transform reverses the ranking and must not agree
    assert aurc([-c for c in confidences], risks) != pytest.approx(baseline)


# ===========================================================================
# REFUTED CLAIMS, PINNED AS FALSE.
#
# Each of the following was asserted in a draft of docs/main.tex, believed, and then
# refuted by an independent adversarial check. Every one had the same shape: a clean
# derivation, asserted before it was tested. Twice the disconfirming evidence was
# already in hand and was explained away. They are pinned here as FALSE so that no one
# -- including a future draft of this paper -- re-derives them.
# ===========================================================================



# The probability-unit baselines: the ONLY scores that do not claim the flooring
# convention. Everything else in the flat dict does. Kept in sync with the identical set
# in tests/test_selective.py by
# test_the_two_suites_agree_on_which_scores_are_baselines, so the two cannot drift.
_PIXEL_BASELINES = frozenset({
    "mean_max_prob",
    "p05_max_prob",
    "mean_margin",
    "mean_collision_prob",
    "mmmc",
    "neg_gen",
    "neg_mean_entropy",
    "neg_fg_entropy",
    "neg_boundary_entropy",
    "neg_interior_entropy",
    "neg_low_conf_fraction",
    "neg_tail_uncertainty",
    "worst_patch_max_prob",
})


def _claims_the_flooring_convention(key):
    """Which scores must floor a total detection failure?

    The metric-aligned ones -- the band, r_2, the quadrature ladder, the area readouts
    and SDC -- because they estimate the expected value of a segmentation metric, and a
    prediction of nothing has the worst possible value of one. The probability-unit
    BASELINES (mean max softmax, entropy, margin, ...) do NOT: to them a confidently
    empty map is a confident map. That is not a defect to patch but the paper's own
    criticism of them, and it is pinned in
    test_the_BASELINES_do_not_floor_a_detection_failure_and_that_is_the_point.

    Excluded too: the ``diag_`` diagnostics (not scores), and the ``_skip`` / ``_lo``
    families (documented-broken ablation components, pinned separately).

    THIS PREDICATE USED TO BE A CASE-SENSITIVE ALLOWLIST, and it went stale. Its only
    ladder clause was ``key.startswith("neg_rM")``, so when the candidate node rules
    shipped as ``neg_rmid2_*`` / ``neg_rvtx2_*`` (lowercase 'r', no capital M) they
    matched NOTHING and were silently skipped -- 4 of the 174 keys, unchecked, while the
    caller's docstring claimed to loop over "EVERY key". An allowlist fails open; that
    is the same enumerate-N-of-M failure that let the original orientation inversion
    ship. It is now an EXCLUSION list: anything not explicitly excluded is checked, so a
    newly added score is covered the day it is added.
    """
    if key.startswith("diag_") or "_skip" in key or "_lo@" in key:
        return False
    return key not in _PIXEL_BASELINES


def test_the_exact_identity_pairs_each_level_with_the_gap_BELOW_it():
    """REFUTED: the paper wrote sum_k (v_{k+1} - v_k) H(v_k) -- the gap ABOVE.

    Since Y_t = {i : p_i >= t} uses a NON-STRICT inequality, Y_t is constant on the
    half-open interval (v_{k-1}, v_k] and equals Y_{v_k} there. Each level therefore
    carries the gap BELOW it. The wrong pairing is not a rounding error: on codex's
    knots it gives 0.566 against a true 1.697.
    """
    # The off-by-one's real damage is that it DROPS THE EMPTY-MASK ATOM: it assigns the
    # top weight (1 - v_m) to H(v_m) instead of to L(empty, Yhat) = diag. So it is
    # invisible on a map whose probabilities reach 1 (tiny atom) and on a smooth map
    # with thousands of near-equal gaps -- which is exactly why it survived review. Here
    # the map maxes out at 0.5, so the atom carries HALF the integral's weight.
    prob = np.full((24, 24), 0.1)
    prob[4:20, 4:20] = 0.3
    prob[8:16, 8:16] = 0.5
    tensor, mask = torch.tensor(prob), torch.tensor(prob >= 0.5)
    diagonal = math.hypot(*prob.shape)

    def height(t):
        risk = quadrature_risks(tensor, mask, [float(t)])
        return risk["hd95"] if risk is not None else diagonal

    values = np.unique(prob)
    below, previous = 0.0, 0.0
    for value in values:                       # each level carries the gap BELOW it
        below += (value - previous) * height(value)
        previous = value
    below += (1.0 - values[-1]) * diagonal     # the empty-mask atom

    above = 0.0                                # the ORIGINAL, WRONG pairing
    for value, nxt in zip(values, np.append(values[1:], 1.0)):
        above += (nxt - value) * height(value)

    reference = float(np.mean([height(t) for t in (np.arange(4000) + 0.5) / 4000]))
    assert below == pytest.approx(reference, rel=0.02)
    assert abs(above - reference) / reference > 0.50   # the old form drops the atom


def test_h_at_one_is_the_diagonal_only_when_the_map_does_not_saturate():
    """REFUTED: "H(1) = L(empty, Yhat) = diag" was asserted unconditionally.

    Y_1 = {p >= 1} is NON-empty whenever the map attains 1.0 exactly -- which a
    confident float32 softmax routinely does. Then H(1) is not the diagonal at all.
    """
    diagonal = math.hypot(20, 20)

    unsaturated = torch.full((20, 20), 0.3, dtype=torch.float64)
    unsaturated[5:15, 5:15] = 0.99
    risk = quadrature_risks(unsaturated, unsaturated >= 0.5, [1.0])
    assert risk is None  # empty level set -> the bounded convention applies

    saturated = torch.full((20, 20), 0.3, dtype=torch.float64)
    saturated[5:15, 5:15] = 1.0
    assert (saturated >= 1.0).any()  # Y_1 is NOT empty
    risk = quadrature_risks(saturated, saturated >= 0.5, [1.0])
    assert risk is not None
    assert risk["hd95"] == pytest.approx(0.0)
    assert risk["hd95"] != pytest.approx(diagonal)


def test_hd95_zero_does_not_imply_set_equality():
    """THE ROOT CAUSE OF FOUR SEPARATE ERRORS IN THIS PAPER.

    HD95 is a 95th PERCENTILE, not a metric. It does not obey the triangle inequality
    and HD95(A, B) = 0 does NOT imply A = B: a small disagreement is truncated away by
    the percentile. This single fact broke (1) Prop 5's band-to-r_2 bridge, (2) the
    claim "H(t*) = 0 iff C = 2", (3) the strictness of the V-shape, and (4) the reading
    of a zero band width as demonstrated certainty.
    """
    a = np.zeros((40, 40), dtype=bool)
    a[10:30, 10:30] = True
    b = a.copy()
    b[10, 10] = False
    assert not np.array_equal(a, b)

    def surface(mask):
        return mask & ~ndimage.binary_erosion(mask)

    sa, sb = surface(a), surface(b)
    pooled = np.hstack([
        ndimage.distance_transform_edt(~sb)[sa],
        ndimage.distance_transform_edt(~sa)[sb],
    ])
    assert float(np.percentile(pooled, 95)) == pytest.approx(0.0)


def test_the_argmax_floor_is_not_the_vertex_and_H_can_vanish_for_C_gt_2():
    """TWO REFUTED CLAIMS, on one fixture.

    (5) "the vertex of H is the argmax floor (1/C softmax, 1/2 CLIPSeg)". The floor is
        a LOWER BOUND on the winning probability -- a map's winner need never descend to
        it. Here a 3-way softmax whose winner bottoms out at 0.40 has H(1/3) > 0 while
        H(0.35) = 0.

    (6) "H(t*) = 0 iff C = 2", in its REVERSE direction. This same C=3 map has
        Y_{0.35} = Yhat_c EXACTLY and H(0.35) = 0, so C = 2 is not necessary for a
        vanishing minimum. (The forward direction is handled in
        test_h_at_one_half_is_zero_for_binary_maps_only.) The evidence was already
        sitting in this fixture, unlabelled, while the iff was taught 350 lines above.

    What IS provable is only that m_c := min in-mask probability satisfies m_c >= floor,
    and that Y_t contains Yhat iff t <= m_c. The vertex has NO closed form.
    """
    probs = np.empty((3, 11, 11))
    probs[0], probs[1], probs[2] = 0.34, 0.50, 0.16
    central = np.zeros((11, 11), dtype=bool)
    central[4:7, 4:7] = True
    probs[0, central], probs[1, central], probs[2, central] = 0.40, 0.35, 0.25
    assert np.allclose(probs.sum(axis=0), 1)
    assert probs[0][probs.argmax(axis=0) == 0].min() == pytest.approx(0.40)

    prediction = torch.tensor(probs.argmax(axis=0) == 0)
    channel = torch.tensor(probs[0])
    at_floor = quadrature_risks(channel, prediction, [1 / 3])["hd95"]
    above_floor = quadrature_risks(channel, prediction, [0.35])["hd95"]
    assert at_floor > 0.0
    assert above_floor == pytest.approx(0.0)

    # ...and this REFUTES "H(t*) = 0 iff C = 2" in the reverse direction: C is 3, yet
    # the level set at 0.35 is exactly the prediction and H vanishes there.
    assert probs.shape[0] == 3
    assert np.array_equal(probs[0] >= 0.35, prediction.numpy())


def test_H_is_not_guaranteed_unimodal():
    """REFUTED: "H is unimodal" / "H grows monotonically toward BOTH tails".

    obs:vshape is an OBSERVATION, not a theorem, and rem:vshapefails gives the
    counterexample: H can fall, RISE, and fall again. On a map whose prediction is a
    single pixel, H = 27.3, 4.0, 19.8, 0.0 at t = 0.01, 0.10, 0.30, 0.45.

    Two live docstrings used to assert unimodality as fact, and one of them USED IT AS A
    PREMISE -- deriving the bounded-variation hypothesis of eq:koksma from "split H at
    the vertex into two monotone halves". That derivation is unsound. The BV property
    Koksma actually needs comes from H being PIECEWISE CONSTANT (finitely many distinct
    probability values) and BOUNDED (rem:bounded), neither of which mentions the shape.
    So: no theorem here may assume a single vertex, two monotone halves, or a minimum
    that is also a global one.
    """
    height, width = 31, 31
    p_class = np.full((height, width), 0.05)
    row, col = np.ogrid[:height, :width]
    disk = (row - 15) ** 2 + (col - 6) ** 2 <= 4**2
    p_class[disk] = 0.20        # many low-distance boundary samples
    p_class[15, 28] = 0.40      # one remote, higher-confidence false component
    p_class[15, 6] = 0.90       # the sole argmax-positive pixel
    probs = np.stack([1 - p_class, p_class])
    assert np.allclose(probs.sum(axis=0), 1)
    predicted = probs.argmax(axis=0) == 1
    assert predicted.sum() == 1  # the prediction really is a single pixel

    channel, mask = torch.tensor(p_class), torch.tensor(predicted)
    heights = [
        quadrature_risks(channel, mask, [t])["hd95"]
        for t in (0.01, 0.10, 0.30, 0.45)
    ]

    # fall, RISE, fall: no single vertex splits this into two monotone halves
    assert heights[1] < heights[0]        # falls
    assert heights[2] > heights[1]        # then RISES -- unimodality is dead here
    assert heights[3] < heights[2]        # then falls again
    assert heights[3] == pytest.approx(0.0)
    # two distinct local minima, so "the" vertex does not exist on this map
    assert sum(
        heights[i] < heights[i - 1] and (i == 3 or heights[i] < heights[i + 1])
        for i in range(1, 4)
    ) == 2


def test_minimizing_the_koksma_bound_does_not_minimize_the_error():
    """REFUTED INFERENCE: "the midpoint rule minimizes the star-discrepancy, hence the
    Koksma bound, hence it is the right node set."

    A node set with a strictly WORSE bound can achieve a strictly BETTER error.
    Minimizing a worst-case bound says nothing about the error on the integrand one
    actually has. THE SAME FALLACY APPEARED TWICE: it also underpinned a
    sqrt(V_L / V_R) node-allocation rule that provably minimizes the same bound and
    measurably fails to reduce the error.
    """
    def integrand(t):            # a step: int_0^1 = 1.5
        return 0.0 if t < 0.5 else 3.0

    def estimate(nodes):
        return sum(integrand(t) for t in nodes) / len(nodes)

    def star_discrepancy(nodes):
        nodes = sorted(nodes)
        n = len(nodes)
        return max(
            max(abs((i + 1) / n - x), abs(i / n - x)) for i, x in enumerate(nodes)
        )

    midpoint = [0.25, 0.75]
    coarser = [0.6, 0.9]
    assert star_discrepancy(coarser) > star_discrepancy(midpoint)   # WORSE bound
    assert abs(estimate(midpoint) - 1.5) == pytest.approx(0.0)
    assert abs(estimate(coarser) - 1.5) == pytest.approx(1.5)       # and worse error
    # ...but the ORDER of the bound does not determine the ORDER of the error:
    exact_but_lopsided = [0.4, 0.5]     # estimate = 1.5, exactly right
    assert star_discrepancy(exact_but_lopsided) > star_discrepancy(midpoint)
    assert abs(estimate(exact_but_lopsided) - 1.5) == pytest.approx(0.0)


def test_detection_failure_is_least_confident_under_EVERY_score():
    """THE INVARIANT THAT WOULD HAVE CAUGHT THE HISTORICAL BUG.

    An image with no predicted foreground class is a total detection failure -- true
    risk 0.55-0.84 against dataset means of 0.04-0.40 -- and must be the LEAST confident
    image under EVERY score. Distance scores are negated, so aggregating an empty
    present-class list to 0.0 put these images at the score's MAXIMUM. That inverted the
    paper's headline (band-vs-SDC on overlap: 2/8 -> 6/8 once fixed).

    _claims_the_flooring_convention is an EXCLUSION list, so every key that is not a
    named baseline, a diagnostic, or a documented-broken family is checked -- a newly
    added score is covered the day it is added. It used to be an ALLOWLIST keyed on
    "neg_rM", and the four candidate-node scores (neg_rmid2_*, neg_rvtx2_*) fell straight
    through it: 4 keys silently unchecked while this docstring claimed to loop over
    "EVERY key". The ``checked`` counter below is the guard against that recurring --
    a predicate that quietly stops matching now fails loudly instead of passing
    vacuously.
    """
    failure = torch.zeros(2, 40, 40)
    failure[0], failure[1] = 0.98, 0.02
    sharp = torch.zeros(2, 40, 40)
    sharp[0], sharp[1] = 0.02, 0.98
    sharp[0, :10, :], sharp[1, :10, :] = 0.98, 0.02
    fuzzy = torch.full((40, 40), 0.55)

    bad = image_confidence_scores(failure, [0.1])
    assert bad["diag_no_present_class"] == 1.0
    checked = 0
    for other in (sharp, torch.stack([1 - fuzzy, fuzzy])):
        good = image_confidence_scores(other, [0.1])
        checked = 0
        for key, value in bad.items():
            if not _claims_the_flooring_convention(key):
                continue
            checked += 1
            assert value <= good[key] + 1e-9, key
    # the candidate node rules are among the keys checked -- they were not, before
    for key in ("neg_rmid2_pmean", "neg_rvtx2_pmean",
                "neg_rmid2_dice_pmean", "neg_rvtx2_dice_pmean"):
        assert _claims_the_flooring_convention(key), key
    # non-vacuity: 174 keys, minus 13 baselines, 5 diag_, 10 _skip and 4 _lo@
    assert checked > 55, f"the predicate went vacuous: only {checked} keys checked"


def test_the_BASELINES_do_not_floor_a_detection_failure_and_that_is_the_point():
    """A NEGATIVE CLAIM, pinned -- and it is one of the paper's arguments.

    The flooring convention binds the scores that are aligned to a segmentation metric.
    The probability-unit BASELINES neither claim it nor could satisfy it: a confidently
    -empty prediction is, to mean-max-softmax, a confident image. aMSP reads 0.98 on a
    total detection failure against 0.55 on a merely fuzzy one -- it is dominated by the
    confident background, which is precisely the structural criticism the paper makes of
    the whole aggregate-a-per-pixel-uncertainty family.

    So this is not a bug to fix. It is pinned because it explains why those baselines
    collapse on the detection-failure-heavy conditions (CLIPSeg/VOC: 53-67% of images
    carry a hallucinated or missed class).
    """
    failure = torch.zeros(2, 40, 40)
    failure[0], failure[1] = 0.98, 0.02
    fuzzy = torch.full((40, 40), 0.55)
    bad = image_confidence_scores(failure, [0.1])
    fuzz = image_confidence_scores(torch.stack([1 - fuzzy, fuzzy]), [0.1])

    # the baselines rank the catastrophic failure ABOVE the merely uncertain image
    assert bad["mean_max_prob"] > fuzz["mean_max_prob"]
    assert bad["neg_mean_entropy"] > fuzz["neg_mean_entropy"]
    # while every metric-aligned score of ours floors it
    for key in ("neg_band_width_pmean_sym@0.1", "neg_r2_pmean_sym@0.1",
                "sdc_pmean", "levelset_dice_pmean@0.1"):
        assert bad[key] <= fuzz[key] + 1e-9, key


def test_the_skip_and_lo_families_are_anti_informative_and_are_pinned_as_such():
    """PINNED FAILURES: 14 keys in the flat dict violate sharp-beats-fuzzy.

    ``hd95_skip`` drops empty level sets and renormalizes, so a maximally-uncertain
    CONSTANT map -- whose surviving level sets all coincide with its own argmax --
    scores a *zero* skip-risk, i.e. PERFECT confidence. The ``_lo`` keys are components
    of r_2, not standalone scores. Both ship in the flat dict and are ranked by the
    analysis as if they were candidate scores. They are ablations and diagnostics; this
    test exists so that none is promoted to the default without someone noticing.
    """
    sharp = torch.zeros(32, 32)
    sharp[8:24, 8:24] = 1.0
    fuzzy = torch.full((32, 32), 0.55)
    s = image_confidence_scores(torch.stack([1 - sharp, sharp]), [0.1])
    f = image_confidence_scores(torch.stack([1 - fuzzy, fuzzy]), [0.1])

    broken = [k for k in s if not k.startswith("diag_") and not s[k] > f[k]]
    assert broken, "the known-broken families vanished -- update this test"
    for key in broken:
        assert "_skip" in key or "_lo@" in key, f"a NEW score is anti-informative: {key}"


def test_H_at_one_half_is_zero_only_when_no_pixel_ties_at_one_half():
    """REFUTED PRECONDITION: "for C = 2, Y_{1/2} = Yhat, so H(1/2) = 0" -- but only if
    nothing ties at exactly 1/2.

    The two sets are defined with inequalities of DIFFERENT STRICTNESS. torch.argmax
    breaks ties toward the lowest index, which is the background, so Yhat = {p > 1/2}
    STRICTLY -- while the level set Y_{1/2} = {p >= 1/2} includes the tied pixels. With
    121 pixels at exactly 0.5 we get |Yhat| = 9 against |Y_{1/2}| = 121 and
    H(1/2) = 5.66, not 0.

    This was the one case believed airtight. It is not, and the gap is a tie-breaking
    convention, which is precisely the kind of detail a proof sketch skips.
    """
    tied = torch.full((11, 11), 0.5, dtype=torch.float64)
    tied[4:7, 4:7] = 0.9
    prediction = torch.stack([1 - tied, tied]).argmax(dim=0) == 1
    assert int(prediction.sum()) == 9              # ties went to BACKGROUND
    assert int((tied >= 0.5).sum()) == 121         # the level set kept them
    assert quadrature_risks(tied, prediction, [0.5])["hd95"] > 1.0

    untied = torch.full((11, 11), 0.3, dtype=torch.float64)
    untied[4:7, 4:7] = 0.9
    prediction = torch.stack([1 - untied, untied]).argmax(dim=0) == 1
    assert quadrature_risks(untied, prediction, [0.5])["hd95"] == pytest.approx(0.0)


def _faint_present_class(size=48, classes=21):
    """A softmax map with one confident object and one HALLUCINATED class.

    The hallucinated class wins the argmax on a whole block at p ~ 1/C = 0.048 -- below
    alpha = 0.1, and below mid2's lowest node 0.25 -- so its aggressive set and its
    mid2 level set are both EMPTY. This is the map every dropped-present-class bug
    fires on, and the one the M-ladder contamination was found with.
    """
    logits = torch.zeros(classes, size, size)
    logits[0] = 4.0                                  # confident background
    logits[3, 10:30, 10:30] = 8.0                    # a confident object
    logits[:, 36:44, 36:44] = 0.0                    # ...and a near-uniform corner,
    logits[7, 36:44, 36:44] = 0.001                  # where class 7 wins by a hair
    return torch.softmax(logits, dim=0)


def test_the_candidate_node_rules_saturate_a_present_class_rather_than_dropping_it():
    """REGRESSION: the M-ladder's contamination, committed a SECOND time in the P1
    experiment that is supposed to settle node placement.

    The candidate node rules (mid2, vtx2) used to filter ``if risk is not None`` instead
    of routing through _present_readouts, so a class that WINS the argmax while its
    lowest-node level set is empty was DELETED from the present-class mean rather than
    saturated at the diagonal (rem:bounded). It fired ASYMMETRICALLY between the two
    rules being compared -- mid2's lowest node is 0.25, so a class winning at p ~ 1/21 =
    0.048 was dropped; vtx2's lowest node is 0.5 * m_c <= m_c, so by prop:floor(i) its
    level set always contains the prediction and never empties. The two rules were
    therefore averaged over DIFFERENT CLASS SETS on the same image: exactly the defect
    of rem:mladdercontam, in the experiment meant to replace the contaminated one.

    Measured before the fix: neg_rmid2_pmean = -0.0 -- the score's MAXIMUM, perfect
    confidence -- on an image containing a wholly hallucinated object.
    """
    probs = _faint_present_class()
    prediction = probs.argmax(dim=0)
    present = sorted(set(prediction.unique().tolist()) - {0})
    assert present == [3, 7], present
    # class 7 is present in the argmax but its whole channel sits below alpha and below
    # mid2's lowest node: this is the class that used to vanish
    assert float(probs[7].max()) < 0.1
    assert float(probs[7][prediction == 7].min()) > 1 / 21 - 1e-3   # prop:floor(ii)

    scores = image_confidence_scores(probs, [0.1])
    diagonal = math.hypot(48, 48)

    # class 3 reads ~0, class 7 saturates at the diagonal -> the mean is diag/2. Never 0.
    assert scores["neg_rmid2_pmean"] == pytest.approx(-diagonal / 2, abs=1e-6)
    assert scores["neg_rmid2_dice_pmean"] == pytest.approx(-0.5, abs=1e-9)
    assert scores["neg_rmid2_pmean"] < -1.0, "the hallucinated class was DROPPED again"

    # vtx2 never had the bug (its lowest node cannot empty), and must stay unchanged
    assert scores["neg_rvtx2_pmean"] < -1.0

    # and both candidate rules must now sit in the same range as every correctly
    # aggregated estimator on this image -- that is what "same class set" means
    reference = scores["neg_rM32_pmean@mid"]
    for key in ("neg_rmid2_pmean", "neg_rvtx2_pmean", "neg_rM2_pmean@gl"):
        assert abs(scores[key] - reference) < 0.25 * diagonal, key


def test_which_constructions_can_drop_a_present_class_and_which_cannot():
    """WHY THE DROPPED-CLASS BUG WAS CONFINED TO SOFTMAX MULTI-CLASS.

    The bug: a class can WIN the argmax while its aggressive set {p_c >= alpha} is
    EMPTY, and the aggregation dropped it instead of saturating it at the diagonal.
    That requires the argmax floor to sit BELOW alpha, and whether it does is a
    property of the probability construction, not of the code:

      softmax, C = 21 : a winner needs only p_c >= 1/C = 0.048  <  alpha = 0.1  -> CAN fire
      softmax, C = 2  : a winner needs p_c >= 1/2               >  alpha       -> cannot
      CLIPSeg         : bg := 1 - max_c p_c, so a foreground class wins only if
                        p_c > 1 - p_c, i.e. p_c > 1/2           >  alpha       -> cannot

    This is what bounds the blast radius to the two DeepLabV3/VOC conditions, and it is
    why the two LARGEST dense-vs-M=2 gains (both on CLIPSeg/VOC) are structurally clean.
    Pinned because a contaminated result was reported before this was understood.
    """
    generator = torch.Generator().manual_seed(17)
    alpha = 0.1

    # softmax C=21: an argmax winner CAN sit below alpha
    reachable = False
    for _ in range(200):
        probs = torch.softmax(torch.randn(21, 16, 16, generator=generator) * 0.3, dim=0)
        prediction = probs.argmax(dim=0)
        for c in range(1, 21):
            mask = prediction == c
            if mask.any() and float(probs[c].max()) < alpha:
                reachable = True
    assert reachable, "a C=21 softmax argmax winner below alpha should be reachable"

    # CLIPSeg: a winning foreground class ALWAYS exceeds 1/2, hence always exceeds alpha
    for _ in range(100):
        logits = torch.randn(21, 16, 16, generator=generator) * 2
        foreground = torch.sigmoid(logits[1:])
        probs = torch.cat(
            [1 - foreground.max(dim=0, keepdim=True).values, foreground], dim=0
        )
        prediction = probs.argmax(dim=0)
        for c in range(1, 21):
            mask = prediction == c
            if mask.any():
                assert float(probs[c][mask].min()) > 0.5 > alpha

    # binary softmax: likewise
    for _ in range(100):
        probs = torch.softmax(torch.randn(2, 16, 16, generator=generator) * 2, dim=0)
        prediction = probs.argmax(dim=0)
        mask = prediction == 1
        if mask.any():
            assert float(probs[1][mask].min()) > 0.5 > alpha


def test_the_two_suites_agree_on_which_scores_are_baselines():
    """The flooring predicate is an EXCLUSION list in both suites, so the excluded set
    is the only thing standing between a new score and the invariant. If the two lists
    drift, one suite starts checking a score the other exempts -- and an allowlist that
    silently stops matching is precisely how the candidate node rules shipped unchecked.
    """
    from tests.test_selective import _PIXEL_BASELINES as OTHER

    assert _PIXEL_BASELINES == OTHER


def test_the_importance_rule_does_not_guarantee_it_reaches_the_empty_level_set():
    """REFUTED IMPLEMENTATION CLAIM: importance_nodes' anchored edges were said to put
    "node_0 below min p and node_{M-1} above max p, so both tails and the empty-mask
    atom are actually evaluated".

    They are bin MIDPOINTS, not bounds. node_{M-1} = (q_{M-1} + 1)/2 exceeds max p only
    when q_{M-1} > 2 max p - 1, which fails on any confident map. So the top bin still
    carries the atom's WEIGHT (1 - q_{M-1}, which contains the atom's mass 1 - max p)
    while its node evaluates a NON-EMPTY level set -- the diagonal that atom is worth is
    still not charged. The rule remains a valid quadrature (the weights telescope to 1)
    and it is a FAILED ABLATION either way; what is pinned here is that the guarantee
    does not hold, so nobody re-derives a consistency argument from it.
    """
    prob = np.full((32, 32), 0.1)
    prob[8:24, 8:24] = 0.9
    for count in (2, 4, 8):                       # the deployed importance rules
        nodes, weights = importance_nodes(prob, count)
        assert sum(weights) == pytest.approx(1.0)   # still a valid quadrature
        assert nodes[-1] <= prob.max(), count       # ...but the top node does NOT clear
        assert nodes[0] >= prob.min(), count        # ...nor the bottom one undercut
        # Y_t = {p >= t} is NON-STRICT, so even node == max p leaves a non-empty set
        assert (prob >= nodes[-1]).any(), count

    # a SATURATING map can never reach the empty level set at all, at any M
    saturated = np.full((32, 32), 0.3)
    saturated[8:24, 8:24] = 1.0
    for count in (2, 8, 64):
        nodes, _ = importance_nodes(saturated, count)
        assert (saturated >= nodes[-1]).any(), count
        assert 1 - saturated.max() == 0.0        # the atom's MASS degrades to zero...
    # ...while H(1) = 0, not diag: prop:floor(iii)
    tensor = torch.tensor(saturated)
    risk = quadrature_risks(tensor, tensor >= 0.5, [1.0])
    assert risk is not None
    assert risk["hd95"] == pytest.approx(0.0)
