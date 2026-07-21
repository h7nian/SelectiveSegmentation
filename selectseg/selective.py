"""Image-level confidence scores for selective segmentation.

Implements the bi-level set distance-field proposal (METHODS.md §6) together
with the single-pass baselines it is compared against, plus the AURC metric
for risk–coverage evaluation. All confidence scores are oriented so that
higher means more confident.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def bilevel_band_widths(prob, alpha, conservative_threshold=None,
                        conservative_mask=None):
    """95th-percentile widths, in pixels, of one class's probability band.

    ``prob`` is the (H, W) probability map of a single foreground class.
    The band lies between the aggressive level set {p >= alpha} and the
    conservative one {p >= t_hi}, where ``conservative_threshold`` t_hi
    defaults to the constant 1 - alpha. Passing the per-image threshold of
    :func:`rank_anchored_threshold` instead gives the rank-anchored band: the
    two variants share this one code path, and therefore every convention
    below, differing only in where the inner contour is drawn.

    ``conservative_mask``, when given, is intersected into the conservative set
    (Y_hi = {p >= t_hi} & mask). It is only ever passed the class's own argmax
    mask, and only on the rank-anchored path, where an order-statistic t_hi can
    drop below the argmax's own floor and {p >= t_hi} would otherwise spill
    outside the class's prediction — see :func:`rank_anchored_threshold`. With
    the constant t_hi = 1 - alpha > 1/2 the intersection is a no-op on any
    normalized map, and it is *not* applied there, so the constant-threshold
    scores keep their own convention exactly.

    Three robust Hausdorff-style widths are returned: ``outward`` reads the
    distance field of the conservative surface on the aggressive surface
    (catches fog bulges and detached mid-confidence blobs), ``inward`` reads
    the opposite direction (catches erosion of the confident core, but is blind
    to detached blobs), and ``symmetric`` pools both reading sets before the
    percentile — the bidirectional analogue matching the evaluation metric's
    convention.

    Returns ``None`` when the aggressive set is empty (class confidently
    absent). Returns the image diagonal for all variants when the
    conservative set is empty while the aggressive one is not (a band with
    no inner anchor — maximal uncertainty).

    Note what a width of *zero* means here, since the score is negated and zero
    is therefore its maximum: the two contours coincide. Under the constant
    threshold that can only happen if p jumps from below alpha to above
    1 - alpha, i.e. genuine near-binary certainty. Under a clipped t_hi it can
    also happen because the map is *flat* — the two level sets then coincide
    with no confidence having been demonstrated at all. That is why
    :func:`rank_anchored_threshold` refuses to return a threshold that collapses
    the band, rather than this function guarding on Y_hi == Y_lo, which would
    wrongly punish the genuinely sharp map.
    """
    prob = prob.numpy()
    if conservative_threshold is None:
        conservative_threshold = 1 - alpha
    aggressive = prob >= alpha
    if not aggressive.any():
        return None
    conservative = prob >= conservative_threshold
    if conservative_mask is not None:
        conservative = conservative & conservative_mask.numpy()
    if not conservative.any():
        diagonal = math.hypot(*prob.shape)
        return {"outward": diagonal, "inward": diagonal, "symmetric": diagonal}
    aggressive_surface = _surface(aggressive)
    conservative_surface = _surface(conservative)
    outward = ndimage.distance_transform_edt(~conservative_surface)[
        aggressive_surface
    ]
    inward = ndimage.distance_transform_edt(~aggressive_surface)[
        conservative_surface
    ]
    return {
        "outward": float(np.percentile(outward, 95)),
        "inward": float(np.percentile(inward, 95)),
        "symmetric": float(np.percentile(np.hstack([outward, inward]), 95)),
    }


def _rank_anchored_percentile(prob, predicted, alpha, q):
    """min(1 - alpha, P_q(in-mask p)) — the raw clip, numpy in, float out."""
    values = prob[predicted]
    if values.size == 0:
        return None
    return float(min(1 - alpha, np.percentile(values, 100 * q)))


def _clip_collapses(prob, predicted, alpha, threshold):
    """True when a clipped conservative threshold carries no information.

    Two ways the clip can collapse, both of which make a *negated distance*
    score read its maximum (zero width, i.e. perfect confidence) on a map that
    has demonstrated no confidence at all:

    * no pixel lies in [alpha, t_hi) — the conservative set then swallows the
      whole aggressive set, the two contours coincide, and every band width is
      0. This fires on any mask that is flat at any level >= alpha, and it also
      covers the inverted case t_hi <= alpha (the interval is empty), where
      Y_hi would be a strict *superset* of Y_lo and the returned "width" would
      not be a band width at all. t_hi <= alpha is reachable: an argmax winner
      of a C-way softmax needs only p >= 1/C (0.048 for VOC's C=21), well below
      the alphas in use.
    * no in-mask pixel lies below t_hi — the conservative set is then the whole
      predicted mask, so the r_2 conservative term HD95(Y_hi, Yhat) is 0. The
      clip has again been placed where the map's own distribution ends, and the
      estimator stops probing the part of [0, 1] the map never reaches, which is
      exactly the part carrying the uncertainty.

    Both are refusals to bet against the model's own peak, the same pathology
    :func:`quadrature_risks` documents for ``hd95_skip`` (a constant-0.55 map
    scoring a *zero* skip-risk). The caller reverts to the constant threshold,
    which either yields a real band or saturates honestly at the diagonal.
    """
    if not ((prob >= alpha) & (prob < threshold)).any():
        return True
    return not (predicted & (prob < threshold)).any()


def rank_anchored_threshold(prob, predicted, alpha, q=0.95):
    """Conservative threshold clipped to an order statistic of the class's mask.

    AN ABLATION, NOT THE DEPLOYED SCORE. Read the caveats below before using it.

    The constant conservative threshold 1 - alpha is a bet on each image's own
    upper tail: {p >= 1 - alpha} is non-empty only if max_i p_i >= 1 - alpha, a
    condition a global alpha has no reason to secure and that diffuse
    (zero-shot, per-class sigmoid) maps rarely reach. When it fails the band has
    no inner anchor and :func:`bilevel_band_widths` saturates at the image
    diagonal, so the score carries no ranking information on the images that
    fire it — 99.6% of CLIPSeg-general Pet at alpha=0.01, 14.7% at alpha=0.1.

    Anchoring the threshold to a rank inside the class's own predicted mask
    ``predicted`` (its argmax mask Yhat) removes the bet::

        t_hi = min(1 - alpha, P_q({p_i : i in Yhat}))       (q = 0.95)

    the conservative set being Y_hi = {p >= t_hi} & Yhat. Two guards apply, and
    both are load-bearing:

    * **Nesting.** Y_hi ⊆ Yhat is guaranteed by the level set alone only for
      C = 2, where every in-mask probability exceeds 1/2 and so does their q-th
      percentile. It fails both ways for C > 2. Under a *normalized* (softmax)
      map the argmax winner's floor is 1/C, so in-mask probabilities — and t_hi
      with them — routinely sit below 1/2, and a pixel another class wins can
      still clear t_hi; the constant threshold, needing two classes above 0.7 on
      a simplex, could never do this. Under a *sigmoid* map (CLIPSeg, background
      = 1 - max foreground) in-mask p does exceed 1/2, but {p >= 1/2} is not the
      argmax mask, because several prompts can clear 1/2 at one pixel and only
      the max wins. Y_hi is therefore intersected with Yhat whenever the clip is
      active. Left un-intersected, a leaked Y_hi can be *twice the size of the
      prediction* and report a width of 0 — maximal confidence — on an image the
      constant threshold rates at the diagonal.
    * **Collapse.** If the clipped band would carry no information at all
      (:func:`_clip_collapses`: a flat mask, or t_hi <= alpha), the clip is
      abandoned and the constant threshold 1 - alpha returned. The rank-anchored
      score is then bit-for-bit the constant one on that class, which is the
      honest outcome: it either has a real band or it saturates.

    After the guards, alpha < t_hi <= 1 - alpha always, and Y_hi is non-empty
    and nested whenever the clip is active.

    WHAT THIS DOES *NOT* BUY, contrary to what an earlier version of this
    docstring and of the paper claimed:

    * It is **not** confined to the degenerate images, and is therefore **not** a
      strict information gain. The clip fires whenever P_q(in-mask) < 1 - alpha,
      which strictly contains the saturated set {max p < 1 - alpha} and in
      practice is several times larger. On every clipped-but-healthy class it
      *reparameterizes* a score that was working. The variants are identical
      exactly where the clip is inactive or reverted, which ``diag_clipped@``
      counts — not where ``diag_saturated@`` is 0.
    * When the clip *is* active, {p >= t_hi} & Yhat is by definition the top
      1 - q of the mask **by area** (measured: |Y_hi| / |Yhat| = 0.050 +- 0.0004
      whenever the clip fires). A percentile of the in-mask values *is* an area
      quantile of the mask; there is no daylight between an order-statistic
      anchor and the top-k|Yhat| rule. The band width then reads the distance
      from the skirt {p >= alpha} to a fixed-area core, whose inner term is pure
      object geometry, and a synthetic study finds the ranking degrades in every
      regime tested (Kendall tau against true risk falling ~0.10-0.15) even as
      the ties vanish. This is the collapse into a shape statistic the design
      set out to avoid, and it is why these scores ship as an ablation.

    Returns ``None`` when ``predicted`` is empty: an absent class has no mask to
    take the order statistic of, and so no rank-anchored band.
    """
    prob = prob.numpy()
    predicted = predicted.numpy()
    threshold = _rank_anchored_percentile(prob, predicted, alpha, q)
    if threshold is None:
        return None
    if threshold < 1 - alpha and _clip_collapses(prob, predicted, alpha, threshold):
        return 1 - alpha
    return threshold


def bilevel_overlap(prob, alpha):
    """IoU and Dice agreement between one class's two nested level sets.

    The aggressive set {p >= alpha} contains the conservative one
    {p >= 1 - alpha}, so the overlap reduces to an area ratio: high overlap
    = a thin probability band (confident), low overlap = a wide one. This is
    the region-based counterpart to :func:`bilevel_band_widths`, which
    measures the band's spatial extent instead; both share the bi-level
    frame and differ only in how band discrepancy is quantified.

    Returns ``None`` when the class is absent (aggressive set empty); 0 for
    both measures when the conservative set is empty while the aggressive
    one is not (maximal uncertainty).
    """
    prob = prob.numpy()
    aggressive = int((prob >= alpha).sum())
    if aggressive == 0:
        return None
    conservative = int((prob >= 1 - alpha).sum())
    return {
        "iou": conservative / aggressive,
        "dice": 2 * conservative / (conservative + aggressive),
    }


# Nodes of the two-point Gauss-Legendre rule mapped to (0, 1): 1/2 -+ 1/(2 sqrt 3).
_GAUSS_LEGENDRE_2 = (0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3))

# Quadrature rules for the M-point estimator, as (rule, M) pairs. ("mid", 32)
# is the dense reference the cheap rules are ranked against.
QUADRATURE_RULES = (("gl", 2), ("mid", 4), ("mid", 8), ("mid", 16), ("mid", 32))
DENSE_RULE = ("mid", 32)


def quadrature_nodes(rule, count):
    """Thresholds t_1 < ... < t_M of a quadrature rule on the unit interval.

    ``gl`` is the two-point Gauss-Legendre rule, whose nodes integrate cubics
    exactly; ``mid`` is the M-point midpoint rule, whose nodes (m - 1/2)/M are
    equispaced and, unlike a closed rule, never degenerate at the endpoints.
    Both carry equal weights 1/M, so callers can leave the weights unset.

    Neither endpoint is a safe node. t = 0 makes every pixel foreground. t = 1 is
    the empty level set only on a map with max_i p_ci < 1; on a float32 softmax that
    has SATURATED to exactly 1.0 -- which a confident network routinely does --
    Y_1 = {p >= 1} is the whole saturated core, and H(1) is NOT the diagonal
    (prop:floor(iii): measured 0.000 against diag = 28.28). An earlier version of this
    docstring justified avoiding t = 1 by claiming the level set there is "almost
    none"; that premise is REFUTED, and saturation makes the endpoint a worse node
    rather than a better one -- H collapses toward 0 instead of saturating honestly.
    The conclusion (avoid both endpoints) survives; the reason given for it did not.
    """
    if rule == "gl":
        if count != 2:
            raise ValueError("the Gauss-Legendre rule is only tabulated for M=2")
        return list(_GAUSS_LEGENDRE_2)
    if rule == "mid":
        return [(index + 0.5) / count for index in range(count)]
    raise ValueError(f"unknown quadrature rule {rule!r}")


def importance_nodes(prob, count):
    """Nodes at the map's own probability quantiles, with the matching weights.

    A FAILED ABLATION, KEPT AS A CAUTIONARY ONE, AND EVERY PARAGRAPH BELOW IS
    WRITTEN AS SUCH. It is worse than the equispaced midpoint rule at every M
    below 16, and the reason is worth stating, because it is the same reason
    :func:`rank_anchored_threshold` failed: both pull a node *toward* the
    probability mass, and the probability mass is exactly where the integrand is
    smallest.

    WHY IT FAILS.  On realistic maps the integrand H(t) = L(Y_t, Yhat_c) is
    EMPIRICALLY V-shaped in t (obs:vshape -- an OBSERVATION, NOT A THEOREM): it falls
    to a minimum NEAR m_c := min{p_ci : i in Yhat_c}, the smallest probability the map
    attains inside its own prediction, then rises again toward both tails. That is
    enough to sink this ablation: the probability VALUES concentrate near the same
    place the integrand is smallest, so quantile nodes land where H has nothing to
    contribute, while the integral's mass lives in the TAILS.

    FOUR THINGS AN EARLIER VERSION OF THIS DOCSTRING CLAIMED ARE FALSE AND STAND
    RETRACTED (rem:vshapefails). They are spelled out rather than deleted, because
    the repo's failure mode is a clean derivation propagating unchallenged:

    * The vertex is NOT the argmax floor phi (1/C for a softmax; 1/2 for a binary map
      and for CLIPSeg's bg = 1 - max_c p_c). prop:floor gives only m_c >= phi: phi is
      a LOWER BOUND on m_c, not the location of the minimum. On a 3-way softmax whose
      winner never descends below 0.40, H(1/3) = 5.66 while H(0.35) = 0. There is NO
      CLOSED FORM for the vertex t*; m_c is the best cheap ESTIMATE, and it is an
      estimate.
    * H(t*) = 0 is NOT equivalent to C = 2. It fails both ways: a C=3 map with
      Y_{1/3} = Yhat_c has H(1/3) = 0, and a BINARY map with pixels tied at exactly
      1/2 has H(1/2) > 0, because argmax breaks ties toward background so
      Yhat = {p > 1/2} STRICTLY while Y_{1/2} = {p >= 1/2} keeps the ties
      (|Yhat| = 9 against |Y_{1/2}| = 121, H(1/2) = 5.66).
    * H is NOT unimodal. It can fall, rise and fall again: 27.3, 4.0, 19.8, 0.0 at
      t = 0.01, 0.10, 0.30, 0.45 on a map whose prediction is a single pixel. So
      "grows monotonically toward both tails" is not available as a premise.
    * H(1) is NOT always the diagonal. That holds only when max_i p_ci < 1. A
      confident float32 softmax saturates to exactly 1.0, and then Y_1 = {p >= 1} is
      NON-empty and H(1) = 0, not diag (prop:floor(iii)).

    None of the four is needed here. The one fact the ablation's failure rests on --
    the minimum sits where the probability mass is -- is empirical, and it is enough:
    on one binary map the M=2 quantile nodes land at 0.359 and 0.597, hugging the
    minimum, while the midpoint rule's 0.25 and 0.75 sit where the mass is. The gap
    weights are correct -- the estimand is untouched, which is what distinguishes this
    failure from :func:`rank_anchored_threshold`'s -- but the nodes are
    unrepresentative of the bins they stand for, and the estimate collapses.

    THE ARGUMENT THAT MOTIVATED THIS, AND WHY IT IS WRONG.  H is piecewise constant
    in t, jumping only where t crosses one of the image's own probability values, so
    the reasoning went: equispaced nodes waste evaluations on the plateaus (where
    Y_t does not move at all) and undersample the ranges where the probability
    values are dense (where it moves fastest); place the nodes at the quantiles of
    the map's own histogram and weight them by the gaps they span, and one gets
    textbook importance sampling with g = the density of the probability values,

        t_m = F^{-1}((m - 1/2) / M),   w_m proportional to 1 / f(t_m).

    The step that is true: substituting u = F(t) turns dt into du / f(t), so the
    *gap* between consecutive edges in probability space is exactly the weight that
    leaves the estimand unchanged. The exact identity is eq:orderstat, and its
    INDEXING is the whole content of it:

        int_0^1 H(t) dt = sum_{k=1..m} (v_k - v_{k-1}) H(Y_{v_k}) + (1 - v_m) * diag,

    with v_1 < ... < v_m the DISTINCT probability values and v_0 := 0. Because
    Y_t = {p >= t} is NON-STRICT, Y_t is constant on (v_{k-1}, v_k] and equals
    Y_{v_k} there, so each level carries the gap BELOW it, NOT above; and the trailing
    term is the atom Q_p places on the EMPTY mask, of mass 1 - max_i p_i and worth diag
    (rem:atom, rem:bounded).

    An earlier version of this docstring wrote this as
    sum_k (p_(k+1) - p_(k)) H(p_(k)) -- each level paired with the gap ABOVE it, and
    no atom term. That form is REFUTED. It is not a rounding error: it understates the
    integral badly (0.566 against a true 1.697 on the paper's worked example), because
    it charges the top gap 1 - max p at H({p >= max p}) ~ 0 instead of at the diagonal.
    It is pinned as false in
    tests/test_theory.py::test_the_exact_identity_pairs_each_level_with_the_gap_BELOW_it.

    THE WEIGHTS REALLY ARE LOAD-BEARING: moving the nodes and keeping equal
    weights would silently estimate a different functional -- one whose marginals
    are the image's own empirical CDF F(p_i) rather than p_i -- breaking the
    calibration the level-set posterior rests on. Adapting the *nodes* is safe
    because every image still estimates the same integral, so the cross-image
    ranking (all AURC depends on) survives; adapting the *estimand* is not.

    The step that is false: "where the values are dense" is not "where H moves
    fastest" in any sense that matters to the integral. H's *total variation* is
    concentrated at the vertex, but its *magnitude* is concentrated at the tails,
    and quadrature error is driven by magnitude. The right importance density is
    proportional to |H(t)|, which is tail-heavy: one should oversample AWAY from
    where the probability mass sits, not toward it. The equispaced rule already
    does the safe thing, and simply raising M is the effective fix. This also
    retro-justifies the constant threshold's nodes (alpha, 1 - alpha) = (0.1, 0.9):
    they are in the tails, where H is large.

    IMPLEMENTATION NOTE (the bug this used to have).  Both the bin edges *and* the
    representative nodes are built from the anchored edges [0, q_1, ..., q_{M-1}, 1],
    node m being the midpoint of bin m. An earlier version anchored only the edges
    and took the nodes from ``np.quantile(unique(p), ...)``, whose largest node is
    at most max_i p_i -- so the top bin carried the weight 1 - max_i p_i that Q_p
    puts on the EMPTY mask while its node never reached the empty level set, and
    the diagonal that atom is worth was silently replaced by H({p >= max p}) ~ 0.
    That made the rule *inconsistent* (the error plateaued instead of vanishing as
    M grew) and scored a two-valued map at a perfectly-confident risk of exactly 0.

    Taking the nodes from the same ANCHORED edges is what makes the weights telescope
    to 1 and gives the top bin the weight 1 - q_{M-1}, which CONTAINS the atom's mass
    1 - max_i p_i; and it at least ALLOWS the outer nodes to escape [min p, max p] as
    M grows, which the old edges made impossible. It does NOT guarantee that they do
    so at any given M, and an earlier version of this note claimed it did. That claim
    is FALSE and is retracted: node_0 = q_1/2 and node_{M-1} = (q_{M-1} + 1)/2 are bin
    MIDPOINTS, not bounds. Measured on the two-valued 0.1/0.9 map, node_{M-1} is 0.75
    at M=2, 0.85 at M=4 and exactly 0.90 = max p at M=8 -- and since Y_t = {p >= t} is
    NON-STRICT, even that last one has a non-empty level set. The diagonal is first
    charged at M=16. On a realistic 21-way softmax channel the top node stays below
    max p at every M up to 64. So the top bin still charges the atom's weight at a
    non-empty level set, and the residual inconsistency the note above describes is
    reduced, not eliminated. (Only splitting the top bin AT max_i p_i, and charging
    [max p, 1] the diagonal with weight 1 - max p, would eliminate it -- that is
    exactly eq:orderstat's final term. Forcing the top NODE above max p would not: it
    would charge the whole bin weight 1 - q_{M-1} at the diagonal, which overshoots.)

    Nothing here may assume H(1) = diag. On a map that SATURATES to exactly 1.0 --
    which float32 softmax routinely does -- {p >= t} is non-empty for every t <= 1, so
    NO node reaches the empty level set, H(1) = 0 rather than diag (prop:floor(iii)),
    and the atom's mass 1 - max_i p_i is simply 0. The mass degrades gracefully; the
    endpoint VALUE does not.

    The ablation still loses to the midpoint rule, and now loses for the reason stated
    above rather than for a second, avoidable one.

    Returns (nodes, weights). The weights are the gaps of the anchored edges and
    therefore telescope to exactly 1 by construction. A degenerate map (a single
    distinct probability v) needs no special case: the interior quantiles all land
    on v, giving the two-bin rule (v/2, (v+1)/2) with weights (v, 1 - v), which is
    the exact integral of a map whose only jump is at v.
    """
    values = np.unique(prob)
    # bin edges in probability space, anchored at 0 and 1 so the weights are the
    # gaps of the exact identity and integrate over the whole unit interval. The
    # outer bins carry the tails, and the upper one carries the atom Q_p places on
    # the empty mask, weight 1 - max_i p_i; normalizing over [min p, max p] instead
    # would silently drop them, and those tails are where the max-softmax signal
    # enters the score.
    interior = np.quantile(values, [index / count for index in range(1, count)])
    edges = np.concatenate([[0.0], interior, [1.0]])
    # representative point of each bin: its midpoint. Taken from the *anchored* edges,
    # so the top node CAN rise above max p as M grows -- but it is a midpoint, not a
    # bound, and at small M it does not (0.75 at M=2 on a 0.1/0.9 map, against
    # max p = 0.9). See the IMPLEMENTATION NOTE: the atom's WEIGHT is covered, its
    # VALUE is not.
    nodes = 0.5 * (edges[:-1] + edges[1:])
    weights = np.diff(edges)
    return list(nodes), list(weights)


def _level_set_losses(prob, predicted, nodes, restrict=None):
    """Per-node HD95 and 1 - Dice between {p >= t} and the hard prediction.

    ``prob`` and ``predicted`` are numpy arrays: one class's probability map
    and its argmax mask. Each HD95 is the pooled bidirectional 95th-percentile
    surface distance used by the band width and by the evaluation metric.

    ``restrict``, when given, is a list aligned with ``nodes`` whose entries are
    boolean masks (or ``None``) intersected into that node's level set. Only
    :func:`bilevel_r2` uses it, and only to nest the rank-anchored conservative
    set inside the prediction; leaving it unset reproduces the plain level sets
    exactly.

    One EDT of the prediction's surface serves every node — the distance field
    of the predicted boundary does not depend on t — so M nodes cost M EDTs
    rather than 2M; only the other reading direction, distances measured away
    from the level set's own surface, has to be recomputed per node.

    Also returns which nodes are non-degenerate. An empty level set has no
    surface: its HD95 is undefined and saturates at the image diagonal (the
    convention of :func:`bilevel_band_widths`, and the exact counterpart of
    Dice's own 0 there), and its Dice loss is 1.
    """
    diagonal = math.hypot(*prob.shape)
    if not predicted.any():
        count = len(nodes)
        return [diagonal] * count, [1.0] * count, [False] * count
    predicted_surface = _surface(predicted)
    to_predicted = ndimage.distance_transform_edt(~predicted_surface)
    distances, dice_losses, defined = [], [], []
    for position, node in enumerate(nodes):
        level = prob >= node
        if restrict is not None and restrict[position] is not None:
            level = level & restrict[position]
        if not level.any():
            distances.append(diagonal)
            dice_losses.append(1.0)
            defined.append(False)
            continue
        level_surface = _surface(level)
        outward = to_predicted[level_surface]
        inward = ndimage.distance_transform_edt(~level_surface)[predicted_surface]
        distances.append(float(np.percentile(np.hstack([outward, inward]), 95)))
        dice_losses.append(
            1 - 2 * float((level & predicted).sum())
            / float(level.sum() + predicted.sum())
        )
        defined.append(True)
    return distances, dice_losses, defined


def bilevel_r2(prob, predicted, alpha, conservative_threshold=None,
               conservative_mask=None):
    """Two-point quadrature risk of one class: the estimator the theory derives.

    Where :func:`bilevel_band_widths` measures the two outer level sets against
    *each other*, this measures each against the model's hard prediction
    ``predicted`` (the class's argmax mask), which is what the M=2 rule with
    nodes (alpha, t_hi) and equal weights actually asks for::

        r_2 = 1/2 [ HD95(Y_lo, Yhat) + HD95(Y_hi, Yhat) ]

    with Y_lo = {p >= alpha} and Y_hi = {p >= t_hi}. ``conservative_threshold``
    t_hi defaults to the constant 1 - alpha; passing the per-image threshold of
    :func:`rank_anchored_threshold` gives the rank-anchored r_2, the estimator
    counterpart of the rank-anchored band, on this same code path.
    ``conservative_mask`` intersects Y_hi with the class's prediction, exactly as
    in :func:`bilevel_band_widths` and for the same reason (a clipped t_hi can
    fall below the argmax floor); it is passed only on the rank-anchored path.

    The two quadrature terms are returned separately (``lo``, ``hi``) alongside
    their mean (``symmetric``), which is r_2 itself. Each term uses the same
    machinery as the band width — surface = mask minus its erosion, one EDT per
    set, pooled bidirectional 95th percentile — so the two scores are directly
    comparable and their ranking difference is attributable to the estimator
    alone.

    Conventions follow :func:`bilevel_band_widths`: ``None`` when the
    aggressive set is empty (class confidently absent), and the image diagonal
    for a term whose level set is empty (undefined distance, read as maximal
    uncertainty). Under the constant threshold and alpha < 1/2 only the
    conservative ``hi`` term can saturate this way, since a non-empty aggressive
    set is guaranteed above. Under the rank-anchored one it saturates exactly
    when the clip has been reverted (:func:`rank_anchored_threshold`), i.e. when
    the clipped conservative set would have carried no information; where the
    clip survives, Y_hi is non-empty by construction.
    """
    prob = prob.numpy()
    predicted = predicted.numpy()
    if conservative_threshold is None:
        conservative_threshold = 1 - alpha
    if not (prob >= alpha).any():
        return None
    restrict = None
    if conservative_mask is not None:
        restrict = [None, conservative_mask.numpy()]
    distances, _, _ = _level_set_losses(
        prob, predicted, [alpha, conservative_threshold], restrict
    )
    return {
        "lo": distances[0],
        "hi": distances[1],
        "symmetric": 0.5 * (distances[0] + distances[1]),
    }


def quadrature_risks(prob, predicted, nodes, weights=None):
    """M-point quadrature risks r_M = sum_m w_m L(Y_{t_m}, Yhat) for one class.

    Computed for both losses of the distance-vs-area ablation: L = HD95
    (``hd95``) and L = 1 - Dice (``dice``), each against ``predicted``, the
    class's argmax mask. ``weights`` defaults to the equal weights 1/M carried
    by :func:`quadrature_nodes`. The cost is M EDTs on top of the forward pass.

    Empty level sets — the nodes above the largest probability the map attains
    — are handled two ways. ``hd95`` saturates them at the image diagonal, the
    convention of :func:`bilevel_band_widths`; ``hd95_skip`` drops them and
    renormalizes the surviving weights. The two disagree sharply, and not
    subtly: for a map whose probabilities never reach the upper nodes,
    saturation reads maximal uncertainty while skipping reads perfect
    confidence on whatever evidence remains (a constant-0.55 map scores a
    *zero* skip-risk, its level sets agreeing exactly with its own argmax at
    every surviving node). ``hd95`` is therefore the reported convention and
    ``hd95_skip`` the ablation of it. The Dice loss needs no such choice: an
    empty level set overlaps nothing, so its loss is exactly 1.

    Returns ``None`` when even the lowest node's level set is empty (class
    confidently absent), matching :func:`bilevel_band_widths`.
    """
    prob = prob.numpy()
    nodes = list(nodes)
    if not (prob >= min(nodes)).any():
        return None
    if weights is None:
        weights = [1 / len(nodes)] * len(nodes)
    distances, dice_losses, defined = _level_set_losses(
        prob, predicted.numpy(), nodes
    )
    saturated = float(np.dot(weights, distances))
    kept = sum(
        weight for weight, ok in zip(weights, defined) if ok
    )
    skipped = sum(
        weight * distance
        for weight, distance, ok in zip(weights, distances, defined)
        if ok
    )
    return {
        "hd95": saturated,
        "hd95_skip": float(skipped / kept) if kept else saturated,
        "dice": float(np.dot(weights, dice_losses)),
    }


def _worst_patch(max_prob, patch=32):
    """Mean confidence of the least confident patch (patch aggregation)."""
    kernel = min(patch, max_prob.shape[-2], max_prob.shape[-1])
    pooled = F.avg_pool2d(
        max_prob[None, None], kernel_size=kernel, stride=max(1, kernel // 2)
    )
    return pooled.min().item()


def _tail_mean(uncertainty, threshold):
    """Mean of the uncertainty values above ``threshold`` (0 if none)."""
    tail = uncertainty.flatten()
    tail = tail[tail > threshold]
    return tail.mean().item() if tail.numel() else 0.0


def _median_min_max(values):
    """Median(v) + min(v)/max(v) over a per-pixel map v (the MMMC form)."""
    flat = values.flatten()
    return (
        flat.median() + flat.min() / flat.max().clamp_min(1e-12)
    ).item()


def _spatial_scores(prediction, foreground_predicted, entropy, uncertainty, dist=4.0):
    """Boundary-aware confidence scores from the predicted argmax map.

    Splits the image at the predicted class boundary (any pixel adjacent to
    a different argmax label) into a band within ``dist`` pixels of it and
    the rest. Returns Boundary Uncertainty Concentration (BUC, Zeevi et al.,
    UNCV 2025), boundary-band mean entropy (MetaSeg, Rottmann et al., IJCNN
    2020), and interior mean entropy over predicted foreground away from the
    boundary (Zenk et al., MedIA 2024). Entropies are negated for confidence
    orientation; empty regions fall back to the global mean.

    BUC is a *density* ratio, not a share of mass: mean uncertainty near the
    boundary over the sum of the near and the far means,

        buc = mean(u | near) / (mean(u | near) + mean(u | far)),

    so 0.5 is the neutral point of a spatially uniform uncertainty map (higher =
    concentrated at the boundary, where it belongs; lower = smeared through the
    interior, which is the pathology). The mass share sum(u | near) / sum(u) is a
    *different* functional -- on a uniform map it equals the near-band's area
    fraction rather than 0.5, so it is confounded by object size and has no natural
    neutral point. The docstring used to describe the mass share while the code
    computed the density ratio; the code is the deliberate one and is unchanged.
    """
    labels = prediction.numpy()
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[1:] |= labels[:-1] != labels[1:]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    entropy = entropy.numpy()
    uncertainty = uncertainty.numpy()
    if boundary.any():
        near = ndimage.distance_transform_edt(~boundary) <= dist
    else:
        near = np.zeros(labels.shape, dtype=bool)
    interior = foreground_predicted.numpy() & ~near
    inside = uncertainty[near].mean() if near.any() else 0.0
    outside = uncertainty[~near].mean() if (~near).any() else 0.0
    total = inside + outside
    return {
        "buc": float(inside / total) if total > 0 else 0.5,
        "neg_boundary_entropy": -float(
            entropy[near].mean() if near.any() else entropy.mean()
        ),
        "neg_interior_entropy": -float(
            entropy[interior].mean() if interior.any() else entropy.mean()
        ),
    }


def _present_readouts(readouts, present, saturated):
    """Per-class readouts for the present classes, with ``None`` SATURATED.

    ``readouts`` maps a class index to the dict returned by one of the per-class
    functions above, or to ``None``. Those functions return ``None`` on an empty
    aggressive set / an empty lowest level set, documented as "class confidently
    absent" -- which is the right reading for the ALL-class band
    (``neg_band_width@``), where an absent class genuinely contributes nothing.

    It is the wrong reading for a class that is *present* in the argmax, and the
    two are not the same set. eq:nesting's left containment Yhat_c subset Y_lo is
    FALSE for the argmax mask: a C-way softmax argmax winner needs only p >= 1/C
    (0.048 for VOC's C=21, well below alpha = 0.1), and a quadrature rule's lowest
    node is lower still (0.211 for gl-2, 0.0156 for mid-32). So a class can win the
    argmax somewhere -- hallucinating a whole object -- and still have an empty
    {p >= alpha}, whereupon the readout is ``None``.

    Dropping that ``None`` from the present-class mean deletes the most uncertain
    class in the image and makes the image read MORE confident: the class-level
    twin of the detection-failure inversion documented below, and it fires whenever
    SOME but not all present classes are sub-alpha, so ``diag_no_present_class``
    does not catch it. rem:bounded says what to do instead -- L(empty, Yhat) is the
    image diagonal for a distance loss, 0 for Dice/IoU (loss 1) -- and that is what
    ``saturated`` carries. It is exactly the value :func:`bilevel_band_widths`
    already returns one branch down, when the *conservative* set is the empty one.

    Substituting rather than dropping also keeps every aggregation over the SAME
    class set, which the M-ladder and the band-vs-r_2 surrogate ablation compare
    across: otherwise the cheap rules (whose lowest node is high) and the dense
    reference (whose lowest node is below the argmax floor) average over different
    classes on the same image, and the ablation measures class bookkeeping instead
    of node placement.
    """
    return [
        saturated if readouts[index] is None else readouts[index]
        for index in sorted(present)
    ]


def image_confidence_scores(probs, alphas):
    """Image-level confidence scores from (C, H, W) class-probability maps.

    Rows need not sum to one (CLIPSeg's background is 1 - max foreground),
    so the distribution is renormalized for the entropy score.

    Band widths are aggregated two ways per alpha: over every foreground
    class whose aggressive set is non-empty (``band_width`` — worst case,
    but an absent class with mid-range probabilities can hijack it), and
    over only the classes present in the argmax prediction
    (``band_width_pmax`` / ``band_width_pmean``). An image with NO present class
    is a total detection failure and takes each score's floor, never 0; a present
    class whose own readout is undefined saturates rather than dropping out of the
    mean (:func:`_present_readouts`). Both conventions are rem:bounded, and both
    exist because their absence makes the worst images of a split read as the most
    confident — which is a bug this code has had, twice.

    The two quadrature estimators — ``r2`` (:func:`bilevel_r2`) and ``rM``
    (:func:`quadrature_risks`) — use the present-class aggregation only, since
    both are defined against the class's argmax mask and an absent class has
    none.

    ``mband`` is the mask-intersected band of eq:mband, Y_hi := {p >= 1 - alpha}
    AND Yhat_c, which repairs the nesting failure that per-prompt sigmoid maps
    exhibit (see ``diag_nesting_leak@``). It is reported in both aggregations
    alongside the plain band. The plain band remains the REPORTING default
    (scripts/analyze_selective.py::DEFAULT_BAND) only so that every number in the paper
    is comparable to a single fixed score; the paper RECOMMENDS eq:mband as the
    DEPLOYED form of the band on any non-normalized map, since it is a provable no-op
    where the nesting leak is zero and strictly better where it is not (it closes
    42-77% of the band's gap to SDC on the sigmoid conditions). An earlier version of
    this docstring called the plain band "the deployed default", which inverts the
    paper's guidance -- "deployed" is the word main.tex reserves for eq:mband. Nothing
    numerical turns on the reporting choice (it flips no cell), but do not read
    DEFAULT_BAND as an endorsement.

    Each of the band width and r_2 also comes in a rank-anchored ``q`` variant
    (``qband_width``, ``qr2``) — **an ablation, not the default score** — which
    replaces the constant conservative threshold 1 - alpha with the per-class
    one of :func:`rank_anchored_threshold`. The two variants agree bit-for-bit
    on every class whose clip is inactive *or reverted*, and differ only where
    the clip is active, which ``diag_clipped@`` counts. That is strictly more
    than the saturated classes ``diag_saturated@`` counts, so a ranking
    difference between the variants is *not* attributable to the degenerate
    images alone. The q variants are present-class scores for the same reason
    the estimators are: the order statistic is taken inside the class's own
    predicted mask.
    """
    max_prob = probs.max(dim=0).values
    normalized = (probs / probs.sum(dim=0).clamp_min(1e-12)).clamp_min(1e-12)
    entropy = -(normalized * normalized.log()).sum(dim=0)
    top2 = normalized.topk(2, dim=0).values
    prediction = probs.argmax(dim=0)
    foreground_predicted = prediction > 0
    scores = {
        "mean_max_prob": max_prob.mean().item(),
        # numpy percentile avoids torch.quantile's 2**24-element limit
        "p05_max_prob": float(np.percentile(max_prob.numpy(), 5)),
        "mean_margin": (top2[0] - top2[1]).mean().item(),
        "neg_mean_entropy": -entropy.mean().item(),
        # entropy over predicted foreground only: robust to the large
        # confident background dominating the full-image mean
        "neg_fg_entropy": -(
            entropy[foreground_predicted].mean().item()
            if foreground_predicted.any()
            else entropy.mean().item()
        ),
        # threshold aggregation (ValUES): share of low-confidence pixels
        "neg_low_conf_fraction": -(max_prob < 0.75).float().mean().item(),
        # Median-Min-Max Confidence baseline from the SDC paper
        # (arXiv:2402.10665), applied to the confidence map so that higher
        # means more confident (the min/max term is monotone here).
        "mmmc": _median_min_max(max_prob),
        # DOCTOR (Granese et al., NeurIPS 2021): mean collision probability
        # sum_c p^2 of the renormalized class distribution (higher = more
        # confident; this is 1 - Gini impurity).
        "mean_collision_prob": (normalized**2).sum(dim=0).mean().item(),
        # Generalized Entropy (Liu et al., CVPR 2023): mean of the
        # gamma-entropy sum_c p^g (1-p)^g, g=0.1; a sharper transform than
        # collision probability (the g=1 case). Higher spread = less confident.
        "neg_gen": -(
            normalized**0.1 * (1 - normalized) ** 0.1
        ).sum(dim=0).mean().item(),
        # threshold-level aggregation (ValUES): mean severity of the
        # uncertain tail, complementing the count in neg_low_conf_fraction
        "neg_tail_uncertainty": -_tail_mean(1 - max_prob, 0.25),
        # patch aggregation (ValUES): confidence of the worst 32x32 patch
        # (kernel clamped for images smaller than the patch)
        "worst_patch_max_prob": _worst_patch(max_prob),
        **_spatial_scores(prediction, foreground_predicted, entropy, 1 - max_prob),
    }
    predicted_present = set(torch.unique(prediction).tolist()) - {0}
    # An image with no predicted foreground class is a total detection
    # failure, not a confident background: it must receive the *lowest*
    # confidence under every score. Distance-type scores are negated, so
    # their floor is the saturating width (the image diagonal, matching
    # bilevel_band_widths' own empty-conservative-set convention); area-type
    # scores and SDC are oriented so 0 is already their floor. Aggregating an
    # empty present-class list to 0.0 would instead make these images the
    # *most* confident of the split -- they are among the worst.
    diagonal = math.hypot(*prediction.shape)
    # The same convention one level down, per CLASS: rem:bounded's saturating
    # value for a present class whose readout is undefined. See
    # _present_readouts -- a present class must never be *dropped* from an
    # aggregation, only charged this.
    saturated_widths = {
        "outward": diagonal, "inward": diagonal, "symmetric": diagonal,
    }
    saturated_r2 = {"lo": diagonal, "hi": diagonal, "symmetric": diagonal}
    saturated_quadrature = {
        "hd95": diagonal, "hd95_skip": diagonal, "dice": 1.0,
    }
    saturated_overlap = {"iou": 0.0, "dice": 0.0}
    # a total detection failure: no foreground class predicted anywhere
    scores["diag_no_present_class"] = float(not predicted_present)
    # Soft Dice Confidence (arXiv:2402.10665): the soft Dice between the
    # hard prediction and the model's own probability map, i.e. the
    # expected Dice under the model's marginals. Per present class, with
    # the paper's empty-prediction convention (confidence 0).
    sdc_values = []
    for index in predicted_present:
        class_mask = prediction == index
        class_probs = probs[index]
        sdc_values.append(
            2 * class_probs[class_mask].sum().item()
            / (class_probs.sum().item() + class_mask.sum().item())
        )
    scores["sdc_pmean"] = (
        sum(sdc_values) / len(sdc_values) if sdc_values else 0.0
    )
    scores["sdc_pmin"] = min(sdc_values) if sdc_values else 0.0
    boundary_pixels = 0
    if foreground_predicted.any():
        mask = foreground_predicted.numpy()
        boundary_pixels = int(_surface(mask).sum())
    foreground = probs[1:]
    for alpha in alphas:
        widths = {
            index: bilevel_band_widths(probs[index], alpha)
            for index in range(1, probs.shape[0])
        }
        # the ALL-class band: here a None really does mean "class confidently
        # absent" (it never wins the argmax either), and dropping it is right.
        outward_all = [w["outward"] for w in widths.values() if w is not None]
        # the PRESENT-class bands: a None here is a class the model hallucinated
        # below alpha, and it saturates rather than vanishing (_present_readouts).
        present = _present_readouts(widths, predicted_present, saturated_widths)
        in_band = ((foreground >= alpha) & (foreground <= 1 - alpha)).any(dim=0)
        key = f"{alpha:g}"
        # Label-free degeneracy diagnostic. A present class whose conservative
        # set {p >= 1 - alpha} is empty has no inner anchor, so its width
        # saturates at the image diagonal and carries no ranking information.
        # A fixed alpha in probability space cannot avoid this: the condition
        # is 1 - alpha <= max_i p_i, a property of each image's own upper tail,
        # and diffuse (zero-shot) maps rarely reach it. Selecting alpha against
        # this rate needs no ground truth.
        scores[f"diag_saturated@{key}"] = float(
            sum(
                1
                for index in predicted_present
                if not (probs[index] >= 1 - alpha).any()
            )
            / len(predicted_present)
            if predicted_present
            else 1.0
        )
        # Label-free nesting diagnostic. The bi-level construction assumes
        # Y_hi = {p_c >= 1 - alpha} is contained in the class's argmax mask
        # Yhat_c; the band width is only a reading of *that class's* boundary
        # uncertainty if it is. A normalized softmax cannot break this (two
        # classes cannot both clear 1 - alpha >= 0.7 on the simplex), and a
        # binary map cannot either -- but CLIPSeg scores each prompt with an
        # independent sigmoid and sets background to 1 - max_c p_c, so
        # semantically overlapping prompts (cat/dog, cow/horse/sheep) can both
        # clear the threshold while only one wins the argmax. The leaked pixels
        # then pull the conservative contour outside the predicted object
        # entirely. Reported as the share of conservative-set pixels that fall
        # outside their own class's prediction, so it is comparable across
        # conditions and needs no ground truth.
        #
        # The loop runs over EVERY foreground class, not only the present ones.
        # The mechanism is that the *loser* of a co-firing pair keeps a
        # conservative set outside its own prediction -- and the extreme case of
        # that is a loser which wins the argmax NOWHERE, whose conservative set is
        # therefore 100% leaked. Restricting to predicted_present would drop
        # exactly those from both numerator and denominator and report 0.0 leak on
        # the worst instances of the very mechanism the diagnostic names. A class
        # that is genuinely absent has an empty conservative set and is skipped by
        # the guard below, so this is a no-op on softmax and binary maps (verified
        # in tests/test_selective.py) and widens the count only where a sigmoid map
        # really does leak.
        leaked = kept = 0
        for index in range(1, probs.shape[0]):
            conservative = probs[index] >= 1 - alpha
            if not conservative.any():
                continue
            inside = conservative & (prediction == index)
            kept += int(conservative.sum())
            leaked += int(conservative.sum()) - int(inside.sum())
        scores[f"diag_nesting_leak@{key}"] = float(leaked / kept) if kept else 0.0
        # Mask-intersected band: Y_hi := {p_c >= 1 - alpha} AND Yhat_c, which
        # restores the nesting the diagnostic above shows the sigmoid maps
        # break. It is a provable no-op wherever the leak is zero -- on a
        # normalized softmax the conservative set already lies inside the
        # prediction -- so it cannot cost anything on DeepLabV3, and it is the
        # direct repair for CLIPSeg, whose co-firing prompts are the whole
        # source of the leak. Reported alongside the constant band in both
        # aggregations (pmean and pmax), so the max variant of the repair is
        # evaluable too.
        #
        # The un-intersected neg_band_width_pmean_sym@0.1 stays the REPORTING default
        # (analyze_selective.DEFAULT_BAND) purely so every table number is comparable
        # to one fixed score -- NOT because it is preferred. The paper recommends the
        # intersected form as the DEPLOYED band on any non-normalized map, and
        # docs/FINDINGS.md says it should be the default; it flips no cell either way.
        # An earlier version of this comment said "the DEPLOYED default remains the
        # un-intersected [band], and the paper's tables say so" -- that inverts the
        # paper, whose tables were deliberately held fixed so as NOT to settle the
        # deployment question. If the recommendation is adopted, DEFAULT_BAND moves
        # with it and every table must be regenerated.
        masked = {
            index: bilevel_band_widths(
                probs[index], alpha, conservative_mask=(prediction == index)
            )
            for index in sorted(predicted_present)
        }
        masked_present = _present_readouts(
            masked, predicted_present, saturated_widths
        )
        for direction, suffix in (
            ("outward", ""), ("inward", "_in"), ("symmetric", "_sym"),
        ):
            values = [w[direction] for w in masked_present]
            scores[f"neg_mband_width_pmax{suffix}@{key}"] = -(
                max(values) if values else diagonal
            )
            scores[f"neg_mband_width_pmean{suffix}@{key}"] = -(
                sum(values) / len(values) if values else diagonal
            )
        scores[f"neg_band_width@{key}"] = -(
            max(outward_all) if outward_all else diagonal
        )
        for direction, suffix in (
            ("outward", ""), ("inward", "_in"), ("symmetric", "_sym"),
        ):
            values = [w[direction] for w in present]
            scores[f"neg_band_width_pmax{suffix}@{key}"] = -(
                max(values) if values else diagonal
            )
            scores[f"neg_band_width_pmean{suffix}@{key}"] = -(
                sum(values) / len(values) if values else diagonal
            )
        # band area as a share of the image — the crudest area readout of the
        # band. It is NEGATED, so unlike levelset_*/SDC (whose 0 is their floor)
        # its 0 is its CEILING: a confidently-empty background has no pixel in
        # [alpha, 1 - alpha] at all, so an unguarded -mean(in_band) would hand a
        # total detection failure the maximum of the score's [-1, 0] range and
        # rank the worst images of the split as the most confident -- the exact
        # inversion the comment above forbids, which this score alone used to
        # commit. -1.0 is its own floor, the area-type analogue of -diagonal.
        scores[f"neg_band_fraction@{key}"] = -(
            in_band.float().mean().item() if predicted_present else 1.0
        )
        # crude mean width: band area per predicted boundary pixel — the
        # perimeter-normalized ablation of the EDT-based width above. With no
        # predicted boundary there is no perimeter to normalize by, so this
        # too saturates rather than reading as confident.
        scores[f"neg_band_per_boundary@{key}"] = -(
            in_band.sum().item() / boundary_pixels
            if boundary_pixels
            else diagonal
        )
        # region-based level-set agreement (IoU / Dice between Y_high and
        # Y_low), the area counterpart to the distance-based band width;
        # present-class mean, oriented so higher = more confident.
        overlaps = _present_readouts(
            {
                index: bilevel_overlap(probs[index], alpha)
                for index in predicted_present
            },
            predicted_present,
            saturated_overlap,
        )
        for measure in ("iou", "dice"):
            values = [o[measure] for o in overlaps]
            scores[f"levelset_{measure}_pmean@{key}"] = (
                sum(values) / len(values) if values else 0.0
            )
        # two-point quadrature r_2: the estimator the theory derives, measured
        # against the hard prediction instead of between the two level sets.
        #
        # The band width above is a posterior-SPREAD SURROGATE for it -- and the
        # bridge between them is NOT PROVED for what this code computes. prop:band
        # gives HD(Y_lo, Y_hi) <= 2 r_2 only for the MAX Hausdorff distance, by the
        # triangle inequality. Everything here is HD95 -- a 95th PERCENTILE of pooled
        # surface distances, which is NOT a metric and does not obey the triangle
        # inequality -- so the bound does not transfer (rem:nobridge), and it fails by
        # an UNBOUNDED factor: on a disk plus two thin spikes the implemented band
        # width is 13.41 px while 2 r_2 = 0, each spike being individually truncated
        # away by the percentile. There is no equality condition either: an earlier
        # version of this comment said the band width is "equal to 2 r_2 under coherent
        # nesting, a lower bound otherwise". BOTH halves are REFUTED -- the paper
        # explicitly claims no equality condition (measured frequency 0/398, mean ratio
        # 0.665), and the band is not even a lower bound on 2 r_2 under HD95.
        #
        # Comparing the RANKINGS of the two is therefore the whole point of the
        # surrogate ablation (scripts/analyze_selective.py::spearman_band_vs_r2): an
        # EMPIRICAL question, not a corollary. The measured rho in [0.974, 0.992] is
        # the only thing connecting the deployed band to the r_2 it stands in for.
        # Both quadrature terms are also reported on their own.
        risks = _present_readouts(
            {
                index: bilevel_r2(probs[index], prediction == index, alpha)
                for index in sorted(predicted_present)
            },
            predicted_present,
            saturated_r2,
        )
        for term, suffix in (("symmetric", "_sym"), ("lo", "_lo"), ("hi", "_hi")):
            values = [r[term] for r in risks]
            scores[f"neg_r2_pmax{suffix}@{key}"] = -(
                max(values) if values else diagonal
            )
            scores[f"neg_r2_pmean{suffix}@{key}"] = -(
                sum(values) / len(values) if values else diagonal
            )
        # rank-anchored ('q') band and r_2 -- AN ABLATION, NOT THE DEFAULT. The
        # same two scores with the conservative threshold clipped to the 95th
        # percentile of the class's probabilities inside its own predicted mask,
        # the conservative set intersected with that mask whenever the clip is
        # active (nesting), and the clip abandoned altogether where it would
        # collapse the band (see rank_anchored_threshold). Only a *present* class
        # has an argmax mask to take the order statistic of, so unlike
        # neg_band_width@ there is no all-class counterpart.
        thresholds, qmasks, reverted = {}, {}, 0
        for index in sorted(predicted_present):
            mask = prediction == index
            # both non-None for every present class: its argmax mask is non-empty
            raw = _rank_anchored_percentile(
                probs[index].numpy(), mask.numpy(), alpha, 0.95
            )
            threshold = rank_anchored_threshold(probs[index], mask, alpha)
            reverted += raw < 1 - alpha and threshold == 1 - alpha
            # after the guard the band is a band: never inverted (t_hi > alpha,
            # which an argmax floor of 1/C < alpha would otherwise permit) and
            # never wider than the constant one (t_hi <= 1 - alpha).
            assert alpha < threshold <= 1 - alpha, (alpha, threshold)
            thresholds[index] = threshold
            # nest Y_hi inside the prediction only where the clip is active: with
            # t_hi = 1 - alpha the intersection is a no-op on a normalized map,
            # and omitting it there keeps the two variants bit-for-bit equal
            # (including on sigmoid maps, where the constant Y_hi is the one that
            # can leak and both variants must then leak the same way).
            qmasks[index] = mask if threshold < 1 - alpha else None
        # Companion diagnostics to diag_saturated@. diag_clipped@ is the share of
        # present classes whose clip is active *after* the collapse guard, i.e.
        # exactly the classes on which the two variants differ at all -- strictly
        # more than the saturated ones, so the fix is a reparameterization of the
        # score on the clipped-but-healthy classes, not only a repair of the dead
        # ones. diag_clip_reverted@ is the share whose clip was abandoned: a flat
        # in-mask distribution, or t_hi <= alpha. An image with no predicted class
        # has no order statistic at all, so it counts as fully reverted (both
        # variants give it the diagonal) and as unclipped.
        scores[f"diag_clipped@{key}"] = float(
            sum(1 for t in thresholds.values() if t < 1 - alpha)
            / len(thresholds)
            if thresholds
            else 0.0
        )
        scores[f"diag_clip_reverted@{key}"] = float(
            reverted / len(thresholds) if thresholds else 1.0
        )
        qwidths = _present_readouts(
            {
                index: bilevel_band_widths(
                    probs[index], alpha, thresholds[index], qmasks[index]
                )
                for index in sorted(predicted_present)
            },
            predicted_present,
            saturated_widths,
        )
        for direction, suffix in (
            ("outward", ""), ("inward", "_in"), ("symmetric", "_sym"),
        ):
            values = [w[direction] for w in qwidths]
            scores[f"neg_qband_width_pmax{suffix}@{key}"] = -(
                max(values) if values else diagonal
            )
            scores[f"neg_qband_width_pmean{suffix}@{key}"] = -(
                sum(values) / len(values) if values else diagonal
            )
        qrisks = _present_readouts(
            {
                index: bilevel_r2(
                    probs[index],
                    prediction == index,
                    alpha,
                    thresholds[index],
                    qmasks[index],
                )
                for index in sorted(predicted_present)
            },
            predicted_present,
            saturated_r2,
        )
        for term, suffix in (("symmetric", "_sym"), ("lo", "_lo"), ("hi", "_hi")):
            values = [r[term] for r in qrisks]
            scores[f"neg_qr2_pmax{suffix}@{key}"] = -(
                max(values) if values else diagonal
            )
            scores[f"neg_qr2_pmean{suffix}@{key}"] = -(
                sum(values) / len(values) if values else diagonal
            )
    # M-point quadrature: the same expected loss at M in {2, 4, 8, 16, 32},
    # under the distance loss (HD95) and its area counterpart (1 - Dice). The
    # dense M=32 rule is the reference the cheap rules are ranked against; the
    # nodes are fixed by the rule, so unlike the band scores these do not
    # depend on alpha.
    # importance-sampled quadrature: same estimand, nodes at the map's own
    # probability quantiles, weights = the gaps they span (see importance_nodes).
    # The comparison against the equispaced "mid" rule at the SAME M is the test
    # of whether adaptive node placement buys accuracy for free -- which requires
    # the two sides to average over the SAME class set, hence _present_readouts on
    # both. A FAILED ABLATION: the nodes land at the vertex of the V, where the
    # integrand is smallest. Kept, and pinned as failed, so nobody promotes it.
    for count in (2, 4, 8):
        risks = _present_readouts(
            {
                index: quadrature_risks(
                    probs[index],
                    prediction == index,
                    *importance_nodes(probs[index].numpy(), count),
                )
                for index in sorted(predicted_present)
            },
            predicted_present,
            saturated_quadrature,
        )
        for measure, infix, empty in (
            ("hd95", "", diagonal), ("dice", "_dice", 1.0),
        ):
            values = [r[measure] for r in risks]
            scores[f"neg_rM{count}{infix}_pmean@imp"] = -(
                sum(values) / len(values) if values else empty
            )
    # --- CANDIDATE NODE RULES, the P1 experiment -------------------------------
    # Node placement WAS an open empirical question; the eight real conditions have now
    # DECIDED it, in the NEGATIVE. Against the deployed band under the penalized
    # boundary risk: dense M=32 wins 8/8, mid2 (0.25,0.75) wins 6/8, and the
    # vertex-aware vtx2 wins only 4/8 -- the WORST of the candidates, and worse than the
    # band it was meant to repair on the two conditions where its mechanism can even
    # operate. vtx2 is the FOURTH principled adaptive node rule to fail empirically,
    # joining rank_anchored_threshold, importance_nodes, and the sqrt(V_L/V_R)
    # allocation. It stays here as a pinned FAILED ABLATION so nobody re-promotes it;
    # do not read the "measures better on multi-class" star-discrepancy framing below as
    # a recommendation -- it is exactly the intuition the real maps refuted. (Two
    # synthetic generators had already disagreed about the optimum, 0.25/0.75 against
    # 0.50/0.78, and the minimax argument licenses nothing: a node set with a WORSE
    # Koksma bound can achieve a BETTER error.)
    #
    #   mid2  : the binary-derived midpoint (0.25, 0.75). This is eq:derivednodes with
    #           the vertex ESTIMATE t_hat pinned to 1/2 instead of to m_c, and it is
    #           the vertex-aware rule ONLY when m_c = 1/2 exactly. An earlier version
    #           of this comment said it is "correct when the vertex of H sits at 1/2,
    #           which holds for a binary map and for CLIPSeg's bg = 1 - max(fg)". That
    #           is REFUTED. 1/2 is only the argmax FLOOR phi on those two constructions
    #           (prop:floor(ii)), and the floor is a LOWER BOUND on m_c, not the vertex:
    #           on a CLIPSeg-style map the measured t* is 0.550 with m_c = 0.533, so m_c
    #           is 3x the closer predictor. Even on a binary map 1/2 is the vertex only
    #           if NO pixel ties at exactly 1/2 -- argmax breaks ties toward background,
    #           so Yhat = {p > 1/2} STRICTLY while Y_{1/2} = {p >= 1/2} keeps them. For
    #           a C-way softmax it is nowhere near (phi = 1/C = 0.048 on VOC).
    #   vtx2  : split at m_c = min in-mask probability, the tightest threshold whose
    #           level set still contains the prediction, then take the midpoint of each
    #           half. Weights are the subinterval widths, so ONLY the node placement
    #           adapts and the estimand int_0^1 H stays exactly the same -- which is
    #           what keeps this safe where rank_anchored_threshold and importance_nodes
    #           were not. m_c is an ESTIMATE of the vertex, not the vertex: there is no
    #           closed form (the argmax floor is only a lower bound on it).
    #
    # The straddle hypothesis (rem:straddle) -- that dense M=32 beat M=2 on VOC because
    # a 21-way softmax puts its vertex near 0.06, so BOTH the deployed (0.1,0.9) and the
    # midpoint (0.25,0.75) sit on the same side and never straddle the minimum -- has
    # been REFUTED twice on clean data. (i) vtx2 straddles the vertex by construction
    # and LOSES (above). (ii) The band->M32 gain is LARGEST exactly where the mechanism
    # is switched OFF: on the two true-softmax conditions (phi=0.048, band unstraddled)
    # the gain is +6.9%/+5.6%, SIGNIFICANTLY SMALLER (non-overlapping CIs) than the two
    # CLIPSeg-VOC conditions (phi=1/2, band already straddles) at +17.6%/+14.7%. The
    # contamination it was once confounded with (rem:mladdercontam) turned out never to
    # have fired at the deployed alpha, and the M-ladder reproduced to the digit -- so
    # the explanandum (5.6-17.6%) stands, but the explanation does not. What dense
    # actually buys decomposes into readout + placement + count, and the count leg is
    # significantly NEGATIVE on 1 of 8 conditions (a second borderline): "more nodes
    # help" is not the reading.
    #
    # AGGREGATION. Both rules go through _present_readouts, like every other rung of
    # the ladder. They used to filter `if risk is not None`, which DROPPED a present
    # class whose level set was empty instead of saturating it -- verbatim the defect
    # rem:mladdercontam describes, still live in the P1 experiment that is supposed to
    # settle node placement. It fired asymmetrically between the two rules being
    # compared: mid2's lowest node is 0.25, so a class winning the argmax at p ~ 1/21 =
    # 0.048 has an empty {p >= 0.25} and was deleted, while vtx2's lowest node
    # 0.5 * m_c always sits below m_c and so never empties. Measured on a 21-way softmax
    # with one hallucinated class, neg_rmid2_pmean read -0.0 -- MAXIMAL CONFIDENCE on an
    # image containing a hallucinated object -- against -33.9 for the correctly
    # saturated value and -33.4 for dense M=32. The two rules were being compared over
    # DIFFERENT CLASS SETS on the same image, which is the contaminated ladder's error
    # committed a second time.
    for label, count in (("mid2", 2), ("vtx2", 2)):
        readouts = {}
        for index in sorted(predicted_present):
            mask = prediction == index
            channel = probs[index]
            if label == "mid2":
                nodes, weights = [0.25, 0.75], [0.5, 0.5]
            else:
                m_c = float(channel[mask].min())
                # a degenerate m_c leaves no room for a node on one side; fall back to
                # the midpoint rule rather than emit a rule with a zero-width half
                if not 1e-3 < m_c < 1 - 1e-3:
                    nodes, weights = [0.25, 0.75], [0.5, 0.5]
                else:
                    nodes = [0.5 * m_c, 0.5 * (m_c + 1.0)]
                    weights = [m_c, 1.0 - m_c]
            readouts[index] = quadrature_risks(channel, mask, nodes, weights=weights)
        risks = _present_readouts(
            readouts, predicted_present, saturated_quadrature
        )
        for measure, infix, empty in (("hd95", "", diagonal), ("dice", "_dice", 1.0)):
            values = [r[measure] for r in risks]
            # key order matches the rest of the ladder (neg_rM32_dice_pmean@mid), so the
            # suite's "which scores floor at -diag and which at -1?" dispatch keeps
            # working: the Dice loss is bounded by 1, the distance losses by the diagonal
            scores[f"neg_r{label}{infix}_pmean"] = -(
                sum(values) / len(values) if values else empty
            )

    for rule, count in QUADRATURE_RULES:
        nodes = quadrature_nodes(rule, count)
        # a present class whose peak sits below this rule's LOWEST node reads
        # None -- and min(nodes) is rule-dependent (0.211 for gl-2 down to 0.0156
        # for mid-32), so dropping it would make each rung of the ladder average
        # over a different class set and turn the quadrature ablation into a
        # measurement of class bookkeeping. It saturates instead.
        risks = _present_readouts(
            {
                index: quadrature_risks(probs[index], prediction == index, nodes)
                for index in sorted(predicted_present)
            },
            predicted_present,
            saturated_quadrature,
        )
        for measure, infix, empty in (
            ("hd95", "", diagonal),
            ("hd95_skip", "_skip", diagonal),
            # the Dice loss is bounded by 1, so a total detection failure
            # takes the maximal loss rather than the diagonal
            ("dice", "_dice", 1.0),
        ):
            values = [r[measure] for r in risks]
            scores[f"neg_rM{count}{infix}_pmax@{rule}"] = -(
                max(values) if values else empty
            )
            scores[f"neg_rM{count}{infix}_pmean@{rule}"] = -(
                sum(values) / len(values) if values else empty
            )
    return scores


def risk_coverage_curve(confidences, risks):
    """Risk–coverage curve: coverage grid k/n and the mean risk of the k
    most confident samples, for k = 1..n."""
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    coverages, curve = [], []
    running = 0.0
    for count, index in enumerate(order, start=1):
        running += risks[index]
        coverages.append(count / len(order))
        curve.append(running / count)
    return coverages, curve


def aurc(confidences, risks):
    """Area under the risk–coverage curve (lower is better)."""
    _, curve = risk_coverage_curve(confidences, risks)
    return sum(curve) / len(curve)
