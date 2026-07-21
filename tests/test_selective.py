"""Unit tests for the selective-segmentation confidence scores (CPU)."""

import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from selectseg.selective import (
    DENSE_RULE,
    QUADRATURE_RULES,
    _level_set_losses,
    aurc,
    bilevel_band_widths,
    bilevel_overlap,
    bilevel_r2,
    image_confidence_scores,
    importance_nodes,
    quadrature_nodes,
    quadrature_risks,
    rank_anchored_threshold,
    risk_coverage_curve,
)


def _foreground_probs(prob):
    """(2, H, W) probabilities for a single-class foreground map."""
    return torch.stack([1 - prob, prob])


def _predicted(prob):
    """The class's argmax mask, the Yhat the quadrature estimators score."""
    return _foreground_probs(prob).argmax(dim=0) == 1


def _sharp():
    """A confident map: a 0.99 core inside a 0.4 ring.

    The argmax mask is the core, whose probabilities are all 0.99 — above
    1 - alpha for the alphas used here, so the rank-anchored clip is inactive.
    """
    prob = torch.zeros(40, 40)
    prob[6:34, 6:34] = 0.4
    prob[10:30, 10:30] = 0.99
    return prob


def _diffuse(core=slice(15, 25)):
    """A diffuse map that never reaches 1 - alpha: the degenerate case.

    A 20x20 square at 0.55 with a 0.6 core. The argmax mask is the whole square
    (every in-mask p > 1/2) and the core is a quarter of it, so the 95th in-mask
    percentile is 0.6 while the conservative set {p >= 0.9} is empty. Shrinking
    ``core`` widens the band without changing either fact — which is how two
    such maps can be told apart only by the rank-anchored score.
    """
    prob = torch.zeros(40, 40)
    prob[10:30, 10:30] = 0.55
    prob[core, core] = 0.6
    return prob


def _softmax_plurality_leak():
    """A normalized 3-class map whose argmax winner sits below 1/2.

    The pathology the constant threshold is immune to and the clip is not. A
    softmax argmax winner only needs p >= 1/C, so class 1 owns its mask at
    p = 0.35 (rim) / 0.42 (core) — its 95th in-mask percentile, and hence the
    clipped t_hi, is 0.42, far below 1/2. Class 1 also sits at 0.44 inside class
    2's mask, where class 2 outbids it: {p_1 >= t_hi} therefore leaks *outside*
    class 1's own prediction. Two classes can never both clear the constant
    1 - alpha >= 0.7 on a simplex, so this is unreachable without the clip.

    Returns (probs, class-1 mask). A 0.20 skirt keeps [alpha, t_hi) populated so
    the collapse guard does not fire and the leak is actually exercised.
    """
    zeros = torch.zeros(40, 40)
    class_1, class_2 = zeros.clone(), zeros.clone()
    background = torch.ones(40, 40)
    # skirt: predicted background, but inside the aggressive set
    class_1[3:17, 3:17], class_2[3:17, 3:17], background[3:17, 3:17] = 0.20, 0.10, 0.70
    # class 1's mask: rim at 0.35, core at 0.42 (both below 1/2, both winners)
    class_1[5:15, 5:15], class_2[5:15, 5:15], background[5:15, 5:15] = 0.35, 0.33, 0.32
    class_1[7:13, 7:13], class_2[7:13, 7:13], background[7:13, 7:13] = 0.42, 0.30, 0.28
    # class 2's mask: class 1 is a close second here, above t_hi = 0.42
    class_1[25:35, 25:35], class_2[25:35, 25:35], background[25:35, 25:35] = (
        0.44, 0.45, 0.11
    )
    probs = torch.stack([background, class_1, class_2])
    assert torch.allclose(probs.sum(dim=0), torch.ones(40, 40))  # normalized
    return probs, probs.argmax(dim=0) == 1


def test_band_widths_zero_for_sharp_mask():
    prob = torch.zeros(32, 32)
    prob[8:24, 8:24] = 1.0
    widths = bilevel_band_widths(prob, alpha=0.05)
    assert widths == {"outward": 0.0, "inward": 0.0, "symmetric": 0.0}


def test_band_widths_none_when_class_absent():
    assert bilevel_band_widths(torch.zeros(16, 16), alpha=0.05) is None


def test_band_widths_diagonal_when_no_conservative_set():
    prob = torch.full((30, 40), 0.5)
    widths = bilevel_band_widths(prob, alpha=0.05)
    for value in widths.values():
        assert value == pytest.approx(math.hypot(30, 40))


def test_band_widths_match_ring_geometry():
    # 0.99 core, a 4-pixel 0.5 ring around it: edge width 4, corners ~5.66
    prob = torch.zeros(40, 40)
    prob[6:34, 6:34] = 0.5
    prob[10:30, 10:30] = 0.99
    widths = bilevel_band_widths(prob, alpha=0.05)
    for direction in ("outward", "inward", "symmetric"):
        assert 4.0 <= widths[direction] <= math.hypot(4, 4) + 1e-6


def test_bilevel_overlap_area_ratio():
    # conservative set = 4 pixels (p>=0.95), aggressive set = 16 pixels
    # (p>=0.05): IoU = 4/16, Dice = 2*4/(4+16)
    prob = torch.zeros(16, 16)
    prob[:4, :4] = 0.5   # aggressive only (16 px total incl. core)
    prob[:2, :2] = 0.99  # conservative core (4 px)
    overlap = bilevel_overlap(prob, alpha=0.05)
    assert overlap["iou"] == pytest.approx(4 / 16)
    assert overlap["dice"] == pytest.approx(2 * 4 / (4 + 16))


def test_bilevel_overlap_edge_cases():
    assert bilevel_overlap(torch.zeros(8, 8), alpha=0.05) is None  # absent
    fuzzy = torch.full((8, 8), 0.5)  # aggressive full, conservative empty
    overlap = bilevel_overlap(fuzzy, alpha=0.05)
    assert overlap == {"iou": 0.0, "dice": 0.0}
    sharp = torch.ones(8, 8)  # both sets identical
    overlap = bilevel_overlap(sharp, alpha=0.05)
    assert overlap == {"iou": 1.0, "dice": 1.0}


def test_band_width_directions_differ_on_detached_blob():
    # confident core plus a detached mid-confidence blob: the outward
    # reading sees the blob, the inward reading is blind to it.
    prob = torch.zeros(32, 32)
    prob[8:16, 8:16] = 0.99
    prob[24:30, 24:30] = 0.5
    widths = bilevel_band_widths(prob, alpha=0.1)
    assert widths["inward"] == 0.0
    assert widths["outward"] > 10.0
    assert widths["symmetric"] > 0.0


def test_image_confidence_scores_orientation():
    sharp = torch.zeros(32, 32)
    sharp[8:24, 8:24] = 1.0
    fuzzy = torch.full((32, 32), 0.55)  # predicted foreground, wide band
    alphas = [0.05]
    sharp_scores = image_confidence_scores(
        torch.stack([1 - sharp, sharp]), alphas
    )
    fuzzy_scores = image_confidence_scores(
        torch.stack([1 - fuzzy, fuzzy]), alphas
    )
    for key in (
        "neg_band_width@0.05",
        "neg_band_width_pmax@0.05",
        "neg_band_width_pmean@0.05",
        "neg_band_width_pmax_in@0.05",
        "neg_band_width_pmean_in@0.05",
        "neg_band_width_pmax_sym@0.05",
        "neg_band_width_pmean_sym@0.05",
        # the rank-anchored variants must survive the repo's canonical
        # maximally-uncertain fixture too: the flat 0.55 map is a plateau, on
        # which an unguarded clip puts Y_hi = Y_lo and reports width 0, i.e.
        # *maximal* confidence, tying the fuzziest map with the sharpest one.
        "neg_qband_width_pmax_sym@0.05",
        "neg_qband_width_pmean_sym@0.05",
        "neg_qband_width_pmax@0.05",
        "neg_qband_width_pmean@0.05",
        "neg_qr2_pmax_sym@0.05",
        "neg_qr2_pmean_sym@0.05",
        "neg_band_fraction@0.05",
        "neg_band_per_boundary@0.05",
        "mean_max_prob",
        "p05_max_prob",
        "mean_margin",
        "neg_mean_entropy",
        "neg_fg_entropy",
        "neg_low_conf_fraction",
        "mmmc",
        "sdc_pmean",
        "sdc_pmin",
        "mean_collision_prob",
        "neg_gen",
        "neg_tail_uncertainty",
        "neg_boundary_entropy",
        "neg_interior_entropy",
        "buc",
        "worst_patch_max_prob",
    ):
        assert sharp_scores[key] > fuzzy_scores[key], key


def test_mmmc_ranks_confident_image_higher():
    # a confident image (max_prob 0.9 everywhere) vs an uncertain one (half
    # the pixels at prob 0.5, i.e. max_prob 0.5): MMMC on the confidence map
    # must rank the confident image higher — the audit caught the fix's
    # predecessor doing the opposite.
    confident = torch.full((16, 16), 0.9)
    uncertain = torch.full((16, 16), 0.9)
    uncertain[:8] = 0.5  # half the pixels maximally uncertain (max_prob 0.5)
    conf_scores = image_confidence_scores(
        torch.stack([1 - confident, confident]), [0.1]
    )
    unc_scores = image_confidence_scores(
        torch.stack([1 - uncertain, uncertain]), [0.1]
    )
    assert conf_scores["mmmc"] > unc_scores["mmmc"]


def test_soft_dice_confidence_hand_computed():
    # probs 0.8/0.6 predicted foreground, 0.4/0.0 background:
    # SDC = 2*(0.8+0.6) / ((0.8+0.6+0.4+0.0) + 2)
    foreground = torch.tensor([[0.8, 0.6], [0.4, 0.0]])
    probs = torch.stack([1 - foreground, foreground])
    scores = image_confidence_scores(probs, [0.1])
    assert scores["sdc_pmean"] == pytest.approx(2.8 / 3.8)
    assert scores["sdc_pmin"] == pytest.approx(2.8 / 3.8)


def test_boundary_scores_concentrate_uncertainty_at_boundary():
    # a foreground square with a soft (uncertain) rim and a sharp core:
    # uncertainty should sit near the predicted boundary, so BUC > 0.5.
    foreground = torch.zeros(40, 40)
    foreground[8:32, 8:32] = 1.0
    foreground[8:32, 8:10] = 0.55  # left rim: predicted fg but uncertain
    foreground[8:32, 30:32] = 0.55  # right rim
    probs = torch.stack([1 - foreground, foreground])
    scores = image_confidence_scores(probs, [0.1])
    assert scores["buc"] > 0.5
    # interior (sharp core) is more confident than the boundary band
    assert scores["neg_interior_entropy"] > scores["neg_boundary_entropy"]


def test_generalized_entropy_reduces_to_gini_at_gamma_one():
    # sanity: at gamma=1, sum_c p(1-p) = 1 - sum_c p^2 = 1 - gini
    probs = torch.tensor([[[0.7]], [[0.3]]])
    normalized = probs / probs.sum(dim=0)
    gini = (normalized**2).sum().item()
    gamma_one = (normalized * (1 - normalized)).sum().item()
    assert gamma_one == pytest.approx(1 - gini)


def test_soft_dice_confidence_empty_prediction_is_zero():
    foreground = torch.full((4, 4), 0.2)  # never wins the argmax
    probs = torch.stack([1 - foreground, foreground])
    scores = image_confidence_scores(probs, [0.1])
    assert scores["sdc_pmean"] == 0.0
    assert scores["sdc_pmin"] == 0.0


def test_present_class_aggregation_ignores_absent_class_noise():
    # class 1: sharp, predicted present; class 2: mid-probability blob that
    # never wins the argmax — it must not affect the present-class scores.
    class_1 = torch.zeros(32, 32)
    class_1[8:16, 8:16] = 0.99
    class_2 = torch.zeros(32, 32)
    class_2[20:28, 20:28] = 0.45
    background = (1 - class_1 - class_2).clamp_min(0)
    probs = torch.stack([background, class_1, class_2])
    scores = image_confidence_scores(probs, [0.1])
    diagonal = math.hypot(32, 32)
    assert scores["neg_band_width@0.1"] == pytest.approx(-diagonal)
    assert scores["neg_band_width_pmax@0.1"] == pytest.approx(0.0)
    assert scores["neg_band_width_pmean@0.1"] == pytest.approx(0.0)


def test_quadrature_nodes_are_the_tabulated_rules():
    gauss = quadrature_nodes("gl", 2)
    assert gauss == pytest.approx([0.2113248654, 0.7886751346])
    assert quadrature_nodes("mid", 4) == pytest.approx([0.125, 0.375, 0.625, 0.875])
    # every tabulated rule stays strictly inside the open interval
    for rule, count in QUADRATURE_RULES:
        nodes = quadrature_nodes(rule, count)
        assert len(nodes) == count
        assert all(0 < node < 1 for node in nodes)
        assert nodes == sorted(nodes)
    with pytest.raises(ValueError):
        quadrature_nodes("simpson", 3)


def test_r2_zero_for_sharp_mask():
    prob = torch.zeros(32, 32)
    prob[8:24, 8:24] = 1.0
    risk = bilevel_r2(prob, _predicted(prob), alpha=0.05)
    assert risk == {"lo": 0.0, "hi": 0.0, "symmetric": 0.0}


def test_r2_none_when_class_absent():
    prob = torch.zeros(16, 16)
    assert bilevel_r2(prob, _predicted(prob), alpha=0.05) is None


def test_r2_hi_term_saturates_when_conservative_set_empty():
    # a 0.6 map: predicted foreground everywhere, so the aggressive set agrees
    # with the prediction exactly (lo = 0), but no pixel reaches 0.95, leaving
    # the conservative term undefined and saturated at the image diagonal.
    prob = torch.full((30, 40), 0.6)
    risk = bilevel_r2(prob, _predicted(prob), alpha=0.05)
    diagonal = math.hypot(30, 40)
    assert risk["lo"] == 0.0
    assert risk["hi"] == pytest.approx(diagonal)
    assert risk["symmetric"] == pytest.approx(diagonal / 2)


def test_band_width_is_twice_r2_under_coherent_nesting():
    # Prop. 4's equality case: a 0.99 core inside a 0.4 ring. The argmax
    # prediction coincides with the conservative set, so the predicted contour
    # lies on every shortest path from the aggressive contour to the
    # conservative one and the triangle inequality is tight.
    prob = torch.zeros(40, 40)
    prob[6:34, 6:34] = 0.4
    prob[10:30, 10:30] = 0.99
    predicted = _predicted(prob)
    widths = bilevel_band_widths(prob, alpha=0.05)
    risk = bilevel_r2(prob, predicted, alpha=0.05)
    assert risk["hi"] == 0.0  # Yhat == Y_hi here
    assert risk["lo"] == pytest.approx(widths["symmetric"])
    assert widths["symmetric"] == pytest.approx(2 * risk["symmetric"])


def test_r2_is_the_two_point_case_of_the_m_point_rule():
    prob = torch.zeros(40, 40)
    prob[6:34, 6:34] = 0.4
    prob[10:30, 10:30] = 0.99
    predicted = _predicted(prob)
    risk = bilevel_r2(prob, predicted, alpha=0.1)
    quadrature = quadrature_risks(prob, predicted, [0.1, 0.9])
    assert quadrature["hd95"] == pytest.approx(risk["symmetric"])


def test_quadrature_zero_for_sharp_mask():
    # p in {0, 1}: every level set equals the prediction, so both losses
    # vanish at every node, for every rule.
    prob = torch.zeros(32, 32)
    prob[8:24, 8:24] = 1.0
    predicted = _predicted(prob)
    for rule, count in QUADRATURE_RULES:
        risk = quadrature_risks(prob, predicted, quadrature_nodes(rule, count))
        assert risk["hd95"] == 0.0
        assert risk["hd95_skip"] == 0.0
        assert risk["dice"] == 0.0


def test_quadrature_none_when_class_absent():
    prob = torch.zeros(16, 16)
    nodes = quadrature_nodes("mid", 4)
    assert quadrature_risks(prob, _predicted(prob), nodes) is None


def test_quadrature_dice_hand_computed():
    # row 0 at p=0.9 (predicted), row 1 at p=0.3 (not predicted). At t=0.2 the
    # level set is both rows (8 px) against a 4 px prediction: Dice = 8/12, so
    # the loss is 1/3. At t=0.5 the level set is the prediction: loss 0.
    prob = torch.zeros(4, 4)
    prob[0] = 0.9
    prob[1] = 0.3
    risk = quadrature_risks(prob, _predicted(prob), [0.2, 0.5])
    assert risk["dice"] == pytest.approx((1 / 3 + 0.0) / 2)
    assert risk["hd95"] > 0.0


def test_quadrature_weights_are_honoured():
    prob = torch.zeros(4, 4)
    prob[0] = 0.9
    prob[1] = 0.3
    predicted = _predicted(prob)
    # all the mass on the degenerate-free upper node: loss 0
    assert quadrature_risks(prob, predicted, [0.2, 0.5], [0.0, 1.0])["dice"] == 0.0
    assert quadrature_risks(prob, predicted, [0.2, 0.5], [1.0, 0.0])[
        "dice"
    ] == pytest.approx(1 / 3)


def test_quadrature_saturates_empty_level_sets_and_skipping_does_not():
    # a constant 0.55 map: predicted foreground everywhere, but the two upper
    # midpoint nodes have empty level sets. Saturation charges the image
    # diagonal there (maximal uncertainty); skipping renormalizes onto the two
    # surviving nodes, where the level set agrees with the argmax exactly, and
    # so calls this maximally fuzzy image perfectly confident. This is why
    # hd95 (saturating) is the reported convention and hd95_skip the ablation.
    prob = torch.full((16, 16), 0.55)
    risk = quadrature_risks(prob, _predicted(prob), quadrature_nodes("mid", 4))
    diagonal = math.hypot(16, 16)
    assert risk["hd95"] == pytest.approx(diagonal / 2)  # 2 of 4 nodes empty
    assert risk["hd95_skip"] == 0.0
    assert risk["dice"] == pytest.approx(0.5)  # Dice loss is 1 on an empty set


def test_quadrature_scores_orientation():
    # the estimators of §6.2, on the same sharp-vs-fuzzy pair as the band
    # scores. The '_lo' term and the skip convention are excluded: both are 0
    # for either image (the fuzzy map's aggressive set *is* its argmax), which
    # is a fact about them worth stating, not an orientation failure.
    sharp = torch.zeros(32, 32)
    sharp[8:24, 8:24] = 1.0
    fuzzy = torch.full((32, 32), 0.55)
    sharp_scores = image_confidence_scores(_foreground_probs(sharp), [0.05])
    fuzzy_scores = image_confidence_scores(_foreground_probs(fuzzy), [0.05])
    keys = [
        "neg_r2_pmean_sym@0.05",
        "neg_r2_pmax_sym@0.05",
        "neg_r2_pmean_hi@0.05",
        "neg_r2_pmax_hi@0.05",
    ]
    for rule, count in QUADRATURE_RULES:
        keys += [
            f"neg_rM{count}_pmean@{rule}",
            f"neg_rM{count}_pmax@{rule}",
            f"neg_rM{count}_dice_pmean@{rule}",
            f"neg_rM{count}_dice_pmax@{rule}",
        ]
    for key in keys:
        assert sharp_scores[key] > fuzzy_scores[key], key
    for key in (
        "neg_r2_pmean_lo@0.05",
        f"neg_rM{DENSE_RULE[1]}_skip_pmean@{DENSE_RULE[0]}",
    ):
        assert sharp_scores[key] == 0.0 and fuzzy_scores[key] == 0.0


def test_quadrature_scores_present_in_flat_score_dict():
    prob = torch.zeros(32, 32)
    prob[8:24, 8:24] = 0.99
    scores = image_confidence_scores(_foreground_probs(prob), [0.05, 0.1])
    dense = f"neg_rM{DENSE_RULE[1]}_pmean@{DENSE_RULE[0]}"
    assert dense in scores  # the dense reference the M-ablation ranks against
    assert all(isinstance(value, float) for value in scores.values())
    # the deployed band width and its scores must survive untouched
    assert "neg_band_width_pmean_sym@0.1" in scores


def test_quadrature_scores_saturate_when_no_class_is_predicted():
    # Nothing wins the argmax, so the present-class aggregation is empty. This
    # is a *total detection failure*, not a confident background, and it must
    # take the worst value of each loss -- the image diagonal for the distance
    # rules, 1 for the bounded Dice loss. Falling back to 0 would instead make
    # these images the most confident of the split, because the scores are
    # negated. See test_detection_failure_is_least_confident_not_most.
    prob = torch.full((8, 8), 0.2)
    scores = image_confidence_scores(_foreground_probs(prob), [0.1])
    diagonal = math.hypot(8, 8)
    assert scores["neg_r2_pmean_sym@0.1"] == pytest.approx(-diagonal)
    assert scores["neg_rM4_pmean@mid"] == pytest.approx(-diagonal)
    assert scores["neg_rM4_dice_pmean@mid"] == pytest.approx(-1.0)


def test_risk_coverage_curve_hand_computed():
    coverages, curve = risk_coverage_curve([1.0, 0.0], [0.0, 1.0])
    assert coverages == [0.5, 1.0]
    assert curve == [0.0, 0.5]


def test_aurc_hand_computed():
    # perfect ordering: the risky image is accepted last
    assert aurc([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.25)
    # inverted ordering: the risky image is accepted first
    assert aurc([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.75)
    # constant risk: any ordering gives that risk
    assert aurc([0.3, 0.9, 0.1], [0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_detection_failure_is_least_confident_not_most():
    """An image with no predicted foreground class must rank last, not first.

    Negated distance scores floor at 0, so aggregating an empty present-class
    list to 0.0 would make a total detection failure the *most* confident image
    of the split -- while SDC, whose 0.0 is its minimum, puts it last. The two
    would rank the same images at opposite ends. Every score must agree that a
    detection failure is the worst case.
    """
    # background everywhere: no foreground class is ever the argmax
    failure = torch.zeros(2, 40, 40)
    failure[0], failure[1] = 0.9, 0.1
    # a sharp, confident foreground square: the best case
    sharp = torch.zeros(2, 40, 40)
    sharp[0], sharp[1] = 0.02, 0.98
    sharp[0, :10, :], sharp[1, :10, :] = 0.98, 0.02

    bad = image_confidence_scores(failure, [0.1])
    good = image_confidence_scores(sharp, [0.1])

    assert bad["diag_no_present_class"] == 1.0
    assert good["diag_no_present_class"] == 0.0
    diagonal = math.hypot(40, 40)
    for key in (
        "neg_band_width_pmean_sym@0.1",
        "neg_band_width_pmax_sym@0.1",
        "neg_band_width@0.1",
        "neg_r2_pmean_sym@0.1",
        "neg_rM32_pmean@mid",
    ):
        assert bad[key] == pytest.approx(-diagonal), key
        assert bad[key] < good[key], key
    # the area scores and SDC already floor at 0; they must agree in direction
    for key in ("sdc_pmean", "levelset_dice_pmean@0.1", "levelset_iou_pmean@0.1"):
        assert bad[key] == 0.0, key
        assert bad[key] < good[key], key


def test_rank_anchored_threshold_clips_to_the_constant_when_confident():
    prob = _sharp()
    # P95 inside the argmax mask is 0.99, above 1 - alpha, so the min() returns
    # the constant threshold untouched.
    assert rank_anchored_threshold(prob, _predicted(prob), alpha=0.05) == 0.95


def test_rank_anchored_threshold_follows_the_mask_when_diffuse():
    prob = _diffuse()
    # the map never reaches 1 - alpha = 0.9; the clip binds at the mask's own
    # 95th percentile instead
    threshold = rank_anchored_threshold(prob, _predicted(prob), alpha=0.1)
    assert threshold == pytest.approx(0.6)


def test_rank_anchored_threshold_none_without_a_predicted_mask():
    # an absent class has no mask to take the order statistic of
    prob = torch.full((8, 8), 0.2)
    assert rank_anchored_threshold(prob, _predicted(prob), alpha=0.1) is None


def test_rank_anchored_band_equals_the_constant_band_when_confident():
    # The key property of the fix: where the constant threshold is achievable
    # the clip is inactive and every rank-anchored score is bit-for-bit the one
    # it generalizes. The fix is a strict information gain on the degenerate
    # images, not a reparameterization of the score on all of them.
    scores = image_confidence_scores(_foreground_probs(_sharp()), [0.05])
    assert scores["diag_clipped@0.05"] == 0.0
    for aggregation in ("pmax", "pmean"):
        for suffix in ("", "_in", "_sym"):
            key = f"{aggregation}{suffix}@0.05"
            assert (
                scores[f"neg_qband_width_{key}"]
                == scores[f"neg_band_width_{key}"]
            ), key
        for suffix in ("_sym", "_lo", "_hi"):
            key = f"{aggregation}{suffix}@0.05"
            assert scores[f"neg_qr2_{key}"] == scores[f"neg_r2_{key}"], key


def test_rank_anchored_band_is_finite_where_the_constant_one_saturates():
    # The degeneracy: no pixel reaches 1 - alpha = 0.9, so the conservative set
    # is empty, the band has no inner anchor and saturates at the image
    # diagonal -- a constant, carrying no ranking information. The rank-anchored
    # band has an inner anchor by construction and stays strictly finite.
    scores = image_confidence_scores(_foreground_probs(_diffuse()), [0.1])
    diagonal = math.hypot(40, 40)
    assert scores["diag_saturated@0.1"] == 1.0
    assert scores["diag_clipped@0.1"] == 1.0
    assert scores["neg_band_width_pmean_sym@0.1"] == pytest.approx(-diagonal)
    for key in ("neg_qband_width_pmean_sym@0.1", "neg_qr2_pmean_sym@0.1"):
        # strictly finite (a real band width, not the saturating constant) and
        # strictly narrower than the diagonal it replaces
        assert -diagonal < scores[key] < 0.0, key
        assert scores[key] > scores["neg_band_width_pmean_sym@0.1"], key


def test_rank_anchored_band_separates_images_the_constant_one_ties():
    # Two diffuse maps with visibly different bands -- a 3-pixel gap and a
    # 7-pixel one between the confident core and the mask boundary. Both
    # saturate to the same diagonal under the constant threshold, so the score
    # cannot tell them apart at all; the rank-anchored one ranks the tighter
    # band as the more confident. This is the tie the fix exists to break.
    tight = image_confidence_scores(
        _foreground_probs(_diffuse(core=slice(13, 27))), [0.1]
    )
    loose = image_confidence_scores(
        _foreground_probs(_diffuse(core=slice(17, 23))), [0.1]
    )
    key = "neg_band_width_pmean_sym@0.1"
    assert tight[key] == loose[key] == pytest.approx(-math.hypot(40, 40))
    assert (
        tight["neg_qband_width_pmean_sym@0.1"]
        > loose["neg_qband_width_pmean_sym@0.1"]
    )


def test_rank_anchored_nesting_holds_for_binary_maps():
    # The bi-level frame needs Y_hi subset Yhat subset Y_lo. For C = 2 -- and
    # ONLY for C = 2 -- the level set alone secures it: the threshold is a
    # percentile of probabilities that all exceed 1/2 (they won the argmax), so
    # it exceeds 1/2 too and its level set stays inside the prediction. See
    # test_rank_anchored_nesting_needs_the_mask_for_multiclass_maps for what
    # happens when C > 2, which is the case this suite used to have no coverage of.
    for prob in (_sharp(), _diffuse()):
        predicted = _predicted(prob)
        threshold = rank_anchored_threshold(prob, predicted, alpha=0.1)
        conservative = prob >= threshold
        aggressive = prob >= 0.1
        assert conservative.any()  # non-empty by construction: no saturation
        assert torch.all(predicted[conservative])  # Y_hi subset Yhat
        assert torch.all(aggressive[predicted])  # Yhat subset Y_lo


def test_rank_anchored_nesting_needs_the_mask_for_multiclass_maps():
    # C > 2, normalized: the argmax winner's floor is 1/C, so a clipped t_hi can
    # sit below 1/2 and {p >= t_hi} escapes the class's own prediction. The
    # escape is not cosmetic -- the leaked blob is a slab of "conservative" set
    # right underneath the aggressive contour, so the un-nested band reads as
    # MORE confident than the nested one. bilevel_band_widths must therefore be
    # given the mask, which is what image_confidence_scores does whenever the
    # clip is active.
    probs, mask = _softmax_plurality_leak()
    threshold = rank_anchored_threshold(probs[1], mask, alpha=0.1)
    assert threshold == pytest.approx(0.42)  # clipped, and below 1/2
    escaped = probs[1] >= threshold
    assert not torch.all(mask[escaped])  # the leak is real: Y_hi \ Yhat nonempty
    assert torch.all(mask[escaped & mask])  # intersecting restores nesting

    leaked = bilevel_band_widths(probs[1], 0.1, threshold)
    nested = bilevel_band_widths(probs[1], 0.1, threshold, mask)
    assert leaked["outward"] < nested["outward"]  # the leak flatters the image

    # The score must use the nested set. Class 2 is present too, and its mask is
    # flat at 0.45, so its clip collapses and reverts to the constant threshold,
    # which nothing reaches: it contributes the diagonal to the present-class
    # mean. The leaked width would make the image look more confident than that.
    diagonal = math.hypot(40, 40)
    assert rank_anchored_threshold(probs[2], probs.argmax(dim=0) == 2, 0.1) == 0.9
    scores = image_confidence_scores(probs, [0.1])
    assert scores["neg_qband_width_pmean_sym@0.1"] == pytest.approx(
        -(nested["symmetric"] + diagonal) / 2
    )
    assert scores["neg_qband_width_pmean_sym@0.1"] < -(
        leaked["symmetric"] + diagonal
    ) / 2


def test_rank_anchored_keeps_the_constant_convention_on_sigmoid_leaks():
    # CLIPSeg-style: per-class sigmoids, background = 1 - max foreground. Here
    # in-mask p does exceed 1/2, but {p >= 1/2} is NOT the argmax mask -- two
    # prompts can both fire at one pixel and only the max wins. Class 1 clears
    # 1 - alpha inside class 2's mask, so the CONSTANT conservative set already
    # leaks, exactly as it does today. The clip is inactive here, and the fix
    # deliberately does not intersect there, so the two variants stay bit-for-bit
    # equal: any difference between them is attributable to the clip alone.
    class_1, class_2 = torch.zeros(40, 40), torch.zeros(40, 40)
    class_1[5:15, 5:15] = 0.95            # class 1's own mask
    class_1[25:35, 25:35] = 0.95          # co-fires under class 2
    class_2[25:35, 25:35] = 0.97          # ... and loses the argmax there
    foreground = torch.stack([class_1, class_2])
    probs = torch.cat([1 - foreground.max(dim=0).values[None], foreground])
    mask = probs.argmax(dim=0) == 1
    assert not torch.all(mask[class_1 >= 0.9])  # the constant set leaks too
    scores = image_confidence_scores(probs, [0.1])
    assert scores["diag_clipped@0.1"] == 0.0
    for key in ("pmean_sym", "pmax_sym"):
        assert (
            scores[f"neg_qband_width_{key}@0.1"]
            == scores[f"neg_band_width_{key}@0.1"]
        )
        assert scores[f"neg_qr2_{key}@0.1"] == scores[f"neg_r2_{key}@0.1"]


def test_rank_anchored_flat_map_is_not_the_most_confident_image():
    # THE plateau failure. On the repo's canonical maximally-uncertain fixture
    # -- a flat 0.55 map -- the clip lands on 0.55, so Y_hi = Y_lo, every band
    # width is 0 and a negated distance score reads its MAXIMUM: the fuzziest
    # possible image would tie with a perfect one at the top of the ranking.
    # A tie is uninformative; this would be anti-informative. The clip is
    # therefore abandoned on a collapsing band and the constant threshold --
    # which saturates honestly at the diagonal here -- is restored.
    flat = torch.full((32, 32), 0.55)
    scores = image_confidence_scores(_foreground_probs(flat), [0.05])
    diagonal = math.hypot(32, 32)
    assert scores["diag_clip_reverted@0.05"] == 1.0
    assert scores["diag_clipped@0.05"] == 0.0  # reverted, so the variants agree
    for key in ("pmean_sym", "pmax_sym", "pmean", "pmax"):
        assert scores[f"neg_qband_width_{key}@0.05"] == pytest.approx(-diagonal)
        assert (
            scores[f"neg_qband_width_{key}@0.05"]
            == scores[f"neg_band_width_{key}@0.05"]
        )
    assert scores["neg_qr2_pmean_sym@0.05"] == pytest.approx(-diagonal / 2)


def test_rank_anchored_reverts_on_a_plateau_that_the_constant_band_handles():
    # The collapse is not confined to the images the fix targets. This map is
    # NOT degenerate -- a 0.99 core reaches 1 - alpha, so the constant band is
    # well-defined and diag_saturated@ is 0 -- yet the core is only 2% of the
    # mask, so the 95th in-mask percentile lands on the 0.6 plateau and the
    # clipped band collapses to width 0. Reverting must hand back the constant
    # band exactly, not the diagonal and certainly not 0.
    prob = torch.zeros(40, 40)
    prob[10:30, 10:30] = 0.6
    prob[10:13, 10:13] = 0.99  # 9 of 400 mask pixels
    scores = image_confidence_scores(_foreground_probs(prob), [0.1])
    assert scores["diag_saturated@0.1"] == 0.0  # the constant band is fine
    assert scores["diag_clip_reverted@0.1"] == 1.0
    assert scores["diag_clipped@0.1"] == 0.0
    key = "neg_qband_width_pmean_sym@0.1"
    assert scores[key] == scores["neg_band_width_pmean_sym@0.1"]
    assert scores[key] < 0.0  # NOT the zero width of a collapsed band
    assert scores["neg_qr2_pmean_sym@0.1"] == scores["neg_r2_pmean_sym@0.1"]


def test_rank_anchored_reverts_on_a_flat_mask_with_a_skirt():
    # The other way the clip collapses: the aggressive band is populated (a 0.3
    # skirt sits in [alpha, t_hi)), so the *band* does not vanish, but the mask
    # itself is flat, so Y_hi = {p >= t_hi} & Yhat is the whole prediction and
    # the r_2 conservative term HD95(Y_hi, Yhat) is 0 -- the estimator stops
    # probing the upper half of [0, 1] that this map never reaches, which is
    # precisely where its uncertainty lives.
    prob = torch.zeros(40, 40)
    prob[6:34, 6:34] = 0.3
    prob[10:30, 10:30] = 0.6  # flat mask, no interior structure
    scores = image_confidence_scores(_foreground_probs(prob), [0.1])
    diagonal = math.hypot(40, 40)
    assert scores["diag_clip_reverted@0.1"] == 1.0
    assert scores["diag_saturated@0.1"] == 1.0  # nothing reaches 0.9
    assert scores["neg_qr2_pmean_hi@0.1"] == pytest.approx(-diagonal)
    assert scores["neg_qband_width_pmean_sym@0.1"] == pytest.approx(-diagonal)


def test_rank_anchored_threshold_never_falls_below_alpha():
    # t_hi = min(1 - alpha, P95) has no lower clamp, and on a 21-class softmax
    # the argmax floor is 1/21 = 0.048 -- below the deployed alpha = 0.1. An
    # unguarded clip would then put Y_lo strictly INSIDE Y_hi and return a
    # "width" that is not a band width at all, moving the image from the
    # confidence floor to a confident rank. Note max(alpha, .) is NOT the fix:
    # it gives Y_hi == Y_lo, hence width 0, hence maximal confidence.
    logits = torch.zeros(21, 40, 40)
    logits[7, 10:30, 10:30] = 0.3   # a plurality winner at p = 0.063
    logits[7, 18:22, 18:22] = 3.0   # a 1% core above alpha
    probs = logits.softmax(dim=0)
    mask = probs.argmax(dim=0) == 7
    assert mask.any() and torch.allclose(probs.sum(dim=0), torch.ones(40, 40))
    values = probs[7][mask]
    raw = torch.quantile(values, 0.95).item()
    assert raw < 0.1  # the unguarded clip really does invert the level sets
    threshold = rank_anchored_threshold(probs[7], mask, alpha=0.1)
    assert 0.1 < threshold <= 0.9
    assert threshold == pytest.approx(0.9)  # abandoned, back to the constant
    scores = image_confidence_scores(probs, [0.1])
    for key in ("neg_qband_width_pmean_sym@0.1", "neg_qr2_pmean_sym@0.1"):
        assert scores[key] == scores[key.replace("_q", "_")], key


def test_rank_anchored_scores_orientation():
    sharp = torch.zeros(40, 40)
    sharp[10:30, 10:30] = 1.0
    sharp_scores = image_confidence_scores(_foreground_probs(sharp), [0.1])
    diffuse_scores = image_confidence_scores(_foreground_probs(_diffuse()), [0.1])
    for key in (
        "neg_qband_width_pmean_sym@0.1",
        "neg_qband_width_pmax_sym@0.1",
        "neg_qband_width_pmean@0.1",
        "neg_qband_width_pmean_in@0.1",
        "neg_qr2_pmean_sym@0.1",
        "neg_qr2_pmax_sym@0.1",
    ):
        assert sharp_scores[key] > diffuse_scores[key], key
    assert all(isinstance(value, float) for value in diffuse_scores.values())


def test_rank_anchored_scores_saturate_when_no_class_is_predicted():
    # A total detection failure has no predicted mask to anchor to, so the
    # rank-anchored scores fall back to the same worst case as the ones they
    # generalize -- the image diagonal, never 0, which would make the worst
    # image of the split the most confident. See
    # test_detection_failure_is_least_confident_not_most.
    prob = torch.full((8, 8), 0.2)  # nothing wins the argmax
    scores = image_confidence_scores(_foreground_probs(prob), [0.1])
    diagonal = math.hypot(8, 8)
    assert scores["diag_no_present_class"] == 1.0
    # no mask, so no order statistic: the constant convention is used throughout
    # and the two variants agree (they agree on the diagonal, the worst score).
    assert scores["diag_clipped@0.1"] == 0.0
    assert scores["diag_clip_reverted@0.1"] == 1.0
    for key in (
        "neg_qband_width_pmean_sym@0.1",
        "neg_qband_width_pmax_sym@0.1",
        "neg_qr2_pmean_sym@0.1",
        "neg_qr2_pmax_sym@0.1",
    ):
        assert scores[key] == pytest.approx(-diagonal), key


def test_nesting_leak_diagnostic_separates_softmax_from_sigmoid():
    """The conservative set can escape its own class's mask only under sigmoids.

    A normalized softmax cannot leak: two classes cannot both clear
    1 - alpha = 0.9 on the simplex, so {p_c >= 0.9} always wins the argmax.
    CLIPSeg scores each prompt with an independent sigmoid and sets background
    to 1 - max_c p_c, so two co-firing prompts can *both* clear 0.9 while only
    one takes the argmax -- and the loser's conservative set then lies wholly
    outside its own prediction. This is the hypothesised mechanism behind the
    band's losses on CLIPSeg/VOC, and the diagnostic is what would confirm it.
    """
    # softmax: two classes, only one can exceed 0.9 anywhere
    background = torch.full((24, 24), 0.02)
    class_1, class_2 = torch.full((24, 24), 0.95), torch.full((24, 24), 0.03)
    class_1[12:, :], class_2[12:, :] = 0.03, 0.95
    softmax = torch.stack([background, class_1, class_2])
    assert torch.allclose(softmax.sum(dim=0), torch.ones(24, 24))
    assert image_confidence_scores(softmax, [0.1])["diag_nesting_leak@0.1"] == 0.0

    # CLIPSeg-style sigmoid: both prompts fire at 0.95/0.93, background is
    # 1 - max(fg), and class 2 loses the argmax everywhere it fires
    class_1, class_2 = torch.full((24, 24), 0.95), torch.full((24, 24), 0.93)
    sigmoid = torch.stack([1 - torch.maximum(class_1, class_2), class_1, class_2])
    scores = image_confidence_scores(sigmoid, [0.1])
    # THE TOTAL LEAK, and the diagnostic must count it. Class 2 clears 0.9 over
    # the whole frame and wins the argmax NOWHERE, so its conservative set lies
    # entirely outside its (empty) prediction: it is 100% leaked. This is the
    # *extreme* case of the very mechanism the diagnostic names -- the co-firing
    # loser whose Y_hi escapes its own Yhat -- so restricting the loop to the
    # classes present in the argmax (as this diagnostic once did) would drop the
    # worst offenders from both numerator and denominator and report a leak of
    # zero on the images that leak most. Class 1's 576 conservative pixels all sit
    # inside its prediction; class 2's 576 all sit outside its own. 576/1152 = 1/2.
    assert scores["diag_nesting_leak@0.1"] == pytest.approx(0.5)
    # Make class 2 present somewhere as well -- the leak is unchanged. Both
    # prompts clear 0.9 over the whole 24x24 frame, so both conservative sets are
    # the full frame (576 px each, 1152 kept). Class 2 wins the argmax only on its
    # 4x4 corner, so 576 - 16 = 560 of its conservative pixels escape; class 1
    # wins everywhere else, so it loses exactly that corner's 16. The leak is
    # therefore (560 + 16) / 1152 = exactly one half.
    class_2[:4, :4] = 0.99
    sigmoid = torch.stack([1 - torch.maximum(class_1, class_2), class_1, class_2])
    scores = image_confidence_scores(sigmoid, [0.1])
    assert scores["diag_nesting_leak@0.1"] == pytest.approx(0.5)

    # ...and the leak manufactures *false confidence*: the escaped pixels sit in
    # Y_lo and Y_hi alike, so the two contours coincide and the band vanishes.
    # Intersecting Y_hi with the prediction restores the nesting and the signal.
    assert scores["neg_band_width_pmean_sym@0.1"] == pytest.approx(0.0)
    assert scores["neg_mband_width_pmean_sym@0.1"] < -1.0


# ---------------------------------------------------------------------------
# Closed properties over the WHOLE score dict.
#
# Everything above pins named keys. The two tests below iterate the dict itself,
# so a score ADDED LATER cannot inherit a broken orientation in silence -- which
# is exactly how the historical detection-failure inversion survived: the guard
# that was written to prevent it enumerated eight keys out of ninety, and every
# score added after the fix went unpinned.
# ---------------------------------------------------------------------------

# The single-pass pixel baselines. These are allowed -- required, even -- to call
# a confidently-empty background confident: they read the probability map and
# nothing else, and a map that is uniformly certain of "background" IS certain.
# Their blindness to detection failure is not a bug, it is the phenomenon the
# paper measures, and a test that forced them to floor would be pinning the
# opposite of the result. Every OTHER score -- band, mband, qband, r2, qr2, the
# rM ladder, the level-set overlaps, SDC, BUC -- is a statement about the
# predicted object, and an image with no predicted object must take its floor.
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

# The two readouts that legitimately TIE (or invert) on the canonical flat fuzzy
# map, and why. They are excluded by NAME PATTERN, decided in advance -- not by
# "whatever happened to fail" -- so a newly added score cannot fall into the
# exclusion by accident.
#
#   ``_lo@``    the r_2 lower term HD95(Y_lo, Yhat). On a flat 0.55 map the
#               aggressive set IS the argmax mask, so the term is exactly 0 --
#               its best value -- while a sharp map with a soft skirt has a
#               genuine Y_lo \ Yhat and scores worse. That is a true fact about
#               the lower term, and it is why r_2 is reported symmetrized.
#   ``_skip_``  the skip convention, which DROPS empty level sets instead of
#               saturating them. It is shipped as the cautionary ablation of the
#               saturating one precisely because it reads a maximally fuzzy map
#               as perfectly confident (see
#               test_quadrature_saturates_empty_level_sets_and_skipping_does_not).
#               Requiring it to be correctly oriented would be requiring it not
#               to be the thing it is kept to demonstrate.
_UNORIENTED_PATTERNS = ("_lo@", "_skip_")


def _is_oriented(key):
    return not any(pattern in key for pattern in _UNORIENTED_PATTERNS)


def _detection_failure(size=40, classes=2):
    """A CONFIDENTLY empty background: no class wins the argmax anywhere.

    p_fg = 0.02 sits below every alpha in the sweep, which matters. The suite's
    older fixture used p_fg = 0.1 with alpha = 0.1, and the band test is
    inclusive (>= alpha), so the whole image landed *inside* the band and
    neg_band_fraction@ floored at -1.0 by pure coincidence -- masking the fact
    that it did not floor at all. A detection failure that is confidently empty is
    a strictly worse image and the honest fixture.
    """
    probs = torch.full((classes, size, size), 0.02)
    probs[0] = 1.0 - 0.02 * (classes - 1)
    return probs


def test_every_score_is_oriented_sharp_over_fuzzy():
    """CLOSED PROPERTY: every score in the dict must prefer the sharp image.

    Iterates the returned dict rather than a hand-written key list, so adding a
    new score with an inverted sign, or a missing negation, fails here rather than
    silently entering the AURC table with its ranking upside down. The only
    exemptions are the two readouts documented above as deliberately degenerate on
    the flat fixture, and they are excluded by a pattern fixed in advance.
    """
    sharp = _foreground_probs(_sharp())
    fuzzy = _foreground_probs(torch.full((40, 40), 0.55))
    sharp_scores = image_confidence_scores(sharp, [0.1])
    fuzzy_scores = image_confidence_scores(fuzzy, [0.1])

    assert set(sharp_scores) == set(fuzzy_scores)
    checked = 0
    for key in sharp_scores:
        if key.startswith("diag_") or not _is_oriented(key):
            continue
        assert sharp_scores[key] > fuzzy_scores[key], key
        checked += 1
    # a guard on the guard: if a refactor renamed the keys, this test must not
    # quietly become vacuous
    assert checked > 60


def test_every_score_floors_on_a_detection_failure():
    """CLOSED PROPERTY: a total detection failure is the worst image, under every
    score that claims to describe the predicted object.

    THE regression test for the bug the paper says inverted its headline result
    (band-vs-SDC went 2/8 -> 6/8 once it was fixed). The bug was a single
    ``... if values else 0.0`` on a NEGATED distance score, which put a total
    detection failure at the score's *maximum*. Its previous guard enumerated
    eight keys, so every score added afterwards -- the whole eq:mband family, the
    whole importance ladder, neg_band_fraction@ -- was unprotected, and
    neg_band_fraction@ was in fact still committing exactly this bug.

    So this iterates the dict, and asserts three things: the failure is no better
    than either reference image, it sits at its score's exact floor, and the floor
    is the right one -- the image diagonal for a negated distance, -1 for a
    negated bounded fraction or Dice loss, 0 for a positively-oriented area score.
    """
    diagonal = math.hypot(40, 40)
    bad = image_confidence_scores(_detection_failure(), [0.1])
    good = image_confidence_scores(_foreground_probs(_sharp()), [0.1])
    fuzzy = image_confidence_scores(
        _foreground_probs(torch.full((40, 40), 0.55)), [0.1]
    )
    assert bad["diag_no_present_class"] == 1.0

    checked = 0
    for key in bad:
        if key.startswith("diag_") or key in _PIXEL_BASELINES:
            continue
        # never better than a real image, sharp or fuzzy
        assert bad[key] <= good[key], key
        assert bad[key] <= fuzzy[key], key
        # ...and pinned to the exact floor its TYPE demands. The three types are
        # distinguished by name, so a new score is forced into one of them: a
        # positively-oriented area score floors at 0, a negated bounded one (a Dice
        # loss, a fraction) at -1, and a negated distance at -diagonal. Getting the
        # branch wrong is itself the bug -- a negated Dice loss that floored at
        # -diagonal, or a negated distance that floored at -1, would both be
        # mis-scaled against the rest of the split.
        if key.startswith(("levelset_", "sdc_")) or key == "buc":
            assert bad[key] == 0.0, key
        elif "_dice" in key or key.startswith("neg_band_fraction@"):
            assert bad[key] == pytest.approx(-1.0), key
        else:
            assert bad[key] == pytest.approx(-diagonal), key
        checked += 1
    assert checked > 70


def test_band_fraction_floors_on_a_confidently_empty_background():
    """neg_band_fraction@ is a NEGATED area score, so 0 is its ceiling, not floor.

    The sibling conventions do not transfer, and that is what caught this one out.
    levelset_*/SDC are oriented so that 0 is their minimum; the band widths are
    negated distances whose floor is -diagonal. neg_band_fraction@ is a negated
    fraction: bounded in [-1, 0], with 0 -- an image with no pixel in
    [alpha, 1 - alpha] at all -- as its MAXIMUM. A confidently-empty background has
    exactly no such pixel, so an unguarded -mean(in_band) ranked the worst image of
    the split as the single most confident one, above every correctly segmented
    image. Its floor is -1.0, the area-type analogue of -diagonal.

    On the real outputs this put the three worst images of deeplabv3-external/Pet
    at ranks 1, 2 and 3 of 3669.
    """
    bad = image_confidence_scores(_detection_failure(), [0.1])
    assert bad["neg_band_fraction@0.1"] == pytest.approx(-1.0)
    # ...and it is genuinely below a real image, not merely at some constant
    crisp = image_confidence_scores(_foreground_probs(_sharp()), [0.1])
    assert bad["neg_band_fraction@0.1"] < crisp["neg_band_fraction@0.1"]
    # the sibling built from the SAME numerator must floor too
    assert bad["neg_band_per_boundary@0.1"] == pytest.approx(-math.hypot(40, 40))


def test_detection_failure_floors_for_multiclass_and_sigmoid_maps():
    """The floor is not a property of C = 2. It must hold for every construction.

    The deployed conditions are a 21-way softmax (VOC) and a per-prompt sigmoid
    (CLIPSeg), and their detection failures look different: the softmax one is a
    confident background row, the sigmoid one is every prompt firing weakly and
    none winning. Neither had coverage; both must floor.
    """
    for name, probs, size in (
        ("softmax C=21", _detection_failure(size=48, classes=21), 48),
        ("sigmoid C=4", _sigmoid_probs(torch.full((3, 48, 48), 0.03)), 48),
    ):
        scores = image_confidence_scores(probs, [0.1])
        diagonal = math.hypot(size, size)
        assert scores["diag_no_present_class"] == 1.0, name
        for key in (
            "neg_band_width_pmean_sym@0.1",
            "neg_mband_width_pmean_sym@0.1",
            "neg_mband_width_pmax_sym@0.1",
            "neg_qband_width_pmean_sym@0.1",
            "neg_r2_pmean_sym@0.1",
            "neg_qr2_pmean_sym@0.1",
            "neg_rM32_pmean@mid",
            "neg_rM2_pmean@gl",
            "neg_rM4_pmean@imp",
            "neg_band_per_boundary@0.1",
        ):
            assert scores[key] == pytest.approx(-diagonal), (name, key)
        assert scores["neg_band_fraction@0.1"] == pytest.approx(-1.0), name
        assert scores["neg_rM32_dice_pmean@mid"] == pytest.approx(-1.0), name
        for key in ("sdc_pmean", "sdc_pmin", "levelset_iou_pmean@0.1"):
            assert scores[key] == 0.0, (name, key)


# ---------------------------------------------------------------------------
# A PRESENT class must never vanish from an aggregation (rem:bounded).
# ---------------------------------------------------------------------------


def _sigmoid_probs(foreground):
    """CLIPSeg's parameterization: independent per-prompt sigmoids, bg = 1 - max."""
    return torch.cat([1 - foreground.max(dim=0, keepdim=True).values, foreground])


def _faint_present_class(size=48):
    """Two present classes, one of which never reaches alpha. Returns (probs, mask).

    THE case eq:nesting's left containment does not cover. The paper writes
    Y_hi subset Yhat subset Y_lo, but Yhat is the ARGMAX mask, and a C-way softmax
    argmax winner needs only p >= 1/C -- 1/21 = 0.048 for VOC, BELOW the deployed
    alpha = 0.1. So a class can win the argmax over a 144-pixel patch (the model
    hallucinating a whole object) while its aggressive set {p >= alpha} is empty.

    Class 1 is a healthy sharp blob; class 2 wins the argmax at a peak probability
    of ~0.051, just over the 1/21 floor.
    """
    logits = torch.zeros(21, size, size)
    logits[0] = 4.0
    logits[1, 10:26, 10:26] = 6.0
    logits[1, 14:22, 14:22] = 9.0
    logits[0, 30:42, 30:42] = 0.0
    logits[2, 30:42, 30:42] = 0.30
    for index in range(3, 21):
        logits[index, 30:42, 30:42] = 0.25
    probs = logits.softmax(dim=0)
    return probs, probs.argmax(dim=0)


def test_a_present_class_below_alpha_saturates_instead_of_vanishing():
    """A hallucinated class must COST the image, not be deleted from its score.

    bilevel_band_widths / bilevel_r2 / bilevel_overlap / quadrature_risks all
    return None on an empty aggressive set, documented as "class confidently
    absent". For the ALL-class band that reading is right. For a class that is
    *present* in the argmax it is exactly wrong, and the aggregations used to
    filter that None out -- deleting the most uncertain class in the image and
    making the image read MORE confident.

    It is reachable because the argmax floor 1/C sits below alpha: here class 2
    wins the argmax on 144 pixels at p ~ 0.051 while {p_2 >= 0.1} is empty. Under
    rem:bounded an empty level set costs a full image diagonal, so the
    present-class mean must be -(w_1 + diag)/2 -- not -w_1, which is what dropping
    the class gives and which is worth 31 free pixels of confidence on this fixture
    for a whole hallucinated object.

    Note diag_no_present_class is 0 here: the image HAS a present class. This is
    the partial-detection-failure survivor of the bug the paper says it fixed, and
    the DEPLOYED score is one of the ones affected.
    """
    probs, prediction = _faint_present_class()
    diagonal = math.hypot(48, 48)
    assert sorted(set(prediction.unique().tolist()) - {0}) == [1, 2]
    assert probs[2].max() < 0.1                    # below alpha ...
    assert probs[2].max() > 1 / 21                 # ... but above the argmax floor
    assert bilevel_band_widths(probs[2], 0.1) is None  # the None that used to vanish

    healthy = bilevel_band_widths(probs[1], 0.1)
    scores = image_confidence_scores(probs, [0.1])
    assert scores["diag_no_present_class"] == 0.0  # the guard does NOT fire
    assert scores["diag_saturated@0.1"] == 0.5     # ...but this one does

    for suffix, direction in (("", "outward"), ("_in", "inward"), ("_sym", "symmetric")):
        expected = -(healthy[direction] + diagonal) / 2
        for family in ("neg_band_width", "neg_mband_width", "neg_qband_width"):
            assert scores[f"{family}_pmean{suffix}@0.1"] == pytest.approx(
                expected
            ), family
        # the max aggregation must take the diagonal outright
        assert scores[f"neg_band_width_pmax{suffix}@0.1"] == pytest.approx(
            -diagonal
        )
    # the area readouts saturate at their own floor (0), not by omission
    overlap = bilevel_overlap(probs[1], 0.1)
    assert scores["levelset_iou_pmean@0.1"] == pytest.approx(overlap["iou"] / 2)
    assert scores["levelset_dice_pmean@0.1"] == pytest.approx(overlap["dice"] / 2)
    # SDC, which iterates the present classes directly and cannot drop one, saw the
    # hallucinated class all along -- which is how the bug stayed hidden: the score
    # that was wrong and the score it is benchmarked against disagreed loudly, and
    # that looked like a result rather than a defect.
    assert scores["sdc_pmin"] < 0.15


def test_the_quadrature_ladder_averages_over_one_class_set():
    """Every rung of the M-ladder must score the SAME classes, or it measures
    bookkeeping instead of node placement.

    quadrature_risks returns None when the class's peak falls below the rule's
    LOWEST node -- and that node is rule-dependent: 0.211 for Gauss-Legendre M=2,
    0.125 for mid-4, but 0.0156 for the dense mid-32 reference. A present class
    peaking at 0.05 is therefore dropped by the cheap rules and kept (and
    saturated) by the dense one, so spearman_vs_dense -- the quadrature ablation's
    headline column -- would be comparing rules computed over different class sets
    on the same image. The saturating convention makes every rung comparable.
    """
    probs, _ = _faint_present_class()
    diagonal = math.hypot(48, 48)
    predicted = probs.argmax(dim=0) == 2
    # the cheap rules genuinely cannot see this class...
    assert quadrature_risks(probs[2], predicted, quadrature_nodes("gl", 2)) is None
    assert quadrature_risks(probs[2], predicted, quadrature_nodes("mid", 4)) is None
    # ...while the dense one does, and charges it the diagonal
    dense = quadrature_risks(probs[2], predicted, quadrature_nodes("mid", 32))
    assert dense["hd95"] > 0.5 * diagonal

    scores = image_confidence_scores(probs, [0.1])
    # every rung is a 2-class mean, so none of them can read 0 (the free-confidence
    # value a dropped class produces), and all sit within a diagonal of each other
    values = [
        scores[f"neg_rM{count}_pmean@{rule}"] for rule, count in QUADRATURE_RULES
    ]
    values += [scores[f"neg_rM{count}_pmean@imp"] for count in (2, 4, 8)]
    assert all(value < -0.4 * diagonal for value in values)
    assert max(values) - min(values) < diagonal


# ---------------------------------------------------------------------------
# eq:mband -- the nesting repair
# ---------------------------------------------------------------------------


def test_masked_band_is_a_bit_for_bit_no_op_where_the_leak_is_zero():
    """eq:mband is claimed to be a PROVABLE no-op wherever the nesting holds.

    The dominance argument the paper makes for the repair has two halves -- "free
    where it does nothing, strictly better where it does something" -- and this is
    the first half. On a normalized map the conservative set {p >= 1 - alpha}
    already lies inside the argmax mask (two classes cannot both clear 0.7 on the
    simplex), so intersecting with the prediction changes nothing, and the paper
    asserts the no-op holds *to machine precision*. So does this test: exact
    equality, at every alpha in the deployed sweep, in every direction, for both
    aggregations, on both a 21-way softmax and a binary map.

    A tolerance here would be the wrong assertion. The two code paths differ only
    by an intersection that is provably the identity, so any difference at all --
    even one ULP -- would mean they are no longer computing the same thing.
    """
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    torch.manual_seed(0)
    maps = {
        "softmax C=21": torch.softmax(torch.randn(21, 64, 64) * 2, dim=0),
        "binary": torch.softmax(torch.randn(2, 48, 48) * 2, dim=0),
    }
    for name, probs in maps.items():
        scores = image_confidence_scores(probs, alphas)
        for alpha in alphas:
            key = f"{alpha:g}"
            assert scores[f"diag_nesting_leak@{key}"] == 0.0, (name, key)
            for aggregation in ("pmean", "pmax"):
                for suffix in ("", "_in", "_sym"):
                    left = f"neg_mband_width_{aggregation}{suffix}@{key}"
                    right = f"neg_band_width_{aggregation}{suffix}@{key}"
                    assert scores[left] == scores[right], (name, left)


def test_masked_band_repairs_the_false_confidence_a_sigmoid_leak_manufactures():
    """The second half of the dominance argument: strictly better where it acts.

    Two co-firing prompts both clear 1 - alpha while only one wins the argmax, so
    the loser's conservative set lies outside its own prediction. The failure mode
    is worse than a broken invariant: the leaked pixels sit in Y_lo and Y_hi alike,
    the two contours COINCIDE, and the band width reads 0 -- its maximum, maximal
    confidence -- on precisely the image where two classes are fighting.
    """
    class_1, class_2 = torch.zeros(40, 40), torch.zeros(40, 40)
    class_1[5:35, 5:35] = 0.96
    class_2[5:35, 5:35] = 0.93
    class_2[:6, :6] = 0.99  # give class 2 an argmax foothold
    scores = image_confidence_scores(
        _sigmoid_probs(torch.stack([class_1, class_2])), [0.1]
    )
    assert scores["diag_nesting_leak@0.1"] > 0.4
    # the plain band vanishes: false confidence
    assert scores["neg_band_width_pmean_sym@0.1"] == pytest.approx(0.0)
    # the repair restores a real, strictly worse, width
    assert scores["neg_mband_width_pmean_sym@0.1"] < -1.0
    assert (
        scores["neg_mband_width_pmean_sym@0.1"]
        < scores["neg_band_width_pmean_sym@0.1"]
    )


def test_nesting_leak_counts_a_class_that_wins_the_argmax_nowhere():
    """The diagnostic must see the WORST offender, not skip it.

    The mechanism it names is "the LOSER's conservative set escapes its own
    prediction" -- and the extreme case of a loser is one that wins the argmax
    nowhere at all, whose conservative set is therefore 100% leaked. A loop over
    the *present* classes drops exactly that case from both numerator and
    denominator and reports a leak of zero on the image that leaks most. The loop
    runs over every foreground class for that reason.

    The published CLIPSeg leak rates were lower bounds on their own quantity.
    """
    class_1, class_2 = torch.full((24, 24), 0.95), torch.full((24, 24), 0.93)
    probs = _sigmoid_probs(torch.stack([class_1, class_2]))
    assert not (probs.argmax(dim=0) == 2).any()   # class 2 wins NOWHERE
    assert (probs[2] >= 0.9).all()                # ...yet its Y_hi is the whole frame
    scores = image_confidence_scores(probs, [0.1])
    assert scores["diag_nesting_leak@0.1"] == pytest.approx(0.5)

    # and the widened loop is a provable no-op on a softmax: a genuinely absent
    # class has an EMPTY conservative set and is skipped, so nothing is added.
    torch.manual_seed(1)
    for _ in range(5):
        softmax = torch.softmax(torch.randn(21, 32, 32) * 3, dim=0)
        leaks = image_confidence_scores(softmax, [0.05, 0.1, 0.3])
        for alpha in ("0.05", "0.1", "0.3"):
            assert leaks[f"diag_nesting_leak@{alpha}"] == 0.0


# ---------------------------------------------------------------------------
# THE TWO FAILED ABLATIONS. Pinned as FAILURES so nobody promotes them.
# ---------------------------------------------------------------------------


def _clipped_field(seed):
    """A smoothed random binary field on which the rank-anchored clip fires."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.random((64, 64)), 5.0)
    field = (field - field.min()) / (field.max() - field.min())
    return torch.tensor((0.05 + 0.9 * field).astype(np.float32))


def test_rank_anchored_clip_is_an_area_quantile_not_a_probability_anchor():
    """FAILED ABLATION 1, pinned by its mechanism: the clip IS a top-k-by-area rule.

    The clip sets t_hi = P_q({p_i : i in Yhat}) with q = 0.95. A percentile of the
    in-mask probability VALUES is, by definition, an AREA quantile of the mask: the
    conservative set it produces is the top 1 - q = 5% of the prediction by pixel
    count, on every image, whatever the probabilities happen to be. The band then
    reads the distance from the skirt to a core of PINNED RELATIVE AREA, whose
    inner term is pure object geometry -- the shape-statistic collapse the whole
    design exists to avoid, arriving by the back door.

    Measured here: |Y_hi & Yhat| / |Yhat| = 0.05 to three decimals, on every field
    where the clip is active. That number is not a coincidence to be explained, it
    is the definition of what the clip computes, and pinning it is what stops
    somebody reading the ablation's tie-breaking ability as an improvement.
    """
    ratios = []
    for seed in range(20):
        foreground = _clipped_field(seed)
        probs = _foreground_probs(foreground)
        mask = probs.argmax(dim=0) == 1
        if not mask.any():
            continue
        threshold = rank_anchored_threshold(foreground, mask, alpha=0.1)
        if threshold is None or threshold >= 0.9:
            continue  # clip inactive or reverted
        conservative = (foreground >= threshold) & mask
        ratios.append(float(conservative.sum()) / float(mask.sum()))

    assert len(ratios) >= 10  # the clip really does fire, on most maps
    assert float(np.median(ratios)) == pytest.approx(0.05, abs=0.02)
    assert max(ratios) < 0.10  # a fixed *area*, never a fixed probability


def test_the_clip_fires_on_healthy_classes_too_so_it_is_not_a_free_fix():
    """FAILED ABLATION 1, pinned by its scope: it rewrites scores that worked.

    The claim the ablation was promoted on was that it is confined to the
    degenerate images -- the ones whose constant band saturates -- and is therefore
    a strict information gain. It is not. The clip fires whenever
    P95(in-mask) < 1 - alpha, a set that strictly CONTAINS the saturated one
    {max p < 1 - alpha}, so on a clipped-but-healthy class it reparameterizes a
    score that was already working. diag_clipped@ counts the classes on which the
    two variants differ; diag_saturated@ counts the ones the fix was for; and the
    first is routinely positive while the second is zero.
    """
    healthy_but_clipped = 0
    for seed in range(20):
        scores = image_confidence_scores(
            _foreground_probs(_clipped_field(seed)), [0.1]
        )
        # the invariant: the clip is at least as broad as the degeneracy
        assert scores["diag_clipped@0.1"] + scores["diag_clip_reverted@0.1"] >= (
            scores["diag_saturated@0.1"]
        ), seed
        if scores["diag_saturated@0.1"] == 0.0 and scores["diag_clipped@0.1"] > 0.0:
            healthy_but_clipped += 1
            # the constant band is fine here, and the clip changes it anyway
            assert (
                scores["neg_qband_width_pmean_sym@0.1"]
                != scores["neg_band_width_pmean_sym@0.1"]
            )
    assert healthy_but_clipped >= 3


def test_importance_nodes_are_worse_than_equispaced_at_low_m():
    """FAILED ABLATION 2, pinned by its mechanism: the nodes land where H vanishes.

    Placing the nodes at the quantiles of the image's own probability histogram is
    textbook importance sampling and is provably estimand-preserving (the gap
    weights are exactly the ones that leave int H invariant) -- which is what makes
    the failure interesting rather than merely careless. It fails anyway, and worse
    than the equispaced midpoint rule at every M below 16, because the probability
    values concentrate near the integrand's VERTEX, where H is smallest, while the
    integral's mass lives in the tails. So the quantile nodes are drawn precisely
    to the region that contributes nothing.

    Both the outcome and the mechanism are pinned. The outcome: the error against a
    dense reference is larger at M = 2, 4 and 8. The mechanism, stated as a
    measurable quantity: the integrand is SYSTEMATICALLY SMALLER at the nodes the
    importance rule chooses than at the nodes the midpoint rule chooses, at every
    M. That is what "the nodes land where H vanishes" means operationally, and it
    is the sentence the ablation exists to make unforgettable.
    """
    rng = np.random.default_rng(11)
    field = ndimage.gaussian_filter(rng.random((48, 48)), 4.0)
    field = (field - field.min()) / (field.max() - field.min())
    prob = (0.02 + 0.96 * field).astype(np.float32)
    predicted = prob >= 0.5
    tensor, mask = torch.tensor(prob), torch.tensor(predicted)

    reference = quadrature_risks(tensor, mask, quadrature_nodes("mid", 128))["hd95"]
    for count in (2, 4, 8):
        equispaced_nodes = quadrature_nodes("mid", count)
        adaptive_nodes, weights = importance_nodes(prob, count)

        # THE OUTCOME: worse than equispaced, at every M below 16.
        equispaced = quadrature_risks(tensor, mask, equispaced_nodes)["hd95"]
        adaptive = quadrature_risks(tensor, mask, adaptive_nodes, weights)["hd95"]
        assert abs(adaptive - reference) > abs(equispaced - reference), count

        # THE MECHANISM: the adaptive nodes sit where the integrand is smaller.
        heights = lambda nodes: np.mean(  # noqa: E731
            _level_set_losses(prob, predicted, list(nodes))[0]
        )
        assert heights(adaptive_nodes) < heights(equispaced_nodes), count

    # ...and the reason it happens: H vanishes at the vertex and grows toward both
    # tails, while the probability values -- whose quantiles the nodes chase --
    # concentrate at that same vertex. Oversampling the mass is oversampling zero.
    heights = {
        t: _level_set_losses(prob, predicted, [t])[0][0]
        for t in (0.25, 0.5, 0.75, 0.98)
    }
    assert heights[0.5] == 0.0
    assert heights[0.25] > 5.0 and heights[0.98] > 5.0
    assert 0.25 < float(np.median(prob)) < 0.75


def test_importance_nodes_still_reach_both_tails_and_the_empty_mask_atom():
    """A failed ablation must fail for the reason we STATE it fails for.

    The rule used to have a second, avoidable defect on top of the interesting
    one: the bin EDGES were anchored at 0 and 1, so the top bin correctly carried
    the atom Q_p places on the empty mask (weight 1 - max_i p_i), but the NODES
    were quantiles of the map's own values, so the top node could never exceed
    max_i p_i and the empty level set was never evaluated. The diagonal that atom
    is worth was silently replaced by H({p >= max p}) ~ 0.

    That made the estimator INCONSISTENT -- the error plateaued at
    (1 - max p) * diag instead of vanishing as M grew -- and it inflated the very
    failure magnitudes the paper quotes, misattributing the ablation's collapse to
    node placement alone. Taking the nodes from the same anchored edges fixes it.
    The ablation still loses (the test above), which is the point: it now loses
    honestly.

    The fixture is a DIFFUSE map (max p = 0.57), which is where this matters and
    where CLIPSeg lives: Q_p then puts mass 1 - 0.57 = 0.43 on the empty mask,
    worth 0.43 diagonals -- most of the true risk. On a saturated map (max p =
    0.98) the atom is worth almost nothing and the old bug was almost invisible,
    which is why it survived.
    """
    prob = np.clip(
        np.random.default_rng(3).random((24, 24)) * 0.55 + 0.02, 0, 0.60
    ).astype(np.float32)
    predicted = prob >= 0.5
    tensor, mask = torch.tensor(prob), torch.tensor(predicted)
    assert prob.max() < 0.6  # a big empty-mask atom: 1 - max p > 0.4

    # the exact integral, by the piecewise-constant identity of prop:integral
    values = np.unique(prob)
    edges = np.concatenate([[0.0], values, [1.0]])
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    distances, _, _ = _level_set_losses(prob, predicted, list(midpoints))
    exact = float(np.diff(edges) @ np.array(distances))

    errors = []
    for count in (2, 8, 64, 512):
        nodes, weights = importance_nodes(prob, count)
        assert sum(weights) == pytest.approx(1.0)
        # the top node clears max p, so the EMPTY level set -- the atom the top
        # bin's weight stands for -- is actually evaluated, at every M. It used to
        # be unreachable at every M, because the nodes were value quantiles.
        assert max(nodes) > prob.max()
        risk = quadrature_risks(tensor, mask, nodes, weights)
        errors.append(abs(risk["hd95"] - exact))
    # CONSISTENT: the error vanishes with M rather than plateauing
    assert errors[-1] < 0.01 * errors[0]
    assert errors[-1] < 0.1

    # a degenerate (single-valued) map needs no special case, and must not be
    # charged the detection-failure floor: a fully saturated p = 1 map is the most
    # confident map there is, and used to score -diagonal here, i.e. identically to
    # a total detection failure.
    saturated = image_confidence_scores(
        torch.stack([torch.zeros(32, 32), torch.ones(32, 32)]), [0.1]
    )
    for count in (2, 4, 8):
        assert saturated[f"neg_rM{count}_pmean@imp"] == pytest.approx(0.0)
        assert saturated[f"neg_rM{count}_dice_pmean@imp"] == pytest.approx(0.0)


def test_no_failed_ablation_is_the_deployed_default():
    """The deployed score is the constant-threshold band, and stays that way.

    Both ablations above are shipped -- they are the paper's two negative results
    and deleting them would delete the evidence. That makes a guard necessary: the
    default must not quietly become one of them.
    """
    import scripts.analyze_selective as analyze

    assert analyze.DEFAULT_BAND == "neg_band_width_pmean_sym@0.1"
    assert not analyze.DEFAULT_BAND.startswith("neg_qband")   # rank-anchored
    assert "@imp" not in analyze.DEFAULT_BAND                 # importance nodes
    assert "_skip" not in analyze.DEFAULT_BAND                # the skip convention
    # the ablations are still reported, as ablations
    assert analyze.RANK_ANCHORED_BAND in analyze.QUADRATURE_LADDER
    assert analyze.DEFAULT_BAND in analyze.QUADRATURE_LADDER
