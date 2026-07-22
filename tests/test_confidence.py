"""Tests for the strict risk-aligned binary selective-risk framework."""

import math

import numpy as np
import pytest

from selectseg.confidence import (
    foreground_dice_loss,
    levelset_risk,
    midpoint_rule,
    midpoint_loss_indexed_confidences,
    normalized_penalized_hd95,
    paired_cluster_bootstrap_aurc_difference,
    soft_dice_confidence,
    summarize_aurc,
    tie_aware_expected_aurc,
    validate_quadrature,
)


def test_foreground_dice_loss_empty_conventions_and_hand_case():
    empty = np.zeros((2, 3), dtype=bool)
    one = empty.copy()
    one[0, 0] = True
    assert foreground_dice_loss(empty, empty) == 0.0
    assert foreground_dice_loss(empty, one) == 1.0
    assert foreground_dice_loss(one, empty) == 1.0

    label = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    prediction = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
    assert foreground_dice_loss(label, prediction) == pytest.approx(0.5)


def test_binary_losses_validate_masks_and_shapes():
    with pytest.raises(ValueError, match="only binary"):
        foreground_dice_loss(np.array([[0, 2]]), np.array([[0, 1]]))
    with pytest.raises(ValueError, match="shapes differ"):
        normalized_penalized_hd95(
            np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool)
        )
    with pytest.raises(ValueError, match="2D"):
        foreground_dice_loss(np.zeros(3), np.zeros(3))


def test_normalized_penalized_hd95_empty_identical_and_shifted_cases():
    empty = np.zeros((6, 8), dtype=bool)
    first = empty.copy()
    second = empty.copy()
    first[2, 2] = True
    second[2, 4] = True

    assert normalized_penalized_hd95(empty, empty) == 0.0
    assert normalized_penalized_hd95(empty, first) == 1.0
    assert normalized_penalized_hd95(first, empty) == 1.0
    assert normalized_penalized_hd95(first, first) == 0.0
    assert normalized_penalized_hd95(first, second) == pytest.approx(
        2 / math.hypot(6, 8)
    )
    assert normalized_penalized_hd95(second, first) == pytest.approx(
        normalized_penalized_hd95(first, second)
    )


def test_midpoint_rule_and_generic_levelset_risk_hand_computation():
    probability = np.array([[0.9, 0.6], [0.4, 0.1]])
    hard_prediction = probability > 0.5
    # At t=.25: loss=1-4/5=.2. At t=.75: loss=1-2/3=1/3.
    risk = levelset_risk(
        probability,
        hard_prediction,
        foreground_dice_loss,
        nodes=[0.25, 0.75],
        weights=[0.25, 0.75],
    )
    assert risk == pytest.approx(0.25 * 0.2 + 0.75 / 3)

    nodes2, weights2 = midpoint_rule(2)
    nodes32, weights32 = midpoint_rule(32)
    assert nodes2 == pytest.approx([0.25, 0.75])
    assert weights2 == pytest.approx([0.5, 0.5])
    assert len(nodes32) == len(weights32) == 32
    # M changes only the nodes and weights; the probability, action and loss stay fixed.
    for nodes, weights in ((nodes2, weights2), (nodes32, weights32)):
        value = levelset_risk(
            probability,
            hard_prediction,
            normalized_penalized_hd95,
            nodes,
            weights,
        )
        assert 0 <= value <= 1


@pytest.mark.parametrize(
    ("nodes", "weights", "message"),
    [
        ([], None, "non-empty"),
        ([0.0, 0.5], None, "strictly inside"),
        ([0.5, 1.0], None, "strictly inside"),
        ([0.6, 0.4], None, "strictly increasing"),
        ([0.4, 0.4], None, "strictly increasing"),
        ([0.25, 0.75], [1.0], "aligned"),
        ([0.25, 0.75], [1.1, -0.1], "non-negative"),
        ([0.25, 0.75], [0.2, 0.2], "sum to one"),
    ],
)
def test_quadrature_validation_rejects_invalid_rules(nodes, weights, message):
    with pytest.raises(ValueError, match=message):
        validate_quadrature(nodes, weights)


def test_levelset_risk_validates_probability_shape_and_loss_output():
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        levelset_risk(np.full((2, 2), 1.1), mask, foreground_dice_loss, [0.5])
    with pytest.raises(ValueError, match="shapes differ"):
        levelset_risk(np.zeros((3, 2)), mask, foreground_dice_loss, [0.5])
    with pytest.raises(ValueError, match="one scalar"):
        levelset_risk(np.zeros((2, 2)), mask, lambda _a, _b: [0, 1], [0.5])
    with pytest.raises(ValueError, match="non-finite"):
        levelset_risk(np.zeros((2, 2)), mask, lambda _a, _b: np.nan, [0.5])


def test_levelset_risk_handles_empty_empty_per_node_not_by_prediction_floor():
    probability = np.full((4, 4), 0.1)
    hard_prediction = np.zeros((4, 4), dtype=bool)
    # Low node gives nonempty-vs-empty (loss 1); high node gives empty-empty (0).
    nodes = [0.05, 0.5]
    assert levelset_risk(
        probability, hard_prediction, foreground_dice_loss, nodes
    ) == pytest.approx(0.5)
    assert levelset_risk(
        probability, hard_prediction, normalized_penalized_hd95, nodes
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(4))
def test_cached_midpoint_ladder_matches_generic_losses_exactly(seed):
    rng = np.random.default_rng(seed)
    probability = rng.random((15, 17))
    prediction = probability >= (0.35 + 0.1 * seed)
    cached = midpoint_loss_indexed_confidences(
        probability, prediction, counts=(2, 8, 32)
    )
    for count in (2, 8, 32):
        nodes, weights = midpoint_rule(count)
        assert cached[count]["dice"] == pytest.approx(
            -levelset_risk(
                probability,
                prediction,
                foreground_dice_loss,
                nodes,
                weights,
            )
        )
        assert cached[count]["nhd95"] == pytest.approx(
            -levelset_risk(
                probability,
                prediction,
                normalized_penalized_hd95,
                nodes,
                weights,
            )
        )


def test_cached_midpoint_ladder_preserves_per_node_empty_conventions():
    probability = np.full((4, 4), 0.1)
    prediction = np.zeros((4, 4), dtype=bool)
    cached = midpoint_loss_indexed_confidences(
        probability, prediction, counts=(2,)
    )
    # Both midpoint nodes exceed 0.1: every level set and the action are empty.
    assert cached[2] == {"dice": 0.0, "nhd95": 0.0}
    with pytest.raises(ValueError, match="non-empty"):
        midpoint_loss_indexed_confidences(probability, prediction, counts=())
    with pytest.raises(ValueError, match="duplicates"):
        midpoint_loss_indexed_confidences(probability, prediction, counts=(2, 2))


def test_soft_dice_confidence_hand_case_and_empty_convention():
    probability = np.array([[0.8, 0.6], [0.4, 0.0]])
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    assert soft_dice_confidence(probability, prediction) == pytest.approx(2.8 / 3.8)
    assert soft_dice_confidence(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_tie_aware_expected_aurc_is_exact_and_row_order_invariant():
    confidences = np.array([1.0, 1.0, 0.0])
    risks = np.array([0.0, 2.0, 3.0])
    # Average of the two tied orders: (8/9 + 14/9) / 2 = 11/9.
    assert tie_aware_expected_aurc(confidences, risks) == pytest.approx(11 / 9)
    permutation = [1, 0, 2]
    assert tie_aware_expected_aurc(
        confidences[permutation], risks[permutation]
    ) == pytest.approx(11 / 9)


def test_aurc_summary_reports_oracle_random_excess_and_normalized_values():
    risks = np.array([0.0, 1.0])
    perfect = summarize_aurc([1.0, 0.0], risks)
    assert perfect.aurc == pytest.approx(0.25)
    assert perfect.oracle_aurc == pytest.approx(0.25)
    assert perfect.random_aurc == pytest.approx(0.5)
    assert perfect.excess_aurc == pytest.approx(0.0)
    assert perfect.normalized_aurc == pytest.approx(0.0)

    reversed_score = summarize_aurc([0.0, 1.0], risks)
    assert reversed_score.aurc == pytest.approx(0.75)
    assert reversed_score.excess_aurc == pytest.approx(0.5)
    assert reversed_score.normalized_aurc == pytest.approx(2.0)

    unidentifiable = summarize_aurc([3.0, 2.0, 1.0], [0.4, 0.4, 0.4])
    assert unidentifiable.normalized_aurc is None


def test_paired_cluster_bootstrap_is_paired_clustered_and_reproducible():
    risks = np.array([0.0, 1.0, 0.8, 0.2, 0.4, 0.6])
    left = np.array([1.0, 0.0, 0.1, 0.9, 0.8, 0.2])
    right = np.array([0.8, 0.2, 0.9, 0.1, 0.3, 0.7])
    image_ids = ["a", "a", "b", "b", "c", "c"]
    first = paired_cluster_bootstrap_aurc_difference(
        left,
        right,
        risks,
        cluster_ids=image_ids,
        n_resamples=250,
        seed=17,
    )
    second = paired_cluster_bootstrap_aurc_difference(
        left,
        right,
        risks,
        cluster_ids=image_ids,
        n_resamples=250,
        seed=17,
    )
    assert first == second
    assert first.difference == pytest.approx(
        tie_aware_expected_aurc(left, risks)
        - tie_aware_expected_aurc(right, risks)
    )
    assert first.n_observations == 6
    assert first.n_clusters == 3
    assert first.n_resamples == 250
    assert first.seed == 17
    assert first.ci_low <= first.ci_high


def test_paired_cluster_bootstrap_identical_scores_have_zero_interval():
    confidence = [0.9, 0.9, 0.2, 0.1]
    result = paired_cluster_bootstrap_aurc_difference(
        confidence,
        confidence,
        [0.0, 1.0, 0.3, 0.8],
        cluster_ids=[10, 10, 20, 20],
        n_resamples=50,
        seed=3,
    )
    assert result.difference == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0


def test_paired_cluster_bootstrap_accepts_hashable_tuple_image_ids():
    result = paired_cluster_bootstrap_aurc_difference(
        [0.9, 0.8, 0.2],
        [0.8, 0.9, 0.2],
        [0.0, 0.3, 0.8],
        cluster_ids=[("dataset", 1), ("dataset", 1), ("dataset", 2)],
        n_resamples=10,
        seed=1,
    )
    assert result.n_clusters == 2


def test_paired_cluster_bootstrap_validates_cluster_metadata():
    with pytest.raises(ValueError, match="match the data length"):
        paired_cluster_bootstrap_aurc_difference(
            [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], cluster_ids=["one"]
        )
    with pytest.raises(ValueError, match="missing"):
        paired_cluster_bootstrap_aurc_difference(
            [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], cluster_ids=["one", None]
        )
    with pytest.raises(ValueError, match="positive integer"):
        paired_cluster_bootstrap_aurc_difference(
            [1.0], [1.0], [0.0], n_resamples=0
        )
