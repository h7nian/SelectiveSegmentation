import itertools

import numpy as np
import pytest

from selectseg.confidence import foreground_dice_loss, midpoint_rule
from selectseg.counts import (
    action_two_block_dice_confidence,
    count_ladders,
    second_order_dice_similarity,
    shared_threshold_dice_confidence,
)


def test_shared_threshold_matches_direct_mask_evaluation():
    probability = np.array([[0.1, 0.4], [0.6, 0.9]])
    action = probability >= 0.5
    nodes, _ = midpoint_rule(8)
    expected = -np.mean(
        [foreground_dice_loss(probability >= node, action) for node in nodes]
    )
    assert shared_threshold_dice_confidence(probability, action, m=8) == pytest.approx(
        expected
    )


def test_two_block_matches_cartesian_mask_evaluation():
    probability = np.array([[0.05, 0.35, 0.49], [0.51, 0.72, 0.98]])
    action = probability >= 0.5
    nodes, _ = midpoint_rule(4)
    losses = []
    for inside, outside in itertools.product(nodes, repeat=2):
        candidate = np.where(action, probability >= inside, probability >= outside)
        losses.append(foreground_dice_loss(candidate, action))
    assert action_two_block_dice_confidence(
        probability, action, m=4
    ) == pytest.approx(-np.mean(losses))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_empty_and_full_constant_maps_are_total(value):
    probability = np.full((3, 4), value)
    action = probability >= 0.5
    assert shared_threshold_dice_confidence(probability, action) == 0.0
    assert action_two_block_dice_confidence(probability, action) == 0.0


def test_count_ladders_partition_every_candidate_count():
    probability = np.array([[0.1, 0.4], [0.6, 0.9]])
    action = np.array([[False, True], [True, False]])
    overlap, outside = count_ladders(probability, action, m=2)
    nodes, _ = midpoint_rule(2)
    for index, node in enumerate(nodes):
        candidate = probability >= node
        assert overlap[index] == np.logical_and(candidate, action).sum()
        assert outside[index] == np.logical_and(candidate, ~action).sum()


def test_second_order_reduces_to_soft_dice_when_variances_are_zero():
    assert second_order_dice_similarity(10, 8, 2, 0, 0, 0) == pytest.approx(
        16 / 20
    )


def test_inputs_fail_closed():
    probability = np.ones((2, 2))
    with pytest.raises(TypeError):
        action_two_block_dice_confidence(probability, np.ones((2, 2)))
    probability[0, 0] = np.nan
    with pytest.raises(ValueError):
        shared_threshold_dice_confidence(probability, np.ones((2, 2), dtype=bool))
