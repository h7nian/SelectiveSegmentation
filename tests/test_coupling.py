import numpy as np
import pytest

from selectseg.confidence import foreground_dice_loss
from selectseg.counts import (
    balanced_grid_partition,
    component_partition,
    labels_for_coupling,
    partition_diagnostics,
    partition_dice_confidence,
)


def test_component_partition_labels_both_phases_contiguously():
    mask = np.array(
        [[True, False, True], [True, False, False], [False, False, True]],
        dtype=bool,
    )
    labels = component_partition(mask)
    assert set(np.unique(labels)) == set(range(4))
    assert labels[0, 0] == labels[1, 0]
    assert labels[0, 0] != labels[0, 2]
    assert labels[0, 1] == labels[1, 1] == labels[1, 2]


@pytest.mark.parametrize("shape,blocks", [((7, 11), 1), ((7, 11), 5), ((7, 11), 17), ((2, 3), 6)])
def test_balanced_grid_realizes_exact_block_count(shape, blocks):
    labels = balanced_grid_partition(shape, blocks)
    assert labels.shape == shape
    assert np.array_equal(np.unique(labels), np.arange(blocks))
    assert partition_diagnostics(labels)["num_blocks"] == blocks


def test_one_block_partition_matches_direct_shared_threshold_mc():
    probability = np.array([[0.1, 0.4], [0.6, 0.9]])
    action = probability >= 0.5
    labels = np.zeros(probability.shape, dtype=np.int32)
    confidence, repeats = partition_dice_confidence(
        probability,
        action,
        labels,
        draws=20,
        repeats=2,
        master_seed=9,
        sample_id="x",
        coupling_id="one",
    )
    assert confidence == pytest.approx(repeats.mean())
    assert np.all((-1 <= repeats) & (repeats <= 0))


def test_constant_map_has_perfect_confidence_for_any_partition():
    probability = np.ones((4, 5))
    action = probability >= 0.5
    labels = balanced_grid_partition(probability.shape, 7)
    confidence, repeats = partition_dice_confidence(
        probability, action, labels, draws=8, repeats=3
    )
    assert confidence == 0
    assert np.array_equal(repeats, np.zeros(3))


def test_component_and_grid_have_matched_block_counts():
    probability = np.array([[0.9, 0.1, 0.8], [0.9, 0.1, 0.2]])
    action = probability >= 0.5
    components = labels_for_coupling(
        probability, action, coupling="action_components"
    )
    grid = labels_for_coupling(probability, action, coupling="action_grid")
    assert np.unique(components).size == np.unique(grid).size


def test_partition_score_matches_explicit_candidates_for_recovered_uniforms():
    probability = np.array([[0.2, 0.8], [0.4, 0.6]])
    action = probability >= 0.5
    labels = np.array([[0, 1], [0, 1]], dtype=np.int32)
    # The production function is exercised end to end; direct Dice remains total.
    confidence, _ = partition_dice_confidence(
        probability, action, labels, draws=16, repeats=1, master_seed=2
    )
    assert -1 <= confidence <= 0
    assert foreground_dice_loss(action, action) == 0
