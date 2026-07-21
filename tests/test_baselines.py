"""Hand-computed tests for the strong single-map binary baselines."""

import numpy as np
import pytest

from selectseg.baselines import (
    _maximum_valid_patch_mean,
    binary_entropy_bits,
    exact_levelset_dice_confidence,
    strong_binary_confidences,
)


def test_binary_entropy_bits_has_exact_endpoint_and_midpoint_values():
    entropy = binary_entropy_bits(np.array([0.0, 0.5, 1.0]))
    assert entropy == pytest.approx([0.0, 1.0, 0.0])


def test_qfr_uses_predicted_fraction_quantile_and_includes_ties():
    probability = np.linspace(0.01, 0.49, 100).reshape(10, 10)
    prediction = np.zeros((10, 10), dtype=bool)
    prediction.flat[:20] = True
    entropy = binary_entropy_bits(probability)
    threshold = np.quantile(entropy, 0.8)
    expected = -entropy[entropy >= threshold].mean()
    scores = strong_binary_confidences(probability, prediction)
    assert scores["confidence_qfr_entropy"] == pytest.approx(expected)


def test_qfr_empty_prediction_retains_the_maximum_entropy_ties():
    probability = np.full((10, 10), 0.1)
    prediction = np.zeros((10, 10), dtype=bool)
    scores = strong_binary_confidences(probability, prediction)
    assert scores["confidence_qfr_entropy"] == pytest.approx(
        -binary_entropy_bits(probability).max()
    )
    assert scores["confidence_foreground_entropy"] == -1.0


def test_mmmc_matches_parenthesized_entropy_formula_and_zero_endpoint():
    probability = np.linspace(0.0, 0.5, 100).reshape(10, 10)
    prediction = probability >= 0.3
    entropy = binary_entropy_bits(probability)
    expected = -(np.median(entropy) + entropy.min()) / entropy.max()
    scores = strong_binary_confidences(probability, prediction)
    assert scores["confidence_mmmc_entropy"] == pytest.approx(expected)

    sharp = strong_binary_confidences(
        np.zeros((10, 10)), np.zeros((10, 10), dtype=bool)
    )
    assert sharp["confidence_mmmc_entropy"] == 0.0


def test_plm_uses_maximum_valid_stride_one_patch_mean():
    values = np.zeros((11, 12))
    values[:10, 1:11] = 1
    assert _maximum_valid_patch_mean(values, 10) == pytest.approx(1.0)

    probability = np.full((11, 12), 0.5)
    scores = strong_binary_confidences(probability, probability >= 0.5)
    assert scores["confidence_plm10_entropy"] == pytest.approx(-1.0)


def test_exact_levelset_dice_integrates_every_probability_interval():
    probability = np.array([[0.9, 0.6], [0.4, 0.1]])
    prediction = probability >= 0.5
    expected_risk = 0.1 * 1 + 0.3 / 3 + 0.2 * 0 + 0.3 * 0.2 + 0.1 / 3
    assert exact_levelset_dice_confidence(probability, prediction) == pytest.approx(
        -expected_risk
    )


def test_exact_levelset_dice_handles_ties_and_empty_maps():
    empty = np.zeros((2, 3))
    assert exact_levelset_dice_confidence(empty, empty.astype(bool)) == 0.0
    assert exact_levelset_dice_confidence(empty, np.ones((2, 3))) == -1.0

    probability = np.array([[0.8, 0.8], [0.2, 0.2]])
    prediction = probability >= 0.5
    # (0.8,1]: empty-vs-prediction; (0.2,0.8]: exact prediction;
    # (0,0.2]: all four pixels versus the two-pixel prediction.
    assert exact_levelset_dice_confidence(probability, prediction) == pytest.approx(
        -(0.2 + 0.2 / 3)
    )


@pytest.mark.parametrize("seed", range(4))
def test_exact_levelset_dice_matches_direct_interval_enumeration(seed):
    rng = np.random.default_rng(seed)
    probability = rng.choice([0.0, 0.1, 0.35, 0.7, 1.0], size=(7, 9))
    prediction = probability >= (0.3 + 0.1 * seed)
    boundaries = np.unique(np.concatenate(([0.0], probability.ravel(), [1.0])))
    risk = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        level = probability >= upper
        denominator = level.sum() + prediction.sum()
        loss = (
            0.0
            if denominator == 0
            else 1 - 2 * np.logical_and(level, prediction).sum() / denominator
        )
        risk += (upper - lower) * loss
    assert exact_levelset_dice_confidence(probability, prediction) == pytest.approx(
        -risk
    )


def test_baselines_validate_shapes_ranges_masks_and_patch_size():
    with pytest.raises(ValueError, match="one non-empty shape"):
        strong_binary_confidences(np.zeros((10, 10)), np.zeros((9, 10)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_entropy_bits(np.array([1.1]))
    with pytest.raises(ValueError, match="binary"):
        strong_binary_confidences(np.zeros((10, 10)), np.full((10, 10), 2))
    with pytest.raises(ValueError, match="requires both dimensions"):
        strong_binary_confidences(np.zeros((9, 10)), np.zeros((9, 10)))
