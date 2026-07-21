from types import SimpleNamespace

import numpy as np
import pytest

from selectseg.confidence import foreground_dice_loss
from selectseg.studies.counts import score_sample


def _sample(probability, truth=None):
    probability = np.asarray(probability, dtype=np.float32)
    if truth is None:
        truth = probability >= 0.5
    return SimpleNamespace(
        sample_id="case-1",
        index=3,
        foreground_probability=probability,
        truth=np.asarray(truth, dtype=np.uint8),
    )


def test_score_sample_keeps_action_and_risk_fixed():
    sample = _sample([[0.1, 0.4], [0.6, 0.9]], [[0, 1], [1, 1]])
    row = score_sample(sample, gamma=0.5, m=32)
    action = sample.foreground_probability >= 0.5
    assert row["risk_dice"] == foreground_dice_loss(sample.truth, action)
    assert row["confidence_dice_shared_m32_recomputed"] <= 0
    assert row["confidence_dice_action_two_block_m32"] <= 0
    assert row["two_block_covariance"] == 0
    assert row["score_runtime_seconds"] >= 0


def test_soft_dice_is_the_ratio_at_mean_counts():
    sample = _sample([[0.2, 0.4], [0.6, 0.8]])
    row = score_sample(sample, gamma=0.5, m=32)
    expected = 2 * (0.6 + 0.8) / (2 + 0.2 + 0.4 + 0.6 + 0.8)
    assert row["confidence_dice_sdc_recomputed"] == pytest.approx(expected)


def test_score_sample_rejects_noncanonical_payload_dtypes():
    sample = _sample([[0.2, 0.8]])
    sample.foreground_probability = sample.foreground_probability.astype(float)
    with pytest.raises(TypeError):
        score_sample(sample, gamma=0.5, m=32)
