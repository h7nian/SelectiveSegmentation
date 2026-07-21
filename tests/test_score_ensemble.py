"""Unit tests for empirical-posterior and threshold-stability baselines."""

from types import SimpleNamespace

import numpy as np

from selectseg.studies.ensemble import (
    SCORE_FIELDS,
    score_ensemble_sample,
    threshold_iou_stability,
)


def _sample(index, sample_id, probability, truth):
    return SimpleNamespace(
        index=index,
        sample_id=sample_id,
        foreground_probability=np.asarray(probability, dtype=np.float32),
        truth=np.asarray(truth, dtype=np.uint8),
    )


def test_threshold_stability_totalizes_empty_pair_and_uses_nested_iou():
    assert threshold_iou_stability(np.zeros((2, 2))) == 1.0
    probability = np.array([[0.1, 0.3], [0.8, 0.9]])
    assert threshold_iou_stability(probability) == 2 / 3


def test_ensemble_scores_are_aligned_oriented_and_finite():
    truth = [[0, 0], [1, 1]]
    mean = _sample(0, "case", [[0.1, 0.4], [0.7, 0.9]], truth)
    members = (
        _sample(0, "case", [[0.1, 0.2], [0.8, 0.9]], truth),
        _sample(0, "case", [[0.1, 0.6], [0.7, 0.9]], truth),
        _sample(0, "case", [[0.2, 0.4], [0.6, 0.8]], truth),
    )
    row = score_ensemble_sample(mean, members)
    assert set(SCORE_FIELDS).issubset(row)
    assert row["risk_dice"] == 0.0
    assert row["confidence_ensemble_q_dice"] < 0.0
    assert row["confidence_ensemble_all_iou"] < 1.0
    assert all(np.isfinite(row[field]) for field in SCORE_FIELDS)
