"""Focused binary evaluator schema and action-anchoring tests."""

import numpy as np
import pytest

from selectseg.evaluate import _binary_entropy, binary_record
from selectseg.confidence import midpoint_rule


def _rules(*counts):
    return {count: midpoint_rule(count) for count in counts}


def test_binary_record_has_matched_risks_and_loss_indexed_confidences():
    probability = np.full((10, 10), 0.1, dtype=float)
    probability[:3, :3] = np.array(
        [[0.9, 0.8, 0.2], [0.7, 0.4, 0.1], [0.2, 0.1, 0.0]], dtype=float
    )
    truth = np.zeros((10, 10), dtype=bool)
    truth[:3, :3] = np.array(
        [[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool
    )
    row = binary_record(
        probability,
        truth,
        run_id="unit-test",
        image_id="case-7",
        image_index=7,
        class_index=1,
        class_name="lesion",
        decision_threshold=0.5,
        quadrature_rules=_rules(2, 32),
    )

    # Native binary evaluation has exactly one task per image, so these IDs join.
    assert row["sample_id"] == "case-7"
    assert row["risk_dice"] == 0.0
    assert row["risk_nhd95"] == 0.0
    assert 0 <= row["confidence_sdc"] <= 1
    for field in (
        "confidence_dice_exact",
        "confidence_qfr_entropy",
        "confidence_plm10_entropy",
        "confidence_foreground_entropy",
    ):
        assert -1 <= row[field] <= 0
    assert -2 <= row["confidence_mmmc_entropy"] <= 0
    for count in (2, 32):
        assert -1 <= row[f"confidence_dice_m{count}"] <= 0
        assert -1 <= row[f"confidence_nhd95_m{count}"] <= 0
    assert not any(key.startswith("image_mdice") for key in row)


def test_binary_record_anchors_confidence_to_the_declared_gamma():
    probability = np.tile(np.array([[0.8, 0.6], [0.4, 0.2]]), (5, 5))
    truth = np.tile(np.array([[1, 0], [0, 0]], dtype=bool), (5, 5))
    common = dict(
        run_id="unit-test",
        image_id="x",
        image_index=0,
        class_index=1,
        class_name="object",
        quadrature_rules=_rules(2),
    )
    at_half = binary_record(
        probability, truth, decision_threshold=0.5, **common
    )
    at_point_seven = binary_record(
        probability, truth, decision_threshold=0.7, **common
    )
    assert at_half["prediction_foreground_fraction"] == 0.5
    assert at_point_seven["prediction_foreground_fraction"] == 0.25
    assert at_half["risk_dice"] != at_point_seven["risk_dice"]
    assert at_half["confidence_dice_m2"] != at_point_seven["confidence_dice_m2"]


def test_negative_image_is_kept_and_uses_total_empty_conventions():
    probability = np.tile(np.array([[0.9, 0.8], [0.2, 0.1]]), (5, 5))
    truth = np.zeros((10, 10), dtype=bool)
    common = dict(
        run_id="unit-test",
        image_id="x",
        image_index=0,
        class_index=1,
        class_name="object",
        decision_threshold=0.5,
        quadrature_rules=_rules(2),
    )
    row = binary_record(probability, truth, **common)
    assert row["risk_dice"] == 1.0
    assert row["risk_nhd95"] == 1.0
    assert row["risk_hd95_pixels"] == pytest.approx(row["image_diagonal"])

    empty = binary_record(
        np.zeros((10, 10)), truth, **common
    )
    assert empty["risk_dice"] == 0.0
    assert empty["risk_nhd95"] == 0.0
    # This is the published SDC baseline convention, recorded in the manifest.
    assert empty["confidence_sdc"] == 0.0


def test_binary_record_requires_a_total_binary_truth_and_equal_shapes():
    probability = np.full((10, 10), 0.4)
    void_truth = np.zeros((10, 10), dtype=int)
    void_truth[0, :3] = (0, 255, 1)
    with pytest.raises(ValueError, match="no void"):
        binary_record(
            probability,
            void_truth,
            run_id="unit-test",
            image_id="x",
            image_index=0,
            class_index=1,
            class_name="object",
            decision_threshold=0.5,
            quadrature_rules=_rules(2),
        )
    with pytest.raises(ValueError, match="equal shapes"):
        binary_record(
            probability,
            np.ones((11, 10), dtype=bool),
            run_id="unit-test",
            image_id="x",
            image_index=0,
            class_index=1,
            class_name="object",
            decision_threshold=0.5,
            quadrature_rules=_rules(2),
        )


def test_binary_record_rejects_mislabeled_rule_and_invalid_gamma():
    probability = np.full((10, 10), 0.4)
    truth = np.zeros((10, 10), dtype=bool)
    common = dict(
        run_id="unit-test",
        image_id="x",
        image_index=0,
        class_index=1,
        class_name="object",
    )
    with pytest.raises(ValueError, match="decision_threshold"):
        binary_record(
            probability,
            truth,
            decision_threshold=1.0,
            quadrature_rules=_rules(2),
            **common,
        )
    with pytest.raises(ValueError, match="not its midpoint"):
        binary_record(
            probability,
            truth,
            decision_threshold=0.5,
            quadrature_rules={2: (np.array([0.2, 0.8]), np.array([0.5, 0.5]))},
            **common,
        )


def test_binary_entropy_is_finite_at_probability_endpoints():
    values = _binary_entropy(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values[1] == pytest.approx(np.log(2))
    assert values[2] == pytest.approx(0.0, abs=1e-9)
