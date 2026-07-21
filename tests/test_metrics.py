"""Unit tests for the metrics (CPU, no data needed)."""

import pytest
import torch

from selectseg.metrics import (
    ConfusionMatrix,
    HausdorffMetric,
    PerImageMetrics,
    hausdorff_95,
)


def test_hand_computed_binary_case():
    confusion = ConfusionMatrix(("background", "pet"))
    target = torch.tensor([0, 0, 1, 1, 255])
    prediction = torch.tensor([0, 1, 1, 1, 0])
    confusion.update(prediction, target)
    metrics = confusion.compute()
    # matrix = [[1, 1], [0, 2]] once the ignored pixel is dropped
    assert metrics["pixel_accuracy"] == pytest.approx(3 / 4)
    assert metrics["per_class_iou"]["background"] == pytest.approx(1 / 2)
    assert metrics["per_class_iou"]["pet"] == pytest.approx(2 / 3)
    assert metrics["mean_iou"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics["mean_dice"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    # recall: background 1/2, pet 2/2; frequency: both classes half the pixels
    assert metrics["mean_class_accuracy"] == pytest.approx((1 / 2 + 1) / 2)
    assert metrics["fw_iou"] == pytest.approx(1 / 2 * 1 / 2 + 1 / 2 * 2 / 3)


def test_perfect_prediction():
    confusion = ConfusionMatrix(("a", "b", "c"))
    target = torch.tensor([[0, 1], [2, 2]])
    confusion.update(target.clone(), target)
    metrics = confusion.compute()
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["mean_dice"] == 1.0


def test_ignored_pixels_do_not_count():
    confusion = ConfusionMatrix(("a", "b"))
    confusion.update(torch.tensor([1, 1]), torch.tensor([255, 255]))
    assert confusion.matrix.sum() == 0


def test_absent_class_reported_none_and_excluded_from_means():
    confusion = ConfusionMatrix(("a", "b", "c"))
    confusion.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    metrics = confusion.compute()
    assert metrics["per_class_iou"]["c"] is None
    assert metrics["mean_iou"] == 1.0


def test_accumulates_across_updates():
    confusion = ConfusionMatrix(("a", "b"))
    confusion.update(torch.tensor([0]), torch.tensor([0]))
    confusion.update(torch.tensor([0]), torch.tensor([1]))
    assert confusion.compute()["pixel_accuracy"] == pytest.approx(1 / 2)


def test_per_image_metrics_average_images_equally():
    aggregator = PerImageMetrics(("background", "pet"))
    aggregator.update(torch.tensor([0, 1]), torch.tensor([0, 1]))  # perfect
    aggregator.update(
        torch.tensor([0, 1, 1, 1, 0]), torch.tensor([0, 0, 1, 1, 255])
    )  # the hand-computed case: mIoU 7/12, mDice 11/15
    result = aggregator.compute()
    assert result["per_image_mean_iou"] == pytest.approx((1 + 7 / 12) / 2)
    assert result["per_image_mean_dice"] == pytest.approx((1 + 11 / 15) / 2)


def test_per_image_metrics_skip_fully_ignored_images():
    aggregator = PerImageMetrics(("background", "pet"))
    aggregator.update(torch.tensor([0, 1]), torch.tensor([255, 255]))
    result = aggregator.compute()
    assert result["per_image_mean_iou"] is None
    assert result["per_image_mean_dice"] is None


def test_hausdorff_95_identical_masks_is_zero():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:5, 2:5] = True
    assert hausdorff_95(mask, mask.clone()) == 0.0


def test_hausdorff_95_single_pixels_is_euclidean_distance():
    a = torch.zeros(8, 8, dtype=torch.bool)
    b = torch.zeros(8, 8, dtype=torch.bool)
    a[0, 0] = True
    b[3, 4] = True  # 3-4-5 triangle
    assert hausdorff_95(a, b) == pytest.approx(5.0)


def test_hausdorff_95_empty_mask_is_undefined():
    empty = torch.zeros(4, 4, dtype=torch.bool)
    full = torch.ones(4, 4, dtype=torch.bool)
    assert hausdorff_95(empty, full) is None
    assert hausdorff_95(full, empty) is None


def test_hausdorff_metric_aggregation():
    metric = HausdorffMetric(("background", "a", "b"))
    prediction = torch.zeros(8, 8, dtype=torch.long)
    target = torch.zeros(8, 8, dtype=torch.long)
    prediction[0, 0] = 1
    target[3, 4] = 1  # class a: defined, distance 5
    prediction[7, 7] = 2  # class b: predicted but absent from target
    metric.update(prediction, target)
    result = metric.compute()
    assert result["per_class_hd95"]["a"] == pytest.approx(5.0)
    assert result["per_class_hd95"]["b"] is None
    assert result["per_class_hd95_images"] == {"a": 1, "b": 0}
    assert result["per_class_hd95_undefined"] == {"a": 0, "b": 1}
    assert result["mean_hd95"] == pytest.approx(5.0)


def test_hausdorff_metric_excludes_ignored_pixels():
    metric = HausdorffMetric(("background", "a"))
    prediction = torch.zeros(8, 8, dtype=torch.long)
    target = torch.full((8, 8), 255, dtype=torch.long)
    prediction[0, 0] = 1  # falls entirely inside the ignored region
    metric.update(prediction, target)
    result = metric.compute()
    assert result["mean_hd95"] is None
    assert result["per_class_hd95_undefined"] == {"a": 0}
