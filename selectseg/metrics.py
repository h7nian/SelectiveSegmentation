"""Segmentation metrics: confusion-matrix scores and Hausdorff distance."""

import numpy as np
import torch
from scipy import ndimage


class ConfusionMatrix:
    """Accumulates predictions and reports IoU, Dice, and pixel accuracy.

    Pixels whose target or prediction falls outside the class range
    (notably :data:`IGNORE_INDEX` targets) are excluded. Classes absent
    from both predictions and targets are reported as ``None`` and
    excluded from the means.
    """

    def __init__(self, class_names):
        self.class_names = tuple(class_names)
        num_classes = len(self.class_names)
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, prediction, target):
        num_classes = len(self.class_names)
        prediction = prediction.flatten()
        target = target.flatten()
        valid = (
            (target >= 0)
            & (target < num_classes)
            & (prediction >= 0)
            & (prediction < num_classes)
        )
        index = target[valid] * num_classes + prediction[valid]
        counts = torch.bincount(index, minlength=num_classes**2)
        self.matrix += counts.view(num_classes, num_classes)

    def compute(self):
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        target_total = matrix.sum(dim=1)
        prediction_total = matrix.sum(dim=0)
        union = target_total + prediction_total - true_positive
        observed = union > 0
        iou = true_positive[observed] / union[observed]
        dice = 2 * true_positive[observed] / (target_total + prediction_total)[observed]
        in_target = target_total > 0
        class_accuracy = true_positive[in_target] / target_total[in_target]
        frequency = target_total[observed] / matrix.sum()
        per_class_iou = {}
        iou_iterator = iter(iou.tolist())
        for name, seen in zip(self.class_names, observed.tolist()):
            per_class_iou[name] = next(iou_iterator) if seen else None
        return {
            "pixel_accuracy": (true_positive.sum() / matrix.sum()).item(),
            "mean_iou": iou.mean().item(),
            "mean_dice": dice.mean().item(),
            "mean_class_accuracy": class_accuracy.mean().item(),
            "fw_iou": (frequency * iou).sum().item(),
            "per_class_iou": per_class_iou,
        }


class PerImageMetrics:
    """Mean over images of per-image mIoU and mean Dice.

    Each image is scored by its own confusion matrix with the same
    observed-class convention as :class:`ConfusionMatrix`, then images are
    averaged with equal weight — the per-case convention of medical-image
    evaluation, complementary to the dataset-pooled scores where large
    instances dominate a class's total. Images with no valid pixels are
    skipped.
    """

    def __init__(self, class_names):
        self.class_names = tuple(class_names)
        self.total_iou = 0.0
        self.total_dice = 0.0
        self.images = 0

    def update(self, prediction, target):
        image_matrix = ConfusionMatrix(self.class_names)
        image_matrix.update(prediction, target)
        if image_matrix.matrix.sum() == 0:
            return
        scores = image_matrix.compute()
        self.total_iou += scores["mean_iou"]
        self.total_dice += scores["mean_dice"]
        self.images += 1

    def compute(self):
        if not self.images:
            return {"per_image_mean_iou": None, "per_image_mean_dice": None}
        return {
            "per_image_mean_iou": self.total_iou / self.images,
            "per_image_mean_dice": self.total_dice / self.images,
        }


def hausdorff_95(prediction, target):
    """Symmetric 95th-percentile Hausdorff distance between binary masks.

    Follows the medpy convention: pool the directed surface distances from
    both masks and take their 95th percentile, in pixels. Returns ``None``
    when either mask is empty, where the distance is undefined.
    """
    prediction = prediction.numpy().astype(bool)
    target = target.numpy().astype(bool)
    if not prediction.any() or not target.any():
        return None
    pred_surface = prediction & ~ndimage.binary_erosion(prediction)
    target_surface = target & ~ndimage.binary_erosion(target)
    to_target = ndimage.distance_transform_edt(~target_surface)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface)[target_surface]
    return float(np.percentile(np.hstack([to_target, to_pred]), 95))


class HausdorffMetric:
    """Per-class mean HD95 over images, at the masks' native resolution.

    Foreground classes only (the background boundary mirrors the foreground
    ones). HD95 is averaged over images where prediction and target both
    contain the class; images where only one side contains it (a detection
    failure, distance undefined) are tallied separately in
    ``per_class_hd95_undefined`` instead of being folded into the mean.
    Pixels whose target is outside the class range (ignored regions) are
    excluded from both masks.
    """

    def __init__(self, class_names):
        self.class_names = tuple(class_names)
        num_classes = len(self.class_names)
        self.sums = [0.0] * num_classes
        self.images = [0] * num_classes
        self.undefined = [0] * num_classes

    def update(self, prediction, target):
        num_classes = len(self.class_names)
        valid = (target >= 0) & (target < num_classes)
        present = set(torch.unique(target[valid]).tolist())
        present |= set(torch.unique(prediction[valid]).tolist())
        for index in range(1, num_classes):
            if index not in present:
                continue
            distance = hausdorff_95(
                (prediction == index) & valid, (target == index) & valid
            )
            if distance is None:
                self.undefined[index] += 1
            else:
                self.sums[index] += distance
                self.images[index] += 1

    def compute(self):
        per_class = {}
        for name, total, count in zip(self.class_names[1:], self.sums[1:], self.images[1:]):
            per_class[name] = total / count if count else None
        defined = [value for value in per_class.values() if value is not None]
        return {
            "mean_hd95": sum(defined) / len(defined) if defined else None,
            "per_class_hd95": per_class,
            "per_class_hd95_images": dict(
                zip(self.class_names[1:], self.images[1:])
            ),
            "per_class_hd95_undefined": dict(
                zip(self.class_names[1:], self.undefined[1:])
            ),
        }
