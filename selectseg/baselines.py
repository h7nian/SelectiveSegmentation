"""Strong single-map confidence baselines for binary segmentation.

Every score uses only a foreground probability map and its fixed deployed
mask. Scores are oriented so that larger means more confident. Binary entropy
is measured in bits, hence lies in ``[0, 1]``.
"""

import numpy as np


def _validated_inputs(foreground_probability, hard_prediction):
    probability = np.asarray(foreground_probability, dtype=float)
    prediction = np.asarray(hard_prediction)
    if probability.ndim != 2 or prediction.ndim != 2:
        raise ValueError("probability and prediction must be 2D")
    if probability.shape != prediction.shape or 0 in probability.shape:
        raise ValueError("probability and prediction must have one non-empty shape")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("foreground probabilities must be finite and lie in [0, 1]")
    if not np.all((prediction == 0) | (prediction == 1)):
        raise ValueError("hard prediction must be binary")
    return probability, prediction.astype(bool, copy=False)


def binary_entropy_bits(foreground_probability):
    """Elementwise Bernoulli entropy with exact zero at both endpoints."""

    probability = np.asarray(foreground_probability, dtype=float)
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("foreground probabilities must be finite and lie in [0, 1]")
    entropy = np.zeros_like(probability, dtype=float)
    interior = (probability > 0) & (probability < 1)
    p = probability[interior]
    entropy[interior] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return entropy


def _maximum_valid_patch_mean(values, patch_size):
    if isinstance(patch_size, bool) or not isinstance(
        patch_size, (int, np.integer)
    ):
        raise TypeError("patch_size must be a positive integer")
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width = values.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"a {patch_size}x{patch_size} valid patch requires both dimensions "
            f"to be at least {patch_size}; got {(height, width)}"
        )
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    patch_sums = (
        integral[patch_size:, patch_size:]
        - integral[:-patch_size, patch_size:]
        - integral[patch_size:, :-patch_size]
        + integral[:-patch_size, :-patch_size]
    )
    return float(patch_sums.max() / patch_size**2)


def exact_levelset_dice_confidence(foreground_probability, hard_prediction):
    """Exact uniform-threshold Dice confidence in ``O(N log N)`` time.

    The nested level set changes only when the threshold crosses a distinct
    probability. Sorting once therefore integrates every constant interval
    exactly, including the empty level set above the maximum probability.
    """

    probability, prediction = _validated_inputs(
        foreground_probability, hard_prediction
    )
    flat_probability = probability.ravel()
    order = np.argsort(-flat_probability, kind="stable")
    sorted_probability = flat_probability[order]
    sorted_prediction = prediction.ravel()[order]
    prediction_size = int(prediction.sum())
    empty_loss = float(prediction_size > 0)
    positive_count = int(np.count_nonzero(sorted_probability > 0))
    if positive_count == 0:
        return -empty_loss

    positive_probability = sorted_probability[:positive_count]
    group_ends = np.flatnonzero(
        np.r_[positive_probability[:-1] != positive_probability[1:], True]
    )
    level_sizes = group_ends + 1
    intersections = np.cumsum(sorted_prediction[:positive_count], dtype=np.int64)[
        group_ends
    ]
    denominators = level_sizes + prediction_size
    losses = 1 - 2 * intersections / denominators
    values = positive_probability[group_ends]
    interval_widths = values - np.r_[values[1:], 0.0]
    risk = (1 - values[0]) * empty_loss + np.dot(interval_widths, losses)
    return -float(risk)


def strong_binary_confidences(
    foreground_probability,
    hard_prediction,
    *,
    patch_size=10,
):
    """Return QFR, PLM/PLA, MMMC, and foreground-entropy confidences.

    QFR uses the predicted foreground fraction and NumPy's default quantile
    interpolation; ``>=`` retains all ties. PLM is the maximum mean entropy in
    a valid stride-one patch. At a fixed patch size, PLA's maximum entropy sum
    is a positive constant multiple and therefore gives the same ranking.
    Empty deployed masks receive foreground-entropy confidence ``-1``. MMMC is
    defined as zero when the whole entropy map is zero, resolving its ``0/0``
    endpoint without perturbing non-degenerate scores.
    """

    probability, prediction = _validated_inputs(
        foreground_probability, hard_prediction
    )
    entropy = binary_entropy_bits(probability)

    foreground_fraction = float(prediction.mean())
    qfr_threshold = float(np.quantile(entropy, 1 - foreground_fraction))
    qfr = -float(entropy[entropy >= qfr_threshold].mean())

    maximum_entropy = float(entropy.max())
    mmmc = (
        0.0
        if maximum_entropy == 0
        else -float((np.median(entropy) + entropy.min()) / maximum_entropy)
    )
    foreground = -float(entropy[prediction].mean()) if prediction.any() else -1.0

    return {
        "confidence_dice_exact": exact_levelset_dice_confidence(
            probability, prediction
        ),
        "confidence_qfr_entropy": qfr,
        "confidence_plm10_entropy": -_maximum_valid_patch_mean(
            entropy, patch_size
        ),
        "confidence_mmmc_entropy": mmmc,
        "confidence_foreground_entropy": foreground,
    }
