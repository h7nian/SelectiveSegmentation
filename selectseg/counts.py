"""Dice confidence from loss-sufficient count posteriors.

For a fixed deployed action ``A``, foreground Dice depends on a candidate mask
only through ``Z = |Y intersect A|`` and ``W = |Y outside A|``.  This module
therefore works on the induced count posterior instead of constructing full
candidate masks.

``shared_threshold_dice_confidence`` is the usual image-wide comonotone
coupling. ``action_two_block_dice_confidence`` assigns independent uniform
thresholds to ``A`` and its complement. Both retain the input pixel marginals;
only their joint laws differ. The two-block expectation is evaluated with a
deterministic product midpoint rule.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import ndimage

from selectseg.confidence import midpoint_rule


def _validate_inputs(probability, action) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=float)
    action = np.asarray(action)
    if probability.ndim != 2 or 0 in probability.shape:
        raise ValueError("probability must be a non-empty 2D array")
    if action.shape != probability.shape or action.dtype != np.bool_:
        raise TypeError("action must be a boolean array matching probability")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("probability must be finite and lie in [0, 1]")
    return probability, action


def _threshold_counts(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Count entries greater than or equal to every increasing node."""

    sorted_values = np.sort(np.asarray(values, dtype=float))
    return sorted_values.size - np.searchsorted(sorted_values, nodes, side="left")


def _dice_loss_from_counts(
    action_size: int, overlap: np.ndarray, outside: np.ndarray
) -> np.ndarray:
    denominator = action_size + overlap + outside
    loss = np.ones(np.broadcast_shapes(overlap.shape, outside.shape), dtype=float)
    denominator = np.broadcast_to(denominator, loss.shape)
    overlap = np.broadcast_to(overlap, loss.shape)
    nonempty = denominator > 0
    loss[~nonempty] = 0.0
    loss[nonempty] -= 2.0 * overlap[nonempty] / denominator[nonempty]
    return loss


def count_ladders(probability, action, *, m: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint overlap and outside-count ladders for one action."""

    probability, action = _validate_inputs(probability, action)
    nodes, _ = midpoint_rule(m)
    return (
        _threshold_counts(probability[action], nodes),
        _threshold_counts(probability[~action], nodes),
    )


def shared_threshold_dice_confidence(probability, action, *, m: int = 32) -> float:
    """Negative Dice risk under one shared uniform threshold."""

    probability, action = _validate_inputs(probability, action)
    overlap, outside = count_ladders(probability, action, m=m)
    losses = _dice_loss_from_counts(int(action.sum()), overlap, outside)
    confidence = -float(np.mean(losses))
    if not math.isfinite(confidence) or not -1 <= confidence <= 0:
        raise RuntimeError("invalid shared-threshold Dice confidence")
    return confidence


def action_two_block_dice_confidence(probability, action, *, m: int = 32) -> float:
    """Negative Dice risk with independent thresholds inside and outside action.

    The first threshold changes ``Z = |Y intersect A|`` and the second changes
    ``W = |Y outside A|``. The Cartesian midpoint rule has ``m**2`` equally
    weighted states but requires only two one-dimensional count ladders.
    """

    probability, action = _validate_inputs(probability, action)
    overlap, outside = count_ladders(probability, action, m=m)
    losses = _dice_loss_from_counts(
        int(action.sum()), overlap[:, None], outside[None, :]
    )
    confidence = -float(np.mean(losses))
    if not math.isfinite(confidence) or not -1 <= confidence <= 0:
        raise RuntimeError("invalid two-block Dice confidence")
    return confidence


def second_order_dice_similarity(
    action_size: int,
    mean_overlap: float,
    mean_outside: float,
    variance_overlap: float,
    variance_outside: float,
    covariance: float,
) -> float:
    """Second-order expansion exposing the coupling correction to SDC.

    This diagnostic is not used as a production confidence score. It makes
    explicit that SDC is the zeroth-order ratio at the mean counts, while a
    coupling changes Dice through the count variances and covariance.
    """

    values = (
        action_size,
        mean_overlap,
        mean_outside,
        variance_overlap,
        variance_outside,
        covariance,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("moments must be finite")
    if action_size < 0 or min(values[1:5]) < 0:
        raise ValueError("sizes, means, and variances must be non-negative")
    denominator = action_size + mean_overlap + mean_outside
    if denominator == 0:
        return 1.0
    plug_in = 2.0 * mean_overlap / denominator
    correction = (
        -2.0 * (action_size + mean_outside) * variance_overlap
        + 2.0 * (mean_overlap - action_size - mean_outside) * covariance
        + 2.0 * mean_overlap * variance_outside
    ) / denominator**3
    return float(plug_in + correction)


FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)


def component_partition(mask: np.ndarray) -> np.ndarray:
    """Label four-connected components of a binary set and its complement."""

    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or 0 in mask.shape:
        raise TypeError("mask must be a non-empty boolean 2D array")
    foreground, num_foreground = ndimage.label(mask, structure=FOUR_CONNECTED)
    background, num_background = ndimage.label(~mask, structure=FOUR_CONNECTED)
    labels = np.empty(mask.shape, dtype=np.int32)
    labels[mask] = foreground[mask] - 1
    labels[~mask] = background[~mask] - 1 + num_foreground
    expected = num_foreground + num_background
    if expected <= 0 or int(labels.min()) != 0 or int(labels.max()) != expected - 1:
        raise RuntimeError("component partition is not contiguous")
    return labels


def balanced_grid_partition(shape: tuple[int, int], target_blocks: int) -> np.ndarray:
    """Create exactly ``target_blocks`` compact deterministic grid cells."""

    height, width = shape
    if height <= 0 or width <= 0 or not 1 <= target_blocks <= height * width:
        raise ValueError("invalid shape or target block count")
    row_count = min(
        height,
        max(1, round(math.sqrt(target_blocks * height / width))),
        target_blocks,
    )
    base, remainder = divmod(target_blocks, row_count)
    columns_per_row = np.full(row_count, base, dtype=int)
    columns_per_row[:remainder] += 1
    while np.any(columns_per_row > width):
        row_count += 1
        if row_count > min(height, target_blocks):
            raise RuntimeError("could not construct requested grid")
        base, remainder = divmod(target_blocks, row_count)
        columns_per_row = np.full(row_count, base, dtype=int)
        columns_per_row[:remainder] += 1

    row_edges = np.linspace(0, height, row_count + 1, dtype=int)
    labels = np.empty(shape, dtype=np.int32)
    offset = 0
    for row in range(row_count):
        columns = int(columns_per_row[row])
        column_edges = np.linspace(0, width, columns + 1, dtype=int)
        for column in range(columns):
            labels[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ] = offset + column
        offset += columns
    if (
        offset != target_blocks
        or int(labels.min()) != 0
        or int(labels.max()) != target_blocks - 1
    ):
        raise RuntimeError("grid partition did not realize its target block count")
    return labels


def labels_for_coupling(
    probability: np.ndarray,
    action: np.ndarray,
    *,
    coupling: str,
    proposal_threshold: float | None = None,
) -> np.ndarray:
    """Construct the declared component or matched-grid partition."""

    probability, action = _validate_inputs(probability, action)
    allowed = {
        "action_components",
        "action_grid",
        "proposal_components",
        "proposal_grid",
    }
    if coupling not in allowed:
        raise ValueError(f"unsupported partition coupling: {coupling}")
    if coupling.startswith("proposal"):
        if proposal_threshold is None or not 0 < proposal_threshold < 1:
            raise ValueError("proposal coupling requires a threshold in (0, 1)")
        reference = component_partition(probability >= proposal_threshold)
    else:
        if proposal_threshold is not None:
            raise ValueError("action coupling does not accept a proposal threshold")
        reference = component_partition(action)
    if coupling.endswith("components"):
        return reference
    return balanced_grid_partition(probability.shape, int(reference.max()) + 1)


def _partition_seed(
    master_seed: int, sample_id: str, coupling: str, repeat: int
) -> int:
    payload = f"{master_seed}|{sample_id}|{coupling}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _partition_count_draws(
    probability: np.ndarray,
    action: np.ndarray,
    labels: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate inside/outside foreground counts for block thresholds."""

    if labels.shape != probability.shape or labels.dtype.kind not in "iu":
        raise TypeError("labels must be an integer array matching probability")
    num_blocks = int(labels.max()) + 1
    if labels.min() != 0 or uniforms.shape[1] != num_blocks:
        raise ValueError("labels and uniforms must identify contiguous blocks")
    flat_labels = labels.ravel()
    order = np.argsort(flat_labels, kind="stable")
    ordered_labels = flat_labels[order]
    boundaries = np.flatnonzero(np.diff(ordered_labels)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [order.size]))
    if starts.size != num_blocks:
        raise RuntimeError("partition contains empty or missing blocks")

    flat_probability = probability.ravel()
    flat_action = action.ravel()
    overlap = np.zeros(uniforms.shape[0], dtype=np.int64)
    outside = np.zeros(uniforms.shape[0], dtype=np.int64)
    for block, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        indices = order[start:stop]
        thresholds = uniforms[:, block]
        inside_values = np.sort(flat_probability[indices[flat_action[indices]]])
        outside_values = np.sort(flat_probability[indices[~flat_action[indices]]])
        overlap += inside_values.size - np.searchsorted(
            inside_values, thresholds, side="left"
        )
        outside += outside_values.size - np.searchsorted(
            outside_values, thresholds, side="left"
        )
    return overlap, outside


def partition_dice_confidence(
    probability,
    action,
    labels,
    *,
    draws: int = 256,
    repeats: int = 4,
    master_seed: int = 20260721,
    sample_id: str = "sample",
    coupling_id: str = "partition",
) -> tuple[float, np.ndarray]:
    """Estimate negative Dice risk with antithetic per-block thresholds."""

    probability, action = _validate_inputs(probability, action)
    labels = np.asarray(labels)
    if draws <= 0 or draws % 2 or repeats <= 0:
        raise ValueError("draws must be positive and even; repeats must be positive")
    num_blocks = int(labels.max()) + 1
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        generator = np.random.default_rng(
            _partition_seed(master_seed, str(sample_id), coupling_id, repeat)
        )
        half = generator.random((draws // 2, num_blocks), dtype=np.float32)
        uniforms = np.concatenate((half, 1.0 - half), axis=0)
        overlap, outside = _partition_count_draws(
            probability, action, labels, uniforms
        )
        losses = _dice_loss_from_counts(int(action.sum()), overlap, outside)
        estimates[repeat] = -float(np.mean(losses))
    confidence = float(np.mean(estimates))
    if not np.isfinite(estimates).all() or not -1 <= confidence <= 0:
        raise RuntimeError("partition Dice confidence is invalid")
    return confidence, estimates


def partition_diagnostics(labels: np.ndarray) -> dict[str, float | int]:
    """Return block count and largest-block fraction."""

    labels = np.asarray(labels)
    counts = np.bincount(labels.ravel())
    if counts.size == 0 or np.any(counts == 0):
        raise ValueError("labels must define nonempty contiguous blocks")
    return {
        "num_blocks": int(counts.size),
        "largest_block_fraction": float(counts.max() / counts.sum()),
    }
