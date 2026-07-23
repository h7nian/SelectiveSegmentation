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
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from selectseg.confidence import midpoint_rule


@dataclass(frozen=True)
class SpatialCopulaDiceEstimate:
    """One Monte Carlo estimate under a marginal-preserving spatial copula."""

    confidence: float
    spatial_grid_shape: tuple[int, int] | None


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


def _spatial_grid_shape(
    image_shape: tuple[int, int], knot_spacing_diagonal: float
) -> tuple[int, int]:
    """Choose a bilinear-field grid from a resolution-invariant knot spacing."""

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain two positive dimensions")
    if not math.isfinite(knot_spacing_diagonal) or not (
        0 < knot_spacing_diagonal <= 1
    ):
        raise ValueError("knot_spacing_diagonal must lie in (0, 1]")
    spacing_pixels = knot_spacing_diagonal * math.hypot(height, width)

    def axis_size(length: int) -> int:
        if length == 1:
            return 1
        return min(length, max(2, math.ceil((length - 1) / spacing_pixels) + 1))

    return axis_size(height), axis_size(width)


def _linear_interpolation_variance(
    output_size: int, knot_count: int, *, torch, device
):
    """Return pointwise variance of aligned linear interpolation of iid knots."""

    if output_size == 1 or knot_count == 1:
        return torch.ones(output_size, dtype=torch.float32, device=device)
    positions = torch.linspace(
        0,
        knot_count - 1,
        output_size,
        dtype=torch.float32,
        device=device,
    )
    fraction = positions - torch.floor(positions)
    return (1 - fraction).square() + fraction.square()


def sample_spatial_copula_masks(
    probability,
    *,
    posterior_draws: int,
    repeat_index: int,
    global_variance_weight: float,
    spatial_variance_weight: float,
    spatial_knot_spacing_diagonal: float,
    posterior_batch_size: int = 64,
    master_seed: int = 20260721,
    sample_id: str = "sample",
    device: str = "cpu",
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Draw coherent masks while preserving every input pixel marginal.

    This materializing API is intended for small synthetic diagnostics that
    evaluate arbitrary structured losses.  The production Dice scorer below
    remains streaming so native-resolution images do not retain every mask.
    """

    probability = np.asarray(probability, dtype=float)
    action = probability >= 0.5
    probability, _ = _validate_inputs(probability, action)
    if posterior_draws <= 0 or posterior_draws % 2:
        raise ValueError("posterior_draws must be a positive even integer")
    if repeat_index < 0 or posterior_batch_size <= 0 or master_seed < 0:
        raise ValueError("repeat, batch size, and seed must be valid")
    if (
        not math.isfinite(global_variance_weight)
        or not math.isfinite(spatial_variance_weight)
        or global_variance_weight < 0
        or spatial_variance_weight < 0
        or global_variance_weight + spatial_variance_weight > 1 + 1e-12
    ):
        raise ValueError("invalid copula variance weights")
    if not math.isfinite(spatial_knot_spacing_diagonal) or not (
        0 < spatial_knot_spacing_diagonal <= 1
    ):
        raise ValueError("spatial_knot_spacing_diagonal must lie in (0, 1]")

    import torch
    import torch.nn.functional as functional

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    residual_weight = max(
        0.0, 1.0 - global_variance_weight - spatial_variance_weight
    )
    grid_shape = (
        _spatial_grid_shape(probability.shape, spatial_knot_spacing_diagonal)
        if spatial_variance_weight > 0
        else None
    )
    quantile = torch.special.ndtri(
        torch.as_tensor(probability, dtype=torch.float32, device=compute_device)
    )
    spatial_sd = None
    if grid_shape is not None:
        row_variance = _linear_interpolation_variance(
            probability.shape[0], grid_shape[0], torch=torch, device=compute_device
        )
        column_variance = _linear_interpolation_variance(
            probability.shape[1], grid_shape[1], torch=torch, device=compute_device
        )
        spatial_sd = torch.sqrt(row_variance[:, None] * column_variance[None, :])

    coupling_id = (
        "spatial-copula-masks|"
        f"global={float(global_variance_weight).hex()}|"
        f"spatial={float(spatial_variance_weight).hex()}|"
        f"spacing={float(spatial_knot_spacing_diagonal).hex()}"
    )
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(
        _partition_seed(master_seed, str(sample_id), coupling_id, repeat_index)
        % (2**63 - 1)
    )
    output = []
    remaining_pairs = posterior_draws // 2
    while remaining_pairs:
        batch_size = min(posterior_batch_size, remaining_pairs)
        latent = None
        if global_variance_weight > 0:
            latent = math.sqrt(global_variance_weight) * torch.randn(
                (batch_size, 1, 1),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
        if spatial_variance_weight > 0:
            knots = torch.randn(
                (batch_size, 1, *grid_shape),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
            spatial = functional.interpolate(
                knots,
                size=probability.shape,
                mode="bilinear",
                align_corners=True,
            )[:, 0]
            spatial = math.sqrt(spatial_variance_weight) * spatial / spatial_sd
            latent = spatial if latent is None else latent + spatial
        if residual_weight > 0:
            independent = math.sqrt(residual_weight) * torch.randn(
                (batch_size, *probability.shape),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
            latent = independent if latent is None else latent + independent
        if latent is None:
            raise RuntimeError("copula latent field has no variance component")
        for signed in (latent, -latent):
            output.append((signed <= quantile).to(device="cpu").numpy())
        remaining_pairs -= batch_size
    masks = np.concatenate(output, axis=0)
    if masks.shape != (posterior_draws, *probability.shape):
        raise RuntimeError("spatial copula sampler emitted an invalid shape")
    return masks.astype(bool, copy=False), grid_shape


def spatial_copula_dice_confidence(
    probability,
    action,
    *,
    posterior_draws: int,
    repeat_index: int,
    global_variance_weight: float,
    spatial_variance_weight: float,
    spatial_knot_spacing_diagonal: float,
    posterior_batch_size: int = 8,
    master_seed: int = 20260721,
    sample_id: str = "sample",
    device: str = "cpu",
) -> SpatialCopulaDiceEstimate:
    """Estimate Dice confidence under a spatial Gaussian-copula posterior.

    The latent field is

    ``sqrt(a) G + sqrt(b) U_i + sqrt(1-a-b) epsilon_i``,

    where ``U`` is a standardized bilinear interpolation of iid Gaussian
    knots.  Every pixel therefore has a standard-normal latent marginal and
    ``Pr(Y_i=1)=p_i`` exactly under the sampling law.  Antithetic latent pairs
    reduce Monte Carlo error.  One call computes exactly one repeat so repeats
    can be scheduled as independent jobs.
    """

    probability, action = _validate_inputs(probability, action)
    numeric_values = {
        "global_variance_weight": global_variance_weight,
        "spatial_variance_weight": spatial_variance_weight,
        "spatial_knot_spacing_diagonal": spatial_knot_spacing_diagonal,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    global_variance_weight = float(global_variance_weight)
    spatial_variance_weight = float(spatial_variance_weight)
    spatial_knot_spacing_diagonal = float(spatial_knot_spacing_diagonal)
    if global_variance_weight < 0 or spatial_variance_weight < 0:
        raise ValueError("variance weights must be non-negative")
    if global_variance_weight + spatial_variance_weight > 1 + 1e-12:
        raise ValueError("global and spatial variance weights cannot sum above one")
    if not 0 < spatial_knot_spacing_diagonal <= 1:
        raise ValueError("spatial_knot_spacing_diagonal must lie in (0, 1]")
    if (
        isinstance(posterior_draws, bool)
        or not isinstance(posterior_draws, (int, np.integer))
        or posterior_draws <= 0
        or posterior_draws % 2
    ):
        raise ValueError("posterior_draws must be a positive even integer")
    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, (int, np.integer))
        or repeat_index < 0
    ):
        raise ValueError("repeat_index must be a non-negative integer")
    if (
        isinstance(posterior_batch_size, bool)
        or not isinstance(posterior_batch_size, (int, np.integer))
        or posterior_batch_size <= 0
    ):
        raise ValueError("posterior_batch_size must be a positive integer")
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, (int, np.integer))
        or master_seed < 0
    ):
        raise ValueError("master_seed must be a non-negative integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")

    # PyTorch supplies the same vectorized implementation on CPU and GPU.  It
    # is imported lazily so deterministic count-only scores remain lightweight.
    import torch
    import torch.nn.functional as functional

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    residual_variance_weight = max(
        0.0, 1.0 - global_variance_weight - spatial_variance_weight
    )
    spatial_grid_shape = (
        _spatial_grid_shape(probability.shape, spatial_knot_spacing_diagonal)
        if spatial_variance_weight > 0
        else None
    )
    probability_tensor = torch.as_tensor(
        probability, dtype=torch.float32, device=compute_device
    )
    quantile = torch.special.ndtri(probability_tensor)
    action_tensor = torch.as_tensor(action, dtype=torch.bool, device=compute_device)
    action_size = int(action.sum())

    spatial_standard_deviation = None
    if spatial_grid_shape is not None:
        row_variance = _linear_interpolation_variance(
            probability.shape[0],
            spatial_grid_shape[0],
            torch=torch,
            device=compute_device,
        )
        column_variance = _linear_interpolation_variance(
            probability.shape[1],
            spatial_grid_shape[1],
            torch=torch,
            device=compute_device,
        )
        spatial_standard_deviation = torch.sqrt(
            row_variance[:, None] * column_variance[None, :]
        )

    coupling_id = (
        "spatial-copula|"
        f"global={global_variance_weight.hex()}|"
        f"spatial={spatial_variance_weight.hex()}|"
        f"spacing={spatial_knot_spacing_diagonal.hex()}"
    )
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(
        _partition_seed(master_seed, str(sample_id), coupling_id, repeat_index)
        % (2**63 - 1)
    )

    loss_sum = 0.0
    remaining_pairs = posterior_draws // 2
    while remaining_pairs:
        batch_size = min(posterior_batch_size, remaining_pairs)
        latent = None
        if global_variance_weight > 0:
            global_component = torch.randn(
                (batch_size, 1, 1),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
            latent = math.sqrt(global_variance_weight) * global_component
        if spatial_variance_weight > 0:
            knot_field = torch.randn(
                (batch_size, 1, *spatial_grid_shape),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
            spatial_component = functional.interpolate(
                knot_field,
                size=probability.shape,
                mode="bilinear",
                align_corners=True,
            )[:, 0]
            spatial_component = spatial_component / spatial_standard_deviation
            weighted_spatial = math.sqrt(spatial_variance_weight) * spatial_component
            latent = weighted_spatial if latent is None else latent + weighted_spatial
        if residual_variance_weight > 0:
            independent_component = torch.randn(
                (batch_size, *probability.shape),
                generator=generator,
                device=compute_device,
                dtype=torch.float32,
            )
            weighted_independent = (
                math.sqrt(residual_variance_weight) * independent_component
            )
            latent = (
                weighted_independent
                if latent is None
                else latent + weighted_independent
            )
        if latent is None:
            raise RuntimeError("copula latent field has no variance component")

        for signed_latent in (latent, -latent):
            candidate = signed_latent <= quantile
            candidate_size = candidate.sum(dim=(-2, -1), dtype=torch.float64)
            overlap = torch.logical_and(candidate, action_tensor).sum(
                dim=(-2, -1), dtype=torch.float64
            )
            denominator = candidate_size + action_size
            losses = torch.where(
                denominator > 0,
                1 - 2 * overlap / denominator,
                torch.zeros_like(denominator),
            )
            loss_sum += float(losses.sum().item())
        remaining_pairs -= batch_size

    confidence = -loss_sum / posterior_draws
    if not math.isfinite(confidence) or not -1 <= confidence <= 0:
        raise RuntimeError("spatial-copula Dice confidence is invalid")
    return SpatialCopulaDiceEstimate(
        confidence=float(confidence), spatial_grid_shape=spatial_grid_shape
    )


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
