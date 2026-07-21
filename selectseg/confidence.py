"""Loss-indexed confidence and selective-risk utilities for binary masks.

This module deliberately has no multiclass or present-class aggregation.  A
single foreground probability map induces nested candidate labels

``Y_t = {i: p_i >= t}``,

and every candidate is compared with one *fixed* deployed hard mask.  Changing
the loss changes the confidence estimand; changing ``M`` changes only the
quadrature rule used to approximate it.

The boundary loss and boundary confidence use exactly the same convention:
HD95 is divided by the image diagonal, empty--empty costs zero, and a
one-sided empty pair costs one.  Thus both binary losses returned here lie in
``[0, 1]``.
"""

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


BinaryLoss = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class AURCSummary:
    """AURC and its oracle/random normalizations (all lower is better).

    ``normalized_aurc`` is normalized excess AURC,

    ``(aurc - oracle_aurc) / (random_aurc - oracle_aurc)``.

    It is ``None`` when every risk is equal, because the oracle and random
    denominators then coincide.  Consequently 0 is oracle performance, 1 is
    random performance, and values above 1 are possible for an anti-informative
    score.
    """

    aurc: float
    oracle_aurc: float
    random_aurc: float
    excess_aurc: float
    normalized_aurc: float | None


@dataclass(frozen=True)
class BootstrapAURCDifference:
    """Paired cluster-bootstrap result for ``AURC(left) - AURC(right)``.

    A positive difference means that the right-hand confidence has lower AURC.
    The interval is the percentile interval over paired cluster resamples.
    """

    difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    n_observations: int
    n_clusters: int
    seed: int


def _as_binary_mask(mask, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary mask, got shape {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must have non-empty spatial dimensions")
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain booleans or numeric 0/1 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def _as_probability_map(probability) -> np.ndarray:
    array = np.asarray(probability, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"foreground probability must be 2D, got shape {array.shape}"
        )
    if 0 in array.shape:
        raise ValueError("foreground probability must have non-empty dimensions")
    if not np.isfinite(array).all():
        raise ValueError("foreground probability contains a non-finite value")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("foreground probabilities must lie in [0, 1]")
    return array


def foreground_dice_loss(label, prediction) -> float:
    """Foreground Dice loss with explicit binary empty-mask conventions.

    The loss is zero for empty--empty, one for a one-sided empty pair, and
    ``1 - 2|A intersect B|/(|A|+|B|)`` otherwise.
    """

    label = _as_binary_mask(label, name="label")
    prediction = _as_binary_mask(prediction, name="prediction")
    if label.shape != prediction.shape:
        raise ValueError(
            f"label and prediction shapes differ: {label.shape} != {prediction.shape}"
        )
    denominator = int(label.sum()) + int(prediction.sum())
    if denominator == 0:
        return 0.0
    intersection = int(np.logical_and(label, prediction).sum())
    return float(1 - 2 * intersection / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def normalized_penalized_hd95(label, prediction) -> float:
    """Pooled bidirectional HD95 divided by the image diagonal.

    Both masks empty costs 0; exactly one empty costs 1.  For two non-empty
    masks, surface distances in both directions are pooled before taking the
    95th percentile, matching :func:`selectseg.metrics.hausdorff_95`, and the
    result is divided by ``hypot(height, width)``.  The returned loss is clipped
    at one only as a numerical safeguard.
    """

    label = _as_binary_mask(label, name="label")
    prediction = _as_binary_mask(prediction, name="prediction")
    if label.shape != prediction.shape:
        raise ValueError(
            f"label and prediction shapes differ: {label.shape} != {prediction.shape}"
        )
    label_present = bool(label.any())
    prediction_present = bool(prediction.any())
    if not label_present and not prediction_present:
        return 0.0
    if label_present != prediction_present:
        return 1.0

    label_surface = _surface(label)
    prediction_surface = _surface(prediction)
    to_prediction = ndimage.distance_transform_edt(~prediction_surface)[label_surface]
    to_label = ndimage.distance_transform_edt(~label_surface)[prediction_surface]
    distance = float(np.percentile(np.concatenate([to_prediction, to_label]), 95))
    diagonal = math.hypot(*label.shape)
    return float(min(1.0, distance / diagonal))


def validate_quadrature(nodes, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Validate a quadrature rule and return float arrays.

    Nodes must be finite, strictly increasing, and strictly inside ``(0, 1)``.
    Weights default to ``1/M``; explicit weights must be finite, non-negative,
    aligned with the nodes, and sum to one.  Zero-weight nodes are accepted
    because they do not change the estimand, though midpoint rules never use
    them.
    """

    node_array = np.asarray(nodes, dtype=float)
    if node_array.ndim != 1 or node_array.size == 0:
        raise ValueError("nodes must be a non-empty one-dimensional sequence")
    if not np.isfinite(node_array).all():
        raise ValueError("nodes contain a non-finite value")
    if np.any((node_array <= 0) | (node_array >= 1)):
        raise ValueError("quadrature nodes must lie strictly inside (0, 1)")
    if np.any(np.diff(node_array) <= 0):
        raise ValueError("quadrature nodes must be strictly increasing")

    if weights is None:
        weight_array = np.full(node_array.size, 1 / node_array.size, dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.ndim != 1 or weight_array.size != node_array.size:
            raise ValueError("weights must be one-dimensional and aligned with nodes")
        if not np.isfinite(weight_array).all():
            raise ValueError("weights contain a non-finite value")
        if np.any(weight_array < 0):
            raise ValueError("quadrature weights must be non-negative")
        if not np.isclose(weight_array.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("quadrature weights must sum to one")
    return node_array, weight_array


def midpoint_rule(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the equal-weight ``M``-point midpoint rule on ``(0, 1)``."""

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be a positive integer")
    if count <= 0:
        raise ValueError("count must be a positive integer")
    nodes = (np.arange(count, dtype=float) + 0.5) / count
    return nodes, np.full(count, 1 / count, dtype=float)


def levelset_risk(
    foreground_probability,
    hard_prediction,
    loss: BinaryLoss,
    nodes,
    weights=None,
) -> float:
    """Compute ``r_M = sum_m w_m L({p >= t_m}, hard_prediction)``.

    ``hard_prediction`` is fixed across all nodes: it is the mask actually
    deployed, not an additional quadrature sample.  The function is agnostic to
    the binary loss; passing :func:`foreground_dice_loss` or
    :func:`normalized_penalized_hd95` changes only the estimand, while using
    ``midpoint_rule(2)`` versus ``midpoint_rule(32)`` changes only numerical
    quadrature.
    """

    probability = _as_probability_map(foreground_probability)
    prediction = _as_binary_mask(hard_prediction, name="hard_prediction")
    if probability.shape != prediction.shape:
        raise ValueError(
            "foreground probability and hard prediction shapes differ: "
            f"{probability.shape} != {prediction.shape}"
        )
    if not callable(loss):
        raise TypeError("loss must be callable")
    node_array, weight_array = validate_quadrature(nodes, weights)

    values = np.empty(node_array.size, dtype=float)
    for position, node in enumerate(node_array):
        value = np.asarray(loss(probability >= node, prediction))
        if value.ndim != 0:
            raise ValueError("loss must return one scalar per mask pair")
        values[position] = float(value)
    if not np.isfinite(values).all():
        raise ValueError("loss returned a non-finite value")
    return float(np.dot(weight_array, values))


def midpoint_loss_indexed_confidences(
    foreground_probability,
    hard_prediction,
    counts: Sequence[int] = (2, 8, 32),
) -> dict[int, dict[str, float]]:
    """Compute Dice- and nHD95-indexed midpoint scores with shared geometry.

    This is the production counterpart of repeated :func:`levelset_risk`
    calls.  It evaluates the union of all requested midpoint nodes once,
    reuses the fixed prediction surface and its distance transform, and then
    aggregates the same node-loss table for each ``M``.  The returned values
    are confidences (negative risks) under keys ``"dice"`` and ``"nhd95"``.

    The optimized path is mathematically identical to calling
    :func:`levelset_risk` separately with :func:`foreground_dice_loss` and
    :func:`normalized_penalized_hd95`; it only avoids duplicate masks and EDTs.
    """

    probability = _as_probability_map(foreground_probability)
    prediction = _as_binary_mask(hard_prediction, name="hard_prediction")
    if probability.shape != prediction.shape:
        raise ValueError(
            "foreground probability and hard prediction shapes differ: "
            f"{probability.shape} != {prediction.shape}"
        )
    requested = list(counts)
    if not requested:
        raise ValueError("counts must be non-empty")
    if len(set(requested)) != len(requested):
        raise ValueError("counts cannot contain duplicates")
    rules = {count: midpoint_rule(count) for count in requested}
    all_nodes = np.unique(np.concatenate([nodes for nodes, _ in rules.values()]))

    diagonal = math.hypot(*probability.shape)
    prediction_size = int(prediction.sum())
    prediction_surface = _surface(prediction) if prediction_size else None
    to_prediction = (
        ndimage.distance_transform_edt(~prediction_surface)
        if prediction_size
        else None
    )
    dice_by_node = np.empty(all_nodes.size, dtype=float)
    nhd95_by_node = np.empty(all_nodes.size, dtype=float)
    for position, node in enumerate(all_nodes):
        level = probability >= node
        level_size = int(level.sum())
        denominator = level_size + prediction_size
        if denominator == 0:
            dice_by_node[position] = 0.0
            nhd95_by_node[position] = 0.0
            continue
        intersection = int(np.logical_and(level, prediction).sum())
        dice_by_node[position] = 1 - 2 * intersection / denominator
        if level_size == 0 or prediction_size == 0:
            nhd95_by_node[position] = 1.0
            continue
        level_surface = _surface(level)
        to_level = ndimage.distance_transform_edt(~level_surface)
        distances = np.concatenate(
            [to_prediction[level_surface], to_level[prediction_surface]]
        )
        nhd95_by_node[position] = min(
            1.0, float(np.percentile(distances, 95)) / diagonal
        )

    result = {}
    for count, (nodes, weights) in rules.items():
        positions = np.searchsorted(all_nodes, nodes)
        if not np.array_equal(all_nodes[positions], nodes):
            raise AssertionError("midpoint node lookup lost exact values")
        result[int(count)] = {
            "dice": -float(np.dot(weights, dice_by_node[positions])),
            "nhd95": -float(np.dot(weights, nhd95_by_node[positions])),
        }
    return result


def soft_dice_confidence(foreground_probability, hard_prediction) -> float:
    """Soft Dice Confidence (ratio of expectations) for one binary foreground.

    The zero-denominator convention is 0, matching the existing SDC baseline's
    empty-prediction convention.  This is intentionally distinct from expected
    Dice under the nested level-set posterior.
    """

    probability = _as_probability_map(foreground_probability)
    prediction = _as_binary_mask(hard_prediction, name="hard_prediction")
    if probability.shape != prediction.shape:
        raise ValueError(
            "foreground probability and hard prediction shapes differ: "
            f"{probability.shape} != {prediction.shape}"
        )
    denominator = float(probability.sum()) + int(prediction.sum())
    if denominator == 0:
        return 0.0
    return float(2 * probability[prediction].sum() / denominator)


def _validated_score_and_risk(confidences, risks) -> tuple[np.ndarray, np.ndarray]:
    confidence_array = np.asarray(confidences, dtype=float)
    risk_array = np.asarray(risks, dtype=float)
    if confidence_array.ndim != 1 or risk_array.ndim != 1:
        raise ValueError("confidences and risks must be one-dimensional")
    if confidence_array.size == 0:
        raise ValueError("confidences and risks must be non-empty")
    if confidence_array.size != risk_array.size:
        raise ValueError("confidences and risks must have the same length")
    if not np.isfinite(confidence_array).all() or not np.isfinite(risk_array).all():
        raise ValueError("confidences and risks must be finite")
    return confidence_array, risk_array


def tie_aware_expected_aurc(confidences, risks) -> float:
    """Expected AURC under a uniform random ordering within every score tie.

    For a tied group with mean risk ``mu``, size ``g``, and cumulative risk
    ``S`` before the group, the expected cumulative risk after accepting ``j``
    of its members is ``S + j*mu``.  Summing these expected prefix risks gives
    the exact expectation over all within-group permutations without sampling.
    Consequently this AURC is invariant to input row order.
    """

    confidence_array, risk_array = _validated_score_and_risk(confidences, risks)
    order = np.argsort(-confidence_array, kind="stable")
    sorted_confidence = confidence_array[order]
    sorted_risk = risk_array[order]
    count = sorted_confidence.size

    # Map every sorted observation to its exact-tie group.  Vectorizing the
    # expected-prefix formula matters because this function runs twice inside
    # every bootstrap replicate.
    group_start = np.empty(count, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_confidence[1:] != sorted_confidence[:-1]
    group_index = np.cumsum(group_start) - 1
    group_count = np.bincount(group_index)
    group_sum = np.bincount(group_index, weights=sorted_risk)
    group_mean = group_sum / group_count
    first_position = np.cumsum(group_count) - group_count
    previous_risk = np.cumsum(group_sum) - group_sum

    position = np.arange(count)
    within_group = position - first_position[group_index] + 1
    expected_cumulative_risk = (
        previous_risk[group_index] + within_group * group_mean[group_index]
    )
    return float(np.mean(expected_cumulative_risk / (position + 1)))


def summarize_aurc(confidences, risks) -> AURCSummary:
    """Return score, oracle, random, excess, and normalized-excess AURC."""

    confidence_array, risk_array = _validated_score_and_risk(confidences, risks)
    score = tie_aware_expected_aurc(confidence_array, risk_array)
    oracle = tie_aware_expected_aurc(-risk_array, risk_array)
    random = float(risk_array.mean())
    excess = score - oracle
    denominator = random - oracle
    normalized = None
    if not np.isclose(denominator, 0.0, rtol=1e-12, atol=1e-15):
        normalized = float(excess / denominator)
    return AURCSummary(
        aurc=score,
        oracle_aurc=oracle,
        random_aurc=random,
        excess_aurc=excess,
        normalized_aurc=normalized,
    )


def _cluster_groups(
    cluster_ids: Sequence[Hashable] | None, count: int
) -> list[np.ndarray]:
    if cluster_ids is None:
        return [np.array([index], dtype=int) for index in range(count)]
    if isinstance(cluster_ids, np.ndarray) and cluster_ids.ndim != 1:
        raise ValueError("cluster_ids must be one-dimensional and match the data length")
    ids = list(cluster_ids)
    if len(ids) != count:
        raise ValueError("cluster_ids must be one-dimensional and match the data length")

    locations: dict[Hashable, list[int]] = {}
    for index, raw_key in enumerate(ids):
        key = raw_key.item() if isinstance(raw_key, np.generic) else raw_key
        if key is None or (isinstance(key, (float, np.floating)) and np.isnan(key)):
            raise ValueError("cluster_ids cannot contain missing values")
        try:
            hash(key)
        except TypeError as error:
            raise TypeError("every cluster id must be hashable") from error
        locations.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=int) for indices in locations.values()]


def paired_cluster_bootstrap_aurc_difference(
    left_confidences,
    right_confidences,
    risks,
    *,
    cluster_ids: Sequence[Hashable] | None = None,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapAURCDifference:
    """Paired cluster bootstrap of ``AURC(left) - AURC(right)``.

    Clusters (normally image IDs) are sampled with replacement, and every row
    belonging to a selected image is copied together.  The same sampled rows
    are used for both confidence scores and the risk, preserving pairing.  If
    ``cluster_ids`` is omitted, each row forms its own cluster.  Every AURC in
    the bootstrap uses :func:`tie_aware_expected_aurc`.
    """

    left, risk_array = _validated_score_and_risk(left_confidences, risks)
    right, right_risk = _validated_score_and_risk(right_confidences, risks)
    if not np.array_equal(risk_array, right_risk):  # defensive: helper saw same input
        raise AssertionError("internal risk validation mismatch")
    if isinstance(n_resamples, bool) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise TypeError("n_resamples must be a positive integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    groups = _cluster_groups(cluster_ids, risk_array.size)
    cluster_count = len(groups)
    singleton_clusters = cluster_count == risk_array.size and all(
        group.size == 1 for group in groups
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for replicate in range(n_resamples):
        if singleton_clusters:
            # Native binary evaluation has one row per image. Avoid building
            # thousands of one-element arrays in every bootstrap replicate.
            indices = rng.integers(0, cluster_count, size=cluster_count)
        else:
            sampled = rng.integers(0, cluster_count, size=cluster_count)
            indices = np.concatenate([groups[position] for position in sampled])
        draws[replicate] = tie_aware_expected_aurc(
            left[indices], risk_array[indices]
        ) - tie_aware_expected_aurc(right[indices], risk_array[indices])

    tail = (1 - confidence_level) / 2
    ci_low, ci_high = np.quantile(draws, [tail, 1 - tail])
    difference = tie_aware_expected_aurc(left, risk_array) - tie_aware_expected_aurc(
        right, risk_array
    )
    return BootstrapAURCDifference(
        difference=float(difference),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        n_observations=int(risk_array.size),
        n_clusters=cluster_count,
        seed=int(seed),
    )
