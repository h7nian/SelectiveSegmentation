"""Exact, lightweight checks for the binary theory in the manuscript."""

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import linprog

from selectseg.binary_baselines import exact_levelset_dice_confidence
from selectseg.binary_framework import foreground_dice_loss


def _piecewise_integral(values, widths):
    return float(np.dot(np.asarray(values, dtype=float), widths))


def _lower_tail_selector(values, masses, coverage):
    """Population quantile selector, constant on every exact-value tie."""

    values = np.asarray(values, dtype=float)
    masses = np.asarray(masses, dtype=float)
    selector = np.zeros(values.size, dtype=float)
    remaining = float(coverage)
    for value in np.unique(values):
        tied = values == value
        tied_mass = float(masses[tied].sum())
        accepted = min(remaining, tied_mass)
        selector[tied] = accepted / tied_mass
        remaining -= accepted
        if remaining <= 1e-14:
            break
    assert abs(float(np.dot(masses, selector)) - coverage) <= 1e-12
    return selector


def _selective_risk(selector, risks, masses, coverage):
    return float(np.dot(masses * selector, risks) / coverage)


def _finite_population_aurc(ordering_risk, true_risk, masses):
    """Integrate population selective risk exactly across tied score blocks."""

    ordering_risk = np.asarray(ordering_risk, dtype=float)
    true_risk = np.asarray(true_risk, dtype=float)
    masses = np.asarray(masses, dtype=float)
    cumulative_mass = 0.0
    cumulative_loss = 0.0
    integral = 0.0
    for value in np.unique(ordering_risk):
        tied = ordering_risk == value
        group_mass = float(masses[tied].sum())
        group_loss = float(np.dot(masses[tied], true_risk[tied]))
        group_mean = group_loss / group_mass
        next_mass = cumulative_mass + group_mass
        coefficient = cumulative_loss - cumulative_mass * group_mean
        integral += group_mean * group_mass
        if cumulative_mass > 0:
            integral += coefficient * math.log(next_mass / cumulative_mass)
        else:
            assert coefficient == pytest.approx(0.0)
        cumulative_mass = next_mass
        cumulative_loss += group_loss
    assert cumulative_mass == pytest.approx(1.0)
    return integral


def _integrated_regret_envelope(error, bound=1.0):
    if error <= 0:
        return 0.0
    if error >= bound:
        return bound
    return error * (1 + math.log(bound / error))


def _jaccard_distance(left, right):
    union = left | right
    if union == 0:
        return 0.0
    return (left ^ right).bit_count() / union.bit_count()


def _dice_loss_bits(label, action):
    denominator = label.bit_count() + action.bit_count()
    if denominator == 0:
        return 0.0
    return 1 - 2 * (label & action).bit_count() / denominator


def _levelset_mask_distribution(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    distribution = np.zeros(1 << probabilities.size, dtype=float)
    boundaries = np.unique(np.r_[0.0, probabilities, 1.0])
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == lower:
            continue
        threshold = (lower + upper) / 2
        mask = sum(
            1 << index
            for index, probability in enumerate(probabilities)
            if probability >= threshold
        )
        distribution[mask] += upper - lower
    assert distribution.sum() == pytest.approx(1.0)
    return distribution


def _dice_fixed_action_moments(distribution, action, pixel_count):
    action_size = action.bit_count()
    beta = np.zeros(pixel_count + 1, dtype=float)
    marginals = np.zeros(pixel_count, dtype=float)
    reciprocal_numerators = np.zeros(pixel_count, dtype=float)
    expected_dice = 0.0
    for label, mass in enumerate(distribution):
        label_size = label.bit_count()
        expected_dice += mass * (1 - _dice_loss_bits(label, action))
        beta[label_size] += mass * (label & action).bit_count()
        if action_size > 0:
            for index in range(pixel_count):
                if label & (1 << index):
                    marginals[index] += mass
                    reciprocal_numerators[index] += mass / (
                        label_size + action_size
                    )
    kappa = np.divide(
        reciprocal_numerators,
        marginals,
        out=np.zeros_like(marginals),
        where=marginals > 0,
    )
    return expected_dice, beta, marginals, kappa


def _finite_wasserstein(probability, approximation, costs):
    size = len(probability)
    equality = np.zeros((2 * size, size * size), dtype=float)
    for left in range(size):
        equality[left, left * size : (left + 1) * size] = 1
    for right in range(size):
        equality[size + right, right::size] = 1
    result = linprog(
        np.asarray(costs, dtype=float).ravel(),
        A_eq=equality,
        b_eq=np.r_[probability, approximation],
        bounds=(0, None),
        method="highs",
    )
    assert result.success
    return float(result.fun)


def _direct_piecewise_dice_confidence(probability, prediction):
    boundaries = np.unique(
        np.r_[0.0, np.asarray(probability, dtype=float).ravel(), 1.0]
    )
    risk = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == lower:
            continue
        threshold = (lower + upper) / 2
        risk += (upper - lower) * foreground_dice_loss(
            probability >= threshold, prediction
        )
    return -risk


def _normalized_penalized_full_hausdorff(left, right, diagonal):
    left = np.asarray(left, dtype=float).reshape(-1, 2)
    right = np.asarray(right, dtype=float).reshape(-1, 2)
    if left.size == 0 and right.size == 0:
        return 0.0
    if (left.size == 0) != (right.size == 0):
        return 1.0
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()) / diagonal)


def _pooled_surface_quantile(left, right, quantile, diagonal):
    left = np.asarray(left, dtype=float).reshape(-1, 2)
    right = np.asarray(right, dtype=float).reshape(-1, 2)
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    pooled = np.r_[distances.min(axis=1), distances.min(axis=0)]
    return float(np.percentile(pooled, 100 * quantile) / diagonal)


def test_uniform_threshold_is_unique_for_arbitrary_one_pixel_maps():
    """A nonuniform threshold law changes the advertised pixel marginal."""

    probability = 0.37
    uniform_inclusion = probability
    triangular_inclusion = probability**2  # density q(t)=2t, CDF F(u)=u^2

    assert uniform_inclusion == pytest.approx(probability)
    assert triangular_inclusion == pytest.approx(0.1369)
    assert triangular_inclusion != pytest.approx(probability)


def test_importance_moments_and_chi_square_identity_exactly():
    """Pin unbiasedness, variance, and the chi-squared identity without MC."""

    widths = np.array([0.5, 0.5])
    integrand = np.array([1.0, 2.0])
    proposal = np.array([0.5, 1.5])
    sample_count = 11

    risk = _piecewise_integral(integrand, widths)
    second_moment = _piecewise_integral(integrand**2 / proposal, widths)
    estimator_variance = (second_moment - risk**2) / sample_count

    oracle = integrand / risk
    chi_square = _piecewise_integral((oracle - proposal) ** 2 / proposal, widths)

    assert _piecewise_integral(proposal, widths) == pytest.approx(1.0)
    assert risk == pytest.approx(3 / 2)
    assert second_moment == pytest.approx(7 / 3)
    assert estimator_variance == pytest.approx(1 / (12 * sample_count))
    assert chi_square == pytest.approx(1 / 27)
    assert risk**2 * chi_square / sample_count == pytest.approx(
        estimator_variance
    )


def test_oracle_proposal_has_zero_variance():
    """Under q*=H/integral(H), every nonzero weighted draw is constant."""

    integrand = np.array([1.0, 2.0])
    risk = 3 / 2
    oracle = integrand / risk
    weighted_draw_values = integrand / oracle

    assert oracle == pytest.approx([2 / 3, 4 / 3])
    assert weighted_draw_values == pytest.approx([risk, risk])
    assert np.var(weighted_draw_values) == pytest.approx(0.0)


def test_defensive_mixture_range_variance_and_tail():
    """Check the defensive floor and one exact finite-sample Hoeffding case."""

    alpha = 0.2
    widths = np.array([0.5, 0.5])
    integrand = np.array([1.0, 2.0])
    candidate = np.array([2.0, 0.0])
    proposal = (1 - alpha) * candidate + alpha
    weighted_values = integrand / proposal

    proposal_masses = widths * proposal
    mean = float(np.dot(proposal_masses, weighted_values))
    variance = float(np.dot(proposal_masses, weighted_values**2) - mean**2)

    assert proposal == pytest.approx([1.8, 0.2])
    assert proposal.min() >= alpha
    assert weighted_values.max() <= integrand.max() / alpha
    assert mean == pytest.approx(3 / 2)
    assert variance == pytest.approx(289 / 36)
    assert variance <= integrand.max() ** 2 / alpha

    sample_count = 5
    deviation = 4.0
    right_mass = float(proposal_masses[1])
    exact_tail = sum(
        math.comb(sample_count, count)
        * right_mass**count
        * (1 - right_mass) ** (sample_count - count)
        for count in range(3, sample_count + 1)
    )
    hoeffding = 2 * math.exp(
        -2
        * sample_count
        * alpha**2
        * deviation**2
        / integrand.max() ** 2
    )

    assert exact_tail == pytest.approx(0.00856)
    assert exact_tail <= hoeffding


def test_population_total_error_on_two_image_atoms():
    """The population bound averages the per-image numerical/model split."""

    population_mass = np.array([0.4, 0.6])
    total_error = np.array([0.3, 0.4])
    variation = np.array([1.0, 1.0])
    posterior_discrepancy = np.array([0.1, 0.2])
    midpoint_count = 2

    observed = float(np.dot(population_mass, total_error))
    bound = float(
        np.dot(population_mass, variation) / (2 * midpoint_count)
        + np.dot(population_mass, posterior_discrepancy)
    )

    assert observed == pytest.approx(0.36)
    assert bound == pytest.approx(0.41)
    assert observed <= bound


def test_l1_score_error_controls_coverage_regret_with_atoms_and_ties():
    """Exhaust the finite three-atom risks, including fractional tie blocks."""

    masses = np.full(3, 1 / 3)
    coverages = (0.01, 1 / 6, 1 / 3, 0.5, 5 / 6, 1.0)
    grid = (0.0, 0.5, 1.0)
    for true_tuple, approximate_tuple in itertools.product(
        itertools.product(grid, repeat=3), repeat=2
    ):
        true_risk = np.asarray(true_tuple)
        approximate_risk = np.asarray(approximate_tuple)
        l1_error = float(np.dot(masses, np.abs(true_risk - approximate_risk)))
        uniform_error = float(np.max(np.abs(true_risk - approximate_risk)))
        for coverage in coverages:
            oracle = _lower_tail_selector(true_risk, masses, coverage)
            plug_in = _lower_tail_selector(approximate_risk, masses, coverage)
            regret = _selective_risk(
                plug_in, true_risk, masses, coverage
            ) - _selective_risk(oracle, true_risk, masses, coverage)
            assert regret >= -1e-12
            # The factor-one L1 result is stronger than the initially proposed
            # 2*eta/c inequality and remains valid for fractional selectors.
            assert regret <= min(1.0, l1_error / coverage) + 1e-12
            assert regret <= min(1.0, 2 * uniform_error) + 1e-12


def test_l1_score_error_controls_population_aurc_regret_exactly():
    """Integrate every three-atom risk curve analytically, without a grid in c."""

    masses = np.full(3, 1 / 3)
    grid = (0.0, 0.5, 1.0)
    for true_tuple, approximate_tuple in itertools.product(
        itertools.product(grid, repeat=3), repeat=2
    ):
        true_risk = np.asarray(true_tuple)
        approximate_risk = np.asarray(approximate_tuple)
        l1_error = float(np.dot(masses, np.abs(true_risk - approximate_risk)))
        regret = _finite_population_aurc(
            approximate_risk, true_risk, masses
        ) - _finite_population_aurc(true_risk, true_risk, masses)
        assert regret >= -1e-12
        assert regret <= _integrated_regret_envelope(l1_error) + 1e-12


def test_nonuniform_atoms_use_randomization_across_the_whole_score_tie():
    true_risk = np.array([0.0, 0.4, 1.0])
    approximate_risk = np.array([0.2, 0.2, 0.8])
    masses = np.array([0.2, 0.3, 0.5])
    coverage = 0.35

    selector = _lower_tail_selector(approximate_risk, masses, coverage)
    assert selector == pytest.approx([0.7, 0.7, 0.0])
    oracle = _lower_tail_selector(true_risk, masses, coverage)
    regret = _selective_risk(
        selector, true_risk, masses, coverage
    ) - _selective_risk(oracle, true_risk, masses, coverage)
    l1_error = float(np.dot(masses, np.abs(true_risk - approximate_risk)))
    assert regret >= 0
    assert regret <= l1_error / coverage


def test_dice_is_two_lipschitz_in_jaccard_geometry_including_empty_masks():
    masks = range(1 << 4)
    for action, left, right in itertools.product(masks, repeat=3):
        difference = abs(
            _dice_loss_bits(left, action) - _dice_loss_bits(right, action)
        )
        assert difference <= 2 * _jaccard_distance(left, right) + 1e-12

    # The constant approaches two: add one action pixel to a large disjoint mask.
    background_size = 20
    action = 1
    left = sum(1 << index for index in range(1, background_size + 1))
    right = left | action
    ratio = abs(
        _dice_loss_bits(left, action) - _dice_loss_bits(right, action)
    ) / _jaccard_distance(left, right)
    assert ratio > 1.9


@pytest.mark.parametrize("seed", range(4))
def test_dice_posterior_discrepancy_is_bounded_by_jaccard_wasserstein(seed):
    rng = np.random.default_rng(seed)
    masks = tuple(range(1 << 3))
    probability = rng.dirichlet(np.ones(len(masks)))
    approximation = rng.dirichlet(np.ones(len(masks)))
    costs = np.array(
        [[_jaccard_distance(left, right) for right in masks] for left in masks]
    )
    wasserstein = _finite_wasserstein(probability, approximation, costs)
    total_variation = 0.5 * float(np.abs(probability - approximation).sum())

    assert wasserstein <= total_variation + 1e-10
    for action in masks:
        loss = np.array([_dice_loss_bits(label, action) for label in masks])
        discrepancy = abs(float(np.dot(probability - approximation, loss)))
        assert discrepancy <= 2 * wasserstein + 1e-9
        assert discrepancy <= total_variation + 1e-10


def test_jaccard_wasserstein_can_be_small_when_total_variation_is_one():
    background_size = 100
    left = sum(1 << index for index in range(background_size))
    right = left | (1 << background_size)
    total_variation = 1.0  # distinct point masses
    wasserstein = _jaccard_distance(left, right)

    assert wasserstein == pytest.approx(1 / (background_size + 1))
    assert wasserstein < total_variation / 100


@pytest.mark.parametrize("seed", range(4))
def test_fixed_action_dice_moment_identities_and_bounds(seed):
    rng = np.random.default_rng(seed)
    pixel_count = 3
    masks = 1 << pixel_count
    true_distribution = rng.dirichlet(np.ones(masks))
    probabilities = rng.uniform(0.05, 0.95, size=pixel_count)
    levelset_distribution = _levelset_mask_distribution(probabilities)

    for action in range(1, masks):
        action_size = action.bit_count()
        true_dice, true_beta, true_marginals, true_kappa = (
            _dice_fixed_action_moments(
                true_distribution, action, pixel_count
            )
        )
        levelset_dice, levelset_beta, levelset_marginals, levelset_kappa = (
            _dice_fixed_action_moments(
                levelset_distribution, action, pixel_count
            )
        )
        assert levelset_marginals == pytest.approx(probabilities)

        true_beta_identity = 2 * sum(
            true_beta[size] / (size + action_size)
            for size in range(1, pixel_count + 1)
        )
        true_kappa_identity = 2 * sum(
            true_marginals[index] * true_kappa[index]
            for index in range(pixel_count)
            if action & (1 << index)
        )
        assert true_dice == pytest.approx(true_beta_identity)
        assert true_dice == pytest.approx(true_kappa_identity)

        discrepancy = abs(true_dice - levelset_dice)
        beta_bound = 2 * sum(
            abs(true_beta[size] - levelset_beta[size])
            / (size + action_size)
            for size in range(1, pixel_count + 1)
        )
        reciprocal_bound = (
            2
            / (action_size + 1)
            * sum(
                abs(true_marginals[index] - probabilities[index])
                for index in range(pixel_count)
                if action & (1 << index)
            )
            + 2
            * sum(
                probabilities[index]
                * abs(true_kappa[index] - levelset_kappa[index])
                for index in range(pixel_count)
                if action & (1 << index)
            )
        )
        assert discrepancy <= beta_bound + 1e-12
        assert discrepancy <= reciprocal_bound + 1e-12


def test_fixed_action_dice_empty_branch_and_independence_nonordering():
    probabilities = np.array([0.5, 0.5])
    levelset = _levelset_mask_distribution(probabilities)
    independent = np.full(4, 0.25)

    empty_discrepancy = abs(independent[0] - levelset[0])
    assert levelset[0] == pytest.approx(1 - probabilities.max())
    assert empty_discrepancy == pytest.approx(0.25)

    action = 1
    _, _, _, independent_kappa = _dice_fixed_action_moments(
        independent, action, pixel_count=2
    )
    _, _, _, levelset_kappa = _dice_fixed_action_moments(
        levelset, action, pixel_count=2
    )
    assert independent_kappa[0] == pytest.approx(5 / 12)
    assert levelset_kappa[0] == pytest.approx(1 / 3)
    assert independent_kappa[0] != pytest.approx(levelset_kappa[0])


@pytest.mark.parametrize("seed", range(5))
def test_exact_dice_matches_piecewise_threshold_integration_with_ties(seed):
    rng = np.random.default_rng(seed)
    probability = rng.choice(
        [0.0, 0.1, 0.35, 0.35, 0.8, 1.0], size=(5, 7)
    )
    prediction = rng.random(probability.shape) > 0.55
    exact = exact_levelset_dice_confidence(probability, prediction)
    direct = _direct_piecewise_dice_confidence(probability, prediction)
    assert exact == pytest.approx(direct, abs=1e-14)


def test_exact_dice_matches_dense_quadrature_and_all_empty_endpoints():
    probability = np.array(
        [[0.0, 0.13, 0.13, 0.41], [0.57, 0.82, 1.0, 1.0]]
    )
    prediction = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool
    )
    exact = exact_levelset_dice_confidence(probability, prediction)
    count = 20_000
    nodes = (np.arange(count, dtype=float) + 0.5) / count
    levels = probability[None, :, :] >= nodes[:, None, None]
    level_sizes = levels.sum(axis=(1, 2))
    prediction_size = int(prediction.sum())
    intersections = np.logical_and(levels, prediction).sum(axis=(1, 2))
    dense_losses = 1 - 2 * intersections / (level_sizes + prediction_size)
    assert exact == pytest.approx(-float(dense_losses.mean()), abs=3e-4)

    empty = np.zeros((2, 3))
    assert exact_levelset_dice_confidence(empty, np.zeros((2, 3))) == 0.0
    assert exact_levelset_dice_confidence(empty, np.ones((2, 3))) == -1.0
    saturated = np.ones((2, 3))
    assert exact_levelset_dice_confidence(saturated, np.ones((2, 3))) == 0.0


def test_penalized_full_hausdorff_is_a_metric_on_surface_sets():
    points = ((0.0, 0.0), (2.0, 0.0), (5.0, 0.0))
    surfaces = [
        tuple(points[index] for index in range(len(points)) if bits & (1 << index))
        for bits in range(1 << len(points))
    ]
    diagonal = 5.0
    distances = np.array(
        [
            [
                _normalized_penalized_full_hausdorff(left, right, diagonal)
                for right in surfaces
            ]
            for left in surfaces
        ]
    )
    assert distances.min() >= 0
    assert distances.max() <= 1
    assert np.diag(distances) == pytest.approx(0.0)
    assert distances == pytest.approx(distances.T)
    for left, middle, right in itertools.product(range(len(surfaces)), repeat=3):
        assert distances[left, right] <= (
            distances[left, middle] + distances[middle, right] + 1e-12
        )
        assert abs(distances[left, middle] - distances[right, middle]) <= (
            distances[left, right] + 1e-12
        )


@pytest.mark.parametrize("seed", range(4))
def test_full_hausdorff_posterior_discrepancy_has_wasserstein_constant_one(seed):
    rng = np.random.default_rng(seed)
    points = ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0))
    surfaces = [
        tuple(points[index] for index in range(len(points)) if bits & (1 << index))
        for bits in range(1 << len(points))
    ]
    costs = np.array(
        [
            [
                _normalized_penalized_full_hausdorff(left, right, 3.0)
                for right in surfaces
            ]
            for left in surfaces
        ]
    )
    probability = rng.dirichlet(np.ones(len(surfaces)))
    approximation = rng.dirichlet(np.ones(len(surfaces)))
    wasserstein = _finite_wasserstein(probability, approximation, costs)
    total_variation = 0.5 * float(np.abs(probability - approximation).sum())
    assert wasserstein <= total_variation + 1e-10
    for action in range(len(surfaces)):
        discrepancy = abs(
            float(np.dot(probability - approximation, costs[:, action]))
        )
        assert discrepancy <= wasserstein + 1e-9


def test_pooled_hd95_can_jump_under_small_full_hausdorff_perturbation():
    # A has 100 exact matches. Y adds one remote point, which is below the 5%
    # tail; Y' adds 20 points within four pixels of that point, changing only
    # the tail multiplicity. Full Hausdorff(Y,Y') is tiny after normalization,
    # while the pooled 95th percentile jumps by an order-one amount.
    action = np.array([(2 * index, 0) for index in range(100)], dtype=float)
    remote = np.array([[5000.0, 5000.0]])
    offsets = np.array(
        [
            (x, y)
            for x in range(-4, 5)
            for y in range(-4, 5)
            if 0 < x * x + y * y <= 16
        ][:20],
        dtype=float,
    )
    label = np.vstack([action, remote])
    perturbed = np.vstack([label, remote + offsets])
    diagonal = 10_000.0

    baseline_hd95 = _pooled_surface_quantile(label, action, 0.95, diagonal)
    perturbed_hd95 = _pooled_surface_quantile(
        perturbed, action, 0.95, diagonal
    )
    ground_change = _normalized_penalized_full_hausdorff(
        label, perturbed, diagonal
    )
    assert baseline_hd95 == 0.0
    assert perturbed_hd95 > 0.6
    assert ground_change <= 4 / diagonal
    assert perturbed_hd95 - baseline_hd95 > 1000 * ground_change
