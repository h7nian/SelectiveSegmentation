import itertools
import json

import numpy as np
import pytest

from scripts.analyze.counts import sdc_bound_audit
from scripts.render.counts import (
    COPULA_METHODS,
    DATASETS,
    MODELS,
    PRIMARY_COPULA_VARIANT,
    _copula_table,
    _load_copula,
)
from selectseg.confidence import foreground_dice_loss, midpoint_rule
from selectseg.counts import (
    action_two_block_dice_confidence,
    count_ladders,
    second_order_dice_similarity,
    sample_spatial_copula_masks,
    shared_threshold_dice_confidence,
    spatial_copula_dice_confidence,
)


def test_shared_threshold_matches_direct_mask_evaluation():
    probability = np.array([[0.1, 0.4], [0.6, 0.9]])
    action = probability >= 0.5
    nodes, _ = midpoint_rule(8)
    expected = -np.mean(
        [foreground_dice_loss(probability >= node, action) for node in nodes]
    )
    assert shared_threshold_dice_confidence(probability, action, m=8) == pytest.approx(
        expected
    )


def test_two_block_matches_cartesian_mask_evaluation():
    probability = np.array([[0.05, 0.35, 0.49], [0.51, 0.72, 0.98]])
    action = probability >= 0.5
    nodes, _ = midpoint_rule(4)
    losses = []
    for inside, outside in itertools.product(nodes, repeat=2):
        candidate = np.where(action, probability >= inside, probability >= outside)
        losses.append(foreground_dice_loss(candidate, action))
    assert action_two_block_dice_confidence(
        probability, action, m=4
    ) == pytest.approx(-np.mean(losses))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_empty_and_full_constant_maps_are_total(value):
    probability = np.full((3, 4), value)
    action = probability >= 0.5
    assert shared_threshold_dice_confidence(probability, action) == 0.0
    assert action_two_block_dice_confidence(probability, action) == 0.0


def test_count_ladders_partition_every_candidate_count():
    probability = np.array([[0.1, 0.4], [0.6, 0.9]])
    action = np.array([[False, True], [True, False]])
    overlap, outside = count_ladders(probability, action, m=2)
    nodes, _ = midpoint_rule(2)
    for index, node in enumerate(nodes):
        candidate = probability >= node
        assert overlap[index] == np.logical_and(candidate, action).sum()
        assert outside[index] == np.logical_and(candidate, ~action).sum()


def test_second_order_reduces_to_soft_dice_when_variances_are_zero():
    assert second_order_dice_similarity(10, 8, 2, 0, 0, 0) == pytest.approx(
        16 / 20
    )


def test_inputs_fail_closed():
    probability = np.ones((2, 2))
    with pytest.raises(TypeError):
        action_two_block_dice_confidence(probability, np.ones((2, 2)))
    probability[0, 0] = np.nan
    with pytest.raises(ValueError):
        shared_threshold_dice_confidence(probability, np.ones((2, 2), dtype=bool))


@pytest.mark.parametrize(
    ("global_weight", "spatial_weight"),
    [(1.0, 0.0), (0.0, 0.0), (0.25, 0.5)],
)
def test_spatial_copula_preserves_single_pixel_marginal(
    global_weight, spatial_weight
):
    probability = np.array([[0.3]])
    action = np.array([[True]])
    estimate = spatial_copula_dice_confidence(
        probability,
        action,
        posterior_draws=20_000,
        repeat_index=0,
        global_variance_weight=global_weight,
        spatial_variance_weight=spatial_weight,
        spatial_knot_spacing_diagonal=0.1,
        posterior_batch_size=1_000,
        master_seed=17,
        device="cpu",
    )
    assert estimate.confidence == pytest.approx(-(1 - probability.item()), abs=0.015)
    assert estimate.spatial_grid_shape == (
        (1, 1) if spatial_weight > 0 else None
    )


def test_spatial_copula_repeat_is_exactly_reproducible():
    probability = np.linspace(0.05, 0.95, 42).reshape(6, 7)
    action = probability >= 0.5
    arguments = {
        "posterior_draws": 32,
        "repeat_index": 3,
        "global_variance_weight": 0.25,
        "spatial_variance_weight": 0.5,
        "spatial_knot_spacing_diagonal": 0.2,
        "posterior_batch_size": 4,
        "master_seed": 123,
        "sample_id": "image-9",
        "device": "cpu",
    }
    left = spatial_copula_dice_confidence(probability, action, **arguments)
    right = spatial_copula_dice_confidence(probability, action, **arguments)
    assert left == right
    assert left.spatial_grid_shape == (4, 5)


def test_materialized_spatial_copula_masks_are_reproducible_and_marginal_preserving():
    probability = np.array([[0.2, 0.4], [0.6, 0.8]])
    arguments = {
        "posterior_draws": 20_000,
        "repeat_index": 1,
        "global_variance_weight": 0.25,
        "spatial_variance_weight": 0.5,
        "spatial_knot_spacing_diagonal": 0.2,
        "posterior_batch_size": 1_000,
        "master_seed": 7,
        "sample_id": "tiny",
        "device": "cpu",
    }
    left, grid = sample_spatial_copula_masks(probability, **arguments)
    right, right_grid = sample_spatial_copula_masks(probability, **arguments)
    assert np.array_equal(left, right)
    assert grid == right_grid == (2, 2)
    assert np.max(np.abs(left.mean(axis=0) - probability)) < 0.015


@pytest.mark.parametrize(
    "overrides",
    [
        {"posterior_draws": 3},
        {"repeat_index": -1},
        {"global_variance_weight": -0.1},
        {"global_variance_weight": 0.8, "spatial_variance_weight": 0.3},
        {"spatial_knot_spacing_diagonal": 0.0},
        {"posterior_batch_size": 0},
    ],
)
def test_spatial_copula_rejects_invalid_parameters(overrides):
    probability = np.array([[0.2, 0.8]])
    action = probability >= 0.5
    arguments = {
        "posterior_draws": 4,
        "repeat_index": 0,
        "global_variance_weight": 0.25,
        "spatial_variance_weight": 0.5,
        "spatial_knot_spacing_diagonal": 0.1,
        "posterior_batch_size": 2,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        spatial_copula_dice_confidence(probability, action, **arguments)


def _copula_analysis_fixture():
    methods = {
        field: {"aurc": 0.1 + 0.001 * index}
        for index, (field, _) in enumerate(COPULA_METHODS)
    }
    return {
        "analysis_id": "dice-spatial-copula-analysis-v1",
        "summary": {"primary_variant": PRIMARY_COPULA_VARIANT},
        "conditions": [
            {
                "dataset": dataset,
                "condition": model,
                "methods": methods,
                "primary_comparison": {
                    "variant_id": PRIMARY_COPULA_VARIANT,
                    "reference": "dice_m32",
                },
            }
            for dataset, _ in DATASETS
            for model, _ in MODELS
        ],
    }


def test_spatial_copula_renderer_is_symmetric_and_explicit(tmp_path):
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(_copula_analysis_fixture()), encoding="utf-8")
    analysis = _load_copula(source)
    table = _copula_table(analysis, "a" * 64)

    assert table.count(r"\begin{tabular*}{\textwidth}") == len(MODELS)
    assert table.count(r"\textbf{Spatial copula") == len(MODELS)
    assert table.count(r"\bestresult{") == len(DATASETS) * len(MODELS)
    assert "four-repeat mean Dice-risk AURC $\\times100$" in table
    assert "prespecified primary variant" in table


def test_spatial_copula_renderer_rejects_missing_method(tmp_path):
    analysis = _copula_analysis_fixture()
    analysis["conditions"][0]["methods"].pop("sdc")
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected method set"):
        _load_copula(source)


def test_sdc_bound_audit_uses_exact_risk_and_separates_empty_actions():
    rows = [
        {
            "action_pixels": 4,
            "confidence_dice_exact": -0.25,
            "confidence_dice_sdc_recomputed": 0.80,
            "shared_variance_overlap": 0.16,
            "shared_variance_outside": 0.64,
        },
        {
            "action_pixels": 0,
            "confidence_dice_exact": -0.3,
            "confidence_dice_sdc_recomputed": 0.0,
            "shared_variance_overlap": 0.0,
            "shared_variance_outside": 1.0,
        },
    ]
    audit = sdc_bound_audit(rows)
    assert audit["num_nonempty_actions"] == 1
    assert audit["num_empty_actions"] == 1
    assert audit["all_nonempty_bounds_hold"]
    overall = audit["overall"]
    assert overall["absolute_sdc_risk_error"]["mean"] == pytest.approx(0.05)
    assert overall["theoretical_bound"]["mean"] == pytest.approx(0.3)
    assert overall["mean_overlap_term"] == pytest.approx(0.2)
    assert overall["mean_outside_term"] == pytest.approx(0.1)
