"""Tests for shared normalized full-HD and pooled-HD95 surface geometry."""

import math

import numpy as np
import pytest
from scipy import ndimage

import selectseg.binary_boundary as binary_boundary
from selectseg.binary_boundary import (
    BoundaryLosses,
    PreparedBoundaryReference,
    normalized_penalized_boundary_losses,
    prepare_boundary_reference,
)
from selectseg.binary_framework import normalized_penalized_hd95


def _naive_dual_edt_oracle(reference, candidate) -> BoundaryLosses:
    """Independent two-EDT oracle with the established surface convention."""

    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    reference_present = bool(reference.any())
    candidate_present = bool(candidate.any())
    if not reference_present and not candidate_present:
        return BoundaryLosses(0.0, 0.0)
    if reference_present != candidate_present:
        return BoundaryLosses(1.0, 1.0)

    reference_surface = reference & ~ndimage.binary_erosion(reference)
    candidate_surface = candidate & ~ndimage.binary_erosion(candidate)
    to_candidate = ndimage.distance_transform_edt(~candidate_surface)
    to_reference = ndimage.distance_transform_edt(~reference_surface)
    distances = np.concatenate(
        [to_candidate[reference_surface], to_reference[candidate_surface]]
    )
    diagonal = math.hypot(*reference.shape)
    return BoundaryLosses(
        min(1.0, float(np.max(distances)) / diagonal),
        min(1.0, float(np.percentile(distances, 95)) / diagonal),
    )


def _boundary_cases():
    empty = np.zeros((11, 13), dtype=bool)
    singleton_corner = empty.copy()
    singleton_corner[0, 0] = True
    singleton_far = empty.copy()
    singleton_far[-1, -1] = True
    full = np.ones_like(empty)
    frame = empty.copy()
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True
    block = empty.copy()
    block[2:8, 3:10] = True
    block_with_hole = block.copy()
    block_with_hole[4:6, 5:8] = False
    return (
        (empty, empty),
        (empty, singleton_corner),
        (singleton_corner, empty),
        (singleton_corner, singleton_corner),
        (singleton_corner, singleton_far),
        (full, frame),
        (block, block_with_hole),
        (frame, block),
    )


@pytest.mark.parametrize(("reference", "candidate"), _boundary_cases())
def test_shared_losses_match_naive_dual_edt_oracle(reference, candidate):
    observed = normalized_penalized_boundary_losses(reference, candidate)
    expected = _naive_dual_edt_oracle(reference, candidate)
    assert observed.nhd == pytest.approx(expected.nhd, abs=0.0)
    assert observed.nhd95 == pytest.approx(expected.nhd95, abs=0.0)
    assert observed.nhd95 == normalized_penalized_hd95(reference, candidate)
    assert 0.0 <= observed.nhd <= 1.0
    assert 0.0 <= observed.nhd95 <= 1.0
    assert observed.nhd95 <= observed.nhd


@pytest.mark.parametrize("seed", range(8))
def test_pooled_nhd95_is_value_identical_to_existing_implementation(seed):
    rng = np.random.default_rng(seed)
    reference = rng.random((17, 19)) < (0.15 + 0.05 * (seed % 3))
    prepared = prepare_boundary_reference(reference)
    for probability in (0.05, 0.2, 0.5, 0.85):
        candidate = rng.random(reference.shape) < probability
        observed = prepared.compare(candidate).nhd95
        expected = normalized_penalized_hd95(reference, candidate)
        assert observed == expected


def test_empty_conventions_and_full_hd_hand_case():
    empty = np.zeros((6, 8), dtype=bool)
    first = empty.copy()
    second = empty.copy()
    first[2, 2] = True
    second[2, 4] = True

    assert prepare_boundary_reference(empty).compare(empty) == (0.0, 0.0)
    assert prepare_boundary_reference(empty).compare(first) == (1.0, 1.0)
    assert prepare_boundary_reference(first).compare(empty) == (1.0, 1.0)
    expected = 2 / math.hypot(6, 8)
    assert prepare_boundary_reference(first).compare(second) == pytest.approx(
        (expected, expected)
    )


def test_prepared_reference_uses_one_new_edt_per_nonempty_candidate(monkeypatch):
    rng = np.random.default_rng(91)
    reference = rng.random((31, 37)) < 0.35
    candidates = [
        rng.random(reference.shape) < probability for probability in (0.2, 0.5, 0.8)
    ]

    real_edt = ndimage.distance_transform_edt
    calls = []

    def counting_edt(*args, **kwargs):
        calls.append(None)
        return real_edt(*args, **kwargs)

    monkeypatch.setattr(binary_boundary.ndimage, "distance_transform_edt", counting_edt)
    prepared = PreparedBoundaryReference.from_mask(reference)
    assert len(calls) == 1
    for index, candidate in enumerate(candidates, start=1):
        prepared.compare(candidate)
        assert len(calls) == 1 + index


def test_empty_pairs_skip_unnecessary_distance_transforms(monkeypatch):
    calls = []

    def unexpected_edt(*_args, **_kwargs):
        calls.append(None)
        raise AssertionError("empty conventions must not run an EDT")

    monkeypatch.setattr(
        binary_boundary.ndimage, "distance_transform_edt", unexpected_edt
    )
    empty = np.zeros((5, 7), dtype=bool)
    nonempty = empty.copy()
    nonempty[2, 3] = True
    prepared = prepare_boundary_reference(empty)
    assert prepared.compare(empty) == (0.0, 0.0)
    assert prepared.compare(nonempty) == (1.0, 1.0)
    assert calls == []


def test_prepared_reference_validates_masks_and_shapes():
    with pytest.raises(ValueError, match="only binary"):
        prepare_boundary_reference(np.array([[0, 2]]))
    with pytest.raises(ValueError, match="2D"):
        prepare_boundary_reference(np.zeros(3, dtype=bool))
    prepared = prepare_boundary_reference(np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="shapes differ"):
        prepared.compare(np.zeros((3, 2), dtype=bool))
