"""Tests for predeclared qualitative selection and artifact-only rendering."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.render_binary_qualitative_cases import (
    CELL_SIZE,
    HEADER_HEIGHT,
    LEFT_MARGIN,
    ROW_HEIGHT,
    compose_dataset_panel,
    load_selected_arrays,
)
from scripts.select_binary_qualitative_cases import (
    CASE_ORDER,
    _canonical_json,
    choose_dataset_cases,
    normalized_safety_ranks,
    validate_selection_id,
    write_selection,
)
from selectseg.binary_artifacts import load_binary_artifact, write_binary_artifact


def _row(
    sample_id,
    *,
    ranks,
    prediction_fraction=0.2,
    risks=(0.2, 0.2, 0.2),
    predicted_risks=(0.2, 0.2, 0.2),
    condition="clipseg-target",
):
    return {
        "dataset": "pet",
        "condition": condition,
        "model": "clipseg" if condition.startswith("clipseg") else "deeplabv3",
        "sample_id": sample_id,
        "image_id": sample_id,
        "image_index": int(sample_id[-1]) if sample_id[-1].isdigit() else 0,
        "height": 2,
        "width": 2,
        "prediction_foreground_fraction": prediction_fraction,
        "truth_foreground_fraction": 0.25,
        "scores": {
            "confidence_dice_m32": -predicted_risks[0],
            "confidence_nhd_m32": -predicted_risks[1],
            "confidence_nhd95_m32": -predicted_risks[2],
        },
        "risks": {
            "risk_dice": risks[0],
            "risk_nhd": risks[1],
            "risk_nhd95": risks[2],
        },
        "normalized_safety_ranks": {
            "confidence_dice_m32": ranks[0],
            "confidence_nhd_m32": ranks[1],
            "confidence_nhd95_m32": ranks[2],
        },
    }


def test_normalized_safety_ranks_are_tie_aware_and_oriented_safest_first():
    ranks = normalized_safety_ranks([0.9, 0.9, 0.2])
    assert ranks.tolist() == pytest.approx([0.25, 0.25, 1.0])
    assert normalized_safety_ranks([0.5]).tolist() == [0.0]


def test_case_rules_avoid_duplicate_underlying_samples_in_declared_order():
    rows = [
        _row("dup0", ranks=(0.0, 1.0, 0.0)),
        _row("rank1", ranks=(0.2, 0.1, 0.9)),
        _row(
            "empty2",
            ranks=(0.4, 0.4, 0.4),
            prediction_fraction=0.0,
            risks=(0.8, 0.7, 0.6),
            predicted_risks=(0.5, 0.5, 0.5),
        ),
        _row(
            "fail3",
            ranks=(0.5, 0.5, 0.5),
            risks=(0.95, 0.4, 0.3),
            predicted_risks=(0.0, 0.4, 0.3),
        ),
    ]

    selected = choose_dataset_cases(rows)
    assert [case["case_type"] for case in selected] == list(CASE_ORDER)
    assert [case["sample_id"] for case in selected] == [
        "dup0",
        "rank1",
        "empty2",
        "fail3",
    ]
    assert all(not case["duplicate_fallback_used"] for case in selected)
    assert selected[0]["selection_objective"] == pytest.approx(1.0)
    assert selected[1]["selection_objective"] == pytest.approx(0.8)
    assert selected[2]["selection_objective_details"]["matched_loss"] == "dice"
    assert selected[3]["selection_objective_details"]["matched_loss"] == "dice"
    assert selected[3]["selection_objective"] == pytest.approx(0.95)


def test_missing_empty_action_is_explicit_and_never_substituted():
    cases = choose_dataset_cases(
        [
            _row("case0", ranks=(0.0, 1.0, 0.2)),
            _row("case1", ranks=(0.4, 0.3, 1.0)),
        ]
    )
    empty = next(case for case in cases if case["case_type"] == "empty_action")
    assert empty["status"] == "unavailable"
    assert "no deployed empty action" in empty["reason"]


def test_selection_content_id_and_publication_are_no_overwrite(tmp_path):
    base = {"schema_version": 1, "value": [1, 2, 3]}
    report = {
        **base,
        "selection_id": hashlib.sha256(_canonical_json(base)).hexdigest()[:16],
    }
    selection_id = validate_selection_id(report)
    output = write_selection(report, tmp_path)
    assert output == tmp_path / selection_id / "selection.json"
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_selection(report, tmp_path)
    changed = dict(report)
    changed["value"] = [9]
    with pytest.raises(ValueError, match="does not match"):
        validate_selection_id(changed)


def _write_artifact(root):
    probability = np.array([[0.1, 0.9], [0.7, 0.2]], dtype=np.float32)
    truth = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    return write_binary_artifact(
        root,
        dataset="toy",
        condition="clipseg-target",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="lesion",
        checkpoint={"path": "outputs/checkpoint.pt", "sha256": "a" * 64, "size_bytes": 1},
        base_model={"name": "clipseg", "source": "vendor/model"},
        source_sha256="b" * 64,
        environment={
            "packages": {
                "python": "test",
                "numpy": "test",
                "torch": "test",
                "torchvision": "test",
                "transformers": "test",
            },
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        preprocessing={
            "model_input": "square resize with antialiasing",
            "probability_to_native_mask": "bilinear interpolation with align_corners=False",
        },
        cohort="held-out toy image",
        sample_ids=["sample-0"],
        samples=[("sample-0", probability, truth)],
        command=["python", "freeze"],
        created_utc="2026-07-19T12:00:00+00:00",
    )


def test_renderer_loads_only_manifest_bound_payload_and_checks_hash(tmp_path):
    manifest_path = _write_artifact(tmp_path / "artifacts")
    artifact = load_binary_artifact(manifest_path, validate_payloads=False)
    entry = artifact.manifest["samples"][0]
    case = {
        "dataset": "toy",
        "condition": "clipseg-target",
        "sample_id": "sample-0",
        "image_index": 0,
        "height": 2,
        "width": 2,
        "provenance": {
            "artifact_id": artifact.manifest["artifact_id"],
            "artifact_manifest_path": str(manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "sample_payload_path": entry["path"],
            "sample_payload_sha256": entry["sha256"],
        },
    }
    probability, truth, provenance = load_selected_arrays(case)
    assert probability.dtype == np.float32
    assert truth.dtype == bool
    assert provenance["payload_sha256"] == entry["sha256"]

    payload = manifest_path.parent / entry["path"]
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selected_arrays(case)


def test_panel_render_is_deterministic_and_contains_boundary_annotations():
    case = {
        "case_type": "confident_failure",
        "status": "selected",
        "condition": "clipseg-target",
        "sample_id": "case-0",
        "selection_objective": 0.7,
        "selection_objective_details": {"matched_loss": "dice"},
        "risks": {"risk_dice": 0.8, "risk_nhd": 0.5, "risk_nhd95": 0.4},
    }
    probability = np.array(
        [[0.1, 0.9, 0.9], [0.1, 0.9, 0.1], [0.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    truth = np.array(
        [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
        dtype=bool,
    )
    arrays = {("clipseg-target", "case-0"): (probability, truth)}
    first = compose_dataset_panel("pet", [case], arrays)
    second = compose_dataset_panel("pet", [case], arrays)
    assert first.size == (
        LEFT_MARGIN + 4 * CELL_SIZE + 5 * 14,
        HEADER_HEIGHT + ROW_HEIGHT + 14,
    )
    assert first.tobytes() == second.tobytes()
    pixels = np.asarray(first)
    assert np.any(np.all(pixels == (0, 220, 80), axis=-1))
    assert np.any(np.all(pixels == (235, 30, 190), axis=-1))
