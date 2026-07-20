"""Tests for analysis-only working-risk and ranking diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_binary import METHODS, RISKS
from scripts.analyze_working_risk_diagnostics import (
    ARTIFACT_TYPE,
    _correlation,
    analyze,
    equal_count_reliability,
    fractional_acceptance_weights,
    load_inputs,
    write_report,
)


SCORE_FIELDS = [field for field, _ in METHODS]
RISK_FIELDS = [field for field, _ in RISKS]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict]:
    specifications = (
        # predicted Dice risk, observed Dice/nHD/nHD95, SDC, prediction/truth size
        (0.20, 0.10, 0.20, 0.15, 0.00, 0.00, 0.00),
        (0.10, 0.20, 0.10, 0.25, 0.85, 0.005, 0.01),
        (0.40, 0.60, 0.50, 0.55, 0.50, 0.05, 0.08),
        (0.80, 0.90, 0.85, 0.95, 0.15, 0.60, 0.70),
    )
    rows = []
    for index, (
        predicted_dice_risk,
        dice_loss,
        nhd_loss,
        nhd95_loss,
        sdc,
        prediction_fraction,
        truth_fraction,
    ) in enumerate(specifications):
        scores = {field: 0.25 + index / 100 for field in SCORE_FIELDS}
        scores.update(
            {
                "confidence_sdc": sdc,
                "confidence_dice_exact": -predicted_dice_risk,
                "confidence_nhd_m32": -(0.15 + index * 0.20),
                "confidence_nhd95_m32": -(0.12 + index * 0.23),
            }
        )
        rows.append(
            {
                "schema_version": 2,
                "run_id": "synthetic-run",
                "sample_id": f"sample-{index}",
                "image_id": f"image-{index}",
                "prediction_foreground_fraction": prediction_fraction,
                "truth_foreground_fraction": truth_fraction,
                "risk_dice": dice_loss,
                "risk_nhd": nhd_loss,
                "risk_nhd95": nhd95_loss,
                **scores,
            }
        )
    return rows


def _write_condition(
    root: Path,
    dataset: str,
    condition: str,
    *,
    rows: list[dict] | None = None,
) -> Path:
    rows = _rows() if rows is None else rows
    directory = root / dataset / condition / "synthetic-run"
    directory.mkdir(parents=True)
    records = directory / "records.jsonl"
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    sample_hash = hashlib.sha256(
        "\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")
    ).hexdigest()
    model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
    manifest = {
        "schema_version": 2,
        "artifact_type": "synthetic-incomplete-assembly",
        "run_id": "synthetic-run",
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "split": "test",
        "num_images": len(rows),
        "num_rows": len(rows),
        "jsonl_sha256": _sha256(records),
        "sample_id_sha256": sample_hash,
        "risk_fields": RISK_FIELDS,
        "auxiliary_fields": [],
        "score_fields": SCORE_FIELDS,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return records


def test_incomplete_analysis_is_order_invariant_and_marks_target_conditions(tmp_path):
    general = _write_condition(tmp_path, "pet", "clipseg-general")
    target = _write_condition(tmp_path, "pet", "clipseg-target")

    first = analyze(
        [target, general], campaign_lock=None, group_bins=2, allow_incomplete=True
    )
    second = analyze(
        [general, target], campaign_lock=None, group_bins=2, allow_incomplete=True
    )

    assert first == second
    assert first["artifact_type"] == ARTIFACT_TYPE
    assert first["schema_version"] == 2
    assert first["condition_sets"]["num_analyzed_conditions"] == 2
    assert first["condition_sets"]["num_target_conditions"] == 1
    assert first["condition_sets"]["target_conditions"] == ["pet/clipseg-target"]
    assert [row["is_target_condition"] for row in first["conditions"]] == [
        False,
        True,
    ]
    assert "does not identify" in first["scope"]["posterior_limitation"]
    assert "do not fit" in first["scope"]["label_use"]

    quality = first["conditions"][0]["model_action_quality"]
    assert quality["mean_dice_coefficient"] == pytest.approx(0.55)
    assert quality["deployed_prediction_empty_rate"] == pytest.approx(0.25)
    assert quality["reference_truth_empty_rate"] == pytest.approx(0.25)
    assert quality["aurc_references"]["risk_dice"][
        "random_ordering_aurc"
    ] == pytest.approx(0.45)
    assert quality["aurc_references"]["risk_dice"]["oracle_aurc"] < 0.45

    discrepancy = first["conditions"][0]["dice_exact_vs_sdc"]
    assert discrepancy["overall"]["n"] == 4
    empty = discrepancy["by_deployed_prediction_mask_size"][0]
    assert empty["stratum"] == "empty"
    assert empty["n"] == 1
    assert empty["mean_signed_difference"] == pytest.approx(0.8)

    reliability = first["conditions"][0]["working_risk_grouped_reliability"]
    assert [item["score_field"] for item in reliability] == [
        "confidence_dice_exact",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    ]
    assert all(item["effective_equal_count_groups"] == 2 for item in reliability)
    agreement = first["conditions"][0]["indexed_score_agreement"]
    assert agreement[0]["left_score"] == "confidence_dice_m32"
    assert agreement[0]["right_score"] == "confidence_nhd_m32"
    assert all(
        "confidence_dice_exact" not in {item["left_score"], item["right_score"]}
        for item in agreement
    )
    assert [
        item["score_field"] for item in first["specification"]["agreement_scores"]
    ] == [
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    ]
    assert [
        item["score_field"]
        for item in first["specification"]["matched_reliability_scores"]
    ][0] == "confidence_dice_exact"

    first_path = write_report(first, tmp_path / "first.json")
    second_path = write_report(second, tmp_path / "second.json")
    assert first_path.read_bytes() == second_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(first, first_path)


def test_fractional_selector_is_tie_aware_and_has_exact_acceptance_mass():
    tied = fractional_acceptance_weights([1.0, 1.0, 0.0, 0.0], 0.25)
    ranked = fractional_acceptance_weights([1.0, 0.0, 0.0, 0.0], 0.25)
    assert tied.tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])
    assert ranked.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert tied.sum() == pytest.approx(1.0)
    overlap = np.minimum(tied, ranked).sum() / np.maximum(tied, ranked).sum()
    assert overlap == pytest.approx(1 / 3)

    all_tied = fractional_acceptance_weights(np.ones(7), 0.5)
    assert all_tied.tolist() == pytest.approx([0.5] * 7)
    assert all_tied.sum() == pytest.approx(3.5)


def test_equal_count_reliability_has_explicit_per_image_and_grouped_metrics():
    result = equal_count_reliability(
        [0.1, 0.2, 0.8, 0.9],
        [0.0, 0.4, 0.6, 1.0],
        ["a", "b", "c", "d"],
        bins=2,
    )
    assert result["bias_predicted_minus_observed"] == pytest.approx(0.0)
    assert result["per_image_mae"] == pytest.approx(0.15)
    assert result["per_image_rmse"] == pytest.approx(np.sqrt(0.025))
    assert result["grouped_ece"] == pytest.approx(0.05)
    assert [group["n"] for group in result["groups"]] == [2, 2]
    assert result["groups"][0][
        "signed_group_gap_predicted_minus_observed"
    ] == pytest.approx(-0.05)

    reversed_result = equal_count_reliability(
        [0.9, 0.8, 0.2, 0.1],
        [1.0, 0.6, 0.4, 0.0],
        ["d", "c", "b", "a"],
        bins=2,
    )
    assert result == reversed_result


def test_correlations_encode_constant_score_as_json_null_not_nan():
    constant = _correlation(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([1.0, 2.0, 3.0]),
        method="spearman",
    )
    assert constant == {
        "defined": False,
        "value": None,
        "undefined_reason": "constant_left_score",
    }
    defined = _correlation(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([3.0, 2.0, 1.0]),
        method="kendall_tau_b",
    )
    assert defined["defined"] is True
    assert defined["value"] == pytest.approx(-1.0)


def test_strict_input_scope_rejects_duplicates_missing_fields_and_incomplete_canonical(
    tmp_path,
):
    records = _write_condition(tmp_path / "valid", "fives", "clipseg-general")
    with pytest.raises(ValueError, match="must be distinct"):
        load_inputs([records, records], allow_incomplete=True)
    with pytest.raises(ValueError, match="exactly the 16"):
        load_inputs([records])

    rows = _rows()
    for row in rows:
        del row["prediction_foreground_fraction"]
    missing = _write_condition(
        tmp_path / "missing", "fives", "clipseg-target", rows=rows
    )
    with pytest.raises(ValueError, match="lacks diagnostic fields"):
        load_inputs([missing], allow_incomplete=True)

    with pytest.raises(ValueError, match="do not supply --campaign-lock"):
        analyze(
            [records],
            campaign_lock=tmp_path / "not-used.lock.json",
            allow_incomplete=True,
        )
