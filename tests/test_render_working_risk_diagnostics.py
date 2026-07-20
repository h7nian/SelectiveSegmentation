"""Contract and deterministic-TeX tests for grouped diagnostic rendering."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.analyze_binary import EXPECTED_CONDITIONS
from scripts.analyze_working_risk_diagnostics import (
    AGREEMENT_SCORES,
    ARTIFACT_TYPE,
    COVERAGES,
    RELIABILITY_SCORES,
    SCHEMA_VERSION,
    TARGET_CONDITIONS,
)
from scripts.render_working_risk_diagnostics import (
    ORDERED_TARGET_CONDITIONS,
    OUTPUT_NAME,
    load_diagnostics,
    main,
    render_diagnostics,
    validate_diagnostics,
    write_output,
)


def _correlation(value):
    return {"defined": True, "value": value, "undefined_reason": None}


def _diagnostics():
    target_index = {key: index for index, key in enumerate(ORDERED_TARGET_CONDITIONS)}
    conditions = []
    for key in reversed(EXPECTED_CONDITIONS):
        dataset, condition = key
        is_target = key in TARGET_CONDITIONS
        index = target_index.get(key, 0)
        dice = 0.70 + 0.01 * index if is_target else 0.50
        quality = {
            "mean_dice_coefficient": dice,
            "mean_dice_loss": 1.0 - dice,
            "mean_normalized_penalized_hd_loss": 0.10 + 0.01 * index,
            "mean_normalized_penalized_hd95_loss": 0.05 + 0.01 * index,
            "deployed_prediction_empty_rate": 0.001 * index,
            "reference_truth_empty_rate": 0.0,
            "aurc_references": {},
        }

        agreements = []
        for left_index, (left, left_label) in enumerate(AGREEMENT_SCORES):
            for right, right_label in AGREEMENT_SCORES[left_index + 1 :]:
                if (left, right) == (
                    "confidence_dice_m32",
                    "confidence_nhd_m32",
                ):
                    spearman = 0.40 + 0.01 * index
                    kendall = 0.30 + 0.01 * index
                    base_jaccard = 0.20
                elif (left, right) == (
                    "confidence_nhd_m32",
                    "confidence_nhd95_m32",
                ):
                    spearman = 0.80 + 0.01 * index
                    kendall = 0.70 + 0.01 * index
                    base_jaccard = 0.60
                else:
                    spearman = 0.60 + 0.01 * index
                    kendall = 0.50 + 0.01 * index
                    base_jaccard = 0.40
                agreements.append(
                    {
                        "left_score": left,
                        "left_label": left_label,
                        "right_score": right,
                        "right_label": right_label,
                        "spearman_rho": _correlation(spearman),
                        "kendall_tau_b": _correlation(kendall),
                        "accepted_set_agreement": [
                            {
                                "coverage": coverage,
                                "tie_aware_fractional_jaccard": (
                                    base_jaccard + coverage / 10 + 0.01 * index
                                ),
                            }
                            for coverage in COVERAGES
                        ],
                    }
                )

        reliability = []
        for score_index, (score, risk, label) in enumerate(RELIABILITY_SCORES):
            reliability.append(
                {
                    "n": 20,
                    "requested_equal_count_groups": 10,
                    "effective_equal_count_groups": 10,
                    "bias_predicted_minus_observed": -0.10 + 0.01 * index,
                    "per_image_mae": 0.10 + 0.01 * index + 0.01 * score_index,
                    "per_image_rmse": 0.15 + 0.01 * index + 0.01 * score_index,
                    "grouped_ece": 0.05 + 0.01 * index + 0.01 * score_index,
                    "groups": [],
                    "score_field": score,
                    "score_label": label,
                    "observed_loss_field": risk,
                    "predicted_working_risk_definition": f"-{score}",
                }
            )
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "model": "clipseg" if condition.startswith("clipseg") else "deeplabv3",
                "is_target_condition": is_target,
                "num_images": 20,
                "model_action_quality": quality,
                "indexed_score_agreement": agreements,
                "dice_exact_vs_sdc": {},
                "working_risk_grouped_reliability": reliability,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scope": {
            "posterior_limitation": (
                "one reference mask per image does not identify the true posterior"
            )
        },
        "specification": {
            "grouping": "synthetic deterministic equal-count grouping",
            "requested_group_bins": 10,
            "accepted_set_coverages": list(COVERAGES),
            "agreement_scores": [
                {"score_field": score, "label": label}
                for score, label in AGREEMENT_SCORES
            ],
            "matched_reliability_scores": [
                {
                    "score_field": score,
                    "matched_observed_loss_field": risk,
                    "label": label,
                }
                for score, risk, label in RELIABILITY_SCORES
            ],
            "mask_size_strata": [],
        },
        "condition_sets": {
            "all_analyzed_conditions": [
                f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
            ],
            "target_condition_definition": "synthetic target definition",
            "target_conditions": [
                f"{dataset}/{condition}"
                for dataset, condition in ORDERED_TARGET_CONDITIONS
            ],
            "num_analyzed_conditions": 16,
            "num_target_conditions": 10,
        },
        "provenance": {"binding": "synthetic-test"},
        "conditions": conditions,
    }


def test_render_has_two_manuscript_ready_tables_fixed_order_and_ranges():
    diagnostics = _diagnostics()
    tex = render_diagnostics(diagnostics, source_hash="a" * 64)

    assert tex.count(r"\begin{table*}[t]") == 2
    assert "% Source diagnostics JSON SHA-256: " + "a" * 64 in tex
    assert r"\label{tab:target-action-quality-diagnostics}" in tex
    assert r"\label{tab:target-working-risk-diagnostic-ranges}" in tex
    assert "descriptive" in tex.lower()
    assert tex.count("ten target conditions") == 2
    assert tex.count("posterior validation") == 2
    assert "not conditional calibration estimates or posterior validation" in tex
    assert tex.count("not independent replicates") == 2
    assert tex.index("Oxford Pet") < tex.index("Kvasir-SEG") < tex.index("FIVES")
    assert "Mean Dice $\\uparrow$ & 0.700 & 0.710" in tex
    assert "Predicted empty & 0.0\\% & 0.1\\%" in tex
    assert tex.count(r"\resizebox{\linewidth}{!}{%") == 2

    assert r"Dice-M32--nHD-M32 & $[0.400,\,0.490]$ & $[0.300,\,0.390]$" in tex
    assert r"nHD-M32--nHD95-M32 & $[0.800,\,0.890]$ & $[0.700,\,0.790]$" in tex
    assert r"Dice-Exact $\to$ Dice & $[-0.100,\,-0.010]$ & $[0.100,\,0.190]$" in tex

    # Rendering is independent of condition-list input order for a fixed source hash.
    ordered = copy.deepcopy(diagnostics)
    ordered["conditions"].reverse()
    assert render_diagnostics(ordered, source_hash="a" * 64) == tex


def test_strict_validation_rejects_scope_join_and_statistic_inconsistencies():
    diagnostics = _diagnostics()
    validate_diagnostics(diagnostics)

    wrong_dice = copy.deepcopy(diagnostics)
    wrong_dice["conditions"][0]["model_action_quality"]["mean_dice_coefficient"] = 0.123
    with pytest.raises(ValueError, match="inconsistent mean Dice"):
        validate_diagnostics(wrong_dice)

    wrong_target = copy.deepcopy(diagnostics)
    wrong_target["condition_sets"]["target_conditions"].pop()
    with pytest.raises(ValueError, match="exact ten target"):
        validate_diagnostics(wrong_target)

    wrong_score_schema = copy.deepcopy(diagnostics)
    wrong_score_schema["specification"]["agreement_scores"][0] = {
        "score_field": "confidence_dice_exact",
        "label": "Dice-Exact",
    }
    with pytest.raises(ValueError, match="three primary M32"):
        validate_diagnostics(wrong_score_schema)

    duplicate = copy.deepcopy(diagnostics)
    duplicate["conditions"][1] = copy.deepcopy(duplicate["conditions"][0])
    with pytest.raises(ValueError, match="duplicate condition"):
        validate_diagnostics(duplicate)

    undefined = copy.deepcopy(diagnostics)
    target_row = next(
        row for row in undefined["conditions"] if row["is_target_condition"]
    )
    target_row["indexed_score_agreement"][0]["spearman_rho"] = {
        "defined": False,
        "value": None,
        "undefined_reason": "constant_left_score",
    }
    with pytest.raises(ValueError, match="must be defined"):
        validate_diagnostics(undefined)


def test_loader_rejects_duplicate_keys_and_nonstandard_constants(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_diagnostics(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_diagnostics(nonfinite)


def test_main_writes_exact_single_name_with_source_hash_and_no_overwrite(
    tmp_path, capsys
):
    source = tmp_path / "diagnostics.json"
    source.write_text(json.dumps(_diagnostics(), indent=2) + "\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "isolated"

    main(["--diagnostics", str(source), "--output-dir", str(output)])
    destination = output / OUTPUT_NAME
    assert destination.is_file()
    assert capsys.readouterr().out.strip() == destination.as_posix()
    assert f"Source diagnostics JSON SHA-256: {source_hash}" in destination.read_text()
    assert [path.name for path in output.iterdir()] == [OUTPUT_NAME]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--diagnostics", str(source), "--output-dir", str(output)])

    second = write_output(
        render_diagnostics(_diagnostics(), source_hash=source_hash),
        tmp_path / "second",
    )
    assert second.read_bytes() == destination.read_bytes()
