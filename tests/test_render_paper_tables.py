"""Contract and rendering tests for scripts/render_paper_tables.py."""

import copy
import json

import pytest

from scripts.render_paper_tables import (
    EXPECTED_CONDITIONS,
    METHODS,
    OUTPUT_NAMES,
    load_analysis,
    main,
    render_tables,
    validate_analysis,
)


def _analysis(*, condition_count=10, samples=10_000):
    conditions = []
    hypotheses = []
    raw_p_values = []
    adjusted_p_values = []
    method_items = list(METHODS.items())
    for condition_index, (dataset, condition_name) in enumerate(
        reversed(EXPECTED_CONDITIONS[:condition_count])
    ):
        risks = {}
        comparisons = {}
        for risk_index, risk_field in enumerate(("risk_dice", "risk_nhd95")):
            oracle = 0.01 + 0.001 * risk_index
            random = 0.81 + 0.001 * risk_index
            methods = {}
            for method_index, (method_field, label) in enumerate(method_items):
                aurc = 0.2 + 0.005 * condition_index + 0.01 * method_index
                aurc += 0.1 * risk_index
                excess = aurc - oracle
                methods[method_field] = {
                    "label": label,
                    "aurc": aurc,
                    "oracle_aurc": oracle,
                    "random_aurc": random,
                    "excess_aurc": excess,
                    "normalized_aurc": excess / (random - oracle),
                }
            risks[risk_field] = {
                "label": (
                    "Dice risk"
                    if risk_field == "risk_dice"
                    else "Normalized penalized HD95 risk"
                ),
                "methods": methods,
                "oracle_aurc": oracle,
                "random_aurc": random,
            }
            difference = (
                methods["confidence_dice_m32"]["aurc"]
                - methods["confidence_nhd95_m32"]["aurc"]
            )
            p_value = min(1.0, 2 / (samples + 1))
            adjusted = min(1.0, 2 * condition_count * p_value)
            comparisons[risk_field] = {
                "left": "confidence_dice_m32",
                "right": "confidence_nhd95_m32",
                "difference_left_minus_right": difference,
                "bootstrap": {
                    "difference": difference,
                    "ci_low": difference - 0.01,
                    "ci_high": difference + 0.01,
                    "confidence_level": 0.95,
                    "p_value": p_value,
                    "n_resamples": samples,
                    "n_observations": 20,
                    "n_clusters": 20,
                    "seed": condition_index * 2 + risk_index,
                },
                "holm_adjusted_p_value": adjusted,
            }
            hypotheses.append(
                {
                    "dataset": dataset,
                    "condition": condition_name,
                    "risk": risk_field,
                    "raw_bootstrap_p_value": p_value,
                    "holm_adjusted_p_value": adjusted,
                }
            )
            raw_p_values.append(p_value)
            adjusted_p_values.append(adjusted)
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition_name,
                "split": "test",
                "num_rows": 20,
                "num_image_clusters": 20,
                "risks": risks,
                "comparisons": comparisons,
            }
        )
    return {
        "schema_version": 1,
        "analysis": {
            "tie_policy": "analytic expectation over random within-tie order",
            "normalized_aurc": "normalized excess AURC",
            "comparison": "Dice-M32 minus nHD95-M32 AURC",
            "bootstrap_samples": samples,
            "confidence_level": 0.95,
            "bootstrap_p_value": "same-resample equal-tail bootstrap p-value",
            "seed": 0,
        },
        "conditions": conditions,
        "multiple_testing": {
            "procedure": "Holm step-down family-wise-error adjustment",
            "family": "all paired cluster bootstrap comparisons",
            "num_hypotheses": 2 * condition_count,
            "hypotheses": hypotheses,
            "raw_bootstrap_p_values": raw_p_values,
            "holm_adjusted_p_values": adjusted_p_values,
            "confidence_intervals": "unadjusted percentile intervals",
            "significance_calls": "not made by this analysis",
        },
    }


def test_complete_render_has_exact_artifacts_order_and_source_numbers():
    result = _analysis()
    tables = render_tables(result, source_hash="a" * 64)

    assert set(tables) == set(OUTPUT_NAMES)
    assert all("COMPLETE FINAL ANALYSIS" in text for text in tables.values())
    assert all(
        "DRAFT GENERATED-TABLE PLACEHOLDER" not in text
        for text in tables.values()
    )
    main_table = tables["main_results.tex"]
    assert main_table.index("Oxford Pet") < main_table.index("Kvasir-SEG")
    assert main_table.index("Kvasir-SEG") < main_table.index("FIVES")
    assert r"\multicolumn{4}{c}{Oxford Pet}" in main_table
    assert r"\multicolumn{3}{c}{Kvasir-SEG}" in main_table
    assert r"\multicolumn{3}{c}{FIVES}" in main_table
    assert (
        "Confidence method & CLIP-G & CLIP-T & DL-T & DL-E & "
        "CLIP-G & CLIP-T & DL-T & CLIP-G & CLIP-T & DL-T"
    ) in main_table
    assert "Dataset & Model condition" not in main_table

    # Every unpooled condition is a column and every method is a row. Verify a
    # complete row directly against source values after canonical ordering.
    ordered = validate_analysis(result)
    expected_cells = []
    for condition in ordered:
        method = condition["risks"]["risk_dice"]["methods"]["confidence_sdc"]
        expected_cells.append(
            rf"\shortstack{{{method['aurc']:.4f}\\"
            rf"({method['normalized_aurc']:.4f})}}"
        )
    assert "SDC & " + " & ".join(expected_cells) + r" \\" in main_table
    assert main_table.count(r"\shortstack{") == 2 * 5 * 10
    for method_label in (
        "SDC",
        "Mean max probability",
        "Negative entropy",
        "Dice-M32",
        "nHD95-M32",
    ):
        assert main_table.count(f"\n{method_label} &") == 2
    assert r"lower is better" in main_table
    assert r"analytic expectation" in main_table

    cross_loss = tables["cross_loss_results.tex"]
    assert r"\multicolumn{4}{c}{Oxford Pet}" in cross_loss
    assert cross_loss.count(r"\shortstack{") == 2 * 2 * 10
    assert cross_loss.count("\nDice-M32 &") == 2
    assert cross_loss.count("\nnHD95-M32 &") == 2
    quadrature = tables["quadrature_ablation.tex"]
    assert r"\multicolumn{3}{c}{Kvasir-SEG}" in quadrature
    assert quadrature.count("\n$M=2$ &") == 2
    assert quadrature.count("\n$M=8$ &") == 2
    assert quadrature.count("\n$M=32$ &") == 2
    assert "no condition is pooled" in quadrature
    statistics = tables["statistical_tests.tex"]
    comparison = ordered[0]["comparisons"]["risk_dice"]
    assert f"{comparison['difference_left_minus_right']:.4f}" in statistics
    assert (
        f"[{comparison['bootstrap']['ci_low']:.4f}, "
        f"{comparison['bootstrap']['ci_high']:.4f}]"
    ) in statistics
    assert "same paired resamples" in statistics
    assert "Holm" in statistics
    assert "unadjusted" in statistics
    assert (
        "Method comparison / condition & Oxford Pet & Kvasir-SEG & FIVES"
        in statistics
    )
    assert "Dataset & Model condition" not in statistics
    assert statistics.count("\nDice-M32 $-$ nHD95-M32 / CLIP-G &") == 2
    assert statistics.count("\nDice-M32 $-$ nHD95-M32 / CLIP-T &") == 2
    assert statistics.count("\nDice-M32 $-$ nHD95-M32 / DL-T &") == 2
    assert statistics.count("\nDice-M32 $-$ nHD95-M32 / DL-E &") == 2
    assert statistics.count(r"\shortstack{") == 2 * 10


def test_final_validation_requires_exact_condition_risk_and_method_sets():
    incomplete = _analysis(condition_count=6, samples=100)
    with pytest.raises(ValueError, match="exactly 10 conditions"):
        validate_analysis(incomplete)
    ordered = validate_analysis(incomplete, allow_incomplete=True)
    assert len(ordered) == 6

    missing_risk = _analysis()
    del missing_risk["conditions"][0]["risks"]["risk_nhd95"]
    with pytest.raises(ValueError, match="exactly the two risks"):
        validate_analysis(missing_risk)

    missing_method = _analysis()
    del missing_method["conditions"][0]["risks"]["risk_dice"]["methods"][
        "confidence_sdc"
    ]
    with pytest.raises(ValueError, match="exactly the nine methods"):
        validate_analysis(missing_method)


def test_validation_rejects_inconsistent_displayed_statistics():
    bad_aurc = _analysis()
    bad_aurc["conditions"][0]["risks"]["risk_dice"]["methods"][
        "confidence_sdc"
    ]["excess_aurc"] += 0.1
    with pytest.raises(ValueError, match="excess_aurc is inconsistent"):
        validate_analysis(bad_aurc)

    bad_delta = _analysis()
    bad_delta["conditions"][0]["comparisons"]["risk_dice"][
        "difference_left_minus_right"
    ] += 0.1
    with pytest.raises(ValueError, match="difference disagrees"):
        validate_analysis(bad_delta)

    bad_hypothesis = _analysis()
    bad_hypothesis["multiple_testing"]["hypotheses"][0][
        "holm_adjusted_p_value"
    ] = 0.999
    with pytest.raises(ValueError, match="arrays disagree"):
        validate_analysis(bad_hypothesis)


def test_validation_rejects_legacy_randomization_and_missing_bootstrap_p_value():
    legacy_metadata = _analysis()
    legacy_metadata["analysis"]["randomization_samples"] = 10_000
    with pytest.raises(ValueError, match="analysis metadata.*schema"):
        validate_analysis(legacy_metadata)

    legacy_comparison = _analysis()
    legacy_comparison["conditions"][0]["comparisons"]["risk_dice"][
        "randomization"
    ] = {"p_value": 0.5}
    with pytest.raises(ValueError, match="comparison schema"):
        validate_analysis(legacy_comparison)

    missing_p = _analysis()
    del missing_p["conditions"][0]["comparisons"]["risk_dice"]["bootstrap"][
        "p_value"
    ]
    with pytest.raises(ValueError, match="bootstrap.*schema"):
        validate_analysis(missing_p)


def test_strict_loader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_analysis(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": NaN}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_analysis(nonfinite)


def test_cli_atomically_writes_only_four_tables(tmp_path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(_analysis(), allow_nan=False) + "\n")
    output = tmp_path / "tables"

    main(["--analysis", str(analysis), "--output-dir", str(output)])

    assert sorted(path.name for path in output.iterdir()) == sorted(OUTPUT_NAMES)
    for name in OUTPUT_NAMES:
        text = (output / name).read_text()
        assert text.endswith("\n")
        assert "AUTO-GENERATED" in text
        assert not list(output.glob(f".{name}.*.tmp"))


def test_source_object_is_not_mutated_by_validation_or_rendering():
    result = _analysis()
    before = copy.deepcopy(result)
    render_tables(result, source_hash="b" * 64)
    assert result == before
