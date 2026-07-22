"""Contract and rendering tests for scripts/render/paper.py."""

import copy
import json

import pytest

from scripts.render.paper import (
    ARCHITECTURE_DOMAIN_CONDITIONS,
    ARCHITECTURE_DOMAIN_HOLM_FAMILY_BY_DATASET,
    COMPLETION_MARKER,
    CONTRASTS,
    CONTROL_CONDITIONS,
    EXPECTED_CONDITIONS,
    HOLM_FAMILY_BY_DATASET,
    MAIN_METHODS,
    METHODS,
    OUTPUT_NAMES,
    RISKS,
    TARGET_CONDITIONS,
    _is_best,
    _method_label,
    load_analysis,
    main,
    render_tables,
    render_extension_tables,
    validate_analysis,
)


def _holm(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _analysis(
    *,
    condition_count=None,
    samples=10_000,
    expected_conditions=EXPECTED_CONDITIONS,
    family_by_dataset=HOLM_FAMILY_BY_DATASET,
):
    if condition_count is None:
        condition_count = len(expected_conditions)
    selected = expected_conditions[:condition_count]
    family_sizes = {}
    for dataset, _ in selected:
        family = family_by_dataset[dataset]
        family_sizes[family] = family_sizes.get(family, 0) + len(CONTRASTS)

    conditions = []
    hypotheses = []
    comparison_refs = {family: [] for family in family_sizes}
    hypothesis_refs = {family: [] for family in family_sizes}
    method_items = tuple(METHODS.items())
    for condition_index, (dataset, condition_name) in enumerate(reversed(selected)):
        risks = {}
        for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
            oracle = 0.01 + 0.001 * risk_index
            random = 0.90 + 0.001 * risk_index
            methods = {}
            for method_index, (method_field, label) in enumerate(method_items):
                aurc = (
                    0.10
                    + 0.003 * condition_index
                    + 0.02 * risk_index
                    + 0.002 * method_index
                )
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
                "label": risk_label,
                "methods": methods,
                "oracle_aurc": oracle,
                "random_aurc": random,
            }

        family = family_by_dataset[dataset]
        comparisons = {}
        for contrast_index, spec in enumerate(CONTRASTS):
            methods = risks[spec.risk]["methods"]
            difference = methods[spec.left]["aurc"] - methods[spec.right]["aurc"]
            p_value = min(
                1.0,
                2 / (samples + 1) + 1e-5 * (condition_index + contrast_index),
            )
            comparison = {
                "name": spec.name,
                "risk": spec.risk,
                "left": spec.left,
                "right": spec.right,
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
                    "seed": condition_index * len(CONTRASTS) + contrast_index,
                },
                "holm_family": family,
                "holm_family_size": family_sizes[family],
                "holm_adjusted_p_value": None,
            }
            hypothesis = {
                "dataset": dataset,
                "condition": condition_name,
                "contrast": spec.name,
                "risk": spec.risk,
                "left": spec.left,
                "right": spec.right,
                "holm_family": family,
                "holm_family_size": family_sizes[family],
                "raw_bootstrap_p_value": p_value,
                "holm_adjusted_p_value": None,
            }
            comparisons[spec.name] = comparison
            hypotheses.append(hypothesis)
            comparison_refs[family].append(comparison)
            hypothesis_refs[family].append(hypothesis)

        conditions.append(
            {
                "condition": condition_name,
                "dataset": dataset,
                "split": "test",
                "num_rows": 20,
                "num_image_clusters": 20,
                "jsonl": f"outputs/{dataset}/{condition_name}.jsonl",
                "manifest": f"outputs/{dataset}/{condition_name}.manifest.json",
                "jsonl_sha256": f"{condition_index:064x}",
                "manifest_sha256": f"{condition_index + 100:064x}",
                "risks": risks,
                "comparisons": comparisons,
                "numerical_validation": {
                    "reference": "confidence_dice_exact",
                    "absolute_error_definition": (
                        "per-image |C_Dice,M - C_Dice,Exact|"
                    ),
                    "rank_agreement_definition": (
                        "Spearman average ranks and Kendall tau-b"
                    ),
                    "exact_match_definition": (
                        "fraction with exact floating-point equality and no tolerance"
                    ),
                    "dice_quadrature": {
                        f"confidence_dice_m{count}": {
                            "m": count,
                            "num_images": 20,
                            "absolute_error": {
                                "mean": 0.01 / count,
                                "median": 0.008 / count,
                                "p95": 0.02 / count,
                                "max": 0.03 / count,
                            },
                            "rank_agreement": {
                                "spearman_rho": 1 - 0.01 / count,
                                "kendall_tau_b": 1 - 0.02 / count,
                            },
                            "exact_match_fraction": count / 64,
                        }
                        for count in (2, 8, 32)
                    },
                },
            }
        )

    families = {}
    for family, comparisons in comparison_refs.items():
        raw = [comparison["bootstrap"]["p_value"] for comparison in comparisons]
        adjusted = _holm(raw)
        for comparison, hypothesis, value in zip(
            comparisons, hypothesis_refs[family], adjusted
        ):
            comparison["holm_adjusted_p_value"] = value
            hypothesis["holm_adjusted_p_value"] = value
        families[family] = {
            "definition": f"synthetic {family} family",
            "num_hypotheses": len(raw),
            "raw_bootstrap_p_values": raw,
            "holm_adjusted_p_values": adjusted,
        }

    provenance_inputs = [
        {
            "logical_id": (
                f"{condition['dataset']}/{condition['condition']}/synthetic-run"
            ),
            "dataset": condition["dataset"],
            "condition": condition["condition"],
            "assembly_run_id": "synthetic-run",
            "assembly_source_sha256": "c" * 64,
            "artifact_id": f"artifact-{index}",
            "manifest_sha256": condition["manifest_sha256"],
            "records_sha256": condition["jsonl_sha256"],
            "sample_id_sha256": f"{index + 200:064x}",
            "num_samples": condition["num_rows"],
        }
        for index, condition in enumerate(conditions)
    ]
    return {
        "schema_version": 2,
        "provenance": {
            "binding": "campaign-lock",
            "campaign_id": "synthetic-campaign",
            "campaign_lock": {
                "logical_name": "campaign.lock.json",
                "sha256": "f" * 64,
            },
            "config_sha256": "e" * 64,
            "analysis_source_sha256": "d" * 64,
            "inputs": provenance_inputs,
        },
        "analysis": {
            "tie_policy": "analytic expectation over random within-tie order",
            "normalized_aurc": "normalized excess AURC",
            "comparisons": "four predeclared adjacent-geometry contrasts",
            "contrast_definitions": [
                {
                    "name": spec.name,
                    "left": spec.left,
                    "right": spec.right,
                    "risk": spec.risk,
                }
                for spec in CONTRASTS
            ],
            "cross_loss_policy": "all other score-risk cells are descriptive",
            "bootstrap_samples": samples,
            "bootstrap_workers": 4,
            "confidence_level": 0.95,
            "bootstrap_p_value": "paired equal-tail bootstrap tail probability",
            "seed": 0,
        },
        "conditions": conditions,
        "multiple_testing": {
            "procedure": "Holm step-down within each family",
            "family_policy": "core and extension are never pooled",
            "families": families,
            "total_hypotheses": len(CONTRASTS) * condition_count,
            "hypotheses": hypotheses,
            "confidence_intervals": "unadjusted percentile intervals",
            "significance_calls": "not made by this analysis",
        },
    }


def test_complete_render_has_declared_artifacts_and_main_orientation():
    result = _analysis()
    tables = render_tables(result, source_hash="a" * 64)

    assert set(tables) == set(OUTPUT_NAMES)
    assert result["multiple_testing"]["total_hypotheses"] == 64
    assert result["multiple_testing"]["families"]["core"]["num_hypotheses"] == 40
    assert result["multiple_testing"]["families"]["extension"]["num_hypotheses"] == 24
    assert all("COMPLETE FINAL ANALYSIS" in table for table in tables.values())

    main = tables["main_results.tex"]
    assert r"\label{tab:main-results}" in main
    assert r"\label{tab:main-results-dl}" in main
    assert r"\label{tab:adjacent-geometry-contrasts}" not in main
    assert (
        "Confidence method & Oxford Pet & Kvasir-SEG & FIVES & ISIC 2018 & TN3K" in main
    )
    assert "Dataset & Model condition" not in main
    assert main.count(r"\begin{table*}[t]") == 2
    # Two model-panel headings plus three symmetric risk headings per panel.
    assert main.count(r"\multicolumn{6}{l}{\textit{") == 2 + 2 * len(RISKS)
    for method_field in MAIN_METHODS:
        assert main.count(f"\n{_method_label(method_field)} &") == 2 * len(RISKS)
    assert "all seven matched-budget baselines" in main
    for spec in CONTRASTS:
        assert spec.name not in main
        assert f"{METHODS[spec.left]} $-$ {METHODS[spec.right]}" not in main
    assert r"\bestresult{" in main
    assert "Dark blue" in main
    assert "raw/Holm-transformed" not in main
    assert "CLIPSeg target (CLIP-T)" in main
    assert "DeepLabV3 target (DL-T)" in main
    assert "CLIP-T:" not in main and "DL-T:" not in main
    assert r"\resizebox{\textwidth}{!}" not in main
    assert main.count(r"\begin{tabular*}") == 2
    assert "significant" not in main.lower()
    assert "significance" not in main.lower()


def test_extension_render_is_separate_compact_and_dataset_oriented():
    result = _analysis(
        expected_conditions=ARCHITECTURE_DOMAIN_CONDITIONS,
        family_by_dataset=ARCHITECTURE_DOMAIN_HOLM_FAMILY_BY_DATASET,
    )

    ordered = validate_analysis(result, design="extension")
    tables = render_extension_tables(result, source_hash="7" * 64)

    assert len(ordered) == 7
    assert set(tables) == {
        "architecture_domain_extension.tex",
        "architecture_domain_extension_full.tex",
    }
    summary = tables["architecture_domain_extension.tex"]
    assert r"\label{tab:architecture-domain-extension}" in summary
    assert "Adjacent-geometry contrast" in summary
    assert "Oxford Pet" in summary and "DUTS" in summary
    assert "SF-T:" in summary and "DL-T:" in summary
    assert summary.count(r"\shortstack{") == len(CONTRASTS) * 6
    full = tables["architecture_domain_extension_full.tex"]
    assert "SegFormer-B2 target (SF-T)" in full
    assert "DeepLabV3 target (DL-T)" in full
    assert full.count(r"\begin{table*}[t]") == len(RISKS)


def test_top1_highlight_uses_the_exact_unrounded_minimum():
    methods = {
        "left": {"aurc": 0.1},
        "near": {"aurc": 0.1 + 5e-11},
    }
    assert _is_best(methods, "left", ("left", "near"))
    assert not _is_best(methods, "near", ("left", "near"))


def test_appendix_tables_cover_17_by_3_and_full_3_by_3_without_pooling():
    tables = render_tables(_analysis(), source_hash="b" * 64)

    target = tables["full_target_results.tex"]
    assert target.count(r"\begin{table*}[t]") == len(RISKS)
    assert target.count(r"\begin{tabular*}{\textwidth}") == 2 * len(RISKS)
    assert r"\resizebox" not in target
    assert target.count(r"{\scriptsize") == len(RISKS)
    assert r"\shortstack{" not in target
    for method_field in METHODS:
        assert target.count(f"\n{_method_label(method_field)} &") == 2 * len(RISKS)
    for panel_label in (
        "CLIPSeg target (CLIP-T)",
        "DeepLabV3 target (DL-T)",
    ):
        assert target.count(panel_label) == len(RISKS)
    for dataset_label in ("Oxford Pet", "Kvasir-SEG", "FIVES", "ISIC 2018", "TN3K"):
        assert target.count(dataset_label) == 2 * len(RISKS)
    assert target.count(r"\bestresult{") == len(TARGET_CONDITIONS) * len(RISKS)
    assert r"raw AURC $\times100$ (nAURC)" in target

    control = tables["complete_results.tex"]
    assert control.count(r"\begin{table*}[t]") == len(RISKS)
    assert control.count(r"\begin{tabular*}{\textwidth}") == len(RISKS)
    assert control.count(r"\begin{tabular*}{0.62\textwidth}") == len(RISKS)
    assert r"\resizebox" not in control
    assert control.count(r"{\scriptsize") == len(RISKS)
    assert r"\shortstack{" not in control
    for method_field in METHODS:
        assert control.count(f"\n{_method_label(method_field)} &") == 2 * len(RISKS)
    for panel_label in (
        "CLIPSeg general (CLIP-G)",
        "DeepLabV3 external (DL-E)",
    ):
        assert control.count(panel_label) == len(RISKS)
    assert control.count("Oxford Pet") == 2 * len(RISKS)
    for dataset_label in ("Kvasir-SEG", "FIVES", "ISIC 2018", "TN3K"):
        assert control.count(dataset_label) == len(RISKS)
    assert control.count(r"\bestresult{") == len(CONTROL_CONDITIONS) * len(RISKS)
    assert r"raw AURC $\times100$ (nAURC)" in control

    cross = tables["cross_loss_results.tex"]
    assert "Full $3\\times3$" in cross
    assert cross.count(r"\begin{tabular*}{\textwidth}") == 3
    assert cross.count(r"\begin{tabular*}{0.62\textwidth}") == 1
    assert r"\resizebox" not in cross
    assert cross.count(r"{\scriptsize") == 1
    assert r"\shortstack{" not in cross
    for method_field in (
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    ):
        assert cross.count(f"\n{_method_label(method_field)} &") == 4 * len(RISKS)
    for panel_label in (
        "CLIPSeg general (CLIP-G)",
        "CLIPSeg target (CLIP-T)",
        "DeepLabV3 target (DL-T)",
        "DeepLabV3 external (DL-E)",
    ):
        assert cross.count(panel_label) == 1
    for risk_label in RISKS.values():
        assert cross.count(rf"\textit{{{risk_label}}}") == 4
    assert cross.count("Oxford Pet") == 4
    for dataset_label in ("Kvasir-SEG", "FIVES", "ISIC 2018", "TN3K"):
        assert cross.count(dataset_label) == 3
    assert cross.count(r"\bestresult{") == len(EXPECTED_CONDITIONS) * len(RISKS)
    assert "descriptive" in cross


def test_quadrature_table_includes_exact_and_all_declared_midpoint_rules():
    table = render_tables(_analysis(), source_hash="c" * 64)["quadrature_ablation.tex"]
    primary, fidelity = table.split(r"\clearpage", 1)

    assert primary.count(r"\begin{tabular*}{\textwidth}") == 2
    assert r"\resizebox" not in primary
    assert r"\resizebox{\textwidth}{!}" in fidelity
    assert r"\shortstack{" not in primary
    assert primary.count(f"\n{_method_label('confidence_dice_exact')} &") == 2
    for count in (2, 8, 32):
        method_field = f"confidence_dice_m{count}"
        assert primary.count(f"\n{_method_label(method_field)} &") == 2
        assert table.count(f"\n{_method_label(method_field)} / ") == 6
    for prefix in ("nHD", "nHD95"):
        for count in (2, 8, 32):
            method_field = f"confidence_{prefix.lower()}_m{count}"
            assert primary.count(f"\n{_method_label(method_field)} &") == 2
    for panel_label in (
        "CLIPSeg target (CLIP-T)",
        "DeepLabV3 target (DL-T)",
    ):
        assert primary.count(panel_label) == 1
    for dataset_label in ("Oxford Pet", "Kvasir-SEG", "FIVES", "ISIC 2018", "TN3K"):
        assert primary.count(dataset_label) == 2
    assert "exact level-set oracle" in table
    assert table.count(r"\bestresult{") == len(RISKS) * len(TARGET_CONDITIONS)
    assert "Dark blue" in table
    assert "lowest unrounded AURC" in table
    assert "AURC need not improve monotonically" in table
    assert r"\label{tab:dice-quadrature-fidelity}" in table
    assert "incomplete drafts" not in table
    for statistic in (
        "Mean abs. error",
        "Median abs. error",
        "P95 abs. error",
        "Max abs. error",
        r"Spearman $\rho$",
        r"Kendall $\tau_b$",
    ):
        assert table.count(f"/ {statistic} &") == 3
    assert "Exact match" not in table
    assert "do not enter the four fixed contrasts" in table


def test_final_renderer_requires_numerical_validation():
    result = _analysis()
    for condition in result["conditions"]:
        del condition["numerical_validation"]
    with pytest.raises(ValueError, match="numerical_validation is required"):
        validate_analysis(result)
    validate_analysis(result, allow_incomplete=True)
    table = render_tables(result, source_hash="9" * 64, allow_incomplete=True)[
        "quadrature_ablation.tex"
    ]
    assert r"\label{tab:quadrature-ablation}" in table
    assert "tab:dice-quadrature-fidelity" not in table


def test_full_contrast_table_reports_four_contrasts_and_all_64_conditions():
    table = render_tables(_analysis(), source_hash="d" * 64)["statistical_tests.tex"]

    assert "subset of the 64 fixed" in table
    assert "40 comparisons" in table
    assert "24 comparisons" in table
    assert r"\label{tab:statistical-tests}" in table
    assert r"\label{tab:statistical-tests-extension}" in table
    assert table.count(r"\shortstack{") == len(CONTRASTS) * 5
    for spec in CONTRASTS:
        row = (
            f"{_method_label(spec.left)} $-$ {_method_label(spec.right)} / "
            f"{RISKS[spec.risk]}"
        )
        assert table.count(f"\n{row} &") == 2
    assert "pointwise 95\\% paired" in table
    assert "Holm" not in table
    assert "tail probabilities" not in table
    assert "descriptive rather than simultaneous tests" in table
    assert "significant" not in table.lower()
    assert "significance" not in table.lower()


def test_validation_rejects_old_schema_and_wrong_dimensions():
    old = _analysis()
    old["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version 1 is obsolete"):
        validate_analysis(old)

    incomplete = _analysis(condition_count=6)
    with pytest.raises(ValueError, match="exactly 16"):
        validate_analysis(incomplete)
    assert len(validate_analysis(incomplete, allow_incomplete=True)) == 6

    missing_risk = _analysis()
    del missing_risk["conditions"][0]["risks"]["risk_nhd"]
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_analysis(missing_risk)

    missing_method = _analysis()
    del missing_method["conditions"][0]["risks"]["risk_dice"]["methods"][
        "confidence_nhd_m32"
    ]
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_analysis(missing_method)

    missing_contrast = _analysis()
    del missing_contrast["conditions"][0]["comparisons"][CONTRASTS[0].name]
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_analysis(missing_contrast)

    bad_numerical = _analysis()
    bad_numerical["conditions"][0]["numerical_validation"]["dice_quadrature"][
        "confidence_dice_m32"
    ]["absolute_error"]["p95"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_analysis(bad_numerical)

    mixed_numerical = _analysis()
    del mixed_numerical["conditions"][0]["numerical_validation"]
    with pytest.raises(ValueError, match="numerical_validation is required"):
        validate_analysis(mixed_numerical)


def test_validation_rejects_inconsistent_statistics_and_holm_contract():
    bad_delta = _analysis()
    comparison = bad_delta["conditions"][0]["comparisons"][CONTRASTS[0].name]
    comparison["difference_left_minus_right"] += 0.1
    with pytest.raises(ValueError, match="difference disagrees"):
        validate_analysis(bad_delta)

    bad_holm = _analysis()
    bad_holm["multiple_testing"]["families"]["core"]["holm_adjusted_p_values"][0] = (
        0.999
    )
    with pytest.raises(ValueError, match="Holm values are inconsistent"):
        validate_analysis(bad_holm)

    bad_definition = _analysis()
    bad_definition["analysis"]["contrast_definitions"][0]["risk"] = "risk_nhd95"
    with pytest.raises(ValueError, match="contrast_definitions"):
        validate_analysis(bad_definition)


def test_strict_loader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 2, "schema_version": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_analysis(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": NaN}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_analysis(nonfinite)


def test_cli_atomically_writes_only_declared_tables(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(_analysis()))
    output_dir = tmp_path / "tables"

    main(["--analysis", str(analysis_path), "--output-dir", str(output_dir)])

    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_NAMES) | {
        COMPLETION_MARKER
    }
    assert all((output_dir / name).read_text().endswith("\n") for name in OUTPUT_NAMES)
    assert (output_dir / COMPLETION_MARKER).read_text().endswith("\n")


def test_validation_and_rendering_do_not_mutate_source():
    result = _analysis()
    original = copy.deepcopy(result)
    validate_analysis(result)
    render_tables(result, source_hash="e" * 64)
    assert result == original
