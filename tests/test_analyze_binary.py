"""Strict I/O, statistics, and artifacts of scripts/analyze_binary.py."""

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from selectseg.binary_framework import tie_aware_expected_aurc

from scripts.analyze_binary import (
    CONTRASTS,
    CSV_NAME,
    EXPECTED_CONDITIONS,
    JSON_NAME,
    LATEX_NAME,
    METHODS,
    RISKS,
    analyze_conditions,
    holm_adjust,
    load_condition,
    main,
    paired_cluster_bootstrap_aurc_test,
    validate_campaign_bound_conditions,
)


def _rows(count=6):
    rows = []
    for index in range(count):
        risk_dice = (index + 1) / (count + 2)
        risk_hd = ((2 * index) % (count + 1) + 0.25) / (count + 2)
        risk_hd95 = ((3 * index) % (count + 1) + 0.5) / (count + 2)
        row = {
            "schema_version": 2,
            "run_id": "synthetic-run",
            "sample_id": f"image-{index}:class-1",
            "image_id": f"image-{index}",
            "image_index": index,
            "class_index": 1,
            "class_name": "foreground",
            "height": 16,
            "width": 20,
            "image_diagonal": (16**2 + 20**2) ** 0.5,
            "risk_dice": risk_dice,
            "risk_nhd": risk_hd,
            "risk_nhd95": risk_hd95,
            "risk_hd_pixels": 40 * risk_hd,
            "risk_hd95_pixels": 40 * risk_hd95,
            "confidence_sdc": 1 - 0.8 * risk_dice,
            "confidence_mean_max_probability": 0.95 - 0.4 * risk_dice,
            "confidence_negative_entropy": -0.5 * risk_dice,
            "confidence_dice_exact": -0.95 * risk_dice,
            "confidence_qfr_entropy": -0.45 * risk_dice,
            "confidence_plm10_entropy": -0.55 * risk_dice,
            "confidence_mmmc_entropy": -0.65 * risk_dice,
            "confidence_foreground_entropy": -0.75 * risk_dice,
            "confidence_dice_m2": -0.9 * risk_dice,
            "confidence_dice_m8": -0.8 * risk_dice,
            "confidence_dice_m32": -risk_dice,
            "confidence_nhd_m2": -0.9 * risk_hd,
            "confidence_nhd_m8": -0.8 * risk_hd,
            "confidence_nhd_m32": -risk_hd,
            "confidence_nhd95_m2": -0.9 * risk_hd95,
            "confidence_nhd95_m8": -0.8 * risk_hd95,
            "confidence_nhd95_m32": -risk_hd95,
        }
        rows.append(row)
    return rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _brute_tie_aware_aurc(confidences, risks):
    """Independent factorial oracle for tiny exact-tie examples."""

    confidences = tuple(confidences)
    risks = np.asarray(risks, dtype=float)
    groups = []
    for score in sorted(set(confidences), reverse=True):
        members = tuple(
            index for index, value in enumerate(confidences) if value == score
        )
        groups.append(tuple(itertools.permutations(members)))
    values = []
    for within_group_orders in itertools.product(*groups):
        order = np.asarray(
            tuple(itertools.chain.from_iterable(within_group_orders)), dtype=int
        )
        prefix_risk = np.cumsum(risks[order]) / np.arange(1, risks.size + 1)
        values.append(float(prefix_risk.mean()))
    return float(np.mean(values))


def _write_condition(
    directory,
    *,
    condition="clipseg-target",
    dataset="pet",
    rows=None,
    manifest_updates=None,
    allow_nan=False,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(_rows() if rows is None else rows)
    jsonl = directory / f"{condition}_{dataset}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, allow_nan=allow_nan) + "\n" for row in rows)
    )
    manifest = {
        "schema_version": 2,
        "run_id": "synthetic-run",
        "condition": condition,
        "dataset": dataset,
        "split": "test",
        "num_rows": len(rows),
        "num_images": len(rows),
        "jsonl_sha256": _sha256(jsonl),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode()
        ).hexdigest(),
        "risk_fields": ["risk_dice", "risk_nhd", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd_pixels", "risk_hd95_pixels"],
        "score_fields": [field for field, _ in METHODS],
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    manifest_path = jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return jsonl, manifest_path


def test_holm_adjust_matches_the_step_down_definition_and_validates():
    # Sorted p-values .01, .03, .04 -> .03, .06, max(.06, .04)=.06.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([0.2, 1.1])
    with pytest.raises(ValueError, match="one-dimensional"):
        holm_adjust([[0.1]])


def test_paired_bootstrap_test_is_deterministic_paired_and_clustered():
    left = [0.9, 0.7, 0.2, 0.1, 0.6, 0.4]
    right = [0.8, 0.6, 0.3, 0.2, 0.4, 0.7]
    risks = [0.0, 0.3, 0.9, 0.7, 0.4, 0.5]
    image_ids = ["a", "a", "b", "b", "c", "c"]
    first = paired_cluster_bootstrap_aurc_test(
        left,
        right,
        risks,
        cluster_ids=image_ids,
        n_resamples=500,
        seed=11,
    )
    second = paired_cluster_bootstrap_aurc_test(
        left,
        right,
        risks,
        cluster_ids=image_ids,
        n_resamples=500,
        seed=11,
    )
    assert first == second
    assert first.n_observations == 6
    assert first.n_clusters == 3
    assert first.n_resamples == 500
    assert 2 / 501 <= first.p_value <= 1
    assert first.ci_low <= first.ci_high

    identical = paired_cluster_bootstrap_aurc_test(
        left,
        left,
        risks,
        cluster_ids=image_ids,
        n_resamples=20,
        seed=2,
    )
    assert identical.difference == 0
    assert identical.ci_low == 0
    assert identical.ci_high == 0
    assert identical.p_value == 1


def test_paired_bootstrap_is_scale_invariant_and_ci_p_value_aligned():
    left = [6, 5, 4, 3, 2, 1]
    right = [1, 2, 3, 4, 5, 6]
    risks = [0, 0, 0.1, 0.8, 0.9, 1]
    kwargs = {
        "cluster_ids": list("abcdef"),
        "n_resamples": 1_000,
        "seed": 2,
    }
    original = paired_cluster_bootstrap_aurc_test(left, right, risks, **kwargs)
    transformed = paired_cluster_bootstrap_aurc_test(
        [17 * value - 9 for value in left],
        [0.01 * value + 100 for value in right],
        risks,
        **kwargs,
    )

    assert transformed == original
    excludes_zero = original.ci_low > 0 or original.ci_high < 0
    assert excludes_zero == (original.p_value < 0.05)
    assert excludes_zero

    rng = np.random.default_rng(kwargs["seed"])
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    risk_array = np.asarray(risks, dtype=float)
    draws = []
    for _ in range(kwargs["n_resamples"]):
        indices = rng.integers(0, len(left), size=len(left))
        draws.append(
            tie_aware_expected_aurc(left_array[indices], risk_array[indices])
            - tie_aware_expected_aurc(right_array[indices], risk_array[indices])
        )
    draws = np.asarray(draws)
    expected_low, expected_high = np.quantile(draws, [0.025, 0.975])
    expected_p = min(
        1.0,
        2
        * min(
            (1 + np.count_nonzero(draws <= 0)) / (len(draws) + 1),
            (1 + np.count_nonzero(draws >= 0)) / (len(draws) + 1),
        ),
    )
    assert original.ci_low == pytest.approx(expected_low)
    assert original.ci_high == pytest.approx(expected_high)
    assert original.p_value == pytest.approx(expected_p)


def test_tie_aware_aurc_matches_factorial_oracle_exhaustively():
    # Exhaust all binary score/risk vectors of length four. This covers one
    # four-way tie, every possible split into two exact-tie blocks, and all
    # risk allocations without reusing the production formula as the oracle.
    for confidences in itertools.product((0.0, 1.0), repeat=4):
        for risks in itertools.product((0.0, 1.0), repeat=4):
            assert tie_aware_expected_aurc(confidences, risks) == pytest.approx(
                _brute_tie_aware_aurc(confidences, risks), abs=1e-14
            )


def test_tie_aware_bootstrap_matches_independent_factorial_oracle():
    left = np.asarray([1.0, 1.0, 0.0, 0.0])
    right = np.asarray([1.0, 0.0, 1.0, 0.0])
    risks = np.asarray([0.0, 0.4, 0.7, 1.0])
    resamples = 250
    seed = 31
    observed = paired_cluster_bootstrap_aurc_test(
        left,
        right,
        risks,
        cluster_ids=["a", "b", "c", "d"],
        n_resamples=resamples,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    oracle_draws = []
    for _ in range(resamples):
        indices = rng.integers(0, left.size, size=left.size)
        oracle_draws.append(
            _brute_tie_aware_aurc(left[indices], risks[indices])
            - _brute_tie_aware_aurc(right[indices], risks[indices])
        )
    oracle_draws = np.asarray(oracle_draws)
    low, high = np.quantile(oracle_draws, [0.025, 0.975])
    lower_tail = (1 + np.count_nonzero(oracle_draws <= 0)) / (resamples + 1)
    upper_tail = (1 + np.count_nonzero(oracle_draws >= 0)) / (resamples + 1)

    assert observed.difference == pytest.approx(
        _brute_tie_aware_aurc(left, risks) - _brute_tie_aware_aurc(right, risks),
        abs=1e-14,
    )
    assert observed.ci_low == pytest.approx(low, abs=1e-14)
    assert observed.ci_high == pytest.approx(high, abs=1e-14)
    assert observed.p_value == pytest.approx(
        min(1.0, 2 * min(lower_tail, upper_tail)), abs=1e-14
    )


def test_load_condition_accepts_a_complete_hashed_schema(tmp_path):
    jsonl, manifest = _write_condition(tmp_path)
    loaded = load_condition(jsonl)
    assert loaded.jsonl_path == jsonl
    assert loaded.manifest_path == manifest
    assert loaded.condition == "clipseg-target"
    assert loaded.dataset == "pet"
    assert len(loaded.rows) == 6


def test_three_loss_schema_and_adjacent_geometry_contrasts_are_declarative():
    assert len(METHODS) == 17
    assert len(RISKS) == 3
    assert [field for field, _ in RISKS] == [
        "risk_dice",
        "risk_nhd",
        "risk_nhd95",
    ]
    assert [
        (contrast.name, contrast.left, contrast.right, contrast.risk)
        for contrast in CONTRASTS
    ] == [
        (
            "dice_vs_nhd_under_dice",
            "confidence_dice_m32",
            "confidence_nhd_m32",
            "risk_dice",
        ),
        (
            "dice_vs_nhd_under_nhd",
            "confidence_dice_m32",
            "confidence_nhd_m32",
            "risk_nhd",
        ),
        (
            "nhd_vs_nhd95_under_nhd",
            "confidence_nhd_m32",
            "confidence_nhd95_m32",
            "risk_nhd",
        ),
        (
            "nhd_vs_nhd95_under_nhd95",
            "confidence_nhd_m32",
            "confidence_nhd95_m32",
            "risk_nhd95",
        ),
    ]


def test_load_condition_accepts_absent_optional_pixel_auxiliaries(tmp_path):
    rows = _rows()
    for row in rows:
        del row["risk_hd_pixels"]
        del row["risk_hd95_pixels"]
    jsonl, _ = _write_condition(
        tmp_path,
        rows=rows,
        manifest_updates={"auxiliary_fields": []},
    )
    loaded = load_condition(jsonl)
    assert "risk_hd_pixels" not in loaded.rows[0]
    assert "risk_hd95_pixels" not in loaded.rows[0]


def test_load_condition_explicitly_rejects_the_old_two_loss_schema(tmp_path):
    rows = _rows()
    old_scores = [
        field for field, _ in METHODS if not field.startswith("confidence_nhd_m")
    ]
    for row in rows:
        del row["risk_nhd"]
        del row["risk_hd_pixels"]
        for field in set(row) - set(old_scores):
            if field.startswith("confidence_nhd_m"):
                del row[field]
    jsonl, _ = _write_condition(
        tmp_path,
        rows=rows,
        manifest_updates={
            "risk_fields": ["risk_dice", "risk_nhd95"],
            "auxiliary_fields": ["risk_hd95_pixels"],
            "score_fields": old_scores,
        },
    )
    with pytest.raises(ValueError, match="lacks required score/risk fields"):
        load_condition(jsonl)


def test_load_condition_matches_evaluator_records_and_manifest_names(tmp_path):
    jsonl, manifest = _write_condition(tmp_path / "run")
    records = jsonl.with_name("records.jsonl")
    canonical_manifest = jsonl.with_name("manifest.json")
    jsonl.rename(records)
    manifest.rename(canonical_manifest)
    loaded = load_condition(records)
    assert loaded.manifest_path == canonical_manifest


def test_load_condition_rejects_hash_row_count_duplicate_and_nonfinite(tmp_path):
    bad_hash, _ = _write_condition(tmp_path / "hash")
    bad_hash.write_text(bad_hash.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_condition(bad_hash)

    bad_count, _ = _write_condition(
        tmp_path / "count", manifest_updates={"num_rows": 99, "num_images": 99}
    )
    with pytest.raises(ValueError, match="row-count mismatch"):
        load_condition(bad_count)

    duplicate_rows = _rows()
    duplicate_rows[1]["sample_id"] = duplicate_rows[0]["sample_id"]
    duplicate, _ = _write_condition(tmp_path / "duplicate", rows=duplicate_rows)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_condition(duplicate)

    nonfinite_rows = _rows()
    nonfinite_rows[0]["risk_dice"] = float("nan")
    nonfinite, _ = _write_condition(
        tmp_path / "nonfinite", rows=nonfinite_rows, allow_nan=True
    )
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_condition(nonfinite)


def test_load_condition_rejects_manifest_row_schema_disagreement(tmp_path):
    missing_rows = _rows()
    del missing_rows[0]["confidence_nhd95_m32"]
    missing, _ = _write_condition(tmp_path / "missing", rows=missing_rows)
    with pytest.raises(ValueError, match="lack required fields"):
        load_condition(missing)

    extra_rows = _rows()
    for row in extra_rows:
        row["confidence_unregistered"] = 0.5
    extra, _ = _write_condition(tmp_path / "extra", rows=extra_rows)
    with pytest.raises(ValueError, match="manifest/row score-risk schema mismatch"):
        load_condition(extra)

    unexpected_risk_rows = _rows()
    for row in unexpected_risk_rows:
        row["risk_unregistered"] = 0.25
    unexpected_risk, _ = _write_condition(
        tmp_path / "unexpected-risk",
        rows=unexpected_risk_rows,
        manifest_updates={
            "risk_fields": [
                "risk_dice",
                "risk_nhd",
                "risk_nhd95",
                "risk_unregistered",
            ]
        },
    )
    with pytest.raises(ValueError, match="exactly the three main risks"):
        load_condition(unexpected_risk)


def test_analyze_conditions_rejects_duplicate_dataset_condition_pairs(tmp_path):
    first, _ = _write_condition(tmp_path / "one")
    second, _ = _write_condition(tmp_path / "two")
    with pytest.raises(ValueError, match="exactly once"):
        analyze_conditions(
            [load_condition(first), load_condition(second)],
            bootstrap_samples=5,
        )


def test_analyze_conditions_requires_declared_complete_matrix_by_default(tmp_path):
    jsonl, _ = _write_condition(tmp_path / "one")
    condition = load_condition(jsonl)
    with pytest.raises(ValueError, match="exactly the 16 declared conditions"):
        analyze_conditions([condition], bootstrap_samples=5)

    result = analyze_conditions([condition], bootstrap_samples=5, allow_incomplete=True)
    assert result["multiple_testing"]["total_hypotheses"] == 4
    assert result["multiple_testing"]["families"]["core"]["num_hypotheses"] == 4
    comparisons = result["conditions"][0]["comparisons"]
    assert set(comparisons) == {contrast.name for contrast in CONTRASTS}
    for contrast in CONTRASTS:
        comparison = comparisons[contrast.name]
        assert comparison["name"] == contrast.name
        assert comparison["risk"] == contrast.risk
        assert comparison["left"] == contrast.left
        assert comparison["right"] == contrast.right
        rows = condition.rows
        expected_difference = tie_aware_expected_aurc(
            [row[contrast.left] for row in rows],
            [row[contrast.risk] for row in rows],
        ) - tie_aware_expected_aurc(
            [row[contrast.right] for row in rows],
            [row[contrast.risk] for row in rows],
        )
        assert comparison["difference_left_minus_right"] == pytest.approx(
            expected_difference
        )

    numerical = result["conditions"][0]["numerical_validation"]
    assert numerical["reference"] == "confidence_dice_exact"
    assert set(numerical["dice_quadrature"]) == {
        "confidence_dice_m2",
        "confidence_dice_m8",
        "confidence_dice_m32",
    }
    reference = np.asarray([row["confidence_dice_exact"] for row in condition.rows])
    approximation = np.asarray([row["confidence_dice_m32"] for row in condition.rows])
    errors = np.abs(approximation - reference)
    m32 = numerical["dice_quadrature"]["confidence_dice_m32"]
    assert m32["m"] == 32
    assert m32["num_images"] == len(condition.rows)
    assert m32["absolute_error"] == pytest.approx(
        {
            "mean": errors.mean(),
            "median": np.median(errors),
            "p95": np.quantile(errors, 0.95),
            "max": errors.max(),
        }
    )
    assert m32["rank_agreement"] == pytest.approx(
        {"spearman_rho": 1.0, "kendall_tau_b": 1.0}
    )
    assert m32["exact_match_fraction"] == 0


def test_numerical_validation_marks_constant_rank_agreement_undefined(tmp_path):
    rows = _rows()
    for row in rows:
        row["confidence_dice_exact"] = 0.25
        row["confidence_dice_m2"] = 0.25
    jsonl, _ = _write_condition(tmp_path, rows=rows)
    result = analyze_conditions(
        [load_condition(jsonl)], bootstrap_samples=2, allow_incomplete=True
    )
    m2 = result["conditions"][0]["numerical_validation"]["dice_quadrature"][
        "confidence_dice_m2"
    ]
    assert m2["absolute_error"] == {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }
    assert m2["rank_agreement"] == {
        "spearman_rho": None,
        "kendall_tau_b": None,
    }
    assert m2["exact_match_fraction"] == 1.0


def test_complete_analysis_uses_separate_core_and_extension_holm_families(tmp_path):
    conditions = []
    for dataset, condition_name in EXPECTED_CONDITIONS:
        jsonl, _ = _write_condition(
            tmp_path / f"{dataset}-{condition_name}",
            dataset=dataset,
            condition=condition_name,
        )
        conditions.append(load_condition(jsonl))

    result = analyze_conditions(conditions, bootstrap_samples=2)
    multiple = result["multiple_testing"]
    assert multiple["total_hypotheses"] == 64
    assert multiple["families"]["core"]["num_hypotheses"] == 40
    assert multiple["families"]["extension"]["num_hypotheses"] == 24
    assert all(
        comparison["holm_family_size"]
        == {"core": 40, "extension": 24}[comparison["holm_family"]]
        for condition in result["conditions"]
        for comparison in condition["comparisons"].values()
    )


def test_cli_explicit_inputs_write_deterministic_json_csv_and_latex(
    tmp_path, monkeypatch
):
    jsonl, _ = _write_condition(tmp_path / "input")
    output_one = tmp_path / "output-one"
    output_two = tmp_path / "output-two"

    base_args = [
        "analyze_binary.py",
        "--inputs",
        str(jsonl),
        "--bootstrap-samples",
        "40",
        "--seed",
        "123",
        "--allow-incomplete",
    ]
    monkeypatch.setattr(sys, "argv", [*base_args, "--output-dir", str(output_one)])
    main()
    monkeypatch.setattr(sys, "argv", [*base_args, "--output-dir", str(output_two)])
    main()

    for filename in (JSON_NAME, CSV_NAME, LATEX_NAME):
        assert (output_one / filename).read_bytes() == (
            output_two / filename
        ).read_bytes()

    summary_text = (output_one / JSON_NAME).read_text()
    assert ": NaN" not in summary_text
    summary = json.loads(summary_text)
    assert len(summary["conditions"]) == 1
    condition = summary["conditions"][0]
    assert set(condition["risks"]) == {"risk_dice", "risk_nhd", "risk_nhd95"}
    assert len(condition["risks"]["risk_dice"]["methods"]) == len(METHODS)
    assert summary["multiple_testing"]["total_hypotheses"] == 4
    assert set(summary["multiple_testing"]["families"]) == {"core"}
    assert "raw_bootstrap_p_values" in summary["multiple_testing"]["families"]["core"]
    assert "randomization" not in summary_text.lower()
    assert summary["multiple_testing"]["significance_calls"] == (
        "not made by this analysis"
    )
    for comparison in condition["comparisons"].values():
        assert 0 <= comparison["bootstrap"]["p_value"] <= 1
        assert 0 <= comparison["holm_adjusted_p_value"] <= 1
        assert comparison["holm_family"] == "core"
        assert comparison["holm_family_size"] == 4
        assert "significant" not in comparison

    csv_lines = (output_one / CSV_NAME).read_text().splitlines()
    assert len(csv_lines) == 1 + len(RISKS) * len(METHODS)
    assert "normalized_aurc" in csv_lines[0]
    csv_rows = list(csv.DictReader(csv_lines))
    numerical_rows = [
        row
        for row in csv_rows
        if row["risk"] == "risk_dice"
        and row["method"]
        in {
            "confidence_dice_m2",
            "confidence_dice_m8",
            "confidence_dice_m32",
        }
    ]
    assert len(numerical_rows) == 3
    assert all(
        row["numerical_reference"] == "confidence_dice_exact" for row in numerical_rows
    )
    assert all(row["absolute_error_p95"] for row in numerical_rows)
    assert all(row["spearman_rho"] == "1" for row in numerical_rows)
    assert all(row["kendall_tau_b"] == "1" for row in numerical_rows)
    assert not any(
        row["numerical_reference"] for row in csv_rows if row["risk"] != "risk_dice"
    )
    latex = (output_one / LATEX_NAME).read_text()
    assert r"\begin{table*}" in latex
    assert "Dice-M32" in latex
    assert "nHD-M32" in latex
    assert "nHD95-M32" in latex
    assert r"AURC $\times100$" in latex
    assert "multiplied by 100 for display only" in latex


def test_campaign_binding_requires_locked_final_assemblies(tmp_path):
    prepared = []
    artifacts = []
    for index, (dataset, condition_name) in enumerate(EXPECTED_CONDITIONS):
        jsonl, manifest_path = _write_condition(
            tmp_path / f"{dataset}-{condition_name}",
            dataset=dataset,
            condition=condition_name,
        )
        manifest = json.loads(manifest_path.read_text())
        artifact = {
            "manifest_path": f"frozen/{dataset}/{condition_name}/manifest.json",
            "manifest_sha256": f"{index + 300:064x}",
            "artifact_id": f"artifact-{index:02d}",
            "dataset": dataset,
            "condition": condition_name,
            "model": "synthetic-model",
            "split": "test",
            "checkpoint_sha256": None,
            "source_sha256": f"{index + 400:064x}",
            "sample_id_sha256": manifest["sample_id_sha256"],
            "num_samples": manifest["num_images"],
        }
        prepared.append((jsonl, manifest_path, manifest, artifact))
        artifacts.append(artifact)

    lock_path = tmp_path / "campaign.lock.json"
    lock = {
        "lock_schema_version": 1,
        "campaign_id": "synthetic-campaign",
        "config": {"path": "configs/final.json", "sha256": "a" * 64},
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "estimator": {
            "spec_path": "configs/estimators/midpoint-v1.json",
            "spec_sha256": "b" * 64,
            "estimator_id": "midpoint-v1",
            "target_measure": "uniform-threshold",
        },
        "paths": {},
        "artifacts": artifacts,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    lock_sha = _sha256(lock_path)

    conditions = []
    for jsonl, manifest_path, manifest, artifact in prepared:
        manifest.update(
            {
                "artifact_type": "selectseg.binary_simulation_assembly",
                "model": artifact["model"],
                "checkpoint": None,
                "assembly": {
                    "assembly_schema_version": 2,
                    "assembly_source_sha256": "c" * 64,
                    "campaign_id": lock["campaign_id"],
                    "campaign_lock_sha256": lock_sha,
                    "artifact_id": artifact["artifact_id"],
                    "artifact_manifest_sha256": artifact["manifest_sha256"],
                    "artifact_source_sha256": artifact["source_sha256"],
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        conditions.append(load_condition(jsonl))

    provenance = validate_campaign_bound_conditions(conditions, lock_path)
    assert provenance["binding"] == "campaign-lock"
    assert provenance["campaign_lock"]["sha256"] == lock_sha
    assert len(provenance["inputs"]) == len(EXPECTED_CONDITIONS)
    assert all(
        "/scratch.global/" not in item["logical_id"] for item in provenance["inputs"]
    )

    bad = conditions[0]
    bad.manifest["assembly"]["campaign_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign_lock_sha256 differs"):
        validate_campaign_bound_conditions(conditions, lock_path)


def test_cli_rejects_removed_randomization_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_binary.py", "--randomization-samples", "10"],
    )
    with pytest.raises(SystemExit):
        main()
