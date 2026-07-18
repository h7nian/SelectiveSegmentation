"""Strict I/O, statistics, and artifacts of scripts/analyze_binary.py."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from selectseg.binary_framework import tie_aware_expected_aurc

from scripts.analyze_binary import (
    CSV_NAME,
    JSON_NAME,
    LATEX_NAME,
    METHODS,
    analyze_conditions,
    holm_adjust,
    load_condition,
    main,
    paired_cluster_bootstrap_aurc_test,
)


def _rows(count=6):
    rows = []
    for index in range(count):
        risk_dice = (index + 1) / (count + 2)
        risk_hd95 = ((3 * index) % (count + 1) + 0.5) / (count + 2)
        row = {
            "schema_version": 1,
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
            "risk_nhd95": risk_hd95,
            "risk_hd95_pixels": 40 * risk_hd95,
            "confidence_sdc": 1 - 0.8 * risk_dice,
            "confidence_mean_max_probability": 0.95 - 0.4 * risk_dice,
            "confidence_negative_entropy": -0.5 * risk_dice,
            "confidence_dice_m2": -0.9 * risk_dice,
            "confidence_dice_m8": -0.8 * risk_dice,
            "confidence_dice_m32": -risk_dice,
            "confidence_nhd95_m2": -0.9 * risk_hd95,
            "confidence_nhd95_m8": -0.8 * risk_hd95,
            "confidence_nhd95_m32": -risk_hd95,
        }
        rows.append(row)
    return rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_condition(
    directory,
    *,
    condition="model-a",
    dataset="dataset-a",
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
        "schema_version": 1,
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
        "risk_fields": ["risk_dice", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd95_pixels"],
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


def test_load_condition_accepts_a_complete_hashed_schema(tmp_path):
    jsonl, manifest = _write_condition(tmp_path)
    loaded = load_condition(jsonl)
    assert loaded.jsonl_path == jsonl
    assert loaded.manifest_path == manifest
    assert loaded.condition == "model-a"
    assert loaded.dataset == "dataset-a"
    assert len(loaded.rows) == 6


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
            "risk_fields": ["risk_dice", "risk_nhd95", "risk_unregistered"]
        },
    )
    with pytest.raises(ValueError, match="exactly the two main risks"):
        load_condition(unexpected_risk)


def test_analyze_conditions_rejects_duplicate_dataset_condition_pairs(tmp_path):
    first, _ = _write_condition(tmp_path / "one")
    second, _ = _write_condition(tmp_path / "two")
    with pytest.raises(ValueError, match="exactly once"):
        analyze_conditions(
            [load_condition(first), load_condition(second)],
            bootstrap_samples=5,
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
    ]
    monkeypatch.setattr(sys, "argv", [*base_args, "--output-dir", str(output_one)])
    main()
    monkeypatch.setattr(sys, "argv", [*base_args, "--output-dir", str(output_two)])
    main()

    for filename in (JSON_NAME, CSV_NAME, LATEX_NAME):
        assert (output_one / filename).read_bytes() == (output_two / filename).read_bytes()

    summary_text = (output_one / JSON_NAME).read_text()
    assert ": NaN" not in summary_text
    summary = json.loads(summary_text)
    assert len(summary["conditions"]) == 1
    condition = summary["conditions"][0]
    assert set(condition["risks"]) == {"risk_dice", "risk_nhd95"}
    assert len(condition["risks"]["risk_dice"]["methods"]) == len(METHODS)
    assert summary["multiple_testing"]["num_hypotheses"] == 2
    assert "raw_bootstrap_p_values" in summary["multiple_testing"]
    assert "randomization" not in summary_text.lower()
    assert summary["multiple_testing"]["significance_calls"] == (
        "not made by this analysis"
    )
    for comparison in condition["comparisons"].values():
        assert 0 <= comparison["bootstrap"]["p_value"] <= 1
        assert 0 <= comparison["holm_adjusted_p_value"] <= 1
        assert "significant" not in comparison

    csv_lines = (output_one / CSV_NAME).read_text().splitlines()
    assert len(csv_lines) == 1 + 2 * len(METHODS)
    assert "normalized_aurc" in csv_lines[0]
    latex = (output_one / LATEX_NAME).read_text()
    assert r"\begin{table*}" in latex
    assert "Dice-M32" in latex
    assert "nHD95-M32" in latex


def test_cli_rejects_removed_randomization_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_binary.py", "--randomization-samples", "10"],
    )
    with pytest.raises(SystemExit):
        main()
