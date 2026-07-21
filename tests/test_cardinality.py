"""Tests for the isolated exact shared-threshold cardinality diagnostics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze.main import EXPECTED_CONDITIONS
from scripts.analyze.cardinality import (
    ARTIFACT_TYPE as ANALYSIS_ARTIFACT_TYPE,
    SCHEMA_VERSION as ANALYSIS_SCHEMA_VERSION,
    SCOPE as ANALYSIS_SCOPE,
    TARGET_CONDITIONS,
    build_analysis,
    summarize_condition,
)
from scripts.render.cardinality import (
    render_analysis,
    validate_analysis,
)
from scripts.submit.main import write_campaign_lock
from scripts.submit.cardinality import plan_cardinality_jobs
from selectseg.artifacts import sha256_file, write_binary_artifact
from selectseg.studies.cardinality import (
    EXPECTED_AUXILIARY_ID,
    EXPECTED_DIAGNOSTIC_SEED,
    PROTOCOL,
    cardinality_cdf_bounds,
    deterministic_pit_randomizer,
    load_cardinality_diagnostic,
    parse_args,
    run_cardinality_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
MIDPOINT_SPEC = ROOT / "configs" / "estimators" / "midpoint-v1.json"


def _artifact(tmp_path):
    samples = (
        (
            "case-a",
            np.array([[0.8, 0.8], [0.3, 0.0]], dtype=np.float32),
            np.array([[1, 0], [1, 0]], dtype=np.uint8),
        ),
        (
            "case-b",
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.uint8),
        ),
    )
    return write_binary_artifact(
        tmp_path / "frozen",
        dataset="pet",
        condition="clipseg-general",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="foreground",
        checkpoint=None,
        base_model={"name": "clipseg", "source": "unit-test"},
        source_sha256="a" * 64,
        environment={
            "packages": {
                "python": "3.12",
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
            "model_input": "none",
            "probability_to_native_mask": "none",
        },
        cohort="two exact-cardinality unit-test masks",
        sample_ids=[row[0] for row in samples],
        samples=samples,
        command=["pytest", "freeze"],
        created_utc="2026-07-20T00:00:00+00:00",
    )


def _auxiliary_lock(tmp_path, artifact):
    frozen = json.loads(artifact.read_text())
    campaign = {
        "lock_schema_version": 1,
        "campaign_id": "unit-cardinality-campaign",
        "config": {"path": "unit-cardinality.json", "sha256": "d" * 64},
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "estimator": {
            "spec_path": str(MIDPOINT_SPEC),
            "spec_sha256": sha256_file(MIDPOINT_SPEC),
            "estimator_id": "midpoint-v1",
            "target_measure": "uniform-threshold",
        },
        "paths": {
            "artifact_output_root": str(tmp_path / "unused-artifacts"),
            "common_output_root": str(tmp_path / "unused-common"),
            "simulation_output_root": str(tmp_path / "unused-simulations"),
            "assembly_output_root": str(tmp_path / "unused-assembled"),
        },
        "artifacts": [
            {
                "manifest_path": str(artifact),
                "manifest_sha256": sha256_file(artifact),
                "artifact_id": frozen["artifact_id"],
                "dataset": frozen["dataset"],
                "condition": frozen["condition"],
                "model": frozen["model"],
                "split": frozen["split"],
                "checkpoint_sha256": None,
                "source_sha256": frozen["source_sha256"],
                "sample_id_sha256": frozen["sample_id_sha256"],
                "num_samples": frozen["num_samples"],
            }
        ],
    }
    campaign_path, campaign_sha256 = write_campaign_lock(
        campaign, tmp_path / "campaign.lock.json"
    )
    spec = {
        "spec_schema_version": 1,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "canonical_campaign_lock": {
            "path": str(campaign_path),
            "sha256": campaign_sha256,
            "campaign_id": campaign["campaign_id"],
        },
        "protocol": PROTOCOL,
        "cpu_partitions": ["agsmall", "amdsmall", "msismall"],
        "output_root": str(tmp_path / "cardinality-output"),
    }
    spec_path = tmp_path / "cardinality-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    lock = {
        "lock_schema_version": 1,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "canonical_campaign_lock": spec["canonical_campaign_lock"],
        "protocol": PROTOCOL,
        "cpu_partitions": spec["cpu_partitions"],
        "output_root": spec["output_root"],
        "artifacts": [
            {
                "manifest_path": str(artifact),
                "manifest_sha256": sha256_file(artifact),
            }
        ],
    }
    lock_path = tmp_path / "cardinality.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock_path


def _run(tmp_path):
    artifact = _artifact(tmp_path)
    lock_path = _auxiliary_lock(tmp_path, artifact)
    arguments = [
        "--auxiliary-lock",
        str(lock_path),
        "--expected-auxiliary-lock-sha256",
        sha256_file(lock_path),
        "--artifact-manifest",
        str(artifact),
        "--expected-artifact-manifest-sha256",
        sha256_file(artifact),
    ]
    return artifact, lock_path, arguments, run_cardinality_diagnostics(
        parse_args(arguments), created_utc="2026-07-20T01:00:00+00:00"
    )


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (0, (0.0, 0.2)),
        (1, (0.2, 0.2)),
        (2, (0.2, 0.7)),
        (3, (0.7, 1.0)),
    ],
)
def test_exact_cardinality_cdf_handles_ties_and_edges(k, expected):
    probability = np.array([0.8, 0.8, 0.3], dtype=np.float64)
    assert cardinality_cdf_bounds(probability, k) == pytest.approx(expected)


def test_cdf_point_masses_sum_to_one_and_match_layer_cake_identity():
    probability = np.array([0.9, 0.6, 0.2], dtype=np.float32)
    bounds = [cardinality_cdf_bounds(probability, k) for k in range(4)]
    masses = np.asarray([upper - lower for lower, upper in bounds])
    assert masses == pytest.approx([0.1, 0.3, 0.4, 0.2])
    assert np.sum(masses) == pytest.approx(1.0)
    assert np.dot(np.arange(4), masses) == pytest.approx(np.sum(probability))
    assert masses[0] == pytest.approx(1.0 - np.max(probability))
    assert cardinality_cdf_bounds(np.zeros(3), 0) == (0.0, 1.0)
    assert cardinality_cdf_bounds(np.ones(3), 3) == (0.0, 1.0)


def test_partition_formula_matches_complete_sort_reference():
    rng = np.random.default_rng(17)
    probability = rng.choice([0.0, 0.1, 0.4, 0.4, 0.9, 1.0], size=101)
    descending = np.sort(probability)[::-1]
    for k in range(probability.size + 1):
        expected_lower = 0.0 if k == 0 else 1.0 - descending[k - 1]
        expected_upper = 1.0 if k == probability.size else 1.0 - descending[k]
        assert cardinality_cdf_bounds(probability, k) == pytest.approx(
            (expected_lower, expected_upper)
        )


def test_randomized_pit_hash_is_stable_paired_and_strictly_inside_unit_interval():
    value = deterministic_pit_randomizer(EXPECTED_DIAGNOSTIC_SEED, "病例-17")
    assert value.hex() == "0x1.de37d25edc0fdp-1"
    assert 0.0 < value < 1.0
    assert value == deterministic_pit_randomizer(
        EXPECTED_DIAGNOSTIC_SEED, "病例-17"
    )
    assert value != deterministic_pit_randomizer(
        EXPECTED_DIAGNOSTIC_SEED + 1, "病例-17"
    )
    assert value != deterministic_pit_randomizer(
        EXPECTED_DIAGNOSTIC_SEED, "病例-18"
    )


def test_independent_artifact_is_exact_append_only_and_fail_closed(tmp_path):
    _, _, arguments, (records_path, manifest_path) = _run(tmp_path)
    loaded = load_cardinality_diagnostic(records_path)
    assert loaded.manifest["scope"]["not_posterior_calibration"].startswith(
        "one annotation"
    )
    assert loaded.manifest["protocol"] == PROTOCOL
    assert loaded.manifest["records_sha256"] == sha256_file(records_path)
    assert len(loaded.records) == 2
    first, second = loaded.records
    assert first["observed_cardinality_probability"] == pytest.approx(0.5)
    assert first["randomized_cardinality_pit"] == pytest.approx(
        first["cardinality_cdf_lower"]
        + first["pit_randomizer"] * first["observed_cardinality_probability"]
    )
    assert second["working_posterior_empty_probability"] == 1.0
    assert second["observed_empty_mask"] is True
    with pytest.raises(FileExistsError, match="already exist"):
        run_cardinality_diagnostics(parse_args(arguments))

    original = records_path.read_bytes()
    records_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_cardinality_diagnostic(records_path)
    assert manifest_path.is_file()


def test_incomplete_analysis_planner_and_condition_summary(tmp_path):
    _, lock_path, _, (records_path, _) = _run(tmp_path)
    loaded = load_cardinality_diagnostic(records_path)
    summary = summarize_condition(loaded)
    assert summary["foreground_fraction_error"]["mean_absolute_error"] >= 0
    assert summary["foreground_fraction_reliability"]["num_bins"] == 2
    assert "not a pointwise posterior-calibration estimate" in summary[
        "randomized_cardinality_pit"
    ]["interpretation"]

    report = build_analysis(lock_path, [records_path], allow_incomplete=True)
    assert report["condition_sets"]["num_conditions"] == 1
    with pytest.raises(ValueError, match="exactly 16"):
        build_analysis(lock_path, [records_path])

    jobs = plan_cardinality_jobs(lock_path)
    assert len(jobs) == 1
    assert jobs[0].phase == "cardinality_diagnostics"
    assert "--partition" in jobs[0].command
    assert "scripts/slurm/run.sbatch" in jobs[0].command


def _complete_renderer_fixture():
    conditions = []
    for dataset, condition in EXPECTED_CONDITIONS:
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "is_target_condition": (dataset, condition) in TARGET_CONDITIONS,
                "foreground_fraction_error": {
                    "signed_bias_predicted_minus_observed": 0.01,
                    "mean_absolute_error": 0.02,
                },
                "empty_mask_identity": {
                    "signed_bias_predicted_minus_observed": -0.01,
                    "mean_absolute_error": 0.03,
                },
                "randomized_cardinality_pit": {
                    "diagnostic_seed": EXPECTED_DIAGNOSTIC_SEED,
                    "kolmogorov_smirnov_distance_to_uniform": 0.1,
                    "outside_central_90_percent_ratio": 0.12,
                    "zero_observed_point_mass_ratio": 0.0,
                    "interpretation": (
                        "pooled label proxy; not a pointwise "
                        "posterior-calibration estimate"
                    ),
                },
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_ARTIFACT_TYPE,
        "analysis_id": "a" * 16,
        "scope": ANALYSIS_SCOPE,
        "protocol": PROTOCOL,
        "condition_sets": {
            "complete": True,
            "num_conditions": 16,
            "num_target_conditions": 10,
            "conditions": [f"{a}/{b}" for a, b in EXPECTED_CONDITIONS],
            "target_conditions": [
                f"{a}/{b}"
                for a, b in EXPECTED_CONDITIONS
                if (a, b) in TARGET_CONDITIONS
            ],
        },
        "provenance": {},
        "conditions": conditions,
    }


def test_renderer_uses_dataset_columns_and_explicit_support_language():
    fixture = _complete_renderer_fixture()
    by_key = validate_analysis(fixture)
    assert len(by_key) == 16
    tex = render_analysis(fixture, source_sha256="f" * 64)
    assert "Oxford Pet & Kvasir-SEG & FIVES & ISIC 2018 & TN3K" in tex
    assert "\\textsc{CLIP-T} / \\textsc{DL-T}" in tex
    assert "Favorable values do not establish pointwise" in tex
    assert "posterior calibration or validate" in tex
    assert "Foreground-mass bias" in tex
    assert r"\mathbb{E}_{Q_p}[|Y|/|\Omega|]=|\Omega|^{-1}\sum_i p_i" in tex
    assert "direct support incompatibility" in tex
    assert tex.count(r"\resizebox{\textwidth}{!}{%") == 1

    broken = copy.deepcopy(fixture)
    broken["scope"]["interpretation_limit"] = "posterior calibrated"
    with pytest.raises(ValueError, match="scope/protocol"):
        validate_analysis(broken)
