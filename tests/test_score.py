import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import selectseg.pipeline.score as scorer_module
import selectseg.pipeline.common as common_module

from selectseg.artifacts import load_binary_artifact, write_binary_artifact
from selectseg.geometry import prepare_boundary_reference
from selectseg.evaluate import binary_record
from selectseg.confidence import midpoint_rule
from selectseg.pipeline.common import (
    AUXILIARY_FIELDS,
    BASE_ROW_FIELDS,
    COMMON_SCORE_FIELDS,
    IDENTITY_ROW_FIELDS,
    RISK_FIELDS,
    parse_args as parse_common_args,
    run_common,
)
from selectseg.pipeline.score import (
    parse_args,
    run_simulation,
)
from selectseg.quadrature import (
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MIDPOINT_SPEC = ROOT / "configs" / "estimators" / "midpoint-v1.json"


def _frozen_artifact(tmp_path):
    probability_a = np.full((12, 12), 0.2, dtype=np.float32)
    probability_a[3:9, 4:8] = np.float32(0.8)
    truth_a = (probability_a >= 0.5).astype(np.uint8)

    horizontal = np.linspace(0.05, 0.95, 12, dtype=np.float32)
    probability_b = np.broadcast_to(horizontal, (12, 12)).copy()
    truth_b = np.zeros((12, 12), dtype=np.uint8)
    truth_b[2:10, 7:11] = 1
    samples = (
        ("sample-a", probability_a, truth_a),
        ("sample-b", probability_b, truth_b),
    )
    return write_binary_artifact(
        tmp_path / "artifacts",
        dataset="toy",
        condition="toy-condition",
        model="toy-model",
        split="test",
        class_index=1,
        class_name="lesion",
        checkpoint=None,
        base_model={"name": "toy-model", "source": "unit-test"},
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
        cohort="two synthetic native binary masks",
        sample_ids=["sample-a", "sample-b"],
        samples=samples,
        command=["pytest", "freeze"],
        created_utc="2026-01-01T00:00:00+00:00",
    )


def _arguments(tmp_path, artifact_manifest, campaign_lock, *, m=2, seed=0):
    return [
        "--campaign-id",
        "unit-campaign",
        "--campaign-lock",
        str(campaign_lock),
        "--expected-campaign-lock-sha256",
        sha256_file(campaign_lock),
        "--artifact-manifest",
        str(artifact_manifest),
        "--expected-artifact-manifest-sha256",
        sha256_file(artifact_manifest),
        "--estimator-spec",
        str(MIDPOINT_SPEC),
        "--expected-estimator-spec-sha256",
        sha256_file(MIDPOINT_SPEC),
        "--gamma",
        "0.5",
        "--m",
        str(m),
        "--seed",
        str(seed),
        "--output-root",
        str(tmp_path / "simulations"),
        "--score-workers",
        "1",
        "--max-pending-scores",
        "1",
    ]


def _common_arguments(tmp_path, artifact_manifest, campaign_lock):
    return [
        "--campaign-id",
        "unit-campaign",
        "--campaign-lock",
        str(campaign_lock),
        "--expected-campaign-lock-sha256",
        sha256_file(campaign_lock),
        "--artifact-manifest",
        str(artifact_manifest),
        "--expected-artifact-manifest-sha256",
        sha256_file(artifact_manifest),
        "--gamma",
        "0.5",
        "--output-root",
        str(tmp_path / "common"),
        "--score-workers",
        "1",
        "--max-pending-scores",
        "1",
    ]


def test_midpoint_v1_is_deterministic_and_seed_locked():
    spec = load_estimator_spec(MIDPOINT_SPEC)
    assert spec.estimator_id == "midpoint-v1"
    assert spec.sha256 == sha256_file(MIDPOINT_SPEC)
    rule = build_threshold_rule(spec, m=4, seed=0)
    np.testing.assert_array_equal(rule.nodes, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_array_equal(rule.weights, [0.25] * 4)
    assert not rule.nodes.flags.writeable
    assert not rule.weights.flags.writeable
    with pytest.raises(ValueError, match="requires seed=0"):
        build_threshold_rule(spec, m=4, seed=1)


@pytest.mark.parametrize("axis", ["--gamma", "--m", "--seed"])
def test_cli_rejects_a_repeated_simulation_axis(axis):
    values = {"--gamma": "0.5", "--m": "2", "--seed": "0"}
    arguments = [
        "--campaign-id",
        "campaign",
        "--campaign-lock",
        "lock.json",
        "--expected-campaign-lock-sha256",
        "0" * 64,
        "--artifact-manifest",
        "manifest.json",
        "--expected-artifact-manifest-sha256",
        "1" * 64,
        "--estimator-spec",
        "midpoint.json",
        "--expected-estimator-spec-sha256",
        "2" * 64,
    ]
    for option in ("--gamma", "--m", "--seed"):
        arguments.extend([option, values[option]])
        if option == axis:
            arguments.extend([option, values[option]])
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_one_simulation_writes_strict_partial_rows_and_hashes(tmp_path):
    artifact_manifest = _frozen_artifact(tmp_path)
    campaign_lock = tmp_path / "campaign-lock.json"
    campaign_lock.write_text('{"campaign_id":"unit-campaign"}\n', encoding="utf-8")
    arguments = _arguments(tmp_path, artifact_manifest, campaign_lock, m=2)

    records_path, manifest_path = run_simulation(parse_args(arguments))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["artifact_type"] == "selectseg.binary_simulation_partial"
    assert manifest["num_images"] == manifest["num_rows"] == len(rows) == 2
    assert manifest["jsonl_sha256"] == sha256_file(records_path)
    assert manifest["score_fields"] == [
        "confidence_dice_m2",
        "confidence_nhd_m2",
        "confidence_nhd95_m2",
    ]
    assert manifest["risk_fields"] == []
    assert manifest["auxiliary_fields"] == []
    assert set(manifest["quadrature"]) == {"2"}
    simulation = manifest["simulation"]
    assert simulation["campaign_id"] == "unit-campaign"
    assert simulation["campaign_lock_path"] == campaign_lock.resolve().as_posix()
    assert (
        simulation["artifact_manifest_path"] == artifact_manifest.resolve().as_posix()
    )
    assert simulation["estimator_spec_path"] == ("configs/estimators/midpoint-v1.json")
    assert simulation["campaign_lock_sha256"] == sha256_file(campaign_lock)
    assert simulation["artifact_manifest_sha256"] == sha256_file(artifact_manifest)
    assert simulation["estimator_spec_sha256"] == sha256_file(MIDPOINT_SPEC)
    assert simulation["artifact_source_sha256"] == "a" * 64
    assert simulation["m"] == 2
    assert simulation["gamma"] == 0.5
    assert simulation["seed"] == 0
    assert len(manifest["source_sha256"]) == 64
    assert manifest["run_id"] == manifest["simulation_id"]

    expected_fields = {
        "schema_version",
        "run_id",
        "sample_id",
        "image_id",
        "image_index",
        "class_index",
        "class_name",
        "height",
        "width",
        "confidence_dice_m2",
        "confidence_nhd_m2",
        "confidence_nhd95_m2",
    }
    for index, row in enumerate(rows):
        assert set(row) == expected_fields
        assert row["run_id"] == manifest["simulation_id"]
        assert row["image_index"] == index
        assert row["sample_id"] == row["image_id"]
        assert row["class_index"] == 1
        assert row["class_name"] == "lesion"
        assert all(np.isfinite(row[field]) for field in manifest["score_fields"])
        assert "confidence_dice_m8" not in row
        assert "confidence_nhd95_m32" not in row
    frozen_samples = tuple(load_binary_artifact(artifact_manifest).iter_samples())
    for row, sample in zip(rows, frozen_samples, strict=True):
        expected = binary_record(
            sample.foreground_probability,
            sample.truth,
            run_id=manifest["simulation_id"],
            image_id=sample.sample_id,
            image_index=sample.index,
            class_index=1,
            class_name="lesion",
            decision_threshold=0.5,
            quadrature_rules={2: midpoint_rule(2)},
        )
        probability = sample.foreground_probability.astype(float)
        prediction = probability >= 0.5
        nodes, weights = midpoint_rule(2)
        boundary = prepare_boundary_reference(prediction)
        expected_nhd = -float(
            np.dot(
                weights, [boundary.compare(probability >= node).nhd for node in nodes]
            )
        )
        for field, value in row.items():
            if field == "schema_version":
                assert value == 2
            elif field == "confidence_nhd_m2":
                assert value == expected_nhd
            else:
                assert value == expected[field]
    assert (
        hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode("utf-8")
        ).hexdigest()
        == manifest["sample_id_sha256"]
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_simulation(parse_args(arguments))


def test_common_job_is_unique_source_of_risks_baselines_and_float_base_fields(
    tmp_path,
):
    artifact_manifest = _frozen_artifact(tmp_path)
    campaign_lock = tmp_path / "campaign-lock.json"
    campaign_lock.write_text('{"campaign_id":"unit-campaign"}\n', encoding="utf-8")
    records_path, manifest_path = run_common(
        parse_common_args(_common_arguments(tmp_path, artifact_manifest, campaign_lock))
    )
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert manifest["artifact_type"] == "selectseg.binary_common_scores"
    assert manifest["score_fields"] == list(COMMON_SCORE_FIELDS)
    assert manifest["risk_fields"] == list(RISK_FIELDS)
    assert manifest["auxiliary_fields"] == list(AUXILIARY_FIELDS)
    expected_fields = (
        set(BASE_ROW_FIELDS)
        | set(RISK_FIELDS)
        | set(AUXILIARY_FIELDS)
        | set(COMMON_SCORE_FIELDS)
    )
    frozen_samples = tuple(load_binary_artifact(artifact_manifest).iter_samples())
    for row, sample in zip(rows, frozen_samples, strict=True):
        assert set(row) == expected_fields
        evaluator = binary_record(
            sample.foreground_probability,
            sample.truth,
            run_id=manifest["run_id"],
            image_id=sample.sample_id,
            image_index=sample.index,
            class_index=1,
            class_name="lesion",
            decision_threshold=0.5,
            quadrature_rules={2: midpoint_rule(2)},
        )
        probability = sample.foreground_probability.astype(float)
        truth = sample.truth.astype(bool)
        prediction = probability >= 0.5
        boundary = prepare_boundary_reference(truth).compare(prediction)
        for field, value in row.items():
            if field == "schema_version":
                assert value == 2
            elif field == "risk_nhd":
                assert value == boundary.nhd
            elif field == "risk_hd_pixels":
                assert value == boundary.nhd * np.hypot(*probability.shape)
            else:
                assert value == evaluator[field]
    # The simulation contract contains only exact identity fields plus M scores.
    assert set(IDENTITY_ROW_FIELDS).isdisjoint(
        set(RISK_FIELDS) | set(COMMON_SCORE_FIELDS)
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_common(
            parse_common_args(
                _common_arguments(tmp_path, artifact_manifest, campaign_lock)
            )
        )


def test_hash_lock_and_midpoint_seed_fail_closed(tmp_path):
    artifact_manifest = _frozen_artifact(tmp_path)
    campaign_lock = tmp_path / "campaign-lock.json"
    campaign_lock.write_text("{}\n", encoding="utf-8")

    bad_hash_args = _arguments(tmp_path, artifact_manifest, campaign_lock)
    position = bad_hash_args.index("--expected-artifact-manifest-sha256") + 1
    bad_hash_args[position] = "f" * 64
    with pytest.raises(ValueError, match="artifact manifest SHA-256 mismatch"):
        run_simulation(parse_args(bad_hash_args))

    bad_seed_args = _arguments(tmp_path, artifact_manifest, campaign_lock, seed=9)
    with pytest.raises(ValueError, match="requires seed=0"):
        run_simulation(parse_args(bad_seed_args))


def test_failed_simulation_leaves_no_published_or_staging_output(tmp_path, monkeypatch):
    artifact_manifest = _frozen_artifact(tmp_path)
    campaign_lock = tmp_path / "campaign-lock.json"
    campaign_lock.write_text("{}\n", encoding="utf-8")

    def fail_score(*args, **kwargs):
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(scorer_module, "score_binary_sample", fail_score)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        run_simulation(
            parse_args(_arguments(tmp_path, artifact_manifest, campaign_lock))
        )
    condition_root = tmp_path / "simulations" / "toy" / "toy-condition"
    assert condition_root.is_dir()
    assert list(condition_root.iterdir()) == []

    monkeypatch.setattr(common_module, "score_binary_common_sample", fail_score)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        run_common(
            parse_common_args(
                _common_arguments(tmp_path, artifact_manifest, campaign_lock)
            )
        )
    common_root = tmp_path / "common" / "toy" / "toy-condition"
    assert common_root.is_dir()
    assert list(common_root.iterdir()) == []
