"""Tests for the separate, content-addressed M=128 reference workflow."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.submit_binary_simulations import (
    build_campaign_lock,
    load_config,
    write_campaign_lock,
)
from scripts.submit_m128_auxiliary import CPU_PARTITIONS, plan_m128_jobs
from selectseg.binary_artifacts import (
    load_binary_artifact,
    write_binary_artifact,
)
from selectseg.binary_boundary import prepare_boundary_reference
from selectseg.score_binary_m128_auxiliary import (
    AUXILIARY_ARTIFACT_TYPE,
    IDENTITY_FIELDS,
    M128_SCORE_FIELDS,
    M32_DIAGNOSTIC_FIELDS,
    parse_args,
    run_auxiliary,
)
from selectseg.score_binary_simulation import score_binary_sample
from selectseg.threshold_estimators import (
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MIDPOINT_SPEC = ROOT / "configs" / "estimators" / "midpoint-v1.json"


def _write_frozen_artifact(tmp_path, *, dataset="toy0"):
    probability_a = np.full((12, 12), 0.2, dtype=np.float32)
    probability_a[3:9, 4:8] = np.float32(0.8)
    truth_a = (probability_a >= 0.5).astype(np.uint8)
    horizontal = np.linspace(0.01, 0.99, 12, dtype=np.float32)
    probability_b = np.broadcast_to(horizontal, (12, 12)).copy()
    truth_b = (probability_b >= 0.6).astype(np.uint8)
    return write_binary_artifact(
        tmp_path / "artifacts",
        dataset=dataset,
        condition="clipseg-general",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="lesion",
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
        cohort="two deterministic masks",
        sample_ids=["sample-a", "sample-b"],
        samples=(
            ("sample-a", probability_a, truth_a),
            ("sample-b", probability_b, truth_b),
        ),
        command=["pytest", "freeze"],
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _locked_campaign(tmp_path, *, condition_count=1):
    artifacts = [
        _write_frozen_artifact(tmp_path / f"artifact-{index}", dataset=f"toy{index}")
        for index in range(condition_count)
    ]
    conditions = [
        {
            "dataset": f"toy{index}",
            "condition": "clipseg-general",
            "model": "clipseg",
            "checkpoint": None,
            "batch_size": 1,
            "expected_num_samples": 2,
        }
        for index in range(condition_count)
    ]
    config_payload = {
        "config_schema_version": 1,
        "campaign_id": "unit-m128-campaign",
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "cpu_partitions": list(CPU_PARTITIONS),
        "estimator_spec": str(MIDPOINT_SPEC),
        "paths": {
            "artifact_output_root": str(tmp_path / "artifacts-unused"),
            "common_output_root": str(tmp_path / "common"),
            "simulation_output_root": str(tmp_path / "simulations"),
            "assembly_output_root": str(tmp_path / "assembled"),
        },
        "conditions": conditions,
    }
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n")
    config = load_config(config_path)
    lock = build_campaign_lock(config, artifacts)
    lock_path, lock_sha256 = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    return artifacts, lock_path, lock_sha256


def _arguments(
    tmp_path,
    artifact_manifest,
    campaign_lock,
    campaign_sha256,
    *,
    diagnostics=True,
):
    arguments = [
        "--campaign-id",
        "unit-m128-campaign",
        "--campaign-lock",
        str(campaign_lock),
        "--expected-campaign-lock-sha256",
        campaign_sha256,
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
        "--output-root",
        str(tmp_path / "m128"),
        "--score-workers",
        "1",
        "--max-pending-scores",
        "1",
    ]
    if diagnostics:
        arguments.append("--include-m32-diagnostics")
    return arguments


def _direct_scores(sample, count):
    spec = load_estimator_spec(MIDPOINT_SPEC)
    rule = build_threshold_rule(spec, m=count, seed=0)
    probability = sample.foreground_probability.astype(float)
    prediction = probability >= 0.5
    prediction_size = int(prediction.sum())
    reference = prepare_boundary_reference(prediction)
    values = []
    for node in rule.nodes:
        level = probability >= node
        denominator = int(level.sum()) + prediction_size
        dice = (
            0.0
            if denominator == 0
            else 1 - (2 * int(np.logical_and(level, prediction).sum()) / denominator)
        )
        boundary = reference.compare(level)
        values.append((dice, boundary.nhd, boundary.nhd95))
    array = np.asarray(values)
    return tuple(-float(np.dot(rule.weights, array[:, index])) for index in range(3))


def test_m128_auxiliary_is_exact_append_only_and_separate_from_schema_v2(tmp_path):
    artifacts, lock_path, lock_sha256 = _locked_campaign(tmp_path)
    arguments = _arguments(
        tmp_path, artifacts[0], lock_path, lock_sha256, diagnostics=True
    )
    records_path, manifest_path = run_auxiliary(parse_args(arguments))
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]

    assert manifest["artifact_type"] == AUXILIARY_ARTIFACT_TYPE
    assert manifest["schema_version"] == 1
    assert manifest["canonical_schema_v2_compatible"] is False
    assert manifest["num_rows"] == manifest["num_images"] == len(rows) == 2
    assert manifest["jsonl_sha256"] == sha256_file(records_path)
    assert manifest["score_fields"] == list(M128_SCORE_FIELDS)
    assert manifest["diagnostic_fields"] == list(M32_DIAGNOSTIC_FIELDS)
    assert set(manifest["quadrature"]) == {"32", "128"}
    assert len(manifest["quadrature"]["128"]["nodes"]) == 128
    assert manifest["provenance"]["campaign_lock_sha256"] == lock_sha256
    assert manifest["provenance"]["artifact_manifest_sha256"] == sha256_file(
        artifacts[0]
    )
    assert manifest["provenance"]["include_m32_diagnostics"] is True

    frozen_samples = tuple(load_binary_artifact(artifacts[0]).iter_samples())
    for row, sample in zip(rows, frozen_samples, strict=True):
        assert set(row) == (
            set(IDENTITY_FIELDS) | set(M128_SCORE_FIELDS) | set(M32_DIAGNOSTIC_FIELDS)
        )
        assert row["schema_version"] == 1
        assert row["run_id"] == manifest["run_id"]
        m128 = _direct_scores(sample, 128)
        m32 = _direct_scores(sample, 32)
        canonical_m32 = score_binary_sample(
            sample,
            simulation_id="canonical-comparison",
            class_index=1,
            class_name="lesion",
            gamma=0.5,
            threshold_rule=build_threshold_rule(
                load_estimator_spec(MIDPOINT_SPEC), m=32, seed=0
            ),
        )
        np.testing.assert_allclose(
            [row[field] for field in M128_SCORE_FIELDS], m128, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            [row[field] for field in M32_DIAGNOSTIC_FIELDS[:3]],
            m32,
            rtol=0,
            atol=1e-15,
        )
        assert [row[field] for field in M32_DIAGNOSTIC_FIELDS[:3]] == [
            canonical_m32["confidence_dice_m32"],
            canonical_m32["confidence_nhd_m32"],
            canonical_m32["confidence_nhd95_m32"],
        ]
        np.testing.assert_allclose(
            [row[field] for field in M32_DIAGNOSTIC_FIELDS[3:]],
            np.subtract(m128, m32),
            rtol=0,
            atol=1e-15,
        )

    with pytest.raises(FileExistsError, match="already exists"):
        run_auxiliary(parse_args(arguments))


def test_diagnostics_flag_changes_content_identity_and_row_schema(tmp_path):
    artifacts, lock_path, lock_sha256 = _locked_campaign(tmp_path)
    with_diagnostics = run_auxiliary(
        parse_args(
            _arguments(tmp_path, artifacts[0], lock_path, lock_sha256, diagnostics=True)
        )
    )[1]
    without_diagnostics = run_auxiliary(
        parse_args(
            _arguments(
                tmp_path, artifacts[0], lock_path, lock_sha256, diagnostics=False
            )
        )
    )[1]
    assert with_diagnostics.parent != without_diagnostics.parent
    manifest = json.loads(without_diagnostics.read_text())
    row = json.loads(
        (without_diagnostics.parent / "records.jsonl").read_text().splitlines()[0]
    )
    assert manifest["diagnostic_fields"] == []
    assert set(manifest["quadrature"]) == {"128"}
    assert set(row) == set(IDENTITY_FIELDS) | set(M128_SCORE_FIELDS)


def test_auxiliary_fails_closed_on_hashes_and_lock_membership(tmp_path):
    artifacts, lock_path, lock_sha256 = _locked_campaign(tmp_path)
    bad_campaign = _arguments(
        tmp_path, artifacts[0], lock_path, lock_sha256, diagnostics=False
    )
    bad_campaign[bad_campaign.index("--expected-campaign-lock-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError, match="campaign lock SHA-256 mismatch"):
        run_auxiliary(parse_args(bad_campaign))

    bad_artifact = _arguments(
        tmp_path, artifacts[0], lock_path, lock_sha256, diagnostics=False
    )
    bad_artifact[bad_artifact.index("--expected-artifact-manifest-sha256") + 1] = (
        "e" * 64
    )
    with pytest.raises(ValueError, match="artifact manifest SHA-256 mismatch"):
        run_auxiliary(parse_args(bad_artifact))

    foreign = _write_frozen_artifact(tmp_path / "foreign", dataset="foreign")
    foreign_arguments = _arguments(
        tmp_path, foreign, lock_path, lock_sha256, diagnostics=False
    )
    with pytest.raises(ValueError, match="occur exactly once"):
        run_auxiliary(parse_args(foreign_arguments))


def test_planner_is_one_cpu_job_per_condition_and_cycles_allowed_partitions(
    tmp_path, capsys
):
    _, lock_path, _ = _locked_campaign(tmp_path, condition_count=4)
    jobs = plan_m128_jobs(
        lock_path,
        output_root="outputs/test-m128",
        include_m32_diagnostics=True,
    )
    assert len(jobs) == 4
    assert [job.key[2] for job in jobs] == [
        "agsmall",
        "amdsmall",
        "msismall",
        "agsmall",
    ]
    assert len({job.key for job in jobs}) == 4
    for job in jobs:
        command = list(job.command)
        assert job.phase == "m128_auxiliary"
        assert command.count("scripts/slurm/score_binary_m128_auxiliary.sbatch") == 1
        assert command.count("--artifact-manifest") == 1
        assert command.count("--expected-artifact-manifest-sha256") == 1
        assert command.count("--expected-campaign-lock-sha256") == 1
        assert command.count("--include-m32-diagnostics") == 1
        assert command[command.index("--partition") + 1] in CPU_PARTITIONS
        assert command[command.index("--account") + 1] == "ssafo"
        assert "--array" not in command
        assert "--gres" not in command
        assert "--gpus" not in command

    wrapper = (ROOT / "scripts/slurm/score_binary_m128_auxiliary.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem=24g" in wrapper
    assert "#SBATCH --time=12:00:00" in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert "#SBATCH --gpus" not in wrapper
    assert "python -m selectseg.score_binary_m128_auxiliary" in wrapper
    assert capsys.readouterr().out == ""


def test_hash_helper_matches_raw_bytes(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"m128-auxiliary\n")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
