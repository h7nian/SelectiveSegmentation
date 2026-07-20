"""Tests for the locked one-(condition,gamma)-per-job sensitivity workflow."""

from __future__ import annotations

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
from scripts.submit_gamma_sensitivity import (
    DEFAULT_AUXILIARY_LOCK,
    parse_args as parse_submit_args,
    plan_gamma_sensitivity_jobs,
)
from selectseg.binary_artifacts import load_binary_artifact, write_binary_artifact
from selectseg.score_binary_common import score_binary_common_sample
from selectseg.score_binary_gamma_sensitivity import (
    AUXILIARY_ARTIFACT_TYPE,
    EXPECTED_CPU_PARTITIONS,
    M32_SCORE_FIELDS,
    OUTPUT_ROW_FIELDS,
    load_auxiliary_lock,
    parse_args,
    run_gamma_sensitivity,
)
from selectseg.score_binary_simulation import score_binary_sample
from selectseg.threshold_estimators import (
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MIDPOINT_SPEC = ROOT / "configs" / "estimators" / "midpoint-v1.json"


def _write_artifact(tmp_path, *, dataset):
    probability_a = np.full((12, 12), 0.2, dtype=np.float32)
    probability_a[2:9, 4:10] = np.float32(0.8)
    truth_a = (probability_a >= 0.6).astype(np.uint8)
    probability_b = np.broadcast_to(
        np.linspace(0.01, 0.99, 12, dtype=np.float32), (12, 12)
    ).copy()
    truth_b = np.zeros((12, 12), dtype=np.uint8)
    truth_b[3:10, 7:11] = 1
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


def _write_locked_workflow(tmp_path, *, condition_count=1):
    artifacts = [
        _write_artifact(tmp_path / f"condition-{index}", dataset=f"toy{index}")
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
        "campaign_id": "unit-gamma-main-v1",
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "cpu_partitions": list(EXPECTED_CPU_PARTITIONS),
        "estimator_spec": str(MIDPOINT_SPEC),
        "paths": {
            "artifact_output_root": str(tmp_path / "unused-artifacts"),
            "common_output_root": str(tmp_path / "common"),
            "simulation_output_root": str(tmp_path / "simulation"),
            "assembly_output_root": str(tmp_path / "assembled"),
        },
        "conditions": conditions,
    }
    config_path = tmp_path / "canonical.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n")
    canonical = build_campaign_lock(load_config(config_path), artifacts)
    campaign_path, campaign_sha256 = write_campaign_lock(
        canonical, tmp_path / "campaign.lock.json"
    )

    directory = tmp_path / "configs" / "auxiliary"
    directory.mkdir(parents=True)
    output_root = tmp_path / "gamma-output"
    common = {
        "auxiliary_id": "unit-gamma-sensitivity-v1",
        "canonical_campaign_lock": {
            "path": str(campaign_path),
            "sha256": campaign_sha256,
            "campaign_id": canonical["campaign_id"],
        },
        "protocol": {
            "gamma_values": [0.3, 0.7],
            "m": 32,
            "quadrature_rule": "midpoint-v1",
            "seed": 0,
        },
        "estimator_spec": {
            "path": str(MIDPOINT_SPEC),
            "sha256": sha256_file(MIDPOINT_SPEC),
        },
        "cpu_partitions": list(EXPECTED_CPU_PARTITIONS),
        "output_root": str(output_root),
    }
    spec = {"spec_schema_version": 1, **common}
    spec_path = directory / "gamma.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    lock = {
        "lock_schema_version": 1,
        **common,
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "artifacts": [
            {
                "manifest_path": entry["manifest_path"],
                "manifest_sha256": entry["manifest_sha256"],
            }
            for entry in canonical["artifacts"]
        ],
    }
    lock_path = directory / "gamma.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return artifacts, lock_path, sha256_file(lock_path), output_root


def _arguments(artifact, lock_path, lock_sha256, *, gamma=0.3):
    return [
        "--auxiliary-lock",
        str(lock_path),
        "--expected-auxiliary-lock-sha256",
        lock_sha256,
        "--artifact-manifest",
        str(artifact),
        "--expected-artifact-manifest-sha256",
        sha256_file(artifact),
        "--gamma",
        str(gamma),
        "--score-workers",
        "1",
        "--max-pending-scores",
        "1",
    ]


def test_repository_lock_expands_to_exactly_32_cpu_jobs():
    submit_args = parse_submit_args([])
    assert submit_args.submit is False
    assert submit_args.receipt is None
    jobs = plan_gamma_sensitivity_jobs(ROOT / DEFAULT_AUXILIARY_LOCK)
    assert len(jobs) == 32
    assert len({job.key for job in jobs}) == 32
    assert {job.key[2] for job in jobs} == {0.3, 0.7}
    assert [job.key[3] for job in jobs[:4]] == [
        "agsmall",
        "amdsmall",
        "msismall",
        "agsmall",
    ]
    for job in jobs:
        command = list(job.command)
        assert job.phase == "gamma_sensitivity"
        assert command.count("scripts/slurm/score_binary_gamma_sensitivity.sbatch") == 1
        assert command.count("--artifact-manifest") == 1
        assert command.count("--gamma") == 1
        assert command[command.index("--partition") + 1] in EXPECTED_CPU_PARTITIONS
        assert "--array" not in command
        assert "--gres" not in command
        assert "--gpus" not in command

    wrapper = (ROOT / "scripts/slurm/score_binary_gamma_sensitivity.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem=24g" in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert "#SBATCH --gpus" not in wrapper


def test_one_job_streams_and_publishes_combined_m32_rows(tmp_path):
    artifacts, lock_path, lock_sha256, _ = _write_locked_workflow(tmp_path)
    arguments = _arguments(artifacts[0], lock_path, lock_sha256, gamma=0.3)
    records_path, manifest_path = run_gamma_sensitivity(parse_args(arguments))
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]

    assert manifest["artifact_type"] == AUXILIARY_ARTIFACT_TYPE
    assert manifest["num_rows"] == manifest["num_images"] == len(rows) == 2
    assert manifest["provenance"]["artifact_passes"] == 1
    assert manifest["provenance"]["auxiliary_lock_sha256"] == lock_sha256
    assert manifest["decision_rule"]["gamma"] == 0.3
    assert set(manifest["quadrature"]) == {"32"}
    assert manifest["jsonl_sha256"] == sha256_file(records_path)

    samples = tuple(load_binary_artifact(artifacts[0]).iter_samples())
    rule = build_threshold_rule(load_estimator_spec(MIDPOINT_SPEC), m=32, seed=0)
    for row, sample in zip(rows, samples, strict=True):
        assert set(row) == set(OUTPUT_ROW_FIELDS)
        expected_common = score_binary_common_sample(
            sample,
            common_id=manifest["run_id"],
            class_index=1,
            class_name="lesion",
            gamma=0.3,
        )
        expected_m32 = score_binary_sample(
            sample,
            simulation_id=manifest["run_id"],
            class_index=1,
            class_name="lesion",
            gamma=0.3,
            threshold_rule=rule,
        )
        assert row == {
            **expected_common,
            **{field: expected_m32[field] for field in M32_SCORE_FIELDS},
        }

    with pytest.raises(FileExistsError, match="already exists"):
        run_gamma_sensitivity(parse_args(arguments))


def test_auxiliary_lock_and_axes_fail_closed(tmp_path):
    artifacts, lock_path, lock_sha256, _ = _write_locked_workflow(tmp_path)
    load_auxiliary_lock(lock_path, expected_sha256=lock_sha256)

    bad_lock = _arguments(artifacts[0], lock_path, lock_sha256, gamma=0.3)
    bad_lock[bad_lock.index("--expected-auxiliary-lock-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError, match="auxiliary lock SHA-256 mismatch"):
        run_gamma_sensitivity(parse_args(bad_lock))

    bad_artifact = _arguments(artifacts[0], lock_path, lock_sha256, gamma=0.3)
    bad_artifact[bad_artifact.index("--expected-artifact-manifest-sha256") + 1] = (
        "e" * 64
    )
    with pytest.raises(ValueError, match="artifact hash differs"):
        run_gamma_sensitivity(parse_args(bad_artifact))

    with pytest.raises(ValueError, match="not uniquely predeclared"):
        run_gamma_sensitivity(
            parse_args(_arguments(artifacts[0], lock_path, lock_sha256, gamma=0.5))
        )

    spec_path = Path(load_auxiliary_lock(lock_path)["spec_path"])
    spec_path.write_text(spec_path.read_text() + " ")
    with pytest.raises(ValueError, match="spec is missing or its bytes changed"):
        load_auxiliary_lock(lock_path)


def test_auxiliary_lock_binds_every_canonical_artifact_hash(tmp_path):
    _, lock_path, _, _ = _write_locked_workflow(tmp_path, condition_count=2)
    lock = json.loads(lock_path.read_text())
    lock["artifacts"][1]["manifest_sha256"] = "0" * 64
    tampered = lock_path.with_name("tampered.lock.json")
    tampered.write_text(json.dumps(lock, indent=2) + "\n")
    with pytest.raises(ValueError, match="differ from the canonical campaign"):
        load_auxiliary_lock(tampered)
