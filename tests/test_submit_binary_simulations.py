"""Tests for deterministic one-simulation-per-job submission planning."""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import submit_binary_simulations
from scripts.submit_binary_simulations import (
    build_campaign_lock,
    canonical_runtime_receipt_path,
    execute_configured_plan,
    execute_plan,
    load_campaign_lock,
    load_config,
    plan_assemble_jobs,
    plan_common_jobs,
    plan_diagnose_jobs,
    plan_freeze_jobs,
    plan_score_jobs,
    recover_configured_submission,
    reconcile_configured_plan,
    write_campaign_lock,
)
from scripts.assemble_binary_simulations import FINAL_SCORE_FIELDS, assemble
from scripts.analyze_binary import load_condition
from selectseg.binary_artifacts import write_binary_artifact
from selectseg.score_binary_common import (
    parse_args as parse_common_args,
    run_common,
)
from selectseg.score_binary_simulation import (
    parse_args as parse_score_args,
    run_simulation,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_config(tmp_path):
    estimator = tmp_path / "midpoint-v1.json"
    estimator.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimator_id": "midpoint-v1",
                "target_measure": "uniform-threshold",
                "rule": "midpoint",
                "randomized": False,
                "required_seed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    config = {
        "config_schema_version": 1,
        "campaign_id": "unit-midpoint-main-v1",
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "estimator_spec": str(estimator),
        "paths": {
            "artifact_output_root": str(tmp_path / "artifacts"),
            "common_output_root": str(tmp_path / "common"),
            "simulation_output_root": str(tmp_path / "simulations"),
            "assembly_output_root": str(tmp_path / "assembled"),
        },
        "conditions": [
            {
                "dataset": "pet",
                "condition": "clipseg-general",
                "model": "clipseg",
                "checkpoint": None,
                "batch_size": 8,
                "expected_num_samples": 2,
            }
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return load_config(path)


def _write_candidate_config(tmp_path):
    legacy = _write_config(tmp_path)
    config = json.loads(legacy.path.read_text())
    config.update(
        config_schema_version=2,
        execution_policy="scheduler-preview-only",
        gpu_partition_candidates=["saffo-a100", "apollo_agate"],
        cpu_partition_candidates=[
            "amdsmall",
            "agsmall",
            "msismall",
            "saffo-2tb",
        ],
    )
    path = tmp_path / "campaign-candidates-v2.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return load_config(path)


def _execute_candidate_runtime(
    config,
    jobs,
    *,
    submit=True,
    receipt_path,
    runner,
    preflight_runner,
    retry_failed_job_ids=(),
    retry_submission_failures=False,
):
    """Exercise future runtime primitives without bypassing the public v2 gate."""

    assert submit is True
    jobs = submit_binary_simulations._validate_runtime_jobs(jobs)
    receipt = submit_binary_simulations._validated_runtime_receipt_path(
        config, jobs, receipt_path
    )
    jobs = submit_binary_simulations.preflight_plan(jobs, runner=preflight_runner)
    return submit_binary_simulations._execute_runtime_plan(
        config,
        jobs,
        receipt_path=receipt,
        runner=runner,
        retry_failed_job_ids=retry_failed_job_ids,
        retry_submission_failures=retry_submission_failures,
    )


def _write_artifact(tmp_path, *, scientific_input=None):
    sample_ids = ["image-0", "image-1"]
    probability = np.linspace(0.05, 0.95, 12 * 12, dtype=np.float32).reshape(12, 12)
    truth = (probability >= 0.55).astype(np.uint8)
    return write_binary_artifact(
        tmp_path / "artifacts",
        dataset="pet",
        condition="clipseg-general",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="foreground",
        checkpoint=None,
        base_model={"name": "clipseg", "source": "synthetic"},
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
            "model_input": "synthetic",
            "probability_to_native_mask": "synthetic",
        },
        cohort="synthetic cohort",
        sample_ids=sample_ids,
        samples=[
            (sample_id, probability.copy(), truth.copy()) for sample_id in sample_ids
        ],
        command=["synthetic-freeze"],
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        scientific_input=scientific_input,
    )


def _write_locked_candidate_config(tmp_path):
    preview = _write_candidate_config(tmp_path)
    payload = json.loads(preview.path.read_text())
    payload.update(
        execution_policy="scientific-input-locked",
        data_root="data",
        scientific_input_lock={
            "path": str(tmp_path / "scientific-input.lock.json"),
            "sha256": "1" * 64,
        },
    )
    path = tmp_path / "campaign-locked-v2.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return load_config(path)


def _fake_scientific_plan(config):
    dataset_binding = {
        "dataset": "pet",
        "path": "locks/pet.json",
        "sha256": "2" * 64,
    }
    components = {
        "datasets": [dataset_binding],
        "source": {"path": "locks/source.json", "sha256": "3" * 64},
        "base_models": {"path": "locks/models.json", "sha256": "4" * 64},
        "checkpoints": {"path": "locks/checkpoints.json", "sha256": "5" * 64},
        "environment": {"path": "locks/environment.json", "sha256": "6" * 64},
    }
    binding = {
        "path": Path(config.data["scientific_input_lock"]["path"]),
        "sha256": "1" * 64,
        "lock": {
            "campaign_id": config.data["campaign_id"],
            "science_config": {"projection_sha256": "7" * 64},
            "components": components,
        },
        "projection": {
            "config": {
                "campaign_id": config.data["campaign_id"],
                "conditions": config.data["conditions"],
            }
        },
        "components": {
            "datasets": {"pet": {}},
            **{
                key: {}
                for key in ("source", "base_models", "checkpoints", "environment")
            },
        },
    }
    return Path(config.data["scientific_input_lock"]["path"]), "1" * 64, binding


def _locked_campaign(tmp_path):
    config = _write_config(tmp_path)
    artifact = _write_artifact(tmp_path)
    lock = build_campaign_lock(config, [artifact])
    lock_path, lock_sha = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    return config, artifact, lock_path, lock_sha


def _locked_candidate_campaign(tmp_path):
    config = _write_candidate_config(tmp_path)
    artifact = _write_artifact(tmp_path)
    lock = build_campaign_lock(config, [artifact])
    lock_path, lock_sha = write_campaign_lock(
        lock, tmp_path / "campaign-candidates-v2.lock.json"
    )
    return config, artifact, lock_path, lock_sha


def test_freeze_and_score_plans_are_one_job_per_cartesian_row(tmp_path):
    config, artifact, lock_path, lock_sha = _locked_campaign(tmp_path)

    freeze_jobs = plan_freeze_jobs(config)
    assert len(freeze_jobs) == 1
    assert freeze_jobs[0].command.count("scripts/slurm/freeze_binary_maps.sbatch") == 1
    freeze_command = list(freeze_jobs[0].command)
    assert freeze_command[freeze_command.index("--expected-num-samples") + 1] == "2"
    assert freeze_command[freeze_command.index("--partition") + 1] == (
        "saffo-a100,apollo_agate"
    )
    assert freeze_command[freeze_command.index("--account") + 1] == "ssafo"
    assert freeze_jobs[0].key[-1] == "saffo-a100,apollo_agate"

    common_jobs = plan_common_jobs(config, lock_path)
    assert len(common_jobs) == 1
    common_command = list(common_jobs[0].command)
    assert common_jobs[0].phase == "common"
    assert common_command.count("scripts/slurm/score_binary_common.sbatch") == 1
    assert common_command[common_command.index("--partition") + 1] == "agsmall"
    assert common_command[common_command.index("--account") + 1] == "ssafo"
    assert "--m" not in common_command
    assert common_command.count("--gamma") == 1

    jobs = plan_score_jobs(config, lock_path)
    assert len(jobs) == 3
    assert [job.key[-2] for job in jobs] == [2, 8, 32]
    assert [job.key[2] for job in jobs] == ["agsmall", "amdsmall", "msismall"]
    assert len({job.key for job in jobs}) == 3
    for job in jobs:
        command = list(job.command)
        assert "--array" not in command
        assert command.count("scripts/slurm/score_binary_simulation.sbatch") == 1
        assert command.count("--partition") == 1
        assert command.count("--account") == 1
        assert command.count("--m") == 1
        assert command.count("--gamma") == 1
        assert command.count("--seed") == 1
        assert command[command.index("--campaign-id") + 1] == config.data["campaign_id"]
        assert command[command.index("--expected-campaign-lock-sha256") + 1] == lock_sha
        assert command[command.index("--artifact-manifest") + 1] == str(
            artifact.resolve()
        )
        assert command[
            command.index("--expected-artifact-manifest-sha256") + 1
        ] == _sha256(artifact)

    diagnostic_jobs = plan_diagnose_jobs(
        config, lock_path, output_root="outputs/unit-diagnostics"
    )
    assert len(diagnostic_jobs) == 1
    diagnostic = list(diagnostic_jobs[0].command)
    assert diagnostic_jobs[0].phase == "diagnose"
    assert diagnostic_jobs[0].key == ("pet", "clipseg-general", "agsmall")
    assert "--array" not in diagnostic
    assert diagnostic.count("scripts/slurm/diagnose_binary_artifact.sbatch") == 1
    assert diagnostic[diagnostic.index("--partition") + 1] == "agsmall"
    assert diagnostic[diagnostic.index("--account") + 1] == "ssafo"
    assert diagnostic[diagnostic.index("--artifact-manifest") + 1] == str(
        artifact.resolve()
    )
    assert diagnostic[
        diagnostic.index("--expected-artifact-manifest-sha256") + 1
    ] == _sha256(artifact)
    assert diagnostic[diagnostic.index("--output-root") + 1] == (
        "outputs/unit-diagnostics"
    )
    assert diagnostic[diagnostic.index("--decision-threshold") + 1] == "0.5"
    assert diagnostic.count("--write-descriptors") == 1

    with pytest.raises(FileNotFoundError, match="scoring outputs are incomplete"):
        plan_assemble_jobs(config, lock_path)


def test_candidate_partition_mode_keeps_one_experiment_per_independent_job(tmp_path):
    config, _, lock_path, _ = _locked_candidate_campaign(tmp_path)
    gpu_request = "saffo-a100,apollo_agate"
    cpu_requests = (
        "amdsmall,agsmall,msismall,saffo-2tb",
        "agsmall,msismall,amdsmall,saffo-2tb",
        "msismall,amdsmall,agsmall,saffo-2tb",
    )

    freeze_jobs = plan_freeze_jobs(config)
    common_jobs = plan_common_jobs(config, lock_path)
    score_jobs = plan_score_jobs(config, lock_path)
    diagnose_jobs = plan_diagnose_jobs(config, lock_path)

    assert len(freeze_jobs) == len(common_jobs) == len(diagnose_jobs) == 1
    assert len(score_jobs) == 3
    assert len({job.key for job in score_jobs}) == 3
    score_identities = {
        (
            command[command.index("--artifact-manifest") + 1],
            command[command.index("--gamma") + 1],
            command[command.index("--m") + 1],
            command[command.index("--seed") + 1],
        )
        for command in (list(job.command) for job in score_jobs)
    }
    assert len(score_identities) == 3

    for job in freeze_jobs:
        command = list(job.command)
        assert command[command.index("--partition") + 1] == gpu_request
        assert command.count("scripts/slurm/freeze_binary_maps.sbatch") == 1
        assert "--array" not in command
        assert not any(token.startswith("--array=") for token in command)

    cpu_jobs = (*common_jobs, *score_jobs, *diagnose_jobs)
    for job in cpu_jobs:
        command = list(job.command)
        assert command.count("--partition") == 1
        assert set(command[command.index("--partition") + 1].split(",")) == {
            "amdsmall",
            "agsmall",
            "msismall",
            "saffo-2tb",
        }
        assert "--array" not in command
        assert not any(token.startswith("--array=") for token in command)
    assert [
        job.command[job.command.index("--partition") + 1] for job in score_jobs
    ] == list(cpu_requests)
    assert (
        common_jobs[0].command[common_jobs[0].command.index("--partition") + 1]
        == cpu_requests[0]
    )
    assert (
        diagnose_jobs[0].command[diagnose_jobs[0].command.index("--partition") + 1]
        == cpu_requests[0]
    )
    assert all(
        job.command.count("scripts/slurm/score_binary_simulation.sbatch") == 1
        for job in score_jobs
    )

    common_command = list(common_jobs[0].command)
    common_wrapper = common_command.index("scripts/slurm/score_binary_common.sbatch")
    _, common_manifest = run_common(
        parse_common_args(common_command[common_wrapper + 1 :])
    )
    for job in score_jobs:
        command = list(job.command)
        score_wrapper = command.index("scripts/slurm/score_binary_simulation.sbatch")
        run_simulation(parse_score_args(command[score_wrapper + 1 :]))
    assemble_jobs = plan_assemble_jobs(config, lock_path)
    assert len(assemble_jobs) == 1
    assemble_command = list(assemble_jobs[0].command)
    assert assemble_command[assemble_command.index("--partition") + 1] == (
        cpu_requests[0]
    )
    assert (
        assemble_command.count("scripts/slurm/assemble_binary_simulations.sbatch") == 1
    )
    assert assemble_command[assemble_command.index("--common") + 1] == str(
        common_manifest
    )
    assert "--array" not in assemble_command
    assert not any(token.startswith("--array=") for token in assemble_command)


def test_scientific_locked_freeze_plan_binds_each_job_to_exact_inputs(
    tmp_path, monkeypatch
):
    config = _write_locked_candidate_config(tmp_path)
    science = _fake_scientific_plan(config)
    monkeypatch.setattr(
        submit_binary_simulations,
        "_scientific_plan_binding",
        lambda candidate, *, mode: science,
    )

    jobs = plan_freeze_jobs(config)
    assert len(jobs) == 1
    command = list(jobs[0].command)
    assert command[command.index("--partition") + 1] == (
        "saffo-a100,apollo_agate"
    )
    assert command[command.index("--data-root") + 1] == "data"
    assert command[command.index("--campaign-config") + 1] == str(config.path)
    assert command[command.index("--expected-campaign-config-sha256") + 1] == (
        config.sha256
    )
    assert command[command.index("--scientific-input-lock") + 1] == str(
        science[0]
    )
    assert command[
        command.index("--expected-scientific-input-lock-sha256") + 1
    ] == science[1]
    assert len(command[command.index("--expected-condition-input-sha256") + 1]) == 64
    assert "--array" not in command


def test_scientific_locked_submit_preflights_then_receipts_without_duplicates(
    tmp_path, monkeypatch
):
    config = _write_locked_candidate_config(tmp_path)
    science = _fake_scientific_plan(config)
    monkeypatch.setattr(
        submit_binary_simulations,
        "_scientific_plan_binding",
        lambda candidate, *, mode: science,
    )
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    preflight_calls = []
    real_calls = []

    def preflight(command, **kwargs):
        preflight_calls.append(tuple(command))
        assert "--test-only" in command and "--parsable" not in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def real(command, **kwargs):
        real_calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="73001\n", stderr="")

    assert execute_configured_plan(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=real,
        preflight_runner=preflight,
    ) == ("73001",)
    assert execute_configured_plan(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=real,
        preflight_runner=preflight,
    ) == ()
    assert len(preflight_calls) == 2
    assert real_calls == [jobs[0].command]
    events = [json.loads(line)["event"] for line in receipt.read_text().splitlines()]
    assert events == ["submitting", "submitted"]


def test_scientific_campaign_lock_propagates_condition_roots(
    tmp_path, monkeypatch
):
    from selectseg import scientific_inputs

    config = _write_locked_candidate_config(tmp_path)
    science = _fake_scientific_plan(config)
    monkeypatch.setattr(
        submit_binary_simulations,
        "_scientific_plan_binding",
        lambda candidate, *, mode: science,
    )
    identity = scientific_inputs.condition_input_identity(
        science[2],
        dataset="pet",
        model="clipseg",
        condition="clipseg-general",
    )
    scientific = {
        **identity["scientific_input_hashes"],
        "condition_input_sha256": identity["scientific_input_sha256"],
    }
    artifact = _write_artifact(tmp_path, scientific_input=scientific)
    lock = build_campaign_lock(config, [artifact])
    assert lock["lock_schema_version"] == 2
    assert lock["scientific_input"] == {
        "root_lock_path": submit_binary_simulations._portable_path(science[0]),
        "root_lock_sha256": science[1],
        "science_projection_sha256": "7" * 64,
    }
    assert lock["artifacts"][0]["scientific_input"] == scientific

    lock_path, _ = write_campaign_lock(lock, tmp_path / "scientific-campaign.lock")
    monkeypatch.setattr(
        scientific_inputs,
        "load_root_lock",
        lambda path, *, expected_sha256: science[2],
    )
    _, _, loaded = load_campaign_lock(lock_path, config=config)
    assert loaded == lock


def test_schema_v1_partition_commands_remain_legacy_compatible(tmp_path):
    config, _, lock_path, _ = _locked_campaign(tmp_path)
    assert config.data["config_schema_version"] == 1
    assert "gpu_partition_candidates" not in config.data
    assert "cpu_partition_candidates" not in config.data
    assert [
        job.command[job.command.index("--partition") + 1]
        for job in plan_freeze_jobs(config)
    ] == ["saffo-a100,apollo_agate"]
    assert [
        job.command[job.command.index("--partition") + 1]
        for job in plan_common_jobs(config, lock_path)
    ] == ["agsmall"]
    score_jobs = plan_score_jobs(config, lock_path)
    assert [
        job.command[job.command.index("--partition") + 1] for job in score_jobs
    ] == ["agsmall", "amdsmall", "msismall"]
    assert [job.key[2] for job in score_jobs] == [
        "agsmall",
        "amdsmall",
        "msismall",
    ]


@pytest.mark.parametrize(
    ("version", "updates", "message"),
    [
        (
            2,
            {"execution_policy": "scheduler-preview-only"},
            "must be provided together",
        ),
        (
            1,
            {
                "gpu_partition_candidates": ["saffo-a100", "apollo_agate"],
                "cpu_partition_candidates": [
                    "amdsmall",
                    "agsmall",
                    "msismall",
                    "saffo-2tb",
                ],
            },
            "require config_schema_version 2",
        ),
        (
            2,
            {
                "execution_policy": "scheduler-preview-only",
                "gpu_partition_candidates": ["saffo-a100", "apollo_agate"],
            },
            "must be provided together",
        ),
        (
            2,
            {
                "execution_policy": "scheduler-preview-only",
                "gpu_partition_candidates": ["apollo_agate", "saffo-a100"],
                "cpu_partition_candidates": [
                    "amdsmall",
                    "agsmall",
                    "msismall",
                    "saffo-2tb",
                ],
            },
            "gpu_partition_candidates must equal",
        ),
    ],
)
def test_candidate_partition_fields_are_versioned_paired_and_exact(
    tmp_path, version, updates, message
):
    config = _write_config(tmp_path)
    payload = json.loads(config.path.read_text())
    payload["config_schema_version"] = version
    payload.update(updates)
    path = tmp_path / "invalid-candidate-config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("version", "policy", "message"),
    [
        (2, None, "execution_policy must equal"),
        (2, "scientific-input-locked", "require data_root"),
        (1, "scheduler-preview-only", "requires config_schema_version 2"),
    ],
)
def test_execution_policy_is_exact_and_schema_v2_only(
    tmp_path, version, policy, message
):
    config = _write_config(tmp_path)
    payload = json.loads(config.path.read_text())
    payload["config_schema_version"] = version
    if version == 2:
        payload.update(
            gpu_partition_candidates=["saffo-a100", "apollo_agate"],
            cpu_partition_candidates=[
                "amdsmall",
                "agsmall",
                "msismall",
                "saffo-2tb",
            ],
        )
    if policy is not None:
        payload["execution_policy"] = policy
    path = tmp_path / "invalid-execution-policy.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_freeze_plan_supports_an_explicit_reproducible_development_subset(tmp_path):
    config = _write_config(tmp_path)
    condition = config.data["conditions"][0]
    condition.update(
        expected_num_samples=2,
        expected_dataset_samples=200,
        freeze_limit=2,
    )
    command = list(plan_freeze_jobs(config)[0].command)
    assert command[command.index("--expected-num-samples") + 1] == "200"
    assert command[command.index("--limit") + 1] == "2"


def test_execute_plan_is_dry_run_by_default_and_submits_each_job_separately(
    tmp_path, capsys
):
    config, _, lock_path, _ = _locked_campaign(tmp_path)
    jobs = plan_score_jobs(config, lock_path)
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=f"job-{len(calls)}\n")

    assert execute_plan(jobs, runner=runner) == ()
    assert calls == []
    output = capsys.readouterr().out
    assert output.count("scripts/slurm/score_binary_simulation.sbatch") == 3
    assert "planned_jobs=3 submitted_jobs=0" in output

    receipt = tmp_path / "score-submissions.jsonl"
    assert execute_plan(jobs, submit=True, receipt_path=receipt, runner=runner) == (
        "job-1",
        "job-2",
        "job-3",
    )
    assert len(calls) == 3
    assert all(call[0][0][0] == "sbatch" for call in calls)
    assert execute_plan(jobs, submit=True, receipt_path=receipt, runner=runner) == ()
    assert len(calls) == 3
    receipt_rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["status"] for row in receipt_rows] == [
        "submitting",
        "submitted",
    ] * 3

    diagnostic_jobs = plan_diagnose_jobs(config, lock_path)
    diagnostic_receipt = tmp_path / "diagnose-submissions.jsonl"
    assert execute_plan(
        diagnostic_jobs,
        submit=True,
        receipt_path=diagnostic_receipt,
        runner=runner,
    ) == ("job-4",)
    assert (
        execute_plan(
            diagnostic_jobs,
            submit=True,
            receipt_path=diagnostic_receipt,
            runner=runner,
        )
        == ()
    )
    diagnostic_rows = [
        json.loads(line) for line in diagnostic_receipt.read_text().splitlines()
    ]
    assert [row["phase"] for row in diagnostic_rows] == ["diagnose", "diagnose"]
    assert [row["status"] for row in diagnostic_rows] == [
        "submitting",
        "submitted",
    ]


@pytest.mark.parametrize("existing_receipt", [False, True])
def test_schema_v2_real_submit_is_rejected_before_scheduler_or_receipt_access(
    tmp_path, existing_receipt
):
    config, _, lock_path, _ = _locked_candidate_campaign(tmp_path)
    jobs = plan_score_jobs(config, lock_path)
    receipt = canonical_runtime_receipt_path(config, "score")
    sentinel = b"sentinel receipt bytes\n"
    if existing_receipt:
        receipt.parent.mkdir(parents=True)
        receipt.write_bytes(sentinel)
        sentinel_time = 1_700_000_000_123_456_789
        os.utime(receipt, ns=(sentinel_time, sentinel_time))
        initial_stat = receipt.stat()

    calls = []

    def forbidden_runner(command, **kwargs):
        calls.append((tuple(command), kwargs))
        raise AssertionError("preview-only rejection must precede scheduler access")

    with pytest.raises(RuntimeError, match="scientific inputs"):
        execute_configured_plan(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=forbidden_runner,
            preflight_runner=forbidden_runner,
        )

    assert calls == []
    if existing_receipt:
        final_stat = receipt.stat()
        assert receipt.read_bytes() == sentinel
        assert final_stat.st_mtime_ns == initial_stat.st_mtime_ns
        assert final_stat.st_size == initial_stat.st_size
    else:
        assert not receipt.exists()
        assert not receipt.parent.exists()


def test_schema_v2_preflights_the_whole_wave_without_real_jobs_or_receipt(tmp_path):
    config, _, lock_path, _ = _locked_candidate_campaign(tmp_path)
    jobs = plan_score_jobs(config, lock_path)
    original_jobs = tuple(jobs)
    candidate_requests = [
        "amdsmall,agsmall,msismall,saffo-2tb",
        "agsmall,msismall,amdsmall,saffo-2tb",
        "msismall,amdsmall,agsmall,saffo-2tb",
    ]
    preflight_calls = []

    def preflight_runner(command, **kwargs):
        command = tuple(command)
        preflight_calls.append((command, kwargs))
        assert "--test-only" in command
        assert "--parsable" not in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def forbidden_real_runner(*args, **kwargs):
        raise AssertionError("scheduler preview must not invoke real sbatch")

    assert (
        execute_configured_plan(
            config,
            jobs,
            scheduler_preflight_only=True,
            runner=forbidden_real_runner,
            preflight_runner=preflight_runner,
        )
        == ()
    )

    assert tuple(jobs) == original_jobs
    assert len(preflight_calls) == len(jobs)
    preflight_commands = [call[0] for call in preflight_calls]
    for job, preflight, candidate_request in zip(
        jobs, preflight_commands, candidate_requests
    ):
        expected_preflight = tuple(
            "--test-only" if token == "--parsable" else token for token in job.command
        )
        assert preflight == expected_preflight
        assert preflight[preflight.index("--partition") + 1] == candidate_request
        assert "--array" not in preflight
    assert not canonical_runtime_receipt_path(config, "score").exists()


def test_scheduler_preflight_surfaces_successful_forecast(tmp_path, capsys):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)

    def runner(command, **kwargs):
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 45,
        }
        return SimpleNamespace(
            returncode=0,
            stdout="start=now partition=saffo-a100\n",
            stderr="scheduler advisory\n" * 426,
        )

    assert submit_binary_simulations.preflight_plan(jobs, runner=runner) == jobs
    captured = capsys.readouterr()
    assert "scheduler_preflight=ok" in captured.out
    assert "start=now partition=saffo-a100" in captured.out
    assert captured.err.count("scheduler advisory") == 1
    assert "occurrences=426" in captured.err


def test_scheduler_preflight_surfaces_failure_reason_and_stays_fail_closed(
    tmp_path, capsys
):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    error = subprocess.CalledProcessError(
        1,
        jobs[0].command,
        output="scheduler stdout reason",
        stderr="scheduler stderr reason",
    )

    def runner(*args, **kwargs):
        raise error

    with pytest.raises(subprocess.CalledProcessError) as raised:
        submit_binary_simulations.preflight_plan(jobs, runner=runner)
    assert raised.value is error
    captured = capsys.readouterr()
    assert "scheduler_preflight=failed" in captured.out
    assert "scheduler stdout reason" in captured.out
    assert "scheduler stderr reason" in captured.err


def test_scheduler_preflight_timeout_is_compact_and_fail_closed(tmp_path, capsys):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    error = subprocess.TimeoutExpired(
        jobs[0].command,
        timeout=45,
        output=(b"socket error\n" * 426),
        stderr=b"scheduler timed out\n",
    )

    def runner(command, **kwargs):
        assert kwargs["timeout"] == 45
        raise error

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        submit_binary_simulations.preflight_plan(jobs, runner=runner)
    assert raised.value is error
    captured = capsys.readouterr()
    assert "scheduler_preflight=failed reason=timeout_after_45s" in captured.out
    assert captured.out.count("socket error") == 1
    assert "occurrences=426" in captured.out
    assert "scheduler timed out" in captured.err


def test_scheduler_output_compaction_preserves_counts_and_caps_distinct_lines():
    lines = ["repeated", "repeated", *[f"distinct-{index}" for index in range(25)]]
    compact = submit_binary_simulations._compact_scheduler_output("\n".join(lines))
    assert "repeated [occurrences=2]" in compact
    assert "distinct-0 [occurrences=1]" in compact
    assert "omitted 6 distinct lines (6 occurrences)" in compact
    assert "distinct-24" not in compact
    assert len(compact) < 10_000


@pytest.mark.parametrize(
    "array_tokens",
    [
        ("--array", "0-1"),
        ("--array=0-1",),
        ("-a", "0-1"),
        ("-a=0-1",),
        ("-a0-1",),
    ],
)
def test_schema_v2_runtime_rejects_every_slurm_array_spelling(
    tmp_path, array_tokens
):
    config = _write_candidate_config(tmp_path)
    job = plan_freeze_jobs(config)[0]
    command = list(job.command)
    wrapper_index = command.index("scripts/slurm/freeze_binary_maps.sbatch")
    command[wrapper_index:wrapper_index] = array_tokens
    array_job = submit_binary_simulations.PlannedJob(
        phase=job.phase,
        key=job.key,
        command=tuple(command),
    )
    with pytest.raises(ValueError, match="forbids Slurm arrays"):
        submit_binary_simulations._validate_runtime_jobs((array_job,))


def test_array_guard_does_not_parse_wrapper_argument_values(tmp_path):
    config = _write_candidate_config(tmp_path)
    job = plan_freeze_jobs(config)[0]
    command = (*job.command, "--artifact-label", "-a0-1")
    safe_job = submit_binary_simulations.PlannedJob(
        phase=job.phase,
        key=job.key,
        command=command,
    )
    assert submit_binary_simulations._validate_runtime_jobs((safe_job,)) == (safe_job,)


def test_canonical_sbatch_wrappers_have_no_active_array_directive():
    repository = Path(__file__).resolve().parents[1]
    wrappers = (
        "scripts/slurm/freeze_binary_maps.sbatch",
        "scripts/slurm/score_binary_common.sbatch",
        "scripts/slurm/score_binary_simulation.sbatch",
        "scripts/slurm/assemble_binary_simulations.sbatch",
        "scripts/slurm/diagnose_binary_artifact.sbatch",
        "scripts/slurm/build_scientific_dataset.sbatch",
    )
    directive = re.compile(
        r"^\s*#SBATCH\s+(?:--array(?:=|\s)|-a(?:=|\d|\s))"
    )
    for relative_path in wrappers:
        text = (repository / relative_path).read_text()
        assert not any(directive.match(line) for line in text.splitlines())


def test_schema_v2_scheduler_preflight_rejects_receipt_before_scheduler(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    calls = []

    def forbidden_runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("receipt validation must precede scheduler access")

    with pytest.raises(ValueError, match="never accepts --receipt"):
        execute_configured_plan(
            config,
            jobs,
            scheduler_preflight_only=True,
            receipt_path=receipt,
            runner=forbidden_runner,
            preflight_runner=forbidden_runner,
        )
    assert calls == []
    assert not receipt.exists()


def test_schema_v2_freeze_preflight_preserves_exact_gpu_candidate_request(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    calls = []

    def runner(command, **kwargs):
        command = tuple(command)
        calls.append(command)
        assert "--test-only" in command
        assert "--parsable" not in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert (
        execute_configured_plan(
            config,
            jobs,
            scheduler_preflight_only=True,
            runner=lambda *args, **kwargs: pytest.fail("real sbatch was called"),
            preflight_runner=runner,
        )
        == ()
    )
    assert len(calls) == 1
    assert "--test-only" in calls[0]
    assert [command[command.index("--partition") + 1] for command in calls] == [
        "saffo-a100,apollo_agate",
    ]
    job_name = jobs[0].command[jobs[0].command.index("--job-name") + 1]
    assert job_name.endswith(f"-c{config.sha256[:16]}")
    assert not canonical_runtime_receipt_path(config, "freeze").exists()


def test_schema_v1_execution_does_not_gain_scheduler_preflight(tmp_path):
    config, _, lock_path, _ = _locked_campaign(tmp_path)
    jobs = plan_score_jobs(config, lock_path)
    receipt = tmp_path / "score-v1.jsonl"
    real_commands = []

    def forbidden_preflight(*args, **kwargs):
        raise AssertionError("schema v1 must not invoke scheduler preflight")

    def real_runner(command, **kwargs):
        real_commands.append(tuple(command))
        return SimpleNamespace(stdout=f"job-{len(real_commands)}\n")

    assert execute_configured_plan(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=real_runner,
        preflight_runner=forbidden_preflight,
    ) == ("job-1", "job-2", "job-3")
    assert real_commands == [job.command for job in jobs]
    assert all("--parsable" in command for command in real_commands)
    assert all("--test-only" not in command for command in real_commands)
    assert all(
        not command[command.index("--job-name") + 1].rsplit("-", 1)[-1].startswith("c")
        for command in real_commands
    )


def test_schema_v2_receipt_path_is_canonical_and_substitution_is_rejected(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    canonical = canonical_runtime_receipt_path(config, "freeze")
    assert canonical == (tmp_path / "receipts" / "freeze.jsonl").resolve()
    substitute = tmp_path / "other-receipts" / "freeze.jsonl"
    calls = []

    def forbidden_runner(command, **kwargs):
        calls.append((command, kwargs))
        raise AssertionError("path validation must precede scheduler access")

    with pytest.raises(ValueError, match="path substitution rejected"):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=substitute,
            runner=forbidden_runner,
            preflight_runner=forbidden_runner,
        )
    with pytest.raises(ValueError, match="path substitution rejected"):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=canonical_runtime_receipt_path(config, "score"),
            runner=forbidden_runner,
            preflight_runner=forbidden_runner,
        )
    assert calls == []
    assert not substitute.exists()
    assert not canonical.exists()


def test_schema_v2_config_requires_one_campaign_root_for_canonical_receipts(tmp_path):
    config = _write_config(tmp_path)
    payload = json.loads(config.path.read_text())
    payload.update(
        config_schema_version=2,
        execution_policy="scheduler-preview-only",
        gpu_partition_candidates=["saffo-a100", "apollo_agate"],
        cpu_partition_candidates=[
            "amdsmall",
            "agsmall",
            "msismall",
            "saffo-2tb",
        ],
    )
    payload["paths"]["simulation_output_root"] = str(
        tmp_path / "different-campaign" / "simulations"
    )
    path = tmp_path / "split-roots-v2.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="sibling directories"):
        load_config(path)


def test_schema_v2_running_and_unknown_jobs_are_never_resubmitted(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    real_calls = []

    def preflight(command, **kwargs):
        assert "--test-only" in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def submit(command, **kwargs):
        real_calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="3001\n", stderr="")

    assert _execute_candidate_runtime(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=submit,
        preflight_runner=preflight,
    ) == ("3001",)
    assert (
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=submit,
            preflight_runner=preflight,
        )
        == ()
    )
    assert real_calls == [jobs[0].command]

    scheduler_calls = []

    def running_scheduler(command, **kwargs):
        scheduler_calls.append(tuple(command))
        if command[0] == "sacct":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command[0] == "squeue"
        return SimpleNamespace(
            returncode=0,
            stdout="3001|RUNNING|saffo-a100|gpu-a|None\n",
            stderr="",
        )

    before = receipt.read_bytes()
    assert (
        reconcile_configured_plan(
            config, jobs, receipt_path=receipt, runner=running_scheduler
        )
        == ()
    )
    assert [command[0] for command in scheduler_calls] == ["sacct", "squeue"]
    assert receipt.read_bytes() == before

    unknown_calls = []

    def unknown_scheduler(command, **kwargs):
        unknown_calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="state remained unknown"):
        reconcile_configured_plan(
            config, jobs, receipt_path=receipt, runner=unknown_scheduler
        )
    assert [command[0] for command in unknown_calls] == ["sacct", "squeue"]
    assert receipt.read_bytes() == before
    assert (
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=submit,
            preflight_runner=preflight,
        )
        == ()
    )
    assert real_calls == [jobs[0].command]


def test_schema_v2_reconcile_records_terminal_facts_once(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")

    def preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def submit(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="3101\n", stderr="")

    _execute_candidate_runtime(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=submit,
        preflight_runner=preflight,
    )
    scheduler_calls = []

    def completed_scheduler(command, **kwargs):
        scheduler_calls.append(tuple(command))
        assert command[0] == "sacct"
        return SimpleNamespace(
            returncode=0,
            stdout="3101|COMPLETED|0:0|saffo-a100|gpu-b|None\n",
            stderr="",
        )

    assert reconcile_configured_plan(
        config, jobs, receipt_path=receipt, runner=completed_scheduler
    ) == (("3101", "completed"),)
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "submitting",
        "submitted",
        "completed",
    ]
    terminal = rows[-1]
    assert terminal["scheduler_state"] == "COMPLETED"
    assert terminal["scheduler_exit_code"] == "0:0"
    assert terminal["scheduler_partition"] == "saffo-a100"
    assert terminal["scheduler_nodes"] == "gpu-b"
    assert terminal["scheduler_reason"] == "None"

    scheduler_calls.clear()
    before = receipt.read_bytes()
    assert (
        reconcile_configured_plan(
            config, jobs, receipt_path=receipt, runner=completed_scheduler
        )
        == ()
    )
    assert scheduler_calls == []
    assert receipt.read_bytes() == before


def test_schema_v2_failed_job_requires_exact_replacement_and_records_lineage(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    submitted_ids = iter(("4001", "4002"))
    submit_calls = []

    def preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def submit(command, **kwargs):
        submit_calls.append(tuple(command))
        return SimpleNamespace(
            returncode=0, stdout=next(submitted_ids) + "\n", stderr=""
        )

    assert _execute_candidate_runtime(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=submit,
        preflight_runner=preflight,
    ) == ("4001",)

    def failed_scheduler(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="4001|OUT_OF_MEMORY|0:125|saffo-a100|gpu-c|oom-kill\n",
            stderr="",
        )

    assert reconcile_configured_plan(
        config, jobs, receipt_path=receipt, runner=failed_scheduler
    ) == (("4001", "failed"),)
    assert (
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=submit,
            preflight_runner=preflight,
        )
        == ()
    )
    assert len(submit_calls) == 1

    with pytest.raises(ValueError, match="does not name a current failed job"):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=submit,
            preflight_runner=preflight,
            retry_failed_job_ids=("4999",),
        )
    assert len(submit_calls) == 1
    assert _execute_candidate_runtime(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=submit,
        preflight_runner=preflight,
        retry_failed_job_ids=("4001",),
    ) == ("4002",)
    assert submit_calls == [jobs[0].command, jobs[0].command]

    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "submitting",
        "submitted",
        "failed",
        "submitting",
        "submitted",
    ]
    replacement = rows[-2:]
    assert {row["attempt"] for row in replacement} == {2}
    assert {row["authorization"] for row in replacement} == {"retry_failed_job_id"}
    assert {row["predecessor_job_id"] for row in replacement} == {"4001"}
    assert replacement[-1]["job_id"] == "4002"
    assert tuple(replacement[-1]["command"]) == jobs[0].command
    assert "--array" not in replacement[-1]["command"]


@pytest.mark.parametrize("crash_window", ["before_scheduler", "after_scheduler"])
def test_schema_v2_dangling_submission_intent_fails_closed(
    tmp_path, monkeypatch, crash_window
):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")

    def preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    if crash_window == "before_scheduler":

        def crashing_submit(command, **kwargs):
            raise KeyboardInterrupt("simulated process death before sbatch returns")

    else:

        def crashing_submit(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="5001\n", stderr="")

        original_append = submit_binary_simulations._append_runtime_receipt_event

        def crash_before_submitted_commit(handle, event):
            if event["event"] == "submitted":
                raise KeyboardInterrupt("simulated death before job-id commit")
            return original_append(handle, event)

        monkeypatch.setattr(
            submit_binary_simulations,
            "_append_runtime_receipt_event",
            crash_before_submitted_commit,
        )

    with pytest.raises(KeyboardInterrupt):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=crashing_submit,
            preflight_runner=preflight,
        )
    if crash_window == "after_scheduler":
        monkeypatch.setattr(
            submit_binary_simulations,
            "_append_runtime_receipt_event",
            original_append,
        )
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["submitting"]

    if crash_window == "before_scheduler":
        recovery_calls = []

        def absent_scheduler(command, **kwargs):
            recovery_calls.append(tuple(command))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with pytest.raises(RuntimeError, match="cannot prove"):
            recover_configured_submission(
                config,
                jobs,
                receipt_path=receipt,
                job_id="5001",
                runner=absent_scheduler,
            )
        assert [command[0] for command in recovery_calls] == ["sacct", "squeue"]
        assert [
            json.loads(line)["event"] for line in receipt.read_text().splitlines()
        ] == ["submitting"]
    else:
        command = jobs[0].command
        expected_name = command[command.index("--job-name") + 1]
        recovery_calls = []

        def matching_scheduler(command, **kwargs):
            recovery_calls.append(tuple(command))
            assert command[0] == "sacct"
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"5001|{expected_name}|ssafo|saffo-a100|RUNNING|0:0|gpu-d|None\n"
                ),
                stderr="",
            )

        assert recover_configured_submission(
            config,
            jobs,
            receipt_path=receipt,
            job_id="5001",
            runner=matching_scheduler,
        ) == ("5001",)
        assert [command[0] for command in recovery_calls] == ["sacct"]
        recovery_calls.clear()

        def forbidden_idempotent_query(command, **kwargs):
            recovery_calls.append(tuple(command))
            raise AssertionError("repeated recovery must use the durable binding")

        assert recover_configured_submission(
            config,
            jobs,
            receipt_path=receipt,
            job_id="5001",
            runner=forbidden_idempotent_query,
        ) == ("5001",)
        assert recovery_calls == []
        recovered_rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        assert [row["event"] for row in recovered_rows] == [
            "submitting",
            "submission_recovered",
        ]
        recovered = recovered_rows[-1]
        assert recovered["job_id"] == "5001"
        assert recovered["scheduler_job_name"] == expected_name
        assert recovered["scheduler_account"] == "ssafo"
        assert recovered["scheduler_partition"] == "saffo-a100"

    scheduler_calls = []

    def forbidden_scheduler(command, **kwargs):
        scheduler_calls.append(tuple(command))
        raise AssertionError("an intent without a durable job id is ambiguous")

    if crash_window == "before_scheduler":
        with pytest.raises(RuntimeError, match="fails closed"):
            reconcile_configured_plan(
                config, jobs, receipt_path=receipt, runner=forbidden_scheduler
            )
        assert scheduler_calls == []

    real_calls = []

    def forbidden_submit(command, **kwargs):
        real_calls.append(tuple(command))
        raise AssertionError("a dangling intent must never be resubmitted")

    if crash_window == "before_scheduler":
        with pytest.raises(RuntimeError, match="fail closed"):
            _execute_candidate_runtime(
                config,
                jobs,
                submit=True,
                receipt_path=receipt,
                runner=forbidden_submit,
                preflight_runner=preflight,
            )
    else:
        assert (
            _execute_candidate_runtime(
                config,
                jobs,
                submit=True,
                receipt_path=receipt,
                runner=forbidden_submit,
                preflight_runner=preflight,
            )
            == ()
        )
    assert real_calls == []


def test_schema_v2_recovery_rejects_wrong_job_id_and_scheduler_identity(
    tmp_path, monkeypatch
):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    original_append = submit_binary_simulations._append_runtime_receipt_event

    def crash_before_submitted_commit(handle, event):
        if event["event"] == "submitted":
            raise KeyboardInterrupt("simulated post-sbatch crash")
        return original_append(handle, event)

    monkeypatch.setattr(
        submit_binary_simulations,
        "_append_runtime_receipt_event",
        crash_before_submitted_commit,
    )
    with pytest.raises(KeyboardInterrupt):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout="5101\n", stderr=""
            ),
            preflight_runner=lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout="", stderr=""
            ),
        )
    monkeypatch.setattr(
        submit_binary_simulations,
        "_append_runtime_receipt_event",
        original_append,
    )
    before = receipt.read_bytes()

    def missing_job(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="cannot prove"):
        recover_configured_submission(
            config,
            jobs,
            receipt_path=receipt,
            job_id="5199",
            runner=missing_job,
        )
    assert receipt.read_bytes() == before

    def wrong_identity(command, **kwargs):
        if command[0] == "sacct":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "5101|unrelated-job|wrong-account|msismall|RUNNING|"
                    "0:0|node-x|None\n"
                ),
                stderr="",
            )
        raise AssertionError("a present sacct row must not fall back to squeue")

    with pytest.raises(ValueError, match="does not match"):
        recover_configured_submission(
            config,
            jobs,
            receipt_path=receipt,
            job_id="5101",
            runner=wrong_identity,
        )
    assert receipt.read_bytes() == before


def test_schema_v2_sbatch_failure_requires_explicit_retry(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    calls = []

    def preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def rejected(command, **kwargs):
        calls.append(tuple(command))
        raise subprocess.CalledProcessError(9, command, stderr="rejected")

    with pytest.raises(subprocess.CalledProcessError):
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=rejected,
            preflight_runner=preflight,
        )
    assert (
        _execute_candidate_runtime(
            config,
            jobs,
            submit=True,
            receipt_path=receipt,
            runner=rejected,
            preflight_runner=preflight,
        )
        == ()
    )
    assert len(calls) == 1

    def accepted(command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="6001\n", stderr="")

    assert _execute_candidate_runtime(
        config,
        jobs,
        submit=True,
        receipt_path=receipt,
        runner=accepted,
        preflight_runner=preflight,
        retry_submission_failures=True,
    ) == ("6001",)
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "submitting",
        "submission_failed",
        "submitting",
        "submitted",
    ]
    assert rows[1]["scheduler_reason"].startswith("CalledProcessError:")
    assert rows[-1]["attempt"] == 2
    assert rows[-1]["authorization"] == "retry_submission_failure"


def test_schema_v2_success_with_ambiguous_job_id_stays_dangling(tmp_path):
    config = _write_candidate_config(tmp_path)
    jobs = plan_freeze_jobs(config)
    receipt = canonical_runtime_receipt_path(config, "freeze")
    real_calls = []

    def preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ambiguous_success(command, **kwargs):
        real_calls.append(tuple(command))
        return SimpleNamespace(
            returncode=0,
            stdout="7001\nwarning-after-id\n",
            stderr="",
        )

    with pytest.raises(ValueError, match="non-array Slurm job id"):
        _execute_candidate_runtime(
            config,
            jobs,
            receipt_path=receipt,
            runner=ambiguous_success,
            preflight_runner=preflight,
        )
    assert [json.loads(line)["event"] for line in receipt.read_text().splitlines()] == [
        "submitting"
    ]
    with pytest.raises(RuntimeError, match="unresolved schema-v2 submission intent"):
        _execute_candidate_runtime(
            config,
            jobs,
            receipt_path=receipt,
            runner=ambiguous_success,
            preflight_runner=preflight,
            retry_submission_failures=True,
        )
    assert real_calls == [jobs[0].command]


def test_lock_creation_requires_explicit_complete_artifacts_and_immutable_payload(
    tmp_path,
):
    config = _write_config(tmp_path)
    with pytest.raises(ValueError, match="exactly once"):
        build_campaign_lock(config, [])

    artifact = _write_artifact(tmp_path)
    lock = build_campaign_lock(config, [artifact])
    lock_path, _ = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    with pytest.raises(FileExistsError, match="already exists"):
        write_campaign_lock(lock, lock_path)

    payload = next(artifact.parent.joinpath("samples").glob("*.npz"))
    payload.write_bytes(payload.read_bytes() + b"tamper")
    jobs = plan_score_jobs(config, lock_path)
    command = list(jobs[0].command)
    wrapper_index = command.index("scripts/slurm/score_binary_simulation.sbatch")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_simulation(parse_score_args(command[wrapper_index + 1 :]))


def test_score_plan_rejects_a_truncated_or_retargeted_campaign_lock(tmp_path):
    config, _, lock_path, _ = _locked_campaign(tmp_path)
    lock = json.loads(lock_path.read_text())

    truncated = tmp_path / "truncated.lock.json"
    lock["artifacts"] = []
    truncated.write_text(json.dumps(lock) + "\n")
    with pytest.raises(ValueError, match="non-empty"):
        plan_score_jobs(config, truncated)

    lock = json.loads(lock_path.read_text())
    lock["paths"]["simulation_output_root"] = str(tmp_path / "retargeted")
    retargeted = tmp_path / "retargeted.lock.json"
    retargeted.write_text(json.dumps(lock) + "\n")
    with pytest.raises(ValueError, match="different output paths"):
        plan_score_jobs(config, retargeted)


def test_lock_rejects_an_artifact_with_the_wrong_predeclared_cohort_size(tmp_path):
    config = _write_config(tmp_path)
    config.data["conditions"][0]["expected_num_samples"] = 3
    artifact = _write_artifact(tmp_path)
    with pytest.raises(ValueError, match="predeclares 3"):
        build_campaign_lock(config, [artifact])


def test_main_config_has_16_conditions_and_48_independent_simulations():
    repository = Path(__file__).resolve().parents[1]
    config = load_config(repository / "configs/binary_midpoint_main.json")
    assert len(config.data["conditions"]) == 16
    protocol = config.data["protocol"]
    assert (
        len(config.data["conditions"])
        * len(protocol["gamma_values"])
        * len(protocol["m_values"])
        * len(protocol["seeds"])
        == 48
    )
    assert len(plan_freeze_jobs(config)) == 16
    assert {job.key[-1] for job in plan_freeze_jobs(config)} == {
        "saffo-a100,apollo_agate"
    }


def test_checked_in_v2_main_campaign_is_isolated_and_plans_independent_jobs(
    monkeypatch,
):
    repository = Path(__file__).resolve().parents[1]
    v1 = load_config(repository / "configs/binary_midpoint_main.json")
    v2 = load_config(repository / "configs/binary_midpoint_main_v2.json")
    gpu_requests = ("saffo-a100,apollo_agate", "apollo_agate,saffo-a100")
    cpu_requests = (
        "amdsmall,agsmall,msismall,saffo-2tb",
        "agsmall,msismall,amdsmall,saffo-2tb",
        "msismall,amdsmall,agsmall,saffo-2tb",
    )

    assert v2.data["config_schema_version"] == 2
    assert v2.data["campaign_id"] == "binary-midpoint-main-v2"
    assert v2.data["execution_policy"] == "scientific-input-locked"
    assert v2.data["data_root"] == "data"
    assert set(v2.data["scientific_input_lock"]) == {"path", "sha256"}
    assert v2.data["campaign_id"] != v1.data["campaign_id"]
    assert v2.data["gpu_partition_candidates"] == gpu_requests[0].split(",")
    assert v2.data["cpu_partition_candidates"] == cpu_requests[0].split(",")
    assert v2.data["protocol"] == v1.data["protocol"]
    assert v2.data["conditions"] == v1.data["conditions"]
    assert v2.data["estimator_spec"] == v1.data["estimator_spec"]
    assert v2.data["paths"] == {
        "artifact_output_root": "outputs/binary_midpoint_main_v2/artifacts",
        "common_output_root": "outputs/binary_midpoint_main_v2/common_scores",
        "simulation_output_root": "outputs/binary_midpoint_main_v2/simulations",
        "assembly_output_root": "outputs/binary_midpoint_main_v2/assembled",
    }
    assert set(v2.data["paths"].values()).isdisjoint(v1.data["paths"].values())

    from selectseg.scientific_inputs import load_root_lock

    science = v2.data["scientific_input_lock"]
    science_path = repository / science["path"]
    science_binding = load_root_lock(science_path, expected_sha256=science["sha256"])
    monkeypatch.setattr(
        submit_binary_simulations,
        "_scientific_plan_binding",
        lambda candidate, *, mode: (science_path, science["sha256"], science_binding),
    )

    freeze_jobs = plan_freeze_jobs(v2)
    assert len(freeze_jobs) == 16
    assert len({job.key for job in freeze_jobs}) == len(freeze_jobs)
    for index, job in enumerate(freeze_jobs):
        command = list(job.command)
        assert command[command.index("--partition") + 1] == gpu_requests[index % 2]
        assert command.count("scripts/slurm/freeze_binary_maps.sbatch") == 1
        assert "--array" not in command
        assert not any(token.startswith("--array=") for token in command)

    lock = {
        "campaign_id": v2.data["campaign_id"],
        "protocol": v2.data["protocol"],
        "estimator": {
            "spec_path": v2.data["estimator_spec"],
            "spec_sha256": "e" * 64,
        },
        "paths": v2.data["paths"],
        "artifacts": [
            {
                "dataset": condition["dataset"],
                "condition": condition["condition"],
                "manifest_path": (
                    f"artifacts/{condition['dataset']}/"
                    f"{condition['condition']}/manifest.json"
                ),
                "manifest_sha256": hashlib.sha256(
                    f"{condition['dataset']}/{condition['condition']}".encode()
                ).hexdigest(),
            }
            for condition in v2.data["conditions"]
        ],
    }
    lock_path = repository / "unused-v2-planner-test.lock.json"

    def load_test_lock(path, *, config=None):
        assert Path(path) == lock_path
        assert config is v2
        return lock_path, "f" * 64, lock

    monkeypatch.setattr(submit_binary_simulations, "load_campaign_lock", load_test_lock)
    common_jobs = plan_common_jobs(v2, lock_path)
    score_jobs = plan_score_jobs(v2, lock_path)
    diagnose_jobs = plan_diagnose_jobs(v2, lock_path)

    assert len(common_jobs) == 16
    assert len(score_jobs) == 48
    assert len(diagnose_jobs) == 16
    for jobs in (common_jobs, score_jobs, diagnose_jobs):
        assert len({job.key for job in jobs}) == len(jobs)
        for index, job in enumerate(jobs):
            command = list(job.command)
            assert command.count("--partition") == 1
            request = command[command.index("--partition") + 1]
            assert request == cpu_requests[index % 3]
            assert set(request.split(",")) == set(cpu_requests[0].split(","))
            assert request.endswith(",saffo-2tb")
            assert "--array" not in command
            assert not any(token.startswith("--array=") for token in command)

    score_identities = {
        (
            command[command.index("--artifact-manifest") + 1],
            command[command.index("--gamma") + 1],
            command[command.index("--m") + 1],
            command[command.index("--seed") + 1],
        )
        for command in (list(job.command) for job in score_jobs)
    }
    assert len(score_identities) == len(score_jobs)
    assert all(
        job.command.count("scripts/slurm/score_binary_simulation.sbatch") == 1
        for job in score_jobs
    )


def test_repository_module_clis_work_without_an_editable_install():
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for module in (
        "selectseg.score_binary_common",
        "scripts.submit_binary_simulations",
        "scripts.assemble_binary_simulations",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_actual_scorer_shards_assemble_without_repeating_inference(tmp_path):
    config, _, lock_path, _ = _locked_campaign(tmp_path)
    common_job = plan_common_jobs(config, lock_path)[0]
    common_command = list(common_job.command)
    common_wrapper = common_command.index("scripts/slurm/score_binary_common.sbatch")
    _, common_manifest = run_common(
        parse_common_args(common_command[common_wrapper + 1 :])
    )
    partials = []
    for job in plan_score_jobs(config, lock_path):
        command = list(job.command)
        wrapper_index = command.index("scripts/slurm/score_binary_simulation.sbatch")
        _, manifest_path = run_simulation(
            parse_score_args(command[wrapper_index + 1 :])
        )
        partials.append(manifest_path)

    assembly_jobs = plan_assemble_jobs(config, lock_path)
    assert len(assembly_jobs) == 1
    assembly_job = assembly_jobs[0]
    command = list(assembly_job.command)
    assert assembly_job.phase == "assemble"
    assert assembly_job.key == ("pet", "clipseg-general", "agsmall")
    assert "--array" not in command
    assert command.count("scripts/slurm/assemble_binary_simulations.sbatch") == 1
    assert command[command.index("--partition") + 1] == "agsmall"
    assert command[command.index("--account") + 1] == "ssafo"
    assert command[command.index("--campaign-lock") + 1] == str(lock_path)
    assert command[command.index("--common") + 1] == str(common_manifest)
    assert command.count("--input") == 3
    planned_inputs = [
        command[index + 1] for index, value in enumerate(command) if value == "--input"
    ]
    assert planned_inputs == [str(path) for path in partials]
    assert (
        command[command.index("--output-root") + 1]
        == (config.data["paths"]["assembly_output_root"])
    )

    target = assemble(
        campaign_lock=lock_path,
        common=common_manifest,
        inputs=partials,
        output_root=tmp_path / "assembled-integration",
    )
    condition = load_condition(target / "records.jsonl")
    assert tuple(condition.manifest["score_fields"]) == FINAL_SCORE_FIELDS
    assert len(condition.rows) == 2
