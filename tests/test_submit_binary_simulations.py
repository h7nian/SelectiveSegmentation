"""Tests for deterministic one-simulation-per-job submission planning."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.submit_binary_simulations import (
    build_campaign_lock,
    execute_plan,
    load_config,
    plan_assemble_jobs,
    plan_common_jobs,
    plan_diagnose_jobs,
    plan_freeze_jobs,
    plan_score_jobs,
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
        gpu_partition_candidates=["saffo-a100", "apollo_agate"],
        cpu_partition_candidates=[
            "saffo-2tb",
            "agsmall",
            "amdsmall",
            "msismall",
        ],
    )
    path = tmp_path / "campaign-candidates-v2.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return load_config(path)


def _write_artifact(tmp_path):
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
    )


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
    cpu_request = "saffo-2tb,agsmall,amdsmall,msismall"

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
        assert command[command.index("--partition") + 1] == cpu_request
        assert "--array" not in command
        assert not any(token.startswith("--array=") for token in command)
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
    assert assemble_command[assemble_command.index("--partition") + 1] == cpu_request
    assert assemble_command.count("scripts/slurm/assemble_binary_simulations.sbatch") == 1
    assert assemble_command[assemble_command.index("--common") + 1] == str(
        common_manifest
    )
    assert "--array" not in assemble_command
    assert not any(token.startswith("--array=") for token in assemble_command)


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
            {},
            "must be provided together",
        ),
        (
            1,
            {
                "gpu_partition_candidates": ["saffo-a100", "apollo_agate"],
                "cpu_partition_candidates": [
                    "saffo-2tb",
                    "agsmall",
                    "amdsmall",
                    "msismall",
                ],
            },
            "require config_schema_version 2",
        ),
        (
            2,
            {"gpu_partition_candidates": ["saffo-a100", "apollo_agate"]},
            "must be provided together",
        ),
        (
            2,
            {
                "gpu_partition_candidates": ["apollo_agate", "saffo-a100"],
                "cpu_partition_candidates": [
                    "saffo-2tb",
                    "agsmall",
                    "amdsmall",
                    "msismall",
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
    assert execute_plan(
        diagnostic_jobs,
        submit=True,
        receipt_path=diagnostic_receipt,
        runner=runner,
    ) == ()
    diagnostic_rows = [
        json.loads(line) for line in diagnostic_receipt.read_text().splitlines()
    ]
    assert [row["phase"] for row in diagnostic_rows] == ["diagnose", "diagnose"]
    assert [row["status"] for row in diagnostic_rows] == [
        "submitting",
        "submitted",
    ]


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
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--input"
    ]
    assert planned_inputs == [str(path) for path in partials]
    assert command[command.index("--output-root") + 1] == (
        config.data["paths"]["assembly_output_root"]
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
