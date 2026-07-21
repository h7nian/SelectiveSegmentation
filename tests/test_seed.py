import copy
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from scripts.maintenance import finalize_seed as scheduler_ledger
from scripts.submit import seed as submit
from scripts.analyze.main import CONTRASTS
from scripts.analyze.seed import _gate_c, _strict_cohort_join
from scripts.render.seed import render_table
from selectseg.seed import downstream
from selectseg.seed import extension as seedext
from selectseg.artifacts import sample_id_sha256, write_binary_artifact


V1_GOLDEN_FILES = {
    "configs/auxiliary/binary_seed_extension-v1.json": (
        "8c861e6421270fc16378ca5408db6abaedd47cf0356b0cb2d82f86f7d2d76696"
    ),
    "configs/auxiliary/binary_seed_extension-v1.lock.json": (
        "3fb7b721a4b54e467383c03b03168166a1d2e9f197d3a26eade0161df931deed"
    ),
}
V1_CHECKPOINT_LOCK_BINDING = {
    "path": Path("outputs/binary_seed_campaign/checkpoints.lock.json"),
    "sha256": "9f5db0ff1c9c6b6abd49fdb4f0b40b4821f7e5c78f43f438ea995ba06968c79f",
}


def _binding():
    """Load immutable v1 metadata without pretending current source is v1 source.

    The public v1 lock deliberately keeps the hashes of the code that actually
    produced the completed seed extension.  The repository has since evolved,
    so production ``load_spec_lock`` must fail closed on that source drift.
    Unit tests for the frozen plan and metadata still need the immutable
    binding; they bypass only the current-worktree source-byte comparison while
    retaining the lock hash, schema, paths, model files, and all other checks.
    """

    verifier = seedext._verify_file_binding

    def verify_historical_binding(binding, *, location):
        if location.startswith("spec_lock.source_files["):
            return Path(binding["path"])
        return verifier(binding, location=location)

    with mock.patch.object(
        seedext, "_verify_file_binding", side_effect=verify_historical_binding
    ):
        return seedext.load_spec_lock()


def _public_v1_golden_binding():
    spec_path = Path("configs/auxiliary/binary_seed_extension-v1.json")
    return {
        "path": Path("configs/auxiliary/binary_seed_extension-v1.lock.json"),
        "sha256": V1_GOLDEN_FILES[
            "configs/auxiliary/binary_seed_extension-v1.lock.json"
        ],
        "spec": json.loads(spec_path.read_text(encoding="utf-8")),
    }


def _canonical_plan_sha256(jobs):
    plan = [
        {
            "phase": job.phase,
            "key": list(job.key),
            "command": list(job.command),
        }
        for job in jobs
    ]
    payload = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_v1_gpu_plan(jobs, *, phase, wrapper, expected_identities):
    assert len(jobs) == 20
    assert {job.phase for job in jobs} == {phase}
    assert {job.key for job in jobs} == expected_identities
    assert len({job.key for job in jobs}) == 20
    for job in jobs:
        assert job.command.count(wrapper) == 1
        assert not any(
            token == "--array" or token.startswith("--array=")
            for token in job.command
        )


def test_v1_public_files_match_sealed_sha256():
    observed = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in V1_GOLDEN_FILES
    }

    assert observed == V1_GOLDEN_FILES


def test_v1_production_loader_fails_closed_after_source_evolution():
    lock = json.loads(
        Path("configs/auxiliary/binary_seed_extension-v1.lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]
        for row in lock["source_files"]
    )
    with pytest.raises(ValueError, match="locked file hash mismatch"):
        seedext.load_spec_lock()


def test_v1_train_and_freeze_plans_match_sealed_golden_replay():
    binding = _public_v1_golden_binding()
    expected_identities = {
        (
            experiment["dataset"]["name"],
            experiment["model"]["name"],
            experiment["training_seed"],
            experiment["gpu_profile"]["partition"],
        )
        for experiment in seedext.iter_experiments(binding["spec"])
    }
    train_jobs = submit.plan_training_jobs(binding)
    checkpoint_binding = {
        "path": V1_CHECKPOINT_LOCK_BINDING["path"],
        "sha256": V1_CHECKPOINT_LOCK_BINDING["sha256"],
    }
    assert checkpoint_binding["path"].as_posix() == binding["spec"]["paths"][
        "checkpoint_lock"
    ]
    freeze_jobs = submit.plan_freeze_jobs(binding, checkpoint_binding)

    _assert_v1_gpu_plan(
        train_jobs,
        phase="seed_train",
        wrapper="scripts/slurm/run.sbatch",
        expected_identities=expected_identities,
    )
    _assert_v1_gpu_plan(
        freeze_jobs,
        phase="seed_freeze",
        wrapper="scripts/slurm/run.sbatch",
        expected_identities=expected_identities,
    )
    assert all("selectseg.seed.extension" in job.command for job in train_jobs)
    assert all("selectseg.seed.extension" in job.command for job in freeze_jobs)


def _valid_train_record(binding, experiment):
    return {
        "train_record_schema_version": seedext.TRAIN_RECORD_SCHEMA_VERSION,
        "auxiliary_id": seedext.EXPECTED_AUXILIARY_ID,
        "created_utc": "2026-07-19T00:00:00+00:00",
        "spec_lock": {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
        },
        "dataset": experiment["dataset"]["name"],
        "model": experiment["model"]["name"],
        "condition": experiment["model"]["condition"],
        "training_seed": experiment["training_seed"],
        "gpu_profile": copy.deepcopy(experiment["gpu_profile"]),
        "runtime": {
            "slurm_job_id": "123456",
            "partition": experiment["gpu_profile"]["partition"],
            "node": "unit-test-node",
            "cuda_device": "NVIDIA A100-SXM4-40GB",
            "environment": copy.deepcopy(binding["spec"]["environment"]),
        },
        "outputs": {},
    }


def _install_train_record_stubs(monkeypatch, record):
    monkeypatch.setattr(seedext, "_train_output_dir", lambda *args: Path("train"))
    monkeypatch.setattr(seedext, "_load_json", lambda path: record)
    monkeypatch.setattr(seedext, "_training_command", lambda *args: ["train"])
    monkeypatch.setattr(
        seedext,
        "_validate_training_outputs",
        lambda binding, experiment, command: {},
    )


def test_load_train_record_accepts_locked_metadata(monkeypatch):
    binding = _binding()
    experiment = next(iter(seedext.iter_experiments(binding["spec"])))
    record = _valid_train_record(binding, experiment)
    _install_train_record_stubs(monkeypatch, record)

    _, loaded, outputs = seedext._load_train_record(binding, experiment)

    assert loaded == record
    assert outputs == {}


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("auxiliary_id", "auxiliary_id"),
        ("condition", "unexpected condition"),
        ("gpu_partition", "locked GPU profile"),
        ("gpu_account", "locked GPU profile"),
        ("gpu_gres", "locked GPU profile"),
        ("runtime_missing", "must contain exactly"),
        ("runtime_extra", "must contain exactly"),
        ("runtime_partition", "runtime partition"),
        ("runtime_environment", "runtime environment"),
        ("runtime_job", "slurm_job_id"),
        ("runtime_node", "node"),
        ("runtime_device_empty", "cuda_device"),
        ("runtime_device_wrong", "not an A100"),
    ],
)
def test_load_train_record_rejects_metadata_tampering(monkeypatch, case, match):
    binding = _binding()
    experiment = next(iter(seedext.iter_experiments(binding["spec"])))
    record = _valid_train_record(binding, experiment)
    if case == "auxiliary_id":
        record["auxiliary_id"] = "retargeted"
    elif case == "condition":
        record["condition"] = "retargeted"
    elif case.startswith("gpu_"):
        record["gpu_profile"][case.removeprefix("gpu_")] = "retargeted"
    elif case == "runtime_missing":
        record["runtime"].pop("node")
    elif case == "runtime_extra":
        record["runtime"]["unexpected"] = "value"
    elif case == "runtime_partition":
        record["runtime"]["partition"] = "retargeted"
    elif case == "runtime_environment":
        record["runtime"]["environment"]["torch"] = "retargeted"
    elif case == "runtime_job":
        record["runtime"]["slurm_job_id"] = ""
    elif case == "runtime_node":
        record["runtime"]["node"] = None
    elif case == "runtime_device_empty":
        record["runtime"]["cuda_device"] = " "
    elif case == "runtime_device_wrong":
        record["runtime"]["cuda_device"] = "NVIDIA H100"
    else:
        raise AssertionError(f"unhandled case {case}")
    _install_train_record_stubs(monkeypatch, record)

    with pytest.raises(ValueError, match=match):
        seedext._load_train_record(binding, experiment)


@pytest.mark.parametrize("model", ["clipseg", "deeplabv3"])
def test_run_freeze_verifies_locked_base_model_before_inference(monkeypatch, model):
    binding = _binding()
    experiment = next(
        entry
        for entry in seedext.iter_experiments(binding["spec"])
        if entry["model"]["name"] == model
    )
    monkeypatch.setattr(seedext, "_verify_slurm_context", lambda profile: None)
    monkeypatch.setattr(seedext, "_verify_environment", lambda spec: None)

    def checked(*args):
        raise RuntimeError("locked base model checked")

    monkeypatch.setattr(seedext, "_verify_base_model_files", checked)

    with pytest.raises(RuntimeError, match="locked base model checked"):
        seedext.run_freeze(
            binding,
            {},
            dataset=experiment["dataset"]["name"],
            model=model,
            seed=experiment["training_seed"],
            expected_partition=experiment["gpu_profile"]["partition"],
        )


def test_static_spec_is_strict_and_grid_is_exactly_twenty():
    binding = _binding()
    experiments = list(seedext.iter_experiments(binding["spec"]))

    assert len(experiments) == 5 * 2 * 2 == 20
    assert (
        len(
            {
                (
                    experiment["dataset"]["name"],
                    experiment["model"]["name"],
                    experiment["training_seed"],
                )
                for experiment in experiments
            }
        )
        == 20
    )
    assert Counter(
        experiment["gpu_profile"]["partition"] for experiment in experiments
    ) == {"saffo-a100": 10, "apollo_agate": 10}

    # Seed, architecture, and dataset are each balanced across the two queues;
    # seed is not globally confounded with hardware assignment.
    for seed in (1, 2):
        assert Counter(
            experiment["gpu_profile"]["partition"]
            for experiment in experiments
            if experiment["training_seed"] == seed
        ) == {"saffo-a100": 5, "apollo_agate": 5}
    for model in ("clipseg", "deeplabv3"):
        assert Counter(
            experiment["gpu_profile"]["partition"]
            for experiment in experiments
            if experiment["model"]["name"] == model
        ) == {"saffo-a100": 5, "apollo_agate": 5}


def test_training_plan_is_one_explicit_gpu_job_per_experiment():
    jobs = submit.plan_training_jobs(_binding())

    assert len(jobs) == 20
    assert {job.phase for job in jobs} == {"seed_train"}
    for job in jobs:
        command = job.command
        assert command[0:2] == ("sbatch", "--parsable")
        assert "--array" not in command
        assert command.count("--partition") == 1
        assert command.count("--account") == 1
        assert command.count("--gres") == 1
        assert command[command.index("--account") + 1] == "ssafo"
        assert command[command.index("--gres") + 1] == "gpu:a100:1"
        assert command.count("--training-seed") == 1
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert submit.CPU_PARTITION_REQUEST not in command


def test_future_cpu_candidate_pool_is_canonical_and_one_job_stays_one_job():
    assert submit.CPU_PARTITION_CANDIDATES == (
        "saffo-2tb",
        "agsmall",
        "amdsmall",
        "msismall",
    )
    assert submit.CPU_PARTITION_REQUEST == (
        "saffo-2tb,agsmall,amdsmall,msismall"
    )
    original = submit.PlannedJob(
        phase="seed_assemble",
        key=(1, "pet", "clipseg-target", "agsmall"),
        command=(
            "sbatch",
            "--parsable",
            "--job-name",
            "one-experiment",
            "--partition",
            "agsmall",
            "wrapper.sbatch",
        ),
    )

    planned = submit._with_cpu_partition_candidates(original)

    assert planned.phase == original.phase
    assert planned.key[:-1] == original.key[:-1]
    assert planned.key[-1] == submit.CPU_PARTITION_REQUEST
    assert planned.command[planned.command.index("--partition") + 1] == (
        submit.CPU_PARTITION_REQUEST
    )
    assert planned.command.count("wrapper.sbatch") == 1
    assert "--array" not in planned.command
    assert original.key[-1] == "agsmall"


def test_dry_run_prints_exactly_twenty_and_never_calls_scheduler(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not call Slurm")

    monkeypatch.setattr(submit, "load_spec_lock", lambda *args, **kwargs: _binding())
    monkeypatch.setattr(subprocess, "run", forbidden)
    submit.main(["--phase", "train"])
    output = capsys.readouterr().out
    assert output.count("scripts/slurm/run.sbatch") == 20
    assert "partition_distribution=saffo-a100:10,apollo_agate:10" in output
    assert "planned_jobs=20 submitted_jobs=0 skipped_jobs=0" in output


def test_seed_analyzer_defaults_to_clean_canonical_v3():
    args = submit.parse_args(["--phase", "analyze"])
    assert args.canonical_analysis == ("outputs/binary_final_v3_analysis/analysis.json")


def test_lock_tampering_fails_before_planning(tmp_path):
    source = Path(seedext.DEFAULT_SPEC_LOCK)
    tampered = tmp_path / source.name
    payload = json.loads(source.read_text())
    payload["scheduler_validation"]["profiles"][0]["gres"] = "gpu:1"
    tampered.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="spec-lock hash mismatch"):
        seedext.load_spec_lock(
            tampered, expected_sha256=seedext.DEFAULT_SPEC_LOCK_SHA256
        )


def _fake_checkpoint_binding(binding, tmp_path, *, portable=False):
    checkpoints = []
    for index, experiment in enumerate(seedext.iter_experiments(binding["spec"])):
        checkpoint = (
            Path(f"checkpoint-{index}.pt")
            if portable
            else tmp_path / f"checkpoint-{index}.pt"
        )
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        dummy = (
            Path(f"metadata-{index}.json")
            if portable
            else tmp_path / f"metadata-{index}.json"
        )
        dummy.write_text("{}\n")
        checkpoints.append(
            {
                "dataset": experiment["dataset"]["name"],
                "model": experiment["model"]["name"],
                "condition": experiment["model"]["condition"],
                "training_seed": experiment["training_seed"],
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": seedext._sha256(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "train_config_path": dummy.as_posix(),
                "train_config_sha256": seedext._sha256(dummy),
                "history_path": dummy.as_posix(),
                "history_sha256": seedext._sha256(dummy),
                "train_record_path": dummy.as_posix(),
                "train_record_sha256": seedext._sha256(dummy),
            }
        )
    checkpoint_path = (
        Path("checkpoints.lock.json")
        if portable
        else tmp_path / "checkpoints.lock.json"
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoint_lock_schema_version": 1,
                "auxiliary_id": seedext.EXPECTED_AUXILIARY_ID,
                "created_utc": "2026-07-19T00:00:00+00:00",
                "spec_lock": {
                    "path": binding["path"].as_posix(),
                    "sha256": binding["sha256"],
                },
                "checkpoints": checkpoints,
            }
        )
        + "\n"
    )
    return seedext.load_checkpoint_lock(binding, checkpoint_path, verify_files=False)


def test_freeze_plan_is_gated_and_has_twenty_independent_jobs(tmp_path, monkeypatch):
    binding = _binding()
    checkpoint_binding = _fake_checkpoint_binding(binding, tmp_path)
    jobs = submit.plan_freeze_jobs(binding, checkpoint_binding)

    assert len(jobs) == 20
    assert {job.phase for job in jobs} == {"seed_freeze"}
    assert Counter(job.key[-1] for job in jobs) == {
        "saffo-a100": 10,
        "apollo_agate": 10,
    }
    for job in jobs:
        command = job.command
        assert "--array" not in command
        assert command.count("--checkpoint-lock") == 1
        assert command.count("--expected-checkpoint-lock-sha256") == 1
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert submit.CPU_PARTITION_REQUEST not in command

    monkeypatch.setattr(submit, "load_spec_lock", lambda *args, **kwargs: binding)
    with pytest.raises(FileNotFoundError, match="gated"):
        submit.main(
            [
                "--phase",
                "freeze",
                "--checkpoint-lock",
                str(tmp_path / "absent.json"),
                "--expected-checkpoint-lock-sha256",
                "0" * 64,
            ]
        )
    with pytest.raises(ValueError, match="requires --expected-checkpoint"):
        submit.main(["--phase", "freeze"])


def test_checkpoint_lock_rejects_wrong_condition_and_self_consistent_retarget(
    monkeypatch, tmp_path
):
    binding = _binding()
    checkpoint_binding = _fake_checkpoint_binding(binding, tmp_path)
    original_rows = copy.deepcopy(checkpoint_binding["lock"]["checkpoints"])

    wrong_condition = tmp_path / "wrong-condition.json"
    payload = copy.deepcopy(checkpoint_binding["lock"])
    payload["checkpoints"][0]["condition"] = "clipseg-general"
    wrong_condition.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="unexpected condition"):
        seedext.load_checkpoint_lock(
            binding,
            wrong_condition,
            expected_sha256=seedext._sha256(wrong_condition),
            verify_files=False,
        )

    by_cell = {
        (row["dataset"], row["model"], row["training_seed"]): row
        for row in original_rows
    }

    def reconstruct_from_record(_binding, experiment):
        cell = (
            experiment["dataset"]["name"],
            experiment["model"]["name"],
            experiment["training_seed"],
        )
        row = by_cell[cell]
        outputs = {
            field: row[field]
            for field in (
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "train_config_path",
                "train_config_sha256",
                "history_path",
                "history_sha256",
            )
        }
        return Path(row["train_record_path"]), {}, outputs

    monkeypatch.setattr(seedext, "_load_train_record", reconstruct_from_record)
    rogue = tmp_path / "rogue.pt"
    rogue.write_bytes(b"self-consistent but not the locked training output")
    payload = copy.deepcopy(checkpoint_binding["lock"])
    payload["checkpoints"][0].update(
        checkpoint_path=rogue.as_posix(),
        checkpoint_sha256=seedext._sha256(rogue),
        checkpoint_size_bytes=rogue.stat().st_size,
    )
    retargeted = tmp_path / "retargeted.json"
    retargeted.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="immutable training record"):
        seedext.load_checkpoint_lock(
            binding,
            retargeted,
            expected_sha256=seedext._sha256(retargeted),
            verify_files=True,
        )


def test_checkpoint_lock_writer_requires_all_twenty_and_never_overwrites(
    monkeypatch, tmp_path
):
    binding = copy.deepcopy(_binding())
    destination = tmp_path / "checkpoints.lock.json"
    binding["spec"]["paths"]["checkpoint_lock"] = destination.as_posix()

    def fake_record(current_binding, experiment):
        index = list(seedext.iter_experiments(current_binding["spec"])).index(
            experiment
        )
        checkpoint = tmp_path / f"checkpoint-{index}.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        config = tmp_path / f"config-{index}.json"
        history = tmp_path / f"history-{index}.json"
        record = tmp_path / f"record-{index}.json"
        for path in (config, history, record):
            path.write_text("{}\n")
        outputs = {
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": seedext._sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "train_config_path": config.as_posix(),
            "train_config_sha256": seedext._sha256(config),
            "history_path": history.as_posix(),
            "history_sha256": seedext._sha256(history),
            "command": ["locked"],
        }
        return record, {}, outputs

    experiments = list(seedext.iter_experiments(binding["spec"]))
    for index in range(len(experiments)):
        (tmp_path / f"record-{index}.json").write_text("{}\n")
    expected_record_set_sha = seedext._training_record_set_sha256(
        [
            {
                "dataset": experiment["dataset"]["name"],
                "model": experiment["model"]["name"],
                "training_seed": experiment["training_seed"],
                "train_record_sha256": seedext._sha256(
                    tmp_path / f"record-{index}.json"
                ),
            }
            for index, experiment in enumerate(experiments)
        ]
    )

    monkeypatch.setattr(seedext, "_load_train_record", fake_record)
    with pytest.raises(TypeError, match="expected_training_record_set_sha256"):
        seedext.write_checkpoint_lock(binding, destination)
    with pytest.raises(ValueError, match="changed after scheduler closure"):
        seedext.write_checkpoint_lock(
            binding,
            destination,
            expected_training_record_set_sha256="f" * 64,
        )
    assert not destination.exists()

    seedext.write_checkpoint_lock(
        binding,
        destination,
        expected_training_record_set_sha256=expected_record_set_sha,
    )
    payload = json.loads(destination.read_text())
    assert len(payload["checkpoints"]) == 20
    with pytest.raises(FileExistsError, match="overwrite"):
        seedext.write_checkpoint_lock(
            binding,
            destination,
            expected_training_record_set_sha256=expected_record_set_sha,
        )


def test_training_record_set_digest_matches_scheduler_contract():
    rows = [
        {
            "dataset": f"dataset-{index}",
            "model": "model",
            "training_seed": 1 + index % 2,
            "train_record_sha256": f"{index + 1:064x}",
        }
        for index in range(20)
    ]
    scheduler_rows = [
        {
            **row,
            "receipt_job_id": str(900000 + index),
            "record_slurm_job_id": str(900000 + index),
        }
        for index, row in enumerate(rows)
    ]

    assert seedext._training_record_set_sha256(rows) == (
        scheduler_ledger._record_set_sha256(scheduler_rows)
    )


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_json_loader_rejects_symlink_path_components(tmp_path, kind):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "payload.json"
    target.write_text("{}\n")
    if kind == "leaf":
        source = tmp_path / "payload-link.json"
        source.symlink_to(target)
    else:
        linked_dir = tmp_path / "linked"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        source = linked_dir / target.name

    with pytest.raises(ValueError, match="symlink path components"):
        seedext._load_json(source)


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_atomic_writer_rejects_symlink_path_components(tmp_path, kind):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    if kind == "leaf":
        target = real_dir / "existing.json"
        target.write_text("{}\n")
        destination = tmp_path / "output.json"
        destination.symlink_to(target)
    else:
        linked_dir = tmp_path / "linked"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        destination = linked_dir / "output.json"

    with pytest.raises(ValueError, match="symlink path components"):
        seedext._atomic_write_new(destination, {"new": True})
    if kind == "ancestor":
        assert not (real_dir / "output.json").exists()


def test_scheduler_preflight_checks_both_profiles_and_fails_closed(monkeypatch):
    jobs = submit.plan_training_jobs(_binding())
    calls = []

    def accepted(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "accepted", "")

    monkeypatch.setattr(submit.subprocess, "run", accepted)
    submit._scheduler_preflight(jobs)
    assert len(calls) == 2
    assert all("--test-only" in command for command in calls)
    assert all("--parsable" not in command for command in calls)

    def rejected(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "invalid account")

    monkeypatch.setattr(submit.subprocess, "run", rejected)
    with pytest.raises(RuntimeError, match="no jobs submitted"):
        submit._scheduler_preflight(jobs)


def test_cpu_candidate_preflight_checks_combined_request_and_fails_closed(
    monkeypatch,
):
    job = submit._with_cpu_partition_candidates(
        submit.PlannedJob(
            phase="seed_assemble",
            key=(1, "pet", "clipseg-target", "agsmall"),
            command=(
                "sbatch",
                "--parsable",
                "--job-name",
                "one-experiment",
                "--partition",
                "agsmall",
                "wrapper.sbatch",
            ),
        )
    )
    calls = []

    def accepted(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "accepted", "")

    monkeypatch.setattr(submit.subprocess, "run", accepted)
    submit._cpu_candidate_preflight((job,))
    assert len(calls) == 1
    assert "--test-only" in calls[0]
    assert "--parsable" not in calls[0]
    assert calls[0][calls[0].index("--partition") + 1] == (
        submit.CPU_PARTITION_REQUEST
    )

    def rejected(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "not eligible")

    monkeypatch.setattr(submit.subprocess, "run", rejected)
    with pytest.raises(RuntimeError, match="no jobs submitted"):
        submit._cpu_candidate_preflight((job,))


def test_wrappers_defer_gpu_profile_to_submitter_and_disable_requeue():
    wrapper = Path("scripts/slurm/run.sbatch").read_text()
    assert "#SBATCH --partition" not in wrapper
    assert "#SBATCH --account" not in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert 'exec "$@"' in wrapper


def test_downstream_design_is_isolated_and_does_not_expand_primary_validator():
    design = submit.downstream_job_design(_binding())
    assert len(design["cells"]) == 20
    assert design["common"]["jobs"] == 20
    assert design["m_score"]["jobs"] == 60
    assert design["assemble"]["jobs"] == 20
    assert design["diagnose"]["jobs"] == 20
    assert all(
        "binary_seed_" in design[phase]["output_root"]
        for phase in (
            "common",
            "m_score",
            "assemble",
            "diagnose",
        )
    )
    assert "do not broaden" in design["guard"]


def _write_fake_freezes(binding, checkpoint_binding):
    sample_ids = ["case-a", "case-b"]
    probability = np.array([[0.1, 0.9], [0.6, 0.2]], dtype=np.float32)
    truth = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    for dataset in binding["spec"]["datasets"]:
        dataset["eval_count"] = len(sample_ids)
        dataset["eval_sample_id_sha256"] = sample_id_sha256(sample_ids)
    records = []
    for experiment in seedext.iter_experiments(binding["spec"]):
        dataset = experiment["dataset"]["name"]
        model = experiment["model"]["name"]
        seed = experiment["training_seed"]
        checkpoint = seedext._checkpoint_entry(checkpoint_binding, dataset, model, seed)
        manifest_path = write_binary_artifact(
            binding["spec"]["paths"]["artifact_root"],
            dataset=dataset,
            condition=experiment["model"]["condition"],
            model=model,
            split=experiment["dataset"]["eval_split"],
            class_index=1,
            class_name="foreground",
            checkpoint={
                "path": checkpoint["checkpoint_path"],
                "sha256": checkpoint["checkpoint_sha256"],
                "size_bytes": checkpoint["checkpoint_size_bytes"],
            },
            base_model={"name": model, "source": "unit-test"},
            source_sha256=hex(seed)[2:].zfill(64),
            environment={
                "packages": {
                    "python": "test",
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
            preprocessing=binding["spec"]["freeze_preprocessing"],
            cohort="unit-test cohort",
            sample_ids=sample_ids,
            samples=[
                (sample_id, probability.copy(), truth.copy())
                for sample_id in sample_ids
            ],
            command=["unit-test-freeze"],
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        artifact = downstream.load_binary_artifact(
            manifest_path, validate_payloads=False
        )
        record_path = downstream._freeze_record_path(binding, experiment)
        seedext._atomic_write_new(
            record_path,
            {
                "freeze_record_schema_version": 1,
                "auxiliary_id": seedext.EXPECTED_AUXILIARY_ID,
                "created_utc": "2026-07-19T00:00:00+00:00",
                "spec_lock": {
                    "path": binding["path"].as_posix(),
                    "sha256": binding["sha256"],
                },
                "checkpoint_lock": {
                    "path": checkpoint_binding["path"].as_posix(),
                    "sha256": checkpoint_binding["sha256"],
                },
                "dataset": dataset,
                "model": model,
                "condition": experiment["model"]["condition"],
                "training_seed": seed,
                "gpu_profile": experiment["gpu_profile"],
                "artifact_manifest_path": manifest_path.as_posix(),
                "artifact_manifest_sha256": artifact.manifest_sha256,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            },
        )
        records.append(record_path)
    return records


def _fake_downstream(tmp_path, monkeypatch):
    base_binding = _binding()
    estimator_path = Path(base_binding["lock"]["estimator_spec"]["path"]).resolve()
    monkeypatch.chdir(tmp_path)
    binding = copy.deepcopy(base_binding)
    binding["lock"]["estimator_spec"]["path"] = estimator_path.as_posix()
    paths = binding["spec"]["paths"]
    paths.update(
        {
            "train_root": "train",
            "checkpoint_lock": "checkpoints.lock.json",
            "artifact_root": "artifacts",
            "freeze_record_root": "freeze-records",
            "common_root": "common",
            "simulation_root": "simulations",
            "assembly_root": "assembled",
            "diagnostic_root": "diagnostics",
            "analysis_root": "analysis",
        }
    )
    checkpoint_binding = _fake_checkpoint_binding(binding, tmp_path, portable=True)
    records = _write_fake_freezes(binding, checkpoint_binding)
    destination = downstream._default_downstream_path(binding)
    downstream.write_downstream_lock(binding, checkpoint_binding, destination)
    monkeypatch.setattr(downstream, "load_spec_lock", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        downstream,
        "load_checkpoint_lock",
        lambda *args, **kwargs: checkpoint_binding,
    )
    loaded = downstream.load_downstream_lock(
        destination, expected_sha256=seedext._sha256(destination)
    )
    return binding, checkpoint_binding, records, loaded


def test_downstream_lock_splits_duplicate_condition_names_by_training_seed(
    tmp_path, monkeypatch
):
    _, _, _, loaded = _fake_downstream(tmp_path, monkeypatch)

    assert [campaign["training_seed"] for campaign in loaded["campaigns"]] == [1, 2]
    assert [
        len(campaign["campaign"]["artifacts"]) for campaign in loaded["campaigns"]
    ] == [10, 10]
    assert len(submit.plan_downstream_jobs(loaded, "common")) == 20
    assert len(submit.plan_downstream_jobs(loaded, "score")) == 60
    assert len(submit.plan_downstream_jobs(loaded, "diagnose")) == 20
    assert all(
        job.key[0] in {1, 2}
        for phase in ("common", "score", "diagnose")
        for job in submit.plan_downstream_jobs(loaded, phase)
    )
    with pytest.raises(FileNotFoundError, match="scoring outputs are incomplete"):
        submit.plan_downstream_jobs(loaded, "assemble")


def test_seed_downstream_commands_are_singular_and_receipts_are_fixed(
    tmp_path, monkeypatch
):
    binding, _, _, loaded = _fake_downstream(tmp_path, monkeypatch)
    expected_counts = {"common": 20, "score": 60, "diagnose": 20}
    for phase, expected in expected_counts.items():
        jobs = submit.plan_downstream_jobs(loaded, phase)
        assert len(jobs) == expected
        assert (
            len({job.command[job.command.index("--job-name") + 1] for job in jobs})
            == expected
        )
        for job in jobs:
            command = job.command
            assert command[command.index("--partition") + 1] != (
                submit.CPU_PARTITION_REQUEST
            )
            assert command.count("--artifact-manifest") == 1
            assert command.count("--expected-artifact-manifest-sha256") == 1
            assert command.count("--output-root") == 1
            assert "--array" not in command
            if phase == "score":
                assert command.count("--m") == 1
                assert command.count("--seed") == 1
            else:
                assert "--m" not in command

    job = submit.plan_downstream_jobs(loaded, "common")[0]
    malformed = submit.PlannedJob(
        phase=job.phase,
        key=job.key,
        command=(*job.command, "--artifact-manifest", "second.json"),
    )
    with pytest.raises(RuntimeError, match="1 occurrence.*artifact-manifest"):
        submit._validate_downstream_job_isolation((malformed,), phase="common")

    submit._validate_receipt_argument(
        binding,
        phase="train",
        submit=True,
        receipt=submit.TRAIN_RECEIPT,
    )
    freeze_receipt = (
        Path(binding["spec"]["paths"]["checkpoint_lock"]).parent
        / "freeze-submissions.jsonl"
    )
    submit._validate_receipt_argument(
        binding,
        phase="freeze",
        submit=True,
        receipt=freeze_receipt,
    )
    with pytest.raises(ValueError, match="fixed duplicate guard"):
        submit._validate_receipt_argument(
            binding,
            phase="freeze",
            submit=True,
            receipt=tmp_path / "new-receipt.jsonl",
        )
    with pytest.raises(ValueError, match="only together"):
        submit._validate_receipt_argument(
            binding,
            phase="freeze",
            submit=False,
            receipt=freeze_receipt,
        )


def test_analysis_and_render_are_singleton_jobs_bound_to_fixed_outputs(
    tmp_path, monkeypatch
):
    binding, _, _, loaded = _fake_downstream(tmp_path, monkeypatch)
    import scripts.analyze.seed as analyzer
    import scripts.render.seed as renderer

    monkeypatch.setattr(
        analyzer, "validate_analysis_inputs", lambda *args, **kwargs: None
    )
    analysis_jobs = submit.plan_analysis_job(
        loaded,
        canonical_analysis="outputs/binary_final_v3_analysis/analysis.json",
        expected_canonical_analysis_sha256="a" * 64,
    )
    assert len(analysis_jobs) == 1
    analysis_command = analysis_jobs[0].command
    assert (
        analysis_command.count("scripts/slurm/run.sbatch")
        == 1
    )
    assert analysis_command.count("--output") == 1
    assert analysis_jobs[0].key[-1] == submit.CPU_PARTITION_REQUEST
    assert analysis_command[analysis_command.index("--partition") + 1] == (
        submit.CPU_PARTITION_REQUEST
    )
    assert (
        analysis_command[analysis_command.index("--output") + 1]
        == (
            Path(binding["spec"]["paths"]["analysis_root"]) / "analysis.json"
        ).as_posix()
    )

    expected_analysis = (
        Path(binding["spec"]["paths"]["analysis_root"]) / "analysis.json"
    )
    downstream_provenance = {
        "path": loaded["path"].as_posix(),
        "sha256": loaded["sha256"],
    }
    monkeypatch.setattr(
        renderer,
        "load_analysis",
        lambda *args, **kwargs: (
            {"provenance": {"downstream_lock": downstream_provenance}},
            {},
            "b" * 64,
        ),
    )
    render_jobs = submit.plan_render_job(
        loaded,
        seed_analysis=expected_analysis,
        expected_seed_analysis_sha256="b" * 64,
    )
    assert len(render_jobs) == 1
    render_command = render_jobs[0].command
    assert (
        render_command.count("scripts/slurm/run.sbatch") == 1
    )
    assert render_jobs[0].key[-1] == submit.CPU_PARTITION_REQUEST
    assert render_command[render_command.index("--partition") + 1] == (
        submit.CPU_PARTITION_REQUEST
    )
    assert render_command[render_command.index("--output") + 1] == (
        expected_analysis.with_name("seed_robustness.tex").as_posix()
    )
    with pytest.raises(ValueError, match="fixed analysis"):
        submit.plan_render_job(
            loaded,
            seed_analysis=tmp_path / "unbound.json",
            expected_seed_analysis_sha256="b" * 64,
        )


def test_downstream_lock_gate_writes_nothing_when_a_freeze_record_is_missing(
    tmp_path, monkeypatch
):
    base_binding = _binding()
    estimator_path = Path(base_binding["lock"]["estimator_spec"]["path"]).resolve()
    monkeypatch.chdir(tmp_path)
    binding = copy.deepcopy(base_binding)
    binding["lock"]["estimator_spec"]["path"] = estimator_path.as_posix()
    binding["spec"]["paths"].update(
        {
            "checkpoint_lock": "checkpoints.lock.json",
            "artifact_root": "artifacts",
            "freeze_record_root": "freeze-records",
            "common_root": "common",
            "simulation_root": "simulations",
            "assembly_root": "assembled",
            "diagnostic_root": "diagnostics",
            "analysis_root": "analysis",
        }
    )
    checkpoint_binding = _fake_checkpoint_binding(binding, tmp_path, portable=True)
    records = _write_fake_freezes(binding, checkpoint_binding)
    records[-1].unlink()
    destination = downstream._default_downstream_path(binding)

    with pytest.raises(FileNotFoundError):
        downstream.write_downstream_lock(binding, checkpoint_binding, destination)
    assert not destination.exists()
    assert not (destination.parent / "seed-1").exists()
    assert not (destination.parent / "seed-2").exists()


def _cell(dataset, condition, signs):
    return {
        "dataset": dataset,
        "condition": condition,
        "summary": {
            "contrasts": {
                contrast.name: {
                    "seed0_is_majority_direction": signs[0] == signs[1],
                    "direction_reversal": -1 in signs and 1 in signs,
                }
                for contrast in CONTRASTS
            }
        },
    }


def test_gate_c_uses_checkpoint_signs_and_never_image_pseudo_replication():
    stable = [_cell(f"d{i}", "clipseg-target", (1, 1, 1)) for i in range(10)]
    assert _gate_c(stable)["fired"] is False

    sensitive = stable.copy()
    for index in range(3):
        sensitive[index] = _cell(f"d{index}", "clipseg-target", (1, -1, 1))
    outcome = _gate_c(sensitive)
    assert outcome["fired"] is True
    assert all(count == 3 for count in outcome["direction_reversal_counts"].values())
    assert outcome["seed0_not_majority_cells"]


def test_seed_cohort_join_tolerates_one_ulp_but_rejects_changed_truth():
    row = {
        "sample_id": "sample-a",
        "image_id": "sample-a",
        "image_index": 0,
        "class_index": 1,
        "class_name": "foreground",
        "height": 100,
        "width": 120,
        "image_diagonal": float(np.hypot(100, 120)),
        "truth_foreground_fraction": 0.25,
    }
    reference = SimpleNamespace(manifest={"sample_id_sha256": "a" * 64}, rows=(row,))
    roundoff_row = copy.deepcopy(row)
    roundoff_row["image_diagonal"] = float(
        np.nextafter(roundoff_row["image_diagonal"], np.inf)
    )
    roundoff = SimpleNamespace(
        manifest={"sample_id_sha256": "a" * 64}, rows=(roundoff_row,)
    )
    _strict_cohort_join(reference, roundoff, context="test")

    changed_row = copy.deepcopy(row)
    changed_row["truth_foreground_fraction"] += 1e-5
    changed = SimpleNamespace(
        manifest={"sample_id_sha256": "a" * 64}, rows=(changed_row,)
    )
    with pytest.raises(ValueError, match="truth_foreground_fraction"):
        _strict_cohort_join(reference, changed, context="test")


def test_seed_renderer_uses_display_only_times_100_scale():
    cells = []
    for dataset in ("pet", "kvasir", "fives", "isic", "tn3k"):
        for condition in ("clipseg-target", "deeplabv3-target"):
            cells.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "summary": {
                        "contrasts": {
                            contrast.name: {
                                "values": {"0": 0.0123, "1": -0.0045, "2": 0.0},
                                "mean": 0.0026,
                                "minimum": -0.0045,
                                "maximum": 0.0123,
                                "range": 0.0168,
                                "sample_standard_deviation": 0.0087,
                            }
                            for contrast in CONTRASTS
                        }
                    },
                }
            )
    analysis = {"gate_c": {"fired": False}}
    by_key = {(cell["dataset"], cell["condition"]): cell for cell in cells}
    rendered = render_table(analysis, by_key, analysis_sha256="a" * 64)

    assert "+1.23 / -0.45 / +0.00" in rendered
    assert "+0.26 $\\pm$ 0.87 [1.68]" in rendered
    assert "mean $\\pm$ sample SD [range]" in rendered
    assert "multiplied by 100 for display only" in rendered
    assert "0.0123" not in rendered
