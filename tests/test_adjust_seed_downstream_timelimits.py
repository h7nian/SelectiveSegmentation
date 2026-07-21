import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import adjust_seed_downstream_timelimits as adjuster
from scripts.submit_binary_simulations import PlannedJob


def _line(value):
    return (adjuster._canonical_json(value) + "\n").encode()


def _planned_jobs(phase, count):
    jobs = []
    wrapper = {
        "seed_common": "scripts/slurm/score_binary_common.sbatch",
        "seed_score": "scripts/slurm/score_binary_simulation.sbatch",
    }[phase]
    for index in range(count):
        partition = ("agsmall", "amdsmall", "msismall")[index % 3]
        job_name = f"synthetic-{phase}-{index}"
        key = (phase, index, partition)
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            job_name,
            "--partition",
            partition,
            "--account",
            "ssafo",
            wrapper,
            "--synthetic-index",
            str(index),
        )
        jobs.append(PlannedJob(phase=phase, key=key, command=command))
    return tuple(jobs)


def _write_receipt(path, jobs, first_job_id):
    rows = []
    for index, job in enumerate(jobs):
        common = {
            "receipt_schema_version": 1,
            "created_utc": "2026-07-20T18:00:00+00:00",
            "phase": job.phase,
            "key": list(job.key),
            "command": list(job.command),
        }
        rows.extend(
            (
                {**common, "status": "submitting", "job_id": None},
                {
                    **common,
                    "status": "submitted",
                    "job_id": str(first_job_id + index),
                },
            )
        )
    payload = b"".join(_line(row) for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _campaign(tmp_path, monkeypatch):
    common_jobs = _planned_jobs("seed_common", adjuster.EXPECTED_COMMON_JOBS)
    score_jobs = _planned_jobs("seed_score", adjuster.EXPECTED_SCORE_JOBS)
    downstream_path = tmp_path / "downstream.lock.json"
    downstream_path.write_text("synthetic downstream lock\n")
    downstream_sha = hashlib.sha256(downstream_path.read_bytes()).hexdigest()
    common_path = tmp_path / "common.jsonl"
    score_path = tmp_path / "score.jsonl"
    common_sha = _write_receipt(common_path, common_jobs, 910000)
    score_sha = _write_receipt(score_path, score_jobs, 920000)
    ledger_path = tmp_path / "scheduler" / "adjustments.jsonl"

    def load_lock(path, *, expected_sha256):
        assert Path(path) == downstream_path
        assert expected_sha256 == downstream_sha
        return {
            "path": downstream_path.resolve(),
            "sha256": downstream_sha,
        }

    def plan(downstream, phase):
        assert downstream["sha256"] == downstream_sha
        return {"common": common_jobs, "score": score_jobs}[phase]

    monkeypatch.setattr(adjuster, "load_downstream_lock", load_lock)
    monkeypatch.setattr(adjuster, "plan_downstream_jobs", plan)
    jobs = (*common_jobs, *score_jobs)
    job_ids = tuple(
        [str(910000 + index) for index in range(len(common_jobs))]
        + [str(920000 + index) for index in range(len(score_jobs))]
    )
    observations = {}
    for job, job_id in zip(jobs, job_ids, strict=True):
        descriptor = adjuster._planned_descriptor(job)
        observations[job_id] = {
            "job_id": job_id,
            "state": "PENDING",
            "timelimit_raw_minutes": adjuster.OLD_TIMELIMIT_MINUTES,
            "partition": descriptor["partition"],
            "job_name": descriptor["job_name"],
            "command": descriptor["expected_scheduler_command"],
        }
    return SimpleNamespace(
        downstream_path=downstream_path,
        downstream_sha=downstream_sha,
        common_path=common_path,
        common_sha=common_sha,
        score_path=score_path,
        score_sha=score_sha,
        ledger_path=ledger_path,
        jobs=jobs,
        job_ids=job_ids,
        observations=observations,
    )


def _kwargs(campaign, *, provider, updater=None, apply=False):
    values = {
        "downstream_lock": campaign.downstream_path,
        "expected_downstream_lock_sha256": campaign.downstream_sha,
        "common_receipt": campaign.common_path,
        "expected_common_receipt_sha256": campaign.common_sha,
        "score_receipt": campaign.score_path,
        "expected_score_receipt_sha256": campaign.score_sha,
        "private_ledger": campaign.ledger_path,
        "provider": provider,
        "apply": apply,
        "now": "2026-07-20T19:00:00+00:00",
    }
    if updater is not None:
        values["updater"] = updater
    return values


def _provider(campaign):
    def provide(job_ids):
        assert tuple(job_ids) == campaign.job_ids
        return {job_id: dict(campaign.observations[job_id]) for job_id in job_ids}

    return provide


def test_dry_run_is_read_only_and_requires_exact_eighty_jobs(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)
    receipt_bytes = (
        campaign.common_path.read_bytes(),
        campaign.score_path.read_bytes(),
    )

    summary = adjuster.adjust_pending_timelimits(
        **_kwargs(campaign, provider=_provider(campaign))
    )

    assert summary["mode"] == "dry-run"
    assert summary["status"] == "ready"
    assert summary["expected_jobs"] == 80
    assert summary["phase_job_counts"] == {"common": 20, "score": 60}
    assert summary["timelimit_raw_minute_counts"] == {"720": 80}
    assert summary["remaining_updates"] == 80
    assert not campaign.ledger_path.exists()
    assert (campaign.common_path.read_bytes(), campaign.score_path.read_bytes()) == (
        receipt_bytes
    )


def test_apply_commits_intent_before_updates_and_verified_applied_event(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path, monkeypatch)
    calls = []

    def updater(job_id):
        rows, _ = adjuster._strict_jsonl(
            campaign.ledger_path.read_bytes(), source=str(campaign.ledger_path)
        )
        assert rows[0]["artifact_type"] == (
            "selectseg.seed_downstream_timelimit_intent"
        )
        assert len(rows) == 1
        calls.append(job_id)
        campaign.observations[job_id]["timelimit_raw_minutes"] = 180

    summary = adjuster.adjust_pending_timelimits(
        **_kwargs(
            campaign,
            provider=_provider(campaign),
            updater=updater,
            apply=True,
        )
    )

    assert calls == list(campaign.job_ids)
    assert summary["status"] == "applied"
    assert summary["timelimit_raw_minute_counts"] == {"180": 80}
    rows, _ = adjuster._strict_jsonl(
        campaign.ledger_path.read_bytes(), source=str(campaign.ledger_path)
    )
    assert [row["artifact_type"] for row in rows] == [
        "selectseg.seed_downstream_timelimit_intent",
        "selectseg.seed_downstream_timelimit_applied",
    ]
    assert len(rows[0]["jobs"]) == len(rows[1]["jobs"]) == 80
    assert {row["timelimit_raw_minutes"] for row in rows[1]["jobs"]} == {180}

    second = adjuster.adjust_pending_timelimits(
        **_kwargs(
            campaign,
            provider=lambda _: pytest.fail("sealed rerun must not query Slurm"),
            updater=lambda _: pytest.fail("sealed rerun must not update Slurm"),
            apply=True,
        )
    )
    assert second["status"] == "already-applied"
    assert len(campaign.ledger_path.read_text().splitlines()) == 2


def test_partial_failure_is_audited_and_same_receipts_resume(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)
    original_receipts = (
        campaign.common_path.read_bytes(),
        campaign.score_path.read_bytes(),
    )
    first_calls = []

    def failing_updater(job_id):
        first_calls.append(job_id)
        if len(first_calls) == 4:
            raise adjuster.SchedulerUpdateError(job_id, 9)
        campaign.observations[job_id]["timelimit_raw_minutes"] = 180

    with pytest.raises(RuntimeError, match="partial scheduler update recorded"):
        adjuster.adjust_pending_timelimits(
            **_kwargs(
                campaign,
                provider=_provider(campaign),
                updater=failing_updater,
                apply=True,
            )
        )
    rows, _ = adjuster._strict_jsonl(
        campaign.ledger_path.read_bytes(), source=str(campaign.ledger_path)
    )
    assert [row["artifact_type"] for row in rows] == [
        "selectseg.seed_downstream_timelimit_intent",
        "selectseg.seed_downstream_timelimit_attempt",
    ]
    assert rows[1]["successful_update_job_ids"] == list(campaign.job_ids[:3])
    assert rows[1]["failed_job_id"] == campaign.job_ids[3]
    assert rows[1]["update_return_code"] == 9

    resumed_calls = []

    def resumed_updater(job_id):
        resumed_calls.append(job_id)
        campaign.observations[job_id]["timelimit_raw_minutes"] = 180

    summary = adjuster.adjust_pending_timelimits(
        **_kwargs(
            campaign,
            provider=_provider(campaign),
            updater=resumed_updater,
            apply=True,
        )
    )
    assert resumed_calls == list(campaign.job_ids[3:])
    assert summary["status"] == "applied"
    rows, _ = adjuster._strict_jsonl(
        campaign.ledger_path.read_bytes(), source=str(campaign.ledger_path)
    )
    assert [row["artifact_type"] for row in rows] == [
        "selectseg.seed_downstream_timelimit_intent",
        "selectseg.seed_downstream_timelimit_attempt",
        "selectseg.seed_downstream_timelimit_applied",
    ]
    assert (
        campaign.common_path.read_bytes(),
        campaign.score_path.read_bytes(),
    ) == original_receipts


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("state", "RUNNING", "must remain PENDING"),
        ("timelimit_raw_minutes", 719, "forbidden TimelimitRaw"),
        ("partition", "other", "locked partition"),
        ("job_name", "other", "locked identity/name"),
        ("command", "/tmp/other.sbatch", "locked command"),
    ),
)
def test_preflight_rejects_state_limit_or_invariant_before_intent(
    tmp_path, monkeypatch, field, value, match
):
    campaign = _campaign(tmp_path, monkeypatch)
    campaign.observations[campaign.job_ids[0]][field] = value
    calls = []

    with pytest.raises(ValueError, match=match):
        adjuster.adjust_pending_timelimits(
            **_kwargs(
                campaign,
                provider=_provider(campaign),
                updater=calls.append,
                apply=True,
            )
        )
    assert calls == []
    assert not campaign.ledger_path.exists()


def test_fixed_receipt_hash_and_global_job_ids_fail_closed(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)
    wrong_sha = "f" * 64
    kwargs = _kwargs(campaign, provider=_provider(campaign))
    kwargs["expected_common_receipt_sha256"] = wrong_sha
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        adjuster.adjust_pending_timelimits(**kwargs)

    duplicate = campaign.job_ids[0]
    original_loader = adjuster._load_fixed_receipt

    def duplicate_loader(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        if kwargs["expected_jobs"] == adjuster.EXPECTED_SCORE_JOBS:
            changed = list(loaded["bindings"])
            changed[0] = {**changed[0], "job_id": duplicate}
            loaded = {**loaded, "bindings": tuple(changed)}
        return loaded

    monkeypatch.setattr(adjuster, "_load_fixed_receipt", duplicate_loader)
    with pytest.raises(ValueError, match="80 globally unique jobs"):
        adjuster.adjust_pending_timelimits(
            **_kwargs(campaign, provider=_provider(campaign))
        )


def test_query_scheduler_cross_checks_raw_limit_and_live_command():
    ids = ("101", "102")
    sacct = "\n".join(
        (
            "101|PENDING|720|agsmall|job-a|",
            "101.batch|PENDING|||",
            "102|PENDING|720|amdsmall|job-b|",
        )
    )
    squeue = "\n".join(
        (
            "101|PENDING|12:00:00|agsmall|job-a|/repo/common.sbatch",
            "102|PENDING|12:00:00|amdsmall|job-b|/repo/score.sbatch",
        )
    )

    def accounting_runner(command, **kwargs):
        assert command[0] == "sacct"
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=sacct)

    def queue_runner(command, **kwargs):
        assert command[0] == "squeue"
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=squeue)

    result = adjuster.query_scheduler(
        ids, accounting_runner=accounting_runner, queue_runner=queue_runner
    )
    assert tuple(result) == ids
    assert result["101"] == {
        "job_id": "101",
        "state": "PENDING",
        "timelimit_raw_minutes": 720,
        "partition": "agsmall",
        "job_name": "job-a",
        "command": "/repo/common.sbatch",
    }


def test_update_command_and_cli_are_narrow():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    adjuster.update_timelimit("12345", runner=runner)
    assert calls == [
        (
            [
                "scontrol",
                "update",
                "JobId=12345",
                "TimeLimit=03:00:00",
            ],
            {"check": False, "capture_output": True, "text": True},
        )
    ]
    assert adjuster.parse_args([]).apply is False
    assert adjuster.parse_args(["--apply"]).apply is True
    with pytest.raises(SystemExit):
        adjuster.parse_args(["--receipt", "replacement.jsonl"])


def test_historical_evidence_records_three_hour_limit_and_headroom():
    evidence = adjuster._evidence()
    assert evidence["historical_canonical_max_elapsed_seconds"] == {
        "common": 1018,
        "score": 3783,
    }
    assert evidence["requested_timelimit_seconds"] == 3 * 60 * 60
    assert evidence["headroom_over_global_historical_max_seconds"] == 10800 - 3783
