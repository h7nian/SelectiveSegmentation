import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.maintenance import finalize_seed as ledger
from scripts.submit.main import PlannedJob


def _line(value):
    return (ledger._canonical_json(value) + "\n").encode()


def _campaign(tmp_path):
    jobs = tuple(
        PlannedJob(
            phase="seed_train",
            key=(f"dataset-{index}", "model", 1, "private-gpu"),
            command=("sbatch", "--parsable", f"job-{index}"),
        )
        for index in range(ledger.EXPECTED_JOBS)
    )
    job_ids = tuple(str(900000 + index) for index in range(ledger.EXPECTED_JOBS))
    receipt_path = tmp_path / "private" / "train.jsonl"
    receipt_path.parent.mkdir()
    receipt_rows = []
    for job, job_id in zip(jobs, job_ids, strict=True):
        common = {
            "receipt_schema_version": 1,
            "created_utc": "2026-07-20T12:00:00+00:00",
            "phase": job.phase,
            "key": list(job.key),
            "command": list(job.command),
        }
        receipt_rows.extend(
            (
                {**common, "status": "submitting", "job_id": None},
                {**common, "status": "submitted", "job_id": job_id},
            )
        )
    receipt_payload = b"".join(_line(row) for row in receipt_rows)
    receipt_path.write_bytes(receipt_payload)
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    spec_sha = "a" * 64

    adjusted_ids = job_ids[:10]
    completed_ids = job_ids[10:]
    first = {
        "adjustment_id": "seed-train-pending-timelimit-test",
        "artifact_type": "selectseg.scheduler_adjustment",
        "created_utc": "2026-07-20T12:01:00+00:00",
        "evidence": {
            "completed_elapsed_seconds": list(range(101, 111)),
            "completed_job_ids": list(completed_ids),
            "maximum_elapsed_seconds": 110,
            "source": "sacct",
        },
        "invariants": {
            "account": "unchanged",
            "command": "unchanged",
            "experiment_identity": "unchanged",
            "gres": "unchanged",
            "job_id": "unchanged",
            "partition": "unchanged",
            "spec_lock": "unchanged",
        },
        "job_ids": list(adjusted_ids),
        "new_time_limit": "02:00:00",
        "old_time_limit": "24:00:00",
        "reason": "same computation, shorter conservative scheduler request",
        "receipt_schema_version": 1,
        "spec_lock_sha256": spec_sha,
        "train_submission_receipt_sha256": receipt_sha,
    }
    first_payload = _line(first)
    second = {
        "artifact_type": "selectseg.scheduler_adjustment_verification",
        "created_utc": "2026-07-20T12:02:00+00:00",
        "jobs": [
            {
                "job_id": job_id,
                "job_name": f"job-{index}",
                "provisional_start": "2026-07-20T13:00:00",
                "reason": "Priority",
                "state": "PENDING",
            }
            for index, job_id in enumerate(adjusted_ids)
        ],
        "parent_adjustment_id": first["adjustment_id"],
        "parent_log_sha256": hashlib.sha256(first_payload).hexdigest(),
        "receipt_schema_version": 1,
        "resource_invariants": {
            "account": "private",
            "command": "train-wrapper",
            "gres": "gpu:1",
            "memory": "64G",
            "no_requeue": True,
            "num_cpus": 16,
            "partition": "private-gpu",
            "time_limit": "02:00:00",
        },
        "source": "scontrol show job -o",
        "verification_id": "seed-train-pending-timelimit-verification-test",
    }
    private_path = tmp_path / "private" / "scheduler.jsonl"
    private_payload = first_payload + _line(second)
    private_path.write_bytes(private_payload)

    records = [
        {
            "dataset": job.key[0],
            "model": job.key[1],
            "training_seed": job.key[2],
            "receipt_job_id": job_id,
            "record_slurm_job_id": job_id,
            "train_record_sha256": f"{index + 1:064x}",
        }
        for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True))
    ]
    record_core = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "training_seed": row["training_seed"],
            "train_record_sha256": row["train_record_sha256"],
        }
        for row in records
    ]
    record_sha = hashlib.sha256(
        ledger._canonical_json(record_core).encode()
    ).hexdigest()
    public_path = tmp_path / "public" / "scheduler.json"
    return SimpleNamespace(
        jobs=jobs,
        job_ids=job_ids,
        receipt_path=receipt_path,
        receipt_sha=receipt_sha,
        spec_sha=spec_sha,
        private_path=private_path,
        private_payload=private_payload,
        records=records,
        record_sha=record_sha,
        public_path=public_path,
    )


def _patch_campaign(monkeypatch, campaign):
    binding = {
        "sha256": campaign.spec_sha,
        "spec": {},
    }
    monkeypatch.setattr(
        ledger.seedext,
        "load_spec_lock",
        lambda path, expected_sha256: binding,
    )
    monkeypatch.setattr(ledger, "plan_training_jobs", lambda binding: campaign.jobs)
    monkeypatch.setattr(
        ledger,
        "_load_records",
        lambda binding, jobs, receipt: (campaign.records, campaign.record_sha),
    )
    monkeypatch.setattr(
        ledger,
        "_load_terminal_records",
        lambda binding, jobs, receipt, accounting: (
            campaign.records,
            campaign.record_sha,
        ),
    )


def _accounting(campaign, *, state="COMPLETED", exit_code="0:0"):
    return {
        job_id: {
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": 100 + index,
            "timelimit_seconds": 7200,
        }
        for index, job_id in enumerate(campaign.job_ids)
    }


def _job_bindings(campaign):
    return {
        (row["dataset"], row["model"], row["training_seed"]): row["receipt_job_id"]
        for row in campaign.records
    }


def _patch_partial_record_campaign(monkeypatch, campaign, train_root, missing_index):
    experiments = [
        {
            "dataset": {"name": job.key[0]},
            "model": {"name": job.key[1]},
            "training_seed": job.key[2],
        }
        for job in campaign.jobs
    ]
    binding = {
        "sha256": campaign.spec_sha,
        "spec": {"paths": {"train_root": str(train_root)}},
    }
    monkeypatch.setattr(
        ledger.seedext,
        "load_spec_lock",
        lambda path, expected_sha256: binding,
    )
    monkeypatch.setattr(ledger, "plan_training_jobs", lambda binding: campaign.jobs)
    monkeypatch.setattr(
        ledger.seedext, "iter_experiments", lambda spec: iter(experiments)
    )

    def load_entry(binding, job, experiment, receipt):
        index = campaign.jobs.index(job)
        if index == missing_index:
            raise FileNotFoundError("missing terminal training record")
        return campaign.records[index]

    monkeypatch.setattr(ledger, "_record_entry", load_entry)
    return experiments[missing_index]


def _finalize(campaign, *, provider, write=True):
    return ledger.finalize_scheduler_ledger(
        spec_lock="synthetic.lock.json",
        expected_spec_lock_sha256=campaign.spec_sha,
        receipt=campaign.receipt_path,
        private_ledger=campaign.private_path,
        public_summary=campaign.public_path,
        provider=provider,
        write=write,
        now="2026-07-20T12:03:00+00:00",
    )


def _bind_fixed_closure_paths(monkeypatch, campaign):
    monkeypatch.setattr(ledger, "TRAIN_RECEIPT", campaign.receipt_path)
    monkeypatch.setattr(ledger, "PRIVATE_LEDGER", campaign.private_path)
    monkeypatch.setattr(ledger, "PUBLIC_SUMMARY", campaign.public_path)


def _public_sha(campaign):
    return hashlib.sha256(campaign.public_path.read_bytes()).hexdigest()


def _write_terminal_failure_closure(campaign):
    accounting = _accounting(campaign)
    accounting[campaign.job_ids[0]].update(state="TIMEOUT", exit_code="0:0")
    core = ledger._event_core(
        parent_hash=hashlib.sha256(campaign.private_payload).hexdigest(),
        spec_sha=campaign.spec_sha,
        receipt_sha=campaign.receipt_sha,
        record_set_sha=campaign.record_sha,
        records=campaign.records,
        accounting=accounting,
    )
    event = ledger._build_event(core, now="2026-07-20T12:03:00+00:00")
    private_payload = campaign.private_payload + _line(event)
    campaign.private_path.write_bytes(private_payload)
    summary = ledger._public_summary(
        event, private_ledger_sha256=hashlib.sha256(private_payload).hexdigest()
    )
    campaign.public_path.parent.mkdir(parents=True)
    campaign.public_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def test_happy_path_is_idempotent_and_public_summary_is_private_free(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    summary = _finalize(campaign, provider=lambda ids: _accounting(campaign))
    assert summary["status"] == "complete"
    assert summary["successful_jobs"] == 20
    assert len(campaign.private_path.read_text().splitlines()) == 3

    public_text = campaign.public_path.read_text()
    for sensitive in [*campaign.job_ids, "dataset-0", "private-gpu", "sbatch"]:
        assert sensitive not in public_text
    loaded = ledger.load_public_scheduler_summary(
        campaign.public_path,
        expected_private_ledger_sha256=summary["bindings"]["private_ledger_sha256"],
        expected_spec_lock_sha256=campaign.spec_sha,
        expected_receipt_sha256=campaign.receipt_sha,
        expected_record_set_sha256=campaign.record_sha,
        require_complete=True,
    )
    assert loaded == summary
    terminal = ledger.load_terminal_scheduler_event(
        campaign.private_path,
        expected_spec_lock_sha256=campaign.spec_sha,
        expected_receipt_sha256=campaign.receipt_sha,
        expected_record_set_sha256=campaign.record_sha,
        expected_job_bindings=_job_bindings(campaign),
        require_complete=True,
    )
    assert (
        terminal["event"]["terminal_event_id"]
        == summary["bindings"]["terminal_event_id"]
    )
    closure = ledger.load_scheduler_accounting_closure(
        campaign.private_path,
        campaign.public_path,
        expected_spec_lock_sha256=campaign.spec_sha,
        expected_receipt_sha256=campaign.receipt_sha,
        expected_record_set_sha256=campaign.record_sha,
        expected_job_bindings=_job_bindings(campaign),
        require_complete=True,
    )
    assert closure["public"] == summary

    original_private = campaign.private_path.read_bytes()
    second = _finalize(
        campaign,
        provider=lambda ids: pytest.fail("idempotent rerun must not query sacct"),
    )
    assert second == summary
    assert campaign.private_path.read_bytes() == original_private


def test_checkpoint_gate_accepts_only_the_complete_fixed_closure(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    summary = _finalize(campaign, provider=lambda ids: _accounting(campaign))
    _bind_fixed_closure_paths(monkeypatch, campaign)

    result = ledger.validate_complete_training_closure(
        {"sha256": campaign.spec_sha, "spec": {}},
        campaign.jobs,
        expected_public_summary_sha256=_public_sha(campaign),
    )

    assert result["public_summary"] == summary
    assert result["training_record_set_sha256"] == campaign.record_sha
    assert result["train_submission_receipt_sha256"] == campaign.receipt_sha


def test_checkpoint_gate_rejects_absent_public_summary(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _bind_fixed_closure_paths(monkeypatch, campaign)

    with pytest.raises(FileNotFoundError, match="required file does not exist"):
        ledger.validate_complete_training_closure(
            {"sha256": campaign.spec_sha, "spec": {}},
            campaign.jobs,
            expected_public_summary_sha256="f" * 64,
        )


def test_checkpoint_gate_rejects_public_summary_hash_mismatch(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _finalize(campaign, provider=lambda ids: _accounting(campaign))
    _bind_fixed_closure_paths(monkeypatch, campaign)

    with pytest.raises(ValueError, match="summary SHA-256 mismatch"):
        ledger.validate_complete_training_closure(
            {"sha256": campaign.spec_sha, "spec": {}},
            campaign.jobs,
            expected_public_summary_sha256="f" * 64,
        )


def test_checkpoint_gate_rejects_terminal_failure_closure(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _write_terminal_failure_closure(campaign)
    _bind_fixed_closure_paths(monkeypatch, campaign)

    with pytest.raises(RuntimeError, match="terminal failures"):
        ledger.validate_complete_training_closure(
            {"sha256": campaign.spec_sha, "spec": {}},
            campaign.jobs,
            expected_public_summary_sha256=_public_sha(campaign),
        )


def test_checkpoint_cli_runs_complete_closure_gate_before_lock_write(monkeypatch):
    from scripts.submit import seed as submit

    binding = {"sha256": "a" * 64, "spec": {"paths": {"checkpoint_lock": "x"}}}
    jobs = tuple(
        PlannedJob("seed_train", (index,), ("sbatch", str(index)))
        for index in range(ledger.EXPECTED_JOBS)
    )
    calls = []
    monkeypatch.setattr(submit, "load_spec_lock", lambda *args, **kwargs: binding)
    monkeypatch.setattr(submit, "plan_training_jobs", lambda value: jobs)

    def gate(value, planned, *, expected_public_summary_sha256):
        assert value is binding
        assert planned == jobs
        assert expected_public_summary_sha256 == "b" * 64
        calls.append("gate")
        return {"training_record_set_sha256": "c" * 64}

    def write(value, destination, *, expected_training_record_set_sha256):
        assert value is binding
        assert destination == "x"
        assert expected_training_record_set_sha256 == "c" * 64
        calls.append("write")
        return destination

    monkeypatch.setattr(ledger, "validate_complete_training_closure", gate)
    monkeypatch.setattr(submit, "write_checkpoint_lock", write)

    with pytest.raises(ValueError, match="requires --expected-scheduler-summary"):
        submit.main(["--phase", "checkpoint-lock", "--write-checkpoint-lock"])
    assert calls == []

    result = submit.main(
        [
            "--phase",
            "checkpoint-lock",
            "--write-checkpoint-lock",
            "--expected-scheduler-summary-sha256",
            "b" * 64,
        ]
    )
    assert result == "x"
    assert calls == ["gate", "write"]

    calls.clear()
    with pytest.raises(ValueError, match="creates the checkpoint lock"):
        submit.main(
            [
                "--phase",
                "checkpoint-lock",
                "--write-checkpoint-lock",
                "--expected-scheduler-summary-sha256",
                "b" * 64,
                "--expected-checkpoint-lock-sha256",
                "d" * 64,
            ]
        )
    assert calls == []


@pytest.mark.parametrize("target", ["receipt_duplicate", "ledger_parent"])
def test_strict_tamper_is_rejected_before_writes(tmp_path, monkeypatch, target):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    if target == "receipt_duplicate":
        payload = campaign.receipt_path.read_text()
        campaign.receipt_path.write_text(
            payload.replace('"phase":', '"phase":"seed_train","phase":', 1)
        )
    else:
        rows = [
            json.loads(line) for line in campaign.private_path.read_text().splitlines()
        ]
        rows[1]["parent_log_sha256"] = "f" * 64
        campaign.private_path.write_bytes(b"".join(_line(row) for row in rows))
    before = campaign.private_path.read_bytes()
    with pytest.raises(ValueError):
        _finalize(campaign, provider=lambda ids: _accounting(campaign))
    assert campaign.private_path.read_bytes() == before
    assert not campaign.public_path.exists()


def test_nonterminal_accounting_causes_no_write(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    accounting = _accounting(campaign)
    accounting[campaign.job_ids[0]]["state"] = "RUNNING"
    with pytest.raises(RuntimeError, match="not all 20"):
        _finalize(campaign, provider=lambda ids: accounting)
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_timeout_is_previewed_but_never_persisted(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    accounting = _accounting(campaign)
    accounting[campaign.job_ids[3]].update(state="TIMEOUT", exit_code="0:0")
    summary = _finalize(campaign, provider=lambda ids: accounting, write=False)
    assert summary["status"] == "terminal_failures"
    assert summary["failed_jobs"] == 1
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()

    with pytest.raises(RuntimeError, match="refusing to persist terminal failures"):
        _finalize(campaign, provider=lambda ids: accounting, write=True)
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_timeout_without_training_record_is_valid_in_failure_preview(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    missing_index = 3
    _patch_partial_record_campaign(
        monkeypatch, campaign, tmp_path / "train", missing_index
    )
    accounting = _accounting(campaign)
    accounting[campaign.job_ids[missing_index]].update(state="TIMEOUT", exit_code="0:0")

    summary = _finalize(campaign, provider=lambda ids: accounting, write=False)
    assert summary["status"] == "terminal_failures"
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_successful_job_without_training_record_is_rejected(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_partial_record_campaign(monkeypatch, campaign, tmp_path / "train", 3)

    with pytest.raises(ValueError, match="successful.*no immutable"):
        _finalize(campaign, provider=lambda ids: _accounting(campaign))
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_failed_job_with_existing_invalid_record_is_rejected(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    train_root = tmp_path / "train"
    missing_index = 3
    experiment = _patch_partial_record_campaign(
        monkeypatch, campaign, train_root, missing_index
    )
    record_path = ledger._expected_record_path(
        {"spec": {"paths": {"train_root": str(train_root)}}}, experiment
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n")
    accounting = _accounting(campaign)
    accounting[campaign.job_ids[missing_index]].update(state="TIMEOUT", exit_code="0:0")

    with pytest.raises(ValueError, match="exists but did not strictly validate"):
        _finalize(campaign, provider=lambda ids: accounting)
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_private_append_interruption_preserves_original_log(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)

    def interrupted(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(ledger.os, "replace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        _finalize(campaign, provider=lambda ids: _accounting(campaign))
    assert campaign.private_path.read_bytes() == campaign.private_payload
    assert not campaign.public_path.exists()


def test_rerun_completes_publication_after_post_append_interruption(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    original_writer = ledger._atomic_write_new_or_identical
    monkeypatch.setattr(
        ledger,
        "_atomic_write_new_or_identical",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("after append")),
    )
    with pytest.raises(OSError, match="after append"):
        _finalize(campaign, provider=lambda ids: _accounting(campaign))
    assert len(campaign.private_path.read_text().splitlines()) == 3
    assert not campaign.public_path.exists()

    monkeypatch.setattr(ledger, "_atomic_write_new_or_identical", original_writer)
    summary = _finalize(
        campaign,
        provider=lambda ids: pytest.fail("terminal event must make rerun offline"),
    )
    assert summary["status"] == "complete"
    assert campaign.public_path.is_file()


def test_symlink_receipt_is_rejected(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    target = campaign.receipt_path
    link = tmp_path / "receipt-link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        ledger.finalize_scheduler_ledger(
            spec_lock="synthetic.lock.json",
            expected_spec_lock_sha256=campaign.spec_sha,
            receipt=link,
            private_ledger=campaign.private_path,
            public_summary=campaign.public_path,
            provider=lambda ids: _accounting(campaign),
            write=True,
        )


def test_sacct_parser_uses_only_exact_top_level_rows_and_normalizes_suffix():
    job_ids = tuple(str(800000 + index) for index in range(20))
    lines = []
    for index, job_id in enumerate(job_ids):
        state = "COMPLETED+" if index == 0 else "COMPLETED"
        lines.append(f"{job_id}|{state}|0:0|{100 + index}|120|")
        # Real sacct output leaves TimelimitRaw empty for job steps and, with
        # parsable2, therefore ends at the delimiter introducing field five.
        lines.append(f"{job_id}.batch|COMPLETED|0:0|{100 + index}|")

    def runner(*args, **kwargs):
        return SimpleNamespace(stdout="\n".join(lines) + "\n")

    rows = ledger.query_sacct(job_ids, runner=runner)
    assert rows[job_ids[0]]["state"] == "COMPLETED"
    assert rows[job_ids[0]]["timelimit_seconds"] == 7200

    duplicate = lines + [f"{job_ids[0]}|COMPLETED|0:0|100|120|"]
    with pytest.raises(ValueError, match="duplicate top-level"):
        ledger.query_sacct(
            job_ids,
            runner=lambda *args, **kwargs: SimpleNamespace(
                stdout="\n".join(duplicate) + "\n"
            ),
        )


@pytest.mark.parametrize("value", ["0900000", "0", "１２３", "1.0", "-1"])
def test_slurm_job_id_must_be_canonical_positive_ascii(value):
    with pytest.raises(ValueError, match="canonical positive"):
        ledger._job_id(value, location="test job id")


def test_public_loader_rejects_nonstandard_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"summary_schema_version":1,"summary_schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        ledger.load_public_scheduler_summary(path)
    path.write_text('{"value":NaN}\n')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        ledger.load_public_scheduler_summary(path)


def test_public_loader_rejects_complete_status_with_noncompleted_states(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    summary = _finalize(campaign, provider=lambda ids: _accounting(campaign))
    summary["state_counts"] = [{"state": "TIMEOUT", "count": 20}]
    campaign.public_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="completed state"):
        ledger.load_public_scheduler_summary(
            campaign.public_path, require_complete=True
        )


def test_complete_terminal_loader_requires_identity_job_bindings(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _finalize(campaign, provider=lambda ids: _accounting(campaign))
    with pytest.raises(TypeError, match="expected_job_bindings"):
        ledger.load_terminal_scheduler_event(
            campaign.private_path, require_complete=True
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_seconds", 0, "total is inconsistent"),
        ("median_seconds", 100.25, "integer or half-integer"),
    ],
)
def test_public_loader_rejects_impossible_summary_statistics(
    tmp_path, monkeypatch, field, value, message
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    summary = _finalize(campaign, provider=lambda ids: _accounting(campaign))
    summary["duration_seconds"][field] = value
    campaign.public_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match=message):
        ledger.load_public_scheduler_summary(campaign.public_path)


def test_terminal_id_excludes_created_utc(tmp_path):
    campaign = _campaign(tmp_path)
    core = ledger._event_core(
        parent_hash="b" * 64,
        spec_sha=campaign.spec_sha,
        receipt_sha=campaign.receipt_sha,
        record_set_sha=campaign.record_sha,
        records=campaign.records,
        accounting=_accounting(campaign),
    )
    first = ledger._build_event(core, now="2026-07-20T12:03:00+00:00")
    second = ledger._build_event(core, now="2026-07-21T12:03:00+00:00")
    assert first["created_utc"] != second["created_utc"]
    assert first["terminal_event_id"] == second["terminal_event_id"]


def test_terminal_event_with_foreign_job_id_is_rejected_without_expected_args(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _finalize(campaign, provider=lambda ids: _accounting(campaign))
    rows = [json.loads(line) for line in campaign.private_path.read_text().splitlines()]
    event = rows[2]
    event["jobs"][0]["receipt_job_id"] = "99999999"
    event["jobs"][0]["record_slurm_job_id"] = "99999999"
    core = {
        key: value
        for key, value in event.items()
        if key not in {"terminal_event_id", "created_utc"}
    }
    event["terminal_event_id"] = hashlib.sha256(
        ledger._canonical_json(core).encode()
    ).hexdigest()
    campaign.private_path.write_bytes(b"".join(_line(row) for row in rows))

    with pytest.raises(ValueError, match="job ids differ"):
        ledger.load_terminal_scheduler_event(
            campaign.private_path,
            expected_job_bindings=_job_bindings(campaign),
        )


def test_terminal_event_rejects_job_ids_swapped_between_experiments(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _finalize(campaign, provider=lambda ids: _accounting(campaign))
    rows = [json.loads(line) for line in campaign.private_path.read_text().splitlines()]
    event = rows[2]
    first = event["jobs"][0]
    second = event["jobs"][1]
    first["receipt_job_id"], second["receipt_job_id"] = (
        second["receipt_job_id"],
        first["receipt_job_id"],
    )
    first["record_slurm_job_id"], second["record_slurm_job_id"] = (
        second["record_slurm_job_id"],
        first["record_slurm_job_id"],
    )
    core = {
        key: value
        for key, value in event.items()
        if key not in {"terminal_event_id", "created_utc"}
    }
    event["terminal_event_id"] = hashlib.sha256(
        ledger._canonical_json(core).encode()
    ).hexdigest()
    campaign.private_path.write_bytes(b"".join(_line(row) for row in rows))

    with pytest.raises(ValueError, match="identity-to-job bindings differ"):
        ledger.load_terminal_scheduler_event(
            campaign.private_path,
            expected_job_bindings=_job_bindings(campaign),
        )


@pytest.mark.parametrize(
    "field",
    ["spec_lock_sha256", "train_submission_receipt_sha256"],
)
def test_terminal_event_is_always_bound_to_base_ledger_without_expected_args(
    tmp_path, monkeypatch, field
):
    campaign = _campaign(tmp_path)
    _patch_campaign(monkeypatch, campaign)
    _finalize(campaign, provider=lambda ids: _accounting(campaign))
    rows = [json.loads(line) for line in campaign.private_path.read_text().splitlines()]
    event = rows[2]
    event[field] = "f" * 64
    core = {
        key: value
        for key, value in event.items()
        if key not in {"terminal_event_id", "created_utc"}
    }
    event["terminal_event_id"] = hashlib.sha256(
        ledger._canonical_json(core).encode()
    ).hexdigest()
    campaign.private_path.write_bytes(b"".join(_line(row) for row in rows))

    with pytest.raises(ValueError, match="binding mismatch"):
        ledger.load_terminal_scheduler_event(
            campaign.private_path,
            expected_job_bindings=_job_bindings(campaign),
        )


def test_record_job_id_mismatch_is_rejected(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    receipt = ledger._validate_receipt(campaign.receipt_path, campaign.jobs)
    experiments = []
    record_paths = []
    records = []
    for index, (job, job_id) in enumerate(
        zip(campaign.jobs, campaign.job_ids, strict=True)
    ):
        path = tmp_path / f"record-{index}.json"
        path.write_text("{}\n")
        record_paths.append(path)
        experiment = {
            "dataset": {"name": job.key[0]},
            "model": {"name": job.key[1], "condition": "target"},
            "training_seed": job.key[2],
            "gpu_profile": {
                "partition": job.key[3],
                "account": "private",
                "gres": "gpu:1",
            },
        }
        experiments.append(experiment)
        records.append(
            {
                "auxiliary_id": ledger.seedext.EXPECTED_AUXILIARY_ID,
                "condition": "target",
                "gpu_profile": experiment["gpu_profile"],
                "runtime": {
                    "slurm_job_id": "999999" if index == 0 else job_id,
                    "partition": job.key[3],
                    "node": "node",
                    "cuda_device": "A100",
                    "environment": {"version": "locked"},
                },
            }
        )
    monkeypatch.setattr(
        ledger.seedext, "iter_experiments", lambda spec: iter(experiments)
    )
    index_by_dataset = {
        experiment["dataset"]["name"]: index
        for index, experiment in enumerate(experiments)
    }

    def fake_load(binding, experiment):
        index = index_by_dataset[experiment["dataset"]["name"]]
        return record_paths[index], records[index], {}

    monkeypatch.setattr(ledger.seedext, "_load_train_record", fake_load)
    binding = {"spec": {"environment": {"version": "locked"}}}
    with pytest.raises(ValueError, match="Slurm id differs"):
        ledger._load_records(binding, campaign.jobs, receipt)
