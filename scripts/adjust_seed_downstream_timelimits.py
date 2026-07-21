"""Auditably shorten the fixed pending seed common/score Slurm requests.

This utility is intentionally campaign-specific.  It accepts only the exact
20-job common receipt and 60-job score receipt already bound to the immutable
seed downstream lock.  A dry run is the default.  ``--apply`` is permitted
only while every job that still needs an update is a top-level ``PENDING`` job
with the locked identity, partition, command, and original 720-minute limit.

The only scheduler mutation is::

    scontrol update JobId=<receipt-job-id> TimeLimit=03:00:00

An append-only, hash-chained intent is durably committed before the first
update.  A verified applied event is committed only after all 80 jobs report
180 minutes.  If an invocation is interrupted or one update fails, rerunning
the same command resumes from the scheduler-observed 720/180-minute split;
the fixed receipts are never changed, cancelled, deleted, or replaced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from scripts import finalize_seed_scheduler_ledger as ledger_utils
from scripts.submit_binary_seed_extension import plan_downstream_jobs
from selectseg.binary_seed_downstream import load_downstream_lock


DOWNSTREAM_LOCK = Path("outputs/binary_seed_campaign/downstream.lock.json")
COMMON_RECEIPT = Path("outputs/binary_seed_campaign/common-submissions.jsonl")
SCORE_RECEIPT = Path("outputs/binary_seed_campaign/score-submissions.jsonl")
PRIVATE_LEDGER = Path(
    "outputs/binary_seed_campaign/downstream-timelimit-adjustments.jsonl"
)

EXPECTED_DOWNSTREAM_LOCK_SHA256 = (
    "8aab3572c7c8d3d15b7b9093deea528c07ab58ed8d772914526f82756823fe73"
)
EXPECTED_COMMON_RECEIPT_SHA256 = (
    "b3c2659dbba35448107de464ccf07d220c9fa4c7bf0c148c3eaa430601a3213e"
)
EXPECTED_SCORE_RECEIPT_SHA256 = (
    "cabec33de7eb0e9e3ca2321d236e4cc9a7dffc55cedb3ce88dd02b9ee633f67f"
)

EVENT_SCHEMA_VERSION = 1
EXPECTED_COMMON_JOBS = 20
EXPECTED_SCORE_JOBS = 60
EXPECTED_JOBS = EXPECTED_COMMON_JOBS + EXPECTED_SCORE_JOBS
OLD_TIMELIMIT_MINUTES = 720
NEW_TIMELIMIT_MINUTES = 180
NEW_TIMELIMIT_TEXT = "03:00:00"
HISTORICAL_CANONICAL_MAX_SECONDS = {"common": 1018, "score": 3783}

_RECEIPT_FIELDS = frozenset(
    {
        "receipt_schema_version",
        "created_utc",
        "phase",
        "key",
        "command",
        "status",
        "job_id",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "job_id",
        "state",
        "timelimit_raw_minutes",
        "partition",
        "job_name",
        "command",
    }
)
_INTENT_JOB_FIELDS = _OBSERVATION_FIELDS | frozenset(
    {"phase", "key", "planned_command_sha256", "wrapper"}
)
_INTENT_FIELDS = frozenset(
    {
        "event_schema_version",
        "artifact_type",
        "intent_id",
        "created_utc",
        "parent_log_sha256",
        "bindings",
        "operation",
        "invariants",
        "evidence",
        "jobs",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "event_schema_version",
        "artifact_type",
        "attempt_id",
        "created_utc",
        "parent_log_sha256",
        "parent_intent_id",
        "already_at_new_limit_job_ids",
        "successful_update_job_ids",
        "failed_job_id",
        "failure_stage",
        "update_return_code",
        "scheduler_snapshot",
    }
)
_APPLIED_FIELDS = frozenset(
    {
        "event_schema_version",
        "artifact_type",
        "applied_id",
        "created_utc",
        "parent_log_sha256",
        "parent_intent_id",
        "operation",
        "jobs",
    }
)
_BINDING_FIELDS = frozenset({"downstream_lock", "common_receipt", "score_receipt"})
_FILE_BINDING_FIELDS = frozenset({"path", "sha256", "expected_jobs"})
_OPERATION_FIELDS = frozenset(
    {"field", "old_timelimit_raw_minutes", "new_timelimit_raw_minutes"}
)
_INVARIANT_FIELDS = frozenset(
    {
        "job_id",
        "experiment_identity",
        "partition",
        "command",
        "receipt",
        "downstream_lock",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "source",
        "historical_canonical_max_elapsed_seconds",
        "requested_timelimit_seconds",
        "headroom_over_global_historical_max_seconds",
    }
)
_ACTIVE_STATES = frozenset(
    {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)


class SchedulerUpdateError(RuntimeError):
    """A single, exact ``scontrol update`` invocation failed."""

    def __init__(self, job_id: str, return_code: int):
        super().__init__(f"scontrol update failed for job {job_id}")
        self.job_id = job_id
        self.return_code = return_code


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _digest(value, *, location):
    return ledger_utils._digest(value, location=location)


def _exact_fields(value, expected, *, location):
    return ledger_utils._exact_fields(value, expected, location=location)


def _utc_timestamp(value, *, location):
    return ledger_utils._utc_timestamp(value, location=location)


def _job_id(value, *, location):
    return ledger_utils._job_id(value, location=location)


def _read_regular(path):
    return ledger_utils._read_regular(path)


def _strict_jsonl(payload, *, source):
    return ledger_utils._strict_jsonl(payload, source=source)


def _normalize_state(value):
    return ledger_utils._normalize_state(value)


def _duration_to_seconds(value, *, location):
    return ledger_utils._duration_to_seconds(value, location=location)


def _now(value=None):
    timestamp = value or datetime.now(timezone.utc).isoformat()
    return _utc_timestamp(timestamp, location="created_utc")


def _flag_value(command, flag, *, location):
    if command.count(flag) != 1:
        raise ValueError(f"{location} must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"{location} has no value for {flag}")
    value = command[index + 1]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} has an invalid value for {flag}")
    return value


def _planned_descriptor(job):
    if job.phase not in {"seed_common", "seed_score"}:
        raise ValueError("adjuster plan contains a non-common/score phase")
    wrapper = {
        "seed_common": "scripts/slurm/score_binary_common.sbatch",
        "seed_score": "scripts/slurm/score_binary_simulation.sbatch",
    }[job.phase]
    if job.command.count(wrapper) != 1:
        raise ValueError(f"{job.phase} plan has the wrong Slurm wrapper")
    if job.command[:2] != ("sbatch", "--parsable"):
        raise ValueError("adjuster accepts only explicit one-job sbatch commands")
    if "--array" in job.command or any(
        token.startswith("--array=") for token in job.command
    ):
        raise ValueError("Slurm arrays are forbidden")
    return {
        "phase": job.phase,
        "key": list(job.key),
        "partition": _flag_value(job.command, "--partition", location=job.phase),
        "job_name": _flag_value(job.command, "--job-name", location=job.phase),
        "wrapper": wrapper,
        "expected_scheduler_command": str((Path.cwd() / wrapper).resolve()),
        "planned_command_sha256": _sha256_bytes(
            _canonical_json(list(job.command)).encode()
        ),
    }


def _load_fixed_receipt(path, *, expected_sha256, planned_jobs, expected_jobs):
    source = Path(path)
    payload = _read_regular(source)
    observed_sha = _sha256_bytes(payload)
    if observed_sha != _digest(expected_sha256, location=f"{source} expected sha256"):
        raise ValueError(f"fixed submission receipt hash mismatch: {source}")
    rows, _ = _strict_jsonl(payload, source=str(source))
    by_identity = {(job.phase, job.key): job for job in planned_jobs}
    if len(planned_jobs) != expected_jobs or len(by_identity) != expected_jobs:
        raise ValueError(f"locked plan must contain exactly {expected_jobs} jobs")
    latest = {}
    for index, row in enumerate(rows, 1):
        location = f"{source}:{index}"
        _exact_fields(row, _RECEIPT_FIELDS, location=location)
        if row["receipt_schema_version"] != 1:
            raise ValueError(f"unsupported receipt schema at {location}")
        _utc_timestamp(row["created_utc"], location=f"{location}.created_utc")
        if not isinstance(row["phase"], str) or not isinstance(row["key"], list):
            raise ValueError(f"invalid receipt identity at {location}")
        try:
            identity = (row["phase"], tuple(row["key"]))
            planned = by_identity[identity]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"receipt contains a job outside the locked plan at {location}"
            ) from error
        if not isinstance(row["command"], list) or tuple(row["command"]) != (
            planned.command
        ):
            raise ValueError(f"receipt command differs from plan at {location}")
        status_value = row["status"]
        if status_value not in {"submitting", "submitted", "failed"}:
            raise ValueError(f"invalid receipt status at {location}")
        if status_value == "submitted":
            _job_id(row["job_id"], location=f"{location}.job_id")
        elif row["job_id"] is not None:
            raise ValueError(f"{location}.job_id must be null")
        previous = latest.get(identity)
        previous_status = None if previous is None else previous["status"]
        allowed = {
            None: {"submitting"},
            "submitting": {"submitted", "failed"},
            "failed": {"submitting"},
            "submitted": set(),
        }[previous_status]
        if status_value not in allowed:
            raise ValueError(f"invalid receipt state transition at {location}")
        latest[identity] = row
    if set(latest) != set(by_identity):
        raise ValueError("receipt does not cover the exact locked plan")
    if any(row["status"] != "submitted" for row in latest.values()):
        raise ValueError("receipt contains unresolved or failed submission intent")
    bindings = []
    for job in planned_jobs:
        row = latest[(job.phase, job.key)]
        bindings.append(
            {
                "job_id": row["job_id"],
                "job": job,
                "descriptor": _planned_descriptor(job),
            }
        )
    job_ids = [item["job_id"] for item in bindings]
    if len(set(job_ids)) != expected_jobs:
        raise ValueError("receipt does not bind unique top-level Slurm job IDs")
    return {
        "path": source,
        "sha256": observed_sha,
        "bindings": tuple(bindings),
    }


def _load_fixed_campaign(
    *,
    downstream_lock,
    expected_downstream_lock_sha256,
    common_receipt,
    expected_common_receipt_sha256,
    score_receipt,
    expected_score_receipt_sha256,
):
    downstream = load_downstream_lock(
        downstream_lock, expected_sha256=expected_downstream_lock_sha256
    )
    if downstream["sha256"] != expected_downstream_lock_sha256:
        raise ValueError("downstream loader returned an inconsistent hash")
    common_jobs = plan_downstream_jobs(downstream, "common")
    score_jobs = plan_downstream_jobs(downstream, "score")
    common = _load_fixed_receipt(
        common_receipt,
        expected_sha256=expected_common_receipt_sha256,
        planned_jobs=common_jobs,
        expected_jobs=EXPECTED_COMMON_JOBS,
    )
    score = _load_fixed_receipt(
        score_receipt,
        expected_sha256=expected_score_receipt_sha256,
        planned_jobs=score_jobs,
        expected_jobs=EXPECTED_SCORE_JOBS,
    )
    bindings = (*common["bindings"], *score["bindings"])
    job_ids = tuple(item["job_id"] for item in bindings)
    if len(bindings) != EXPECTED_JOBS or len(set(job_ids)) != EXPECTED_JOBS:
        raise ValueError("common and score receipts must bind 80 globally unique jobs")
    return {
        "downstream": downstream,
        "common": common,
        "score": score,
        "bindings": bindings,
        "job_ids": job_ids,
    }


def query_scheduler(
    job_ids,
    *,
    accounting_runner=subprocess.run,
    queue_runner=subprocess.run,
):
    """Read exact top-level raw limits plus live immutable job fields."""

    requested = tuple(_job_id(value, location="requested job id") for value in job_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("scheduler query requires unique top-level job IDs")
    requested_set = set(requested)
    accounting_command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(requested),
        "--format=JobIDRaw%32,State%32,TimelimitRaw,Partition%128,JobName%256",
    ]
    accounting_result = accounting_runner(
        accounting_command, check=True, capture_output=True, text=True
    )
    accounting = {}
    for line_number, line in enumerate(accounting_result.stdout.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) == 6 and fields[-1] == "":
            fields.pop()
        if len(fields) != 5:
            raise ValueError(f"unexpected sacct column count on line {line_number}")
        raw_id, raw_state, raw_limit, partition, job_name = (
            field.strip() for field in fields
        )
        if raw_id not in requested_set:
            parents = [
                job_id for job_id in requested if raw_id.startswith(f"{job_id}.")
            ]
            if len(parents) == 1:
                continue
            raise ValueError(f"sacct returned unrelated job row {raw_id!r}")
        if raw_id in accounting:
            raise ValueError(f"sacct returned duplicate top-level row {raw_id}")
        if not raw_limit.isdecimal():
            raise ValueError(f"sacct returned non-integer TimelimitRaw for {raw_id}")
        accounting[raw_id] = {
            "state": _normalize_state(raw_state),
            "timelimit_raw_minutes": int(raw_limit),
            "partition": partition,
            "job_name": job_name,
        }
    if set(accounting) != requested_set:
        raise ValueError("sacct omitted one or more requested top-level jobs")

    queue_command = [
        "squeue",
        "--noheader",
        "--states=all",
        "--jobs",
        ",".join(requested),
        "--format=%.32i|%.32T|%.32l|%.128P|%.256j|%.4096o",
    ]
    queue_result = queue_runner(
        queue_command, check=True, capture_output=True, text=True
    )
    queued = {}
    for line_number, line in enumerate(queue_result.stdout.splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|", 5)]
        if len(fields) != 6:
            raise ValueError(f"unexpected squeue column count on line {line_number}")
        raw_id, raw_state, raw_limit, partition, job_name, command = fields
        if raw_id not in requested_set:
            raise ValueError(f"squeue returned unrelated job row {raw_id!r}")
        if raw_id in queued:
            raise ValueError(f"squeue returned duplicate row {raw_id}")
        if not command or command == "(null)":
            raise ValueError(f"squeue returned no command for {raw_id}")
        queued[raw_id] = {
            "state": _normalize_state(raw_state),
            "timelimit_seconds": _duration_to_seconds(
                raw_limit, location=f"squeue TimeLimit for {raw_id}"
            ),
            "partition": partition,
            "job_name": job_name,
            "command": command,
        }
    if set(queued) != requested_set:
        raise ValueError("squeue omitted one or more requested active jobs")

    result = {}
    for job_id in requested:
        accounting_row = accounting[job_id]
        queue_row = queued[job_id]
        if accounting_row["state"] != queue_row["state"]:
            raise ValueError(f"sacct/squeue state mismatch for job {job_id}")
        if accounting_row["partition"] != queue_row["partition"]:
            raise ValueError(f"sacct/squeue partition mismatch for job {job_id}")
        if accounting_row["job_name"] != queue_row["job_name"]:
            raise ValueError(f"sacct/squeue name mismatch for job {job_id}")
        if (
            accounting_row["timelimit_raw_minutes"] * 60
            != queue_row["timelimit_seconds"]
        ):
            raise ValueError(f"sacct/squeue time-limit mismatch for job {job_id}")
        result[job_id] = {
            "job_id": job_id,
            **accounting_row,
            "command": queue_row["command"],
        }
    return result


def update_timelimit(job_id, *, runner=subprocess.run):
    """Perform the sole allowed scheduler mutation for one receipt job."""

    canonical_id = _job_id(job_id, location="update job id")
    command = [
        "scontrol",
        "update",
        f"JobId={canonical_id}",
        f"TimeLimit={NEW_TIMELIMIT_TEXT}",
    ]
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SchedulerUpdateError(canonical_id, result.returncode)


def _validate_snapshot(snapshot, campaign, *, allowed_limits, require_pending):
    if not isinstance(snapshot, dict) or set(snapshot) != set(campaign["job_ids"]):
        raise ValueError("scheduler provider did not return the exact 80 receipt jobs")
    rows = []
    for binding in campaign["bindings"]:
        job_id = binding["job_id"]
        descriptor = binding["descriptor"]
        row = snapshot[job_id]
        _exact_fields(row, _OBSERVATION_FIELDS, location=f"scheduler[{job_id}]")
        if _job_id(row["job_id"], location=f"scheduler[{job_id}].job_id") != job_id:
            raise ValueError("scheduler observation changed a receipt job ID")
        state_value = _normalize_state(row["state"])
        if state_value not in _ACTIVE_STATES:
            raise ValueError(f"job {job_id} is no longer active")
        if require_pending and state_value != "PENDING":
            raise ValueError("every job requiring adjustment must remain PENDING")
        limit = row["timelimit_raw_minutes"]
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("TimelimitRaw minutes must be an integer")
        if limit not in allowed_limits:
            raise ValueError(f"job {job_id} has forbidden TimelimitRaw={limit} minutes")
        for field in ("partition", "job_name", "command"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"scheduler {field} must be non-empty")
        if row["partition"] != descriptor["partition"]:
            raise ValueError(f"job {job_id} changed its locked partition")
        if row["job_name"] != descriptor["job_name"]:
            raise ValueError(f"job {job_id} changed its locked identity/name")
        if (
            Path(row["command"]).resolve()
            != Path(descriptor["expected_scheduler_command"]).resolve()
        ):
            raise ValueError(f"job {job_id} changed its locked command")
        rows.append(
            {
                "job_id": job_id,
                "state": state_value,
                "timelimit_raw_minutes": limit,
                "partition": row["partition"],
                "job_name": row["job_name"],
                "command": str(Path(row["command"]).resolve()),
            }
        )
    return rows


def _file_binding(path, sha256, expected_jobs):
    return {
        "path": Path(path).as_posix(),
        "sha256": sha256,
        "expected_jobs": expected_jobs,
    }


def _bindings(campaign):
    return {
        "downstream_lock": _file_binding(
            campaign["downstream"]["path"],
            campaign["downstream"]["sha256"],
            EXPECTED_JOBS,
        ),
        "common_receipt": _file_binding(
            campaign["common"]["path"],
            campaign["common"]["sha256"],
            EXPECTED_COMMON_JOBS,
        ),
        "score_receipt": _file_binding(
            campaign["score"]["path"],
            campaign["score"]["sha256"],
            EXPECTED_SCORE_JOBS,
        ),
    }


def _operation():
    return {
        "field": "TimeLimit",
        "old_timelimit_raw_minutes": OLD_TIMELIMIT_MINUTES,
        "new_timelimit_raw_minutes": NEW_TIMELIMIT_MINUTES,
    }


def _invariants():
    return {
        "job_id": "unchanged",
        "experiment_identity": "unchanged",
        "partition": "unchanged",
        "command": "unchanged",
        "receipt": "unchanged",
        "downstream_lock": "unchanged",
    }


def _evidence():
    global_max = max(HISTORICAL_CANONICAL_MAX_SECONDS.values())
    requested_seconds = NEW_TIMELIMIT_MINUTES * 60
    return {
        "source": "completed canonical seed-0 common/score Slurm accounting",
        "historical_canonical_max_elapsed_seconds": dict(
            HISTORICAL_CANONICAL_MAX_SECONDS
        ),
        "requested_timelimit_seconds": requested_seconds,
        "headroom_over_global_historical_max_seconds": requested_seconds - global_max,
    }


def _event_id(core):
    return _sha256_bytes(_canonical_json(core).encode())


def _build_intent(campaign, observed_rows, *, parent_hash, now=None):
    observations = {row["job_id"]: row for row in observed_rows}
    jobs = []
    for binding in campaign["bindings"]:
        descriptor = binding["descriptor"]
        row = observations[binding["job_id"]]
        jobs.append(
            {
                **row,
                "phase": descriptor["phase"],
                "key": descriptor["key"],
                "planned_command_sha256": descriptor["planned_command_sha256"],
                "wrapper": descriptor["wrapper"],
            }
        )
    core = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "artifact_type": "selectseg.seed_downstream_timelimit_intent",
        "parent_log_sha256": parent_hash,
        "bindings": _bindings(campaign),
        "operation": _operation(),
        "invariants": _invariants(),
        "evidence": _evidence(),
        "jobs": jobs,
    }
    return {**core, "intent_id": _event_id(core), "created_utc": _now(now)}


def _build_attempt(
    *,
    intent,
    parent_hash,
    already_new,
    successful,
    failed_job_id,
    failure_stage,
    return_code,
    scheduler_rows,
    now=None,
):
    core = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "artifact_type": "selectseg.seed_downstream_timelimit_attempt",
        "parent_log_sha256": parent_hash,
        "parent_intent_id": intent["intent_id"],
        "already_at_new_limit_job_ids": list(already_new),
        "successful_update_job_ids": list(successful),
        "failed_job_id": failed_job_id,
        "failure_stage": failure_stage,
        "update_return_code": return_code,
        "scheduler_snapshot": scheduler_rows,
    }
    return {**core, "attempt_id": _event_id(core), "created_utc": _now(now)}


def _build_applied(intent, observed_rows, *, parent_hash, now=None):
    core = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "artifact_type": "selectseg.seed_downstream_timelimit_applied",
        "parent_log_sha256": parent_hash,
        "parent_intent_id": intent["intent_id"],
        "operation": _operation(),
        "jobs": observed_rows,
    }
    return {**core, "applied_id": _event_id(core), "created_utc": _now(now)}


def _validate_file_bindings(value, campaign):
    _exact_fields(value, _BINDING_FIELDS, location="intent.bindings")
    expected = _bindings(campaign)
    for name, row in value.items():
        _exact_fields(row, _FILE_BINDING_FIELDS, location=f"bindings.{name}")
        _digest(row["sha256"], location=f"bindings.{name}.sha256")
    if value != expected:
        raise ValueError("timelimit intent is bound to different immutable inputs")


def _validate_operation(value):
    _exact_fields(value, _OPERATION_FIELDS, location="operation")
    if value != _operation():
        raise ValueError("ledger contains a different scheduler operation")


def _validate_intent(row, *, campaign, parent_hash):
    _exact_fields(row, _INTENT_FIELDS, location="timelimit intent")
    if row["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported timelimit-intent schema")
    if row["artifact_type"] != "selectseg.seed_downstream_timelimit_intent":
        raise ValueError("unexpected first downstream scheduler event")
    _utc_timestamp(row["created_utc"], location="intent.created_utc")
    if _digest(row["parent_log_sha256"], location="intent.parent_log_sha256") != (
        parent_hash
    ):
        raise ValueError("intent has an invalid hash-chain parent")
    _validate_file_bindings(row["bindings"], campaign)
    _validate_operation(row["operation"])
    _exact_fields(row["invariants"], _INVARIANT_FIELDS, location="invariants")
    if row["invariants"] != _invariants():
        raise ValueError("intent changed a protected invariant")
    _exact_fields(row["evidence"], _EVIDENCE_FIELDS, location="evidence")
    if row["evidence"] != _evidence():
        raise ValueError("intent contains inconsistent historical evidence")
    if not isinstance(row["jobs"], list) or len(row["jobs"]) != EXPECTED_JOBS:
        raise ValueError("intent must contain exactly 80 jobs")
    expected_by_id = {item["job_id"]: item for item in campaign["bindings"]}
    seen = set()
    for index, job in enumerate(row["jobs"]):
        _exact_fields(job, _INTENT_JOB_FIELDS, location=f"intent.jobs[{index}]")
        job_id = _job_id(job["job_id"], location=f"intent.jobs[{index}].job_id")
        if job_id in seen or job_id not in expected_by_id:
            raise ValueError("intent contains a duplicate or foreign job ID")
        seen.add(job_id)
        descriptor = expected_by_id[job_id]["descriptor"]
        expected_identity = {
            "phase": descriptor["phase"],
            "key": descriptor["key"],
            "partition": descriptor["partition"],
            "job_name": descriptor["job_name"],
            "planned_command_sha256": descriptor["planned_command_sha256"],
            "wrapper": descriptor["wrapper"],
        }
        if any(job[field] != value for field, value in expected_identity.items()):
            raise ValueError("intent job identity differs from the locked plan")
        if job["state"] != "PENDING":
            raise ValueError("intent may be created only for pending jobs")
        if job["timelimit_raw_minutes"] != OLD_TIMELIMIT_MINUTES:
            raise ValueError("intent must record the original 720-minute limit")
        if (
            Path(job["command"]).resolve()
            != Path(descriptor["expected_scheduler_command"]).resolve()
        ):
            raise ValueError("intent records the wrong scheduler command")
    if seen != set(campaign["job_ids"]):
        raise ValueError("intent does not cover the exact 80 receipt jobs")
    core = {
        key: value
        for key, value in row.items()
        if key not in {"intent_id", "created_utc"}
    }
    if _digest(row["intent_id"], location="intent.intent_id") != _event_id(core):
        raise ValueError("intent content ID is invalid")
    return row


def _validate_live_rows(rows, campaign, *, require_new=False):
    if not isinstance(rows, list) or len(rows) != EXPECTED_JOBS:
        raise ValueError("scheduler snapshot must contain exactly 80 jobs")
    mapping = {}
    for index, row in enumerate(rows):
        _exact_fields(row, _OBSERVATION_FIELDS, location=f"snapshot[{index}]")
        job_id = _job_id(row["job_id"], location=f"snapshot[{index}].job_id")
        if job_id in mapping:
            raise ValueError("scheduler snapshot repeats a job ID")
        mapping[job_id] = row
    allowed = (
        {NEW_TIMELIMIT_MINUTES}
        if require_new
        else {
            OLD_TIMELIMIT_MINUTES,
            NEW_TIMELIMIT_MINUTES,
        }
    )
    return _validate_snapshot(
        mapping, campaign, allowed_limits=allowed, require_pending=False
    )


def _validate_attempt(row, *, campaign, intent, parent_hash):
    _exact_fields(row, _ATTEMPT_FIELDS, location="timelimit attempt")
    if row["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported timelimit-attempt schema")
    if row["artifact_type"] != "selectseg.seed_downstream_timelimit_attempt":
        raise ValueError("unexpected downstream scheduler event")
    _utc_timestamp(row["created_utc"], location="attempt.created_utc")
    if _digest(row["parent_log_sha256"], location="attempt.parent_log_sha256") != (
        parent_hash
    ):
        raise ValueError("attempt has an invalid hash-chain parent")
    if row["parent_intent_id"] != intent["intent_id"]:
        raise ValueError("attempt is attached to another intent")
    expected_ids = set(campaign["job_ids"])
    for field in (
        "already_at_new_limit_job_ids",
        "successful_update_job_ids",
    ):
        values = row[field]
        if not isinstance(values, list):
            raise ValueError(f"attempt.{field} must be a list")
        observed = {_job_id(value, location=f"attempt.{field}[]") for value in values}
        if len(observed) != len(values) or not observed <= expected_ids:
            raise ValueError(f"attempt.{field} has duplicate or foreign jobs")
    failed = row["failed_job_id"]
    if failed is not None:
        failed = _job_id(failed, location="attempt.failed_job_id")
        if failed not in expected_ids:
            raise ValueError("attempt failed job is outside the fixed receipts")
    if row["failure_stage"] not in {"scontrol_update", "post_update_verification"}:
        raise ValueError("attempt has an unsupported failure stage")
    return_code = row["update_return_code"]
    if return_code is not None and (
        isinstance(return_code, bool)
        or not isinstance(return_code, int)
        or return_code < 0
    ):
        raise ValueError("attempt update return code must be non-negative or null")
    if row["failure_stage"] == "scontrol_update":
        if failed is None:
            raise ValueError("update failure must identify the failed job")
    elif failed is not None or return_code is not None:
        raise ValueError("post-verification failure cannot name an update failure")
    if row["scheduler_snapshot"] is not None:
        _validate_live_rows(row["scheduler_snapshot"], campaign)
    core = {
        key: value
        for key, value in row.items()
        if key not in {"attempt_id", "created_utc"}
    }
    if _digest(row["attempt_id"], location="attempt.attempt_id") != _event_id(core):
        raise ValueError("attempt content ID is invalid")
    return row


def _validate_applied(row, *, campaign, intent, parent_hash):
    _exact_fields(row, _APPLIED_FIELDS, location="timelimit applied event")
    if row["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported timelimit-applied schema")
    if row["artifact_type"] != "selectseg.seed_downstream_timelimit_applied":
        raise ValueError("unexpected downstream scheduler event")
    _utc_timestamp(row["created_utc"], location="applied.created_utc")
    if _digest(row["parent_log_sha256"], location="applied.parent_log_sha256") != (
        parent_hash
    ):
        raise ValueError("applied event has an invalid hash-chain parent")
    if row["parent_intent_id"] != intent["intent_id"]:
        raise ValueError("applied event is attached to another intent")
    _validate_operation(row["operation"])
    _validate_live_rows(row["jobs"], campaign, require_new=True)
    core = {
        key: value
        for key, value in row.items()
        if key not in {"applied_id", "created_utc"}
    }
    if _digest(row["applied_id"], location="applied.applied_id") != _event_id(core):
        raise ValueError("applied content ID is invalid")
    return row


def _load_ledger(path, campaign):
    source = Path(path)
    if not source.exists():
        return {"payload": b"", "intent": None, "attempts": (), "applied": None}
    payload = _read_regular(source)
    rows, prefix_lengths = _strict_jsonl(payload, source=str(source))
    intent = None
    attempts = []
    applied = None
    for index, row in enumerate(rows):
        parent_length = 0 if index == 0 else prefix_lengths[index - 1]
        parent_hash = _sha256_bytes(payload[:parent_length])
        if index == 0:
            intent = _validate_intent(row, campaign=campaign, parent_hash=parent_hash)
            continue
        if applied is not None:
            raise ValueError("no ledger event may follow the applied closure")
        artifact_type = row.get("artifact_type") if isinstance(row, dict) else None
        if artifact_type == "selectseg.seed_downstream_timelimit_attempt":
            attempts.append(
                _validate_attempt(
                    row,
                    campaign=campaign,
                    intent=intent,
                    parent_hash=parent_hash,
                )
            )
        elif artifact_type == "selectseg.seed_downstream_timelimit_applied":
            applied = _validate_applied(
                row,
                campaign=campaign,
                intent=intent,
                parent_hash=parent_hash,
            )
        else:
            raise ValueError("ledger contains an unsupported event type")
    if intent is None:
        raise ValueError("non-empty scheduler ledger has no intent")
    return {
        "payload": payload,
        "intent": intent,
        "attempts": tuple(attempts),
        "applied": applied,
    }


@contextmanager
def _exclusive_ledger_lock(path):
    destination = Path(path)
    ledger_utils._reject_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ledger_utils._reject_symlink_ancestors(destination.parent)
    lock_path = destination.with_name(f".{destination.name}.lock")
    ledger_utils._reject_symlink_ancestors(lock_path)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("scheduler-ledger lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _commit_event(path, *, expected_payload, event):
    destination = Path(path)
    current = b"" if not destination.exists() else _read_regular(destination)
    if current != expected_payload:
        raise RuntimeError("downstream scheduler ledger changed during adjustment")
    updated = current + (_canonical_json(event) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and _read_regular(destination) != current:
            raise RuntimeError("downstream scheduler ledger changed before commit")
        if destination.exists():
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise RuntimeError(
                    "downstream scheduler ledger appeared before intent commit"
                ) from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def _safe_post_snapshot(provider, campaign):
    try:
        snapshot = provider(campaign["job_ids"])
        return _validate_snapshot(
            snapshot,
            campaign,
            allowed_limits={OLD_TIMELIMIT_MINUTES, NEW_TIMELIMIT_MINUTES},
            require_pending=False,
        )
    except Exception:
        return None


def _summary(*, campaign, ledger, rows, apply, status):
    limits = Counter(row["timelimit_raw_minutes"] for row in rows)
    states = Counter(row["state"] for row in rows)
    return {
        "artifact_type": "selectseg.seed_downstream_timelimit_preview",
        "mode": "apply" if apply else "dry-run",
        "status": status,
        "expected_jobs": EXPECTED_JOBS,
        "phase_job_counts": {
            "common": EXPECTED_COMMON_JOBS,
            "score": EXPECTED_SCORE_JOBS,
        },
        "state_counts": dict(sorted(states.items())),
        "timelimit_raw_minute_counts": {
            str(key): value for key, value in sorted(limits.items())
        },
        "remaining_updates": limits.get(OLD_TIMELIMIT_MINUTES, 0),
        "operation": _operation(),
        "bindings": _bindings(campaign),
        "ledger_sha256": _sha256_bytes(ledger["payload"]),
        "attempt_events": len(ledger["attempts"]),
    }


def adjust_pending_timelimits(
    *,
    downstream_lock=DOWNSTREAM_LOCK,
    expected_downstream_lock_sha256=EXPECTED_DOWNSTREAM_LOCK_SHA256,
    common_receipt=COMMON_RECEIPT,
    expected_common_receipt_sha256=EXPECTED_COMMON_RECEIPT_SHA256,
    score_receipt=SCORE_RECEIPT,
    expected_score_receipt_sha256=EXPECTED_SCORE_RECEIPT_SHA256,
    private_ledger=PRIVATE_LEDGER,
    provider=query_scheduler,
    updater=update_timelimit,
    apply=False,
    now=None,
):
    """Preview or apply the one allowed 720-to-180-minute adjustment."""

    campaign = _load_fixed_campaign(
        downstream_lock=downstream_lock,
        expected_downstream_lock_sha256=expected_downstream_lock_sha256,
        common_receipt=common_receipt,
        expected_common_receipt_sha256=expected_common_receipt_sha256,
        score_receipt=score_receipt,
        expected_score_receipt_sha256=expected_score_receipt_sha256,
    )
    ledger = _load_ledger(private_ledger, campaign)
    if ledger["applied"] is not None:
        rows = ledger["applied"]["jobs"]
        return _summary(
            campaign=campaign,
            ledger=ledger,
            rows=rows,
            apply=apply,
            status="already-applied",
        )
    if not apply:
        snapshot = provider(campaign["job_ids"])
        allowed = (
            {OLD_TIMELIMIT_MINUTES}
            if ledger["intent"] is None
            else {OLD_TIMELIMIT_MINUTES, NEW_TIMELIMIT_MINUTES}
        )
        rows = _validate_snapshot(
            snapshot,
            campaign,
            allowed_limits=allowed,
            require_pending=any(
                row["timelimit_raw_minutes"] == OLD_TIMELIMIT_MINUTES
                for row in snapshot.values()
            ),
        )
        status = "ready" if ledger["intent"] is None else "resume-ready"
        if all(row["timelimit_raw_minutes"] == NEW_TIMELIMIT_MINUTES for row in rows):
            status = "ready-to-seal"
        return _summary(
            campaign=campaign,
            ledger=ledger,
            rows=rows,
            apply=False,
            status=status,
        )

    with _exclusive_ledger_lock(private_ledger):
        ledger = _load_ledger(private_ledger, campaign)
        if ledger["applied"] is not None:
            return _summary(
                campaign=campaign,
                ledger=ledger,
                rows=ledger["applied"]["jobs"],
                apply=True,
                status="already-applied",
            )
        snapshot = provider(campaign["job_ids"])
        allowed = (
            {OLD_TIMELIMIT_MINUTES}
            if ledger["intent"] is None
            else {OLD_TIMELIMIT_MINUTES, NEW_TIMELIMIT_MINUTES}
        )
        provisional_rows = _validate_snapshot(
            snapshot,
            campaign,
            allowed_limits=allowed,
            require_pending=False,
        )
        remaining = [
            row
            for row in provisional_rows
            if row["timelimit_raw_minutes"] == OLD_TIMELIMIT_MINUTES
        ]
        if remaining and any(row["state"] != "PENDING" for row in provisional_rows):
            raise ValueError(
                "all 80 jobs must remain PENDING before any remaining update"
            )
        payload = ledger["payload"]
        intent = ledger["intent"]
        if intent is None:
            if len(remaining) != EXPECTED_JOBS:
                raise ValueError("new intent requires all 80 original 720-minute jobs")
            intent = _build_intent(
                campaign,
                provisional_rows,
                parent_hash=_sha256_bytes(payload),
                now=now,
            )
            payload = _commit_event(
                private_ledger, expected_payload=payload, event=intent
            )
            ledger = {
                "payload": payload,
                "intent": intent,
                "attempts": (),
                "applied": None,
            }

        already_new = [
            row["job_id"]
            for row in provisional_rows
            if row["timelimit_raw_minutes"] == NEW_TIMELIMIT_MINUTES
        ]
        successful = []
        for row in remaining:
            try:
                updater(row["job_id"])
            except Exception as error:
                return_code = (
                    error.return_code
                    if isinstance(error, SchedulerUpdateError)
                    else None
                )
                attempt = _build_attempt(
                    intent=intent,
                    parent_hash=_sha256_bytes(payload),
                    already_new=already_new,
                    successful=successful,
                    failed_job_id=row["job_id"],
                    failure_stage="scontrol_update",
                    return_code=return_code,
                    scheduler_rows=_safe_post_snapshot(provider, campaign),
                    now=now,
                )
                _commit_event(private_ledger, expected_payload=payload, event=attempt)
                raise RuntimeError(
                    "partial scheduler update recorded; rerun the same fixed "
                    "--apply command after confirming all jobs remain pending"
                ) from error
            successful.append(row["job_id"])

        try:
            post_snapshot = provider(campaign["job_ids"])
            post_rows = _validate_snapshot(
                post_snapshot,
                campaign,
                allowed_limits={NEW_TIMELIMIT_MINUTES},
                require_pending=False,
            )
        except Exception as error:
            attempt = _build_attempt(
                intent=intent,
                parent_hash=_sha256_bytes(payload),
                already_new=already_new,
                successful=successful,
                failed_job_id=None,
                failure_stage="post_update_verification",
                return_code=None,
                scheduler_rows=_safe_post_snapshot(provider, campaign),
                now=now,
            )
            _commit_event(private_ledger, expected_payload=payload, event=attempt)
            raise RuntimeError(
                "post-update verification failed; the intent remains resumable"
            ) from error
        applied_event = _build_applied(
            intent,
            post_rows,
            parent_hash=_sha256_bytes(payload),
            now=now,
        )
        final_payload = _commit_event(
            private_ledger, expected_payload=payload, event=applied_event
        )
        final_ledger = _load_ledger(private_ledger, campaign)
        if final_ledger["payload"] != final_payload or final_ledger["applied"] != (
            applied_event
        ):
            raise RuntimeError("applied scheduler closure failed strict round trip")
        return _summary(
            campaign=campaign,
            ledger=final_ledger,
            rows=post_rows,
            apply=True,
            status="applied",
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "commit intent, update only TimeLimit 720->180 minutes, and append "
            "a verified closure; default is a read-only dry run"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = adjust_pending_timelimits(apply=args.apply)
    print(_canonical_json(summary))
    return summary


if __name__ == "__main__":
    main()
