"""Finalize private Slurm accounting for the 20-job seed-training campaign.

This command is deliberately separate from the submission planner.  It is a
read-only ``sacct`` consumer and never submits, updates, cancels, or retries a
job.  Once all 20 top-level jobs are terminal, it validates every record from a
successful job and any record that a failed job nevertheless published.  A
dry run reports terminal failures without mutating the ledger.  Persistent
closure is permitted only for 20 successful jobs, after which one
content-addressed terminal event and a path-free public summary are written.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import statistics
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.submit_binary_seed_extension import plan_training_jobs
from selectseg import binary_seed_extension as seedext


PRIVATE_LEDGER = Path(
    "outputs/binary_seed_extension_campaign/scheduler-adjustments.jsonl"
)
TRAIN_RECEIPT = Path("outputs/binary_seed_extension_campaign/train-submissions.jsonl")
PUBLIC_SUMMARY = Path("outputs/public_seed/seed_scheduler_summary.json")
TERMINAL_EVENT_SCHEMA_VERSION = 1
PUBLIC_SUMMARY_SCHEMA_VERSION = 1
EXPECTED_JOBS = 20

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
_ADJUSTMENT_FIELDS = frozenset(
    {
        "adjustment_id",
        "artifact_type",
        "created_utc",
        "evidence",
        "invariants",
        "job_ids",
        "new_time_limit",
        "old_time_limit",
        "reason",
        "receipt_schema_version",
        "spec_lock_sha256",
        "train_submission_receipt_sha256",
    }
)
_VERIFICATION_FIELDS = frozenset(
    {
        "artifact_type",
        "created_utc",
        "jobs",
        "parent_adjustment_id",
        "parent_log_sha256",
        "receipt_schema_version",
        "resource_invariants",
        "source",
        "verification_id",
    }
)
_TERMINAL_EVENT_FIELDS = frozenset(
    {
        "terminal_event_schema_version",
        "artifact_type",
        "terminal_event_id",
        "created_utc",
        "parent_log_sha256",
        "spec_lock_sha256",
        "train_submission_receipt_sha256",
        "training_record_set_sha256",
        "expected_jobs",
        "jobs",
        "state_counts",
        "duration_seconds",
        "timelimit_seconds",
        "campaign_complete",
        "failed_jobs",
    }
)
_PRIVATE_JOB_FIELDS = frozenset(
    {
        "dataset",
        "model",
        "training_seed",
        "receipt_job_id",
        "record_slurm_job_id",
        "train_record_sha256",
        "state",
        "exit_code",
        "elapsed_seconds",
        "timelimit_seconds",
    }
)
_PUBLIC_FIELDS = frozenset(
    {
        "summary_schema_version",
        "artifact_type",
        "auxiliary_id",
        "status",
        "expected_jobs",
        "terminal_jobs",
        "successful_jobs",
        "failed_jobs",
        "state_counts",
        "duration_seconds",
        "timelimit_seconds",
        "bindings",
    }
)
_PUBLIC_BINDING_FIELDS = frozenset(
    {
        "private_ledger_sha256",
        "spec_lock_sha256",
        "train_submission_receipt_sha256",
        "training_record_set_sha256",
        "terminal_event_id",
    }
)
_SUMMARY_STAT_FIELDS = frozenset(
    {
        "minimum_seconds",
        "maximum_seconds",
        "median_seconds",
        "total_seconds",
    }
)
_STATE_COUNT_FIELDS = frozenset({"state", "count"})

_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
_NONTERMINAL_STATES = frozenset(
    {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(payload, *, source):
    try:
        return json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _exact_fields(value, expected, *, location):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} must contain exactly {sorted(expected)}")


def _nonempty_string(value, *, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _positive_int(value, *, location, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{location} must be a {qualifier} integer")
    return value


def _utc_timestamp(value, *, location):
    _nonempty_string(value, location=location)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{location} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{location} must include a timezone")
    return value


def _job_id(value, *, location):
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or value != str(int(value))
        or int(value) <= 0
    ):
        raise ValueError(
            f"{location} must be a canonical positive top-level Slurm job id"
        )
    return value


def _reject_symlink_ancestors(path):
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path components are forbidden: {path}")


def _read_regular(path):
    source = Path(path)
    _reject_symlink_ancestors(source)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required file does not exist: {source}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"expected a regular file: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    return payload


def _strict_json_file(path):
    return _loads_strict(_read_regular(path).decode("utf-8"), source=str(path))


def _strict_jsonl(payload, *, source):
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{source} must be non-empty and newline terminated")
    rows = []
    prefix_lengths = []
    offset = 0
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), 1):
        if not raw_line.endswith(b"\n") or raw_line in {b"\n", b"\r\n"}:
            raise ValueError(
                f"blank or unterminated JSONL row at {source}:{line_number}"
            )
        if raw_line.endswith(b"\r\n"):
            raise ValueError(f"CRLF JSONL is forbidden at {source}:{line_number}")
        try:
            line = raw_line[:-1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"non-UTF-8 JSONL at {source}:{line_number}") from error
        rows.append(_loads_strict(line, source=f"{source}:{line_number}"))
        offset += len(raw_line)
        prefix_lengths.append(offset)
    return rows, prefix_lengths


def _duration_to_seconds(value, *, location):
    _nonempty_string(value, location=location)
    days = 0
    clock = value
    if "-" in value:
        day_text, clock = value.split("-", 1)
        if not day_text.isdecimal():
            raise ValueError(f"invalid Slurm duration at {location}")
        days = int(day_text)
    parts = clock.split(":")
    if len(parts) not in {2, 3} or not all(part.isdecimal() for part in parts):
        raise ValueError(f"invalid Slurm duration at {location}")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    else:
        hours, minutes, seconds = map(int, parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm duration at {location}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _validate_receipt(path, planned_jobs):
    payload = _read_regular(path)
    rows, _ = _strict_jsonl(payload, source=str(path))
    by_identity = {(job.phase, job.key): job for job in planned_jobs}
    if len(by_identity) != EXPECTED_JOBS or len(planned_jobs) != EXPECTED_JOBS:
        raise ValueError("training plan must contain exactly 20 unique jobs")
    latest = {}
    for index, event in enumerate(rows, 1):
        location = f"{path}:{index}"
        _exact_fields(event, _RECEIPT_FIELDS, location=location)
        if event["receipt_schema_version"] != 1:
            raise ValueError(f"unsupported receipt schema at {location}")
        _utc_timestamp(event["created_utc"], location=f"{location}.created_utc")
        phase = _nonempty_string(event["phase"], location=f"{location}.phase")
        if not isinstance(event["key"], list):
            raise ValueError(f"{location}.key must be a list")
        try:
            identity = (phase, tuple(event["key"]))
            planned = by_identity[identity]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"receipt contains a job outside the locked plan at {location}"
            ) from error
        if (
            not isinstance(event["command"], list)
            or tuple(event["command"]) != planned.command
        ):
            raise ValueError(
                f"receipt command differs from the locked plan at {location}"
            )
        status_value = event["status"]
        if status_value not in {"submitting", "submitted", "failed"}:
            raise ValueError(f"invalid receipt status at {location}")
        if status_value == "submitted":
            _job_id(event["job_id"], location=f"{location}.job_id")
        elif event["job_id"] is not None:
            raise ValueError(f"{location}.job_id must be null for {status_value}")
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
        latest[identity] = event
    if set(latest) != set(by_identity) or len(latest) != EXPECTED_JOBS:
        raise ValueError("receipt does not cover the exact 20-job training plan")
    if any(event["status"] != "submitted" for event in latest.values()):
        raise ValueError(
            "training receipt contains unresolved or failed submission intent"
        )
    job_ids = [event["job_id"] for event in latest.values()]
    if len(set(job_ids)) != EXPECTED_JOBS:
        raise ValueError("training receipt does not bind 20 unique Slurm jobs")
    job_bindings = {}
    for job in planned_jobs:
        if len(job.key) < 3:
            raise ValueError("training job key does not identify dataset/model/seed")
        identity = (
            _nonempty_string(job.key[0], location="training job dataset"),
            _nonempty_string(job.key[1], location="training job model"),
            _positive_int(job.key[2], location="training job seed"),
        )
        event = latest[(job.phase, job.key)]
        if identity in job_bindings:
            raise ValueError("training plan repeats a dataset/model/seed identity")
        job_bindings[identity] = event["job_id"]
    if len(job_bindings) != EXPECTED_JOBS:
        raise ValueError("training receipt lacks 20 identity-to-job bindings")
    return {
        "sha256": _sha256_bytes(payload),
        "by_identity": latest,
        "job_ids": tuple(job_ids),
        "job_bindings": job_bindings,
    }


def _validate_adjustment_row(row, *, receipt, spec_sha256):
    _exact_fields(row, _ADJUSTMENT_FIELDS, location="scheduler adjustment row 1")
    if row["artifact_type"] != "selectseg.scheduler_adjustment":
        raise ValueError("unexpected scheduler adjustment artifact type")
    if row["receipt_schema_version"] != 1:
        raise ValueError("unsupported scheduler adjustment schema")
    _nonempty_string(row["adjustment_id"], location="adjustment_id")
    _utc_timestamp(row["created_utc"], location="adjustment.created_utc")
    if (
        _digest(row["spec_lock_sha256"], location="adjustment.spec_lock_sha256")
        != spec_sha256
    ):
        raise ValueError("scheduler adjustment is bound to another spec lock")
    if (
        _digest(
            row["train_submission_receipt_sha256"], location="adjustment.receipt_sha256"
        )
        != receipt["sha256"]
    ):
        raise ValueError("scheduler adjustment is bound to another receipt")
    if not isinstance(row["job_ids"], list):
        raise ValueError("adjustment.job_ids must be a list")
    adjusted = tuple(
        _job_id(value, location="adjustment.job_ids[]") for value in row["job_ids"]
    )
    if (
        len(adjusted) != 10
        or len(set(adjusted)) != 10
        or not set(adjusted) < set(receipt["job_ids"])
    ):
        raise ValueError("adjustment must bind exactly ten receipt jobs")
    old_seconds = _duration_to_seconds(row["old_time_limit"], location="old_time_limit")
    new_seconds = _duration_to_seconds(row["new_time_limit"], location="new_time_limit")
    if not 0 < new_seconds < old_seconds:
        raise ValueError(
            "scheduler adjustment must strictly reduce a positive time limit"
        )
    _nonempty_string(row["reason"], location="adjustment.reason")
    _exact_fields(
        row["invariants"],
        frozenset(
            {
                "account",
                "command",
                "experiment_identity",
                "gres",
                "job_id",
                "partition",
                "spec_lock",
            }
        ),
        location="adjustment.invariants",
    )
    if set(row["invariants"].values()) != {"unchanged"}:
        raise ValueError("scheduler adjustment changed a locked invariant")
    evidence = row["evidence"]
    _exact_fields(
        evidence,
        frozenset(
            {
                "completed_elapsed_seconds",
                "completed_job_ids",
                "maximum_elapsed_seconds",
                "source",
            }
        ),
        location="adjustment.evidence",
    )
    completed = tuple(
        _job_id(value, location="evidence.completed_job_ids[]")
        for value in evidence["completed_job_ids"]
    )
    elapsed = evidence["completed_elapsed_seconds"]
    if (
        len(completed) != 10
        or len(set(completed)) != 10
        or set(completed) != set(receipt["job_ids"]) - set(adjusted)
        or not isinstance(elapsed, list)
        or len(elapsed) != 10
    ):
        raise ValueError("adjustment evidence must bind the complementary ten jobs")
    for index, seconds in enumerate(elapsed):
        _positive_int(seconds, location=f"completed_elapsed_seconds[{index}]")
    if evidence["maximum_elapsed_seconds"] != max(elapsed):
        raise ValueError("adjustment maximum elapsed time is inconsistent")
    _nonempty_string(evidence["source"], location="adjustment.evidence.source")
    return set(adjusted)


def _validate_verification_row(
    row, *, first_row, first_prefix_sha256, adjusted_job_ids
):
    _exact_fields(row, _VERIFICATION_FIELDS, location="scheduler verification row 2")
    if row["artifact_type"] != "selectseg.scheduler_adjustment_verification":
        raise ValueError("unexpected scheduler verification artifact type")
    if row["receipt_schema_version"] != 1:
        raise ValueError("unsupported scheduler verification schema")
    _nonempty_string(row["verification_id"], location="verification_id")
    _utc_timestamp(row["created_utc"], location="verification.created_utc")
    if row["parent_adjustment_id"] != first_row["adjustment_id"]:
        raise ValueError("scheduler verification has the wrong parent adjustment")
    if (
        _digest(row["parent_log_sha256"], location="verification.parent_log_sha256")
        != first_prefix_sha256
    ):
        raise ValueError("scheduler verification hash chain is invalid")
    _nonempty_string(row["source"], location="verification.source")
    resources = row["resource_invariants"]
    _exact_fields(
        resources,
        frozenset(
            {
                "account",
                "command",
                "gres",
                "memory",
                "no_requeue",
                "num_cpus",
                "partition",
                "time_limit",
            }
        ),
        location="verification.resource_invariants",
    )
    for field in ("account", "command", "gres", "memory", "partition", "time_limit"):
        _nonempty_string(
            resources[field], location=f"verification.resource_invariants.{field}"
        )
    _positive_int(
        resources["num_cpus"], location="verification.resource_invariants.num_cpus"
    )
    if resources["no_requeue"] is not True:
        raise ValueError("verified jobs must retain no-requeue")
    if _duration_to_seconds(
        resources["time_limit"], location="verification.time_limit"
    ) != _duration_to_seconds(
        first_row["new_time_limit"], location="adjustment.new_time_limit"
    ):
        raise ValueError("verification time limit differs from the adjustment")
    if not isinstance(row["jobs"], list) or len(row["jobs"]) != 10:
        raise ValueError("verification must contain exactly ten jobs")
    observed = set()
    for index, job in enumerate(row["jobs"]):
        _exact_fields(
            job,
            frozenset({"job_id", "job_name", "provisional_start", "reason", "state"}),
            location=f"verification.jobs[{index}]",
        )
        observed.add(
            _job_id(job["job_id"], location=f"verification.jobs[{index}].job_id")
        )
        for field in ("job_name", "provisional_start", "reason"):
            _nonempty_string(job[field], location=f"verification.jobs[{index}].{field}")
        if job["state"] != "PENDING":
            raise ValueError(
                "pre-start scheduler verification must contain pending jobs"
            )
    if observed != adjusted_job_ids:
        raise ValueError("scheduler verification covers the wrong adjusted jobs")


def _normalize_state(value):
    raw = _nonempty_string(value, location="sacct State").strip().upper()
    normalized = raw.split()[0].rstrip("+")
    if normalized not in _TERMINAL_STATES | _NONTERMINAL_STATES:
        raise ValueError(f"unsupported Slurm state {value!r}")
    return normalized


def _exit_code(value):
    _nonempty_string(value, location="sacct ExitCode")
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdecimal() for part in parts):
        raise ValueError(f"invalid Slurm ExitCode {value!r}")
    return f"{int(parts[0])}:{int(parts[1])}"


def query_sacct(job_ids, *, runner=subprocess.run):
    """Return normalized top-level accounting rows; never mutate Slurm state."""

    requested = tuple(_job_id(value, location="requested job id") for value in job_ids)
    if len(requested) != EXPECTED_JOBS or len(set(requested)) != EXPECTED_JOBS:
        raise ValueError("sacct query requires exactly 20 unique top-level job ids")
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(requested),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,TimelimitRaw",
    ]
    result = runner(command, check=True, capture_output=True, text=True)
    rows = {}
    requested_set = set(requested)
    for line_number, line in enumerate(result.stdout.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        # Some Slurm builds include a terminal delimiter, while top-level rows
        # from others do not.  An empty TimelimitRaw on a job-step row is still
        # the fifth field and must not be mistaken for that optional delimiter.
        if len(fields) == 6 and fields[-1] == "":
            fields.pop()
        if len(fields) != 5:
            raise ValueError(f"unexpected sacct column count on line {line_number}")
        raw_id, raw_state, raw_exit, raw_elapsed, raw_limit_minutes = fields
        if raw_id not in requested_set:
            parents = [
                job_id for job_id in requested if raw_id.startswith(f"{job_id}.")
            ]
            if len(parents) == 1 and raw_id != f"{parents[0]}.":
                continue
            raise ValueError(
                f"sacct returned an unrelated or ambiguous job row: {raw_id!r}"
            )
        if raw_id in rows:
            raise ValueError(f"sacct returned duplicate top-level job row {raw_id}")
        state_value = _normalize_state(raw_state)
        exit_value = _exit_code(raw_exit)
        if not raw_elapsed.isdecimal() or not raw_limit_minutes.isdecimal():
            raise ValueError(
                f"sacct returned non-integer raw duration for job {raw_id}"
            )
        rows[raw_id] = {
            "state": state_value,
            "exit_code": exit_value,
            "elapsed_seconds": int(raw_elapsed),
            "timelimit_seconds": int(raw_limit_minutes) * 60,
        }
    if set(rows) != requested_set:
        missing = len(requested_set - set(rows))
        raise ValueError(f"sacct omitted {missing} of the 20 requested top-level jobs")
    return rows


def _validate_accounting_rows(rows, job_ids):
    if not isinstance(rows, dict) or set(rows) != set(job_ids):
        raise ValueError("accounting provider must return the exact 20 receipt job ids")
    normalized = {}
    for job_id in job_ids:
        row = rows[job_id]
        _exact_fields(
            row,
            frozenset({"state", "exit_code", "elapsed_seconds", "timelimit_seconds"}),
            location=f"accounting[{job_id}]",
        )
        state_value = _normalize_state(row["state"])
        if state_value in _NONTERMINAL_STATES:
            raise RuntimeError("not all 20 seed-training jobs are terminal")
        exit_value = _exit_code(row["exit_code"])
        elapsed = _positive_int(
            row["elapsed_seconds"], location="elapsed_seconds", allow_zero=True
        )
        limit = _positive_int(row["timelimit_seconds"], location="timelimit_seconds")
        normalized[job_id] = {
            "state": state_value,
            "exit_code": exit_value,
            "elapsed_seconds": elapsed,
            "timelimit_seconds": limit,
        }
    return normalized


def _record_entry(binding, job, experiment, receipt):
    identity = (job.phase, job.key)
    receipt_event = receipt["by_identity"][identity]
    record_path, record, _ = seedext._load_train_record(binding, experiment)
    # Keep these checks here even after the seed-record hardening lands: this
    # tool must independently bind scheduler accounting to the receipt.
    if record.get("auxiliary_id") != seedext.EXPECTED_AUXILIARY_ID:
        raise ValueError("training record has the wrong auxiliary id")
    if record.get("condition") != experiment["model"]["condition"]:
        raise ValueError("training record has the wrong condition")
    if record.get("gpu_profile") != experiment["gpu_profile"]:
        raise ValueError("training record has the wrong GPU profile")
    runtime = record.get("runtime")
    _exact_fields(
        runtime,
        frozenset({"slurm_job_id", "partition", "node", "cuda_device", "environment"}),
        location=f"training record {record_path}.runtime",
    )
    if runtime["slurm_job_id"] != receipt_event["job_id"]:
        raise ValueError("training record Slurm id differs from its receipt")
    if runtime["partition"] != experiment["gpu_profile"]["partition"]:
        raise ValueError("training record partition differs from the locked plan")
    if runtime["environment"] != binding["spec"]["environment"]:
        raise ValueError("training record environment differs from the locked spec")
    for field in ("node", "cuda_device"):
        _nonempty_string(runtime[field], location=f"training record runtime.{field}")
    return {
        "dataset": experiment["dataset"]["name"],
        "model": experiment["model"]["name"],
        "training_seed": experiment["training_seed"],
        "receipt_job_id": receipt_event["job_id"],
        "record_slurm_job_id": runtime["slurm_job_id"],
        "train_record_sha256": _sha256_bytes(_read_regular(record_path)),
    }


def _record_set_sha256(records):
    if len(records) != EXPECTED_JOBS:
        raise RuntimeError("training record closure does not contain 20 identities")
    record_set_core = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "training_seed": row["training_seed"],
            "train_record_sha256": row["train_record_sha256"],
        }
        for row in records
    ]
    return _sha256_bytes(_canonical_json(record_set_core).encode())


def _load_records(binding, planned_jobs, receipt):
    records = [
        _record_entry(binding, job, experiment, receipt)
        for job, experiment in zip(
            planned_jobs, seedext.iter_experiments(binding["spec"]), strict=True
        )
    ]
    return records, _record_set_sha256(records)


def _expected_record_path(binding, experiment):
    return (
        Path(binding["spec"]["paths"]["train_root"])
        / experiment["dataset"]["name"]
        / experiment["model"]["name"]
        / f"seed-{experiment['training_seed']}"
        / "extension_record.json"
    )


def _load_terminal_records(binding, planned_jobs, receipt, accounting):
    """Load records after accounting, allowing absence only for failed jobs."""

    records = []
    for job, experiment in zip(
        planned_jobs, seedext.iter_experiments(binding["spec"]), strict=True
    ):
        receipt_event = receipt["by_identity"][(job.phase, job.key)]
        job_id = receipt_event["job_id"]
        scheduler = accounting[job_id]
        succeeded = (
            scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0"
        )
        try:
            entry = _record_entry(binding, job, experiment, receipt)
        except FileNotFoundError as error:
            record_path = _expected_record_path(binding, experiment)
            _reject_symlink_ancestors(record_path)
            if record_path.exists() or record_path.is_symlink():
                raise ValueError(
                    "failed-job training record exists but did not strictly validate"
                ) from error
            if succeeded:
                raise ValueError(
                    "successful seed-training job has no immutable training record"
                ) from error
            entry = {
                "dataset": experiment["dataset"]["name"],
                "model": experiment["model"]["name"],
                "training_seed": experiment["training_seed"],
                "receipt_job_id": job_id,
                "record_slurm_job_id": None,
                "train_record_sha256": None,
            }
        records.append(entry)
    return records, _record_set_sha256(records)


def _summary_stats(values):
    if len(values) != EXPECTED_JOBS:
        raise ValueError("summary statistics require exactly 20 values")
    return {
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "median_seconds": statistics.median(values),
        "total_seconds": sum(values),
    }


def _state_counts(jobs):
    counts = Counter(job["state"] for job in jobs)
    return [{"state": key, "count": counts[key]} for key in sorted(counts)]


def _event_core(
    *, parent_hash, spec_sha, receipt_sha, record_set_sha, records, accounting
):
    jobs = []
    for record in records:
        scheduler = accounting[record["receipt_job_id"]]
        jobs.append({**record, **scheduler})
    failed = [
        {
            "dataset": job["dataset"],
            "model": job["model"],
            "training_seed": job["training_seed"],
            "receipt_job_id": job["receipt_job_id"],
            "state": job["state"],
            "exit_code": job["exit_code"],
        }
        for job in jobs
        if job["state"] != "COMPLETED" or job["exit_code"] != "0:0"
    ]
    return {
        "terminal_event_schema_version": TERMINAL_EVENT_SCHEMA_VERSION,
        "artifact_type": "selectseg.seed_scheduler_terminal_accounting",
        "parent_log_sha256": parent_hash,
        "spec_lock_sha256": spec_sha,
        "train_submission_receipt_sha256": receipt_sha,
        "training_record_set_sha256": record_set_sha,
        "expected_jobs": EXPECTED_JOBS,
        "jobs": jobs,
        "state_counts": _state_counts(jobs),
        "duration_seconds": _summary_stats([job["elapsed_seconds"] for job in jobs]),
        "timelimit_seconds": _summary_stats([job["timelimit_seconds"] for job in jobs]),
        "campaign_complete": not failed,
        "failed_jobs": failed,
    }


def _build_event(core, *, now=None):
    event_id = _sha256_bytes(_canonical_json(core).encode())
    created = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _utc_timestamp(created, location="terminal event created_utc")
    return {
        **core,
        "terminal_event_id": event_id,
        "created_utc": created,
    }


def _validate_stats(value, *, location):
    _exact_fields(value, _SUMMARY_STAT_FIELDS, location=location)
    for field in _SUMMARY_STAT_FIELDS:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"{location}.{field} must be a finite non-negative number")
    for field in ("minimum_seconds", "maximum_seconds", "total_seconds"):
        if not isinstance(value[field], int):
            raise ValueError(f"{location}.{field} must be an integer number of seconds")
    doubled_median = 2 * value["median_seconds"]
    if not float(doubled_median).is_integer():
        raise ValueError(
            f"{location}.median_seconds must be an integer or half-integer"
        )
    if (
        value["minimum_seconds"] > value["median_seconds"]
        or value["median_seconds"] > value["maximum_seconds"]
    ):
        raise ValueError(f"{location} order statistics are inconsistent")
    if not (
        EXPECTED_JOBS * value["minimum_seconds"]
        <= value["total_seconds"]
        <= EXPECTED_JOBS * value["maximum_seconds"]
    ):
        raise ValueError(f"{location} total is inconsistent with its range")


def _validate_state_counts(value, *, expected_total, location):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    states = []
    total = 0
    for index, row in enumerate(value):
        _exact_fields(row, _STATE_COUNT_FIELDS, location=f"{location}[{index}]")
        state_value = _normalize_state(row["state"])
        if state_value not in _TERMINAL_STATES:
            raise ValueError(f"{location}[{index}] is not terminal")
        states.append(state_value)
        total += _positive_int(row["count"], location=f"{location}[{index}].count")
    if states != sorted(set(states)) or total != expected_total:
        raise ValueError(f"{location} is not canonical or has the wrong total")


def _validate_terminal_event(
    event,
    *,
    parent_hash=None,
    spec_sha=None,
    receipt_sha=None,
    record_set_sha=None,
    expected_job_ids=None,
    expected_job_bindings=None,
):
    _exact_fields(event, _TERMINAL_EVENT_FIELDS, location="terminal scheduler event")
    if event["terminal_event_schema_version"] != TERMINAL_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported terminal-event schema")
    if event["artifact_type"] != "selectseg.seed_scheduler_terminal_accounting":
        raise ValueError("unexpected terminal-event artifact type")
    _utc_timestamp(event["created_utc"], location="terminal_event.created_utc")
    core = {
        key: value
        for key, value in event.items()
        if key not in {"terminal_event_id", "created_utc"}
    }
    expected_id = _sha256_bytes(_canonical_json(core).encode())
    if _digest(event["terminal_event_id"], location="terminal_event_id") != expected_id:
        raise ValueError("terminal_event_id does not match its canonical core")
    bindings = (
        ("parent_log_sha256", parent_hash),
        ("spec_lock_sha256", spec_sha),
        ("train_submission_receipt_sha256", receipt_sha),
        ("training_record_set_sha256", record_set_sha),
    )
    for field, expected in bindings:
        actual = _digest(event[field], location=f"terminal_event.{field}")
        if expected is not None and actual != expected:
            raise ValueError(f"terminal event {field} binding mismatch")
    if event["expected_jobs"] != EXPECTED_JOBS:
        raise ValueError("terminal event expected_jobs must be 20")
    if not isinstance(event["jobs"], list) or len(event["jobs"]) != EXPECTED_JOBS:
        raise ValueError("terminal event must contain exactly 20 jobs")
    identities = set()
    receipt_ids = set()
    failed_expected = []
    for index, job in enumerate(event["jobs"]):
        _exact_fields(
            job, _PRIVATE_JOB_FIELDS, location=f"terminal_event.jobs[{index}]"
        )
        identity = (
            _nonempty_string(job["dataset"], location="job.dataset"),
            _nonempty_string(job["model"], location="job.model"),
            _positive_int(job["training_seed"], location="job.training_seed"),
        )
        identities.add(identity)
        receipt_id = _job_id(job["receipt_job_id"], location="job.receipt_job_id")
        record_job_id = job["record_slurm_job_id"]
        record_sha = job["train_record_sha256"]
        if (record_job_id is None) != (record_sha is None):
            raise ValueError(
                "terminal-event record id and digest must both be null or set"
            )
        if record_job_id is not None:
            if _job_id(record_job_id, location="job.record_slurm_job_id") != receipt_id:
                raise ValueError(
                    "record and receipt Slurm ids differ in terminal event"
                )
            _digest(record_sha, location="job.train_record_sha256")
        receipt_ids.add(receipt_id)
        state_value = _normalize_state(job["state"])
        if state_value not in _TERMINAL_STATES:
            raise ValueError("terminal event contains a nonterminal state")
        exit_value = _exit_code(job["exit_code"])
        _positive_int(
            job["elapsed_seconds"], location="job.elapsed_seconds", allow_zero=True
        )
        _positive_int(job["timelimit_seconds"], location="job.timelimit_seconds")
        succeeded = state_value == "COMPLETED" and exit_value == "0:0"
        if succeeded and record_job_id is None:
            raise ValueError("successful terminal job must bind a training record")
        if not succeeded:
            failed_expected.append(
                {
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "training_seed": job["training_seed"],
                    "receipt_job_id": receipt_id,
                    "state": state_value,
                    "exit_code": exit_value,
                }
            )
    if len(identities) != EXPECTED_JOBS or len(receipt_ids) != EXPECTED_JOBS:
        raise ValueError("terminal event job identities are not one-to-one")
    if expected_job_ids is not None:
        expected_ids = tuple(
            _job_id(value, location="expected terminal job id")
            for value in expected_job_ids
        )
        if (
            len(expected_ids) != EXPECTED_JOBS
            or len(set(expected_ids)) != EXPECTED_JOBS
            or receipt_ids != set(expected_ids)
        ):
            raise ValueError(
                "terminal event job ids differ from the base ledger/receipt"
            )
    if expected_job_bindings is not None:
        if not isinstance(expected_job_bindings, dict):
            raise ValueError("expected terminal job bindings must be a mapping")
        normalized_bindings = {}
        for identity, job_id in expected_job_bindings.items():
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise ValueError(
                    "expected terminal job binding has an invalid identity"
                )
            normalized_identity = (
                _nonempty_string(identity[0], location="expected job dataset"),
                _nonempty_string(identity[1], location="expected job model"),
                _positive_int(identity[2], location="expected job seed"),
            )
            if normalized_identity in normalized_bindings:
                raise ValueError("expected terminal job bindings repeat an identity")
            normalized_bindings[normalized_identity] = _job_id(
                job_id, location="expected terminal identity job id"
            )
        observed_bindings = {
            (job["dataset"], job["model"], job["training_seed"]): job["receipt_job_id"]
            for job in event["jobs"]
        }
        if (
            len(normalized_bindings) != EXPECTED_JOBS
            or normalized_bindings != observed_bindings
        ):
            raise ValueError(
                "terminal event identity-to-job bindings differ from the receipt"
            )
    record_set_core = [
        {
            "dataset": job["dataset"],
            "model": job["model"],
            "training_seed": job["training_seed"],
            "train_record_sha256": job["train_record_sha256"],
        }
        for job in event["jobs"]
    ]
    computed_record_set_sha = _sha256_bytes(_canonical_json(record_set_core).encode())
    if event["training_record_set_sha256"] != computed_record_set_sha:
        raise ValueError("terminal event training-record aggregate is inconsistent")
    if event["failed_jobs"] != failed_expected:
        raise ValueError("terminal event failed-job recovery set is inconsistent")
    if event["campaign_complete"] != (not failed_expected):
        raise ValueError("terminal event campaign_complete is inconsistent")
    _validate_state_counts(
        event["state_counts"],
        expected_total=EXPECTED_JOBS,
        location="terminal_event.state_counts",
    )
    if event["state_counts"] != _state_counts(event["jobs"]):
        raise ValueError("terminal event state counts are inconsistent")
    for field in ("duration_seconds", "timelimit_seconds"):
        _validate_stats(event[field], location=f"terminal_event.{field}")
    if event["duration_seconds"] != _summary_stats(
        [job["elapsed_seconds"] for job in event["jobs"]]
    ):
        raise ValueError("terminal event duration summary is inconsistent")
    if event["timelimit_seconds"] != _summary_stats(
        [job["timelimit_seconds"] for job in event["jobs"]]
    ):
        raise ValueError("terminal event time-limit summary is inconsistent")
    return event


def _load_ledger(path, *, receipt, spec_sha256, record_set_sha256=None):
    payload = _read_regular(path)
    rows, prefix_lengths = _strict_jsonl(payload, source=str(path))
    if len(rows) not in {2, 3}:
        raise ValueError(
            "private scheduler ledger must contain two base rows and at most one terminal event"
        )
    adjusted = _validate_adjustment_row(
        rows[0], receipt=receipt, spec_sha256=spec_sha256
    )
    first_prefix_hash = _sha256_bytes(payload[: prefix_lengths[0]])
    _validate_verification_row(
        rows[1],
        first_row=rows[0],
        first_prefix_sha256=first_prefix_hash,
        adjusted_job_ids=adjusted,
    )
    base_prefix = payload[: prefix_lengths[1]]
    base_hash = _sha256_bytes(base_prefix)
    terminal = None
    if len(rows) == 3:
        terminal = _validate_terminal_event(
            rows[2],
            parent_hash=base_hash,
            spec_sha=spec_sha256,
            receipt_sha=receipt["sha256"],
            record_set_sha=record_set_sha256,
            expected_job_ids=receipt["job_ids"],
            expected_job_bindings=receipt["job_bindings"],
        )
    return {
        "payload": payload,
        "sha256": _sha256_bytes(payload),
        "base_hash": base_hash,
        "terminal": terminal,
    }


def load_terminal_scheduler_event(
    private_ledger,
    *,
    expected_job_bindings,
    expected_spec_lock_sha256=None,
    expected_receipt_sha256=None,
    expected_record_set_sha256=None,
    require_complete=False,
):
    """Load the final event and bind every scientific identity to its receipt."""

    payload = _read_regular(private_ledger)
    rows, prefix_lengths = _strict_jsonl(payload, source=str(private_ledger))
    if len(rows) != 3:
        raise ValueError("private scheduler ledger has no unique terminal event")
    first_row = rows[0]
    first_spec_sha = _digest(
        first_row.get("spec_lock_sha256"),
        location="adjustment.spec_lock_sha256",
    )
    first_receipt_sha = _digest(
        first_row.get("train_submission_receipt_sha256"),
        location="adjustment.train_submission_receipt_sha256",
    )
    if expected_spec_lock_sha256 is not None and first_spec_sha != _digest(
        expected_spec_lock_sha256, location="expected spec-lock sha256"
    ):
        raise ValueError("scheduler ledger spec-lock binding mismatch")
    if expected_receipt_sha256 is not None and first_receipt_sha != _digest(
        expected_receipt_sha256, location="expected receipt sha256"
    ):
        raise ValueError("scheduler ledger receipt binding mismatch")
    preliminary_ids = first_row.get("job_ids")
    evidence = first_row.get("evidence")
    completed_ids = (
        evidence.get("completed_job_ids") if isinstance(evidence, dict) else None
    )
    if not isinstance(preliminary_ids, list) or not isinstance(completed_ids, list):
        raise ValueError("scheduler adjustment job evidence is malformed")
    synthetic_receipt = {
        "sha256": first_receipt_sha,
        "job_ids": tuple([*preliminary_ids, *completed_ids]),
    }
    adjusted = _validate_adjustment_row(
        first_row,
        receipt=synthetic_receipt,
        spec_sha256=first_spec_sha,
    )
    _validate_verification_row(
        rows[1],
        first_row=first_row,
        first_prefix_sha256=_sha256_bytes(payload[: prefix_lengths[0]]),
        adjusted_job_ids=adjusted,
    )
    parent_hash = _sha256_bytes(payload[: prefix_lengths[1]])
    event = _validate_terminal_event(
        rows[2],
        parent_hash=parent_hash,
        spec_sha=first_spec_sha,
        receipt_sha=first_receipt_sha,
        record_set_sha=expected_record_set_sha256,
        expected_job_ids=synthetic_receipt["job_ids"],
        expected_job_bindings=expected_job_bindings,
    )
    if require_complete and not event["campaign_complete"]:
        raise RuntimeError("seed training has terminal failures and is not complete")
    return {"event": event, "private_ledger_sha256": _sha256_bytes(payload)}


def _public_summary(event, *, private_ledger_sha256):
    successful = sum(
        job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
        for job in event["jobs"]
    )
    return {
        "summary_schema_version": PUBLIC_SUMMARY_SCHEMA_VERSION,
        "artifact_type": "selectseg.public_seed_scheduler_summary",
        "auxiliary_id": seedext.EXPECTED_AUXILIARY_ID,
        "status": "complete" if event["campaign_complete"] else "terminal_failures",
        "expected_jobs": EXPECTED_JOBS,
        "terminal_jobs": EXPECTED_JOBS,
        "successful_jobs": successful,
        "failed_jobs": EXPECTED_JOBS - successful,
        "state_counts": event["state_counts"],
        "duration_seconds": event["duration_seconds"],
        "timelimit_seconds": event["timelimit_seconds"],
        "bindings": {
            "private_ledger_sha256": private_ledger_sha256,
            "spec_lock_sha256": event["spec_lock_sha256"],
            "train_submission_receipt_sha256": event["train_submission_receipt_sha256"],
            "training_record_set_sha256": event["training_record_set_sha256"],
            "terminal_event_id": event["terminal_event_id"],
        },
    }


def _validate_public_summary(value):
    _exact_fields(value, _PUBLIC_FIELDS, location="public scheduler summary")
    if value["summary_schema_version"] != PUBLIC_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported public scheduler summary schema")
    if value["artifact_type"] != "selectseg.public_seed_scheduler_summary":
        raise ValueError("unexpected public scheduler summary artifact type")
    if value["auxiliary_id"] != seedext.EXPECTED_AUXILIARY_ID:
        raise ValueError("unexpected public scheduler summary auxiliary id")
    if value["status"] not in {"complete", "terminal_failures"}:
        raise ValueError("invalid public scheduler summary status")
    for field in ("expected_jobs", "terminal_jobs", "successful_jobs", "failed_jobs"):
        _positive_int(value[field], location=f"public_summary.{field}", allow_zero=True)
    if (
        value["expected_jobs"] != EXPECTED_JOBS
        or value["terminal_jobs"] != EXPECTED_JOBS
    ):
        raise ValueError(
            "public scheduler summary must account for exactly 20 terminal jobs"
        )
    if value["successful_jobs"] + value["failed_jobs"] != EXPECTED_JOBS:
        raise ValueError("public scheduler summary job counts are inconsistent")
    if (value["status"] == "complete") != (value["failed_jobs"] == 0):
        raise ValueError("public scheduler status and failed count are inconsistent")
    _validate_state_counts(
        value["state_counts"],
        expected_total=EXPECTED_JOBS,
        location="public_summary.state_counts",
    )
    state_counts = {row["state"]: row["count"] for row in value["state_counts"]}
    completed_count = state_counts.get("COMPLETED", 0)
    if value["successful_jobs"] > completed_count:
        raise ValueError(
            "public scheduler successful count exceeds completed state count"
        )
    if value["status"] == "complete" and state_counts != {"COMPLETED": EXPECTED_JOBS}:
        raise ValueError(
            "complete public scheduler summary requires 20 completed states"
        )
    _validate_stats(
        value["duration_seconds"], location="public_summary.duration_seconds"
    )
    _validate_stats(
        value["timelimit_seconds"], location="public_summary.timelimit_seconds"
    )
    _exact_fields(
        value["bindings"], _PUBLIC_BINDING_FIELDS, location="public_summary.bindings"
    )
    for field in _PUBLIC_BINDING_FIELDS:
        _digest(value["bindings"][field], location=f"public_summary.bindings.{field}")
    return value


def load_public_scheduler_summary(
    path,
    *,
    expected_private_ledger_sha256=None,
    expected_spec_lock_sha256=None,
    expected_receipt_sha256=None,
    expected_record_set_sha256=None,
    require_complete=False,
):
    """Strict public-summary loader with optional reverse bindings."""

    value = _validate_public_summary(_strict_json_file(path))
    expected = {
        "private_ledger_sha256": expected_private_ledger_sha256,
        "spec_lock_sha256": expected_spec_lock_sha256,
        "train_submission_receipt_sha256": expected_receipt_sha256,
        "training_record_set_sha256": expected_record_set_sha256,
    }
    for field, digest in expected.items():
        if digest is not None and value["bindings"][field] != _digest(
            digest, location=f"expected {field}"
        ):
            raise ValueError(f"public scheduler summary {field} binding mismatch")
    if require_complete and value["status"] != "complete":
        raise RuntimeError("public scheduler summary records terminal failures")
    return value


def load_scheduler_accounting_closure(
    private_ledger,
    public_summary,
    *,
    expected_job_bindings,
    expected_spec_lock_sha256=None,
    expected_receipt_sha256=None,
    expected_record_set_sha256=None,
    require_complete=False,
):
    """Load and cross-check the private event and its redacted public view."""

    private = load_terminal_scheduler_event(
        private_ledger,
        expected_spec_lock_sha256=expected_spec_lock_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_record_set_sha256=expected_record_set_sha256,
        expected_job_bindings=expected_job_bindings,
        require_complete=require_complete,
    )
    public = load_public_scheduler_summary(
        public_summary,
        expected_private_ledger_sha256=private["private_ledger_sha256"],
        expected_spec_lock_sha256=expected_spec_lock_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_record_set_sha256=expected_record_set_sha256,
        require_complete=require_complete,
    )
    expected_public = _public_summary(
        private["event"],
        private_ledger_sha256=private["private_ledger_sha256"],
    )
    if public != expected_public:
        raise ValueError("public scheduler summary differs from its private event")
    return {"private": private, "public": public}


def validate_complete_training_closure(
    binding,
    planned_jobs,
    *,
    expected_public_summary_sha256,
):
    """Validate the fixed scheduler closure before sealing checkpoints.

    The checkpoint gate deliberately accepts no path overrides.  It binds the
    canonical training receipt, current strict training records, private
    terminal event, and redacted public summary to the exact locked 20-job
    plan.  The operator-supplied public-summary digest prevents accidentally
    consuming different bytes from those reviewed after finalization.
    """

    expected_summary_sha = _digest(
        expected_public_summary_sha256,
        location="expected public scheduler-summary sha256",
    )
    jobs = tuple(planned_jobs)
    receipt = _validate_receipt(TRAIN_RECEIPT, jobs)
    records, record_set_sha = _load_records(binding, jobs, receipt)
    if len(records) != EXPECTED_JOBS:
        raise RuntimeError("scheduler closure requires exactly 20 training records")

    public_payload = _read_regular(PUBLIC_SUMMARY)
    observed_summary_sha = _sha256_bytes(public_payload)
    if observed_summary_sha != expected_summary_sha:
        raise ValueError("public scheduler summary SHA-256 mismatch")

    closure = load_scheduler_accounting_closure(
        PRIVATE_LEDGER,
        PUBLIC_SUMMARY,
        expected_job_bindings=receipt["job_bindings"],
        expected_spec_lock_sha256=binding["sha256"],
        expected_receipt_sha256=receipt["sha256"],
        expected_record_set_sha256=record_set_sha,
        require_complete=True,
    )

    # Detect changes to the three fixed closure files across validation.  The
    # checkpoint writer immediately revalidates every training record/output.
    if _sha256_bytes(_read_regular(TRAIN_RECEIPT)) != receipt["sha256"]:
        raise RuntimeError("training receipt changed during scheduler closure gate")
    if (
        _sha256_bytes(_read_regular(PRIVATE_LEDGER))
        != closure["private"]["private_ledger_sha256"]
    ):
        raise RuntimeError("private scheduler ledger changed during closure gate")
    if _sha256_bytes(_read_regular(PUBLIC_SUMMARY)) != expected_summary_sha:
        raise RuntimeError("public scheduler summary changed during closure gate")
    return {
        "public_summary": closure["public"],
        "public_summary_sha256": observed_summary_sha,
        "private_ledger_sha256": closure["private"]["private_ledger_sha256"],
        "train_submission_receipt_sha256": receipt["sha256"],
        "training_record_set_sha256": record_set_sha,
    }


def _atomic_write_new_or_identical(path, value):
    destination = Path(path)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _reject_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent)
    if destination.exists():
        existing = _read_regular(destination)
        if existing != encoded:
            raise FileExistsError(
                f"refusing to replace conflicting output {destination}"
            )
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read_regular(destination)
            if existing != encoded:
                raise FileExistsError(
                    f"refusing to replace conflicting output {destination}"
                )
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _preflight_new_or_identical(path, value):
    destination = Path(path)
    _reject_symlink_ancestors(destination)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if destination.exists() and _read_regular(destination) != encoded:
        raise FileExistsError(f"refusing to replace conflicting output {destination}")


def _atomic_logical_append(path, *, expected_payload, event):
    destination = Path(path)
    _reject_symlink_ancestors(destination)
    lock_path = destination.with_name(f".{destination.name}.lock")
    _reject_symlink_ancestors(lock_path)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _read_regular(destination)
        if current != expected_payload:
            raise RuntimeError("private scheduler ledger changed during finalization")
        event_line = (_canonical_json(event) + "\n").encode()
        updated = current + event_line
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            if _read_regular(destination) != current:
                raise RuntimeError("private scheduler ledger changed before commit")
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    return updated


def finalize_scheduler_ledger(
    *,
    spec_lock=seedext.DEFAULT_SPEC_LOCK,
    expected_spec_lock_sha256=None,
    receipt=TRAIN_RECEIPT,
    private_ledger=PRIVATE_LEDGER,
    public_summary=PUBLIC_SUMMARY,
    provider=query_sacct,
    write=False,
    now=None,
):
    expected_spec = expected_spec_lock_sha256
    if (
        expected_spec is None
        and Path(spec_lock).as_posix() == seedext.DEFAULT_SPEC_LOCK
    ):
        expected_spec = seedext.DEFAULT_SPEC_LOCK_SHA256
    if expected_spec is None:
        raise ValueError("a non-default spec lock requires its expected SHA-256")
    binding = seedext.load_spec_lock(spec_lock, expected_sha256=expected_spec)
    planned_jobs = plan_training_jobs(binding)
    receipt_result = _validate_receipt(receipt, planned_jobs)
    # Validate the immutable base ledger before consulting live accounting or
    # any training output.  A genuinely failed job may never have reached the
    # write-once record publication step.
    ledger = _load_ledger(
        private_ledger,
        receipt=receipt_result,
        spec_sha256=binding["sha256"],
    )
    if ledger["terminal"] is None:
        accounting = _validate_accounting_rows(
            provider(receipt_result["job_ids"]), receipt_result["job_ids"]
        )
        records, record_set_sha = _load_terminal_records(
            binding, planned_jobs, receipt_result, accounting
        )
        core = _event_core(
            parent_hash=ledger["base_hash"],
            spec_sha=binding["sha256"],
            receipt_sha=receipt_result["sha256"],
            record_set_sha=record_set_sha,
            records=records,
            accounting=accounting,
        )
        event = _build_event(core, now=now)
        _validate_terminal_event(
            event,
            parent_hash=ledger["base_hash"],
            spec_sha=binding["sha256"],
            receipt_sha=receipt_result["sha256"],
            record_set_sha=record_set_sha,
            expected_job_ids=receipt_result["job_ids"],
            expected_job_bindings=receipt_result["job_bindings"],
        )
        final_payload = ledger["payload"] + (_canonical_json(event) + "\n").encode()
    else:
        event = ledger["terminal"]
        record_set_sha = event["training_record_set_sha256"]
        final_payload = ledger["payload"]
    final_ledger_sha = _sha256_bytes(final_payload)
    summary = _validate_public_summary(
        _public_summary(event, private_ledger_sha256=final_ledger_sha)
    )
    if write and not event["campaign_complete"]:
        raise RuntimeError(
            "refusing to persist terminal failures; design and audit a recovery "
            "receipt before closing the scheduler ledger"
        )
    if write:
        # A conflicting public destination must not cause a private append first.
        _preflight_new_or_identical(public_summary, summary)
        if ledger["terminal"] is None:
            committed = _atomic_logical_append(
                private_ledger,
                expected_payload=ledger["payload"],
                event=event,
            )
            if committed != final_payload:
                raise RuntimeError("private scheduler append produced unexpected bytes")
        _atomic_write_new_or_identical(public_summary, summary)
        loaded = load_public_scheduler_summary(
            public_summary,
            expected_private_ledger_sha256=final_ledger_sha,
            expected_spec_lock_sha256=binding["sha256"],
            expected_receipt_sha256=receipt_result["sha256"],
            expected_record_set_sha256=record_set_sha,
        )
        if loaded != summary:
            raise RuntimeError(
                "published scheduler summary failed round-trip validation"
            )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-lock", default=seedext.DEFAULT_SPEC_LOCK)
    parser.add_argument("--expected-spec-lock-sha256")
    parser.add_argument("--receipt", default=str(TRAIN_RECEIPT))
    parser.add_argument("--private-ledger", default=str(PRIVATE_LEDGER))
    parser.add_argument("--public-summary", default=str(PUBLIC_SUMMARY))
    parser.add_argument(
        "--write",
        action="store_true",
        help="append/publish only after every strict gate passes",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = finalize_scheduler_ledger(
        spec_lock=args.spec_lock,
        expected_spec_lock_sha256=args.expected_spec_lock_sha256,
        receipt=args.receipt,
        private_ledger=args.private_ledger,
        public_summary=args.public_summary,
        write=args.write,
    )
    print(_canonical_json(summary))
    if summary["status"] != "complete":
        raise RuntimeError(
            "terminal accounting recorded failures; campaign is not complete"
        )
    return summary


if __name__ == "__main__":
    main()
