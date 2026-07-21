"""Plan or submit one Slurm job per binary simulation.

``freeze`` plans one GPU inference job per model condition.  Once those jobs
finish, ``lock`` consumes an explicit list of frozen artifact manifests and
writes an immutable campaign lock.  ``common`` then plans one M-independent
CPU scoring job per artifact, while ``score`` expands the locked Cartesian
product ``artifact x gamma x M x seed`` into independent M-specific CPU jobs.
After scoring, ``assemble`` derives each condition's four content-addressed
shard paths from the lock (never from a glob), and ``diagnose`` plans one
read-only diagnostic per locked artifact.  Every compute phase uses one
independent Slurm job per planned row; Slurm arrays are intentionally not used.

All submission phases are dry-run by default. Schema v1 retains its historical
execution behavior. Schema v2 supports either a fail-closed scheduler preview
or a real ``scientific-input-locked`` campaign: the latter verifies the exact
dataset, checkpoint, base-model, source, and environment bindings before a
whole-wave scheduler preflight, canonical receipt, or real ``sbatch`` call.
Lock files are written only with ``--write-lock`` and are never overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CONFIG_SCHEMA_VERSION = 1
CANDIDATE_CONFIG_SCHEMA_VERSION = 2
LOCK_SCHEMA_VERSION = 1
SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 1
RUNTIME_RECEIPT_SCHEMA_VERSION = 2
SCHEDULER_PREVIEW_ONLY = "scheduler-preview-only"
SCIENTIFIC_INPUT_LOCKED = "scientific-input-locked"
EXPECTED_PROTOCOL = {
    "gamma_values": [0.5],
    "m_values": [2, 8, 32],
    "quadrature_rule": "midpoint-v1",
    "seeds": [0],
}
GPU_ACCOUNT = "ssafo"
DEFAULT_CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
GPU_PARTITION_CANDIDATES = ("saffo-a100", "apollo_agate")
CPU_PARTITION_CANDIDATES = ("amdsmall", "agsmall", "msismall", "saffo-2tb")
SCHEDULER_PREFLIGHT_TIMEOUT_SECONDS = 45
SCHEDULER_PREFLIGHT_MAX_DISTINCT_LINES = 20
SCHEDULER_PREFLIGHT_MAX_LINE_CHARACTERS = 400
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "config_schema_version",
        "campaign_id",
        "protocol",
        "estimator_spec",
        "gpu_partitions",
        "paths",
        "conditions",
    }
)
PARTITION_CANDIDATE_CONFIG_FIELDS = frozenset(
    {"gpu_partition_candidates", "cpu_partition_candidates"}
)
OPTIONAL_CONFIG_FIELDS = frozenset(
    {
        "cpu_partitions",
        "execution_policy",
        "data_root",
        "scientific_input_lock",
        *PARTITION_CANDIDATE_CONFIG_FIELDS,
    }
)
REQUIRED_SCIENTIFIC_INPUT_BINDING_FIELDS = frozenset({"path", "sha256"})
REQUIRED_CONDITION_FIELDS = frozenset(
    {
        "dataset",
        "condition",
        "model",
        "checkpoint",
        "batch_size",
        "expected_num_samples",
    }
)
OPTIONAL_CONDITION_FIELDS = frozenset({"expected_dataset_samples", "freeze_limit"})
REQUIRED_PATH_FIELDS = frozenset(
    {
        "artifact_output_root",
        "common_output_root",
        "simulation_output_root",
        "assembly_output_root",
    }
)
REQUIRED_ESTIMATOR_FIELDS = frozenset(
    {
        "schema_version",
        "estimator_id",
        "target_measure",
        "rule",
        "randomized",
        "required_seed",
    }
)
REQUIRED_LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "campaign_id",
        "config",
        "protocol",
        "estimator",
        "paths",
        "artifacts",
    }
)
REQUIRED_SCIENTIFIC_LOCK_FIELDS = REQUIRED_LOCK_FIELDS | {"scientific_input"}
REQUIRED_LOCK_SCIENTIFIC_INPUT_FIELDS = frozenset(
    {"root_lock_path", "root_lock_sha256", "science_projection_sha256"}
)
REQUIRED_LOCK_CONFIG_FIELDS = frozenset({"path", "sha256"})
REQUIRED_LOCK_ESTIMATOR_FIELDS = frozenset(
    {"spec_path", "spec_sha256", "estimator_id", "target_measure"}
)
REQUIRED_LOCK_ARTIFACT_FIELDS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "artifact_id",
        "dataset",
        "condition",
        "model",
        "split",
        "checkpoint_sha256",
        "source_sha256",
        "sample_id_sha256",
        "num_samples",
    }
)
REQUIRED_SCIENTIFIC_LOCK_ARTIFACT_FIELDS = REQUIRED_LOCK_ARTIFACT_FIELDS | {
    "scientific_input"
}
REQUIRED_RECEIPT_FIELDS = frozenset(
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
REQUIRED_RUNTIME_RECEIPT_FIELDS = frozenset(
    {
        "receipt_schema_version",
        "created_utc",
        "campaign_id",
        "config_sha256",
        "event",
        "phase",
        "key",
        "command",
        "attempt",
        "authorization",
        "job_id",
        "predecessor_job_id",
        "scheduler_state",
        "scheduler_exit_code",
        "scheduler_job_name",
        "scheduler_account",
        "scheduler_partition",
        "scheduler_nodes",
        "scheduler_reason",
    }
)
RUNTIME_RECEIPT_EVENTS = frozenset(
    {
        "submitting",
        "submitted",
        "submission_recovered",
        "submission_failed",
        "completed",
        "failed",
    }
)
RUNTIME_AUTHORIZATIONS = frozenset(
    {"initial", "retry_failed_job_id", "retry_submission_failure"}
)
SLURM_ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "SIGNALING",
        "STAGE_OUT",
    }
)
SLURM_FAILED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "STOPPED",
        "TIMEOUT",
    }
)
SLURM_JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:;[A-Za-z0-9_.-]+)?$")


@dataclass(frozen=True)
class Config:
    path: Path
    sha256: str
    data: dict


@dataclass(frozen=True)
class PlannedJob:
    phase: str
    key: tuple
    command: tuple[str, ...]


@dataclass(frozen=True)
class SchedulerObservation:
    disposition: str
    state: str | None = None
    exit_code: str | None = None
    partition: str | None = None
    nodes: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SchedulerJobRecord:
    job_id: str
    job_name: str
    account: str
    partition: str
    state: str
    exit_code: str | None = None
    nodes: str | None = None
    reason: str | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/binary_midpoint_main.json")
    parser.add_argument(
        "--phase",
        choices=("freeze", "lock", "common", "score", "assemble", "diagnose"),
        default="freeze",
    )
    parser.add_argument(
        "--artifact-manifest",
        action="append",
        default=[],
        help="explicit frozen manifest; repeat once per configured condition",
    )
    parser.add_argument("--campaign-lock")
    parser.add_argument(
        "--write-lock",
        help="write the verified lock here during --phase lock (no overwrite)",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--submit",
        action="store_true",
        help="call sbatch; without this flag submission phases are dry-runs",
    )
    execution.add_argument(
        "--scheduler-preflight-only",
        action="store_true",
        help=(
            "schema-v2: run every final command through "
            "sbatch --test-only without opening a receipt or submitting a job"
        ),
    )
    execution.add_argument(
        "--reconcile",
        action="store_true",
        help=(
            "schema-v2 only: query sacct/squeue and append observed terminal "
            "states without submitting jobs"
        ),
    )
    execution.add_argument(
        "--recover-submitted-job-id",
        metavar="JOB_ID",
        help=(
            "schema-v2 only: explicitly bind the one dangling submission "
            "intent to this Slurm job after scheduler identity verification"
        ),
    )
    parser.add_argument(
        "--receipt",
        help=(
            "phase-specific append-only receipt; required with --submit and "
            "used to prevent blind duplicate resubmission"
        ),
    )
    parser.add_argument(
        "--retry-failed-job-id",
        action="append",
        default=[],
        help=(
            "schema-v2 only: explicitly authorize replacement of this exact "
            "terminal failed Slurm job; repeat for multiple jobs"
        ),
    )
    parser.add_argument(
        "--retry-submission-failure",
        action="store_true",
        help=(
            "schema-v2 only: explicitly retry attempts for which sbatch itself "
            "failed and therefore returned no Slurm job id"
        ),
    )
    parser.add_argument(
        "--diagnostic-output-root",
        default="outputs/binary_diagnostics",
        help=(
            "output root used only by --phase diagnose; diagnostic inputs and "
            "their expected hashes are still read exclusively from the lock"
        ),
    )
    return parser.parse_args(argv)


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text, *, source):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _nonempty_string(value, *, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _positive_integer(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _assert_finite(value, *, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, location=f"{location}[{index}]")


def _portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _project_path(config_path, value):
    raw = Path(_nonempty_string(value, location="configured path"))
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    repository_candidate = (config_path.parent.parent / raw).resolve()
    if cwd_candidate.exists() or not repository_candidate.exists():
        return cwd_candidate
    return repository_candidate


def _validate_protocol(protocol, *, location):
    if not isinstance(protocol, dict) or protocol != EXPECTED_PROTOCOL:
        raise ValueError(
            f"{location} must equal the predeclared main protocol "
            f"{_canonical_json(EXPECTED_PROTOCOL)}"
        )


def _runtime_campaign_root(config):
    """Return the single schema-v2 output root that anchors runtime receipts."""

    if config.data["config_schema_version"] != CANDIDATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("canonical runtime receipts are defined only for schema v2")
    output_roots = tuple(
        _project_path(config.path, config.data["paths"][field])
        for field in sorted(REQUIRED_PATH_FIELDS)
    )
    parents = {path.parent for path in output_roots}
    if len(parents) != 1:
        raise ValueError(
            "schema-v2 output roots must be sibling directories under one campaign root"
        )
    return next(iter(parents))


def canonical_runtime_receipt_path(config, phase):
    """Derive the only accepted schema-v2 receipt pathname for a phase."""

    if phase not in {"freeze", "common", "score", "assemble", "diagnose"}:
        raise ValueError(f"phase {phase!r} has no schema-v2 runtime receipt")
    return _runtime_campaign_root(config) / "receipts" / f"{phase}.jsonl"


def _validated_runtime_receipt_path(config, jobs, receipt_path):
    if receipt_path is None:
        raise ValueError("schema-v2 execution requires its canonical --receipt path")
    jobs = _validate_runtime_jobs(jobs)
    phases = {job.phase for job in jobs}
    if len(phases) != 1:
        raise ValueError("a schema-v2 receipt may contain exactly one planned phase")
    phase = next(iter(phases))
    expected = canonical_runtime_receipt_path(config, phase)
    observed = Path(os.path.abspath(os.path.normpath(os.fspath(receipt_path))))
    if observed != expected:
        raise ValueError(
            "schema-v2 receipt path substitution rejected: expected "
            f"{expected}, received {observed}"
        )
    if expected.parent.is_symlink():
        raise ValueError(
            f"schema-v2 receipt directory must not be a symlink: {expected.parent}"
        )
    return expected


def load_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"campaign config does not exist: {path}")
    raw = path.read_bytes()
    data = _loads_strict(raw.decode("utf-8"), source=str(path))
    keys = set(data) if isinstance(data, dict) else set()
    if (
        not isinstance(data, dict)
        or not REQUIRED_CONFIG_FIELDS <= keys
        or keys - REQUIRED_CONFIG_FIELDS - OPTIONAL_CONFIG_FIELDS
    ):
        raise ValueError(
            f"config must contain required fields {sorted(REQUIRED_CONFIG_FIELDS)} "
            f"and only optional fields {sorted(OPTIONAL_CONFIG_FIELDS)}"
        )
    _assert_finite(data, location=str(path))
    config_schema_version = data["config_schema_version"]
    if isinstance(config_schema_version, bool) or config_schema_version not in (
        CONFIG_SCHEMA_VERSION,
        CANDIDATE_CONFIG_SCHEMA_VERSION,
    ):
        raise ValueError(
            "config_schema_version must equal "
            f"{CONFIG_SCHEMA_VERSION} or {CANDIDATE_CONFIG_SCHEMA_VERSION}"
        )
    candidate_fields = keys & PARTITION_CANDIDATE_CONFIG_FIELDS
    has_execution_policy = "execution_policy" in keys
    execution_policy = data.get("execution_policy")
    if candidate_fields and config_schema_version != CANDIDATE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "partition candidate fields require config_schema_version "
            f"{CANDIDATE_CONFIG_SCHEMA_VERSION}"
        )
    if has_execution_policy and (
        config_schema_version != CANDIDATE_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(
            "execution_policy requires config_schema_version "
            f"{CANDIDATE_CONFIG_SCHEMA_VERSION}"
        )
    if (
        config_schema_version == CANDIDATE_CONFIG_SCHEMA_VERSION
        and candidate_fields != PARTITION_CANDIDATE_CONFIG_FIELDS
    ):
        raise ValueError(
            "gpu_partition_candidates and cpu_partition_candidates must be "
            "provided together"
        )
    if config_schema_version == CANDIDATE_CONFIG_SCHEMA_VERSION and (
        execution_policy not in {SCHEDULER_PREVIEW_ONLY, SCIENTIFIC_INPUT_LOCKED}
    ):
        raise ValueError(
            "schema-v2 execution_policy must equal "
            f"{SCHEDULER_PREVIEW_ONLY!r} or {SCIENTIFIC_INPUT_LOCKED!r}"
        )
    scientific_fields = {"data_root", "scientific_input_lock"} & keys
    if config_schema_version != CANDIDATE_CONFIG_SCHEMA_VERSION and scientific_fields:
        raise ValueError("scientific input bindings require config_schema_version 2")
    if execution_policy == SCHEDULER_PREVIEW_ONLY and scientific_fields:
        raise ValueError(
            "scheduler-preview-only configs cannot claim scientific input bindings"
        )
    if execution_policy == SCIENTIFIC_INPUT_LOCKED:
        if scientific_fields != {"data_root", "scientific_input_lock"}:
            raise ValueError(
                "scientific-input-locked configs require data_root and "
                "scientific_input_lock together"
            )
        data_root = data["data_root"]
        if data_root != "data":
            raise ValueError("the locked binary campaign requires data_root='data'")
        binding = data["scientific_input_lock"]
        if (
            not isinstance(binding, dict)
            or set(binding) != REQUIRED_SCIENTIFIC_INPUT_BINDING_FIELDS
        ):
            raise ValueError(
                "scientific_input_lock must contain exactly path and sha256"
            )
        _nonempty_string(binding["path"], location=f"{path}.scientific_input_lock.path")
        _digest(binding["sha256"], location=f"{path}.scientific_input_lock.sha256")
    _nonempty_string(data["campaign_id"], location=f"{path}.campaign_id")
    _validate_protocol(data["protocol"], location=f"{path}.protocol")
    if (
        not isinstance(data["paths"], dict)
        or set(data["paths"]) != REQUIRED_PATH_FIELDS
    ):
        raise ValueError(
            f"{path}.paths must contain exactly {sorted(REQUIRED_PATH_FIELDS)}"
        )
    for field in REQUIRED_PATH_FIELDS:
        _nonempty_string(data["paths"][field], location=f"{path}.paths.{field}")
    _nonempty_string(data["estimator_spec"], location=f"{path}.estimator_spec")
    partitions = data["gpu_partitions"]
    if (
        not isinstance(partitions, list)
        or not partitions
        or not all(isinstance(value, str) and value.strip() for value in partitions)
        or len(partitions) != len(set(partitions))
    ):
        raise ValueError(f"{path}.gpu_partitions must be a non-empty unique list")
    cpu_partitions = data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
    if (
        not isinstance(cpu_partitions, list)
        or not cpu_partitions
        or not all(isinstance(value, str) and value.strip() for value in cpu_partitions)
        or len(cpu_partitions) != len(set(cpu_partitions))
    ):
        raise ValueError(f"{path}.cpu_partitions must be a non-empty unique list")
    if candidate_fields:
        expected_candidates = {
            "gpu_partition_candidates": list(GPU_PARTITION_CANDIDATES),
            "cpu_partition_candidates": list(CPU_PARTITION_CANDIDATES),
        }
        for field, expected in expected_candidates.items():
            if data[field] != expected:
                raise ValueError(f"{path}.{field} must equal {expected!r}")
    conditions = data["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"{path}.conditions must be a non-empty list")
    seen = set()
    for index, condition in enumerate(conditions):
        location = f"{path}.conditions[{index}]"
        keys = set(condition) if isinstance(condition, dict) else set()
        if (
            not isinstance(condition, dict)
            or not REQUIRED_CONDITION_FIELDS <= keys
            or keys - REQUIRED_CONDITION_FIELDS - OPTIONAL_CONDITION_FIELDS
        ):
            raise ValueError(
                f"{location} must contain required fields "
                f"{sorted(REQUIRED_CONDITION_FIELDS)} and only optional fields "
                f"{sorted(OPTIONAL_CONDITION_FIELDS)}"
            )
        for field in ("dataset", "condition", "model"):
            _nonempty_string(condition[field], location=f"{location}.{field}")
        if condition["checkpoint"] is not None:
            _nonempty_string(condition["checkpoint"], location=f"{location}.checkpoint")
        _positive_integer(condition["batch_size"], location=f"{location}.batch_size")
        artifact_samples = _positive_integer(
            condition["expected_num_samples"],
            location=f"{location}.expected_num_samples",
        )
        dataset_samples = _positive_integer(
            condition.get("expected_dataset_samples", artifact_samples),
            location=f"{location}.expected_dataset_samples",
        )
        freeze_limit = condition.get("freeze_limit")
        if freeze_limit is None:
            if artifact_samples != dataset_samples:
                raise ValueError(
                    f"{location} requires freeze_limit when artifact and full-dataset "
                    "sample counts differ"
                )
        else:
            freeze_limit = _positive_integer(
                freeze_limit, location=f"{location}.freeze_limit"
            )
            if freeze_limit != artifact_samples or freeze_limit > dataset_samples:
                raise ValueError(
                    f"{location}.freeze_limit must equal expected_num_samples and "
                    "not exceed expected_dataset_samples"
                )
        if condition["model"] not in {"clipseg", "deeplabv3"}:
            raise ValueError(f"{location}.model is unsupported")
        expected_condition = {
            ("clipseg", False): "clipseg-general",
            ("clipseg", True): "clipseg-target",
            ("deeplabv3", False): "deeplabv3-external",
            ("deeplabv3", True): "deeplabv3-target",
        }[(condition["model"], condition["checkpoint"] is not None)]
        if condition["condition"] != expected_condition:
            raise ValueError(
                f"{location}.condition must be {expected_condition!r} for its "
                "model/checkpoint combination"
            )
        key = (condition["dataset"], condition["condition"])
        if key in seen:
            raise ValueError(f"duplicate configured condition {key}")
        seen.add(key)
    config = Config(path=path, sha256=_sha256_bytes(raw), data=data)
    if config_schema_version == CANDIDATE_CONFIG_SCHEMA_VERSION:
        _runtime_campaign_root(config)
    return config


def _rotated_partition_request(partitions, index):
    partitions = tuple(partitions)
    if not partitions:
        raise ValueError("partition candidates must not be empty")
    offset = index % len(partitions)
    return ",".join((*partitions[offset:], *partitions[:offset]))


def _gpu_partition_request(config, index):
    partitions = config.data.get(
        "gpu_partition_candidates", config.data["gpu_partitions"]
    )
    if "gpu_partition_candidates" in config.data:
        return _rotated_partition_request(partitions, index)
    return ",".join(partitions)


def _cpu_partition_request(config, index):
    candidates = config.data.get("cpu_partition_candidates")
    if candidates is not None:
        # Rotate the three general CPU queues so a complete wave does not give
        # every job the same first preference.  Keep the private 2-TB queue as
        # a fallback in every request: current Slurm test-only forecasts show
        # that placing it first can reserve an otherwise runnable job far in
        # the future even when a general queue can start immediately.
        primary, private_fallback = candidates[:-1], candidates[-1]
        if not primary:
            raise ValueError("CPU candidates require a non-private first choice")
        return f"{_rotated_partition_request(primary, index)},{private_fallback}"
    partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
    return partitions[index % len(partitions)]


def _scientific_plan_binding(config, *, mode="fast"):
    """Verify and load the pre-freeze lock before planning any locked job."""

    if config.data.get("execution_policy") != SCIENTIFIC_INPUT_LOCKED:
        raise ValueError("campaign is not authorized by a scientific input lock")
    declared = config.data["scientific_input_lock"]
    lock_path = _project_path(config.path, declared["path"])
    expected_sha256 = declared["sha256"]
    from selectseg.scientific_inputs import load_root_lock, verify_root_lock

    verification = verify_root_lock(
        lock_path,
        expected_sha256=expected_sha256,
        mode=mode,
    )
    if verification["campaign_id"] != config.data["campaign_id"]:
        raise ValueError("scientific input lock has a different campaign_id")
    binding = load_root_lock(lock_path, expected_sha256=expected_sha256)
    return lock_path, expected_sha256, binding


def _scheduler_job_name(config, base):
    """Keep v1 names byte-stable and bind v2 names to exact config bytes."""

    if config.data["config_schema_version"] != CANDIDATE_CONFIG_SCHEMA_VERSION:
        return base[:128]
    suffix = f"-c{config.sha256[:16]}"
    return f"{base[: 128 - len(suffix)]}{suffix}"


def _load_estimator(config):
    path = _project_path(config.path, config.data["estimator_spec"])
    if not path.is_file():
        raise FileNotFoundError(f"estimator spec does not exist: {path}")
    raw = path.read_bytes()
    spec = _loads_strict(raw.decode("utf-8"), source=str(path))
    if not isinstance(spec, dict) or set(spec) != REQUIRED_ESTIMATOR_FIELDS:
        raise ValueError(
            f"estimator spec must contain exactly {sorted(REQUIRED_ESTIMATOR_FIELDS)}"
        )
    expected = {
        "schema_version": 1,
        "estimator_id": "midpoint-v1",
        "target_measure": "uniform-threshold",
        "rule": "midpoint",
        "randomized": False,
        "required_seed": 0,
    }
    if spec != expected:
        raise ValueError("the main campaign requires the frozen midpoint-v1 spec")
    return path, _sha256_bytes(raw), spec


def plan_freeze_jobs(config):
    output_root = config.data["paths"]["artifact_output_root"]
    jobs = []
    scientific_plan = None
    if config.data.get("execution_policy") == SCIENTIFIC_INPUT_LOCKED:
        scientific_plan = _scientific_plan_binding(config, mode="fast")
    for condition_index, condition in enumerate(config.data["conditions"]):
        partition_request = _gpu_partition_request(config, condition_index)
        checkpoint = condition["checkpoint"] or "-"
        key = (condition["dataset"], condition["condition"], partition_request)
        job_name = f"selseg-freeze-{condition['dataset']}-{condition['condition']}"
        command = [
            "sbatch",
            "--parsable",
            "--job-name",
            _scheduler_job_name(config, job_name),
            "--partition",
            partition_request,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/freeze_binary_maps.sbatch",
            condition["model"],
            condition["dataset"],
            checkpoint,
            "--output-dir",
            output_root,
            "--num-workers",
            "4",
            "--batch-size",
            str(condition["batch_size"]),
            "--expected-num-samples",
            str(
                condition.get(
                    "expected_dataset_samples", condition["expected_num_samples"]
                )
            ),
        ]
        if condition.get("freeze_limit") is not None:
            command.extend(["--limit", str(condition["freeze_limit"])])
        if scientific_plan is not None:
            from selectseg.scientific_inputs import condition_input_identity

            lock_path, lock_sha256, binding = scientific_plan
            identity = condition_input_identity(
                binding,
                dataset=condition["dataset"],
                model=condition["model"],
                condition=condition["condition"],
            )
            command.extend(
                [
                    "--data-root",
                    config.data["data_root"],
                    "--campaign-config",
                    str(config.path),
                    "--expected-campaign-config-sha256",
                    config.sha256,
                    "--scientific-input-lock",
                    str(lock_path),
                    "--expected-scientific-input-lock-sha256",
                    lock_sha256,
                    "--expected-condition-input-sha256",
                    identity["scientific_input_sha256"],
                ]
            )
        jobs.append(PlannedJob(phase="freeze", key=key, command=tuple(command)))
    return tuple(jobs)


def _load_frozen_artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen artifact manifest does not exist: {path}")
    # Campaign planning validates immutable manifest bytes and structure only.
    # Decompressing tens of gigabytes on a login node is both redundant and
    # unsafe; each independent scorer validates its payloads while streaming.
    from selectseg.binary_artifacts import load_binary_artifact

    canonical_artifact = load_binary_artifact(path, validate_payloads=False)
    raw = path.read_bytes()
    if canonical_artifact.manifest_sha256 != _sha256_bytes(raw):
        raise RuntimeError("canonical artifact reader returned an inconsistent hash")
    return path, canonical_artifact.manifest_sha256, canonical_artifact.manifest


def build_campaign_lock(config, artifact_manifests):
    if len(artifact_manifests) != len(config.data["conditions"]):
        raise ValueError(
            "--artifact-manifest must be supplied exactly once per configured "
            f"condition ({len(config.data['conditions'])} required)"
        )
    resolved = [Path(path).resolve() for path in artifact_manifests]
    if len(set(resolved)) != len(resolved):
        raise ValueError("artifact manifest inputs must be distinct")
    loaded = [_load_frozen_artifact(path) for path in resolved]
    by_key = {}
    for manifest_path, manifest_sha, manifest in loaded:
        key = (manifest["dataset"], manifest["condition"])
        if key in by_key:
            raise ValueError(f"duplicate frozen artifact for condition {key}")
        by_key[key] = (manifest_path, manifest_sha, manifest)

    scientifically_locked = (
        config.data.get("execution_policy") == SCIENTIFIC_INPUT_LOCKED
    )
    scientific_plan = (
        _scientific_plan_binding(config, mode="fast")
        if scientifically_locked
        else None
    )
    artifacts = []
    for expected in config.data["conditions"]:
        key = (expected["dataset"], expected["condition"])
        if key not in by_key:
            raise ValueError(f"missing explicit frozen artifact for condition {key}")
        manifest_path, manifest_sha, manifest = by_key[key]
        for field in ("dataset", "condition", "model"):
            if manifest[field] != expected[field]:
                raise ValueError(f"frozen artifact {key} has unexpected {field}")
        if manifest["num_samples"] != expected["expected_num_samples"]:
            raise ValueError(
                f"frozen artifact {key} has {manifest['num_samples']} samples; "
                f"the campaign predeclares {expected['expected_num_samples']}"
            )
        checkpoint = manifest["checkpoint"]
        if (checkpoint is None) != (expected["checkpoint"] is None):
            raise ValueError(f"frozen artifact {key} has unexpected checkpoint state")
        if checkpoint is not None:
            expected_checkpoint = _project_path(config.path, expected["checkpoint"])
            observed_checkpoint = _project_path(config.path, checkpoint["path"])
            if expected_checkpoint != observed_checkpoint:
                raise ValueError(f"frozen artifact {key} used a different checkpoint")
            if not observed_checkpoint.is_file():
                raise FileNotFoundError(
                    f"locked checkpoint does not exist: {observed_checkpoint}"
                )
            if _sha256(observed_checkpoint) != checkpoint["sha256"].lower():
                raise ValueError(
                    f"checkpoint SHA-256 mismatch for frozen artifact {key}"
                )
        checkpoint_sha = None if checkpoint is None else checkpoint["sha256"].lower()
        artifact_entry = {
            "manifest_path": _portable_path(manifest_path),
            "manifest_sha256": manifest_sha,
            "artifact_id": manifest["artifact_id"],
            "dataset": manifest["dataset"],
            "condition": manifest["condition"],
            "model": manifest["model"],
            "split": manifest["split"],
            "checkpoint_sha256": checkpoint_sha,
            "source_sha256": manifest["source_sha256"].lower(),
            "sample_id_sha256": manifest["sample_id_sha256"].lower(),
            "num_samples": manifest["num_samples"],
        }
        if scientifically_locked:
            if manifest.get("schema_version") != 3:
                raise ValueError(
                    f"frozen artifact {key} lacks schema-3 scientific provenance"
                )
            from selectseg.scientific_inputs import condition_input_identity

            _, _, science_binding = scientific_plan
            identity = condition_input_identity(
                science_binding,
                dataset=expected["dataset"],
                model=expected["model"],
                condition=expected["condition"],
            )
            expected_scientific = {
                **identity["scientific_input_hashes"],
                "condition_input_sha256": identity[
                    "scientific_input_sha256"
                ],
            }
            if manifest.get("scientific_input") != expected_scientific:
                raise ValueError(
                    f"frozen artifact {key} scientific inputs differ from the root lock"
                )
            artifact_entry["scientific_input"] = expected_scientific
        elif manifest.get("schema_version") != 2:
            raise ValueError("legacy campaign locks require schema-2 frozen artifacts")
        artifacts.append(artifact_entry)
    if set(by_key) != {
        (condition["dataset"], condition["condition"])
        for condition in config.data["conditions"]
    }:
        raise ValueError("artifact inputs contain an undeclared condition")
    estimator_path, estimator_sha, estimator = _load_estimator(config)
    result = {
        "lock_schema_version": (
            SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION
            if scientifically_locked
            else LOCK_SCHEMA_VERSION
        ),
        "campaign_id": config.data["campaign_id"],
        "config": {
            "path": _portable_path(config.path),
            "sha256": config.sha256,
        },
        "protocol": config.data["protocol"],
        "estimator": {
            "spec_path": _portable_path(estimator_path),
            "spec_sha256": estimator_sha,
            "estimator_id": estimator["estimator_id"],
            "target_measure": estimator["target_measure"],
        },
        "paths": config.data["paths"],
        "artifacts": artifacts,
    }
    if scientifically_locked:
        science_path, science_sha256, science_binding = scientific_plan
        result["scientific_input"] = {
            "root_lock_path": _portable_path(science_path),
            "root_lock_sha256": science_sha256,
            "science_projection_sha256": science_binding["lock"][
                "science_config"
            ]["projection_sha256"],
        }
    return result


def write_campaign_lock(lock, output_path):
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"campaign lock already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(lock, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link is an atomic no-replace publication: two
        # concurrent writers cannot both install the same lock pathname.
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"campaign lock already exists: {output_path}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(
            output_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output_path, _sha256_bytes(payload)


def load_campaign_lock(path, *, config=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"campaign lock does not exist: {path}")
    raw = path.read_bytes()
    lock = _loads_strict(raw.decode("utf-8"), source=str(path))
    if not isinstance(lock, dict):
        raise ValueError(f"campaign lock must contain one object: {path}")
    lock_schema_version = lock.get("lock_schema_version")
    expected_lock_fields = (
        REQUIRED_SCIENTIFIC_LOCK_FIELDS
        if lock_schema_version == SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION
        else REQUIRED_LOCK_FIELDS
    )
    if set(lock) != expected_lock_fields:
        raise ValueError(
            f"campaign lock must contain exactly {sorted(expected_lock_fields)}"
        )
    _assert_finite(lock, location=str(path))
    if lock_schema_version not in {
        LOCK_SCHEMA_VERSION,
        SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"{path}.lock_schema_version must equal {LOCK_SCHEMA_VERSION} or "
            f"{SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION}"
        )
    _nonempty_string(lock.get("campaign_id"), location=f"{path}.campaign_id")
    _validate_protocol(lock.get("protocol"), location=f"{path}.protocol")
    provenance = lock.get("config")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != REQUIRED_LOCK_CONFIG_FIELDS
    ):
        raise ValueError(
            f"{path}.config must contain exactly {sorted(REQUIRED_LOCK_CONFIG_FIELDS)}"
        )
    _nonempty_string(provenance["path"], location=f"{path}.config.path")
    _digest(provenance["sha256"], location=f"{path}.config.sha256")
    paths = lock.get("paths")
    if not isinstance(paths, dict) or set(paths) != REQUIRED_PATH_FIELDS:
        raise ValueError(
            f"{path}.paths must contain exactly {sorted(REQUIRED_PATH_FIELDS)}"
        )
    for field in REQUIRED_PATH_FIELDS:
        _nonempty_string(paths[field], location=f"{path}.paths.{field}")
    if config is not None:
        if lock.get("campaign_id") != config.data["campaign_id"]:
            raise ValueError("campaign lock and config have different campaign_id")
        if provenance["sha256"] != config.sha256:
            raise ValueError("campaign lock was created from different config bytes")
        if paths != config.data["paths"]:
            raise ValueError("campaign lock and config have different output paths")
    scientific_binding = None
    if lock_schema_version == SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION:
        scientific = lock["scientific_input"]
        if (
            not isinstance(scientific, dict)
            or set(scientific) != REQUIRED_LOCK_SCIENTIFIC_INPUT_FIELDS
        ):
            raise ValueError(
                f"{path}.scientific_input must contain exactly "
                f"{sorted(REQUIRED_LOCK_SCIENTIFIC_INPUT_FIELDS)}"
            )
        scientific_path = _project_path(path, scientific["root_lock_path"])
        scientific_sha256 = _digest(
            scientific["root_lock_sha256"],
            location=f"{path}.scientific_input.root_lock_sha256",
        )
        science_projection_sha256 = _digest(
            scientific["science_projection_sha256"],
            location=f"{path}.scientific_input.science_projection_sha256",
        )
        from selectseg.scientific_inputs import load_root_lock

        scientific_binding = load_root_lock(
            scientific_path, expected_sha256=scientific_sha256
        )
        if (
            scientific_binding["lock"]["science_config"]["projection_sha256"]
            != science_projection_sha256
        ):
            raise ValueError("campaign lock scientific projection changed")
        if scientific_binding["lock"]["campaign_id"] != lock["campaign_id"]:
            raise ValueError("campaign and scientific-input locks have different IDs")
        if config is not None:
            if config.data.get("execution_policy") != SCIENTIFIC_INPUT_LOCKED:
                raise ValueError(
                    "scientific campaign lock requires a scientific-input-locked config"
                )
            configured = config.data["scientific_input_lock"]
            if (
                scientific_path != _project_path(config.path, configured["path"])
                or scientific_sha256 != configured["sha256"]
            ):
                raise ValueError("campaign lock names a different scientific input lock")
    elif config is not None and config.data.get("execution_policy") == (
        SCIENTIFIC_INPUT_LOCKED
    ):
        raise ValueError("scientific-input-locked config requires campaign lock schema 2")
    estimator = lock.get("estimator")
    if (
        not isinstance(estimator, dict)
        or set(estimator) != REQUIRED_LOCK_ESTIMATOR_FIELDS
    ):
        raise ValueError(
            f"{path}.estimator must contain exactly "
            f"{sorted(REQUIRED_LOCK_ESTIMATOR_FIELDS)}"
        )
    estimator_path = _project_path(path, estimator.get("spec_path"))
    estimator_sha = _digest(
        estimator.get("spec_sha256"), location=f"{path}.estimator.spec_sha256"
    )
    if not estimator_path.is_file() or _sha256(estimator_path) != estimator_sha:
        raise ValueError("locked estimator spec is missing or its bytes changed")
    if config is not None:
        expected_path, expected_sha, expected_estimator = _load_estimator(config)
        if estimator_path != expected_path or estimator_sha != expected_sha:
            raise ValueError("campaign lock names a different estimator spec")
        if estimator["estimator_id"] != expected_estimator["estimator_id"]:
            raise ValueError("campaign lock has a different estimator_id")
        if estimator["target_measure"] != expected_estimator["target_measure"]:
            raise ValueError("campaign lock has a different target_measure")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{path}.artifacts must be a non-empty list")
    seen = set()
    for index, artifact in enumerate(artifacts):
        expected_artifact_fields = (
            REQUIRED_SCIENTIFIC_LOCK_ARTIFACT_FIELDS
            if lock_schema_version == SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION
            else REQUIRED_LOCK_ARTIFACT_FIELDS
        )
        if (
            not isinstance(artifact, dict)
            or set(artifact) != expected_artifact_fields
        ):
            raise ValueError(
                f"{path}.artifacts[{index}] must contain exactly "
                f"{sorted(expected_artifact_fields)}"
            )
        manifest_path = _project_path(path, artifact.get("manifest_path"))
        expected_sha = _digest(
            artifact.get("manifest_sha256"),
            location=f"{path}.artifacts[{index}].manifest_sha256",
        )
        observed_path, observed_sha, manifest = _load_frozen_artifact(manifest_path)
        if observed_path != manifest_path or observed_sha != expected_sha:
            raise ValueError(f"locked frozen artifact changed: {manifest_path}")
        key = (artifact.get("dataset"), artifact.get("condition"))
        if key in seen:
            raise ValueError(f"campaign lock contains duplicate condition {key}")
        seen.add(key)
        exact = {
            "artifact_id": manifest["artifact_id"],
            "dataset": manifest["dataset"],
            "condition": manifest["condition"],
            "model": manifest["model"],
            "split": manifest["split"],
            "checkpoint_sha256": (
                None
                if manifest["checkpoint"] is None
                else manifest["checkpoint"]["sha256"]
            ),
            "source_sha256": manifest["source_sha256"],
            "sample_id_sha256": manifest["sample_id_sha256"],
            "num_samples": manifest["num_samples"],
        }
        if lock_schema_version == SCIENTIFIC_CAMPAIGN_LOCK_SCHEMA_VERSION:
            if manifest.get("schema_version") != 3:
                raise ValueError(
                    f"scientific campaign artifact {key} is not schema 3"
                )
            exact["scientific_input"] = manifest["scientific_input"]
            from selectseg.scientific_inputs import condition_input_identity

            identity = condition_input_identity(
                scientific_binding,
                dataset=manifest["dataset"],
                model=manifest["model"],
                condition=manifest["condition"],
            )
            expected_scientific = {
                **identity["scientific_input_hashes"],
                "condition_input_sha256": identity[
                    "scientific_input_sha256"
                ],
            }
            if manifest["scientific_input"] != expected_scientific:
                raise ValueError(
                    f"scientific provenance differs for frozen artifact {key}"
                )
        elif manifest.get("schema_version") != 2:
            raise ValueError(f"legacy campaign artifact {key} is not schema 2")
        for field, value in exact.items():
            if _canonical_json(artifact.get(field)) != _canonical_json(value):
                raise ValueError(
                    f"locked field {field!r} differs for frozen artifact {key}"
                )
    if config is not None:
        expected_conditions = {
            (condition["dataset"], condition["condition"]): condition
            for condition in config.data["conditions"]
        }
        if seen != set(expected_conditions):
            raise ValueError(
                "campaign lock artifact conditions differ from the campaign config"
            )
        if len(artifacts) != len(config.data["conditions"]):
            raise ValueError("campaign lock has the wrong artifact count")
        for artifact in artifacts:
            expected = expected_conditions[(artifact["dataset"], artifact["condition"])]
            if artifact["model"] != expected["model"]:
                raise ValueError("campaign lock artifact model differs from config")
            if artifact["num_samples"] != expected["expected_num_samples"]:
                raise ValueError(
                    "campaign lock artifact cohort size differs from config"
                )
            if (artifact["checkpoint_sha256"] is None) != (
                expected["checkpoint"] is None
            ):
                raise ValueError(
                    "campaign lock artifact checkpoint state differs from config"
                )
            if expected["checkpoint"] is not None:
                expected_checkpoint = _project_path(config.path, expected["checkpoint"])
                if not expected_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"configured checkpoint is missing: {expected_checkpoint}"
                    )
                if _sha256(expected_checkpoint) != artifact["checkpoint_sha256"]:
                    raise ValueError(
                        "campaign lock artifact checkpoint differs from config"
                    )
    return path, _sha256_bytes(raw), lock


def plan_score_jobs(config, campaign_lock):
    lock_path, lock_sha, lock = load_campaign_lock(campaign_lock, config=config)
    estimator = lock["estimator"]
    estimator_path = _project_path(lock_path, estimator["spec_path"])
    output_root = lock.get("paths", {}).get("simulation_output_root")
    _nonempty_string(output_root, location=f"{lock_path}.paths.simulation_output_root")
    jobs = []
    for artifact in lock["artifacts"]:
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        for gamma in lock["protocol"]["gamma_values"]:
            for count in lock["protocol"]["m_values"]:
                for seed in lock["protocol"]["seeds"]:
                    partition = _cpu_partition_request(config, len(jobs))
                    key = (
                        artifact["dataset"],
                        artifact["condition"],
                        partition,
                        gamma,
                        count,
                        seed,
                    )
                    gamma_tag = str(gamma).replace(".", "p")
                    job_name = (
                        f"selseg-sim-{artifact['dataset']}-{artifact['condition']}-"
                        f"g{gamma_tag}-m{count}-s{seed}"
                    )
                    command = (
                        "sbatch",
                        "--parsable",
                        "--job-name",
                        _scheduler_job_name(config, job_name),
                        "--partition",
                        partition,
                        "--account",
                        GPU_ACCOUNT,
                        "scripts/slurm/score_binary_simulation.sbatch",
                        "--campaign-id",
                        lock["campaign_id"],
                        "--campaign-lock",
                        str(lock_path),
                        "--expected-campaign-lock-sha256",
                        lock_sha,
                        "--artifact-manifest",
                        str(artifact_path),
                        "--expected-artifact-manifest-sha256",
                        artifact["manifest_sha256"],
                        "--estimator-spec",
                        str(estimator_path),
                        "--expected-estimator-spec-sha256",
                        estimator["spec_sha256"],
                        "--gamma",
                        str(gamma),
                        "--m",
                        str(count),
                        "--seed",
                        str(seed),
                        "--output-root",
                        output_root,
                    )
                    jobs.append(PlannedJob(phase="score", key=key, command=command))
    expected_count = (
        len(config.data["conditions"])
        * len(lock["protocol"]["gamma_values"])
        * len(lock["protocol"]["m_values"])
        * len(lock["protocol"]["seeds"])
    )
    if len(jobs) != expected_count or len({job.key for job in jobs}) != expected_count:
        raise RuntimeError("Cartesian simulation expansion was not one-to-one")
    return tuple(jobs)


def plan_common_jobs(config, campaign_lock):
    """Plan exactly one M-independent CPU job per locked artifact."""

    lock_path, lock_sha, lock = load_campaign_lock(campaign_lock, config=config)
    gamma_values = lock["protocol"]["gamma_values"]
    if len(gamma_values) != 1:
        raise ValueError("the common phase requires exactly one locked gamma")
    gamma = gamma_values[0]
    output_root = lock.get("paths", {}).get("common_output_root")
    _nonempty_string(output_root, location=f"{lock_path}.paths.common_output_root")
    jobs = []
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = _cpu_partition_request(config, artifact_index)
        key = (artifact["dataset"], artifact["condition"], partition, gamma)
        gamma_tag = str(gamma).replace(".", "p")
        job_name = (
            f"selseg-common-{artifact['dataset']}-{artifact['condition']}-g{gamma_tag}"
        )
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            _scheduler_job_name(config, job_name),
            "--partition",
            partition,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/score_binary_common.sbatch",
            "--campaign-id",
            lock["campaign_id"],
            "--campaign-lock",
            str(lock_path),
            "--expected-campaign-lock-sha256",
            lock_sha,
            "--artifact-manifest",
            str(artifact_path),
            "--expected-artifact-manifest-sha256",
            artifact["manifest_sha256"],
            "--gamma",
            str(gamma),
            "--output-root",
            output_root,
        )
        jobs.append(PlannedJob(phase="common", key=key, command=command))
    expected_count = len(config.data["conditions"])
    if len(jobs) != expected_count or len({job.key for job in jobs}) != expected_count:
        raise RuntimeError("common-job expansion was not one-to-one with artifacts")
    return tuple(jobs)


def _expected_score_manifests(config, lock_path, lock_sha, lock, artifact):
    """Derive the only common/M manifests compatible with the locked run.

    The scorer output directories are content addressed.  Recomputing their
    IDs from the exact identity functions avoids both operator-supplied shard
    paths and directory discovery.  A source change intentionally points at a
    new ID, so stale shards cannot be assembled into a campaign planned from
    different code bytes.
    """

    from selectseg.score_binary_common import (
        _common_id,
        _source_fingerprint as common_source_fingerprint,
    )
    from selectseg.score_binary_simulation import (
        _simulation_id,
        _source_fingerprint as simulation_source_fingerprint,
    )

    gamma_values = lock["protocol"]["gamma_values"]
    seed_values = lock["protocol"]["seeds"]
    if len(gamma_values) != 1 or len(seed_values) != 1:
        raise ValueError("assembly requires exactly one locked gamma and seed")
    gamma = gamma_values[0]
    seed = seed_values[0]
    common_id = _common_id(
        campaign_id=lock["campaign_id"],
        campaign_lock_sha256=lock_sha,
        artifact_manifest_sha256=artifact["manifest_sha256"],
        source_sha256=common_source_fingerprint(),
        gamma=gamma,
    )
    common_root = _project_path(config.path, lock["paths"]["common_output_root"])
    common_manifest = (
        common_root
        / artifact["dataset"]
        / artifact["condition"]
        / common_id
        / "manifest.json"
    )

    simulation_root = _project_path(
        config.path, lock["paths"]["simulation_output_root"]
    )
    simulation_source_sha256 = simulation_source_fingerprint()
    simulation_manifests = []
    for count in lock["protocol"]["m_values"]:
        simulation_id = _simulation_id(
            campaign_id=lock["campaign_id"],
            campaign_lock_sha256=lock_sha,
            artifact_manifest_sha256=artifact["manifest_sha256"],
            estimator_spec_sha256=lock["estimator"]["spec_sha256"],
            source_sha256=simulation_source_sha256,
            gamma=gamma,
            m=count,
            seed=seed,
        )
        simulation_manifests.append(
            simulation_root
            / artifact["dataset"]
            / artifact["condition"]
            / simulation_id
            / "manifest.json"
        )
    return common_manifest, tuple(simulation_manifests)


def plan_assemble_jobs(config, campaign_lock, *, allow_existing_output=False):
    """Plan one strict assembly job per locked artifact, without discovery."""

    lock_path, lock_sha, lock = load_campaign_lock(campaign_lock, config=config)
    from scripts.assemble_binary_simulations import (
        load_campaign_lock as load_assembly_lock,
        prepare_assembly,
    )

    assembly_lock = load_assembly_lock(lock_path)
    output_root = lock["paths"]["assembly_output_root"]
    _nonempty_string(output_root, location=f"{lock_path}.paths.assembly_output_root")
    jobs = []
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        common_manifest, simulation_manifests = _expected_score_manifests(
            config, lock_path, lock_sha, lock, artifact
        )
        required = (common_manifest, *simulation_manifests)
        missing = [path for path in required if not path.is_file()]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "locked scoring outputs are incomplete for "
                f"{(artifact['dataset'], artifact['condition'])}: {formatted}"
            )
        # Validate every manifest, records hash, row schema, ordered sample ID,
        # and lock-bound provenance before a submission intent is recorded.
        dataset, condition, run_id, _, _ = prepare_assembly(
            assembly_lock, common_manifest, simulation_manifests
        )
        if (dataset, condition) != (artifact["dataset"], artifact["condition"]):
            raise RuntimeError("derived assembly inputs target a different condition")
        resolved_output_root = _project_path(config.path, output_root)
        target = resolved_output_root / dataset / condition / run_id
        if not allow_existing_output and (target.exists() or target.is_symlink()):
            raise FileExistsError(f"assembled output already exists: {target}")

        partition = _cpu_partition_request(config, artifact_index)
        key = (artifact["dataset"], artifact["condition"], partition)
        job_name = f"selseg-assemble-{artifact['dataset']}-{artifact['condition']}"
        command = [
            "sbatch",
            "--parsable",
            "--job-name",
            _scheduler_job_name(config, job_name),
            "--partition",
            partition,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/assemble_binary_simulations.sbatch",
            "--campaign-lock",
            str(lock_path),
            "--common",
            str(common_manifest),
        ]
        for manifest in simulation_manifests:
            command.extend(("--input", str(manifest)))
        command.extend(("--output-root", output_root))
        jobs.append(PlannedJob(phase="assemble", key=key, command=tuple(command)))
    expected_count = len(config.data["conditions"])
    if len(jobs) != expected_count or len({job.key for job in jobs}) != expected_count:
        raise RuntimeError("assembly-job expansion was not one-to-one with artifacts")
    return tuple(jobs)


def plan_diagnose_jobs(
    config,
    campaign_lock,
    *,
    output_root="outputs/binary_diagnostics",
):
    """Plan one read-only diagnostic job per lock-listed frozen artifact."""

    lock_path, _, lock = load_campaign_lock(campaign_lock, config=config)
    _nonempty_string(output_root, location="diagnostic output root")
    gamma_values = lock["protocol"]["gamma_values"]
    if len(gamma_values) != 1:
        raise ValueError("diagnostics require exactly one locked decision threshold")
    gamma = gamma_values[0]
    jobs = []
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = _cpu_partition_request(config, artifact_index)
        key = (artifact["dataset"], artifact["condition"], partition)
        job_name = f"selseg-diagnose-{artifact['dataset']}-{artifact['condition']}"
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            _scheduler_job_name(config, job_name),
            "--partition",
            partition,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/diagnose_binary_artifact.sbatch",
            "--artifact-manifest",
            str(artifact_path),
            "--expected-artifact-manifest-sha256",
            artifact["manifest_sha256"],
            "--output-root",
            output_root,
            "--decision-threshold",
            str(gamma),
            "--write-descriptors",
        )
        jobs.append(PlannedJob(phase="diagnose", key=key, command=command))
    expected_count = len(config.data["conditions"])
    if len(jobs) != expected_count or len({job.key for job in jobs}) != expected_count:
        raise RuntimeError("diagnostic-job expansion was not one-to-one with artifacts")
    return tuple(jobs)


def _append_receipt_event(handle, job, *, status, job_id):
    event = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": job.phase,
        "key": list(job.key),
        "command": list(job.command),
        "status": status,
        "job_id": job_id,
    }
    handle.seek(0, os.SEEK_END)
    handle.write(_canonical_json(event) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_receipt_events(handle, jobs, *, source):
    by_identity = {(job.phase, job.key): job for job in jobs}
    latest = {}
    handle.seek(0)
    for line_number, line in enumerate(handle, start=1):
        location = f"{source}:{line_number}"
        if not line.strip():
            raise ValueError(f"blank submission-receipt row at {location}")
        event = _loads_strict(line, source=location)
        if not isinstance(event, dict) or set(event) != REQUIRED_RECEIPT_FIELDS:
            raise ValueError(f"invalid submission-receipt schema at {location}")
        _assert_finite(event, location=location)
        if event["receipt_schema_version"] != RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported receipt schema at {location}")
        _nonempty_string(event["created_utc"], location=f"{location}.created_utc")
        phase = _nonempty_string(event["phase"], location=f"{location}.phase")
        if not isinstance(event["key"], list):
            raise ValueError(f"{location}.key must be a list")
        key = tuple(event["key"])
        identity = (phase, key)
        if identity not in by_identity:
            raise ValueError(
                f"submission receipt contains a job outside this plan at {location}"
            )
        command = event["command"]
        if (
            not isinstance(command, list)
            or tuple(command) != by_identity[identity].command
        ):
            raise ValueError(f"submission command changed at {location}")
        if event["status"] not in {"submitting", "submitted", "failed"}:
            raise ValueError(f"invalid submission status at {location}")
        if event["status"] == "submitted":
            _nonempty_string(event["job_id"], location=f"{location}.job_id")
        elif event["job_id"] is not None:
            raise ValueError(f"{location}.job_id must be null for this status")
        previous = latest.get(identity)
        previous_status = None if previous is None else previous["status"]
        allowed = {
            None: {"submitting"},
            "submitting": {"submitted", "failed"},
            "failed": {"submitting"},
            "submitted": set(),
        }[previous_status]
        if event["status"] not in allowed:
            raise ValueError(f"invalid receipt state transition at {location}")
        latest[identity] = event
    return latest


def _test_only_command(job):
    """Return the scheduler-validation command for one unchanged job identity."""

    command = list(job.command)
    if not command or command[0] != "sbatch":
        raise ValueError(f"planned job {job.key} is not an sbatch command")
    if command.count("--parsable") != 1 or "--test-only" in command:
        raise ValueError(
            f"planned job {job.key} must contain exactly one --parsable option"
        )
    command[command.index("--parsable")] = "--test-only"
    return tuple(command)


def _compact_scheduler_output(value):
    """Bound scheduler diagnostics while retaining line identities and counts."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    else:
        value = str(value)
    counts = {}
    for line in value.splitlines():
        line = line.rstrip()
        if not line:
            continue
        counts[line] = counts.get(line, 0) + 1
    if not counts and value.strip():
        counts[value.strip()] = 1

    rendered = []
    entries = list(counts.items())
    shown = entries[:SCHEDULER_PREFLIGHT_MAX_DISTINCT_LINES]
    for line, count in shown:
        if len(line) > SCHEDULER_PREFLIGHT_MAX_LINE_CHARACTERS:
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]
            line = (
                line[:SCHEDULER_PREFLIGHT_MAX_LINE_CHARACTERS]
                + f"... [truncated sha256={digest}]"
            )
        rendered.append(f"{line} [occurrences={count}]")
    omitted = entries[SCHEDULER_PREFLIGHT_MAX_DISTINCT_LINES:]
    if omitted:
        rendered.append(
            "... omitted "
            f"{len(omitted)} distinct lines ({sum(count for _, count in omitted)} "
            "occurrences)"
        )
    return "\n".join(rendered)


def _report_scheduler_preflight(job, *, status, stdout=None, stderr=None, reason=None):
    """Expose Slurm's selected partition/start forecast for operator review."""

    message = f"{job.key}: scheduler_preflight={status}"
    if reason:
        message += f" reason={reason}"
    print(message)
    stdout = _compact_scheduler_output(stdout)
    stderr = _compact_scheduler_output(stderr)
    if stdout:
        print(f"{job.key}: scheduler stdout: {stdout}")
    if stderr:
        print(f"{job.key}: scheduler stderr: {stderr}", file=sys.stderr)


def preflight_plan(jobs, *, runner=subprocess.run):
    """Validate a complete schema-v2 wave without submitting or writing receipts."""

    jobs = tuple(jobs)
    for job in jobs:
        command = _test_only_command(job)
        try:
            result = runner(
                list(command),
                check=True,
                capture_output=True,
                text=True,
                timeout=SCHEDULER_PREFLIGHT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            _report_scheduler_preflight(
                job,
                status="failed",
                stdout=getattr(error, "stdout", None),
                stderr=getattr(error, "stderr", None),
                reason=f"timeout_after_{error.timeout}s",
            )
            raise
        except subprocess.CalledProcessError as error:
            _report_scheduler_preflight(
                job,
                status="failed",
                stdout=getattr(error, "stdout", None),
                stderr=getattr(error, "stderr", None),
            )
            raise
        except OSError as error:
            _report_scheduler_preflight(
                job,
                status="failed",
                stderr=f"{type(error).__name__}: {error}",
            )
            raise
        returncode = getattr(result, "returncode", 0)
        if returncode != 0:
            _report_scheduler_preflight(
                job,
                status="failed",
                stdout=getattr(result, "stdout", None),
                stderr=getattr(result, "stderr", None),
            )
            raise subprocess.CalledProcessError(
                returncode,
                list(command),
                output=getattr(result, "stdout", None),
                stderr=getattr(result, "stderr", None),
            )
        _report_scheduler_preflight(
            job,
            status="ok",
            stdout=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
        )
    return jobs


def _has_slurm_array_option(command):
    """Inspect only the sbatch option prefix, never wrapper arguments."""

    wrapper_index = next(
        (
            index
            for index, token in enumerate(command[1:], start=1)
            if isinstance(token, str) and token.endswith(".sbatch")
        ),
        len(command),
    )
    for token in command[1:wrapper_index]:
        if not isinstance(token, str):
            continue
        if token == "--array" or token.startswith("--array="):
            return True
        if token == "-a" or token.startswith("-a=") or re.fullmatch(r"-a\d.*", token):
            return True
    return False


def _validate_runtime_jobs(jobs):
    jobs = tuple(jobs)
    if not jobs:
        raise ValueError("schema-v2 runtime execution requires a non-empty plan")
    identities = set()
    phases = set()
    for job in jobs:
        identity = (job.phase, job.key)
        if identity in identities:
            raise ValueError(f"duplicate schema-v2 job identity {identity}")
        identities.add(identity)
        phases.add(job.phase)
        if not job.command or job.command[0] != "sbatch":
            raise ValueError(f"schema-v2 job {job.key} is not an sbatch command")
        if _has_slurm_array_option(job.command):
            raise ValueError("schema-v2 runtime execution forbids Slurm arrays")
    if len(phases) != 1:
        raise ValueError("schema-v2 runtime execution requires one phase per wave")
    return jobs


def _validate_slurm_job_id(value, *, location):
    value = _nonempty_string(value, location=location)
    if not SLURM_JOB_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{location} is not a non-array Slurm job id")
    return value


def _runtime_event(
    config,
    job,
    *,
    event,
    attempt,
    authorization,
    job_id=None,
    predecessor_job_id=None,
    scheduler_state=None,
    scheduler_exit_code=None,
    scheduler_job_name=None,
    scheduler_account=None,
    scheduler_partition=None,
    scheduler_nodes=None,
    scheduler_reason=None,
):
    return {
        "receipt_schema_version": RUNTIME_RECEIPT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign_id": config.data["campaign_id"],
        "config_sha256": config.sha256,
        "event": event,
        "phase": job.phase,
        "key": list(job.key),
        "command": list(job.command),
        "attempt": attempt,
        "authorization": authorization,
        "job_id": job_id,
        "predecessor_job_id": predecessor_job_id,
        "scheduler_state": scheduler_state,
        "scheduler_exit_code": scheduler_exit_code,
        "scheduler_job_name": scheduler_job_name,
        "scheduler_account": scheduler_account,
        "scheduler_partition": scheduler_partition,
        "scheduler_nodes": scheduler_nodes,
        "scheduler_reason": scheduler_reason,
    }


def _append_runtime_receipt_event(handle, event):
    """Append one locked schema-v2 event and make it durable before returning."""

    payload = (_canonical_json(event) + "\n").encode("utf-8")
    handle.flush()
    written = os.write(handle.fileno(), payload)
    if written != len(payload):
        raise OSError(
            f"short schema-v2 receipt append ({written} of {len(payload)} bytes)"
        )
    os.fsync(handle.fileno())


def _nullable_string(value, *, location):
    if value is not None:
        _nonempty_string(value, location=location)
    return value


def _normalized_slurm_state(value):
    if value is None:
        return None
    value = value.strip().upper()
    if not value:
        return None
    return value.split(None, 1)[0].rstrip("+")


def _load_runtime_receipt_events(
    handle, config, jobs, *, source, include_job_owners=False
):
    """Validate the complete append-only runtime ledger and return latest rows."""

    jobs = _validate_runtime_jobs(jobs)
    by_identity = {(job.phase, job.key): job for job in jobs}
    latest = {}
    job_owners = {}
    handle.seek(0)
    for line_number, raw_line in enumerate(handle, start=1):
        location = f"{source}:{line_number}"
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid UTF-8 receipt row at {location}") from error
        if not line.strip():
            raise ValueError(f"blank schema-v2 receipt row at {location}")
        event = _loads_strict(line, source=location)
        if not isinstance(event, dict) or set(event) != REQUIRED_RUNTIME_RECEIPT_FIELDS:
            raise ValueError(f"invalid schema-v2 receipt fields at {location}")
        _assert_finite(event, location=location)
        if event["receipt_schema_version"] != RUNTIME_RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema-v2 receipt version at {location}")
        _nonempty_string(event["created_utc"], location=f"{location}.created_utc")
        if event["campaign_id"] != config.data["campaign_id"]:
            raise ValueError(f"receipt campaign changed at {location}")
        if event["config_sha256"] != config.sha256:
            raise ValueError(f"receipt config binding changed at {location}")
        phase = _nonempty_string(event["phase"], location=f"{location}.phase")
        if not isinstance(event["key"], list):
            raise ValueError(f"{location}.key must be a list")
        identity = (phase, tuple(event["key"]))
        if identity not in by_identity:
            raise ValueError(f"receipt contains a job outside this plan at {location}")
        if (
            not isinstance(event["command"], list)
            or tuple(event["command"]) != by_identity[identity].command
        ):
            raise ValueError(f"submission identity changed at {location}")
        kind = event["event"]
        if kind not in RUNTIME_RECEIPT_EVENTS:
            raise ValueError(f"invalid runtime event at {location}")
        attempt = event["attempt"]
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise ValueError(f"{location}.attempt must be a positive integer")
        authorization = event["authorization"]
        if authorization not in RUNTIME_AUTHORIZATIONS:
            raise ValueError(f"invalid retry authorization at {location}")
        predecessor = _nullable_string(
            event["predecessor_job_id"],
            location=f"{location}.predecessor_job_id",
        )
        if predecessor is not None:
            _validate_slurm_job_id(
                predecessor, location=f"{location}.predecessor_job_id"
            )
        job_id = _nullable_string(event["job_id"], location=f"{location}.job_id")
        if job_id is not None:
            _validate_slurm_job_id(job_id, location=f"{location}.job_id")
            owner = (identity, attempt)
            if job_id in job_owners and job_owners[job_id] != owner:
                raise ValueError(f"Slurm job id is reused at {location}")
            job_owners[job_id] = owner
            if job_id == predecessor:
                raise ValueError(f"replacement reused its predecessor at {location}")

        scheduler_fields = (
            "scheduler_state",
            "scheduler_exit_code",
            "scheduler_job_name",
            "scheduler_account",
            "scheduler_partition",
            "scheduler_nodes",
            "scheduler_reason",
        )
        for field in scheduler_fields:
            _nullable_string(event[field], location=f"{location}.{field}")
        if kind == "submitting":
            if job_id is not None or any(
                event[field] is not None for field in scheduler_fields
            ):
                raise ValueError(
                    f"submitting event carries scheduler data at {location}"
                )
        elif kind == "submitted":
            if job_id is None or any(
                event[field] is not None for field in scheduler_fields
            ):
                raise ValueError(f"invalid submitted event at {location}")
        elif kind == "submission_recovered":
            required_recovery_fields = (
                "scheduler_state",
                "scheduler_job_name",
                "scheduler_account",
                "scheduler_partition",
            )
            if job_id is None or any(
                event[field] is None for field in required_recovery_fields
            ):
                raise ValueError(f"recovered event lacks scheduler proof at {location}")
        elif kind == "submission_failed":
            if (
                job_id is not None
                or any(event[field] is not None for field in scheduler_fields[:-1])
                or event["scheduler_reason"] is None
            ):
                raise ValueError(f"invalid submission-failure event at {location}")
        else:
            state = _normalized_slurm_state(event["scheduler_state"])
            if job_id is None or state is None or event["scheduler_exit_code"] is None:
                raise ValueError(f"terminal event lacks Slurm facts at {location}")
            if kind == "completed":
                if state != "COMPLETED" or event["scheduler_exit_code"] != "0:0":
                    raise ValueError(f"invalid completed event at {location}")
            elif state not in SLURM_FAILED_STATES and not (
                state == "COMPLETED" and event["scheduler_exit_code"] != "0:0"
            ):
                raise ValueError(f"invalid failed event at {location}")

        previous = latest.get(identity)
        if previous is None:
            if not (
                kind == "submitting"
                and attempt == 1
                and authorization == "initial"
                and predecessor is None
            ):
                raise ValueError(f"invalid initial runtime event at {location}")
        else:
            previous_kind = previous["event"]
            if previous_kind == "submitting":
                if kind not in {
                    "submitted",
                    "submission_recovered",
                    "submission_failed",
                }:
                    raise ValueError(f"invalid runtime transition at {location}")
                if (
                    attempt != previous["attempt"]
                    or authorization != previous["authorization"]
                    or predecessor != previous["predecessor_job_id"]
                ):
                    raise ValueError(f"attempt identity changed at {location}")
            elif previous_kind in {"submitted", "submission_recovered"}:
                if kind not in {"completed", "failed"}:
                    raise ValueError(f"invalid runtime transition at {location}")
                if (
                    attempt != previous["attempt"]
                    or authorization != previous["authorization"]
                    or predecessor != previous["predecessor_job_id"]
                    or job_id != previous["job_id"]
                ):
                    raise ValueError(f"submitted job identity changed at {location}")
            elif previous_kind == "failed":
                if not (
                    kind == "submitting"
                    and attempt == previous["attempt"] + 1
                    and authorization == "retry_failed_job_id"
                    and predecessor == previous["job_id"]
                ):
                    raise ValueError(
                        f"unauthorized failed-job replacement at {location}"
                    )
            elif previous_kind == "submission_failed":
                if not (
                    kind == "submitting"
                    and attempt == previous["attempt"] + 1
                    and authorization == "retry_submission_failure"
                    and predecessor == previous["predecessor_job_id"]
                ):
                    raise ValueError(f"unauthorized submission retry at {location}")
            else:
                raise ValueError(f"completed job has later events at {location}")
        latest[identity] = event
    if include_job_owners:
        return latest, job_owners
    return latest


def _open_runtime_receipt(path, *, create):
    flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"schema-v2 receipt is not a private regular file: {path}")
    return os.fdopen(descriptor, "a+b", buffering=0)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _submission_failure_reason(error):
    message = f"{type(error).__name__}: {error}".replace("\x00", "")
    return message[:4096] or type(error).__name__


def _execute_runtime_plan(
    config,
    jobs,
    *,
    receipt_path,
    runner,
    retry_failed_job_ids=(),
    retry_submission_failures=False,
):
    jobs = _validate_runtime_jobs(jobs)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    requested_retries = tuple(retry_failed_job_ids)
    if len(requested_retries) != len(set(requested_retries)):
        raise ValueError("--retry-failed-job-id values must be unique")
    for index, job_id in enumerate(requested_retries):
        _validate_slurm_job_id(job_id, location=f"retry_failed_job_ids[{index}]")
    requested_retries = set(requested_retries)
    created = not receipt_path.exists()
    job_ids = []
    skipped = 0
    with _open_runtime_receipt(receipt_path, create=True) as receipt:
        fcntl.flock(receipt.fileno(), fcntl.LOCK_EX)
        if created:
            _fsync_directory(receipt_path.parent)
        latest, job_owners = _load_runtime_receipt_events(
            receipt,
            config,
            jobs,
            source=receipt_path,
            include_job_owners=True,
        )
        dangling = [
            job.key
            for job in jobs
            if latest.get((job.phase, job.key), {}).get("event") == "submitting"
        ]
        if dangling:
            raise RuntimeError(
                "unresolved schema-v2 submission intent(s) have no durably "
                f"recorded Slurm job id; fail closed and inspect Slurm: {dangling}"
            )
        failed_by_id = {
            event["job_id"]: identity
            for identity, event in latest.items()
            if event["event"] == "failed"
        }
        unknown_retries = requested_retries - set(failed_by_id)
        if unknown_retries:
            raise ValueError(
                "replacement authorization does not name a current failed job: "
                f"{sorted(unknown_retries)}"
            )

        for job in jobs:
            identity = (job.phase, job.key)
            previous = latest.get(identity)
            previous_kind = None if previous is None else previous["event"]
            if previous_kind in {"submitted", "submission_recovered", "completed"}:
                skipped += 1
                print(f"{job.key}: already {previous_kind} as {previous['job_id']}")
                continue
            if (
                previous_kind == "failed"
                and previous["job_id"] not in requested_retries
            ):
                skipped += 1
                print(
                    f"{job.key}: failed as {previous['job_id']}; replacement "
                    "requires --retry-failed-job-id"
                )
                continue
            if previous_kind == "submission_failed" and not retry_submission_failures:
                skipped += 1
                print(
                    f"{job.key}: sbatch failed; retry requires "
                    "--retry-submission-failure"
                )
                continue

            if previous is None:
                attempt = 1
                authorization = "initial"
                predecessor = None
            elif previous_kind == "failed":
                attempt = previous["attempt"] + 1
                authorization = "retry_failed_job_id"
                predecessor = previous["job_id"]
            else:
                attempt = previous["attempt"] + 1
                authorization = "retry_submission_failure"
                predecessor = previous["predecessor_job_id"]
            intent = _runtime_event(
                config,
                job,
                event="submitting",
                attempt=attempt,
                authorization=authorization,
                predecessor_job_id=predecessor,
            )
            _append_runtime_receipt_event(receipt, intent)
            try:
                result = runner(
                    list(job.command),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                failed = _runtime_event(
                    config,
                    job,
                    event="submission_failed",
                    attempt=attempt,
                    authorization=authorization,
                    predecessor_job_id=predecessor,
                    scheduler_reason=_submission_failure_reason(error),
                )
                _append_runtime_receipt_event(receipt, failed)
                raise
            returncode = getattr(result, "returncode", 0)
            if returncode != 0:
                error = subprocess.CalledProcessError(
                    returncode,
                    list(job.command),
                    output=getattr(result, "stdout", None),
                    stderr=getattr(result, "stderr", None),
                )
                failed = _runtime_event(
                    config,
                    job,
                    event="submission_failed",
                    attempt=attempt,
                    authorization=authorization,
                    predecessor_job_id=predecessor,
                    scheduler_reason=_submission_failure_reason(error),
                )
                _append_runtime_receipt_event(receipt, failed)
                raise error

            # A zero exit status means Slurm may already own a real job.  Any
            # local parsing or duplicate-ID failure after this point must leave
            # the durable intent dangling for explicit scheduler recovery; it
            # must never become an automatically retryable submission failure.
            job_id = _validate_slurm_job_id(
                getattr(result, "stdout", "").strip(),
                location=f"sbatch output for {job.key}",
            )
            if job_id in job_owners:
                raise ValueError(
                    f"sbatch reused already recorded Slurm job id {job_id}"
                )
            submitted = _runtime_event(
                config,
                job,
                event="submitted",
                attempt=attempt,
                authorization=authorization,
                job_id=job_id,
                predecessor_job_id=predecessor,
            )
            _append_runtime_receipt_event(receipt, submitted)
            latest[identity] = submitted
            job_owners[job_id] = (identity, attempt)
            job_ids.append(job_id)
            print(f"{job.key}: {job_id}")
    print(
        f"planned_jobs={len(jobs)} submitted_jobs={len(job_ids)} skipped_jobs={skipped}"
    )
    return tuple(job_ids)


def _query_command(runner, command):
    try:
        result = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if getattr(result, "returncode", 0) != 0:
        return None
    return getattr(result, "stdout", "")


def _split_scheduler_row(line, expected_fields):
    fields = [field.strip() for field in line.rstrip("\n").split("|")]
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) != expected_fields:
        return None
    return fields


def _scheduler_job_record(job_id, *, runner=subprocess.run):
    """Return scheduler-owned identity fields for an exact non-array job id."""

    query_id = job_id.split(";", 1)[0]
    sacct_output = _query_command(
        runner,
        (
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--jobs",
            query_id,
            "--format=JobIDRaw,JobName%128,Account,Partition,State,ExitCode,NodeList,Reason",
        ),
    )
    if sacct_output is not None:
        for line in sacct_output.splitlines():
            fields = _split_scheduler_row(line, 8)
            if fields is None or fields[0] != query_id:
                continue
            _, name, account, partition, state, exit_code, nodes, reason = fields
            if name and account and partition and state:
                return SchedulerJobRecord(
                    job_id=job_id,
                    job_name=name,
                    account=account,
                    partition=partition,
                    state=state,
                    exit_code=exit_code or None,
                    nodes=nodes or None,
                    reason=reason or None,
                )
    squeue_output = _query_command(
        runner,
        (
            "squeue",
            "--noheader",
            "--jobs",
            query_id,
            "--format=%i|%.128j|%a|%P|%T|%N|%R",
        ),
    )
    if squeue_output is not None:
        for line in squeue_output.splitlines():
            fields = _split_scheduler_row(line, 7)
            if fields is None or fields[0] != query_id:
                continue
            _, name, account, partition, state, nodes, reason = fields
            if name and account and partition and state:
                return SchedulerJobRecord(
                    job_id=job_id,
                    job_name=name,
                    account=account,
                    partition=partition,
                    state=state,
                    nodes=nodes or None,
                    reason=reason or None,
                )
    return None


def _sbatch_option(job, option):
    values = []
    command = job.command
    for index, token in enumerate(command):
        if token == option:
            if index + 1 >= len(command):
                raise ValueError(f"planned job {job.key} has no value for {option}")
            values.append(command[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"planned job {job.key} must carry exactly one {option} value")
    return values[0]


def _verify_recovered_scheduler_identity(job, record):
    expected_name = _sbatch_option(job, "--job-name")
    expected_account = _sbatch_option(job, "--account")
    expected_partitions = set(_sbatch_option(job, "--partition").split(","))
    observed_partitions = set(record.partition.split(","))
    mismatches = []
    if record.job_name != expected_name:
        mismatches.append(f"job_name={record.job_name!r} expected {expected_name!r}")
    if record.account != expected_account:
        mismatches.append(f"account={record.account!r} expected {expected_account!r}")
    if not observed_partitions or not observed_partitions <= expected_partitions:
        mismatches.append(
            f"partition={record.partition!r} outside {sorted(expected_partitions)!r}"
        )
    if mismatches:
        raise ValueError(
            "scheduler job does not match the dangling planned identity: "
            + "; ".join(mismatches)
        )


def _scheduler_observation(job_id, *, runner=subprocess.run):
    query_id = job_id.split(";", 1)[0]
    sacct_output = _query_command(
        runner,
        (
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--jobs",
            query_id,
            "--format=JobIDRaw,State,ExitCode,Partition,NodeList,Reason",
        ),
    )
    if sacct_output is not None:
        for line in sacct_output.splitlines():
            fields = _split_scheduler_row(line, 6)
            if fields is None or fields[0] != query_id:
                continue
            _, raw_state, exit_code, partition, nodes, reason = fields
            state = _normalized_slurm_state(raw_state)
            observation = SchedulerObservation(
                disposition="unknown",
                state=raw_state or None,
                exit_code=exit_code or None,
                partition=partition or None,
                nodes=nodes or None,
                reason=reason or None,
            )
            if state in SLURM_ACTIVE_STATES:
                return SchedulerObservation(
                    **{**observation.__dict__, "disposition": "active"}
                )
            if state == "COMPLETED" and exit_code == "0:0":
                return SchedulerObservation(
                    **{**observation.__dict__, "disposition": "completed"}
                )
            if state in SLURM_FAILED_STATES or (
                state == "COMPLETED" and exit_code and exit_code != "0:0"
            ):
                return SchedulerObservation(
                    **{**observation.__dict__, "disposition": "failed"}
                )

    squeue_output = _query_command(
        runner,
        (
            "squeue",
            "--noheader",
            "--jobs",
            query_id,
            "--format=%i|%T|%P|%N|%R",
        ),
    )
    if squeue_output is not None:
        for line in squeue_output.splitlines():
            fields = _split_scheduler_row(line, 5)
            if fields is None or fields[0] != query_id:
                continue
            _, raw_state, partition, nodes, reason = fields
            if _normalized_slurm_state(raw_state) in SLURM_ACTIVE_STATES:
                return SchedulerObservation(
                    disposition="active",
                    state=raw_state or None,
                    partition=partition or None,
                    nodes=nodes or None,
                    reason=reason or None,
                )
    return SchedulerObservation(disposition="unknown")


def recover_configured_submission(
    config,
    jobs,
    *,
    receipt_path,
    job_id,
    runner=subprocess.run,
):
    """Auditably bind one dangling intent to an operator-supplied Slurm id."""

    if config.data["config_schema_version"] != CANDIDATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("submission recovery is available only for schema v2")
    jobs = _validate_runtime_jobs(jobs)
    receipt_path = _validated_runtime_receipt_path(config, jobs, receipt_path)
    job_id = _validate_slurm_job_id(job_id, location="recover_submitted_job_id")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"schema-v2 receipt does not exist: {receipt_path}")
    with _open_runtime_receipt(receipt_path, create=False) as receipt:
        fcntl.flock(receipt.fileno(), fcntl.LOCK_EX)
        latest, job_owners = _load_runtime_receipt_events(
            receipt,
            config,
            jobs,
            source=receipt_path,
            include_job_owners=True,
        )
        dangling = [
            (job, latest[(job.phase, job.key)])
            for job in jobs
            if latest.get((job.phase, job.key), {}).get("event") == "submitting"
        ]
        if not dangling:
            matching = [
                event
                for event in latest.values()
                if event.get("job_id") == job_id
                and event["event"]
                in {"submitted", "submission_recovered", "completed", "failed"}
            ]
            if matching:
                print(f"job {job_id} was already durably bound; recovery is idempotent")
                return (job_id,)
            raise RuntimeError("receipt has no dangling submission intent to recover")
        if len(dangling) != 1:
            raise RuntimeError(
                "receipt has multiple dangling intents; recovery refuses to guess identity"
            )
        job, intent = dangling[0]
        if job_id in job_owners:
            raise ValueError(
                f"recovery job id {job_id} is already bound to another attempt"
            )
        record = _scheduler_job_record(job_id, runner=runner)
        if record is None:
            raise RuntimeError(
                f"Slurm cannot prove that job {job_id} exists; intent remains dangling"
            )
        _verify_recovered_scheduler_identity(job, record)
        recovered = _runtime_event(
            config,
            job,
            event="submission_recovered",
            attempt=intent["attempt"],
            authorization=intent["authorization"],
            job_id=job_id,
            predecessor_job_id=intent["predecessor_job_id"],
            scheduler_state=record.state,
            scheduler_exit_code=record.exit_code,
            scheduler_job_name=record.job_name,
            scheduler_account=record.account,
            scheduler_partition=record.partition,
            scheduler_nodes=record.nodes,
            scheduler_reason=record.reason,
        )
        _append_runtime_receipt_event(receipt, recovered)
    print(f"{job.key}: recovered {job_id} after validating name/account/partition")
    return (job_id,)


def reconcile_configured_plan(
    config,
    jobs,
    *,
    receipt_path,
    runner=subprocess.run,
):
    """Append actual terminal Slurm facts for submitted schema-v2 jobs."""

    if config.data["config_schema_version"] != CANDIDATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("runtime reconciliation is available only for schema v2")
    jobs = _validate_runtime_jobs(jobs)
    receipt_path = _validated_runtime_receipt_path(config, jobs, receipt_path)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"schema-v2 receipt does not exist: {receipt_path}")
    transitions = []
    active = 0
    unknown = []
    with _open_runtime_receipt(receipt_path, create=False) as receipt:
        fcntl.flock(receipt.fileno(), fcntl.LOCK_EX)
        latest = _load_runtime_receipt_events(
            receipt, config, jobs, source=receipt_path
        )
        dangling = [
            job.key
            for job in jobs
            if latest.get((job.phase, job.key), {}).get("event") == "submitting"
        ]
        if dangling:
            raise RuntimeError(
                "unresolved schema-v2 submission intent has no recorded job id; "
                f"scheduler recovery is ambiguous and therefore fails closed: {dangling}"
            )
        for job in jobs:
            previous = latest.get((job.phase, job.key))
            if previous is None or previous["event"] not in {
                "submitted",
                "submission_recovered",
            }:
                continue
            observation = _scheduler_observation(previous["job_id"], runner=runner)
            if observation.disposition == "active":
                active += 1
                print(f"{job.key}: {previous['job_id']} remains {observation.state}")
                continue
            if observation.disposition == "unknown":
                unknown.append(previous["job_id"])
                print(f"{job.key}: {previous['job_id']} scheduler state is unknown")
                continue
            event = _runtime_event(
                config,
                job,
                event=observation.disposition,
                attempt=previous["attempt"],
                authorization=previous["authorization"],
                job_id=previous["job_id"],
                predecessor_job_id=previous["predecessor_job_id"],
                scheduler_state=observation.state,
                scheduler_exit_code=observation.exit_code,
                scheduler_partition=observation.partition,
                scheduler_nodes=observation.nodes,
                scheduler_reason=observation.reason,
            )
            _append_runtime_receipt_event(receipt, event)
            transitions.append((previous["job_id"], observation.disposition))
            print(
                f"{job.key}: {previous['job_id']} -> {observation.disposition} "
                f"({observation.state}, {observation.exit_code})"
            )
    print(
        f"reconciled_terminal={len(transitions)} active={active} unknown={len(unknown)}"
    )
    if unknown:
        raise RuntimeError(
            "Slurm state remained unknown; receipt stays submitted and no "
            f"replacement is permitted: {unknown}"
        )
    return tuple(transitions)


def execute_plan(
    jobs,
    *,
    submit=False,
    receipt_path=None,
    runner=subprocess.run,
):
    jobs = tuple(jobs)
    job_ids = []
    if not submit:
        for job in jobs:
            print(shlex.join(job.command))
        print(f"planned_jobs={len(jobs)} submitted_jobs=0 skipped_jobs=0")
        return ()
    if receipt_path is None:
        raise ValueError("--submit requires an append-only --receipt path")
    receipt_path = Path(receipt_path).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with receipt_path.open("a+", encoding="utf-8") as receipt:
        fcntl.flock(receipt.fileno(), fcntl.LOCK_EX)
        latest = _load_receipt_events(receipt, jobs, source=receipt_path)
        for job in jobs:
            identity = (job.phase, job.key)
            previous = latest.get(identity)
            if previous is not None and previous["status"] == "submitted":
                skipped += 1
                print(f"{job.key}: already submitted as {previous['job_id']}")
                continue
            if previous is not None and previous["status"] == "submitting":
                raise RuntimeError(
                    f"unresolved submission intent for {job.key}; inspect Slurm "
                    "before reconciling the receipt"
                )
            _append_receipt_event(receipt, job, status="submitting", job_id=None)
            try:
                result = runner(
                    list(job.command),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                _append_receipt_event(receipt, job, status="failed", job_id=None)
                raise
            job_id = result.stdout.strip()
            if not job_id:
                _append_receipt_event(receipt, job, status="failed", job_id=None)
                raise RuntimeError(f"sbatch returned no job id for {job.key}")
            _append_receipt_event(receipt, job, status="submitted", job_id=job_id)
            latest[identity] = {"status": "submitted", "job_id": job_id}
            job_ids.append(job_id)
            print(f"{job.key}: {job_id}")
    print(
        f"planned_jobs={len(jobs)} submitted_jobs={len(job_ids)} skipped_jobs={skipped}"
    )
    return tuple(job_ids)


def execute_configured_plan(
    config,
    jobs,
    *,
    submit=False,
    scheduler_preflight_only=False,
    receipt_path=None,
    runner=subprocess.run,
    preflight_runner=subprocess.run,
    retry_failed_job_ids=(),
    retry_submission_failures=False,
):
    """Execute v1, preview v2, or run a scientifically locked v2 wave."""

    if config.data["config_schema_version"] == CANDIDATE_CONFIG_SCHEMA_VERSION:
        policy = config.data["execution_policy"]
        if policy == SCHEDULER_PREVIEW_ONLY:
            if submit:
                raise RuntimeError(
                    "schema-v2 is scheduler-preview-only because scientific inputs "
                    "are not independently bound; use a dry-run or "
                    "--scheduler-preflight-only"
                )
            if retry_failed_job_ids or retry_submission_failures:
                raise ValueError(
                    "schema-v2 preview does not accept retry authorization"
                )
            if receipt_path is not None:
                raise ValueError("schema-v2 scheduler preview never accepts --receipt")
            if scheduler_preflight_only:
                jobs = preflight_plan(
                    _validate_runtime_jobs(jobs), runner=preflight_runner
                )
                print(
                    f"planned_jobs={len(jobs)} scheduler_preflight_jobs={len(jobs)} "
                    "submitted_jobs=0"
                )
                return ()
            return execute_plan(jobs, submit=False, runner=runner)

        _scientific_plan_binding(config, mode="fast")
        jobs = _validate_runtime_jobs(jobs)
        if scheduler_preflight_only:
            if receipt_path is not None:
                raise ValueError("scheduler preflight never accepts --receipt")
            if retry_failed_job_ids or retry_submission_failures:
                raise ValueError("scheduler preflight never accepts retry authorization")
            jobs = preflight_plan(jobs, runner=preflight_runner)
            print(
                f"planned_jobs={len(jobs)} scheduler_preflight_jobs={len(jobs)} "
                "submitted_jobs=0"
            )
            return ()
        if not submit:
            if receipt_path is not None:
                raise ValueError("dry-run planning does not accept --receipt")
            if retry_failed_job_ids or retry_submission_failures:
                raise ValueError("retry authorization requires --submit")
            return execute_plan(jobs, submit=False, runner=runner)
        receipt = _validated_runtime_receipt_path(config, jobs, receipt_path)
        jobs = preflight_plan(jobs, runner=preflight_runner)
        return _execute_runtime_plan(
            config,
            jobs,
            receipt_path=receipt,
            runner=runner,
            retry_failed_job_ids=retry_failed_job_ids,
            retry_submission_failures=retry_submission_failures,
        )
    if scheduler_preflight_only:
        raise ValueError(
            "--scheduler-preflight-only is reserved for schema-v2 preview configs"
        )
    if retry_failed_job_ids or retry_submission_failures:
        raise ValueError("schema-v2 retry authorization is not valid for schema v1")
    return execute_plan(
        jobs,
        submit=submit,
        receipt_path=receipt_path,
        runner=runner,
    )


def _validate_execution_request(config, args):
    """Enforce the v2 authorization boundary before planning or external I/O."""

    is_v2 = config.data["config_schema_version"] == CANDIDATE_CONFIG_SCHEMA_VERSION
    if not is_v2:
        if args.scheduler_preflight_only:
            raise ValueError(
                "--scheduler-preflight-only is reserved for schema-v2 preview configs"
            )
        return
    if config.data["execution_policy"] == SCHEDULER_PREVIEW_ONLY:
        if args.submit:
            raise RuntimeError(
                "schema-v2 is scheduler-preview-only because scientific inputs are "
                "not independently bound; use a dry-run or "
                "--scheduler-preflight-only"
            )
        if args.reconcile or args.recover_submitted_job_id:
            raise RuntimeError(
                "schema-v2 scheduler preview has no real submission to reconcile "
                "or recover"
            )
        if args.retry_failed_job_id or args.retry_submission_failure:
            raise RuntimeError("schema-v2 scheduler preview has no attempts to retry")
        if args.receipt:
            raise ValueError("schema-v2 scheduler preview never accepts --receipt")
        if args.phase == "lock" or args.write_lock:
            raise RuntimeError(
                "schema-v2 scheduler preview cannot create a scientific campaign lock"
            )
        return

    # This happens before job planning, receipt creation, scheduler access, or
    # output publication. Any scientific-input drift therefore fails closed.
    _scientific_plan_binding(config, mode="fast")


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    _validate_execution_request(config, args)
    if (args.reconcile or args.recover_submitted_job_id) and (
        config.data["config_schema_version"] != CANDIDATE_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(
            "runtime reconciliation/recovery is available only for schema-v2 campaigns"
        )
    if (args.retry_failed_job_id or args.retry_submission_failure) and not args.submit:
        raise ValueError("retry authorization flags require --submit")

    def dispatch(jobs):
        if args.recover_submitted_job_id:
            return recover_configured_submission(
                config,
                jobs,
                receipt_path=args.receipt,
                job_id=args.recover_submitted_job_id,
            )
        if args.reconcile:
            return reconcile_configured_plan(
                config,
                jobs,
                receipt_path=args.receipt,
            )
        return execute_configured_plan(
            config,
            jobs,
            submit=args.submit,
            scheduler_preflight_only=args.scheduler_preflight_only,
            receipt_path=args.receipt,
            retry_failed_job_ids=args.retry_failed_job_id,
            retry_submission_failures=args.retry_submission_failure,
        )

    if args.phase == "freeze":
        if args.artifact_manifest or args.campaign_lock or args.write_lock:
            raise ValueError("freeze phase does not accept lock/artifact inputs")
        return dispatch(plan_freeze_jobs(config))
    if args.phase == "lock":
        if args.submit:
            raise ValueError("lock phase never invokes sbatch; omit --submit")
        if args.reconcile:
            raise ValueError("lock phase has no scheduler state to reconcile")
        if args.recover_submitted_job_id:
            raise ValueError("lock phase has no scheduler submission to recover")
        if args.retry_failed_job_id or args.retry_submission_failure:
            raise ValueError("lock phase has no scheduler attempts to retry")
        if args.receipt:
            raise ValueError("lock phase does not submit jobs; omit --receipt")
        if args.campaign_lock:
            raise ValueError("lock phase creates a lock; omit --campaign-lock")
        lock = build_campaign_lock(config, args.artifact_manifest)
        if args.write_lock:
            path, digest = write_campaign_lock(lock, args.write_lock)
            print(f"saved {path}")
            print(f"campaign_lock_sha256={digest}")
        else:
            print(json.dumps(lock, indent=2, allow_nan=False))
        print("planned_jobs=0 submitted_jobs=0")
        return ()
    if args.artifact_manifest or args.write_lock:
        raise ValueError(
            f"{args.phase} phase reads artifacts only through --campaign-lock"
        )
    if not args.campaign_lock:
        raise ValueError(f"{args.phase} phase requires --campaign-lock")
    if args.phase == "common":
        return dispatch(plan_common_jobs(config, args.campaign_lock))
    if args.phase == "score":
        jobs = plan_score_jobs(config, args.campaign_lock)
    elif args.phase == "assemble":
        jobs = plan_assemble_jobs(
            config,
            args.campaign_lock,
            allow_existing_output=(
                args.reconcile or bool(args.recover_submitted_job_id)
            ),
        )
    else:
        jobs = plan_diagnose_jobs(
            config,
            args.campaign_lock,
            output_root=args.diagnostic_output_root,
        )
    return dispatch(jobs)


if __name__ == "__main__":
    main()
