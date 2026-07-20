"""Build a strict, path-free release for the training-seed extension.

This exporter is intentionally specific to the fixed five-dataset, two-model,
two-extension-seed protocol.  It does not copy the checkpoint or downstream
locks: those files contain operational paths and scheduler metadata.  Instead
it recursively validates the private closure and constructs two public files
from explicit whitelists:

``seed_robustness_analysis.json``
    The complete numerical seed analysis with source paths replaced by stable
    logical identifiers.

``seed_provenance.json``
    A guard manifest binding the private locks, public analysis, checkpoints,
    frozen/assembled/diagnostic artifacts, all eight submission receipts, the
    terminal scheduler ledger, code fingerprints, and the rendered table.

The payload is published before the guard.  A missing guard therefore denotes
an interrupted, incomplete publication and must never be mirrored.  Existing
identical files are accepted; conflicting bytes are never replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from scripts import analyze_binary_seed_extension as seed_analysis_analyzer
from scripts.analyze_binary import load_condition
from scripts.analyze_binary_seed_extension import (
    ANALYSIS_SCHEMA_VERSION,
    analyze_seed_extension,
    validate_analysis_document,
)
from scripts.finalize_seed_scheduler_ledger import (
    _job_id as _canonical_slurm_job_id,
    _validate_public_summary as _validate_scheduler_public_summary,
    load_scheduler_accounting_closure,
)
from scripts.render_binary_seed_extension import render_table
from scripts.submit_binary_seed_extension import (
    DOWNSTREAM_RECEIPT_NAMES,
    TRAIN_RECEIPT,
    _expected_receipt_path,
    plan_downstream_jobs,
    plan_freeze_jobs,
    plan_training_jobs,
)
from scripts.submit_binary_simulations import (
    DEFAULT_CPU_PARTITIONS,
    GPU_ACCOUNT,
    PlannedJob,
    _expected_score_manifests,
    _load_receipt_events,
)
from scripts.assemble_binary_simulations import (
    load_campaign_lock as load_assembly_lock,
    prepare_assembly,
)
from selectseg.binary_artifacts import load_binary_artifact
from selectseg.binary_diagnostics import load_binary_diagnostics
from selectseg.binary_seed_downstream import load_downstream_lock
from selectseg.binary_seed_extension import (
    EXPECTED_AUXILIARY_ID,
    EXPECTED_DATASETS,
    EXPECTED_MODELS,
    EXPECTED_M_VALUES,
    EXPECTED_QUADRATURE_SEED,
    EXPECTED_TRAINING_SEEDS,
    _sha256,
    iter_experiments,
    load_checkpoint_lock,
    load_spec_lock,
)


PUBLIC_ANALYSIS_SCHEMA_VERSION = 1
PUBLIC_PROVENANCE_SCHEMA_VERSION = 1
PUBLIC_ANALYSIS_ARTIFACT_TYPE = "selectseg.binary_seed_public_analysis"
PUBLIC_PROVENANCE_ARTIFACT_TYPE = "selectseg.binary_seed_public_provenance"
PUBLIC_ANALYSIS_BASENAME = "seed_robustness_analysis.json"
PUBLIC_SCHEDULER_BASENAME = "seed_scheduler_summary.json"
PUBLIC_PROVENANCE_BASENAME = "seed_provenance.json"
EXPECTED_EXTENSION_CELLS = 20
EXPECTED_ANALYSIS_CELLS = 10
EXPECTED_TOTAL_JOBS = 162
PUBLIC_PHASES = (
    "train",
    "freeze",
    "common",
    "score",
    "assemble",
    "diagnose",
    "analyze",
    "render",
)
EXPECTED_PHASE_COUNTS = {
    "train": 20,
    "freeze": 20,
    "common": 20,
    "score": 60,
    "assemble": 20,
    "diagnose": 20,
    "analyze": 1,
    "render": 1,
}
PHASE_UNITS = {
    "train": "one-dataset-model-training-seed",
    "freeze": "one-trained-checkpoint",
    "common": "one-frozen-seed-artifact",
    "score": "one-frozen-seed-artifact-one-quadrature-budget",
    "assemble": "one-seed-condition",
    "diagnose": "one-frozen-seed-artifact",
    "analyze": "complete-three-seed-grid",
    "render": "one-seed-robustness-table",
}

_PUBLIC_ANALYSIS_FIELDS = frozenset(
    {"schema_version", "artifact_type", "analysis", "provenance", "cells", "gate_c"}
)
_PUBLIC_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "campaign",
        "datasets",
        "base_models",
        "code",
        "training",
        "cells",
        "phases",
        "scheduler",
        "analysis",
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]*\Z")
_FORBIDDEN_KEYS = frozenset(
    {
        "path",
        "command",
        "job_id",
        "receipt_job_id",
        "record_slurm_job_id",
        "created_utc",
        "account",
        "partition",
        "node",
        "cuda_device",
        "gpu_profile",
        "output_dir",
        "data_root",
    }
)
_FORBIDDEN_KEY_FRAGMENTS = (
    "access_token",
    "api_key",
    "apikey",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
)
_FORBIDDEN_TEXT = (
    re.compile(
        r"(?i)(?:^|[\s\"'=:(])/(?![/\s])"
        r"(?:[A-Z0-9._-]+/)*[A-Z0-9._-]+"
    ),
    re.compile(r"(?i)\b[A-Z]:[\\/]"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?i)(?:\b(?:https?|ftp|ssh|file)://|\bgit@|\bwww\.)"),
    re.compile(r"(?i)\b(?:saffo(?:-a100)?|apollo_agate|ssafo)\b"),
    re.compile(r"(?i)\b(?:sbatch|squeue|sacct|srun|slurm)\b"),
    re.compile(
        r"(?i)(?:"
        + "github"
        + r"_pat_|"
        + "ghp"
        + r"_|"
        + "olp"
        + r"_|"
        + "sk"
        + r"-proj-)"
    ),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|bearer)\b\s*[:=]"
    ),
    re.compile(r"(?i)-----BEGIN [A-Z ]*" + "PRIVATE" + r" KEY-----"),
)

_PUBLIC_ANALYSIS_PROVENANCE_FIELDS = frozenset(
    {
        "source_analysis_sha256",
        "downstream_lock_sha256",
        "canonical_seed0",
        "analysis_source_sha256",
    }
)
_PUBLIC_ANALYSIS_CANONICAL_FIELDS = frozenset(
    {"analysis_sha256", "campaign_lock_sha256"}
)
_PUBLIC_ANALYSIS_CELL_FIELDS = frozenset(
    {"dataset", "condition", "model", "num_images_per_seed", "sources", "summary"}
)
_PUBLIC_ANALYSIS_SOURCE_FIELDS = frozenset(
    {"logical_id", "records_sha256", "manifest_sha256"}
)
_CAMPAIGN_FIELDS = frozenset(
    {
        "auxiliary_id",
        "spec_sha256",
        "spec_lock_sha256",
        "checkpoint_lock_sha256",
        "downstream_lock_sha256",
        "canonical_seed0",
        "seed_campaigns",
        "protocol",
        "estimator",
        "grid",
    }
)
_CAMPAIGN_CANONICAL_FIELDS = frozenset(
    {"campaign_id", "campaign_lock_sha256", "analysis_sha256"}
)
_SEED_CAMPAIGN_FIELDS = frozenset(
    {"training_seed", "campaign_id", "config_sha256", "campaign_lock_sha256"}
)
_PROTOCOL_FIELDS = frozenset(
    {
        "reference_training_seed",
        "training_seeds",
        "checkpoint_rule",
        "gamma",
        "m_values",
        "quadrature_rule",
        "quadrature_seed",
    }
)
_ESTIMATOR_FIELDS = frozenset({"estimator_id", "target_measure", "spec_sha256"})
_GRID_FIELDS = frozenset(
    {
        "dataset_count",
        "model_count",
        "extension_cell_count",
        "three_seed_analysis_cell_count",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "dataset_id",
        "train_split",
        "train_count",
        "train_sample_id_sha256",
        "eval_split",
        "eval_count",
        "eval_sample_id_sha256",
    }
)
_BASE_MODEL_FIELDS = frozenset({"model_id", "identifier", "revision", "files"})
_BASE_MODEL_FILE_FIELDS = frozenset({"logical_name", "sha256"})
_CODE_FIELDS = frozenset(
    {
        "locked_source_files",
        "freeze_source_sha256",
        "common_source_sha256",
        "simulation_source_sha256",
        "assembly_source_sha256",
        "diagnostic_source_sha256",
        "analysis_source_sha256",
        "renderer_source_sha256",
        "exporter_source_sha256",
    }
)
_LOCKED_SOURCE_FIELDS = frozenset({"logical_name", "sha256"})
_TRAINING_FIELDS = frozenset(
    {
        "logical_id",
        "dataset_id",
        "model_id",
        "condition_id",
        "training_seed",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "train_config_sha256",
        "history_sha256",
        "history_record_count",
        "train_record_sha256",
    }
)
_CELL_FIELDS = frozenset(
    {
        "logical_id",
        "dataset_id",
        "model_id",
        "condition_id",
        "training_seed",
        "num_samples",
        "sample_id_sha256",
        "frozen",
        "assembly",
        "diagnostics",
    }
)
_FROZEN_FIELDS = frozenset(
    {"artifact_id", "manifest_sha256", "source_sha256", "checkpoint_sha256"}
)
_ASSEMBLY_FIELDS = frozenset(
    {
        "run_id",
        "manifest_sha256",
        "records_sha256",
        "assembly_source_sha256",
        "common_manifest_sha256",
        "simulation_manifest_sha256",
    }
)
_DIAGNOSTIC_FIELDS = frozenset(
    {
        "diagnostic_id",
        "summary_sha256",
        "source_sha256",
        "descriptor_sha256",
        "descriptor_row_count",
    }
)
_PHASE_FIELDS = frozenset(
    {
        "phase_id",
        "unit",
        "planned_jobs",
        "submitted_jobs",
        "completed_jobs",
        "receipt_sha256",
        "logical_job_bundle_sha256",
        "completion_evidence_sha256",
    }
)
_PROVENANCE_ANALYSIS_FIELDS = frozenset(
    {
        "schema_version",
        "source_analysis_sha256",
        "portable_analysis_sha256",
        "analysis_source_sha256",
        "renderer_source_sha256",
        "table_sha256",
        "cell_count",
        "seed_count",
        "gate_c_fired",
        "gate_c_sha256",
        "training_bundle_sha256",
        "frozen_bundle_sha256",
        "assembly_bundle_sha256",
        "diagnostic_bundle_sha256",
    }
)
_CONDITION_BY_MODEL = {
    "clipseg": "clipseg-target",
    "deeplabv3": "deeplabv3-target",
}
_EXPECTED_LOCKED_SOURCES = (
    "selectseg/train.py",
    "selectseg/data.py",
    "selectseg/models.py",
    "selectseg/freeze_binary_maps.py",
    "selectseg/binary_artifacts.py",
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


def _strict_loads(payload: bytes | str, *, source: str):
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"non-UTF-8 JSON in {source}") from error
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    _finite_tree(value, location=source)
    return value


def _finite_tree(value, *, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, location=f"{location}[{index}]")


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_bytes(value) -> bytes:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _strict_loads(payload, source="generated public JSON")
    _scan_public_payload(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value, *, location):
    if not isinstance(value, str) or _DIGEST.fullmatch(value.lower()) is None:
        raise ValueError(f"{location} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _exact_fields(value, expected, *, location):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{location} must contain exactly {sorted(expected)}")
    return value


def _positive_int(value, *, location, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{location} must be a {qualifier} integer")
    return value


def _finite_number(value, *, location, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    if lower is not None and result < lower:
        raise ValueError(f"{location} must be at least {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{location} must be at most {upper}")
    return result


def _safe_identifier(value, *, location):
    if (
        not isinstance(value, str)
        or _SAFE_IDENTIFIER.fullmatch(value) is None
        or value.startswith("/")
        or ".." in PurePosixPath(value).parts
        or "\\" in value
    ):
        raise ValueError(f"{location} must be a safe portable identifier")
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


def _read_regular(path, *, name):
    source = Path(path)
    _reject_symlink_ancestors(source)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {source}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be a regular file: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _regular_with_digest(path, expected, *, name):
    payload = _read_regular(path, name=name)
    observed = _sha256_bytes(payload)
    if observed != _digest(expected, location=f"expected {name} SHA-256"):
        raise ValueError(f"{name} SHA-256 mismatch")
    return Path(path), payload, observed


def _scan_public_tree(value, *, location="public"):
    if isinstance(value, dict):
        forbidden = set(value) & _FORBIDDEN_KEYS
        if forbidden:
            raise ValueError(f"{location} contains forbidden keys {sorted(forbidden)}")
        for key, child in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key.endswith("_path") or any(
                fragment in normalized_key for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise ValueError(
                    f"{location} contains private infrastructure or a "
                    "credential-bearing key"
                )
            _scan_public_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_public_tree(child, location=f"{location}[{index}]")
    elif isinstance(value, str):
        for pattern in _FORBIDDEN_TEXT:
            if pattern.search(value):
                raise ValueError(
                    f"{location} contains private infrastructure or a secret"
                )


def _scan_public_payload(payload: bytes):
    text = payload.decode("utf-8")
    for pattern in _FORBIDDEN_TEXT:
        if pattern.search(text):
            raise ValueError("generated public payload contains private infrastructure")
    _scan_public_tree(_strict_loads(payload, source="generated public payload"))


def _atomic_new_or_identical(path, payload):
    destination = Path(path)
    _reject_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent)
    if destination.exists() or destination.is_symlink():
        existing = _read_regular(destination, name="existing publication file")
        if existing != payload:
            raise FileExistsError(f"refusing to replace conflicting {destination}")
        return "unchanged"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read_regular(destination, name="raced publication file")
            if existing != payload:
                raise FileExistsError(f"refusing to replace conflicting {destination}")
            return "unchanged"
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return "published"


def _preflight_new_or_identical(path, payload):
    """Reject a conflicting destination before either publication write."""

    destination = Path(path)
    _reject_symlink_ancestors(destination)
    if destination.exists() or destination.is_symlink():
        existing = _read_regular(destination, name="existing publication file")
        if existing != payload:
            raise FileExistsError(f"refusing to replace conflicting {destination}")


def publish_payload_then_guard(
    *, analysis_path, analysis_payload, guard_path, guard_payload
):
    """Publish payload first and provenance guard last, without replacement."""

    if Path(analysis_path).resolve(strict=False) == Path(guard_path).resolve(
        strict=False
    ):
        raise ValueError("public analysis and provenance guard must be different files")
    # Validate both byte strings before publishing either file.
    _scan_public_payload(analysis_payload)
    _scan_public_payload(guard_payload)
    _preflight_new_or_identical(analysis_path, analysis_payload)
    _preflight_new_or_identical(guard_path, guard_payload)
    analysis_status = _atomic_new_or_identical(analysis_path, analysis_payload)
    # If this call is interrupted here, the payload remains deliberately
    # unguarded.  Sync/anonymous release code must require the guard.
    guard_status = _atomic_new_or_identical(guard_path, guard_payload)
    return {"analysis": analysis_status, "guard": guard_status}


def _resolve(value):
    path = Path(value)
    candidate = path if path.is_absolute() else Path.cwd() / path
    _reject_symlink_ancestors(candidate)
    return candidate.absolute()


def _file_binding(path, expected_sha, *, name):
    source, payload, digest = _regular_with_digest(path, expected_sha, name=name)
    _strict_loads(payload, source=str(source))
    return source.resolve(), digest


def _training_record_set_sha256(checkpoint_binding):
    rows = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "training_seed": row["training_seed"],
            "train_record_sha256": row["train_record_sha256"],
        }
        for row in checkpoint_binding["lock"]["checkpoints"]
    ]
    if len(rows) != EXPECTED_EXTENSION_CELLS:
        raise ValueError("checkpoint lock does not contain exactly 20 training records")
    return _sha256_bytes(_canonical_json(rows).encode("ascii"))


def _validate_scheduler_closure(
    *,
    private_ledger,
    public_summary,
    spec_lock_sha256,
    train_receipt_sha256,
    training_record_set_sha256,
    train_job_bindings,
):
    closure = load_scheduler_accounting_closure(
        private_ledger,
        public_summary,
        expected_spec_lock_sha256=spec_lock_sha256,
        expected_receipt_sha256=train_receipt_sha256,
        expected_record_set_sha256=training_record_set_sha256,
        expected_job_bindings=train_job_bindings,
        require_complete=True,
    )
    return copy.deepcopy(closure["public"])


def _expected_assemble_jobs(downstream_binding):
    """Reconstruct completed-state assembly jobs without an absence check."""

    jobs = []
    for campaign_binding in downstream_binding["campaigns"]:
        seed = campaign_binding["training_seed"]
        config = campaign_binding["config"]
        lock_path = campaign_binding["campaign_lock_path"]
        lock_sha = campaign_binding["campaign_lock_sha256"]
        lock = campaign_binding["campaign"]
        assembly_lock = load_assembly_lock(lock_path)
        partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
        for artifact_index, artifact in enumerate(lock["artifacts"]):
            common, simulations = _expected_score_manifests(
                config, lock_path, lock_sha, lock, artifact
            )
            dataset, condition, _, _, _ = prepare_assembly(
                assembly_lock, common, simulations
            )
            partition = partitions[artifact_index % len(partitions)]
            command = [
                "sbatch",
                "--parsable",
                "--job-name",
                f"selseg-seed-s{seed}-selseg-assemble-{dataset}-{condition}"[:128],
                "--partition",
                partition,
                "--account",
                GPU_ACCOUNT,
                "scripts/slurm/assemble_binary_simulations.sbatch",
                "--campaign-lock",
                str(lock_path),
                "--common",
                str(common),
            ]
            for simulation in simulations:
                command.extend(("--input", str(simulation)))
            command.extend(("--output-root", lock["paths"]["assembly_output_root"]))
            jobs.append(
                PlannedJob(
                    phase="seed_assemble",
                    key=(seed, dataset, condition, partition),
                    command=tuple(command),
                )
            )
    if len(jobs) != 20 or len({(job.phase, job.key) for job in jobs}) != 20:
        raise RuntimeError("completed-state assembly plan must contain 20 jobs")
    return tuple(jobs)


def _expected_analysis_job(
    downstream_binding, *, canonical_analysis, expected_canonical_analysis_sha256
):
    output = (
        Path(downstream_binding["binding"]["spec"]["paths"]["analysis_root"])
        / "analysis.json"
    )
    partition = "agsmall"
    return (
        PlannedJob(
            phase="seed_analyze",
            key=("all-seeds", partition),
            command=(
                "sbatch",
                "--parsable",
                "--job-name",
                "selseg-seed-analysis",
                "--partition",
                partition,
                "--account",
                GPU_ACCOUNT,
                "scripts/slurm/analyze_binary_seed_extension.sbatch",
                "--downstream-lock",
                downstream_binding["path"].as_posix(),
                "--expected-downstream-lock-sha256",
                downstream_binding["sha256"],
                "--canonical-analysis",
                str(canonical_analysis),
                "--expected-canonical-analysis-sha256",
                expected_canonical_analysis_sha256,
                "--output",
                output.as_posix(),
            ),
        ),
    )


def _expected_render_job(downstream_binding, *, seed_analysis, seed_analysis_sha256):
    analysis_path = Path(seed_analysis)
    partition = "agsmall"
    return (
        PlannedJob(
            phase="seed_render",
            key=("seed-robustness-table", partition),
            command=(
                "sbatch",
                "--parsable",
                "--job-name",
                "selseg-seed-render",
                "--partition",
                partition,
                "--account",
                GPU_ACCOUNT,
                "scripts/slurm/render_binary_seed_extension.sbatch",
                "--analysis",
                analysis_path.as_posix(),
                "--expected-analysis-sha256",
                seed_analysis_sha256,
                "--output",
                analysis_path.with_name("seed_robustness.tex").as_posix(),
            ),
        ),
    )


def _expected_job_plans(
    binding,
    checkpoint_binding,
    downstream_binding,
    *,
    canonical_analysis,
    expected_canonical_analysis_sha256,
    seed_analysis,
    seed_analysis_sha256,
):
    return {
        "train": plan_training_jobs(binding),
        "freeze": plan_freeze_jobs(binding, checkpoint_binding),
        "common": plan_downstream_jobs(downstream_binding, "common"),
        "score": plan_downstream_jobs(downstream_binding, "score"),
        "assemble": _expected_assemble_jobs(downstream_binding),
        "diagnose": plan_downstream_jobs(downstream_binding, "diagnose"),
        "analyze": _expected_analysis_job(
            downstream_binding,
            canonical_analysis=canonical_analysis,
            expected_canonical_analysis_sha256=expected_canonical_analysis_sha256,
        ),
        "render": _expected_render_job(
            downstream_binding,
            seed_analysis=seed_analysis,
            seed_analysis_sha256=seed_analysis_sha256,
        ),
    }


def _science_key(phase, job):
    key = job.key
    if phase in {"train", "freeze"}:
        return [key[0], key[1], key[2]]
    if phase == "common":
        return [key[0], key[1], key[2], key[4]]
    if phase == "score":
        return [key[0], key[1], key[2], key[4], key[5], key[6]]
    if phase in {"assemble", "diagnose"}:
        return [key[0], key[1], key[2]]
    return [key[0]]


def _validate_receipt(path, jobs, *, phase):
    source = Path(path)
    _reject_symlink_ancestors(source)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{phase} receipt must be a regular file: {source}")
    with source.open("r", encoding="utf-8") as handle:
        latest = _load_receipt_events(handle, jobs, source=source)
    expected = {(job.phase, job.key) for job in jobs}
    if set(latest) != expected or len(latest) != EXPECTED_PHASE_COUNTS[phase]:
        raise ValueError(f"{phase} receipt does not cover its exact expected plan")
    submitted = [event for event in latest.values() if event["status"] == "submitted"]
    if len(submitted) != len(jobs):
        raise ValueError(f"{phase} receipt has unresolved or failed submission intents")
    job_ids = [
        _canonical_slurm_job_id(
            event["job_id"], location=f"{phase} receipt submitted job id"
        )
        for event in submitted
    ]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError(f"{phase} receipt reuses a submitted job id")
    science_keys = [_science_key(phase, job) for job in jobs]
    if len({_canonical_json(key) for key in science_keys}) != len(jobs):
        raise ValueError(f"{phase} plan fuses or duplicates scientific experiments")
    if any("--array" in job.command for job in jobs):
        raise ValueError(f"{phase} plan contains a forbidden Slurm array")
    result = {
        "receipt_sha256": _sha256(source),
        "logical_job_bundle_sha256": _sha256_bytes(
            _canonical_json(science_keys).encode("ascii")
        ),
        "count": len(jobs),
        "job_ids": tuple(job_ids),
    }
    if phase == "train":
        bindings = {}
        for job in jobs:
            identity = (job.key[0], job.key[1], job.key[2])
            event = latest[(job.phase, job.key)]
            if identity in bindings:
                raise ValueError("training receipt repeats a scientific identity")
            bindings[identity] = event["job_id"]
        if len(bindings) != EXPECTED_EXTENSION_CELLS:
            raise ValueError("training receipt must bind 20 scientific identities")
        result["job_bindings"] = bindings
    return result


def _validate_all_receipts(binding, plans, *, train_receipt, downstream_receipts):
    if set(downstream_receipts) != set(DOWNSTREAM_RECEIPT_NAMES):
        raise ValueError(
            "downstream receipts must contain exactly freeze/common/score/assemble/"
            "diagnose/analyze/render"
        )
    if Path(train_receipt).resolve() != Path(TRAIN_RECEIPT).resolve():
        raise ValueError(f"train receipt must use the fixed path {TRAIN_RECEIPT}")
    result = {"train": _validate_receipt(train_receipt, plans["train"], phase="train")}
    for phase in DOWNSTREAM_RECEIPT_NAMES:
        path = downstream_receipts[phase]
        expected_path = _expected_receipt_path(binding, phase)
        if Path(path).resolve() != Path(expected_path).resolve():
            raise ValueError(f"{phase} receipt must use the fixed path {expected_path}")
        result[phase] = _validate_receipt(path, plans[phase], phase=phase)
    if tuple(result) != PUBLIC_PHASES:
        raise RuntimeError("receipt phases are not in the fixed public order")
    if sum(item["count"] for item in result.values()) != EXPECTED_TOTAL_JOBS:
        raise RuntimeError("receipt closure must contain exactly 162 jobs")
    all_job_ids = [job_id for item in result.values() for job_id in item["job_ids"]]
    if len(set(all_job_ids)) != EXPECTED_TOTAL_JOBS:
        raise ValueError("the eight receipts do not bind 162 globally unique jobs")
    return result


def _base_models(spec_lock):
    rows = spec_lock["base_model_files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("seed spec lock has no base-model file bindings")
    clip = []
    deeplab = []
    revision = None
    for row in rows:
        _exact_fields(row, {"path", "sha256"}, location="base_model_files[]")
        path = PurePosixPath(row["path"])
        digest = _digest(row["sha256"], location="base model file digest")
        if "models--CIDAS--clipseg-rd64-refined" in path.parts:
            try:
                index = path.parts.index("snapshots")
                candidate_revision = path.parts[index + 1]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "CLIPSeg base-model path lacks an immutable revision"
                ) from error
            if revision is not None and revision != candidate_revision:
                raise ValueError("CLIPSeg base-model files name multiple revisions")
            revision = candidate_revision
            clip.append({"logical_name": path.name, "sha256": digest})
        elif path.name == "deeplabv3_resnet50_coco-cd0a2569.pth":
            deeplab.append({"logical_name": "weights", "sha256": digest})
        else:
            raise ValueError("unrecognized base-model file in seed spec lock")
    if revision is None or not clip or len(deeplab) != 1:
        raise ValueError("base-model bindings are incomplete")
    clip.sort(key=lambda item: item["logical_name"])
    return [
        {
            "model_id": "clipseg",
            "identifier": "CIDAS/clipseg-rd64-refined",
            "revision": revision,
            "files": clip,
        },
        {
            "model_id": "deeplabv3",
            "identifier": "torchvision/DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
            "revision": "torchvision-0.27.1",
            "files": deeplab,
        },
    ]


def _datasets(spec):
    result = []
    for row in spec["datasets"]:
        result.append(
            {
                "dataset_id": row["name"],
                "train_split": row["train_split"],
                "train_count": row["train_count"],
                "train_sample_id_sha256": row["train_sample_id_sha256"],
                "eval_split": row["eval_split"],
                "eval_count": row["eval_count"],
                "eval_sample_id_sha256": row["eval_sample_id_sha256"],
            }
        )
    if tuple(row["dataset_id"] for row in result) != EXPECTED_DATASETS:
        raise ValueError("public dataset summary differs from the fixed grid")
    return result


def _training(checkpoint_binding):
    result = []
    for row in checkpoint_binding["lock"]["checkpoints"]:
        history_path = _resolve(row["history_path"])
        _, history_payload, _ = _regular_with_digest(
            history_path,
            row["history_sha256"],
            name="seed training history",
        )
        history = _strict_loads(
            history_payload,
            source=str(history_path),
        )
        if not isinstance(history, list) or len(history) != 40:
            raise ValueError("seed training history must contain exactly 40 epochs")
        result.append(
            {
                "logical_id": f"{row['dataset']}/{row['model']}/seed-{row['training_seed']}",
                "dataset_id": row["dataset"],
                "model_id": row["model"],
                "condition_id": row["condition"],
                "training_seed": row["training_seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_size_bytes": row["checkpoint_size_bytes"],
                "train_config_sha256": row["train_config_sha256"],
                "history_sha256": row["history_sha256"],
                "history_record_count": len(history),
                "train_record_sha256": row["train_record_sha256"],
            }
        )
    if len(result) != 20 or len({row["logical_id"] for row in result}) != 20:
        raise ValueError("public training summary must contain 20 unique cells")
    return result


def _diagnostics_by_artifact(paths):
    if len(paths) != EXPECTED_EXTENSION_CELLS:
        raise ValueError("exactly 20 diagnostic summaries must be supplied explicitly")
    result = {}
    seen_paths = set()
    for path in paths:
        resolved = _resolve(path)
        if resolved in seen_paths:
            raise ValueError("duplicate diagnostic summary input")
        seen_paths.add(resolved)
        loaded = load_binary_diagnostics(resolved, validate_descriptors=True)
        artifact_id = loaded.summary["artifact"]["artifact_id"]
        if artifact_id in result:
            raise ValueError("multiple diagnostics bind the same frozen artifact")
        result[artifact_id] = loaded
    if len(result) != EXPECTED_EXTENSION_CELLS:
        raise ValueError("diagnostic closure does not contain 20 artifacts")
    return result


def _manifest_binding(path, expected_sha, *, name):
    source, payload, observed = _regular_with_digest(path, expected_sha, name=name)
    manifest = _strict_loads(payload, source=str(source))
    if not isinstance(manifest, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return source, observed, manifest


def _cell_closure(downstream_binding, analysis, diagnostic_paths):
    diagnostics = _diagnostics_by_artifact(diagnostic_paths)
    checkpoint_by_cell = {
        (row["dataset"], row["model"], row["training_seed"]): row
        for row in downstream_binding["checkpoint_binding"]["lock"]["checkpoints"]
    }
    freezes = {
        (row["dataset"], row["model"], row["training_seed"]): row
        for campaign in downstream_binding["lock"]["campaigns"]
        for row in campaign["freeze_records"]
    }
    campaign_by_seed = {
        row["training_seed"]: row for row in downstream_binding["campaigns"]
    }
    analysis_by_key = {
        (row["dataset"], row["condition"]): row for row in analysis["cells"]
    }
    result = []
    stage = {
        "freeze": set(),
        "common": set(),
        "simulation": set(),
        "assembly": set(),
        "diagnostic": set(),
    }
    used_diagnostics = set()
    for experiment in iter_experiments(downstream_binding["binding"]["spec"]):
        dataset = experiment["dataset"]["name"]
        model = experiment["model"]["name"]
        condition = experiment["model"]["condition"]
        seed = experiment["training_seed"]
        key = (dataset, model, seed)
        checkpoint = checkpoint_by_cell[key]
        freeze = freezes[key]

        # The downstream loader checks the manifest; publication additionally
        # hashes every probability/truth payload once.
        artifact = load_binary_artifact(
            _resolve(freeze["artifact_manifest_path"]), validate_payloads=True
        )
        if artifact.manifest_sha256 != freeze["artifact_manifest_sha256"]:
            raise ValueError("frozen artifact changed after the downstream lock")
        manifest = artifact.manifest
        stage["freeze"].add(
            _digest(manifest["source_sha256"], location="freeze source")
        )

        source = analysis_by_key[(dataset, condition)]["sources"][str(seed)]
        records_path = _resolve(source["records"])
        manifest_path = _resolve(source["manifest"])
        loaded_condition = load_condition(records_path)
        if loaded_condition.manifest_path.absolute() != manifest_path:
            raise ValueError("seed analysis names a different assembly manifest")
        if (
            _sha256_bytes(_read_regular(manifest_path, name="assembly manifest"))
            != source["manifest_sha256"]
        ):
            raise ValueError("seed analysis assembly manifest changed")
        assembly = loaded_condition.manifest
        if (
            loaded_condition.manifest["jsonl_sha256"] != source["records_sha256"]
            or assembly["dataset"] != dataset
            or assembly["condition"] != condition
            or assembly["model"] != model
            or assembly["num_images"] != experiment["dataset"]["eval_count"]
            or assembly["sample_id_sha256"]
            != experiment["dataset"]["eval_sample_id_sha256"]
            or assembly["checkpoint"]["sha256"] != checkpoint["checkpoint_sha256"]
        ):
            raise ValueError("seed assembly differs from its locked cell")
        assembly_binding = assembly["assembly"]
        campaign = campaign_by_seed[seed]
        if (
            assembly_binding["campaign_lock_sha256"] != campaign["campaign_lock_sha256"]
            or assembly_binding["artifact_manifest_sha256"]
            != freeze["artifact_manifest_sha256"]
        ):
            raise ValueError("seed assembly is bound to a different campaign/artifact")
        stage["common"].add(
            _digest(assembly["source_sha256"], location="common source")
        )
        stage["assembly"].add(
            _digest(
                assembly_binding["assembly_source_sha256"],
                location="assembly source",
            )
        )
        common_path = _resolve(assembly_binding["common_manifest"]["path"])
        _, common_sha, common_manifest = _manifest_binding(
            common_path,
            assembly_binding["common_manifest_sha256"],
            name="common-score manifest",
        )
        if common_manifest["jsonl_sha256"] != assembly_binding["common_jsonl_sha256"]:
            raise ValueError("common shard differs from the assembly")
        simulation_shas = {}
        for count in EXPECTED_M_VALUES:
            shard = assembly_binding["simulation_manifests"][str(count)]
            _, shard_sha, shard_manifest = _manifest_binding(
                _resolve(shard["path"]), shard["sha256"], name="simulation manifest"
            )
            if shard_manifest["jsonl_sha256"] != shard["jsonl_sha256"]:
                raise ValueError("simulation shard differs from the assembly")
            stage["simulation"].add(
                _digest(shard_manifest["source_sha256"], location="simulation source")
            )
            simulation_shas[str(count)] = shard_sha

        diagnostic = diagnostics.get(manifest["artifact_id"])
        if diagnostic is None:
            raise ValueError("missing diagnostic for one frozen seed artifact")
        used_diagnostics.add(manifest["artifact_id"])
        summary = diagnostic.summary
        if (
            summary["artifact"]["manifest_sha256"] != artifact.manifest_sha256
            or summary["artifact"]["sample_id_sha256"] != manifest["sample_id_sha256"]
            or summary["artifact"]["num_samples"] != manifest["num_samples"]
            or summary["artifact"]["dataset"] != dataset
            or summary["artifact"]["condition"] != condition
            or summary["artifact"]["model"] != model
        ):
            raise ValueError("diagnostic summary differs from its frozen artifact")
        descriptor = summary["descriptors"]
        if (
            descriptor["included"] is not True
            or descriptor["num_rows"] != manifest["num_samples"]
        ):
            raise ValueError("seed diagnostic must include one descriptor per image")
        stage["diagnostic"].add(
            _digest(summary["source_sha256"], location="diagnostic source")
        )
        result.append(
            {
                "logical_id": f"{dataset}/{condition}/seed-{seed}",
                "dataset_id": dataset,
                "model_id": model,
                "condition_id": condition,
                "training_seed": seed,
                "num_samples": manifest["num_samples"],
                "sample_id_sha256": manifest["sample_id_sha256"],
                "frozen": {
                    "artifact_id": manifest["artifact_id"],
                    "manifest_sha256": artifact.manifest_sha256,
                    "source_sha256": manifest["source_sha256"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                },
                "assembly": {
                    "run_id": assembly["run_id"],
                    "manifest_sha256": source["manifest_sha256"],
                    "records_sha256": source["records_sha256"],
                    "assembly_source_sha256": assembly_binding[
                        "assembly_source_sha256"
                    ],
                    "common_manifest_sha256": common_sha,
                    "simulation_manifest_sha256": simulation_shas,
                },
                "diagnostics": {
                    "diagnostic_id": summary["diagnostic_id"],
                    "summary_sha256": diagnostic.summary_sha256,
                    "source_sha256": summary["source_sha256"],
                    "descriptor_sha256": descriptor["sha256"],
                    "descriptor_row_count": descriptor["num_rows"],
                },
            }
        )
    if len(result) != 20 or len({row["logical_id"] for row in result}) != 20:
        raise ValueError("seed artifact closure must contain 20 unique cells")
    if used_diagnostics != set(diagnostics):
        raise ValueError("a diagnostic input lies outside the locked seed grid")
    code = {}
    for phase, values in stage.items():
        if len(values) != 1:
            raise ValueError(
                f"{phase} artifacts were produced by multiple source bundles"
            )
        code[f"{phase}_source_sha256"] = next(iter(values))
    return result, code


def _renderer_source_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts" / "render_binary_seed_extension.py",
        root / "scripts" / "analyze_binary_seed_extension.py",
        root / "scripts" / "analyze_binary.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_read_regular(path, name="renderer source"))
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_analysis(analysis, *, source_sha256, downstream_sha256, canonical):
    cells = []
    for cell in analysis["cells"]:
        sources = {}
        for seed in (0, 1, 2):
            source = cell["sources"][str(seed)]
            manifest_path = _resolve(source["manifest"])
            _, manifest_payload, _ = _regular_with_digest(
                manifest_path,
                source["manifest_sha256"],
                name="seed assembly manifest",
            )
            manifest = _strict_loads(
                manifest_payload,
                source=str(manifest_path),
            )
            sources[str(seed)] = {
                "logical_id": (
                    f"seed-{seed}/{cell['dataset']}/{cell['condition']}/"
                    f"{manifest['run_id']}"
                ),
                "records_sha256": source["records_sha256"],
                "manifest_sha256": source["manifest_sha256"],
            }
        cells.append(
            {
                "dataset": cell["dataset"],
                "condition": cell["condition"],
                "model": cell["model"],
                "num_images_per_seed": cell["num_images_per_seed"],
                "sources": sources,
                "summary": copy.deepcopy(cell["summary"]),
            }
        )
    return {
        "schema_version": PUBLIC_ANALYSIS_SCHEMA_VERSION,
        "artifact_type": PUBLIC_ANALYSIS_ARTIFACT_TYPE,
        "analysis": copy.deepcopy(analysis["analysis"]),
        "provenance": {
            "source_analysis_sha256": source_sha256,
            "downstream_lock_sha256": downstream_sha256,
            "canonical_seed0": {
                "analysis_sha256": canonical["sha256"],
                "campaign_lock_sha256": canonical["campaign_lock_sha256"],
            },
            "analysis_source_sha256": analysis["provenance"]["analysis_source_sha256"],
        },
        "cells": cells,
        "gate_c": copy.deepcopy(analysis["gate_c"]),
    }


def _validate_public_analysis(value):
    _scan_public_tree(value)
    _exact_fields(value, _PUBLIC_ANALYSIS_FIELDS, location="public seed analysis")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != PUBLIC_ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("unsupported public seed-analysis schema")
    if value["artifact_type"] != PUBLIC_ANALYSIS_ARTIFACT_TYPE:
        raise ValueError("unexpected public seed-analysis artifact type")

    provenance = _exact_fields(
        value["provenance"],
        _PUBLIC_ANALYSIS_PROVENANCE_FIELDS,
        location="public analysis.provenance",
    )
    for field in (
        "source_analysis_sha256",
        "downstream_lock_sha256",
        "analysis_source_sha256",
    ):
        _digest(provenance[field], location=f"public analysis.provenance.{field}")
    canonical = _exact_fields(
        provenance["canonical_seed0"],
        _PUBLIC_ANALYSIS_CANONICAL_FIELDS,
        location="public analysis.provenance.canonical_seed0",
    )
    for field in _PUBLIC_ANALYSIS_CANONICAL_FIELDS:
        _digest(
            canonical[field],
            location=f"public analysis.provenance.canonical_seed0.{field}",
        )

    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != EXPECTED_ANALYSIS_CELLS:
        raise ValueError("public seed analysis must contain ten cells")
    reconstructed_cells = []
    logical_ids = set()
    for index, cell in enumerate(value["cells"]):
        _exact_fields(
            cell,
            _PUBLIC_ANALYSIS_CELL_FIELDS,
            location=f"public analysis.cells[{index}]",
        )
        sources = cell["sources"]
        if not isinstance(sources, dict) or set(sources) != {"0", "1", "2"}:
            raise ValueError("public analysis source seeds are incomplete")
        reconstructed_sources = {}
        for seed, source in sources.items():
            _exact_fields(
                source,
                _PUBLIC_ANALYSIS_SOURCE_FIELDS,
                location=f"public analysis source {seed}",
            )
            logical_id = _safe_identifier(
                source["logical_id"], location="assembly logical_id"
            )
            parts = PurePosixPath(logical_id).parts
            if len(parts) != 4 or parts[:3] != (
                f"seed-{seed}",
                cell["dataset"],
                cell["condition"],
            ):
                raise ValueError(
                    "public analysis source logical_id differs from its cell/seed"
                )
            _safe_identifier(parts[3], location="assembly run_id")
            if logical_id in logical_ids:
                raise ValueError("public analysis reuses an assembly logical_id")
            logical_ids.add(logical_id)
            _digest(source["records_sha256"], location="records_sha256")
            _digest(source["manifest_sha256"], location="manifest_sha256")
            reconstructed_sources[seed] = {
                "records": f"{logical_id}/records.jsonl",
                "records_sha256": source["records_sha256"],
                "manifest": f"{logical_id}/manifest.json",
                "manifest_sha256": source["manifest_sha256"],
            }
        reconstructed_cells.append(
            {
                "dataset": cell["dataset"],
                "condition": cell["condition"],
                "model": cell["model"],
                "num_images_per_seed": cell["num_images_per_seed"],
                "sources": reconstructed_sources,
                "summary": copy.deepcopy(cell["summary"]),
            }
        )

    # Reconstruct the private analyzer's path-bearing envelope with inert,
    # portable logical names.  Its strict validator then checks the complete
    # metadata schema, the fixed dataset/condition grid, every RISKS x METHODS
    # three-seed statistic, every CONTRASTS identity, and the recomputed Gate C
    # decision.  No private path is reintroduced into the released document.
    reconstructed = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": copy.deepcopy(value["analysis"]),
        "provenance": {
            "downstream_lock": {
                "path": "portable/downstream-lock",
                "sha256": provenance["downstream_lock_sha256"],
            },
            "canonical_seed0": {
                "path": "portable/canonical-seed0-analysis",
                "sha256": canonical["analysis_sha256"],
                "campaign_lock_path": "portable/canonical-seed0-campaign-lock",
                "campaign_lock_sha256": canonical["campaign_lock_sha256"],
            },
            "analysis_source_sha256": provenance["analysis_source_sha256"],
        },
        "cells": reconstructed_cells,
        "gate_c": copy.deepcopy(value["gate_c"]),
    }
    seed_analysis_analyzer.validate_analysis_document(
        reconstructed, require_current_source=False
    )
    _scan_public_tree(value)
    return value


def _phase_records(receipts, *, evidence):
    records = []
    for phase in PUBLIC_PHASES:
        receipt = receipts[phase]
        records.append(
            {
                "phase_id": phase,
                "unit": PHASE_UNITS[phase],
                "planned_jobs": receipt["count"],
                "submitted_jobs": receipt["count"],
                "completed_jobs": receipt["count"],
                "receipt_sha256": receipt["receipt_sha256"],
                "logical_job_bundle_sha256": receipt["logical_job_bundle_sha256"],
                "completion_evidence_sha256": evidence[phase],
            }
        )
    return records


def _validate_public_provenance(value):
    _scan_public_tree(value)
    _exact_fields(value, _PUBLIC_PROVENANCE_FIELDS, location="public seed provenance")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != PUBLIC_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported public seed-provenance schema")
    if value["artifact_type"] != PUBLIC_PROVENANCE_ARTIFACT_TYPE:
        raise ValueError("unexpected public seed-provenance artifact type")

    campaign = _exact_fields(
        value["campaign"], _CAMPAIGN_FIELDS, location="provenance.campaign"
    )
    if campaign["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("public campaign has an unexpected auxiliary_id")
    for field in (
        "spec_sha256",
        "spec_lock_sha256",
        "checkpoint_lock_sha256",
        "downstream_lock_sha256",
    ):
        _digest(campaign[field], location=f"provenance.campaign.{field}")
    canonical = _exact_fields(
        campaign["canonical_seed0"],
        _CAMPAIGN_CANONICAL_FIELDS,
        location="provenance.campaign.canonical_seed0",
    )
    if canonical["campaign_id"] != "binary-midpoint-main-v1":
        raise ValueError("public provenance is not anchored to the canonical seed 0")
    for field in ("campaign_lock_sha256", "analysis_sha256"):
        _digest(canonical[field], location=f"canonical_seed0.{field}")

    seed_campaigns = campaign["seed_campaigns"]
    if not isinstance(seed_campaigns, list) or len(seed_campaigns) != 2:
        raise ValueError("public campaign must contain exactly seed 1 and seed 2")
    for index, row in enumerate(seed_campaigns):
        location = f"provenance.campaign.seed_campaigns[{index}]"
        _exact_fields(row, _SEED_CAMPAIGN_FIELDS, location=location)
        seed = _positive_int(row["training_seed"], location=f"{location}.training_seed")
        if seed != EXPECTED_TRAINING_SEEDS[index]:
            raise ValueError("public seed campaigns are not in the fixed order")
        if row["campaign_id"] != f"binary-seed{seed}-v1":
            raise ValueError("public seed campaign has an unexpected campaign_id")
        for field in ("config_sha256", "campaign_lock_sha256"):
            _digest(row[field], location=f"{location}.{field}")

    protocol = _exact_fields(
        campaign["protocol"], _PROTOCOL_FIELDS, location="provenance.campaign.protocol"
    )
    if (
        _positive_int(
            protocol["reference_training_seed"],
            location="protocol.reference_training_seed",
            allow_zero=True,
        )
        != 0
    ):
        raise ValueError("reference training seed must be zero")
    if (
        not isinstance(protocol["training_seeds"], list)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in protocol["training_seeds"]
        )
        or tuple(protocol["training_seeds"]) != EXPECTED_TRAINING_SEEDS
    ):
        raise ValueError("public protocol training seeds differ from the fixed design")
    if protocol["checkpoint_rule"] != "final_epoch_40":
        raise ValueError(
            "public protocol checkpoint rule differs from the fixed design"
        )
    if _finite_number(protocol["gamma"], location="protocol.gamma") != 0.5:
        raise ValueError("public protocol gamma differs from the fixed design")
    if (
        not isinstance(protocol["m_values"], list)
        or any(
            isinstance(count, bool) or not isinstance(count, int)
            for count in protocol["m_values"]
        )
        or tuple(protocol["m_values"]) != EXPECTED_M_VALUES
    ):
        raise ValueError(
            "public protocol quadrature budgets differ from the fixed design"
        )
    if protocol["quadrature_rule"] != "midpoint-v1":
        raise ValueError(
            "public protocol quadrature rule differs from the fixed design"
        )
    if (
        _positive_int(
            protocol["quadrature_seed"],
            location="protocol.quadrature_seed",
            allow_zero=True,
        )
        != EXPECTED_QUADRATURE_SEED
    ):
        raise ValueError(
            "public protocol quadrature seed differs from the fixed design"
        )

    estimator = _exact_fields(
        campaign["estimator"],
        _ESTIMATOR_FIELDS,
        location="provenance.campaign.estimator",
    )
    if estimator["estimator_id"] != "midpoint-v1":
        raise ValueError("public estimator has an unexpected estimator_id")
    if estimator["target_measure"] != "uniform-threshold":
        raise ValueError("public estimator has an unexpected target measure")
    _digest(estimator["spec_sha256"], location="estimator.spec_sha256")
    grid = _exact_fields(
        campaign["grid"], _GRID_FIELDS, location="provenance.campaign.grid"
    )
    expected_grid = {
        "dataset_count": len(EXPECTED_DATASETS),
        "model_count": len(EXPECTED_MODELS),
        "extension_cell_count": EXPECTED_EXTENSION_CELLS,
        "three_seed_analysis_cell_count": EXPECTED_ANALYSIS_CELLS,
    }
    for field, expected in expected_grid.items():
        observed = _positive_int(grid[field], location=f"campaign.grid.{field}")
        if observed != expected:
            raise ValueError("public campaign grid differs from the fixed design")

    datasets = value["datasets"]
    if not isinstance(datasets, list) or len(datasets) != len(EXPECTED_DATASETS):
        raise ValueError("public provenance must contain the fixed five datasets")
    dataset_by_id = {}
    expected_splits = {
        "pet": ("trainval", "test"),
        "kvasir": ("train", "test"),
        "fives": ("train", "test"),
        "isic": ("train", "test"),
        "tn3k": ("train", "test"),
    }
    for index, row in enumerate(datasets):
        location = f"provenance.datasets[{index}]"
        _exact_fields(row, _DATASET_FIELDS, location=location)
        dataset = row["dataset_id"]
        if dataset != EXPECTED_DATASETS[index]:
            raise ValueError("public datasets are not in the fixed order")
        if (row["train_split"], row["eval_split"]) != expected_splits[dataset]:
            raise ValueError(f"public split protocol changed for {dataset}")
        for field in ("train_count", "eval_count"):
            _positive_int(row[field], location=f"{location}.{field}")
        for field in ("train_sample_id_sha256", "eval_sample_id_sha256"):
            _digest(row[field], location=f"{location}.{field}")
        dataset_by_id[dataset] = row

    base_models = value["base_models"]
    if not isinstance(base_models, list) or len(base_models) != len(EXPECTED_MODELS):
        raise ValueError("public provenance must bind exactly two base models")
    expected_model_metadata = {
        "clipseg": ("CIDAS/clipseg-rd64-refined", None),
        "deeplabv3": (
            "torchvision/DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
            "torchvision-0.27.1",
        ),
    }
    for index, row in enumerate(base_models):
        location = f"provenance.base_models[{index}]"
        _exact_fields(row, _BASE_MODEL_FIELDS, location=location)
        model = row["model_id"]
        if model != EXPECTED_MODELS[index]:
            raise ValueError("public base models are not in the fixed order")
        expected_identifier, expected_revision = expected_model_metadata[model]
        if row["identifier"] != expected_identifier:
            raise ValueError(f"public base-model identifier changed for {model}")
        revision = _safe_identifier(row["revision"], location=f"{location}.revision")
        if model == "clipseg":
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError("CLIPSeg revision must be an immutable commit digest")
        elif revision != expected_revision:
            raise ValueError("DeepLabV3 revision differs from the locked stack")
        files = row["files"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"{location}.files must be a non-empty list")
        logical_names = []
        for file_index, file_row in enumerate(files):
            file_location = f"{location}.files[{file_index}]"
            _exact_fields(file_row, _BASE_MODEL_FILE_FIELDS, location=file_location)
            logical_names.append(
                _safe_identifier(
                    file_row["logical_name"], location=f"{file_location}.logical_name"
                )
            )
            _digest(file_row["sha256"], location=f"{file_location}.sha256")
        if logical_names != sorted(set(logical_names)):
            raise ValueError(f"{location}.files must be unique and canonical")
        if model == "deeplabv3" and logical_names != ["weights"]:
            raise ValueError("DeepLabV3 must bind its single locked weight file")

    code = _exact_fields(value["code"], _CODE_FIELDS, location="provenance.code")
    locked_sources = code["locked_source_files"]
    if not isinstance(locked_sources, list) or len(locked_sources) != len(
        _EXPECTED_LOCKED_SOURCES
    ):
        raise ValueError("public code binding must contain five locked source files")
    for index, row in enumerate(locked_sources):
        location = f"provenance.code.locked_source_files[{index}]"
        _exact_fields(row, _LOCKED_SOURCE_FIELDS, location=location)
        if row["logical_name"] != _EXPECTED_LOCKED_SOURCES[index]:
            raise ValueError("public locked source files differ from the fixed bundle")
        _safe_identifier(row["logical_name"], location=f"{location}.logical_name")
        _digest(row["sha256"], location=f"{location}.sha256")
    for field in _CODE_FIELDS - {"locked_source_files"}:
        _digest(code[field], location=f"provenance.code.{field}")

    expected_identities = [
        (dataset, model, seed)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for seed in EXPECTED_TRAINING_SEEDS
    ]
    training = value["training"]
    if not isinstance(training, list) or len(training) != EXPECTED_EXTENSION_CELLS:
        raise ValueError("public provenance must contain 20 training cells")
    training_by_identity = {}
    for index, row in enumerate(training):
        location = f"provenance.training[{index}]"
        _exact_fields(row, _TRAINING_FIELDS, location=location)
        identity = (row["dataset_id"], row["model_id"], row["training_seed"])
        if identity != expected_identities[index]:
            raise ValueError("public training cells are not the fixed ordered grid")
        dataset, model, seed = identity
        if isinstance(seed, bool) or seed not in EXPECTED_TRAINING_SEEDS:
            raise ValueError(f"{location}.training_seed is invalid")
        if row["condition_id"] != _CONDITION_BY_MODEL[model]:
            raise ValueError(f"{location}.condition_id differs from its model")
        if row["logical_id"] != f"{dataset}/{model}/seed-{seed}":
            raise ValueError(f"{location}.logical_id differs from its identity")
        _safe_identifier(row["logical_id"], location=f"{location}.logical_id")
        _positive_int(
            row["checkpoint_size_bytes"], location=f"{location}.checkpoint_size_bytes"
        )
        if row["history_record_count"] != 40 or isinstance(
            row["history_record_count"], bool
        ):
            raise ValueError("public training history must contain exactly 40 epochs")
        for field in (
            "checkpoint_sha256",
            "train_config_sha256",
            "history_sha256",
            "train_record_sha256",
        ):
            _digest(row[field], location=f"{location}.{field}")
        training_by_identity[identity] = row

    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != EXPECTED_EXTENSION_CELLS:
        raise ValueError("public provenance must contain 20 artifact cells")
    artifact_ids = set()
    run_ids = set()
    for index, row in enumerate(cells):
        location = f"provenance.cells[{index}]"
        _exact_fields(row, _CELL_FIELDS, location=location)
        identity = (row["dataset_id"], row["model_id"], row["training_seed"])
        if identity != expected_identities[index]:
            raise ValueError("public artifact cells are not the fixed ordered grid")
        dataset, model, seed = identity
        if isinstance(seed, bool) or seed not in EXPECTED_TRAINING_SEEDS:
            raise ValueError(f"{location}.training_seed is invalid")
        condition = _CONDITION_BY_MODEL[model]
        if row["condition_id"] != condition:
            raise ValueError(f"{location}.condition_id differs from its model")
        if row["logical_id"] != f"{dataset}/{condition}/seed-{seed}":
            raise ValueError(f"{location}.logical_id differs from its identity")
        _safe_identifier(row["logical_id"], location=f"{location}.logical_id")
        expected_dataset = dataset_by_id[dataset]
        if (
            row["num_samples"] != expected_dataset["eval_count"]
            or isinstance(row["num_samples"], bool)
            or row["sample_id_sha256"] != expected_dataset["eval_sample_id_sha256"]
        ):
            raise ValueError(
                f"{location} cohort differs from the public dataset binding"
            )
        _digest(row["sample_id_sha256"], location=f"{location}.sample_id_sha256")

        frozen = _exact_fields(
            row["frozen"], _FROZEN_FIELDS, location=f"{location}.frozen"
        )
        artifact_id = _safe_identifier(
            frozen["artifact_id"], location=f"{location}.frozen.artifact_id"
        )
        if artifact_id in artifact_ids:
            raise ValueError("public artifact cells reuse one frozen artifact_id")
        artifact_ids.add(artifact_id)
        for field in ("manifest_sha256", "source_sha256", "checkpoint_sha256"):
            _digest(frozen[field], location=f"{location}.frozen.{field}")
        if frozen["source_sha256"] != code["freeze_source_sha256"]:
            raise ValueError("frozen cell source differs from the code bundle")
        if (
            frozen["checkpoint_sha256"]
            != training_by_identity[identity]["checkpoint_sha256"]
        ):
            raise ValueError("frozen cell checkpoint differs from its training cell")

        assembly = _exact_fields(
            row["assembly"], _ASSEMBLY_FIELDS, location=f"{location}.assembly"
        )
        run_id = _safe_identifier(
            assembly["run_id"], location=f"{location}.assembly.run_id"
        )
        if run_id in run_ids:
            raise ValueError("public artifact cells reuse one assembly run_id")
        run_ids.add(run_id)
        for field in (
            "manifest_sha256",
            "records_sha256",
            "assembly_source_sha256",
            "common_manifest_sha256",
        ):
            _digest(assembly[field], location=f"{location}.assembly.{field}")
        if assembly["assembly_source_sha256"] != code["assembly_source_sha256"]:
            raise ValueError("assembly cell source differs from the code bundle")
        simulations = _exact_fields(
            assembly["simulation_manifest_sha256"],
            {str(count) for count in EXPECTED_M_VALUES},
            location=f"{location}.assembly.simulation_manifest_sha256",
        )
        for count in EXPECTED_M_VALUES:
            _digest(
                simulations[str(count)],
                location=(f"{location}.assembly.simulation_manifest_sha256.{count}"),
            )

        diagnostics = _exact_fields(
            row["diagnostics"],
            _DIAGNOSTIC_FIELDS,
            location=f"{location}.diagnostics",
        )
        _safe_identifier(
            diagnostics["diagnostic_id"],
            location=f"{location}.diagnostics.diagnostic_id",
        )
        for field in ("summary_sha256", "source_sha256", "descriptor_sha256"):
            _digest(diagnostics[field], location=f"{location}.diagnostics.{field}")
        if diagnostics["source_sha256"] != code["diagnostic_source_sha256"]:
            raise ValueError("diagnostic cell source differs from the code bundle")
        if diagnostics["descriptor_row_count"] != row["num_samples"] or isinstance(
            diagnostics["descriptor_row_count"], bool
        ):
            raise ValueError("diagnostic descriptor count differs from its cohort")

    phases = value["phases"]
    if not isinstance(phases, list) or len(phases) != len(PUBLIC_PHASES):
        raise ValueError("public provenance must contain the fixed eight phases")
    for index, phase in enumerate(phases):
        _exact_fields(
            phase,
            _PHASE_FIELDS,
            location=f"phases[{index}]",
        )
        if phase["phase_id"] != PUBLIC_PHASES[index]:
            raise ValueError("public provenance must contain the fixed eight phases")
        expected = EXPECTED_PHASE_COUNTS[phase["phase_id"]]
        count_fields = ("planned_jobs", "submitted_jobs", "completed_jobs")
        observed_counts = [
            _positive_int(phase[field], location=f"phases[{index}].{field}")
            for field in count_fields
        ]
        if any(observed != expected for observed in observed_counts):
            raise ValueError(
                f"phase {phase['phase_id']} count differs from the fixed design"
            )
        for field in (
            "receipt_sha256",
            "logical_job_bundle_sha256",
            "completion_evidence_sha256",
        ):
            _digest(phase[field], location=f"phases[{index}].{field}")
    if sum(row["completed_jobs"] for row in phases) != EXPECTED_TOTAL_JOBS:
        raise ValueError("public provenance must account for exactly 162 jobs")

    phase_by_id = {row["phase_id"]: row for row in phases}
    for phase, unit in PHASE_UNITS.items():
        if phase_by_id[phase]["unit"] != unit:
            raise ValueError(f"phase {phase} has an unexpected scientific unit")

    scheduler_value = value["scheduler"]
    if not isinstance(scheduler_value, dict) or isinstance(
        scheduler_value.get("summary_schema_version"), bool
    ):
        raise ValueError("public scheduler summary schema version must be an integer")
    scheduler = _validate_scheduler_public_summary(scheduler_value)
    if (
        scheduler["status"] != "complete"
        or scheduler["successful_jobs"] != EXPECTED_EXTENSION_CELLS
        or scheduler["failed_jobs"] != 0
    ):
        raise ValueError("public scheduler closure is not a complete 20-job campaign")
    training_record_rows = [
        {
            "dataset": row["dataset_id"],
            "model": row["model_id"],
            "training_seed": row["training_seed"],
            "train_record_sha256": row["train_record_sha256"],
        }
        for row in training
    ]
    training_record_set_sha = _sha256_bytes(
        _canonical_json(training_record_rows).encode("ascii")
    )
    scheduler_bindings = scheduler["bindings"]
    if scheduler_bindings["spec_lock_sha256"] != campaign["spec_lock_sha256"]:
        raise ValueError("scheduler summary binds a different seed spec lock")
    if (
        scheduler_bindings["train_submission_receipt_sha256"]
        != phase_by_id["train"]["receipt_sha256"]
    ):
        raise ValueError("scheduler summary binds a different training receipt")
    if scheduler_bindings["training_record_set_sha256"] != training_record_set_sha:
        raise ValueError("scheduler summary binds a different training-record set")

    analysis = _exact_fields(
        value["analysis"],
        _PROVENANCE_ANALYSIS_FIELDS,
        location="provenance.analysis",
    )
    if (
        isinstance(analysis["schema_version"], bool)
        or analysis["schema_version"] != ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("public provenance names an unsupported analysis schema")
    for field in _PROVENANCE_ANALYSIS_FIELDS - {
        "schema_version",
        "cell_count",
        "seed_count",
        "gate_c_fired",
    }:
        _digest(analysis[field], location=f"provenance.analysis.{field}")
    if analysis["cell_count"] != EXPECTED_ANALYSIS_CELLS or isinstance(
        analysis["cell_count"], bool
    ):
        raise ValueError("public provenance analysis cell count must be ten")
    if analysis["seed_count"] != 3 or isinstance(analysis["seed_count"], bool):
        raise ValueError("public provenance analysis seed count must be three")
    if not isinstance(analysis["gate_c_fired"], bool):
        raise ValueError("public provenance Gate C status must be Boolean")
    if analysis["analysis_source_sha256"] != code["analysis_source_sha256"]:
        raise ValueError("analysis summary differs from the bound analysis source")
    if analysis["renderer_source_sha256"] != code["renderer_source_sha256"]:
        raise ValueError("analysis summary differs from the bound renderer source")

    expected_bundles = {
        "training_bundle_sha256": _sha256_bytes(
            _canonical_json(training).encode("ascii")
        ),
        "frozen_bundle_sha256": _sha256_bytes(
            _canonical_json([row["frozen"] for row in cells]).encode("ascii")
        ),
        "assembly_bundle_sha256": _sha256_bytes(
            _canonical_json([row["assembly"] for row in cells]).encode("ascii")
        ),
        "diagnostic_bundle_sha256": _sha256_bytes(
            _canonical_json([row["diagnostics"] for row in cells]).encode("ascii")
        ),
    }
    for field, expected in expected_bundles.items():
        if analysis[field] != expected:
            raise ValueError(f"public provenance {field} is inconsistent")
    expected_evidence = {
        "train": campaign["checkpoint_lock_sha256"],
        "freeze": campaign["downstream_lock_sha256"],
        "common": analysis["assembly_bundle_sha256"],
        "score": analysis["assembly_bundle_sha256"],
        "assemble": analysis["source_analysis_sha256"],
        "diagnose": analysis["diagnostic_bundle_sha256"],
        "analyze": analysis["source_analysis_sha256"],
        "render": analysis["table_sha256"],
    }
    for phase, expected in expected_evidence.items():
        if phase_by_id[phase]["completion_evidence_sha256"] != expected:
            raise ValueError(f"phase {phase} completion evidence is inconsistent")

    _scan_public_tree(value)
    return value


def load_public_seed_release(
    public_analysis,
    public_scheduler_summary,
    public_provenance,
):
    """Strictly load and join the three path-free public seed documents."""

    paths = tuple(
        Path(path)
        for path in (
            public_analysis,
            public_scheduler_summary,
            public_provenance,
        )
    )
    expected_names = (
        PUBLIC_ANALYSIS_BASENAME,
        PUBLIC_SCHEDULER_BASENAME,
        PUBLIC_PROVENANCE_BASENAME,
    )
    if tuple(path.name for path in paths) != expected_names:
        raise ValueError(
            "public seed release requires its three fixed canonical basenames"
        )
    for path in paths:
        _reject_symlink_ancestors(path)
    if len({path.absolute().parent for path in paths}) != 1:
        raise ValueError("public seed release files must share one directory")

    analysis_payload = _read_regular(paths[0], name="public seed analysis")
    scheduler_payload = _read_regular(paths[1], name="public scheduler summary")
    provenance_payload = _read_regular(paths[2], name="public seed provenance")
    analysis = _validate_public_analysis(
        _strict_loads(analysis_payload, source=str(paths[0]))
    )
    scheduler = _validate_scheduler_public_summary(
        _strict_loads(scheduler_payload, source=str(paths[1]))
    )
    provenance = _validate_public_provenance(
        _strict_loads(provenance_payload, source=str(paths[2]))
    )
    analysis_sha256 = _sha256_bytes(analysis_payload)
    if provenance["analysis"]["portable_analysis_sha256"] != analysis_sha256:
        raise ValueError("public seed provenance does not bind the analysis bytes")
    if provenance["scheduler"] != scheduler:
        raise ValueError("public seed provenance embeds a different scheduler summary")

    analysis_provenance = analysis["provenance"]
    provenance_analysis = provenance["analysis"]
    campaign = provenance["campaign"]
    gate_c_sha256 = _sha256_bytes(_canonical_json(analysis["gate_c"]).encode("ascii"))
    if (
        analysis_provenance["source_analysis_sha256"]
        != provenance_analysis["source_analysis_sha256"]
        or analysis_provenance["analysis_source_sha256"]
        != provenance_analysis["analysis_source_sha256"]
        or analysis_provenance["downstream_lock_sha256"]
        != campaign["downstream_lock_sha256"]
        or analysis_provenance["canonical_seed0"]["analysis_sha256"]
        != campaign["canonical_seed0"]["analysis_sha256"]
        or analysis_provenance["canonical_seed0"]["campaign_lock_sha256"]
        != campaign["canonical_seed0"]["campaign_lock_sha256"]
        or analysis["gate_c"]["fired"] != provenance_analysis["gate_c_fired"]
        or gate_c_sha256 != provenance_analysis["gate_c_sha256"]
    ):
        raise ValueError("public seed analysis and provenance disagree")

    provenance_cells = {
        (cell["dataset_id"], cell["condition_id"], cell["training_seed"]): cell
        for cell in provenance["cells"]
    }
    joined = 0
    for analysis_cell in analysis["cells"]:
        for seed_text in ("1", "2"):
            identity = (
                analysis_cell["dataset"],
                analysis_cell["condition"],
                int(seed_text),
            )
            provenance_cell = provenance_cells.get(identity)
            if provenance_cell is None:
                raise ValueError(
                    "public seed analysis lacks a provenance artifact cell"
                )
            source = analysis_cell["sources"][seed_text]
            run_id = PurePosixPath(source["logical_id"]).parts[-1]
            assembly = provenance_cell["assembly"]
            if (
                analysis_cell["model"] != provenance_cell["model_id"]
                or analysis_cell["num_images_per_seed"]
                != provenance_cell["num_samples"]
                or run_id != assembly["run_id"]
                or source["manifest_sha256"] != assembly["manifest_sha256"]
                or source["records_sha256"] != assembly["records_sha256"]
            ):
                raise ValueError(
                    "public seed analysis source differs from its provenance cell"
                )
            joined += 1
    if joined != EXPECTED_EXTENSION_CELLS:
        raise ValueError("public seed release does not join exactly 20 extension cells")
    return {
        "analysis": analysis,
        "scheduler": scheduler,
        "provenance": provenance,
        "sha256": {
            "analysis": analysis_sha256,
            "scheduler": _sha256_bytes(scheduler_payload),
            "provenance": _sha256_bytes(provenance_payload),
        },
    }


def build_seed_publication(
    *,
    spec_lock,
    expected_spec_lock_sha256,
    checkpoint_lock,
    expected_checkpoint_lock_sha256,
    downstream_lock,
    expected_downstream_lock_sha256,
    canonical_analysis,
    expected_canonical_analysis_sha256,
    seed_analysis,
    expected_seed_analysis_sha256,
    table,
    expected_table_sha256,
    train_receipt,
    downstream_receipts: Mapping[str, str | Path],
    diagnostic_summaries: Sequence[str | Path],
    private_scheduler_ledger,
    public_scheduler_summary,
    public_analysis_output,
    public_provenance_output,
):
    """Validate the private closure and publish deterministic public files."""

    analysis_destination = Path(public_analysis_output)
    scheduler_destination = Path(public_scheduler_summary)
    provenance_destination = Path(public_provenance_output)
    if analysis_destination.name != PUBLIC_ANALYSIS_BASENAME:
        raise ValueError(
            f"public analysis output must be named {PUBLIC_ANALYSIS_BASENAME}"
        )
    if provenance_destination.name != PUBLIC_PROVENANCE_BASENAME:
        raise ValueError(
            f"public provenance output must be named {PUBLIC_PROVENANCE_BASENAME}"
        )
    if scheduler_destination.name != PUBLIC_SCHEDULER_BASENAME:
        raise ValueError(
            f"public scheduler summary must be named {PUBLIC_SCHEDULER_BASENAME}"
        )
    _reject_symlink_ancestors(analysis_destination)
    _reject_symlink_ancestors(scheduler_destination)
    _reject_symlink_ancestors(provenance_destination)
    if (
        len(
            {
                analysis_destination.absolute().parent,
                scheduler_destination.absolute().parent,
                provenance_destination.absolute().parent,
            }
        )
        != 1
    ):
        raise ValueError(
            "public analysis, scheduler, and provenance files must share one directory"
        )

    binding = load_spec_lock(spec_lock, expected_sha256=expected_spec_lock_sha256)
    checkpoint_binding = load_checkpoint_lock(
        binding,
        checkpoint_lock,
        expected_sha256=expected_checkpoint_lock_sha256,
        verify_files=True,
    )
    downstream_binding = load_downstream_lock(
        downstream_lock, expected_sha256=expected_downstream_lock_sha256
    )
    if (
        downstream_binding["binding"]["sha256"] != binding["sha256"]
        or downstream_binding["checkpoint_binding"]["sha256"]
        != checkpoint_binding["sha256"]
    ):
        raise ValueError("checkpoint/downstream locks bind a different seed spec")

    analysis_path, analysis_payload, analysis_sha = _regular_with_digest(
        seed_analysis, expected_seed_analysis_sha256, name="seed analysis"
    )
    analysis = _strict_loads(analysis_payload, source=str(analysis_path))
    by_key = validate_analysis_document(analysis)
    recomputed = analyze_seed_extension(
        downstream_binding,
        canonical_analysis=canonical_analysis,
        expected_canonical_analysis_sha256=expected_canonical_analysis_sha256,
    )
    if analysis != recomputed:
        raise ValueError("seed analysis differs from a strict recomputation")
    canonical_path, canonical_sha = _file_binding(
        canonical_analysis,
        expected_canonical_analysis_sha256,
        name="canonical seed-0 analysis",
    )
    if canonical_sha != analysis["provenance"]["canonical_seed0"]["sha256"]:
        raise ValueError("seed analysis binds a different canonical seed-0 analysis")

    table_path, table_payload, table_sha = _regular_with_digest(
        table, expected_table_sha256, name="rendered seed table"
    )
    expected_table = render_table(
        analysis, by_key, analysis_sha256=analysis_sha
    ).encode("utf-8")
    if table_payload != expected_table:
        raise ValueError("rendered seed table differs from the current renderer")

    plans = _expected_job_plans(
        binding,
        checkpoint_binding,
        downstream_binding,
        canonical_analysis=str(canonical_analysis),
        expected_canonical_analysis_sha256=canonical_sha,
        seed_analysis=analysis_path.as_posix(),
        seed_analysis_sha256=analysis_sha,
    )
    receipts = _validate_all_receipts(
        binding,
        plans,
        train_receipt=train_receipt,
        downstream_receipts=downstream_receipts,
    )
    record_set_sha = _training_record_set_sha256(checkpoint_binding)
    scheduler = _validate_scheduler_closure(
        private_ledger=private_scheduler_ledger,
        public_summary=public_scheduler_summary,
        spec_lock_sha256=binding["sha256"],
        train_receipt_sha256=receipts["train"]["receipt_sha256"],
        training_record_set_sha256=record_set_sha,
        train_job_bindings=receipts["train"]["job_bindings"],
    )

    cells, stage_code = _cell_closure(
        downstream_binding, analysis, diagnostic_summaries
    )
    training = _training(checkpoint_binding)
    training_bundle_sha = _sha256_bytes(_canonical_json(training).encode("ascii"))
    frozen_bundle_sha = _sha256_bytes(
        _canonical_json([row["frozen"] for row in cells]).encode("ascii")
    )
    assembly_bundle_sha = _sha256_bytes(
        _canonical_json([row["assembly"] for row in cells]).encode("ascii")
    )
    diagnostic_bundle_sha = _sha256_bytes(
        _canonical_json([row["diagnostics"] for row in cells]).encode("ascii")
    )

    portable = _portable_analysis(
        analysis,
        source_sha256=analysis_sha,
        downstream_sha256=downstream_binding["sha256"],
        canonical=analysis["provenance"]["canonical_seed0"],
    )
    _validate_public_analysis(portable)
    portable_payload = _json_bytes(portable)
    portable_sha = _sha256_bytes(portable_payload)

    spec = binding["spec"]
    estimator = binding["lock"]["estimator_spec"]
    locked_sources = [
        {
            "logical_name": _safe_identifier(
                row["path"], location="locked source logical name"
            ),
            "sha256": _digest(row["sha256"], location="locked source SHA-256"),
        }
        for row in binding["lock"]["source_files"]
    ]
    code = {
        "locked_source_files": locked_sources,
        **stage_code,
        "analysis_source_sha256": analysis["provenance"]["analysis_source_sha256"],
        "renderer_source_sha256": _renderer_source_sha256(),
        "exporter_source_sha256": _sha256(Path(__file__).resolve()),
    }
    campaign_rows = []
    for campaign in downstream_binding["campaigns"]:
        campaign_rows.append(
            {
                "training_seed": campaign["training_seed"],
                "campaign_id": campaign["campaign"]["campaign_id"],
                "config_sha256": campaign["config"].sha256,
                "campaign_lock_sha256": campaign["campaign_lock_sha256"],
            }
        )
    campaign = {
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "spec_sha256": binding["lock"]["spec"]["sha256"],
        "spec_lock_sha256": binding["sha256"],
        "checkpoint_lock_sha256": checkpoint_binding["sha256"],
        "downstream_lock_sha256": downstream_binding["sha256"],
        "canonical_seed0": {
            "campaign_id": binding["lock"]["canonical_campaign_lock"]["campaign_id"],
            "campaign_lock_sha256": binding["lock"]["canonical_campaign_lock"][
                "sha256"
            ],
            "analysis_sha256": canonical_sha,
        },
        "seed_campaigns": campaign_rows,
        "protocol": {
            "reference_training_seed": 0,
            "training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "checkpoint_rule": spec["protocol"]["checkpoint_rule"],
            "gamma": spec["protocol"]["gamma"],
            "m_values": list(EXPECTED_M_VALUES),
            "quadrature_rule": spec["protocol"]["quadrature_rule"],
            "quadrature_seed": EXPECTED_QUADRATURE_SEED,
        },
        "estimator": {
            "estimator_id": "midpoint-v1",
            "target_measure": "uniform-threshold",
            "spec_sha256": estimator["sha256"],
        },
        "grid": {
            "dataset_count": len(EXPECTED_DATASETS),
            "model_count": len(EXPECTED_MODELS),
            "extension_cell_count": EXPECTED_EXTENSION_CELLS,
            "three_seed_analysis_cell_count": EXPECTED_ANALYSIS_CELLS,
        },
    }
    evidence = {
        "train": checkpoint_binding["sha256"],
        "freeze": downstream_binding["sha256"],
        "common": assembly_bundle_sha,
        "score": assembly_bundle_sha,
        "assemble": analysis_sha,
        "diagnose": diagnostic_bundle_sha,
        "analyze": analysis_sha,
        "render": table_sha,
    }
    gate_digest = _sha256_bytes(_canonical_json(analysis["gate_c"]).encode("ascii"))
    provenance = {
        "schema_version": PUBLIC_PROVENANCE_SCHEMA_VERSION,
        "artifact_type": PUBLIC_PROVENANCE_ARTIFACT_TYPE,
        "campaign": campaign,
        "datasets": _datasets(spec),
        "base_models": _base_models(binding["lock"]),
        "code": code,
        "training": training,
        "cells": cells,
        "phases": _phase_records(receipts, evidence=evidence),
        "scheduler": scheduler,
        "analysis": {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "source_analysis_sha256": analysis_sha,
            "portable_analysis_sha256": portable_sha,
            "analysis_source_sha256": analysis["provenance"]["analysis_source_sha256"],
            "renderer_source_sha256": code["renderer_source_sha256"],
            "table_sha256": table_sha,
            "cell_count": len(analysis["cells"]),
            "seed_count": 3,
            "gate_c_fired": analysis["gate_c"]["fired"],
            "gate_c_sha256": gate_digest,
            "training_bundle_sha256": training_bundle_sha,
            "frozen_bundle_sha256": frozen_bundle_sha,
            "assembly_bundle_sha256": assembly_bundle_sha,
            "diagnostic_bundle_sha256": diagnostic_bundle_sha,
        },
    }
    _validate_public_provenance(provenance)
    provenance_payload = _json_bytes(provenance)
    status = publish_payload_then_guard(
        analysis_path=public_analysis_output,
        analysis_payload=portable_payload,
        guard_path=public_provenance_output,
        guard_payload=provenance_payload,
    )
    return {
        "public_analysis": Path(public_analysis_output),
        "public_provenance": Path(public_provenance_output),
        "public_analysis_sha256": portable_sha,
        "public_provenance_sha256": _sha256_bytes(provenance_payload),
        "status": status,
        "document": provenance,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-lock", required=True)
    parser.add_argument("--expected-spec-lock-sha256", required=True)
    parser.add_argument("--checkpoint-lock", required=True)
    parser.add_argument("--expected-checkpoint-lock-sha256", required=True)
    parser.add_argument("--downstream-lock", required=True)
    parser.add_argument("--expected-downstream-lock-sha256", required=True)
    parser.add_argument("--canonical-analysis", required=True)
    parser.add_argument("--expected-canonical-analysis-sha256", required=True)
    parser.add_argument("--seed-analysis", required=True)
    parser.add_argument("--expected-seed-analysis-sha256", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--expected-table-sha256", required=True)
    parser.add_argument("--train-receipt", required=True)
    parser.add_argument(
        "--phase-receipt",
        action="append",
        nargs=2,
        required=True,
        metavar=("PHASE", "JSONL"),
    )
    parser.add_argument(
        "--diagnostic-summary", action="append", required=True, metavar="JSON"
    )
    parser.add_argument("--private-scheduler-ledger", required=True)
    parser.add_argument("--public-scheduler-summary", required=True)
    parser.add_argument("--public-analysis-output", required=True)
    parser.add_argument("--public-provenance-output", required=True)
    return parser.parse_args(argv)


def _receipt_mapping(values):
    result = {}
    for phase, path in values:
        if phase in result:
            raise ValueError(f"duplicate downstream receipt phase {phase!r}")
        result[phase] = path
    return result


def main(argv=None):
    args = parse_args(argv)
    result = build_seed_publication(
        spec_lock=args.spec_lock,
        expected_spec_lock_sha256=args.expected_spec_lock_sha256,
        checkpoint_lock=args.checkpoint_lock,
        expected_checkpoint_lock_sha256=args.expected_checkpoint_lock_sha256,
        downstream_lock=args.downstream_lock,
        expected_downstream_lock_sha256=args.expected_downstream_lock_sha256,
        canonical_analysis=args.canonical_analysis,
        expected_canonical_analysis_sha256=args.expected_canonical_analysis_sha256,
        seed_analysis=args.seed_analysis,
        expected_seed_analysis_sha256=args.expected_seed_analysis_sha256,
        table=args.table,
        expected_table_sha256=args.expected_table_sha256,
        train_receipt=args.train_receipt,
        downstream_receipts=_receipt_mapping(args.phase_receipt),
        diagnostic_summaries=args.diagnostic_summary,
        private_scheduler_ledger=args.private_scheduler_ledger,
        public_scheduler_summary=args.public_scheduler_summary,
        public_analysis_output=args.public_analysis_output,
        public_provenance_output=args.public_provenance_output,
    )
    print(f"saved {result['public_analysis']}")
    print(f"public_analysis_sha256={result['public_analysis_sha256']}")
    print(f"saved {result['public_provenance']}")
    print(f"public_provenance_sha256={result['public_provenance_sha256']}")
    return result


if __name__ == "__main__":
    main()
