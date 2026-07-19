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

All submission phases are dry-run by default; pass ``--submit`` to call
``sbatch``.  Lock files are written only when ``--write-lock`` is provided and
are never overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CONFIG_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
EXPECTED_PROTOCOL = {
    "gamma_values": [0.5],
    "m_values": [2, 8, 32],
    "quadrature_rule": "midpoint-v1",
    "seeds": [0],
}
GPU_ACCOUNT = "ssafo"
DEFAULT_CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
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
OPTIONAL_CONFIG_FIELDS = frozenset({"cpu_partitions"})
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
    parser.add_argument(
        "--submit",
        action="store_true",
        help="call sbatch; without this flag submission phases are dry-runs",
    )
    parser.add_argument(
        "--receipt",
        help=(
            "phase-specific append-only receipt; required with --submit and "
            "used to prevent blind duplicate resubmission"
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
    if data["config_schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config_schema_version must equal {CONFIG_SCHEMA_VERSION}")
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
    return Config(path=path, sha256=_sha256_bytes(raw), data=data)


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
    partitions = config.data["gpu_partitions"]
    partition_request = ",".join(partitions)
    for condition in config.data["conditions"]:
        checkpoint = condition["checkpoint"] or "-"
        key = (condition["dataset"], condition["condition"], partition_request)
        job_name = f"selseg-freeze-{condition['dataset']}-{condition['condition']}"
        command = [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name[:128],
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
        artifacts.append(
            {
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
        )
    if set(by_key) != {
        (condition["dataset"], condition["condition"])
        for condition in config.data["conditions"]
    }:
        raise ValueError("artifact inputs contain an undeclared condition")
    estimator_path, estimator_sha, estimator = _load_estimator(config)
    return {
        "lock_schema_version": LOCK_SCHEMA_VERSION,
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
    if set(lock) != REQUIRED_LOCK_FIELDS:
        raise ValueError(
            f"campaign lock must contain exactly {sorted(REQUIRED_LOCK_FIELDS)}"
        )
    _assert_finite(lock, location=str(path))
    if lock.get("lock_schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError(f"{path}.lock_schema_version must equal {LOCK_SCHEMA_VERSION}")
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
        if (
            not isinstance(artifact, dict)
            or set(artifact) != REQUIRED_LOCK_ARTIFACT_FIELDS
        ):
            raise ValueError(
                f"{path}.artifacts[{index}] must contain exactly "
                f"{sorted(REQUIRED_LOCK_ARTIFACT_FIELDS)}"
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
    cpu_partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
    for artifact in lock["artifacts"]:
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        for gamma in lock["protocol"]["gamma_values"]:
            for count in lock["protocol"]["m_values"]:
                for seed in lock["protocol"]["seeds"]:
                    partition = cpu_partitions[len(jobs) % len(cpu_partitions)]
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
                        job_name[:128],
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
    cpu_partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = cpu_partitions[artifact_index % len(cpu_partitions)]
        key = (artifact["dataset"], artifact["condition"], partition, gamma)
        gamma_tag = str(gamma).replace(".", "p")
        job_name = (
            f"selseg-common-{artifact['dataset']}-{artifact['condition']}-g{gamma_tag}"
        )
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            job_name[:128],
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
    common_root = _project_path(
        config.path, lock["paths"]["common_output_root"]
    )
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


def plan_assemble_jobs(config, campaign_lock):
    """Plan one strict assembly job per locked artifact, without discovery."""

    lock_path, lock_sha, lock = load_campaign_lock(campaign_lock, config=config)
    from scripts.assemble_binary_simulations import (
        load_campaign_lock as load_assembly_lock,
        prepare_assembly,
    )

    assembly_lock = load_assembly_lock(lock_path)
    output_root = lock["paths"]["assembly_output_root"]
    _nonempty_string(output_root, location=f"{lock_path}.paths.assembly_output_root")
    cpu_partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
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
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"assembled output already exists: {target}")

        partition = cpu_partitions[artifact_index % len(cpu_partitions)]
        key = (artifact["dataset"], artifact["condition"], partition)
        job_name = f"selseg-assemble-{artifact['dataset']}-{artifact['condition']}"
        command = [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name[:128],
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
    cpu_partitions = config.data.get("cpu_partitions", list(DEFAULT_CPU_PARTITIONS))
    jobs = []
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = cpu_partitions[artifact_index % len(cpu_partitions)]
        key = (artifact["dataset"], artifact["condition"], partition)
        job_name = f"selseg-diagnose-{artifact['dataset']}-{artifact['condition']}"
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            job_name[:128],
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


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.phase == "freeze":
        if args.artifact_manifest or args.campaign_lock or args.write_lock:
            raise ValueError("freeze phase does not accept lock/artifact inputs")
        return execute_plan(
            plan_freeze_jobs(config),
            submit=args.submit,
            receipt_path=args.receipt,
        )
    if args.phase == "lock":
        if args.submit:
            raise ValueError("lock phase never invokes sbatch; omit --submit")
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
        return execute_plan(
            plan_common_jobs(config, args.campaign_lock),
            submit=args.submit,
            receipt_path=args.receipt,
        )
    if args.phase == "score":
        jobs = plan_score_jobs(config, args.campaign_lock)
    elif args.phase == "assemble":
        jobs = plan_assemble_jobs(config, args.campaign_lock)
    else:
        jobs = plan_diagnose_jobs(
            config,
            args.campaign_lock,
            output_root=args.diagnostic_output_root,
        )
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
