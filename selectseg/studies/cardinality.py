"""Exact cardinality diagnostics for the shared-threshold working posterior.

For ``T ~ Uniform(0, 1)`` and ``Y_T = {i: p_i >= T}``, let
``K(T) = |Y_T|``.  This module evaluates the exact discrete CDF interval at
the one observed reference-mask cardinality ``k``:

``F_-(k) = Q_p(K < k)`` and ``F(k) = Q_p(K <= k)``.

A predeclared diagnostic seed and the sample identifier determine a stable
SHA-256 pseudo-random variate used inside ``[F_-(k), F(k)]``.  Pooling these
randomized PIT values can falsify aggregate cardinality implications of the
working posterior.  With one annotation per image it cannot identify a
pointwise conditional cardinality law, establish posterior calibration, or
validate the shared-threshold coupling.  Labels are diagnostic outcomes only
and must never fit, tune, or select a confidence score.

The workflow is independent of ``selectseg.studies.diagnostics`` and publishes
new content-addressed, no-overwrite artifacts.  It does not modify the locked
v1 diagnostic schema or any existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import string
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.submit.main import load_campaign_lock
from selectseg.artifacts import (
    FrozenBinarySample,
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
    sample_id_sha256,
    sha256_file,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_cardinality_diagnostics"
RECORDS_NAME = "records.jsonl"
MANIFEST_NAME = "manifest.json"
EXPECTED_AUXILIARY_ID = "binary-cardinality-diagnostics-v1"
EXPECTED_DIAGNOSTIC_SEED = 20_260_720
EXPECTED_CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
RANDOMIZER = "sha256-top52-midpoint-v1"

PROTOCOL = {
    "working_posterior": "shared-threshold",
    "threshold_distribution": "Uniform(0,1)",
    "mask_rule": "Y_T={i:foreground_probability_i>=T}",
    "cardinality": "K(T)=|Y_T|",
    "cdf_lower": "F_-(k)=Q_p(K<k)",
    "cdf_upper": "F(k)=Q_p(K<=k)",
    "randomized_pit": "F_-(k)+V[F(k)-F_-(k)]",
    "diagnostic_seed": EXPECTED_DIAGNOSTIC_SEED,
    "randomizer": RANDOMIZER,
    "empty_mask_identity": "Q_p(K=0)=1-max_i foreground_probability_i",
}

RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "artifact_id",
        "sample_index",
        "sample_id",
        "height",
        "width",
        "pixels",
        "truth_cardinality",
        "truth_foreground_fraction",
        "working_posterior_mean_cardinality",
        "working_posterior_mean_foreground_fraction",
        "signed_cardinality_error",
        "absolute_cardinality_error",
        "signed_foreground_fraction_error",
        "absolute_foreground_fraction_error",
        "cardinality_cdf_lower",
        "cardinality_cdf_upper",
        "observed_cardinality_probability",
        "pit_randomizer",
        "randomized_cardinality_pit",
        "working_posterior_empty_probability",
        "observed_empty_mask",
    }
)

_LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "auxiliary_id",
        "spec",
        "canonical_campaign_lock",
        "protocol",
        "cpu_partitions",
        "output_root",
        "artifacts",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "spec_schema_version",
        "auxiliary_id",
        "canonical_campaign_lock",
        "protocol",
        "cpu_partitions",
        "output_root",
    }
)
_FILE_BINDING_FIELDS = frozenset({"path", "sha256"})
_CAMPAIGN_BINDING_FIELDS = frozenset({"path", "sha256", "campaign_id"})
_ARTIFACT_BINDING_FIELDS = frozenset({"manifest_path", "manifest_sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "created_utc",
        "dataset",
        "condition",
        "model",
        "split",
        "num_images",
        "num_rows",
        "sample_id_sha256",
        "scope",
        "protocol",
        "record_fields",
        "environment",
        "provenance",
        "records_sha256",
        "command",
    }
)

SCOPE = {
    "role": (
        "single-label aggregate falsification and label-proxy diagnostic for "
        "cardinality implications of the shared-threshold working posterior"
    ),
    "not_posterior_calibration": (
        "one annotation per image cannot identify the pointwise conditional "
        "cardinality law, establish posterior calibration, or validate Q_p"
    ),
    "label_use_policy": (
        "reference masks are diagnostic outcomes only and must not fit, tune, "
        "or select a confidence score"
    ),
}


@dataclass(frozen=True)
class CardinalityAuxiliaryBinding:
    """Validated immutable bindings for the auxiliary diagnostic."""

    path: Path
    sha256: str
    data: Mapping[str, Any]
    spec_path: Path
    campaign_path: Path
    campaign_sha256: str
    campaign: Mapping[str, Any]
    artifacts: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class CardinalityDiagnostic:
    """One strictly validated per-condition diagnostic artifact."""

    records_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    manifest_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return str(self.manifest["dataset"]), str(self.manifest["condition"])


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-lock", required=True)
    parser.add_argument("--expected-auxiliary-lock-sha256", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path, *, name: str):
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"{name} must be a regular, non-symlink file: {path}"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _digest(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in value):
        raise ValueError(f"{location} must contain exactly 64 hexadecimal characters")
    return normalized


def _project_path(binding_path: Path, value: Any, *, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty path string")
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    repository_candidate = (binding_path.resolve().parents[2] / raw).resolve()
    if cwd_candidate.exists() or not repository_candidate.exists():
        return cwd_candidate
    return repository_candidate


def _validate_file_binding(value: Any, *, location: str) -> None:
    if not isinstance(value, dict) or set(value) != _FILE_BINDING_FIELDS:
        raise ValueError(f"{location} must contain exactly path and sha256")
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError(f"{location}.path must be non-empty")
    _digest(value["sha256"], location=f"{location}.sha256")


def _validate_campaign_binding(value: Any, *, location: str) -> None:
    if not isinstance(value, dict) or set(value) != _CAMPAIGN_BINDING_FIELDS:
        raise ValueError(
            f"{location} must contain exactly path, sha256, and campaign_id"
        )
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError(f"{location}.path must be non-empty")
    _digest(value["sha256"], location=f"{location}.sha256")
    if not isinstance(value["campaign_id"], str) or not value["campaign_id"]:
        raise ValueError(f"{location}.campaign_id must be non-empty")


def load_auxiliary_lock(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> CardinalityAuxiliaryBinding:
    """Load the auxiliary lock and verify its complete canonical projection."""

    lock_path = Path(path).resolve()
    lock = _load_strict_json(lock_path, name="cardinality auxiliary lock")
    if set(lock) != _LOCK_FIELDS or lock.get("lock_schema_version") != 1:
        raise ValueError("cardinality auxiliary lock has an unexpected schema")
    lock_sha256 = sha256_file(lock_path)
    if expected_sha256 is not None and lock_sha256 != _digest(
        expected_sha256, location="expected auxiliary lock SHA-256"
    ):
        raise ValueError("cardinality auxiliary lock SHA-256 mismatch")
    if lock.get("auxiliary_id") != EXPECTED_AUXILIARY_ID:
        raise ValueError("cardinality auxiliary lock has an unexpected auxiliary_id")
    _validate_file_binding(lock.get("spec"), location="auxiliary lock spec")
    _validate_campaign_binding(
        lock.get("canonical_campaign_lock"),
        location="auxiliary lock canonical_campaign_lock",
    )
    if lock.get("protocol") != PROTOCOL:
        raise ValueError("cardinality auxiliary lock protocol is not frozen v1")
    if tuple(lock.get("cpu_partitions", ())) != EXPECTED_CPU_PARTITIONS:
        raise ValueError("cardinality auxiliary lock has unexpected CPU partitions")
    if not isinstance(lock.get("output_root"), str) or not lock["output_root"]:
        raise ValueError("cardinality auxiliary output_root must be non-empty")

    spec_path = _project_path(
        lock_path, lock["spec"]["path"], location="auxiliary spec path"
    )
    if sha256_file(spec_path) != _digest(
        lock["spec"]["sha256"], location="auxiliary spec SHA-256"
    ):
        raise ValueError("cardinality auxiliary spec is missing or changed")
    spec = _load_strict_json(spec_path, name="cardinality auxiliary spec")
    if set(spec) != _SPEC_FIELDS or spec.get("spec_schema_version") != 1:
        raise ValueError("cardinality auxiliary spec has an unexpected schema")
    if spec != {
        "spec_schema_version": 1,
        "auxiliary_id": lock["auxiliary_id"],
        "canonical_campaign_lock": lock["canonical_campaign_lock"],
        "protocol": lock["protocol"],
        "cpu_partitions": lock["cpu_partitions"],
        "output_root": lock["output_root"],
    }:
        raise ValueError("cardinality auxiliary spec and lock disagree")

    campaign_binding = lock["canonical_campaign_lock"]
    campaign_path = _project_path(
        lock_path,
        campaign_binding["path"],
        location="canonical campaign lock path",
    )
    campaign_path, campaign_sha256, campaign = load_campaign_lock(campaign_path)
    if campaign_sha256 != _digest(
        campaign_binding["sha256"], location="canonical campaign SHA-256"
    ):
        raise ValueError("canonical campaign lock bytes differ from auxiliary lock")
    if campaign["campaign_id"] != campaign_binding["campaign_id"]:
        raise ValueError("canonical campaign_id differs from auxiliary lock")

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("cardinality auxiliary lock artifacts must be non-empty")
    projection = []
    for index, entry in enumerate(artifacts):
        if not isinstance(entry, dict) or set(entry) != _ARTIFACT_BINDING_FIELDS:
            raise ValueError(f"auxiliary artifacts[{index}] has an invalid schema")
        resolved = _project_path(
            lock_path,
            entry.get("manifest_path"),
            location=f"auxiliary artifacts[{index}].manifest_path",
        )
        projection.append(
            (
                resolved,
                _digest(
                    entry.get("manifest_sha256"),
                    location=f"auxiliary artifacts[{index}].manifest_sha256",
                ),
            )
        )
    if len({path for path, _ in projection}) != len(projection):
        raise ValueError("cardinality auxiliary lock has duplicate artifacts")
    canonical_projection = [
        (
            _project_path(
                campaign_path,
                entry["manifest_path"],
                location=f"canonical artifacts[{index}].manifest_path",
            ),
            entry["manifest_sha256"],
        )
        for index, entry in enumerate(campaign["artifacts"])
    ]
    if projection != canonical_projection:
        raise ValueError("auxiliary artifacts differ from the canonical campaign")
    return CardinalityAuxiliaryBinding(
        lock_path,
        lock_sha256,
        lock,
        spec_path,
        campaign_path,
        campaign_sha256,
        campaign,
        tuple(projection),
    )


def cardinality_cdf_bounds(
    foreground_probability: np.ndarray, observed_cardinality: int
) -> tuple[float, float]:
    """Return exact ``(Q_p(K<k), Q_p(K<=k))`` using two order statistics.

    Repeated probability values are retained.  Consequently an unattainable
    cardinality has equal lower and upper CDF bounds, exactly as required.
    The operation uses ``numpy.partition`` and is linear-time on average.
    """

    probability = np.asarray(foreground_probability)
    if probability.ndim == 0 or probability.size == 0:
        raise ValueError("foreground_probability must contain at least one value")
    if not np.issubdtype(probability.dtype, np.floating):
        raise TypeError("foreground_probability must have a floating dtype")
    flat = probability.reshape(-1)
    if not np.all(np.isfinite(flat)) or np.any((flat < 0) | (flat > 1)):
        raise ValueError("foreground_probability must be finite and lie in [0,1]")
    if (
        isinstance(observed_cardinality, (bool, np.bool_))
        or not isinstance(observed_cardinality, (int, np.integer))
    ):
        raise TypeError("observed_cardinality must be an integer")
    k = int(observed_cardinality)
    pixels = int(flat.size)
    if not 0 <= k <= pixels:
        raise ValueError("observed_cardinality must lie in [0, num_pixels]")

    if k == 0:
        lower = 0.0
        upper = 1.0 - float(np.max(flat))
    elif k == pixels:
        lower = 1.0 - float(np.min(flat))
        upper = 1.0
    else:
        # Ascending positions N-k-1 and N-k are q_{k+1} and q_k when the
        # probabilities are denoted q_1 >= ... >= q_N.
        selected = np.partition(flat, (pixels - k - 1, pixels - k))
        q_k_plus_one = float(selected[pixels - k - 1])
        q_k = float(selected[pixels - k])
        lower = 1.0 - q_k
        upper = 1.0 - q_k_plus_one
    lower = min(1.0, max(0.0, lower))
    upper = min(1.0, max(0.0, upper))
    if lower > upper:
        raise RuntimeError("computed cardinality CDF bounds are inconsistent")
    return lower, upper


def deterministic_pit_randomizer(seed: int, sample_id: str) -> float:
    """Map the frozen seed and sample ID to a reproducible open-unit variate."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("diagnostic seed must be an integer in [0, 2^63)")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    payload = (
        b"selectseg-cardinality-pit\0"
        + str(seed).encode("ascii")
        + b"\0"
        + sample_id.encode("utf-8")
    )
    word = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    cell = word >> 12
    # Midpoints of 2^52 equal cells are exactly representable here and avoid
    # both endpoints, independent of NumPy/Python RNG implementations.
    return (cell + 0.5) / float(2**52)


def cardinality_record(
    sample: FrozenBinarySample,
    *,
    run_id: str,
    artifact_id: str,
    diagnostic_seed: int = EXPECTED_DIAGNOSTIC_SEED,
) -> dict[str, Any]:
    """Build one exact cardinality/PIT label-proxy diagnostic record."""

    probability = sample.foreground_probability
    truth = sample.truth
    if probability.shape != truth.shape or probability.ndim != 2:
        raise ValueError("probability and truth must be matching two-dimensional masks")
    pixels = int(probability.size)
    truth_cardinality = int(np.count_nonzero(truth))
    mean_cardinality = float(np.sum(probability, dtype=np.float64))
    mean_fraction = mean_cardinality / pixels
    truth_fraction = truth_cardinality / pixels
    signed_cardinality_error = mean_cardinality - truth_cardinality
    signed_fraction_error = mean_fraction - truth_fraction
    cdf_lower, cdf_upper = cardinality_cdf_bounds(
        probability, truth_cardinality
    )
    randomizer = deterministic_pit_randomizer(diagnostic_seed, sample.sample_id)
    randomized_pit = cdf_lower + randomizer * (cdf_upper - cdf_lower)
    empty_probability = 1.0 - float(np.max(probability))
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "sample_index": sample.index,
        "sample_id": sample.sample_id,
        "height": int(probability.shape[0]),
        "width": int(probability.shape[1]),
        "pixels": pixels,
        "truth_cardinality": truth_cardinality,
        "truth_foreground_fraction": truth_fraction,
        "working_posterior_mean_cardinality": mean_cardinality,
        "working_posterior_mean_foreground_fraction": mean_fraction,
        "signed_cardinality_error": signed_cardinality_error,
        "absolute_cardinality_error": abs(signed_cardinality_error),
        "signed_foreground_fraction_error": signed_fraction_error,
        "absolute_foreground_fraction_error": abs(signed_fraction_error),
        "cardinality_cdf_lower": cdf_lower,
        "cardinality_cdf_upper": cdf_upper,
        "observed_cardinality_probability": cdf_upper - cdf_lower,
        "pit_randomizer": randomizer,
        "randomized_cardinality_pit": randomized_pit,
        "working_posterior_empty_probability": empty_probability,
        "observed_empty_mask": truth_cardinality == 0,
    }
    _validate_record(record, location=f"sample {sample.index}")
    return record


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "selectseg/studies/cardinality.py",
        "selectseg/artifacts.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        source = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_id(lock_sha256: str, artifact_sha256: str, source_sha256: str) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "auxiliary_lock_sha256": lock_sha256,
        "artifact_manifest_sha256": artifact_sha256,
        "source_sha256": source_sha256,
        "diagnostic_seed": EXPECTED_DIAGNOSTIC_SEED,
        "randomizer": RANDOMIZER,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_cardinality_diagnostics(
    args, *, created_utc: str | None = None
) -> tuple[Path, Path]:
    """Run and atomically publish one lock-listed condition diagnostic."""

    expected_lock_sha256 = _digest(
        args.expected_auxiliary_lock_sha256,
        location="--expected-auxiliary-lock-sha256",
    )
    expected_artifact_sha256 = _digest(
        args.expected_artifact_manifest_sha256,
        location="--expected-artifact-manifest-sha256",
    )
    binding = load_auxiliary_lock(
        args.auxiliary_lock, expected_sha256=expected_lock_sha256
    )
    artifact_path = Path(args.artifact_manifest).resolve()
    matches = [item for item in binding.artifacts if item[0] == artifact_path]
    if len(matches) != 1:
        raise ValueError("artifact manifest must occur exactly once in auxiliary lock")
    if matches[0][1] != expected_artifact_sha256:
        raise ValueError("artifact SHA-256 differs from auxiliary lock")
    artifact = load_binary_artifact(artifact_path, validate_payloads=False)
    if artifact.manifest_sha256 != expected_artifact_sha256:
        raise ValueError("artifact manifest SHA-256 mismatch")

    source_sha256 = _source_fingerprint()
    run_id = _run_id(binding.sha256, artifact.manifest_sha256, source_sha256)
    frozen = artifact.manifest
    destination = (
        Path(binding.data["output_root"])
        / str(frozen["dataset"])
        / str(frozen["condition"])
        / str(frozen["artifact_id"])
        / run_id
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cardinality diagnostics already exist: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=destination.parent)
    )
    records_path = staging / RECORDS_NAME
    manifest_path = staging / MANIFEST_NAME
    sample_ids = []
    row_count = 0
    try:
        with records_path.open("x", encoding="utf-8") as output:
            for expected_index, sample in enumerate(artifact.iter_samples()):
                if sample.index != expected_index:
                    raise ValueError("artifact samples are not contiguous and ordered")
                sample_ids.append(sample.sample_id)
                record = cardinality_record(
                    sample,
                    run_id=run_id,
                    artifact_id=str(frozen["artifact_id"]),
                )
                output.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                row_count += 1
            output.flush()
            os.fsync(output.fileno())
        if row_count != frozen["num_samples"]:
            raise RuntimeError("diagnostic row count differs from frozen artifact")
        if sample_id_sha256(sample_ids) != frozen["sample_id_sha256"]:
            raise RuntimeError("diagnostic sample order differs from frozen artifact")
        timestamp = created_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "run_id": run_id,
            "created_utc": timestamp,
            "dataset": frozen["dataset"],
            "condition": frozen["condition"],
            "model": frozen["model"],
            "split": frozen["split"],
            "num_images": frozen["num_samples"],
            "num_rows": row_count,
            "sample_id_sha256": frozen["sample_id_sha256"],
            "scope": SCOPE,
            "protocol": PROTOCOL,
            "record_fields": sorted(RECORD_FIELDS),
            "environment": {
                "packages": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
                "device": "cpu",
            },
            "provenance": {
                "auxiliary_id": binding.data["auxiliary_id"],
                "auxiliary_lock_path": _portable_path(binding.path),
                "auxiliary_lock_sha256": binding.sha256,
                "auxiliary_spec_path": _portable_path(binding.spec_path),
                "auxiliary_spec_sha256": binding.data["spec"]["sha256"],
                "canonical_campaign_id": binding.campaign["campaign_id"],
                "canonical_campaign_lock_path": _portable_path(
                    binding.campaign_path
                ),
                "canonical_campaign_lock_sha256": binding.campaign_sha256,
                "artifact_id": frozen["artifact_id"],
                "artifact_manifest_path": _portable_path(artifact.manifest_path),
                "artifact_manifest_sha256": artifact.manifest_sha256,
                "source_sha256": source_sha256,
                "artifact_passes": 1,
            },
            "records_sha256": sha256_file(records_path),
            "command": [
                "python",
                "-m",
                "selectseg.studies.cardinality",
                *args.command_arguments,
            ],
        }
        _validate_manifest(manifest, location="constructed manifest")
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        fsync_directory(staging)
        publish_directory_no_replace(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / RECORDS_NAME, destination / MANIFEST_NAME


def _finite(value: Any, *, location: str, minimum=0.0, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{location} must be finite and >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def _integer(value: Any, *, location: str, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _validate_record(record: Any, *, location: str) -> None:
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise ValueError(f"{location} has an invalid cardinality record schema")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{location}.schema_version is unsupported")
    for field in ("run_id", "artifact_id", "sample_id"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{location}.{field} must be non-empty")
    height = _integer(record["height"], location=f"{location}.height", minimum=1)
    width = _integer(record["width"], location=f"{location}.width", minimum=1)
    pixels = _integer(record["pixels"], location=f"{location}.pixels", minimum=1)
    _integer(record["sample_index"], location=f"{location}.sample_index")
    if pixels != height * width:
        raise ValueError(f"{location}.pixels is inconsistent")
    truth = _integer(
        record["truth_cardinality"], location=f"{location}.truth_cardinality"
    )
    if truth > pixels:
        raise ValueError(f"{location}.truth_cardinality exceeds pixels")
    unit_fields = (
        "truth_foreground_fraction",
        "working_posterior_mean_foreground_fraction",
        "absolute_foreground_fraction_error",
        "cardinality_cdf_lower",
        "cardinality_cdf_upper",
        "observed_cardinality_probability",
        "pit_randomizer",
        "randomized_cardinality_pit",
        "working_posterior_empty_probability",
    )
    for field in unit_fields:
        _finite(record[field], location=f"{location}.{field}", maximum=1.0)
    for field in (
        "working_posterior_mean_cardinality",
        "absolute_cardinality_error",
    ):
        _finite(record[field], location=f"{location}.{field}", maximum=pixels)
    for field, limit in (
        ("signed_cardinality_error", pixels),
        ("signed_foreground_fraction_error", 1.0),
    ):
        _finite(
            record[field],
            location=f"{location}.{field}",
            minimum=-limit,
            maximum=limit,
        )
    lower = float(record["cardinality_cdf_lower"])
    upper = float(record["cardinality_cdf_upper"])
    pit = float(record["randomized_cardinality_pit"])
    if not lower <= pit <= upper:
        raise ValueError(f"{location}.randomized_cardinality_pit leaves CDF interval")
    if not math.isclose(
        record["observed_cardinality_probability"],
        upper - lower,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} has inconsistent CDF point mass")
    expected_pit = lower + record["pit_randomizer"] * (upper - lower)
    if not math.isclose(pit, expected_pit, abs_tol=1e-15):
        raise ValueError(f"{location} has inconsistent randomized PIT")
    expected_truth_fraction = truth / pixels
    if not math.isclose(
        record["truth_foreground_fraction"], expected_truth_fraction, abs_tol=1e-12
    ):
        raise ValueError(f"{location} has inconsistent truth fraction")
    expected_signed = (
        record["working_posterior_mean_cardinality"] - truth
    )
    if not math.isclose(
        record["working_posterior_mean_foreground_fraction"],
        record["working_posterior_mean_cardinality"] / pixels,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} has inconsistent posterior mean fraction")
    if not math.isclose(
        record["signed_cardinality_error"], expected_signed, abs_tol=1e-9
    ):
        raise ValueError(f"{location} has inconsistent cardinality error")
    if not math.isclose(
        record["signed_foreground_fraction_error"],
        expected_signed / pixels,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} has inconsistent fraction error")
    if not math.isclose(
        record["absolute_cardinality_error"], abs(expected_signed), abs_tol=1e-9
    ) or not math.isclose(
        record["absolute_foreground_fraction_error"],
        abs(expected_signed / pixels),
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} has inconsistent absolute error")
    if record["observed_empty_mask"] is not (truth == 0):
        raise ValueError(f"{location}.observed_empty_mask is inconsistent")


def _validate_manifest(manifest: Any, *, location: str) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError(f"{location} has an invalid manifest schema")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
    ):
        raise ValueError(f"{location} has an unsupported artifact type/schema")
    if (
        not isinstance(manifest["run_id"], str)
        or len(manifest["run_id"]) != 16
        or any(char not in "0123456789abcdef" for char in manifest["run_id"])
    ):
        raise ValueError(f"{location}.run_id must be 16 lowercase hexadecimal digits")
    for field in ("dataset", "condition", "model", "split", "created_utc"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"{location}.{field} must be non-empty")
    num_images = _integer(
        manifest["num_images"], location=f"{location}.num_images", minimum=1
    )
    if manifest["num_rows"] != num_images:
        raise ValueError(f"{location}.num_rows must equal num_images")
    _digest(manifest["sample_id_sha256"], location=f"{location}.sample_id_sha256")
    if manifest["scope"] != SCOPE or manifest["protocol"] != PROTOCOL:
        raise ValueError(f"{location} does not declare the frozen scope/protocol")
    if manifest["record_fields"] != sorted(RECORD_FIELDS):
        raise ValueError(f"{location}.record_fields is inconsistent")
    if not isinstance(manifest["environment"], dict):
        raise ValueError(f"{location}.environment must be an object")
    provenance = manifest["provenance"]
    required_provenance = {
        "auxiliary_id",
        "auxiliary_lock_path",
        "auxiliary_lock_sha256",
        "auxiliary_spec_path",
        "auxiliary_spec_sha256",
        "canonical_campaign_id",
        "canonical_campaign_lock_path",
        "canonical_campaign_lock_sha256",
        "artifact_id",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "source_sha256",
        "artifact_passes",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError(f"{location}.provenance has an invalid schema")
    for field in (
        "auxiliary_lock_sha256",
        "auxiliary_spec_sha256",
        "canonical_campaign_lock_sha256",
        "artifact_manifest_sha256",
        "source_sha256",
    ):
        _digest(provenance[field], location=f"{location}.provenance.{field}")
    if provenance["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError(f"{location}.provenance.auxiliary_id is inconsistent")
    if provenance["artifact_passes"] != 1:
        raise ValueError(f"{location} must declare exactly one artifact pass")
    expected_run_id = _run_id(
        provenance["auxiliary_lock_sha256"],
        provenance["artifact_manifest_sha256"],
        provenance["source_sha256"],
    )
    if manifest["run_id"] != expected_run_id:
        raise ValueError(f"{location}.run_id differs from its content identity")
    _digest(manifest["records_sha256"], location=f"{location}.records_sha256")
    if not isinstance(manifest["command"], list) or not all(
        isinstance(token, str) for token in manifest["command"]
    ):
        raise ValueError(f"{location}.command must be a string list")


def load_cardinality_diagnostic(
    records_path: str | os.PathLike[str],
) -> CardinalityDiagnostic:
    """Strictly load a records/manifest pair and validate row identity."""

    records_path = Path(records_path)
    if (
        records_path.name != RECORDS_NAME
        or not records_path.is_file()
        or records_path.is_symlink()
    ):
        raise FileNotFoundError(f"expected a regular {RECORDS_NAME}: {records_path}")
    manifest_path = records_path.parent / MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_strict_json(manifest_path, name="cardinality manifest")
    _validate_manifest(manifest, location=str(manifest_path))
    if records_path.parent.name != manifest["run_id"]:
        raise ValueError("diagnostic directory name differs from run_id")
    if sha256_file(records_path) != manifest["records_sha256"]:
        raise ValueError("cardinality records SHA-256 mismatch")

    records = []
    sample_ids = []
    with records_path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                raise ValueError(f"blank line in cardinality records at {index + 1}")
            try:
                record = json.loads(
                    line,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_unique_object,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid record line {index + 1}: {error}") from error
            _validate_record(record, location=f"records[{index}]")
            if record["sample_index"] != index:
                raise ValueError("cardinality records are not contiguous and ordered")
            if record["run_id"] != manifest["run_id"]:
                raise ValueError("cardinality record run_id differs from manifest")
            if record["artifact_id"] != manifest["provenance"]["artifact_id"]:
                raise ValueError("cardinality record artifact_id differs from manifest")
            expected_randomizer = deterministic_pit_randomizer(
                EXPECTED_DIAGNOSTIC_SEED, record["sample_id"]
            )
            if record["pit_randomizer"] != expected_randomizer:
                raise ValueError("cardinality record PIT randomizer is not reproducible")
            sample_ids.append(record["sample_id"])
            records.append(record)
    if len(records) != manifest["num_rows"]:
        raise ValueError("cardinality records count differs from manifest")
    if sample_id_sha256(sample_ids) != manifest["sample_id_sha256"]:
        raise ValueError("cardinality records sample identity differs from manifest")
    return CardinalityDiagnostic(
        records_path.resolve(),
        manifest_path.resolve(),
        manifest,
        tuple(records),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    records, manifest = run_cardinality_diagnostics(parse_args(argv))
    print(f"saved {records}")
    print(f"saved {manifest}")


if __name__ == "__main__":
    main()
