"""Immutable probability/truth artifacts for binary-segmentation simulations.

The expensive model forward pass is independent of the confidence simulation.
This module freezes each native-resolution foreground-probability map together
with its held-out binary truth mask.  Later jobs can therefore vary the hard
decision threshold, quadrature rule, or Monte Carlo seed without rerunning the
model or silently changing the evaluated cohort.

An artifact is a directory containing ``manifest.json`` and one ``.npz`` file
per sample.  Publication is atomic and refuses to replace an existing artifact.
Readers validate the exact schema, content hashes, ordered sample identity,
member paths, array dtypes/shapes, and numerical domains before exposing data.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "selectseg.frozen_binary_maps"
MANIFEST_NAME = "manifest.json"
PROBABILITY_KEY = "foreground_probability"
TRUTH_KEY = "truth"

_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX_16 = re.compile(r"[0-9a-f]{16}\Z")
_PATH_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_id",
        "created_utc",
        "dataset",
        "condition",
        "model",
        "split",
        "class_index",
        "class_name",
        "num_samples",
        "checkpoint",
        "base_model",
        "source_sha256",
        "environment",
        "preprocessing",
        "cohort",
        "sample_id_sha256",
        "samples",
        "command",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "index",
        "sample_id",
        "path",
        "sha256",
        "height",
        "width",
        "probability_dtype",
        "truth_dtype",
    }
)
_BASE_MODEL_FIELDS = frozenset({"name", "source"})
_PREPROCESSING_FIELDS = frozenset({"model_input", "probability_to_native_mask"})
_ENVIRONMENT_FIELDS = frozenset(
    {"packages", "device", "cuda_runtime", "cuda_device", "autocast_dtype"}
)
_PACKAGE_FIELDS = frozenset({"python", "numpy", "torch", "torchvision", "transformers"})
_CHECKPOINT_FIELDS = frozenset({"path", "sha256", "size_bytes"})


@dataclass(frozen=True)
class FrozenBinarySample:
    """One validated sample from a frozen binary-map artifact."""

    index: int
    sample_id: str
    foreground_probability: np.ndarray
    truth: np.ndarray


@dataclass(frozen=True)
class FrozenBinaryArtifact:
    """A structurally validated frozen artifact with an ordered sample reader."""

    manifest_path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str

    @property
    def artifact_dir(self) -> Path:
        return self.manifest_path.parent

    def iter_samples(self) -> Iterator[FrozenBinarySample]:
        """Yield samples in manifest order, validating every payload on access."""

        entries = self.manifest["samples"]
        assert isinstance(entries, list)  # established by the manifest validator
        for entry in entries:
            assert isinstance(entry, dict)
            yield _load_sample(self.artifact_dir, entry)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_sha256(sample_ids: Sequence[str]) -> str:
    """Hash an ordered, unambiguous sequence of validated sample identifiers."""

    validated = [
        _validate_sample_id(value, location="sample_ids") for value in sample_ids
    ]
    return hashlib.sha256("\n".join(validated).encode("utf-8")).hexdigest()


def artifact_identity(manifest: Mapping[str, object]) -> str:
    """Derive the artifact ID from inference and ordered-cohort identity fields."""

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")
    ordered_ids = []
    for index, entry in enumerate(samples):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest.samples[{index}] must be an object")
        ordered_ids.append(
            _validate_sample_id(
                entry.get("sample_id"),
                location=f"manifest.samples[{index}].sample_id",
            )
        )
    checkpoint = manifest.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise ValueError("manifest.checkpoint must be null or an object")
    payload = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "dataset": manifest.get("dataset"),
        "condition": manifest.get("condition"),
        "model": manifest.get("model"),
        "split": manifest.get("split"),
        "class_index": manifest.get("class_index"),
        "class_name": manifest.get("class_name"),
        "checkpoint_sha256": None if checkpoint is None else checkpoint.get("sha256"),
        "base_model": manifest.get("base_model"),
        "source_sha256": manifest.get("source_sha256"),
        "environment": manifest.get("environment"),
        "preprocessing": manifest.get("preprocessing"),
        "cohort": manifest.get("cohort"),
        "sample_ids": ordered_ids,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_binary_artifact(
    manifest_path: str | os.PathLike[str],
    *,
    validate_payloads: bool = True,
) -> FrozenBinaryArtifact:
    """Load and strictly validate one frozen artifact.

    Parameters
    ----------
    manifest_path:
        Explicit path to the artifact's ``manifest.json``.  Directory guesses
        are deliberately rejected so campaign locks identify one exact file.
    validate_payloads:
        If true (the default), eagerly validate every sample hash and array.
        A scorer may pass false and consume :meth:`iter_samples`, which performs
        the same payload validation exactly once while streaming.
    """

    source = Path(manifest_path)
    if source.name != MANIFEST_NAME or not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"expected a regular, non-symlink {MANIFEST_NAME}: {source}"
        )
    manifest_bytes = source.read_bytes()
    manifest = _loads_strict_json(manifest_bytes, source=str(source))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain one JSON object: {source}")
    _validate_manifest(manifest, source=source)
    artifact = FrozenBinaryArtifact(
        manifest_path=source.resolve(),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    if validate_payloads:
        for _ in artifact.iter_samples():
            pass
    return artifact


def write_binary_artifact(
    output_root: str | os.PathLike[str],
    *,
    dataset: str,
    condition: str,
    model: str,
    split: str,
    class_index: int,
    class_name: str,
    checkpoint: Mapping[str, object] | None,
    base_model: Mapping[str, object],
    source_sha256: str,
    environment: Mapping[str, object],
    preprocessing: Mapping[str, object],
    cohort: str,
    sample_ids: Sequence[str],
    samples: Iterable[tuple[str, np.ndarray, np.ndarray]],
    command: Sequence[str],
    created_utc: str,
) -> Path:
    """Stream samples into a new atomically published artifact.

    The return value is the final ``manifest.json`` path.  Existing targets are
    never reused or replaced, including when an identical invocation is rerun.
    """

    dataset = _validate_path_token(dataset, location="dataset")
    condition = _validate_path_token(condition, location="condition")
    model = _validate_path_token(model, location="model")
    split = _validate_path_token(split, location="split")
    class_index = _strict_int(class_index, location="class_index", minimum=0)
    if class_index != 1:
        raise ValueError("class_index must equal 1 for a native binary artifact")
    class_name = _nonempty_string(class_name, location="class_name")
    cohort = _nonempty_string(cohort, location="cohort")
    source_sha256 = _hash_string(source_sha256, location="source_sha256")
    environment_value = _environment(environment, location="environment")
    created_utc = _utc_timestamp(created_utc, location="created_utc")
    checkpoint_value = _checkpoint(checkpoint, location="checkpoint")
    base_model_value = _fixed_string_mapping(
        base_model, _BASE_MODEL_FIELDS, location="base_model"
    )
    preprocessing_value = _fixed_string_mapping(
        preprocessing, _PREPROCESSING_FIELDS, location="preprocessing"
    )
    command_value = _command(command)
    ordered_ids = [
        _validate_sample_id(value, location=f"sample_ids[{index}]")
        for index, value in enumerate(sample_ids)
    ]
    if not ordered_ids:
        raise ValueError("sample_ids cannot be empty")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("sample_ids must be unique")

    provisional_entries = [
        {
            "index": index,
            "sample_id": sample_id,
            "path": f"samples/{index:08d}.npz",
            "sha256": "0" * 64,
            "height": 1,
            "width": 1,
            "probability_dtype": "float32",
            "truth_dtype": "uint8",
        }
        for index, sample_id in enumerate(ordered_ids)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_id": "",
        "created_utc": created_utc,
        "dataset": dataset,
        "condition": condition,
        "model": model,
        "split": split,
        "class_index": class_index,
        "class_name": class_name,
        "num_samples": len(ordered_ids),
        "checkpoint": checkpoint_value,
        "base_model": base_model_value,
        "source_sha256": source_sha256,
        "environment": environment_value,
        "preprocessing": preprocessing_value,
        "cohort": cohort,
        "sample_id_sha256": sample_id_sha256(ordered_ids),
        "samples": provisional_entries,
        "command": command_value,
    }
    manifest["artifact_id"] = artifact_identity(manifest)
    artifact_id = manifest["artifact_id"]

    parent = Path(output_root) / dataset / condition
    parent.mkdir(parents=True, exist_ok=True)
    final_dir = parent / artifact_id
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"artifact already exists: {final_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.tmp-", dir=parent))
    try:
        sample_dir = staging / "samples"
        sample_dir.mkdir()
        entries = []
        iterator = iter(samples)
        for index, expected_id in enumerate(ordered_ids):
            try:
                sample_id, probability, truth = next(iterator)
            except StopIteration as error:
                raise ValueError(
                    f"samples ended at index {index}; expected {len(ordered_ids)}"
                ) from error
            sample_id = _validate_sample_id(
                sample_id, location=f"samples[{index}].sample_id"
            )
            if sample_id != expected_id:
                raise ValueError(
                    f"sample order mismatch at index {index}: "
                    f"expected {expected_id!r}, got {sample_id!r}"
                )
            probability, truth = _validate_arrays(
                probability, truth, location=f"samples[{index}]"
            )
            relative = f"samples/{index:08d}.npz"
            destination = staging / relative
            with destination.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    foreground_probability=np.ascontiguousarray(probability),
                    truth=np.ascontiguousarray(truth),
                )
                handle.flush()
                os.fsync(handle.fileno())
            entries.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "path": relative,
                    "sha256": sha256_file(destination),
                    "height": int(probability.shape[0]),
                    "width": int(probability.shape[1]),
                    "probability_dtype": "float32",
                    "truth_dtype": "uint8",
                }
            )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"samples contains more than the declared {len(ordered_ids)} entries"
            )

        manifest["samples"] = entries
        # File hashes and shapes do not enter the inference identity; nevertheless,
        # check that replacing provisional entries did not change that identity.
        if artifact_identity(manifest) != artifact_id:
            raise RuntimeError("artifact identity changed while writing payloads")
        _validate_manifest(
            manifest,
            source=staging / MANIFEST_NAME,
            require_directory_name=False,
        )
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(sample_dir)
        fsync_directory(staging)
        publish_directory_no_replace(staging, final_dir)
        fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final_dir / MANIFEST_NAME


def _validate_manifest(
    manifest: dict,
    *,
    source: Path,
    require_directory_name: bool = True,
) -> None:
    _exact_keys(manifest, _MANIFEST_FIELDS, location=str(source))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"{source}.schema_version must equal {SCHEMA_VERSION}")
    if manifest["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError(f"{source}.artifact_type must equal {ARTIFACT_TYPE!r}")
    artifact_id = manifest["artifact_id"]
    if not isinstance(artifact_id, str) or _HEX_16.fullmatch(artifact_id) is None:
        raise ValueError(
            f"{source}.artifact_id must be 16 lowercase hexadecimal digits"
        )
    _utc_timestamp(manifest["created_utc"], location=f"{source}.created_utc")
    for field in ("dataset", "condition", "model", "split"):
        _validate_path_token(manifest[field], location=f"{source}.{field}")
    class_index = _strict_int(
        manifest["class_index"], location=f"{source}.class_index", minimum=0
    )
    if class_index != 1:
        raise ValueError(f"{source}.class_index must equal 1")
    _nonempty_string(manifest["class_name"], location=f"{source}.class_name")
    num_samples = _strict_int(
        manifest["num_samples"], location=f"{source}.num_samples", minimum=1
    )
    _checkpoint(manifest["checkpoint"], location=f"{source}.checkpoint")
    _fixed_string_mapping(
        manifest["base_model"],
        _BASE_MODEL_FIELDS,
        location=f"{source}.base_model",
    )
    _hash_string(manifest["source_sha256"], location=f"{source}.source_sha256")
    _environment(manifest["environment"], location=f"{source}.environment")
    _fixed_string_mapping(
        manifest["preprocessing"],
        _PREPROCESSING_FIELDS,
        location=f"{source}.preprocessing",
    )
    _nonempty_string(manifest["cohort"], location=f"{source}.cohort")
    _command(manifest["command"])

    entries = manifest["samples"]
    if not isinstance(entries, list) or len(entries) != num_samples:
        raise ValueError(
            f"{source}.samples must contain exactly num_samples={num_samples} entries"
        )
    sample_ids = []
    for index, entry in enumerate(entries):
        location = f"{source}.samples[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location} must be an object")
        _exact_keys(entry, _SAMPLE_FIELDS, location=location)
        if (
            _strict_int(entry["index"], location=f"{location}.index", minimum=0)
            != index
        ):
            raise ValueError(
                f"{location}.index must equal its ordered position {index}"
            )
        sample_ids.append(
            _validate_sample_id(entry["sample_id"], location=f"{location}.sample_id")
        )
        expected_path = f"samples/{index:08d}.npz"
        if entry["path"] != expected_path:
            raise ValueError(f"{location}.path must equal {expected_path!r}")
        _safe_member_path(source.parent, entry["path"], location=f"{location}.path")
        _hash_string(entry["sha256"], location=f"{location}.sha256")
        _strict_int(entry["height"], location=f"{location}.height", minimum=1)
        _strict_int(entry["width"], location=f"{location}.width", minimum=1)
        if entry["probability_dtype"] != "float32":
            raise ValueError(f"{location}.probability_dtype must equal 'float32'")
        if entry["truth_dtype"] != "uint8":
            raise ValueError(f"{location}.truth_dtype must equal 'uint8'")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{source}.samples contains duplicate sample identifiers")
    expected_sample_hash = sample_id_sha256(sample_ids)
    if manifest["sample_id_sha256"] != expected_sample_hash:
        raise ValueError(f"{source}.sample_id_sha256 does not match ordered samples")
    if artifact_identity(manifest) != artifact_id:
        raise ValueError(f"{source}.artifact_id does not match inference identity")
    if require_directory_name and source.parent.name != artifact_id:
        raise ValueError(
            f"artifact directory must be named by artifact_id {artifact_id!r}: {source.parent}"
        )


def _load_sample(artifact_dir: Path, entry: Mapping[str, object]) -> FrozenBinarySample:
    index = int(entry["index"])
    location = f"samples[{index}]"
    payload_path = _safe_member_path(
        artifact_dir, entry["path"], location=f"{location}.path"
    )
    if not payload_path.is_file() or payload_path.is_symlink():
        raise FileNotFoundError(
            f"{location} is not a regular non-symlink file: {payload_path}"
        )
    if sha256_file(payload_path) != entry["sha256"]:
        raise ValueError(f"{location} SHA-256 mismatch: {payload_path}")
    try:
        with zipfile.ZipFile(payload_path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{location} is not a valid NPZ archive: {payload_path}"
        ) from error
    expected_names = {f"{PROBABILITY_KEY}.npy", f"{TRUTH_KEY}.npy"}
    if len(names) != 2 or set(names) != expected_names:
        raise ValueError(
            f"{location} must contain exactly {sorted(expected_names)}, got {names}"
        )
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            if len(payload.files) != 2 or set(payload.files) != {
                PROBABILITY_KEY,
                TRUTH_KEY,
            }:
                raise ValueError(f"{location} has an unexpected NPZ member schema")
            probability = payload[PROBABILITY_KEY]
            truth = payload[TRUTH_KEY]
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith(location):
            raise
        raise ValueError(f"cannot load {location} payload: {payload_path}") from error
    probability, truth = _validate_arrays(probability, truth, location=location)
    expected_shape = (int(entry["height"]), int(entry["width"]))
    if probability.shape != expected_shape:
        raise ValueError(
            f"{location} shape {probability.shape} does not match manifest {expected_shape}"
        )
    probability.setflags(write=False)
    truth.setflags(write=False)
    return FrozenBinarySample(
        index=index,
        sample_id=str(entry["sample_id"]),
        foreground_probability=probability,
        truth=truth,
    )


def _validate_arrays(probability, truth, *, location: str):
    probability = np.asarray(probability)
    truth = np.asarray(truth)
    if probability.dtype != np.dtype("float32"):
        raise ValueError(f"{location}.foreground_probability must have dtype float32")
    if truth.dtype != np.dtype("uint8"):
        raise ValueError(f"{location}.truth must have dtype uint8")
    if probability.ndim != 2 or truth.ndim != 2:
        raise ValueError(f"{location} arrays must both be two-dimensional")
    if probability.shape != truth.shape:
        raise ValueError(f"{location} probability/truth shapes must be equal")
    if not probability.size:
        raise ValueError(f"{location} arrays cannot be empty")
    if not np.isfinite(probability).all():
        raise ValueError(f"{location}.foreground_probability must be finite")
    if not np.logical_and(probability >= 0, probability <= 1).all():
        raise ValueError(f"{location}.foreground_probability must lie in [0, 1]")
    if not np.logical_or(truth == 0, truth == 1).all():
        raise ValueError(f"{location}.truth must contain only 0 and 1")
    return probability, truth


def _loads_strict_json(payload: bytes, *, source: str):
    def reject_constant(value):
        raise ValueError(f"{source} contains non-finite JSON constant {value!r}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not valid UTF-8 JSON") from error


def _exact_keys(value: Mapping, expected: frozenset[str], *, location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{location} schema mismatch: missing={missing}, extra={extra}"
        )


def _strict_int(value, *, location: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _nonempty_string(value, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{location} must be a nonempty, trimmed string without NUL")
    return value


def _validate_sample_id(value, *, location: str) -> str:
    value = _nonempty_string(value, location=location)
    if "\n" in value or "\r" in value:
        raise ValueError(f"{location} cannot contain line breaks")
    return value


def _validate_path_token(value, *, location: str) -> str:
    value = _nonempty_string(value, location=location)
    if _PATH_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{location} is not a safe path token: {value!r}")
    return value


def _hash_string(value, *, location: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{location} must be 64 lowercase hexadecimal digits")
    return value


def _fixed_string_mapping(value, fields: frozenset[str], *, location: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    _exact_keys(value, fields, location=location)
    return {
        key: _nonempty_string(value[key], location=f"{location}.{key}")
        for key in sorted(fields)
    }


def _environment(value, *, location: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    _exact_keys(value, _ENVIRONMENT_FIELDS, location=location)
    packages = value["packages"]
    if not isinstance(packages, Mapping):
        raise ValueError(f"{location}.packages must be an object")
    _exact_keys(packages, _PACKAGE_FIELDS, location=f"{location}.packages")
    normalized_packages = {}
    for package in sorted(_PACKAGE_FIELDS):
        package_value = packages[package]
        if package == "python" or package_value is not None:
            package_value = _nonempty_string(
                package_value, location=f"{location}.packages.{package}"
            )
        normalized_packages[package] = package_value

    device = _nonempty_string(value["device"], location=f"{location}.device")
    autocast_dtype = _nonempty_string(
        value["autocast_dtype"], location=f"{location}.autocast_dtype"
    )
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"{location}.device must equal 'cpu' or 'cuda'")
    if device == "cuda":
        cuda_runtime = _nonempty_string(
            value["cuda_runtime"], location=f"{location}.cuda_runtime"
        )
        cuda_device = _nonempty_string(
            value["cuda_device"], location=f"{location}.cuda_device"
        )
        if autocast_dtype != "bfloat16":
            raise ValueError(f"{location}.autocast_dtype must equal 'bfloat16' on CUDA")
    else:
        if value["cuda_runtime"] is not None or value["cuda_device"] is not None:
            raise ValueError(f"{location} CPU provenance cannot name CUDA")
        if autocast_dtype != "disabled":
            raise ValueError(f"{location}.autocast_dtype must equal 'disabled' on CPU")
        cuda_runtime = None
        cuda_device = None
    return {
        "packages": normalized_packages,
        "device": device,
        "cuda_runtime": cuda_runtime,
        "cuda_device": cuda_device,
        "autocast_dtype": autocast_dtype,
    }


def _checkpoint(value, *, location: str):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be null or an object")
    _exact_keys(value, _CHECKPOINT_FIELDS, location=location)
    path = _portable_provenance_path(value["path"], location=f"{location}.path")
    digest = _hash_string(value["sha256"], location=f"{location}.sha256")
    size = _strict_int(
        value["size_bytes"], location=f"{location}.size_bytes", minimum=1
    )
    return {"path": path, "sha256": digest, "size_bytes": size}


def _portable_provenance_path(value, *, location: str) -> str:
    value = _nonempty_string(value, location=location)
    if "\\" in value:
        raise ValueError(f"{location} must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{location} must be a normalized relative path")
    return value


def _command(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("command must be a nonempty sequence of strings")
    return [
        _nonempty_string(item, location=f"command[{index}]")
        for index, item in enumerate(value)
    ]


def _utc_timestamp(value, *, location: str) -> str:
    value = _nonempty_string(value, location=location)
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{location} must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{location} must include a UTC offset")
    if timestamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{location} must be expressed in UTC")
    return value


def _safe_member_path(root: Path, value, *, location: str) -> Path:
    value = _portable_provenance_path(value, location=location)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(value).parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{location} escapes the artifact directory") from error
    relative_parts = PurePosixPath(value).parts
    cursor = root
    for part in relative_parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{location} traverses a symlink: {cursor}")
    return candidate


def fsync_directory(path: Path) -> None:
    """Flush directory metadata after creating or publishing children."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically rename ``staging`` while refusing any existing destination."""

    # Linux renameat2 gives true atomic no-replace publication.  The lock-based
    # fallback retains no-overwrite semantics for cooperating writers on systems
    # whose C library does not expose renameat2.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        if error_number not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise OSError(error_number, os.strerror(error_number), destination)

    lock_path = destination.with_name(f".{destination.name}.publish.lock")
    descriptor = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"artifact already exists: {destination}")
        os.rename(staging, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
