"""Strictly assemble one common shard and three independent M shards.

The caller explicitly supplies one M-independent common-score manifest and
exactly the M=2/8/32 simulation manifests.  No input is discovered.  Every
manifest and JSONL hash, schema, artifact identity, sample order, and stable
identity field is checked before producing the 17-score analyzer schema.
Derived floating-point values exist only in the common shard and are therefore
never compared across compute nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from scripts.submit_binary_simulations import load_campaign_lock as validate_lock
from selectseg.binary_artifacts import fsync_directory, publish_directory_no_replace
from selectseg.score_binary_common import (
    AUXILIARY_FIELDS,
    BASE_ROW_FIELDS,
    COMMON_ARTIFACT_TYPE,
    COMMON_SCORE_FIELDS,
    EXPECTED_M_VALUES,
    FINAL_SCORE_FIELDS,
    IDENTITY_JOIN_FIELDS,
    IDENTITY_ROW_FIELDS,
    M_SCORE_FIELDS,
    RISK_FIELDS,
    ROW_SCHEMA_VERSION,
    SIMULATION_ARTIFACT_TYPE,
)


ASSEMBLY_SCHEMA_VERSION = 2
EXPECTED_GAMMA = 0.5
EXPECTED_SEED = 0
EXPECTED_QUADRATURE_RULE = "midpoint-v1"

COMMON_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "common_id",
        "created_utc",
        "condition",
        "model",
        "dataset",
        "split",
        "num_images",
        "num_rows",
        "checkpoint",
        "base_model",
        "source_sha256",
        "environment",
        "cohort",
        "decision_rule",
        "preprocessing",
        "losses",
        "risk_fields",
        "auxiliary_fields",
        "score_fields",
        "void_policy",
        "sdc_empty_convention",
        "sample_id_sha256",
        "jsonl_sha256",
        "common",
        "command",
    }
)
SIMULATION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "simulation_id",
        "created_utc",
        "condition",
        "model",
        "dataset",
        "split",
        "num_images",
        "num_rows",
        "checkpoint",
        "base_model",
        "source_sha256",
        "environment",
        "cohort",
        "decision_rule",
        "preprocessing",
        "losses",
        "risk_fields",
        "auxiliary_fields",
        "score_fields",
        "quadrature",
        "void_policy",
        "sdc_empty_convention",
        "sample_id_sha256",
        "jsonl_sha256",
        "simulation",
        "command",
    }
)
COMMON_PROVENANCE_FIELDS = frozenset(
    {
        "common_id",
        "campaign_id",
        "campaign_lock_path",
        "campaign_lock_sha256",
        "artifact_id",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "artifact_source_sha256",
        "gamma",
    }
)
SIMULATION_PROVENANCE_FIELDS = frozenset(
    {
        "simulation_id",
        "campaign_id",
        "campaign_lock_path",
        "campaign_lock_sha256",
        "artifact_id",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "artifact_source_sha256",
        "estimator_spec_path",
        "estimator_spec_sha256",
        "estimator_id",
        "target_measure",
        "gamma",
        "m",
        "quadrature_rule",
        "seed",
    }
)
EXACT_JOIN_MANIFEST_FIELDS = (
    "condition",
    "model",
    "dataset",
    "split",
    "num_images",
    "num_rows",
    "checkpoint",
    "base_model",
    "cohort",
    "decision_rule",
    "preprocessing",
    "void_policy",
    "sdc_empty_convention",
    "sample_id_sha256",
)


@dataclass(frozen=True)
class CampaignLock:
    path: Path
    sha256: str
    data: dict


@dataclass(frozen=True)
class CommonRun:
    manifest_path: Path
    manifest_sha256: str
    records_path: Path
    manifest: dict
    rows: tuple[dict, ...]


@dataclass(frozen=True)
class SimulationRun:
    manifest_path: Path
    manifest_sha256: str
    records_path: Path
    manifest: dict
    rows: tuple[dict, ...]
    m: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument(
        "--common", required=True, help="the one explicit common-score manifest"
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="M-specific manifest; repeat exactly for M=2,8,32",
    )
    parser.add_argument("--output-root", required=True)
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


def _resolve_recorded_path(lock_path, value):
    raw = Path(_nonempty_string(value, location="recorded path"))
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    lock_candidate = (lock_path.parent / raw).resolve()
    if cwd_candidate.exists() or not lock_candidate.exists():
        return cwd_candidate
    return lock_candidate


def load_campaign_lock(path):
    resolved, digest, data = validate_lock(path)
    return CampaignLock(path=resolved, sha256=digest, data=data)


def _lock_artifact(lock, *, dataset, condition):
    matches = [
        artifact
        for artifact in lock.data["artifacts"]
        if artifact["dataset"] == dataset and artifact["condition"] == condition
    ]
    if len(matches) != 1:
        raise ValueError(
            "campaign lock must contain exactly one matching artifact for "
            f"{(dataset, condition)}; found {len(matches)}"
        )
    return matches[0]


def _field_list(manifest, name, *, location, allow_empty=False):
    value = manifest.get(name)
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(field, str) and field for field in value)
        or len(value) != len(set(value))
    ):
        qualifier = "possibly-empty" if allow_empty else "non-empty"
        raise ValueError(f"{location}.{name} must be a {qualifier} unique field list")
    return tuple(value)


def _load_manifest_and_records(path, *, required_fields, description):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} manifest does not exist: {path}")
    records_path = path.with_name("records.jsonl")
    if not records_path.is_file():
        raise FileNotFoundError(f"records are missing beside {path}")
    raw = path.read_bytes()
    manifest = _loads_strict(raw.decode("utf-8"), source=str(path))
    if not isinstance(manifest, dict) or set(manifest) != required_fields:
        raise ValueError(
            f"{description} manifest {path} must contain exactly "
            f"{sorted(required_fields)}"
        )
    _assert_finite(manifest, location=str(path))
    if manifest["schema_version"] != ROW_SCHEMA_VERSION:
        raise ValueError(f"{path}.schema_version must equal {ROW_SCHEMA_VERSION}")
    for field in (
        "run_id",
        "created_utc",
        "condition",
        "model",
        "dataset",
        "split",
    ):
        _nonempty_string(manifest[field], location=f"{path}.{field}")
    for field in ("num_images", "num_rows"):
        _positive_integer(manifest[field], location=f"{path}.{field}")
    if manifest["num_images"] != manifest["num_rows"]:
        raise ValueError(f"{path} must contain exactly one row per image")
    _digest(manifest["source_sha256"], location=f"{path}.source_sha256")
    expected_jsonl = _digest(manifest["jsonl_sha256"], location=f"{path}.jsonl_sha256")
    _digest(manifest["sample_id_sha256"], location=f"{path}.sample_id_sha256")
    if _sha256(records_path) != expected_jsonl:
        raise ValueError(f"JSONL SHA-256 mismatch for {records_path}")
    return path, raw, records_path, manifest


def _verify_lock_and_artifact_provenance(
    provenance, *, required_fields, lock, manifest, path, identity_field
):
    if not isinstance(provenance, dict) or set(provenance) != required_fields:
        raise ValueError(
            f"{path} provenance must contain exactly {sorted(required_fields)}"
        )
    if provenance["campaign_id"] != lock.data["campaign_id"]:
        raise ValueError(f"{path} belongs to a different campaign")
    if provenance["campaign_lock_sha256"].lower() != lock.sha256:
        raise ValueError(f"{path} was scored against different campaign-lock bytes")
    if _resolve_recorded_path(lock.path, provenance["campaign_lock_path"]) != lock.path:
        raise ValueError(f"{path} names a different campaign-lock path")
    if manifest["run_id"] != provenance[identity_field]:
        raise ValueError(f"{path}.run_id and {identity_field} must agree")
    artifact = _lock_artifact(
        lock, dataset=manifest["dataset"], condition=manifest["condition"]
    )
    expected = {
        "artifact_id": artifact["artifact_id"],
        "artifact_manifest_sha256": artifact["manifest_sha256"].lower(),
        "artifact_source_sha256": artifact["source_sha256"].lower(),
    }
    for field, value in expected.items():
        observed = provenance[field]
        if field.endswith("sha256") and isinstance(observed, str):
            observed = observed.lower()
        if observed != value:
            raise ValueError(f"{path} provenance field {field!r} differs from lock")
    locked_manifest = _resolve_recorded_path(lock.path, artifact["manifest_path"])
    if (
        _resolve_recorded_path(lock.path, provenance["artifact_manifest_path"])
        != locked_manifest
    ):
        raise ValueError(f"{path} names a different artifact-manifest path")
    if manifest["model"] != artifact["model"] or manifest["split"] != artifact["split"]:
        raise ValueError(f"{path} metadata differs from its locked artifact")
    checkpoint = manifest["checkpoint"]
    checkpoint_sha = None if checkpoint is None else checkpoint.get("sha256")
    if checkpoint_sha != artifact["checkpoint_sha256"]:
        raise ValueError(f"{path}.checkpoint differs from its locked artifact")
    if manifest["num_rows"] != artifact["num_samples"]:
        raise ValueError(f"{path} row count differs from its locked artifact")
    if manifest["sample_id_sha256"].lower() != artifact["sample_id_sha256"].lower():
        raise ValueError(f"{path} sample hash differs from its locked artifact")
    if provenance["gamma"] != EXPECTED_GAMMA:
        raise ValueError(f"{path} must use gamma={EXPECTED_GAMMA}")
    expected_decision = {
        "form": "foreground_probability >= gamma",
        "gamma": EXPECTED_GAMMA,
    }
    if manifest["decision_rule"] != expected_decision:
        raise ValueError(f"{path}.decision_rule is not the locked rule")
    return artifact


def _read_rows(records_path, manifest, *, expected_fields, score_fields, common):
    rows = []
    sample_ids = []
    seen_ids = set()
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{records_path}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank JSONL row at {location}")
            row = _loads_strict(line, source=location)
            _assert_finite(row, location=location)
            if not isinstance(row, dict) or set(row) != set(expected_fields):
                raise ValueError(f"unauthorized row schema at {location}")
            if row["schema_version"] != ROW_SCHEMA_VERSION:
                raise ValueError(f"schema_version mismatch at {location}")
            if row["run_id"] != manifest["run_id"]:
                raise ValueError(f"run_id mismatch at {location}")
            sample_id = _nonempty_string(
                row["sample_id"], location=f"{location}.sample_id"
            )
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {records_path}")
            seen_ids.add(sample_id)
            sample_ids.append(sample_id)
            if row["image_id"] != sample_id:
                raise ValueError(f"sample_id/image_id mismatch at {location}")
            _nonempty_string(row["class_name"], location=f"{location}.class_name")
            for field in ("image_index", "class_index", "height", "width"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{location}.{field} must be an integer")
            if row["image_index"] != line_number - 1:
                raise ValueError(f"non-contiguous image_index at {location}")
            if row["class_index"] < 0 or row["height"] <= 0 or row["width"] <= 0:
                raise ValueError(f"invalid index or image dimensions at {location}")
            for field in score_fields:
                if isinstance(row[field], bool) or not isinstance(
                    row[field], (int, float)
                ):
                    raise ValueError(f"{location}.{field} must be numeric")
            if common:
                for field in (
                    "image_diagonal",
                    "truth_foreground_fraction",
                    "prediction_foreground_fraction",
                    *RISK_FIELDS,
                    *AUXILIARY_FIELDS,
                ):
                    if isinstance(row[field], bool) or not isinstance(
                        row[field], (int, float)
                    ):
                        raise ValueError(f"{location}.{field} must be numeric")
                if row["image_diagonal"] <= 0:
                    raise ValueError(f"{location}.image_diagonal must be positive")
                if (
                    not 0 <= row["truth_foreground_fraction"] <= 1
                    or not 0 <= row["prediction_foreground_fraction"] <= 1
                ):
                    raise ValueError(f"{location} contains an invalid fraction")
                if any(not 0 <= row[field] <= 1 for field in RISK_FIELDS):
                    raise ValueError(f"{location} contains a risk outside [0,1]")
                if any(row[field] < 0 for field in AUXILIARY_FIELDS):
                    raise ValueError(
                        f"{location} contains a negative auxiliary distance"
                    )
            rows.append(row)
    if len(rows) != manifest["num_rows"]:
        raise ValueError(f"row-count mismatch for {records_path}")
    sample_hash = _sha256_bytes("\n".join(sample_ids).encode())
    if sample_hash != manifest["sample_id_sha256"].lower():
        raise ValueError(f"sample_id_sha256 mismatch for {records_path}")
    return tuple(rows)


def load_common(path, *, lock):
    path, raw, records_path, manifest = _load_manifest_and_records(
        path,
        required_fields=COMMON_MANIFEST_FIELDS,
        description="common-score",
    )
    if manifest["artifact_type"] != COMMON_ARTIFACT_TYPE:
        raise ValueError(f"{path}.artifact_type must equal {COMMON_ARTIFACT_TYPE!r}")
    if manifest["common_id"] != manifest["run_id"]:
        raise ValueError(f"{path}.common_id and run_id must agree")
    _verify_lock_and_artifact_provenance(
        manifest["common"],
        required_fields=COMMON_PROVENANCE_FIELDS,
        lock=lock,
        manifest=manifest,
        path=path,
        identity_field="common_id",
    )
    if _field_list(manifest, "risk_fields", location=str(path)) != RISK_FIELDS:
        raise ValueError(f"{path}.risk_fields must equal {RISK_FIELDS}")
    if (
        _field_list(manifest, "auxiliary_fields", location=str(path))
        != AUXILIARY_FIELDS
    ):
        raise ValueError(f"{path}.auxiliary_fields must equal {AUXILIARY_FIELDS}")
    scores = _field_list(manifest, "score_fields", location=str(path))
    if scores != COMMON_SCORE_FIELDS:
        raise ValueError(f"{path}.score_fields must equal {COMMON_SCORE_FIELDS}")
    expected_fields = (
        set(BASE_ROW_FIELDS)
        | set(RISK_FIELDS)
        | set(AUXILIARY_FIELDS)
        | set(COMMON_SCORE_FIELDS)
    )
    rows = _read_rows(
        records_path,
        manifest,
        expected_fields=expected_fields,
        score_fields=COMMON_SCORE_FIELDS,
        common=True,
    )
    return CommonRun(path, _sha256_bytes(raw), records_path, manifest, rows)


def _validate_midpoint(quadrature, count, *, location):
    if not isinstance(quadrature, dict) or set(quadrature) != {str(count)}:
        raise ValueError(f"{location} must contain exactly M={count}")
    rule = quadrature[str(count)]
    expected_nodes = [(index + 0.5) / count for index in range(count)]
    expected_weights = [1 / count] * count
    if not isinstance(rule, dict) or set(rule) != {"rule", "nodes", "weights"}:
        raise ValueError(f"{location}.{count} has an invalid schema")
    if rule["rule"] != "midpoint":
        raise ValueError(f"{location}.{count} must use midpoint quadrature")
    if rule["nodes"] != expected_nodes or rule["weights"] != expected_weights:
        raise ValueError(f"{location}.{count} has unexpected nodes or weights")


def load_simulation(path, *, lock):
    path, raw, records_path, manifest = _load_manifest_and_records(
        path,
        required_fields=SIMULATION_MANIFEST_FIELDS,
        description="M-specific simulation",
    )
    if manifest["artifact_type"] != SIMULATION_ARTIFACT_TYPE:
        raise ValueError(
            f"{path}.artifact_type must equal {SIMULATION_ARTIFACT_TYPE!r}"
        )
    if manifest["simulation_id"] != manifest["run_id"]:
        raise ValueError(f"{path}.simulation_id and run_id must agree")
    provenance = manifest["simulation"]
    _verify_lock_and_artifact_provenance(
        provenance,
        required_fields=SIMULATION_PROVENANCE_FIELDS,
        lock=lock,
        manifest=manifest,
        path=path,
        identity_field="simulation_id",
    )
    count = provenance["m"]
    if isinstance(count, bool) or count not in EXPECTED_M_VALUES:
        raise ValueError(f"{path}.simulation.m must be one of {EXPECTED_M_VALUES}")
    if provenance["seed"] != EXPECTED_SEED:
        raise ValueError(f"{path}.simulation.seed must equal {EXPECTED_SEED}")
    if provenance["quadrature_rule"] != EXPECTED_QUADRATURE_RULE:
        raise ValueError(f"{path} uses the wrong quadrature rule")
    estimator = lock.data["estimator"]
    checks = {
        "estimator_spec_sha256": estimator["spec_sha256"].lower(),
        "estimator_id": estimator["estimator_id"],
        "target_measure": estimator["target_measure"],
    }
    for field, expected in checks.items():
        observed = provenance[field]
        if field.endswith("sha256") and isinstance(observed, str):
            observed = observed.lower()
        if observed != expected:
            raise ValueError(f"{path}.simulation.{field} differs from lock")
    estimator_path = _resolve_recorded_path(lock.path, estimator["spec_path"])
    if (
        _resolve_recorded_path(lock.path, provenance["estimator_spec_path"])
        != estimator_path
    ):
        raise ValueError(f"{path} names a different estimator-spec path")
    if (
        _field_list(manifest, "risk_fields", location=str(path), allow_empty=True) != ()
        or _field_list(
            manifest, "auxiliary_fields", location=str(path), allow_empty=True
        )
        != ()
    ):
        raise ValueError(f"{path} M-specific rows must not declare risks")
    scores = _field_list(manifest, "score_fields", location=str(path))
    if scores != M_SCORE_FIELDS[count]:
        raise ValueError(f"{path}.score_fields must equal {M_SCORE_FIELDS[count]}")
    _validate_midpoint(manifest["quadrature"], count, location=f"{path}.quadrature")
    expected_fields = set(IDENTITY_ROW_FIELDS) | set(M_SCORE_FIELDS[count])
    rows = _read_rows(
        records_path,
        manifest,
        expected_fields=expected_fields,
        score_fields=M_SCORE_FIELDS[count],
        common=False,
    )
    return SimulationRun(path, _sha256_bytes(raw), records_path, manifest, rows, count)


def _require_exact_join(common, simulations):
    common_ids = tuple(row["sample_id"] for row in common.rows)
    for count in EXPECTED_M_VALUES:
        run = simulations[count]
        for field in EXACT_JOIN_MANIFEST_FIELDS:
            if _canonical_json(common.manifest[field]) != _canonical_json(
                run.manifest[field]
            ):
                raise ValueError(f"M={count}: manifest field {field!r} differs")
        run_ids = tuple(row["sample_id"] for row in run.rows)
        if common_ids != run_ids:
            raise ValueError(f"M={count}: ordered sample_id cohort differs")
        for common_row, run_row in zip(common.rows, run.rows, strict=True):
            for field in IDENTITY_JOIN_FIELDS:
                if _canonical_json(common_row[field]) != _canonical_json(
                    run_row[field]
                ):
                    raise ValueError(
                        f"M={count}: identity field {field!r} differs for "
                        f"sample_id={common_row['sample_id']!r}"
                    )


def _assembly_source_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "selectseg" / "score_binary_common.py",
        root / "selectseg" / "score_binary_simulation.py",
        root / "scripts" / "submit_binary_simulations.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_assembly(lock, common_manifest, inputs):
    if len(inputs) != len(EXPECTED_M_VALUES):
        raise ValueError(
            f"exactly {len(EXPECTED_M_VALUES)} --input manifests are required"
        )
    resolved = [Path(path).resolve() for path in inputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--input manifests must be distinct")
    common = load_common(common_manifest, lock=lock)
    loaded = [load_simulation(path, lock=lock) for path in resolved]
    simulations = {}
    for run in loaded:
        if run.m in simulations:
            raise ValueError(f"duplicate M={run.m} simulation input")
        simulations[run.m] = run
    if set(simulations) != set(EXPECTED_M_VALUES):
        raise ValueError(
            f"simulation inputs must contain exactly M={EXPECTED_M_VALUES}"
        )
    _require_exact_join(common, simulations)

    identity = {
        "assembly_schema_version": ASSEMBLY_SCHEMA_VERSION,
        "assembly_source_sha256": _assembly_source_sha256(),
        "campaign_lock_sha256": lock.sha256,
        "artifact_id": common.manifest["common"]["artifact_id"],
        "artifact_manifest_sha256": common.manifest["common"][
            "artifact_manifest_sha256"
        ].lower(),
        "artifact_source_sha256": common.manifest["common"][
            "artifact_source_sha256"
        ].lower(),
        "common_manifest_sha256": common.manifest_sha256,
        "common_jsonl_sha256": common.manifest["jsonl_sha256"].lower(),
        "simulation_manifests": {
            str(count): simulations[count].manifest_sha256
            for count in EXPECTED_M_VALUES
        },
        "simulation_jsonl": {
            str(count): simulations[count].manifest["jsonl_sha256"].lower()
            for count in EXPECTED_M_VALUES
        },
    }
    run_id = _sha256_bytes(_canonical_json(identity).encode())[:16]
    assembled_rows = []
    for row_index, common_row in enumerate(common.rows):
        row = dict(common_row)
        row["run_id"] = run_id
        for count in EXPECTED_M_VALUES:
            shard_row = simulations[count].rows[row_index]
            for field in M_SCORE_FIELDS[count]:
                row[field] = shard_row[field]
        assembled_rows.append(row)
    expected_final_fields = (
        set(BASE_ROW_FIELDS)
        | set(RISK_FIELDS)
        | set(AUXILIARY_FIELDS)
        | set(FINAL_SCORE_FIELDS)
    )
    if any(set(row) != expected_final_fields for row in assembled_rows):
        raise RuntimeError("assembled rows do not have the analyzer schema")
    records = "".join(_canonical_json(row) + "\n" for row in assembled_rows).encode()

    manifest = {
        key: value
        for key, value in common.manifest.items()
        if key not in {"common_id", "created_utc", "common", "jsonl_sha256"}
    }
    manifest.update(
        {
            "run_id": run_id,
            "artifact_type": "selectseg.binary_simulation_assembly",
            "score_fields": list(FINAL_SCORE_FIELDS),
            "quadrature": {
                str(count): simulations[count].manifest["quadrature"][str(count)]
                for count in EXPECTED_M_VALUES
            },
            "jsonl_sha256": _sha256_bytes(records),
            "command": ["python", "-m", "scripts.assemble_binary_simulations"],
            "assembly": {
                **identity,
                "campaign_id": lock.data["campaign_id"],
                "campaign_lock_path": _portable_path(lock.path),
                "common_manifest": {
                    "path": _portable_path(common.manifest_path),
                    "sha256": common.manifest_sha256,
                    "jsonl_sha256": common.manifest["jsonl_sha256"].lower(),
                },
                "simulation_manifests": {
                    str(count): {
                        "path": _portable_path(simulations[count].manifest_path),
                        "sha256": simulations[count].manifest_sha256,
                        "jsonl_sha256": simulations[count]
                        .manifest["jsonl_sha256"]
                        .lower(),
                    }
                    for count in EXPECTED_M_VALUES
                },
            },
        }
    )
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    return (
        common.manifest["dataset"],
        common.manifest["condition"],
        run_id,
        records,
        manifest_bytes,
    )


def assemble(
    *,
    campaign_lock,
    inputs,
    output_root,
    common=None,
    common_manifest=None,
):
    if (common is None) == (common_manifest is None):
        raise ValueError("supply exactly one common manifest")
    explicit_common = common if common is not None else common_manifest
    lock = load_campaign_lock(campaign_lock)
    dataset, condition, run_id, records, manifest = prepare_assembly(
        lock, explicit_common, inputs
    )
    output_root = Path(output_root).resolve()
    target = output_root / dataset / condition / run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"assembled output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=target.parent))
    try:
        records_path = temporary / "records.jsonl"
        manifest_path = temporary / "manifest.json"
        with records_path.open("xb") as handle:
            handle.write(records)
            handle.flush()
            os.fsync(handle.fileno())
        with manifest_path.open("xb") as handle:
            handle.write(manifest)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(temporary)
        publish_directory_no_replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def main(argv=None):
    args = parse_args(argv)
    target = assemble(
        campaign_lock=args.campaign_lock,
        common=args.common,
        inputs=args.inputs,
        output_root=args.output_root,
    )
    print(f"saved {target / 'records.jsonl'}")
    print(f"saved {target / 'manifest.json'}")


if __name__ == "__main__":
    main()
