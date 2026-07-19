"""Export a deterministic, redacted provenance summary for public release.

The exporter consumes only explicit files.  It binds the final selective-risk
analysis and the aggregate diagnostics to the exact campaign-lock bytes,
checks that every submission receipt is resolved and complete, and publishes
only logical identifiers, SHA-256 digests, and count summaries.  Scheduler
metadata, commands, filesystem paths, timestamps, and environment identities
are deliberately excluded from the output schema.

Example::

    python -m scripts.export_public_provenance \
      --campaign-lock outputs/binary_campaign/campaign.lock.json \
      --analysis outputs/binary_final/analysis.json \
      --diagnostics-analysis outputs/binary_final/diagnostics_analysis.json \
      --phase-receipt freeze outputs/binary_campaign/freeze.receipt.jsonl \
      --phase-receipt common outputs/binary_campaign/common.receipt.jsonl \
      --phase-receipt score outputs/binary_campaign/score.receipt.jsonl \
      --phase-receipt assemble outputs/binary_campaign/assemble.receipt.jsonl \
      --phase-receipt diagnose outputs/binary_campaign/diagnose.receipt.jsonl \
      --training-config pet/clipseg/seed-0 path/to/train_config.json \
      --training-history pet/clipseg/seed-0 path/to/history.json \
      --base-model clipseg CIDAS/clipseg-rd64-refined REVISION WEIGHT_SHA256 \
      --base-model deeplabv3 torchvision/DeepLabV3_ResNet50 REVISION WEIGHT_SHA256 \
      --output public_provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PUBLIC_SCHEMA_VERSION = 1
PUBLIC_ARTIFACT_TYPE = "selectseg.public_provenance"
LOCK_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 2
DIAGNOSTICS_SCHEMA_VERSION = 1
DIAGNOSTICS_ARTIFACT_TYPE = "selectseg.binary_diagnostics_analysis"
RECEIPT_SCHEMA_VERSION = 1
PHASES = ("freeze", "common", "score", "assemble", "diagnose")

LOCK_KEYS = frozenset(
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
LOCK_ARTIFACT_KEYS = frozenset(
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
ANALYSIS_KEYS = frozenset(
    {"schema_version", "provenance", "analysis", "conditions", "multiple_testing"}
)
ANALYSIS_PROVENANCE_KEYS = frozenset(
    {
        "binding",
        "campaign_id",
        "campaign_lock",
        "config_sha256",
        "analysis_source_sha256",
        "inputs",
    }
)
DIAGNOSTICS_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "campaign",
        "scope",
        "aggregation",
        "conditions",
    }
)
RECEIPT_KEYS = frozenset(
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
PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "campaign",
        "analyses",
        "phases",
        "base_models",
        "training",
    }
)

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]*\Z")
_SAFE_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_FORBIDDEN_PUBLIC_FRAGMENTS = (
    "api-key",
    "api_key",
    "apikey",
    "bearer",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)
_CREDENTIAL_PREFIXES = (
    "sk" + "-",
    "gh" + "p_",
    "github" + "_pat_",
)


@dataclass(frozen=True)
class BaseModelSpec:
    """One explicitly identified pretrained base model."""

    model_id: str
    identifier: str
    revision: str
    weights_sha256: str


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--diagnostics-analysis", required=True)
    parser.add_argument(
        "--phase-receipt",
        action="append",
        nargs=2,
        required=True,
        metavar=("PHASE", "RECEIPT_JSONL"),
    )
    parser.add_argument(
        "--training-config",
        action="append",
        nargs=2,
        default=[],
        metavar=("LOGICAL_ID", "TRAIN_CONFIG_JSON"),
    )
    parser.add_argument(
        "--training-history",
        action="append",
        nargs=2,
        default=[],
        metavar=("LOGICAL_ID", "HISTORY_JSON"),
    )
    parser.add_argument(
        "--base-model",
        action="append",
        nargs=4,
        required=True,
        metavar=("MODEL_ID", "IDENTIFIER", "REVISION", "WEIGHTS_SHA256"),
    )
    parser.add_argument("--output", required=True)
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
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    _assert_finite(value, location=str(source))
    return value


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required JSON input does not exist: {path}")
    return _loads_strict(path.read_text(encoding="utf-8"), source=path)


def _assert_finite(value, *, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, location=f"{location}[{index}]")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, *, location):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _positive_integer(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _nonnegative_integer(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a nonnegative integer")
    return value


def _exact_mapping(value, fields, *, location):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{location} must contain exactly {sorted(fields)}")
    return value


def _safe_identifier(value, *, location, revision=False):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{location} must be a non-empty public identifier")
    pattern = _SAFE_REVISION if revision else _SAFE_IDENTIFIER
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{location} is not a portable public identifier")
    if (
        value.startswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{location} must not be an absolute or traversing path")
    lowered = value.lower()
    if any(fragment in lowered for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS):
        raise ValueError(f"{location} contains a secret-like marker")
    if lowered.startswith(_CREDENTIAL_PREFIXES):
        raise ValueError(f"{location} resembles a credential")
    return value


def _sequence(value, *, location, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{location} must be a {qualifier}list")
    return value


def _validate_lock(path):
    lock = _load_json(path)
    _exact_mapping(lock, LOCK_KEYS, location="campaign lock")
    if lock["lock_schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported campaign lock schema")
    campaign_id = _safe_identifier(lock["campaign_id"], location="campaign_id")

    config = _exact_mapping(lock["config"], {"path", "sha256"}, location="lock.config")
    config_sha = _digest(config["sha256"], location="lock.config.sha256")
    protocol = _exact_mapping(
        lock["protocol"],
        {"gamma_values", "m_values", "quadrature_rule", "seeds"},
        location="lock.protocol",
    )
    for field in ("gamma_values", "m_values", "seeds"):
        values = _sequence(protocol[field], location=f"lock.protocol.{field}")
        if len(values) != len({json.dumps(item, sort_keys=True) for item in values}):
            raise ValueError(f"lock.protocol.{field} must not contain duplicates")
    for value in protocol["m_values"]:
        _positive_integer(value, location="lock.protocol.m_values[]")
    for value in protocol["seeds"]:
        _nonnegative_integer(value, location="lock.protocol.seeds[]")
    for value in protocol["gamma_values"]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("lock.protocol.gamma_values[] must be numeric")
    quadrature_rule = _safe_identifier(
        protocol["quadrature_rule"], location="lock.protocol.quadrature_rule"
    )

    estimator = lock["estimator"]
    if not isinstance(estimator, dict):
        raise ValueError("lock.estimator must be an object")
    for field in ("estimator_id", "target_measure", "spec_sha256"):
        if field not in estimator:
            raise ValueError(f"lock.estimator is missing {field}")
    estimator_id = _safe_identifier(
        estimator["estimator_id"], location="lock.estimator.estimator_id"
    )
    target_measure = _safe_identifier(
        estimator["target_measure"], location="lock.estimator.target_measure"
    )
    estimator_sha = _digest(
        estimator["spec_sha256"], location="lock.estimator.spec_sha256"
    )

    artifacts = _sequence(lock["artifacts"], location="lock.artifacts")
    by_key = {}
    training_checkpoints = {}
    datasets = {}
    model_ids = set()
    for index, artifact in enumerate(artifacts):
        location = f"lock.artifacts[{index}]"
        _exact_mapping(artifact, LOCK_ARTIFACT_KEYS, location=location)
        dataset = _safe_identifier(artifact["dataset"], location=f"{location}.dataset")
        condition = _safe_identifier(
            artifact["condition"], location=f"{location}.condition"
        )
        model = _safe_identifier(artifact["model"], location=f"{location}.model")
        key = (dataset, condition)
        if key in by_key:
            raise ValueError(f"campaign lock contains duplicate condition {key}")
        sample_count = _positive_integer(
            artifact["num_samples"], location=f"{location}.num_samples"
        )
        for field in ("manifest_sha256", "source_sha256", "sample_id_sha256"):
            _digest(artifact[field], location=f"{location}.{field}")
        checkpoint_sha = artifact["checkpoint_sha256"]
        if checkpoint_sha is not None:
            checkpoint_sha = _digest(
                checkpoint_sha, location=f"{location}.checkpoint_sha256"
            )
            training_key = (dataset, model)
            previous = training_checkpoints.get(training_key)
            if previous is not None and previous != checkpoint_sha:
                raise ValueError(
                    f"lock has multiple checkpoint hashes for {dataset}/{model}"
                )
            training_checkpoints[training_key] = checkpoint_sha
        by_key[key] = artifact
        datasets.setdefault(dataset, {"conditions": 0, "samples": 0})
        datasets[dataset]["conditions"] += 1
        datasets[dataset]["samples"] += sample_count
        model_ids.add(model)

    summary = {
        "campaign_id": campaign_id,
        "lock_sha256": _sha256(path),
        "config_sha256": config_sha,
        "protocol": {
            "gamma_count": len(protocol["gamma_values"]),
            "m_count": len(protocol["m_values"]),
            "seed_count": len(protocol["seeds"]),
            "quadrature_rule": quadrature_rule,
        },
        "estimator": {
            "estimator_id": estimator_id,
            "target_measure": target_measure,
            "spec_sha256": estimator_sha,
        },
        "condition_count": len(artifacts),
        "locked_sample_count": sum(item["num_samples"] for item in artifacts),
        "datasets": [
            {
                "dataset_id": dataset,
                "condition_count": values["conditions"],
                "locked_sample_count": values["samples"],
            }
            for dataset, values in sorted(datasets.items())
        ],
    }
    return lock, by_key, training_checkpoints, model_ids, summary


def _condition_keys(conditions, *, location, count_field):
    keys = set()
    total_count = 0
    secondary_count = 0
    for index, condition in enumerate(_sequence(conditions, location=location)):
        item_location = f"{location}[{index}]"
        if not isinstance(condition, dict):
            raise ValueError(f"{item_location} must be an object")
        for field in ("dataset", "condition", count_field):
            if field not in condition:
                raise ValueError(f"{item_location} is missing {field}")
        key = (
            _safe_identifier(condition["dataset"], location=f"{item_location}.dataset"),
            _safe_identifier(
                condition["condition"], location=f"{item_location}.condition"
            ),
        )
        if key in keys:
            raise ValueError(f"{location} contains duplicate condition {key}")
        keys.add(key)
        total_count += _positive_integer(
            condition[count_field], location=f"{item_location}.{count_field}"
        )
        if count_field == "num_rows":
            secondary_count += _positive_integer(
                condition.get("num_image_clusters"),
                location=f"{item_location}.num_image_clusters",
            )
        else:
            secondary_count += _positive_integer(
                condition.get("num_pixels"), location=f"{item_location}.num_pixels"
            )
    return keys, total_count, secondary_count


def _validate_analysis(path, *, lock_summary, locked_keys):
    result = _load_json(path)
    _exact_mapping(result, ANALYSIS_KEYS, location="final analysis")
    if result["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported final analysis schema")
    provenance = _exact_mapping(
        result["provenance"], ANALYSIS_PROVENANCE_KEYS, location="analysis.provenance"
    )
    if provenance["binding"] != "campaign-lock":
        raise ValueError("final analysis is not campaign-lock bound")
    if provenance["campaign_id"] != lock_summary["campaign_id"]:
        raise ValueError("analysis campaign_id differs from the campaign lock")
    campaign_lock = _exact_mapping(
        provenance["campaign_lock"],
        {"logical_name", "sha256"},
        location="analysis.provenance.campaign_lock",
    )
    observed_lock_sha = _digest(
        campaign_lock["sha256"],
        location="analysis.provenance.campaign_lock.sha256",
    )
    if observed_lock_sha != lock_summary["lock_sha256"]:
        raise ValueError("analysis lock SHA-256 does not match campaign-lock bytes")
    if (
        _digest(
            provenance["config_sha256"], location="analysis.provenance.config_sha256"
        )
        != lock_summary["config_sha256"]
    ):
        raise ValueError("analysis config SHA-256 differs from the campaign lock")
    _digest(
        provenance["analysis_source_sha256"],
        location="analysis.provenance.analysis_source_sha256",
    )
    inputs = _sequence(provenance["inputs"], location="analysis.provenance.inputs")
    if len(inputs) != len(locked_keys):
        raise ValueError("final analysis input count differs from the campaign lock")
    input_keys = set()
    for index, item in enumerate(inputs):
        location = f"analysis.provenance.inputs[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{location} must be an object")
        for field in ("dataset", "condition", "manifest_sha256", "records_sha256"):
            if field not in item:
                raise ValueError(f"{location} is missing {field}")
        key = (
            _safe_identifier(item["dataset"], location=f"{location}.dataset"),
            _safe_identifier(item["condition"], location=f"{location}.condition"),
        )
        if key in input_keys:
            raise ValueError(f"analysis provenance contains duplicate condition {key}")
        input_keys.add(key)
        _digest(item["manifest_sha256"], location=f"{location}.manifest_sha256")
        _digest(item["records_sha256"], location=f"{location}.records_sha256")
    if input_keys != locked_keys:
        raise ValueError("final analysis provenance conditions differ from the lock")
    keys, row_count, cluster_count = _condition_keys(
        result["conditions"], location="analysis.conditions", count_field="num_rows"
    )
    if keys != locked_keys:
        raise ValueError("final analysis conditions differ from the campaign lock")
    if not isinstance(result["analysis"], dict) or not isinstance(
        result["multiple_testing"], dict
    ):
        raise ValueError("final analysis metadata must be objects")
    return {
        "artifact_sha256": _sha256(path),
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "condition_count": len(keys),
        "row_count": row_count,
        "image_cluster_count": cluster_count,
    }


def _validate_diagnostics(path, *, lock_summary, locked_keys):
    result = _load_json(path)
    _exact_mapping(result, DIAGNOSTICS_KEYS, location="diagnostics analysis")
    if result["schema_version"] != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("unsupported diagnostics analysis schema")
    if result["artifact_type"] != DIAGNOSTICS_ARTIFACT_TYPE:
        raise ValueError("unsupported diagnostics artifact type")
    campaign = result["campaign"]
    if not isinstance(campaign, dict):
        raise ValueError("diagnostics campaign must be an object")
    required = {
        "campaign_id",
        "lock_sha256",
        "num_locked_conditions",
        "num_analyzed_conditions",
        "complete_predeclared_campaign",
    }
    missing = required - set(campaign)
    if missing:
        raise ValueError(f"diagnostics campaign is missing {sorted(missing)}")
    if campaign["campaign_id"] != lock_summary["campaign_id"]:
        raise ValueError("diagnostics campaign_id differs from the campaign lock")
    if (
        _digest(campaign["lock_sha256"], location="diagnostics.campaign.lock_sha256")
        != lock_summary["lock_sha256"]
    ):
        raise ValueError("diagnostics lock SHA-256 does not match campaign-lock bytes")
    condition_count = len(locked_keys)
    if (
        campaign["num_locked_conditions"] != condition_count
        or campaign["num_analyzed_conditions"] != condition_count
        or campaign["complete_predeclared_campaign"] is not True
    ):
        raise ValueError("diagnostics analysis is not a complete locked campaign")
    keys, image_count, pixel_count = _condition_keys(
        result["conditions"],
        location="diagnostics.conditions",
        count_field="num_images",
    )
    if keys != locked_keys:
        raise ValueError("diagnostics conditions differ from the campaign lock")
    return {
        "artifact_sha256": _sha256(path),
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "condition_count": len(keys),
        "image_count": image_count,
        "pixel_count": pixel_count,
    }


def _receipt_identity(key, *, location):
    values = _sequence(key, location=location)
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"{location}[{index}] must be a JSON scalar")
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_receipt(path, phase, *, expected_count):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required phase receipt does not exist: {path}")
    latest = {}
    commands = {}
    submitted_job_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank receipt row at {location}")
            event = _loads_strict(line, source=location)
            _exact_mapping(event, RECEIPT_KEYS, location=location)
            if event["receipt_schema_version"] != RECEIPT_SCHEMA_VERSION:
                raise ValueError(f"unsupported receipt schema at {location}")
            if event["phase"] != phase:
                raise ValueError(f"receipt phase mismatch at {location}")
            identity = _receipt_identity(event["key"], location=f"{location}.key")
            command = event["command"]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) and item for item in command)
            ):
                raise ValueError(f"{location}.command must be a non-empty string list")
            command_tuple = tuple(command)
            if identity in commands and commands[identity] != command_tuple:
                raise ValueError(f"submission command changed at {location}")
            commands[identity] = command_tuple
            status = event["status"]
            if status not in {"submitting", "submitted", "failed"}:
                raise ValueError(f"invalid receipt status at {location}")
            if status == "submitted":
                job_id = event["job_id"]
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError(f"{location}.job_id must identify a submitted job")
                if job_id in submitted_job_ids:
                    raise ValueError(
                        f"duplicate submitted job identifier at {location}"
                    )
                submitted_job_ids.add(job_id)
            elif event["job_id"] is not None:
                raise ValueError(f"{location}.job_id must be null for status={status}")
            previous = latest.get(identity)
            allowed = {
                None: {"submitting"},
                "submitting": {"submitted", "failed"},
                "failed": {"submitting"},
                "submitted": set(),
            }[previous]
            if status not in allowed:
                raise ValueError(f"invalid receipt state transition at {location}")
            latest[identity] = status
    if not latest:
        raise ValueError(f"{phase} receipt is empty")
    unresolved = sum(status != "submitted" for status in latest.values())
    if unresolved:
        raise ValueError(
            f"{phase} receipt contains {unresolved} unresolved submissions"
        )
    submitted_count = len(latest)
    if submitted_count != expected_count:
        raise ValueError(
            f"{phase} receipt has {submitted_count} submitted jobs; "
            f"expected {expected_count}"
        )
    return {
        "phase_id": phase,
        "receipt_sha256": _sha256(path),
        "submitted_count": submitted_count,
        # Completion is established by the lock (freeze), final analysis
        # (common/score/assemble), or diagnostics analysis (diagnose).
        "completed_count": expected_count,
    }


def _pairs(values, *, label):
    result = {}
    for key, value in values.items() if isinstance(values, dict) else values:
        key = _safe_identifier(key, location=f"{label} logical ID")
        if key in result:
            raise ValueError(f"duplicate {label} logical ID {key!r}")
        result[key] = Path(value)
    return result


def _validate_training(config_paths, history_paths, *, training_checkpoints):
    configs = _pairs(config_paths, label="training config")
    histories = _pairs(history_paths, label="training history")
    unknown_histories = set(histories) - set(configs)
    if unknown_histories:
        raise ValueError(
            f"training histories lack matching configs: {sorted(unknown_histories)}"
        )
    observed_pairs = {}
    entries = []
    for logical_id, path in sorted(configs.items()):
        config = _load_json(path)
        if not isinstance(config, dict):
            raise ValueError(f"training config {logical_id!r} must be an object")
        for field in ("dataset", "model", "seed"):
            if field not in config:
                raise ValueError(f"training config {logical_id!r} is missing {field}")
        dataset = _safe_identifier(
            str(config["dataset"]), location=f"{logical_id}.dataset"
        )
        model = _safe_identifier(str(config["model"]), location=f"{logical_id}.model")
        seed = str(config["seed"])
        if not seed.isdigit():
            raise ValueError(f"training config {logical_id!r} has a non-integer seed")
        expected_logical_id = f"{dataset}/{model}/seed-{int(seed)}"
        if logical_id != expected_logical_id:
            raise ValueError(
                f"training logical ID {logical_id!r} does not match config identity "
                f"{expected_logical_id!r}"
            )
        pair = (dataset, model)
        if pair in observed_pairs:
            raise ValueError(
                f"multiple training configs provided for {dataset}/{model}"
            )
        observed_pairs[pair] = logical_id
        if pair not in training_checkpoints:
            raise ValueError(f"training config {logical_id!r} has no locked checkpoint")
        history_path = histories.get(logical_id)
        if history_path is None:
            history_sha = None
            history_count = 0
        else:
            history = _load_json(history_path)
            if not isinstance(history, list) or not all(
                isinstance(record, dict) for record in history
            ):
                raise ValueError(f"training history {logical_id!r} must be a JSON list")
            history_sha = _sha256(history_path)
            history_count = len(history)
        entries.append(
            {
                "logical_id": logical_id,
                "checkpoint_sha256": training_checkpoints[pair],
                "config_sha256": _sha256(path),
                "history_sha256": history_sha,
                "history_record_count": history_count,
            }
        )
    missing = set(training_checkpoints) - set(observed_pairs)
    if missing:
        missing_ids = [f"{dataset}/{model}" for dataset, model in sorted(missing)]
        raise ValueError(
            f"missing training configs for locked checkpoints: {missing_ids}"
        )
    return entries


def _validate_base_models(base_models, *, expected_model_ids):
    entries = []
    seen = set()
    for spec in base_models:
        if not isinstance(spec, BaseModelSpec):
            spec = BaseModelSpec(*spec)
        model_id = _safe_identifier(spec.model_id, location="base model ID")
        if model_id in seen:
            raise ValueError(f"duplicate base model ID {model_id!r}")
        seen.add(model_id)
        identifier = _safe_identifier(
            spec.identifier, location=f"base_models.{model_id}.identifier"
        )
        revision = _safe_identifier(
            spec.revision,
            location=f"base_models.{model_id}.revision",
            revision=True,
        )
        if revision.lower() in {"default", "latest", "main", "master", "unknown"}:
            raise ValueError(f"base model {model_id!r} revision is not immutable")
        entries.append(
            {
                "model_id": model_id,
                "identifier": identifier,
                "revision": revision,
                "weights_sha256": _digest(
                    spec.weights_sha256,
                    location=f"base_models.{model_id}.weights_sha256",
                ),
            }
        )
    if seen != set(expected_model_ids):
        raise ValueError(
            "base-model IDs differ from campaign models: "
            f"expected={sorted(expected_model_ids)}, observed={sorted(seen)}"
        )
    return sorted(entries, key=lambda item: item["model_id"])


def validate_public_provenance(result):
    """Validate the exact redacted output schema before publication."""

    _exact_mapping(result, PUBLIC_KEYS, location="public provenance")
    if result["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise ValueError("unsupported public provenance schema")
    if result["artifact_type"] != PUBLIC_ARTIFACT_TYPE:
        raise ValueError("unsupported public provenance artifact type")
    campaign = _exact_mapping(
        result["campaign"],
        {
            "campaign_id",
            "lock_sha256",
            "config_sha256",
            "protocol",
            "estimator",
            "condition_count",
            "locked_sample_count",
            "datasets",
        },
        location="public provenance.campaign",
    )
    _safe_identifier(campaign["campaign_id"], location="campaign.campaign_id")
    for field in ("lock_sha256", "config_sha256"):
        _digest(campaign[field], location=f"campaign.{field}")
    _positive_integer(campaign["condition_count"], location="campaign.condition_count")
    locked_sample_count = _positive_integer(
        campaign["locked_sample_count"], location="campaign.locked_sample_count"
    )
    protocol = _exact_mapping(
        campaign["protocol"],
        {"gamma_count", "m_count", "seed_count", "quadrature_rule"},
        location="campaign.protocol",
    )
    for field in ("gamma_count", "m_count", "seed_count"):
        _positive_integer(protocol[field], location=f"campaign.protocol.{field}")
    _safe_identifier(
        protocol["quadrature_rule"], location="campaign.protocol.quadrature_rule"
    )
    estimator = _exact_mapping(
        campaign["estimator"],
        {"estimator_id", "target_measure", "spec_sha256"},
        location="campaign.estimator",
    )
    _safe_identifier(
        estimator["estimator_id"], location="campaign.estimator.estimator_id"
    )
    _safe_identifier(
        estimator["target_measure"], location="campaign.estimator.target_measure"
    )
    _digest(estimator["spec_sha256"], location="campaign.estimator.spec_sha256")
    if not isinstance(campaign["datasets"], list) or not campaign["datasets"]:
        raise ValueError("campaign.datasets must be a non-empty list")
    dataset_ids = []
    dataset_conditions = 0
    dataset_samples = 0
    for index, dataset in enumerate(campaign["datasets"]):
        dataset = _exact_mapping(
            dataset,
            {"dataset_id", "condition_count", "locked_sample_count"},
            location=f"campaign.datasets[{index}]",
        )
        dataset_ids.append(
            _safe_identifier(
                dataset["dataset_id"],
                location=f"campaign.datasets[{index}].dataset_id",
            )
        )
        dataset_conditions += _positive_integer(
            dataset["condition_count"],
            location=f"campaign.datasets[{index}].condition_count",
        )
        dataset_samples += _positive_integer(
            dataset["locked_sample_count"],
            location=f"campaign.datasets[{index}].locked_sample_count",
        )
    if dataset_ids != sorted(set(dataset_ids)):
        raise ValueError("campaign datasets must be unique and sorted")
    if dataset_conditions != campaign["condition_count"]:
        raise ValueError("campaign dataset condition counts are inconsistent")
    if dataset_samples != locked_sample_count:
        raise ValueError("campaign dataset sample counts are inconsistent")

    analyses = _exact_mapping(
        result["analyses"], {"selective", "diagnostics"}, location="analyses"
    )
    expected_analysis_fields = {
        "selective": {
            "artifact_sha256",
            "schema_version",
            "condition_count",
            "row_count",
            "image_cluster_count",
        },
        "diagnostics": {
            "artifact_sha256",
            "schema_version",
            "condition_count",
            "image_count",
            "pixel_count",
        },
    }
    for name, fields in expected_analysis_fields.items():
        analysis = _exact_mapping(analyses[name], fields, location=f"analyses.{name}")
        _digest(
            analysis["artifact_sha256"], location=f"analyses.{name}.artifact_sha256"
        )
        for field in fields - {"artifact_sha256"}:
            _positive_integer(analysis[field], location=f"analyses.{name}.{field}")

    phases = result["phases"]
    if not isinstance(phases, list) or [
        item.get("phase_id") for item in phases
    ] != list(PHASES):
        raise ValueError("public provenance phases are missing or out of order")
    for index, phase in enumerate(phases):
        _exact_mapping(
            phase,
            {"phase_id", "receipt_sha256", "submitted_count", "completed_count"},
            location=f"phases[{index}]",
        )
        _digest(phase["receipt_sha256"], location=f"phases[{index}].receipt_sha256")
        submitted = _positive_integer(
            phase["submitted_count"], location=f"phases[{index}].submitted_count"
        )
        completed = _positive_integer(
            phase["completed_count"], location=f"phases[{index}].completed_count"
        )
        if completed != submitted:
            raise ValueError(f"phases[{index}] is not complete")

    if not isinstance(result["base_models"], list) or not result["base_models"]:
        raise ValueError("base_models must be a non-empty list")
    model_ids = []
    for index, model in enumerate(result["base_models"]):
        _exact_mapping(
            model,
            {"model_id", "identifier", "revision", "weights_sha256"},
            location=f"base_models[{index}]",
        )
        model_ids.append(
            _safe_identifier(
                model["model_id"], location=f"base_models[{index}].model_id"
            )
        )
        _safe_identifier(
            model["identifier"], location=f"base_models[{index}].identifier"
        )
        _safe_identifier(
            model["revision"], location=f"base_models[{index}].revision", revision=True
        )
        _digest(
            model["weights_sha256"], location=f"base_models[{index}].weights_sha256"
        )
    if model_ids != sorted(set(model_ids)):
        raise ValueError("base models must be unique and sorted")

    if not isinstance(result["training"], list):
        raise ValueError("training must be a list")
    training_ids = []
    for index, training in enumerate(result["training"]):
        _exact_mapping(
            training,
            {
                "logical_id",
                "checkpoint_sha256",
                "config_sha256",
                "history_sha256",
                "history_record_count",
            },
            location=f"training[{index}]",
        )
        training_ids.append(
            _safe_identifier(
                training["logical_id"], location=f"training[{index}].logical_id"
            )
        )
        for field in ("checkpoint_sha256", "config_sha256"):
            _digest(training[field], location=f"training[{index}].{field}")
        if training["history_sha256"] is not None:
            _digest(
                training["history_sha256"],
                location=f"training[{index}].history_sha256",
            )
        _nonnegative_integer(
            training["history_record_count"],
            location=f"training[{index}].history_record_count",
        )
        if training["history_sha256"] is None and training["history_record_count"] != 0:
            raise ValueError(f"training[{index}] has history counts without a hash")
    if training_ids != sorted(set(training_ids)):
        raise ValueError("training entries must be unique and sorted")
    return result


def build_public_provenance(
    campaign_lock,
    analysis,
    diagnostics_analysis,
    *,
    phase_receipts: Mapping[str, str | Path],
    training_configs: Mapping[str, str | Path],
    training_histories: Mapping[str, str | Path],
    base_models: Sequence[BaseModelSpec | Sequence[str]],
):
    """Build one strict public summary without copying sensitive input fields."""

    _, locked, training_checkpoints, model_ids, campaign = _validate_lock(campaign_lock)
    locked_keys = set(locked)
    selective = _validate_analysis(
        analysis, lock_summary=campaign, locked_keys=locked_keys
    )
    diagnostics = _validate_diagnostics(
        diagnostics_analysis, lock_summary=campaign, locked_keys=locked_keys
    )

    receipts = _pairs(phase_receipts, label="phase receipt")
    if set(receipts) != set(PHASES):
        raise ValueError(
            f"phase receipts must contain exactly {list(PHASES)}; "
            f"got {sorted(receipts)}"
        )
    condition_count = campaign["condition_count"]
    protocol = campaign["protocol"]
    expected_counts = {
        "freeze": condition_count,
        "common": condition_count * protocol["gamma_count"],
        "score": (
            condition_count
            * protocol["gamma_count"]
            * protocol["m_count"]
            * protocol["seed_count"]
        ),
        "assemble": condition_count,
        "diagnose": condition_count,
    }
    phases = [
        _validate_receipt(receipts[phase], phase, expected_count=expected_counts[phase])
        for phase in PHASES
    ]
    training = _validate_training(
        training_configs,
        training_histories,
        training_checkpoints=training_checkpoints,
    )
    models = _validate_base_models(base_models, expected_model_ids=model_ids)
    result = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "artifact_type": PUBLIC_ARTIFACT_TYPE,
        "campaign": campaign,
        "analyses": {"selective": selective, "diagnostics": diagnostics},
        "phases": phases,
        "base_models": models,
        "training": training,
    }
    return validate_public_provenance(result)


def write_public_provenance(result, output):
    """Atomically write deterministic, sorted JSON."""

    validate_public_provenance(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def _mapping_from_cli(values, *, label):
    mapping = {}
    for key, path in values:
        if key in mapping:
            raise ValueError(f"duplicate {label} {key!r}")
        mapping[key] = path
    return mapping


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    result = build_public_provenance(
        args.campaign_lock,
        args.analysis,
        args.diagnostics_analysis,
        phase_receipts=_mapping_from_cli(args.phase_receipt, label="phase receipt"),
        training_configs=_mapping_from_cli(
            args.training_config, label="training config"
        ),
        training_histories=_mapping_from_cli(
            args.training_history, label="training history"
        ),
        base_models=[BaseModelSpec(*values) for values in args.base_model],
    )
    write_public_provenance(result, args.output)
    print("saved deterministic public provenance")


if __name__ == "__main__":
    main()
