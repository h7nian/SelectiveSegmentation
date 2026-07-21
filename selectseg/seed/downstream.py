"""Immutable downstream locks for the binary training-seed extension.

The canonical binary campaign keys artifacts by ``(dataset, condition)``.
Consequently, seed-1 and seed-2 artifacts cannot safely share one canonical
campaign lock: their paper-facing condition names intentionally agree.  This
module creates one ordinary, canonical-compatible campaign per training seed
and binds both campaigns, all 20 freeze records, and the complete checkpoint
lock in one write-once downstream lock.

No scorer or assembler is forked here.  After this gate, the existing common,
M-specific, assembly, and diagnostic planners can consume the two ordinary
campaign locks without weakening the seed-0 validators.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.submit.main import (
    Config,
    build_campaign_lock,
    load_campaign_lock,
    load_config,
    write_campaign_lock,
)
from selectseg.artifacts import load_binary_artifact
from selectseg.seed.extension import (
    EXPECTED_AUXILIARY_ID,
    EXPECTED_TRAINING_SEEDS,
    _atomic_write_new,
    _checkpoint_entry,
    _digest,
    _exact_fields,
    _load_json,
    _sha256,
    iter_experiments,
    load_checkpoint_lock,
    load_spec_lock,
)


DOWNSTREAM_LOCK_SCHEMA_VERSION = 1
FREEZE_RECORD_SCHEMA_VERSION = 1
EXPECTED_CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
EXPECTED_CANONICAL_PROTOCOL = {
    "gamma_values": [0.5],
    "m_values": [2, 8, 32],
    "quadrature_rule": "midpoint-v1",
    "seeds": [0],
}

_FREEZE_RECORD_FIELDS = frozenset(
    {
        "freeze_record_schema_version",
        "auxiliary_id",
        "created_utc",
        "spec_lock",
        "checkpoint_lock",
        "dataset",
        "model",
        "condition",
        "training_seed",
        "gpu_profile",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "checkpoint_sha256",
    }
)
_DOWNSTREAM_LOCK_FIELDS = frozenset(
    {
        "downstream_lock_schema_version",
        "auxiliary_id",
        "created_utc",
        "spec_lock",
        "checkpoint_lock",
        "campaigns",
    }
)
_CAMPAIGN_BINDING_FIELDS = frozenset(
    {
        "training_seed",
        "campaign_id",
        "config_path",
        "config_sha256",
        "campaign_lock_path",
        "campaign_lock_sha256",
        "freeze_records",
    }
)
_FREEZE_BINDING_FIELDS = frozenset(
    {
        "dataset",
        "model",
        "condition",
        "training_seed",
        "freeze_record_path",
        "freeze_record_sha256",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "artifact_id",
        "checkpoint_sha256",
        "sample_id_sha256",
        "num_samples",
    }
)


def _canonical_config_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _portable(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _freeze_record_path(binding, experiment) -> Path:
    return (
        Path(binding["spec"]["paths"]["freeze_record_root"])
        / experiment["dataset"]["name"]
        / experiment["model"]["name"]
        / f"seed-{experiment['training_seed']}.json"
    )


def load_freeze_record(binding, checkpoint_binding, experiment):
    """Validate one predictable, checkpoint-bound freeze record and artifact."""

    record_path = _freeze_record_path(binding, experiment)
    record = _load_json(record_path)
    _exact_fields(
        record, _FREEZE_RECORD_FIELDS, location=f"freeze record {record_path}"
    )
    dataset = experiment["dataset"]["name"]
    model = experiment["model"]["name"]
    condition = experiment["model"]["condition"]
    seed = experiment["training_seed"]
    if record["freeze_record_schema_version"] != FREEZE_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported freeze-record schema")
    if record["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("freeze record has the wrong auxiliary_id")
    record_spec_lock = record["spec_lock"]
    if (
        not isinstance(record_spec_lock, dict)
        or set(record_spec_lock) != {"path", "sha256"}
        or _resolve(record_spec_lock["path"]) != binding["path"].resolve()
        or record_spec_lock["sha256"] != binding["sha256"]
    ):
        raise ValueError("freeze record is bound to a different spec lock")
    record_checkpoint_lock = record["checkpoint_lock"]
    if (
        not isinstance(record_checkpoint_lock, dict)
        or set(record_checkpoint_lock) != {"path", "sha256"}
        or _resolve(record_checkpoint_lock["path"])
        != checkpoint_binding["path"].resolve()
        or record_checkpoint_lock["sha256"] != checkpoint_binding["sha256"]
    ):
        raise ValueError("freeze record is bound to a different checkpoint lock")
    if (
        record["dataset"],
        record["model"],
        record["condition"],
        record["training_seed"],
    ) != (dataset, model, condition, seed):
        raise ValueError("freeze-record identity differs from its locked experiment")
    if record["gpu_profile"] != experiment["gpu_profile"]:
        raise ValueError("freeze-record GPU profile differs from its locked assignment")

    checkpoint = _checkpoint_entry(checkpoint_binding, dataset, model, seed)
    checkpoint_sha = _digest(
        record["checkpoint_sha256"], location="freeze_record.checkpoint_sha256"
    )
    if checkpoint_sha != checkpoint["checkpoint_sha256"]:
        raise ValueError("freeze record names a different checkpoint")
    artifact_manifest_sha = _digest(
        record["artifact_manifest_sha256"],
        location="freeze_record.artifact_manifest_sha256",
    )
    artifact_path = _resolve(record["artifact_manifest_path"])
    artifact = load_binary_artifact(artifact_path, validate_payloads=False)
    if artifact.manifest_sha256 != artifact_manifest_sha:
        raise ValueError("freeze record artifact-manifest SHA-256 changed")
    manifest = artifact.manifest
    expected = {
        "dataset": dataset,
        "model": model,
        "condition": condition,
        "num_samples": experiment["dataset"]["eval_count"],
        "sample_id_sha256": experiment["dataset"]["eval_sample_id_sha256"],
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise ValueError(f"frozen seed artifact has unexpected {field}")
    if manifest["checkpoint"] is None:
        raise ValueError("target seed artifact must name its trained checkpoint")
    if manifest["checkpoint"]["sha256"] != checkpoint_sha:
        raise ValueError("artifact and checkpoint lock name different checkpoint bytes")
    if _resolve(manifest["checkpoint"]["path"]) != _resolve(
        checkpoint["checkpoint_path"]
    ):
        raise ValueError("artifact and checkpoint lock name different checkpoint paths")
    return {
        "dataset": dataset,
        "model": model,
        "condition": condition,
        "training_seed": seed,
        "freeze_record_path": _portable(record_path),
        "freeze_record_sha256": _sha256(record_path),
        "artifact_manifest_path": _portable(artifact.manifest_path),
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "artifact_id": manifest["artifact_id"],
        "checkpoint_sha256": checkpoint_sha,
        "sample_id_sha256": manifest["sample_id_sha256"],
        "num_samples": manifest["num_samples"],
    }


def validate_freeze_records(binding, checkpoint_binding):
    """Return the exact 20 validated freeze bindings in locked grid order."""

    records = [
        load_freeze_record(binding, checkpoint_binding, experiment)
        for experiment in iter_experiments(binding["spec"])
    ]
    cells = {(row["dataset"], row["model"], row["training_seed"]) for row in records}
    artifact_paths = {row["artifact_manifest_path"] for row in records}
    artifact_ids = {row["artifact_id"] for row in records}
    if len(records) != 20 or len(cells) != 20:
        raise ValueError("freeze gate requires the exact 20-cell seed grid")
    if len(artifact_paths) != 20 or len(artifact_ids) != 20:
        raise ValueError("every seed cell must have a distinct frozen artifact")
    return tuple(records)


def _campaign_config(binding, checkpoint_binding, seed):
    estimator_path = binding["lock"]["estimator_spec"]["path"]
    conditions = []
    for experiment in iter_experiments(binding["spec"]):
        if experiment["training_seed"] != seed:
            continue
        dataset = experiment["dataset"]
        model = experiment["model"]
        checkpoint = _checkpoint_entry(
            checkpoint_binding, dataset["name"], model["name"], seed
        )
        conditions.append(
            {
                "dataset": dataset["name"],
                "condition": model["condition"],
                "model": model["name"],
                "checkpoint": checkpoint["checkpoint_path"],
                "batch_size": dataset["freeze_batch_size"],
                "expected_num_samples": dataset["eval_count"],
            }
        )
    if len(conditions) != 10:
        raise RuntimeError("each training-seed campaign must contain ten conditions")
    paths = binding["spec"]["paths"]
    return {
        "config_schema_version": 1,
        "campaign_id": f"binary-seed{seed}-v1",
        "protocol": EXPECTED_CANONICAL_PROTOCOL,
        "estimator_spec": estimator_path,
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "cpu_partitions": list(EXPECTED_CPU_PARTITIONS),
        "paths": {
            "artifact_output_root": paths["artifact_root"],
            "common_output_root": paths["common_root"],
            "simulation_output_root": paths["simulation_root"],
            "assembly_output_root": paths["assembly_root"],
        },
        "conditions": conditions,
    }


def _config_object(path: Path, payload: dict) -> Config:
    encoded = _canonical_config_bytes(payload)
    return Config(
        path=path.resolve(),
        sha256=hashlib.sha256(encoded).hexdigest(),
        data=payload,
    )


def _default_downstream_path(binding) -> Path:
    checkpoint_path = Path(binding["spec"]["paths"]["checkpoint_lock"])
    return checkpoint_path.with_name("downstream.lock.json")


def write_downstream_lock(binding, checkpoint_binding, output_path=None):
    """Publish two canonical campaign locks after all 20 freezes validate.

    Validation is completed and both canonical locks are built in memory before
    any file is published.  Every destination is write-once.
    """

    destination = Path(output_path or _default_downstream_path(binding))
    if destination != _default_downstream_path(binding):
        raise ValueError(
            f"downstream lock must be written to {_default_downstream_path(binding)}"
        )
    freeze_records = validate_freeze_records(binding, checkpoint_binding)
    parent = destination.parent
    plans = []
    for seed in EXPECTED_TRAINING_SEEDS:
        seed_dir = parent / f"seed-{seed}"
        config_path = seed_dir / "campaign.json"
        campaign_lock_path = seed_dir / "campaign.lock.json"
        for path in (config_path, campaign_lock_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing to overwrite downstream file: {path}")
        payload = _campaign_config(binding, checkpoint_binding, seed)
        config = _config_object(config_path, payload)
        seed_records = [row for row in freeze_records if row["training_seed"] == seed]
        campaign = build_campaign_lock(
            config, [row["artifact_manifest_path"] for row in seed_records]
        )
        plans.append((seed, config, campaign_lock_path, campaign, seed_records))

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite downstream lock: {destination}")
    campaign_bindings = []
    for seed, config, campaign_lock_path, campaign, seed_records in plans:
        _atomic_write_new(config.path, config.data)
        observed_config = load_config(config.path)
        if observed_config.sha256 != config.sha256:
            raise RuntimeError("published seed campaign config bytes changed")
        written_lock, campaign_sha = write_campaign_lock(campaign, campaign_lock_path)
        campaign_bindings.append(
            {
                "training_seed": seed,
                "campaign_id": config.data["campaign_id"],
                "config_path": _portable(config.path),
                "config_sha256": config.sha256,
                "campaign_lock_path": _portable(written_lock),
                "campaign_lock_sha256": campaign_sha,
                "freeze_records": seed_records,
            }
        )
    payload = {
        "downstream_lock_schema_version": DOWNSTREAM_LOCK_SCHEMA_VERSION,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        # Reusing the checkpoint-lock timestamp keeps the lock deterministic.
        "created_utc": checkpoint_binding["lock"]["created_utc"],
        "spec_lock": {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
        },
        "checkpoint_lock": {
            "path": checkpoint_binding["path"].as_posix(),
            "sha256": checkpoint_binding["sha256"],
        },
        "campaigns": campaign_bindings,
    }
    _atomic_write_new(destination, payload)
    print(f"saved {destination}")
    print(f"downstream_lock_sha256={_sha256(destination)}")
    return destination


def load_downstream_lock(path, *, expected_sha256):
    """Load and recursively validate a complete seed downstream lock."""

    source = Path(path)
    observed_sha = _sha256(source)
    if observed_sha != _digest(
        expected_sha256, location="expected downstream-lock sha256"
    ):
        raise ValueError("seed downstream-lock hash mismatch")
    lock = _load_json(source)
    _exact_fields(lock, _DOWNSTREAM_LOCK_FIELDS, location="seed downstream lock")
    if lock["downstream_lock_schema_version"] != DOWNSTREAM_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported seed downstream-lock schema")
    if lock["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("downstream lock has the wrong auxiliary_id")
    for name in ("spec_lock", "checkpoint_lock"):
        _exact_fields(
            lock[name], frozenset({"path", "sha256"}), location=f"lock.{name}"
        )
        _digest(lock[name]["sha256"], location=f"lock.{name}.sha256")
    binding = load_spec_lock(
        Path(lock["spec_lock"]["path"]),
        expected_sha256=lock["spec_lock"]["sha256"],
    )
    checkpoint_binding = load_checkpoint_lock(
        binding,
        Path(lock["checkpoint_lock"]["path"]),
        expected_sha256=lock["checkpoint_lock"]["sha256"],
        verify_files=True,
    )
    if lock["created_utc"] != checkpoint_binding["lock"]["created_utc"]:
        raise ValueError("downstream-lock timestamp differs from its checkpoint lock")
    if not isinstance(lock["campaigns"], list) or len(lock["campaigns"]) != 2:
        raise ValueError("downstream lock must contain exactly two seed campaigns")

    campaigns = []
    observed_seeds = set()
    observed_cells = set()
    for entry in lock["campaigns"]:
        _exact_fields(entry, _CAMPAIGN_BINDING_FIELDS, location="lock.campaigns[]")
        seed = entry["training_seed"]
        if seed not in EXPECTED_TRAINING_SEEDS or seed in observed_seeds:
            raise ValueError("downstream lock has a duplicate or invalid training seed")
        observed_seeds.add(seed)
        if entry["campaign_id"] != f"binary-seed{seed}-v1":
            raise ValueError("seed campaign has an unexpected campaign_id")
        config_path = _resolve(entry["config_path"])
        campaign_path = _resolve(entry["campaign_lock_path"])
        if _sha256(config_path) != _digest(
            entry["config_sha256"], location="campaign.config_sha256"
        ):
            raise ValueError("seed campaign config bytes changed")
        if _sha256(campaign_path) != _digest(
            entry["campaign_lock_sha256"],
            location="campaign.campaign_lock_sha256",
        ):
            raise ValueError("seed campaign-lock bytes changed")
        config = load_config(config_path)
        expected_config = _campaign_config(binding, checkpoint_binding, seed)
        if config.data != expected_config or config.sha256 != entry["config_sha256"]:
            raise ValueError("seed campaign config differs from the locked design")
        lock_path, lock_sha, campaign = load_campaign_lock(campaign_path, config=config)
        if lock_sha != entry["campaign_lock_sha256"]:
            raise ValueError("canonical seed campaign returned an inconsistent hash")
        records = entry["freeze_records"]
        if not isinstance(records, list) or len(records) != 10:
            raise ValueError("each seed campaign must bind ten freeze records")
        expected_experiments = [
            experiment
            for experiment in iter_experiments(binding["spec"])
            if experiment["training_seed"] == seed
        ]
        expected_by_cell = {
            (experiment["dataset"]["name"], experiment["model"]["name"]): experiment
            for experiment in expected_experiments
        }
        artifacts_by_cell = {
            (artifact["dataset"], artifact["model"]): artifact
            for artifact in campaign["artifacts"]
        }
        if set(artifacts_by_cell) != set(expected_by_cell):
            raise ValueError("seed campaign artifact grid differs from its design")
        for row in records:
            _exact_fields(row, _FREEZE_BINDING_FIELDS, location="freeze binding")
            cell = (row["dataset"], row["model"])
            if cell not in expected_by_cell or row["training_seed"] != seed:
                raise ValueError("freeze binding is outside its seed campaign")
            full_cell = (row["dataset"], row["model"], seed)
            if full_cell in observed_cells:
                raise ValueError("duplicate freeze binding in downstream lock")
            observed_cells.add(full_cell)
            current = load_freeze_record(
                binding, checkpoint_binding, expected_by_cell[cell]
            )
            if row != current:
                raise ValueError(
                    "freeze record or artifact changed after downstream lock"
                )
            artifact = artifacts_by_cell[cell]
            for field in (
                "manifest_sha256",
                "artifact_id",
                "checkpoint_sha256",
                "sample_id_sha256",
                "num_samples",
            ):
                row_field = (
                    "artifact_manifest_sha256" if field == "manifest_sha256" else field
                )
                if artifact[field] != row[row_field]:
                    raise ValueError(
                        f"canonical campaign and freeze binding differ on {field}"
                    )
            if _resolve(artifact["manifest_path"]) != _resolve(
                row["artifact_manifest_path"]
            ):
                raise ValueError("canonical campaign names a different artifact path")
        campaigns.append(
            {
                "training_seed": seed,
                "config": config,
                "campaign_lock_path": lock_path,
                "campaign_lock_sha256": lock_sha,
                "campaign": campaign,
            }
        )
    expected_cells = {
        (
            experiment["dataset"]["name"],
            experiment["model"]["name"],
            experiment["training_seed"],
        )
        for experiment in iter_experiments(binding["spec"])
    }
    if (
        observed_seeds != set(EXPECTED_TRAINING_SEEDS)
        or observed_cells != expected_cells
    ):
        raise ValueError("downstream lock does not cover the exact 20-cell seed grid")
    campaigns.sort(key=lambda item: item["training_seed"])
    return {
        "path": source.resolve(),
        "sha256": observed_sha,
        "lock": lock,
        "binding": binding,
        "checkpoint_binding": checkpoint_binding,
        "campaigns": tuple(campaigns),
    }
