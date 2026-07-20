"""Strict execution helpers for the target-model training-seed extension.

The primary seed-0 campaign is never modified.  This module consumes a frozen
auxiliary specification, runs exactly one seed-extension training or freeze
experiment, and writes no-overwrite provenance records.  A checkpoint lock is
created only after all 20 training records validate; freeze jobs cannot be
planned or run before that gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from selectseg.binary_artifacts import load_binary_artifact, sha256_file


DEFAULT_SPEC_LOCK = "configs/auxiliary/binary_seed_extension-v1.lock.json"
DEFAULT_SPEC_LOCK_SHA256 = (
    "3fb7b721a4b54e467383c03b03168166a1d2e9f197d3a26eade0161df931deed"
)
SPEC_SCHEMA_VERSION = 1
SPEC_LOCK_SCHEMA_VERSION = 1
CHECKPOINT_LOCK_SCHEMA_VERSION = 1
TRAIN_RECORD_SCHEMA_VERSION = 1
FREEZE_RECORD_SCHEMA_VERSION = 1
EXPECTED_AUXILIARY_ID = "binary-target-seed-extension-v1"
EXPECTED_DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
EXPECTED_MODELS = ("clipseg", "deeplabv3")
EXPECTED_TRAINING_SEEDS = (1, 2)
EXPECTED_M_VALUES = (2, 8, 32)
EXPECTED_GAMMA = 0.5
EXPECTED_QUADRATURE_SEED = 0
EXPECTED_GPU_PROFILES = (
    ("saffo-a100", "ssafo", "gpu:a100:1"),
    ("apollo_agate", "ssafo", "gpu:a100:1"),
)
EXPECTED_DATASET_PROTOCOL = {
    "pet": ("trainval", "test", ["cat", "dog"], [1, 1], 8),
    "kvasir": ("train", "test", ["polyp"], [1], 8),
    "fives": ("train", "test", ["retinal blood vessels"], [1], 4),
    "isic": ("train", "test", ["skin lesion"], [1], 4),
    "tn3k": ("train", "test", ["thyroid nodule"], [1], 4),
}
EXPECTED_MODEL_PROTOCOL = {
    "clipseg": {
        "name": "clipseg",
        "condition": "clipseg-target",
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "num_workers": 8,
        "optimizer": "AdamW",
        "scheduler": "polynomial_power_0.9_per_step",
        "gradient_clip_norm": 1.0,
        "autocast_dtype": "bfloat16",
        "image_size": 352,
        "trainable_scope": "decoder_only",
    },
    "deeplabv3": {
        "name": "deeplabv3",
        "condition": "deeplabv3-target",
        "epochs": 40,
        "batch_size": 16,
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "num_workers": 8,
        "optimizer": "SGD_momentum_0.9",
        "scheduler": "polynomial_power_0.9_per_step",
        "gradient_clip_norm": 1.0,
        "autocast_dtype": "bfloat16",
        "image_size": 512,
        "trainable_scope": "end_to_end",
    },
}

_SPEC_FIELDS = frozenset(
    {
        "spec_schema_version",
        "auxiliary_id",
        "primary_campaign",
        "protocol",
        "data_root",
        "datasets",
        "models",
        "training_preprocessing",
        "freeze_preprocessing",
        "environment",
        "gpu_profiles",
        "partition_assignment",
        "paths",
        "downstream_job_design",
    }
)
_SPEC_LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "auxiliary_id",
        "spec",
        "canonical_campaign_lock",
        "estimator_spec",
        "source_files",
        "base_model_files",
        "reference_seed0_train_configs",
        "scheduler_validation",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "name",
        "train_split",
        "eval_split",
        "train_count",
        "train_sample_id_sha256",
        "eval_count",
        "eval_sample_id_sha256",
        "prompts",
        "prompt_classes",
        "freeze_batch_size",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "name",
        "condition",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "optimizer",
        "scheduler",
        "gradient_clip_norm",
        "autocast_dtype",
        "image_size",
        "trainable_scope",
    }
)
_CHECKPOINT_LOCK_FIELDS = frozenset(
    {
        "checkpoint_lock_schema_version",
        "auxiliary_id",
        "created_utc",
        "spec_lock",
        "checkpoints",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "dataset",
        "model",
        "condition",
        "training_seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "train_config_path",
        "train_config_sha256",
        "history_path",
        "history_sha256",
        "train_record_path",
        "train_record_sha256",
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


def _load_json(path):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"expected a regular non-symlink JSON file: {source}")
    try:
        value = json.loads(
            source.read_text(),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    return value


def _sha256(path):
    return sha256_file(Path(path))


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


def _positive_int(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _portable_path(value, *, location):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{location} must be a non-empty relative path")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{location} cannot traverse parents")
    return path


def _verify_file_binding(binding, *, location):
    _exact_fields(binding, frozenset({"path", "sha256"}), location=location)
    path = _portable_path(binding["path"], location=f"{location}.path")
    expected = _digest(binding["sha256"], location=f"{location}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"locked file does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"locked file hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _validate_spec(spec):
    _exact_fields(spec, _SPEC_FIELDS, location="seed-extension spec")
    if spec["spec_schema_version"] != SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported seed-extension spec schema")
    if spec["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("unexpected seed-extension auxiliary_id")
    if spec["primary_campaign"] != {
        "campaign_id": "binary-midpoint-main-v1",
        "training_seed": 0,
    }:
        raise ValueError("seed extension must remain anchored to primary seed 0")
    if spec["protocol"] != {
        "training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "gamma": EXPECTED_GAMMA,
        "m_values": list(EXPECTED_M_VALUES),
        "quadrature_rule": "midpoint-v1",
        "quadrature_seed": EXPECTED_QUADRATURE_SEED,
        "checkpoint_rule": "final_epoch_40",
    }:
        raise ValueError("seed-extension protocol differs from the frozen design")
    if spec["data_root"] != "data":
        raise ValueError("seed extension requires the locked data root 'data'")

    datasets = spec["datasets"]
    if not isinstance(datasets, list) or len(datasets) != len(EXPECTED_DATASETS):
        raise ValueError("seed extension requires exactly five dataset entries")
    for entry in datasets:
        _exact_fields(entry, _DATASET_FIELDS, location="spec.datasets[]")
        _positive_int(entry["train_count"], location="dataset.train_count")
        _positive_int(entry["eval_count"], location="dataset.eval_count")
        _digest(
            entry["train_sample_id_sha256"],
            location="dataset.train_sample_id_sha256",
        )
        _digest(
            entry["eval_sample_id_sha256"],
            location="dataset.eval_sample_id_sha256",
        )
        actual_protocol = (
            entry["train_split"],
            entry["eval_split"],
            entry["prompts"],
            entry["prompt_classes"],
            entry["freeze_batch_size"],
        )
        if actual_protocol != EXPECTED_DATASET_PROTOCOL.get(entry["name"]):
            raise ValueError(f"dataset protocol changed for {entry['name']}")
    if tuple(entry["name"] for entry in datasets) != EXPECTED_DATASETS:
        raise ValueError(
            "dataset order/content differs from the frozen five-dataset grid"
        )

    models = spec["models"]
    if not isinstance(models, list) or len(models) != len(EXPECTED_MODELS):
        raise ValueError("seed extension requires exactly two target architectures")
    for entry in models:
        _exact_fields(entry, _MODEL_FIELDS, location="spec.models[]")
        if entry != EXPECTED_MODEL_PROTOCOL.get(entry["name"]):
            raise ValueError(f"training schedule changed for {entry['name']}")
    if tuple(entry["name"] for entry in models) != EXPECTED_MODELS:
        raise ValueError("model order/content differs from the frozen two-model grid")
    if tuple(entry["condition"] for entry in models) != (
        "clipseg-target",
        "deeplabv3-target",
    ):
        raise ValueError("only target-adapted conditions belong in this extension")
    if spec["training_preprocessing"] != {
        "minimum_random_scale": 0.5,
        "maximum_random_scale": 2.0,
        "random_crop": True,
        "horizontal_flip_probability": 0.5,
        "negative_prompt_probability": 0.25,
        "drop_last": True,
    }:
        raise ValueError("training preprocessing differs from seed 0")
    if spec["freeze_preprocessing"] != {
        "model_input": "square resize with antialiasing",
        "probability_to_native_mask": (
            "bilinear interpolation with align_corners=False"
        ),
    }:
        raise ValueError("freeze preprocessing differs from seed 0")
    if spec["environment"] != {
        "python": "3.12.4",
        "numpy": "2.5.1",
        "torch": "2.12.1",
        "torchvision": "0.27.1",
        "transformers": "5.13.0",
        "cuda_runtime": "13.0",
    }:
        raise ValueError("seed-extension environment is not the locked seed-0 stack")

    profiles = spec["gpu_profiles"]
    if not isinstance(profiles, list):
        raise ValueError("gpu_profiles must be a list")
    profile_values = []
    for entry in profiles:
        _exact_fields(
            entry,
            frozenset({"partition", "account", "gres"}),
            location="spec.gpu_profiles[]",
        )
        profile_values.append((entry["partition"], entry["account"], entry["gres"]))
    if tuple(profile_values) != EXPECTED_GPU_PROFILES:
        raise ValueError("GPU account/GRES profiles differ from the validated tuples")
    if spec["partition_assignment"] != "balanced_dataset_model_seed_parity_v1":
        raise ValueError("unexpected partition-assignment rule")

    paths = spec["paths"]
    expected_path_fields = frozenset(
        {
            "train_root",
            "checkpoint_lock",
            "artifact_root",
            "freeze_record_root",
            "common_root",
            "simulation_root",
            "assembly_root",
            "diagnostic_root",
            "analysis_root",
        }
    )
    _exact_fields(paths, expected_path_fields, location="spec.paths")
    for key, value in paths.items():
        path = _portable_path(value, location=f"spec.paths.{key}")
        if path.parts[0] != "outputs" or "seed" not in value:
            raise ValueError(
                f"extension path is not isolated from primary outputs: {value}"
            )
    if spec["downstream_job_design"] != {
        "common_jobs": 20,
        "m_score_jobs": 60,
        "assembly_jobs": 20,
        "diagnostic_jobs": 20,
        "analysis_jobs": 1,
        "render_jobs": 1,
    }:
        raise ValueError("downstream job counts differ from the predeclared design")


def load_spec_lock(path=DEFAULT_SPEC_LOCK, *, expected_sha256=None):
    """Load and rehash the immutable extension spec and all small dependencies."""

    source = Path(path)
    actual_lock_sha256 = _sha256(source)
    expected = expected_sha256
    if expected is None and source.as_posix() == DEFAULT_SPEC_LOCK:
        expected = DEFAULT_SPEC_LOCK_SHA256
    if expected is None:
        raise ValueError("a non-default spec lock requires --expected-spec-lock-sha256")
    if actual_lock_sha256 != _digest(expected, location="expected spec-lock sha256"):
        raise ValueError("seed-extension spec-lock hash mismatch")
    lock = _load_json(source)
    _exact_fields(lock, _SPEC_LOCK_FIELDS, location="seed-extension spec lock")
    if lock["lock_schema_version"] != SPEC_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported seed-extension spec-lock schema")
    if lock["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("unexpected spec-lock auxiliary_id")

    spec_path = _verify_file_binding(lock["spec"], location="spec_lock.spec")
    spec = _load_json(spec_path)
    _validate_spec(spec)

    canonical_path = _verify_file_binding(
        {
            "path": lock["canonical_campaign_lock"].get("path"),
            "sha256": lock["canonical_campaign_lock"].get("sha256"),
        },
        location="spec_lock.canonical_campaign_lock",
    )
    if set(lock["canonical_campaign_lock"]) != {"path", "sha256", "campaign_id"}:
        raise ValueError("canonical_campaign_lock contains unexpected fields")
    canonical = _load_json(canonical_path)
    if (
        lock["canonical_campaign_lock"]["campaign_id"] != "binary-midpoint-main-v1"
        or canonical.get("campaign_id") != "binary-midpoint-main-v1"
    ):
        raise ValueError("extension is not bound to the canonical primary campaign")
    if canonical.get("protocol", {}).get("seeds") != [0]:
        raise ValueError("canonical campaign is no longer the locked seed-0 campaign")

    estimator_path = _verify_file_binding(
        lock["estimator_spec"], location="spec_lock.estimator_spec"
    )
    estimator = _load_json(estimator_path)
    if (
        estimator.get("estimator_id") != "midpoint-v1"
        or estimator.get("required_seed") != 0
    ):
        raise ValueError("downstream estimator is not the deterministic midpoint rule")

    for index, binding in enumerate(lock["source_files"]):
        _verify_file_binding(binding, location=f"spec_lock.source_files[{index}]")
    if {entry["path"] for entry in lock["source_files"]} != {
        "selectseg/train.py",
        "selectseg/data.py",
        "selectseg/models.py",
        "selectseg/freeze_binary_maps.py",
        "selectseg/binary_artifacts.py",
    }:
        raise ValueError("spec lock must bind the exact training/freeze source set")
    for index, binding in enumerate(lock["reference_seed0_train_configs"]):
        _exact_fields(
            binding,
            frozenset({"dataset", "model", "path", "sha256"}),
            location=f"reference_seed0_train_configs[{index}]",
        )
        _verify_file_binding(
            {"path": binding["path"], "sha256": binding["sha256"]},
            location=f"reference_seed0_train_configs[{index}]",
        )
    expected_references = {
        (dataset, model) for dataset in EXPECTED_DATASETS for model in EXPECTED_MODELS
    }
    actual_references = {
        (entry["dataset"], entry["model"])
        for entry in lock["reference_seed0_train_configs"]
    }
    if actual_references != expected_references or len(actual_references) != 10:
        raise ValueError("spec lock must bind exactly ten primary training configs")

    validation = lock["scheduler_validation"]
    _exact_fields(
        validation,
        frozenset(
            {"checked_date", "method", "profiles", "local_configuration_evidence"}
        ),
        location="spec_lock.scheduler_validation",
    )
    scheduler_profiles = tuple(
        (entry["partition"], entry["account"], entry["gres"])
        for entry in validation["profiles"]
        if isinstance(entry, dict)
        and set(entry) == {"partition", "account", "gres", "result"}
        and entry["result"] == "accepted"
    )
    if scheduler_profiles != EXPECTED_GPU_PROFILES:
        raise ValueError("both GPU profiles must have accepted scheduler validation")

    if not isinstance(lock["base_model_files"], list) or not lock["base_model_files"]:
        raise ValueError("base_model_files cannot be empty")
    for index, binding in enumerate(lock["base_model_files"]):
        _exact_fields(
            binding,
            frozenset({"path", "sha256"}),
            location=f"spec_lock.base_model_files[{index}]",
        )
        _portable_path(
            binding["path"], location=f"spec_lock.base_model_files[{index}].path"
        )
        _digest(
            binding["sha256"],
            location=f"spec_lock.base_model_files[{index}].sha256",
        )
    if len(lock["base_model_files"]) != 8:
        raise ValueError("spec lock must bind seven CLIPSeg files and one DeepLab file")

    return {
        "path": source,
        "sha256": actual_lock_sha256,
        "lock": lock,
        "spec": spec,
        "canonical": canonical,
    }


def _dataset_entry(spec, dataset):
    matches = [entry for entry in spec["datasets"] if entry["name"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} is absent or duplicated in the spec")
    return matches[0]


def _model_entry(spec, model):
    matches = [entry for entry in spec["models"] if entry["name"] == model]
    if len(matches) != 1:
        raise ValueError(f"model {model!r} is absent or duplicated in the spec")
    return matches[0]


def iter_experiments(spec):
    """Yield the locked 5 x 2 x 2 grid in deterministic order."""

    for dataset_index, dataset in enumerate(spec["datasets"]):
        for model_index, model in enumerate(spec["models"]):
            for seed_index, seed in enumerate(spec["protocol"]["training_seeds"]):
                profile_index = (dataset_index + model_index + seed_index) % 2
                yield {
                    "dataset": dataset,
                    "model": model,
                    "training_seed": seed,
                    "gpu_profile": spec["gpu_profiles"][profile_index],
                }


def _experiment(binding, dataset, model, seed):
    if isinstance(seed, bool) or seed not in EXPECTED_TRAINING_SEEDS:
        raise ValueError("extension training seed must be exactly 1 or 2")
    matches = [
        experiment
        for experiment in iter_experiments(binding["spec"])
        if experiment["dataset"]["name"] == dataset
        and experiment["model"]["name"] == model
        and experiment["training_seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("requested experiment is absent or duplicated")
    return matches[0]


def _verify_base_model_files(binding, model):
    relevant = []
    for entry in binding["lock"]["base_model_files"]:
        path = entry["path"]
        if model == "clipseg" and "models--CIDAS--clipseg" in path:
            relevant.append(entry)
        if model == "deeplabv3" and "deeplabv3_resnet50" in path:
            relevant.append(entry)
    if not relevant:
        raise ValueError(f"no locked base-model files found for {model}")
    for index, entry in enumerate(relevant):
        _verify_file_binding(entry, location=f"base_model_files[{model}][{index}]")


def _reference_config(binding, experiment):
    dataset = experiment["dataset"]["name"]
    model = experiment["model"]["name"]
    matches = [
        entry
        for entry in binding["lock"]["reference_seed0_train_configs"]
        if entry["dataset"] == dataset and entry["model"] == model
    ]
    if len(matches) != 1:
        raise ValueError("missing unique primary training-config reference")
    config = _load_json(matches[0]["path"])
    model_spec = experiment["model"]
    expected = {
        "model": model,
        "dataset": dataset,
        "data_root": binding["spec"]["data_root"],
        "output_dir": f"outputs/binary_train/{dataset}/{model}/seed-0",
        "epochs": str(model_spec["epochs"]),
        "batch_size": str(model_spec["batch_size"]),
        "lr": str(model_spec["learning_rate"]),
        "weight_decay": str(model_spec["weight_decay"]),
        "num_workers": str(model_spec["num_workers"]),
        "seed": "0",
        "limit_batches": "None",
    }
    if config != expected:
        raise ValueError(
            f"locked seed-0 config for {dataset}/{model} no longer matches the extension"
        )
    return matches[0]


def _paired_stems(images_dir, masks_dir, *, mask_suffix=""):
    suffixes = {".png", ".jpg", ".jpeg"}
    images = {
        path.stem
        for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    masks = {
        path.stem.removesuffix(mask_suffix)
        for path in Path(masks_dir).iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if images != masks:
        raise ValueError(
            f"image/mask sample identities differ: {images_dir}, {masks_dir}"
        )
    return images


def _sample_ids(dataset, split, data_root):
    root = Path(data_root)
    if dataset == "pet":
        split_file = root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
        return [
            line.split()[0]
            for line in split_file.read_text().splitlines()
            if line and not line.startswith("#")
        ]
    if dataset == "kvasir":
        stems = _paired_stems(
            root / "Kvasir-SEG" / "images", root / "Kvasir-SEG" / "masks"
        )
        ranked = sorted(
            stems,
            key=lambda stem: (hashlib.sha256(stem.encode()).digest(), stem),
        )
        train_count = 4 * len(ranked) // 5
        selected = ranked[:train_count] if split == "train" else ranked[train_count:]
        return sorted(selected)
    if dataset == "fives":
        split_root = root / "FIVES" / split
        return sorted(
            _paired_stems(split_root / "Original", split_root / "Ground truth")
        )
    if dataset == "isic":
        release = "Training" if split == "train" else "Test"
        base = root / "ISIC2018"
        return sorted(
            _paired_stems(
                base / f"ISIC2018_Task1-2_{release}_Input",
                base / f"ISIC2018_Task1_{release}_GroundTruth",
                mask_suffix="_segmentation",
            )
        )
    if dataset == "tn3k":
        release = "trainval" if split == "train" else "test"
        extracted = root / "TN3K" / "extracted" / "Thyroid Dataset" / "tn3k"
        normalized = root / "TN3K" / "tn3k"
        base = extracted if extracted.is_dir() else normalized
        return sorted(
            _paired_stems(base / f"{release}-image", base / f"{release}-mask")
        )
    raise ValueError(f"unsupported seed-extension dataset {dataset!r}")


def _verify_cohorts(binding, dataset_entry):
    for kind in ("train", "eval"):
        split = dataset_entry[f"{kind}_split"]
        ids = _sample_ids(dataset_entry["name"], split, binding["spec"]["data_root"])
        actual_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if len(ids) != dataset_entry[f"{kind}_count"]:
            raise ValueError(f"{dataset_entry['name']} {kind} cohort count changed")
        if actual_hash != dataset_entry[f"{kind}_sample_id_sha256"]:
            raise ValueError(f"{dataset_entry['name']} {kind} cohort identity changed")


def _environment_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "torchvision", "transformers"):
        versions[package] = importlib.metadata.version(package)
    import torch

    versions["cuda_runtime"] = torch.version.cuda
    return versions


def _verify_environment(spec):
    actual = _environment_versions()
    if actual != spec["environment"]:
        raise ValueError(
            f"runtime environment differs from the locked seed-0 environment: {actual}"
        )
    return actual


def _require_cuda():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("seed-extension training requires an allocated CUDA GPU")


def _verify_slurm_context(profile):
    if not os.environ.get("SLURM_JOB_ID"):
        return
    if os.environ.get("SLURM_JOB_PARTITION") != profile["partition"]:
        raise RuntimeError("Slurm partition differs from the planned GPU profile")
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    if account is not None and account != profile["account"]:
        raise RuntimeError("Slurm account differs from the planned GPU profile")


def _atomic_write_new(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
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
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _train_output_dir(spec, dataset, model, seed):
    return Path(spec["paths"]["train_root"]) / dataset / model / f"seed-{seed}"


def _training_command(binding, experiment):
    dataset = experiment["dataset"]["name"]
    model = experiment["model"]["name"]
    seed = experiment["training_seed"]
    model_spec = experiment["model"]
    output_dir = _train_output_dir(binding["spec"], dataset, model, seed)
    return [
        sys.executable,
        "-m",
        "selectseg.train",
        "--model",
        model,
        "--dataset",
        dataset,
        "--data-root",
        binding["spec"]["data_root"],
        "--output-dir",
        output_dir.as_posix(),
        "--epochs",
        str(model_spec["epochs"]),
        "--batch-size",
        str(model_spec["batch_size"]),
        "--lr",
        str(model_spec["learning_rate"]),
        "--weight-decay",
        str(model_spec["weight_decay"]),
        "--num-workers",
        str(model_spec["num_workers"]),
        "--seed",
        str(seed),
    ]


def _validate_training_outputs(binding, experiment, command):
    dataset = experiment["dataset"]["name"]
    model = experiment["model"]["name"]
    seed = experiment["training_seed"]
    model_spec = experiment["model"]
    output_dir = _train_output_dir(binding["spec"], dataset, model, seed)
    config_path = output_dir / "train_config.json"
    history_path = output_dir / "history.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    for path in (config_path, history_path, checkpoint_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"training output is missing, empty, or a symlink: {path}")
    expected_config = {
        "model": model,
        "dataset": dataset,
        "data_root": binding["spec"]["data_root"],
        "output_dir": output_dir.as_posix(),
        "epochs": str(model_spec["epochs"]),
        "batch_size": str(model_spec["batch_size"]),
        "lr": str(model_spec["learning_rate"]),
        "weight_decay": str(model_spec["weight_decay"]),
        "num_workers": str(model_spec["num_workers"]),
        "seed": str(seed),
        "limit_batches": "None",
    }
    if _load_json(config_path) != expected_config:
        raise ValueError(
            f"training config differs from the locked command: {config_path}"
        )
    history = _load_json(history_path)
    if not isinstance(history, list) or len(history) != model_spec["epochs"]:
        raise ValueError("history does not contain exactly 40 completed epochs")
    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "mean_training_loss",
            "learning_rate",
        }:
            raise ValueError("training history has an unexpected schema")
        if row["epoch"] != index:
            raise ValueError(
                "training history is not a complete ordered epoch sequence"
            )
        for field in ("mean_training_loss", "learning_rate"):
            if not isinstance(row[field], (int, float)) or not math.isfinite(
                row[field]
            ):
                raise ValueError(f"training history contains non-finite {field}")
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "train_config_path": config_path.as_posix(),
        "train_config_sha256": _sha256(config_path),
        "history_path": history_path.as_posix(),
        "history_sha256": _sha256(history_path),
        "command": command,
    }


def run_training(binding, *, dataset, model, seed, expected_partition):
    experiment = _experiment(binding, dataset, model, seed)
    profile = experiment["gpu_profile"]
    if expected_partition != profile["partition"]:
        raise ValueError(
            "requested partition differs from the balanced locked assignment"
        )
    _verify_slurm_context(profile)
    _reference_config(binding, experiment)
    _verify_base_model_files(binding, model)
    _verify_cohorts(binding, experiment["dataset"])
    environment = _verify_environment(binding["spec"])
    _require_cuda()
    output_dir = _train_output_dir(binding["spec"], dataset, model, seed)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to reuse training output; quarantine it first: {output_dir}"
        )
    command = _training_command(binding, experiment)
    subprocess.run(command, check=True)
    outputs = _validate_training_outputs(binding, experiment, command)
    import torch

    record = {
        "train_record_schema_version": TRAIN_RECORD_SCHEMA_VERSION,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_lock": {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
        },
        "dataset": dataset,
        "model": model,
        "condition": experiment["model"]["condition"],
        "training_seed": seed,
        "gpu_profile": profile,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", expected_partition),
            "node": os.environ.get("SLURMD_NODENAME"),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "environment": environment,
        },
        "outputs": outputs,
    }
    record_path = output_dir / "extension_record.json"
    _atomic_write_new(record_path, record)
    print(f"saved {record_path}")
    return record_path


def _load_train_record(binding, experiment):
    dataset = experiment["dataset"]["name"]
    model = experiment["model"]["name"]
    seed = experiment["training_seed"]
    output_dir = _train_output_dir(binding["spec"], dataset, model, seed)
    record_path = output_dir / "extension_record.json"
    record = _load_json(record_path)
    required = {
        "train_record_schema_version",
        "auxiliary_id",
        "created_utc",
        "spec_lock",
        "dataset",
        "model",
        "condition",
        "training_seed",
        "gpu_profile",
        "runtime",
        "outputs",
    }
    _exact_fields(record, frozenset(required), location=f"train record {record_path}")
    if record["train_record_schema_version"] != TRAIN_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported training-record schema")
    if record["spec_lock"] != {
        "path": binding["path"].as_posix(),
        "sha256": binding["sha256"],
    }:
        raise ValueError("training record is bound to a different spec lock")
    if (record["dataset"], record["model"], record["training_seed"]) != (
        dataset,
        model,
        seed,
    ):
        raise ValueError("training record identity differs from its expected cell")
    expected_command = _training_command(binding, experiment)
    outputs = _validate_training_outputs(binding, experiment, expected_command)
    if record["outputs"] != outputs:
        raise ValueError("training output hashes differ from the immutable record")
    return record_path, record, outputs


def _checkpoint_entry_from_training_record(binding, experiment):
    """Reconstruct one canonical lock row from its predictable train record."""

    record_path, _, outputs = _load_train_record(binding, experiment)
    return {
        "dataset": experiment["dataset"]["name"],
        "model": experiment["model"]["name"],
        "condition": experiment["model"]["condition"],
        "training_seed": experiment["training_seed"],
        "checkpoint_path": outputs["checkpoint_path"],
        "checkpoint_sha256": outputs["checkpoint_sha256"],
        "checkpoint_size_bytes": outputs["checkpoint_size_bytes"],
        "train_config_path": outputs["train_config_path"],
        "train_config_sha256": outputs["train_config_sha256"],
        "history_path": outputs["history_path"],
        "history_sha256": outputs["history_sha256"],
        "train_record_path": record_path.as_posix(),
        "train_record_sha256": _sha256(record_path),
    }


def write_checkpoint_lock(binding, output_path=None):
    """Validate all 20 final checkpoints and atomically publish their lock."""

    destination = Path(output_path or binding["spec"]["paths"]["checkpoint_lock"])
    expected_destination = Path(binding["spec"]["paths"]["checkpoint_lock"])
    if destination != expected_destination:
        raise ValueError(f"checkpoint lock must be written to {expected_destination}")
    checkpoints = [
        _checkpoint_entry_from_training_record(binding, experiment)
        for experiment in iter_experiments(binding["spec"])
    ]
    if len(checkpoints) != 20:
        raise RuntimeError("checkpoint lock does not contain exactly 20 experiments")
    payload = {
        "checkpoint_lock_schema_version": CHECKPOINT_LOCK_SCHEMA_VERSION,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_lock": {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
        },
        "checkpoints": checkpoints,
    }
    _atomic_write_new(destination, payload)
    print(f"saved {destination}")
    print(f"checkpoint_lock_sha256={_sha256(destination)}")
    return destination


def load_checkpoint_lock(binding, path, *, expected_sha256=None, verify_files=True):
    source = Path(path)
    actual_sha256 = _sha256(source)
    if expected_sha256 is not None and actual_sha256 != _digest(
        expected_sha256, location="expected checkpoint-lock sha256"
    ):
        raise ValueError("checkpoint-lock hash mismatch")
    lock = _load_json(source)
    _exact_fields(lock, _CHECKPOINT_LOCK_FIELDS, location="checkpoint lock")
    if lock["checkpoint_lock_schema_version"] != CHECKPOINT_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-lock schema")
    if lock["auxiliary_id"] != EXPECTED_AUXILIARY_ID:
        raise ValueError("checkpoint lock has the wrong auxiliary_id")
    if lock["spec_lock"] != {
        "path": binding["path"].as_posix(),
        "sha256": binding["sha256"],
    }:
        raise ValueError("checkpoint lock is bound to a different spec lock")
    expected_experiments = {
        (
            experiment["dataset"]["name"],
            experiment["model"]["name"],
            experiment["training_seed"],
        ): experiment
        for experiment in iter_experiments(binding["spec"])
    }
    expected_cells = set(expected_experiments)
    actual_cells = set()
    for index, entry in enumerate(lock["checkpoints"]):
        _exact_fields(
            entry,
            _CHECKPOINT_FIELDS,
            location=f"checkpoint_lock.checkpoints[{index}]",
        )
        cell = (entry["dataset"], entry["model"], entry["training_seed"])
        if cell in actual_cells:
            raise ValueError(f"duplicate checkpoint cell {cell}")
        experiment = expected_experiments.get(cell)
        if experiment is None:
            raise ValueError(f"checkpoint cell {cell} is outside the locked grid")
        if entry["condition"] != experiment["model"]["condition"]:
            raise ValueError(f"checkpoint cell {cell} has an unexpected condition")
        actual_cells.add(cell)
        _positive_int(entry["checkpoint_size_bytes"], location="checkpoint_size_bytes")
        for field in (
            "checkpoint_sha256",
            "train_config_sha256",
            "history_sha256",
            "train_record_sha256",
        ):
            _digest(entry[field], location=f"checkpoint.{field}")
        if verify_files:
            # Reconstruct the entry from the predictable training directory and
            # immutable extension record.  Hashing files named by the lock alone
            # would allow a self-consistent but retargeted lock to pass if an
            # operator accidentally supplied that lock's newly computed digest.
            expected_entry = _checkpoint_entry_from_training_record(binding, experiment)
            if entry != expected_entry:
                raise ValueError(
                    f"checkpoint cell {cell} differs from its immutable training record"
                )
    if actual_cells != expected_cells or len(actual_cells) != 20:
        raise ValueError(
            "checkpoint lock must contain the exact 20-cell extension grid"
        )
    return {"path": source, "sha256": actual_sha256, "lock": lock}


def _checkpoint_entry(checkpoint_binding, dataset, model, seed):
    matches = [
        entry
        for entry in checkpoint_binding["lock"]["checkpoints"]
        if (entry["dataset"], entry["model"], entry["training_seed"])
        == (dataset, model, seed)
    ]
    if len(matches) != 1:
        raise ValueError("missing unique checkpoint entry")
    return matches[0]


def run_freeze(
    binding,
    checkpoint_binding,
    *,
    dataset,
    model,
    seed,
    expected_partition,
):
    experiment = _experiment(binding, dataset, model, seed)
    profile = experiment["gpu_profile"]
    if expected_partition != profile["partition"]:
        raise ValueError(
            "requested partition differs from the balanced locked assignment"
        )
    _verify_slurm_context(profile)
    _verify_environment(binding["spec"])
    _verify_cohorts(binding, experiment["dataset"])
    entry = _checkpoint_entry(checkpoint_binding, dataset, model, seed)
    expected_entry = _checkpoint_entry_from_training_record(binding, experiment)
    if entry != expected_entry:
        raise ValueError("target checkpoint differs from its immutable training record")

    record_path = (
        Path(binding["spec"]["paths"]["freeze_record_root"])
        / dataset
        / model
        / f"seed-{seed}.json"
    )
    if record_path.exists() or record_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite freeze record: {record_path}")
    from selectseg.freeze_binary_maps import main as freeze_main

    arguments = [
        "--model",
        model,
        "--dataset",
        dataset,
        "--data-root",
        binding["spec"]["data_root"],
        "--checkpoint",
        entry["checkpoint_path"],
        "--output-dir",
        binding["spec"]["paths"]["artifact_root"],
        "--batch-size",
        str(experiment["dataset"]["freeze_batch_size"]),
        "--num-workers",
        "4",
        "--expected-num-samples",
        str(experiment["dataset"]["eval_count"]),
        "--require-cuda",
    ]
    manifest_path = freeze_main(arguments)
    artifact = load_binary_artifact(manifest_path, validate_payloads=False)
    manifest = artifact.manifest
    if (
        manifest["dataset"] != dataset
        or manifest["model"] != model
        or manifest["condition"] != experiment["model"]["condition"]
        or manifest["num_samples"] != experiment["dataset"]["eval_count"]
        or manifest["sample_id_sha256"]
        != experiment["dataset"]["eval_sample_id_sha256"]
        or manifest["checkpoint"]["sha256"] != entry["checkpoint_sha256"]
    ):
        raise ValueError("frozen artifact does not match its seed-extension cell")
    record = {
        "freeze_record_schema_version": FREEZE_RECORD_SCHEMA_VERSION,
        "auxiliary_id": EXPECTED_AUXILIARY_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_lock": {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
        },
        "checkpoint_lock": {
            "path": checkpoint_binding["path"].as_posix(),
            "sha256": checkpoint_binding["sha256"],
        },
        "dataset": dataset,
        "model": model,
        "condition": experiment["model"]["condition"],
        "training_seed": seed,
        "gpu_profile": profile,
        "artifact_manifest_path": Path(manifest_path).as_posix(),
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "checkpoint_sha256": entry["checkpoint_sha256"],
    }
    _atomic_write_new(record_path, record)
    print(f"saved {record_path}")
    return record_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("train", "freeze"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--spec-lock", default=DEFAULT_SPEC_LOCK)
        subparser.add_argument("--expected-spec-lock-sha256", required=True)
        subparser.add_argument("--dataset", required=True, choices=EXPECTED_DATASETS)
        subparser.add_argument("--model", required=True, choices=EXPECTED_MODELS)
        subparser.add_argument("--training-seed", required=True, type=int)
        subparser.add_argument("--expected-partition", required=True)
    freeze_parser = subparsers.choices["freeze"]
    freeze_parser.add_argument("--checkpoint-lock", required=True)
    freeze_parser.add_argument("--expected-checkpoint-lock-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    binding = load_spec_lock(
        args.spec_lock, expected_sha256=args.expected_spec_lock_sha256
    )
    if args.phase == "train":
        return run_training(
            binding,
            dataset=args.dataset,
            model=args.model,
            seed=args.training_seed,
            expected_partition=args.expected_partition,
        )
    checkpoint_binding = load_checkpoint_lock(
        binding,
        args.checkpoint_lock,
        expected_sha256=args.expected_checkpoint_lock_sha256,
        verify_files=False,
    )
    return run_freeze(
        binding,
        checkpoint_binding,
        dataset=args.dataset,
        model=args.model,
        seed=args.training_seed,
        expected_partition=args.expected_partition,
    )


if __name__ == "__main__":
    main()
