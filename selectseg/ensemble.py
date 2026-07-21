"""Build a frozen artifact by averaging locked training-seed probabilities.

The builder consumes one immutable source lock and exactly one dataset/model
cell.  It streams aligned samples from seeds 0, 1, and 2, checks byte-identical
truth masks, accumulates probabilities in float64, and atomically publishes a
standard frozen binary artifact.  Downstream scorers therefore need no special
ensemble code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import (
    load_binary_artifact,
    sha256_file,
    write_binary_artifact,
)


SOURCE_LOCK_SCHEMA_VERSION = 1
SOURCE_LOCK_ARTIFACT_TYPE = "selectseg.ensemble_source_lock"
TRAINING_SEEDS = (0, 1, 2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--expected-source-lock-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=("clipseg-target", "deeplabv3-target"),
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "torchvision", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "selectseg/__init__.py",
        "selectseg/artifacts.py",
        "selectseg/ensemble.py",
        "scripts/slurm/run.sbatch",
        "scripts/slurm/env.sh",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _select_cell(source_lock: dict, dataset: str, condition: str) -> dict:
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported ensemble source-lock schema")
    if source_lock.get("artifact_type") != SOURCE_LOCK_ARTIFACT_TYPE:
        raise ValueError("unexpected ensemble source-lock artifact type")
    cells = source_lock.get("cells")
    if not isinstance(cells, list):
        raise ValueError("source_lock.cells must be a list")
    matches = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("dataset") == dataset
        and cell.get("condition") == condition
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one locked cell for {dataset}/{condition}, got {len(matches)}"
        )
    return matches[0]


def _load_sources(cell: dict):
    sources = cell.get("sources")
    if not isinstance(sources, list) or [
        item.get("training_seed") for item in sources
    ] != list(TRAINING_SEEDS):
        raise ValueError("locked sources must be ordered training seeds [0, 1, 2]")

    artifacts = []
    for source in sources:
        manifest_path = Path(source["manifest_path"])
        expected_hash = source["manifest_sha256"]
        if sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"source manifest SHA-256 mismatch: {manifest_path}")
        artifact = load_binary_artifact(manifest_path, validate_payloads=False)
        if artifact.manifest_sha256 != expected_hash:
            raise ValueError(f"source manifest changed while loading: {manifest_path}")
        for field in (
            "artifact_id",
            "dataset",
            "condition",
            "model",
            "split",
            "num_samples",
            "sample_id_sha256",
        ):
            if artifact.manifest[field] != source[field]:
                raise ValueError(f"locked {field} mismatch: {manifest_path}")
        artifacts.append(artifact)
    return tuple(artifacts)


def _validate_shared_metadata(cell: dict, artifacts) -> None:
    reference = artifacts[0].manifest
    expected = {
        "dataset": cell["dataset"],
        "condition": cell["condition"],
        "model": cell["model"],
        "num_samples": cell["expected_num_samples"],
    }
    for artifact in artifacts:
        for field, value in expected.items():
            if artifact.manifest[field] != value:
                raise ValueError(
                    f"source {artifact.manifest_path} has {field}={artifact.manifest[field]!r}; "
                    f"expected {value!r}"
                )
        for field in (
            "split",
            "class_index",
            "class_name",
            "sample_id_sha256",
            "preprocessing",
            "cohort",
        ):
            if artifact.manifest[field] != reference[field]:
                raise ValueError(f"ensemble sources disagree on {field}")


def load_locked_ensemble_sources(
    source_lock_path: str | Path,
    expected_source_lock_sha256: str,
    dataset: str,
    condition: str,
):
    """Load and validate one immutable three-seed ensemble cell.

    Both the probability-mean builder and downstream empirical-posterior
    scorers use this single validation path, so source identity and sample
    alignment cannot drift between auxiliary experiments.
    """

    lock_path = Path(source_lock_path)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise FileNotFoundError(f"source lock is not a regular file: {lock_path}")
    lock_hash = sha256_file(lock_path)
    if lock_hash != expected_source_lock_sha256:
        raise ValueError("ensemble source-lock SHA-256 differs from the planned job")
    source_lock = _load_json(lock_path)
    cell = _select_cell(source_lock, dataset, condition)
    artifacts = _load_sources(cell)
    _validate_shared_metadata(cell, artifacts)
    return source_lock, cell, artifacts


def _ensemble_samples(artifacts):
    iterators = [artifact.iter_samples() for artifact in artifacts]
    for samples in zip(*iterators, strict=True):
        reference = samples[0]
        for sample in samples[1:]:
            if (
                sample.index != reference.index
                or sample.sample_id != reference.sample_id
            ):
                raise ValueError("ensemble source sample order changed")
            if (
                sample.foreground_probability.shape
                != reference.foreground_probability.shape
            ):
                raise ValueError(
                    f"probability shape mismatch for {reference.sample_id}"
                )
            if not np.array_equal(sample.truth, reference.truth):
                raise ValueError(f"truth mismatch for {reference.sample_id}")
        probability = np.zeros(reference.foreground_probability.shape, dtype=np.float64)
        for sample in samples:
            probability += sample.foreground_probability
        probability = np.asarray(probability / len(samples), dtype=np.float32)
        yield reference.sample_id, probability, reference.truth


def build_ensemble_artifact(args, argv=None) -> Path:
    lock_path = Path(args.source_lock)
    lock_hash = sha256_file(lock_path)
    _, cell, artifacts = load_locked_ensemble_sources(
        lock_path,
        args.expected_source_lock_sha256,
        args.dataset,
        args.condition,
    )

    reference = artifacts[0].manifest
    sample_ids = [entry["sample_id"] for entry in reference["samples"]]
    lock_size = lock_path.stat().st_size
    if lock_size <= 0:
        raise ValueError("ensemble source lock cannot be empty")
    command_argv = list(sys.argv[1:] if argv is None else argv)
    return write_binary_artifact(
        args.output_root,
        dataset=cell["dataset"],
        condition=cell["condition"],
        model=cell["model"],
        split=reference["split"],
        class_index=reference["class_index"],
        class_name=reference["class_name"],
        checkpoint={
            "path": lock_path.name,
            "sha256": lock_hash,
            "size_bytes": lock_size,
        },
        base_model={
            "name": f"{cell['model']}-three-seed-probability-ensemble",
            "source": "arithmetic mean of locked training seeds 0, 1, and 2",
        },
        source_sha256=_source_fingerprint(),
        environment={
            "packages": _package_versions(),
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        preprocessing=reference["preprocessing"],
        cohort=reference["cohort"],
        sample_ids=sample_ids,
        samples=_ensemble_samples(artifacts),
        command=[
            "python",
            "-m",
            "selectseg.ensemble",
            *command_argv,
        ],
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def main(argv=None):
    args = parse_args(argv)
    manifest_path = build_ensemble_artifact(args, argv=argv)
    print(f"saved {manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
