"""Score one M-specific binary threshold simulation from a frozen artifact.

One invocation binds one immutable campaign lock, one frozen probability-map
artifact, one estimator specification, one deployment threshold ``gamma``, one
quadrature size ``M``, and one seed.  It never performs model inference.  The
partial JSONL contains stable identity fields plus only the loss-indexed
confidence fields belonging to the requested ``M``.  Risks, auxiliary values,
and common baselines are produced once by :mod:`selectseg.score_binary_common`.

This module deliberately imports neither PyTorch nor any inference code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import string
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from selectseg.binary_artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
)
from selectseg.binary_boundary import prepare_boundary_reference
from selectseg.score_binary_common import (
    AUXILIARY_FIELDS,
    COMMON_SCORE_FIELDS,
    IDENTITY_ROW_FIELDS,
    M_SCORE_FIELDS,
    RISK_FIELDS,
    ROW_SCHEMA_VERSION,
    SIMULATION_ARTIFACT_TYPE,
)
from selectseg.threshold_estimators import (
    ThresholdRule,
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


SIMULATION_SCHEMA_VERSION = ROW_SCHEMA_VERSION
ARTIFACT_TYPE = SIMULATION_ARTIFACT_TYPE
__all__ = (
    "AUXILIARY_FIELDS",
    "COMMON_SCORE_FIELDS",
    "M_SCORE_FIELDS",
    "RISK_FIELDS",
    "parse_args",
    "run_simulation",
    "score_binary_sample",
)


class _StoreExactlyOnce(argparse.Action):
    """Reject repeated scalar simulation axes instead of silently taking the last."""

    def __call__(self, parser, namespace, values, option_string=None):
        marker = f"_seen_{self.dest}"
        if getattr(namespace, marker, False):
            raise argparse.ArgumentError(
                self, f"{option_string} may be supplied exactly once"
            )
        setattr(namespace, marker, True)
        setattr(namespace, self.dest, values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument("--expected-campaign-lock-sha256", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    parser.add_argument("--estimator-spec", required=True)
    parser.add_argument("--expected-estimator-spec-sha256", required=True)
    parser.add_argument("--gamma", required=True, type=float, action=_StoreExactlyOnce)
    parser.add_argument("--m", required=True, type=int, action=_StoreExactlyOnce)
    parser.add_argument("--seed", required=True, type=int, action=_StoreExactlyOnce)
    parser.add_argument("--output-root", default="outputs/binary_simulations")
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--max-pending-scores", type=int, default=16)
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _strict_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in value):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return normalized


def _validate_args(args):
    if not isinstance(args.campaign_id, str) or not args.campaign_id.strip():
        raise ValueError("--campaign-id must be a non-empty string")
    if not math.isfinite(args.gamma) or not 0 < args.gamma < 1:
        raise ValueError("--gamma must be finite and lie strictly inside (0, 1)")
    if isinstance(args.m, bool) or args.m not in M_SCORE_FIELDS:
        raise ValueError(f"--m must be one of {tuple(M_SCORE_FIELDS)}")
    if isinstance(args.seed, bool):
        raise TypeError("--seed must be an integer")
    if args.score_workers <= 0:
        raise ValueError("--score-workers must be positive")
    if args.max_pending_scores < args.score_workers:
        raise ValueError("--max-pending-scores must be at least --score-workers")
    for attribute in (
        "expected_campaign_lock_sha256",
        "expected_artifact_manifest_sha256",
        "expected_estimator_spec_sha256",
    ):
        setattr(
            args,
            attribute,
            _strict_sha256(getattr(args, attribute), name=f"--{attribute}"),
        )


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _verify_file(path: str | Path, expected_sha256: str, *, name: str) -> str:
    resolved = Path(path).resolve()
    unresolved = Path(path)
    if not unresolved.is_file() or unresolved.is_symlink():
        raise FileNotFoundError(
            f"{name} must be a regular, non-symlink file: {unresolved}"
        )
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _source_fingerprint() -> str:
    """Hash every local module that defines a partial simulation row."""

    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "selectseg/score_binary_simulation.py",
        "selectseg/score_binary_common.py",
        "selectseg/threshold_estimators.py",
        "selectseg/binary_artifacts.py",
        "selectseg/binary_boundary.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source dependency is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _m_score_fields(m: int) -> tuple[str, str, str]:
    try:
        return M_SCORE_FIELDS[m]
    except KeyError as error:
        raise ValueError(f"unsupported predeclared M={m}") from error


def score_binary_sample(
    sample,
    *,
    simulation_id: str,
    class_index: int,
    class_name: str,
    gamma: float,
    threshold_rule: ThresholdRule,
):
    """Return stable identity fields and the three scores for one M."""

    raw_probability = np.asarray(sample.foreground_probability)
    if raw_probability.dtype != np.float32:
        raise TypeError("frozen foreground_probability must have dtype float32")
    if raw_probability.ndim != 2:
        raise ValueError("frozen probability must have one 2D shape")
    if not np.isfinite(raw_probability).all() or np.any(
        (raw_probability < 0) | (raw_probability > 1)
    ):
        raise ValueError("frozen probability must be finite and lie in [0, 1]")
    if threshold_rule.estimator_id != "midpoint-v1":
        raise ValueError("the scorer currently supports only midpoint-v1")

    # Match ``binary_eval.binary_record`` exactly: its first operation promotes
    # the frozen float32 map to NumPy's float64 before every derived quantity.
    probability = raw_probability.astype(float)
    prediction = probability >= gamma
    height, width = probability.shape
    prediction_size = int(prediction.sum())
    boundary_reference = prepare_boundary_reference(prediction)
    dice_losses = np.empty(threshold_rule.m, dtype=float)
    nhd_losses = np.empty(threshold_rule.m, dtype=float)
    nhd95_losses = np.empty(threshold_rule.m, dtype=float)
    for position, node in enumerate(threshold_rule.nodes):
        level = probability >= node
        denominator = int(level.sum()) + prediction_size
        if denominator == 0:
            dice_losses[position] = 0.0
        else:
            intersection = int(np.logical_and(level, prediction).sum())
            dice_losses[position] = 1 - 2 * intersection / denominator
        boundary = boundary_reference.compare(level)
        nhd_losses[position] = boundary.nhd
        nhd95_losses[position] = boundary.nhd95

    indexed = {
        "dice": -float(np.dot(threshold_rule.weights, dice_losses)),
        "nhd": -float(np.dot(threshold_rule.weights, nhd_losses)),
        "nhd95": -float(np.dot(threshold_rule.weights, nhd95_losses)),
    }
    dice_field, nhd_field, nhd95_field = _m_score_fields(threshold_rule.m)

    row = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "run_id": simulation_id,
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "class_index": int(class_index),
        "class_name": str(class_name),
        "height": int(height),
        "width": int(width),
    }
    row[dice_field] = float(indexed["dice"])
    row[nhd_field] = float(indexed["nhd"])
    row[nhd95_field] = float(indexed["nhd95"])
    if set(row) != set(IDENTITY_ROW_FIELDS) | {
        dice_field,
        nhd_field,
        nhd95_field,
    }:
        raise RuntimeError("M-specific scorer produced an unexpected row schema")
    return row


def _manifest(
    args,
    *,
    artifact,
    estimator_spec,
    threshold_rule,
    campaign_lock_sha256,
    source_sha256,
    simulation_id,
):
    artifact_manifest = artifact.manifest
    dice_field, nhd_field, nhd95_field = _m_score_fields(args.m)
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": simulation_id,
        "simulation_id": simulation_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "condition": artifact_manifest["condition"],
        "model": artifact_manifest["model"],
        "dataset": artifact_manifest["dataset"],
        "split": artifact_manifest["split"],
        "num_images": artifact_manifest["num_samples"],
        "checkpoint": artifact_manifest["checkpoint"],
        "base_model": artifact_manifest["base_model"],
        "source_sha256": source_sha256,
        "environment": {"packages": _package_versions(), "device": "cpu"},
        "cohort": artifact_manifest["cohort"],
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": args.gamma,
        },
        "preprocessing": artifact_manifest["preprocessing"],
        "losses": {
            "dice": "negative midpoint expectation of foreground 1-Dice",
            "nhd": "negative midpoint expectation of normalized penalized full HD",
            "nhd95": "negative midpoint expectation of normalized penalized HD95",
        },
        "risk_fields": [],
        "auxiliary_fields": [],
        "score_fields": [dice_field, nhd_field, nhd95_field],
        "quadrature": {
            str(args.m): {
                "rule": "midpoint",
                "nodes": threshold_rule.nodes.tolist(),
                "weights": threshold_rule.weights.tolist(),
            }
        },
        "void_policy": "total binary domain; void labels are forbidden",
        "sdc_empty_convention": (
            "published SDC baseline: confidence 0 when p and hard mask are empty"
        ),
        "sample_id_sha256": artifact_manifest["sample_id_sha256"],
        "simulation": {
            "campaign_id": args.campaign_id,
            "campaign_lock_path": _portable_path(args.campaign_lock),
            "campaign_lock_sha256": campaign_lock_sha256,
            "artifact_manifest_path": _portable_path(artifact.manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "artifact_id": artifact_manifest["artifact_id"],
            "artifact_source_sha256": artifact_manifest["source_sha256"],
            "estimator_spec_path": _portable_path(estimator_spec.path),
            "estimator_spec_sha256": estimator_spec.sha256,
            "estimator_id": estimator_spec.estimator_id,
            "target_measure": estimator_spec.target_measure,
            "gamma": args.gamma,
            "m": args.m,
            "quadrature_rule": estimator_spec.estimator_id,
            "seed": args.seed,
            "simulation_id": simulation_id,
        },
        "command": [
            "python",
            "-m",
            "selectseg.score_binary_simulation",
            *args.command_arguments,
        ],
    }


def _simulation_id(
    *,
    campaign_id: str,
    campaign_lock_sha256: str,
    artifact_manifest_sha256: str,
    estimator_spec_sha256: str,
    source_sha256: str,
    gamma: float,
    m: int,
    seed: int,
) -> str:
    identity = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_lock_sha256": campaign_lock_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "estimator_spec_sha256": estimator_spec_sha256,
        "source_sha256": source_sha256,
        "gamma_hex": gamma.hex(),
        "m": m,
        "seed": seed,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def run_simulation(args):
    _validate_args(args)
    campaign_sha256 = _verify_file(
        args.campaign_lock,
        args.expected_campaign_lock_sha256,
        name="campaign lock",
    )
    _verify_file(
        args.artifact_manifest,
        args.expected_artifact_manifest_sha256,
        name="artifact manifest",
    )
    _verify_file(
        args.estimator_spec,
        args.expected_estimator_spec_sha256,
        name="estimator spec",
    )
    estimator_spec = load_estimator_spec(args.estimator_spec)
    if estimator_spec.sha256 != args.expected_estimator_spec_sha256:
        raise RuntimeError("estimator loader returned an inconsistent SHA-256")
    threshold_rule = build_threshold_rule(estimator_spec, m=args.m, seed=args.seed)
    # The ordered streaming iterator performs the same strict payload checks;
    # avoid eagerly loading every (potentially large) map a second time.
    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise RuntimeError("artifact loader returned an inconsistent manifest SHA-256")
    source_sha256 = _source_fingerprint()
    simulation_id = _simulation_id(
        campaign_id=args.campaign_id,
        campaign_lock_sha256=campaign_sha256,
        artifact_manifest_sha256=artifact.manifest_sha256,
        estimator_spec_sha256=estimator_spec.sha256,
        source_sha256=source_sha256,
        gamma=args.gamma,
        m=args.m,
        seed=args.seed,
    )

    artifact_manifest = artifact.manifest
    if artifact_manifest.get("class_index") != 1:
        raise ValueError("frozen binary artifact must declare class_index=1")
    class_name = artifact_manifest.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("frozen binary artifact must declare a non-empty class_name")

    output_dir = (
        Path(args.output_root)
        / artifact_manifest["dataset"]
        / artifact_manifest["condition"]
        / simulation_id
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"simulation output already exists: {output_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{simulation_id}.tmp-", dir=output_dir.parent)
    )
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"

    manifest = _manifest(
        args,
        artifact=artifact,
        estimator_spec=estimator_spec,
        threshold_rule=threshold_rule,
        campaign_lock_sha256=campaign_sha256,
        source_sha256=source_sha256,
        simulation_id=simulation_id,
    )
    try:
        pending = deque()
        row_count = 0
        sample_ids = []
        with (
            ThreadPoolExecutor(max_workers=args.score_workers) as score_pool,
            records_path.open("x", encoding="utf-8") as output,
        ):
            for expected_index, sample in enumerate(artifact.iter_samples()):
                if sample.index != expected_index:
                    raise ValueError(
                        "frozen artifact iterator is not in contiguous index order"
                    )
                sample_ids.append(str(sample.sample_id))
                pending.append(
                    score_pool.submit(
                        score_binary_sample,
                        sample,
                        simulation_id=simulation_id,
                        class_index=artifact_manifest["class_index"],
                        class_name=class_name,
                        gamma=args.gamma,
                        threshold_rule=threshold_rule,
                    )
                )
                if len(pending) < args.max_pending_scores:
                    continue
                record = pending.popleft().result()
                output.write(json.dumps(record, allow_nan=False) + "\n")
                row_count += 1
            while pending:
                record = pending.popleft().result()
                output.write(json.dumps(record, allow_nan=False) + "\n")
                row_count += 1
            output.flush()
            os.fsync(output.fileno())

        if row_count != artifact_manifest["num_samples"]:
            raise RuntimeError(
                f"scored {row_count} rows for "
                f"{artifact_manifest['num_samples']} samples"
            )
        observed_sample_sha256 = hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest()
        if observed_sample_sha256 != artifact_manifest["sample_id_sha256"]:
            raise RuntimeError("scored sample identifiers do not match frozen artifact")
        manifest["num_rows"] = row_count
        manifest["jsonl_sha256"] = sha256_file(records_path)
        with manifest_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        fsync_directory(staging)
        publish_directory_no_replace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir / "records.jsonl", output_dir / "manifest.json"


def main(argv=None):
    records_path, manifest_path = run_simulation(parse_args(argv))
    print(f"saved {records_path}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
