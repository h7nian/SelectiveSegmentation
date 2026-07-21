"""Score the M-independent part of one frozen binary-map artifact.

One invocation streams one canonical frozen artifact and writes the risks,
auxiliary values, and probability-only/deployment-mask baselines that are
independent of the quadrature size ``M``.  The resulting rows are the unique
source of those floating-point values: independent ``M`` jobs never recompute
them, which both saves CPU time and avoids cross-node floating-point joins.

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

from selectseg.artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
)
from selectseg.baselines import strong_binary_confidences
from selectseg.geometry import prepare_boundary_reference
from selectseg.confidence import foreground_dice_loss, soft_dice_confidence
from selectseg.quadrature import sha256_file


ROW_SCHEMA_VERSION = 2
COMMON_ARTIFACT_TYPE = "selectseg.binary_common_scores"
SIMULATION_ARTIFACT_TYPE = "selectseg.binary_simulation_partial"
EXPECTED_M_VALUES = (2, 8, 32)

# Keep the loss-specific schema in one import-safe location so the scorer,
# assembler, and tests cannot silently disagree about the final row contract.
RISK_FIELDS = ("risk_dice", "risk_nhd", "risk_nhd95")
AUXILIARY_FIELDS = ("risk_hd_pixels", "risk_hd95_pixels")
COMMON_SCORE_FIELDS = (
    "confidence_sdc",
    "confidence_mean_max_probability",
    "confidence_negative_entropy",
    "confidence_dice_exact",
    "confidence_qfr_entropy",
    "confidence_plm10_entropy",
    "confidence_mmmc_entropy",
    "confidence_foreground_entropy",
)
M_SCORE_FIELDS = {
    count: (
        f"confidence_dice_m{count}",
        f"confidence_nhd_m{count}",
        f"confidence_nhd95_m{count}",
    )
    for count in EXPECTED_M_VALUES
}
FINAL_SCORE_FIELDS = (
    *COMMON_SCORE_FIELDS,
    *(field for count in EXPECTED_M_VALUES for field in M_SCORE_FIELDS[count]),
)

# M jobs repeat only values whose equality is independent of floating-point
# libraries and execution nodes.  The common artifact alone owns all derived
# floating-point base fields.
IDENTITY_ROW_FIELDS = (
    "schema_version",
    "run_id",
    "sample_id",
    "image_id",
    "image_index",
    "class_index",
    "class_name",
    "height",
    "width",
)
BASE_ROW_FIELDS = (
    *IDENTITY_ROW_FIELDS,
    "image_diagonal",
    "truth_foreground_fraction",
    "prediction_foreground_fraction",
)
IDENTITY_JOIN_FIELDS = tuple(
    field for field in IDENTITY_ROW_FIELDS if field != "run_id"
)


class _StoreExactlyOnce(argparse.Action):
    """Reject a repeated scalar axis instead of silently keeping its last value."""

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
    parser.add_argument("--gamma", required=True, type=float, action=_StoreExactlyOnce)
    parser.add_argument("--output-root", default="outputs/binary_common_scores")
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
    if args.score_workers <= 0:
        raise ValueError("--score-workers must be positive")
    if args.max_pending_scores < args.score_workers:
        raise ValueError("--max-pending-scores must be at least --score-workers")
    for attribute in (
        "expected_campaign_lock_sha256",
        "expected_artifact_manifest_sha256",
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
    unresolved = Path(path)
    resolved = unresolved.resolve()
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
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "selectseg/pipeline/common.py",
        "selectseg/quadrature.py",
        "selectseg/artifacts.py",
        "selectseg/baselines.py",
        "selectseg/geometry.py",
        "selectseg/confidence.py",
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


def _binary_entropy_nats(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def _validated_arrays(sample) -> tuple[np.ndarray, np.ndarray]:
    raw_probability = np.asarray(sample.foreground_probability)
    truth = np.asarray(sample.truth)
    if raw_probability.dtype != np.float32:
        raise TypeError("frozen foreground_probability must have dtype float32")
    if truth.dtype != np.uint8:
        raise TypeError("frozen truth must have dtype uint8")
    if raw_probability.ndim != 2 or truth.shape != raw_probability.shape:
        raise ValueError("frozen probability and truth must have one equal 2D shape")
    if not np.isfinite(raw_probability).all() or np.any(
        (raw_probability < 0) | (raw_probability > 1)
    ):
        raise ValueError("frozen probability must be finite and lie in [0, 1]")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("frozen truth must contain only 0 and 1")
    # Preserve the canonical evaluator's float32-to-float64 promotion.
    return raw_probability.astype(float), truth.astype(bool, copy=False)


def score_binary_common_sample(
    sample,
    *,
    common_id: str,
    class_index: int,
    class_name: str,
    gamma: float,
):
    """Return one complete M-independent analyzer row."""

    probability, truth = _validated_arrays(sample)
    prediction = probability >= gamma
    height, width = probability.shape
    diagonal = math.hypot(height, width)
    boundary_risks = prepare_boundary_reference(truth).compare(prediction)
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "run_id": common_id,
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "class_index": int(class_index),
        "class_name": str(class_name),
        "height": int(height),
        "width": int(width),
        "image_diagonal": float(diagonal),
        "truth_foreground_fraction": float(truth.mean()),
        "prediction_foreground_fraction": float(prediction.mean()),
        "risk_dice": foreground_dice_loss(truth, prediction),
        "risk_nhd": boundary_risks.nhd,
        "risk_nhd95": boundary_risks.nhd95,
        "risk_hd_pixels": float(boundary_risks.nhd * diagonal),
        "risk_hd95_pixels": float(boundary_risks.nhd95 * diagonal),
        "confidence_sdc": soft_dice_confidence(probability, prediction),
        "confidence_mean_max_probability": float(
            np.maximum(probability, 1 - probability).mean()
        ),
        "confidence_negative_entropy": float(-_binary_entropy_nats(probability).mean()),
    }
    row.update(strong_binary_confidences(probability, prediction))
    expected = (
        set(BASE_ROW_FIELDS)
        | set(RISK_FIELDS)
        | set(AUXILIARY_FIELDS)
        | set(COMMON_SCORE_FIELDS)
    )
    if set(row) != expected:
        raise RuntimeError("M-independent scorer produced an unexpected row schema")
    return row


def _common_id(
    *,
    campaign_id: str,
    campaign_lock_sha256: str,
    artifact_manifest_sha256: str,
    source_sha256: str,
    gamma: float,
) -> str:
    identity = {
        "schema_version": ROW_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_lock_sha256": campaign_lock_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "source_sha256": source_sha256,
        "gamma_hex": gamma.hex(),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _manifest(
    args,
    *,
    artifact,
    campaign_lock_sha256: str,
    source_sha256: str,
    common_id: str,
):
    frozen = artifact.manifest
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "artifact_type": COMMON_ARTIFACT_TYPE,
        "run_id": common_id,
        "common_id": common_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "condition": frozen["condition"],
        "model": frozen["model"],
        "dataset": frozen["dataset"],
        "split": frozen["split"],
        "num_images": frozen["num_samples"],
        "checkpoint": frozen["checkpoint"],
        "base_model": frozen["base_model"],
        "source_sha256": source_sha256,
        "environment": {"packages": _package_versions(), "device": "cpu"},
        "cohort": frozen["cohort"],
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": args.gamma,
        },
        "preprocessing": frozen["preprocessing"],
        "losses": {
            "dice": "foreground 1-Dice; empty-empty=0; one-sided-empty=1",
            "nhd": (
                "full bidirectional digital-surface Hausdorff / image diagonal; "
                "empty-empty=0; one-sided-empty=1"
            ),
            "nhd95": (
                "pooled bidirectional HD95 / image diagonal; "
                "empty-empty=0; one-sided-empty=1"
            ),
        },
        "risk_fields": list(RISK_FIELDS),
        "auxiliary_fields": list(AUXILIARY_FIELDS),
        "score_fields": list(COMMON_SCORE_FIELDS),
        "void_policy": "total binary domain; void labels are forbidden",
        "sdc_empty_convention": (
            "published SDC baseline: confidence 0 when p and hard mask are empty"
        ),
        "sample_id_sha256": frozen["sample_id_sha256"],
        "common": {
            "common_id": common_id,
            "campaign_id": args.campaign_id,
            "campaign_lock_path": _portable_path(args.campaign_lock),
            "campaign_lock_sha256": campaign_lock_sha256,
            "artifact_id": frozen["artifact_id"],
            "artifact_manifest_path": _portable_path(artifact.manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "artifact_source_sha256": frozen["source_sha256"],
            "gamma": args.gamma,
        },
        "command": [
            "python",
            "-m",
            "selectseg.pipeline.common",
            *args.command_arguments,
        ],
    }


def run_common(args):
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
    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise RuntimeError("artifact loader returned an inconsistent manifest SHA-256")
    frozen = artifact.manifest
    if frozen.get("class_index") != 1:
        raise ValueError("frozen binary artifact must declare class_index=1")
    class_name = frozen.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("frozen binary artifact must declare a non-empty class_name")

    source_sha256 = _source_fingerprint()
    common_id = _common_id(
        campaign_id=args.campaign_id,
        campaign_lock_sha256=campaign_sha256,
        artifact_manifest_sha256=artifact.manifest_sha256,
        source_sha256=source_sha256,
        gamma=args.gamma,
    )
    output_dir = (
        Path(args.output_root) / frozen["dataset"] / frozen["condition"] / common_id
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"common-score output already exists: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{common_id}.tmp-", dir=output_dir.parent))
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    manifest = _manifest(
        args,
        artifact=artifact,
        campaign_lock_sha256=campaign_sha256,
        source_sha256=source_sha256,
        common_id=common_id,
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
                        score_binary_common_sample,
                        sample,
                        common_id=common_id,
                        class_index=frozen["class_index"],
                        class_name=class_name,
                        gamma=args.gamma,
                    )
                )
                if len(pending) < args.max_pending_scores:
                    continue
                output.write(
                    json.dumps(pending.popleft().result(), allow_nan=False) + "\n"
                )
                row_count += 1
            while pending:
                output.write(
                    json.dumps(pending.popleft().result(), allow_nan=False) + "\n"
                )
                row_count += 1
            output.flush()
            os.fsync(output.fileno())

        if row_count != frozen["num_samples"]:
            raise RuntimeError(
                f"scored {row_count} rows for {frozen['num_samples']} samples"
            )
        observed_sample_sha256 = hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest()
        if observed_sample_sha256 != frozen["sample_id_sha256"]:
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
    records_path, manifest_path = run_common(parse_args(argv))
    print(f"saved {records_path}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
