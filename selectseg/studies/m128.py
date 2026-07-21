"""Append-only M=128 numerical-reference scores for frozen binary maps.

This scorer is deliberately separate from the canonical schema-v2 campaign.
It streams one lock-listed frozen artifact, evaluates Dice, normalized full
Hausdorff distance (nHD), and normalized pooled HD95 at all 128 midpoint nodes
in the same CPU pass, and publishes a content-addressed auxiliary artifact.

With ``--include-m32-diagnostics`` the same pass also evaluates the disjoint
32-point midpoint grid and records each per-image M=32 estimate together with
the signed M=128-minus-M=32 difference.  These fields are diagnostics only;
they are not canonical schema-v2 score fields and are never assembled into the
locked main campaign by this module.

The output identity binds the exact campaign-lock, frozen-manifest, estimator,
and scorer-source hashes.  Publication is atomic and refuses to replace an
existing directory.
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
from typing import Mapping

import numpy as np

from selectseg.artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
)
from selectseg.geometry import prepare_boundary_reference
from selectseg.quadrature import (
    ThresholdRule,
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


AUXILIARY_SCHEMA_VERSION = 1
AUXILIARY_ARTIFACT_TYPE = "selectseg.binary_m128_auxiliary"
M128 = 128
M32 = 32
MIDPOINT_SEED = 0

IDENTITY_FIELDS = (
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
M128_SCORE_FIELDS = (
    "confidence_dice_m128_aux",
    "confidence_nhd_m128_aux",
    "confidence_nhd95_m128_aux",
)
M32_DIAGNOSTIC_FIELDS = (
    "diagnostic_confidence_dice_m32_recomputed",
    "diagnostic_confidence_nhd_m32_recomputed",
    "diagnostic_confidence_nhd95_m32_recomputed",
    "diagnostic_delta_dice_m128_minus_m32",
    "diagnostic_delta_nhd_m128_minus_m32",
    "diagnostic_delta_nhd95_m128_minus_m32",
)


class _StoreExactlyOnce(argparse.Action):
    """Reject repeated scalar axes rather than silently taking the last one."""

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
    parser.add_argument(
        "--include-m32-diagnostics",
        action="store_true",
        help="also recompute M=32 and store per-image signed M128-M32 deltas",
    )
    parser.add_argument("--output-root", default="outputs/binary_m128_auxiliary")
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


def _validate_args(args) -> None:
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


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_file(path: str | Path, *, name: str):
    source = Path(path).resolve()
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {name} {source}: {error}") from error


def _resolve_locked_path(lock_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    repository_candidate = (lock_path.parent.parent / raw).resolve()
    if cwd_candidate.exists() or not repository_candidate.exists():
        return cwd_candidate
    return repository_candidate


def _validate_campaign_binding(
    args,
    *,
    artifact,
    estimator_spec,
) -> Mapping[str, object]:
    """Verify that the exact artifact and estimator are named by the lock."""

    lock = _strict_json_file(args.campaign_lock, name="campaign lock")
    if not isinstance(lock, dict):
        raise ValueError("campaign lock must contain one JSON object")
    if lock.get("campaign_id") != args.campaign_id:
        raise ValueError("campaign lock has a different campaign_id")
    lock_path = Path(args.campaign_lock).resolve()

    protocol = lock.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("campaign lock must declare a protocol object")
    gamma_values = protocol.get("gamma_values")
    if not isinstance(gamma_values, list) or len(gamma_values) != 1:
        raise ValueError("auxiliary scorer requires exactly one locked gamma")
    locked_gamma = gamma_values[0]
    if (
        isinstance(locked_gamma, bool)
        or not isinstance(locked_gamma, (int, float))
        or float(locked_gamma).hex() != args.gamma.hex()
    ):
        raise ValueError("requested gamma differs from the campaign lock")

    estimator = lock.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError("campaign lock must declare an estimator object")
    locked_estimator_path = _resolve_locked_path(
        lock_path, estimator.get("spec_path"), name="locked estimator path"
    )
    if locked_estimator_path != Path(args.estimator_spec).resolve():
        raise ValueError("requested estimator path differs from the campaign lock")
    if estimator.get("spec_sha256") != args.expected_estimator_spec_sha256:
        raise ValueError("requested estimator hash differs from the campaign lock")
    if estimator_spec.sha256 != estimator.get("spec_sha256"):
        raise ValueError("loaded estimator hash differs from the campaign lock")
    if estimator_spec.estimator_id != estimator.get("estimator_id"):
        raise ValueError("loaded estimator_id differs from the campaign lock")
    if estimator_spec.target_measure != estimator.get("target_measure"):
        raise ValueError("loaded estimator target differs from the campaign lock")

    entries = lock.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("campaign lock must declare a non-empty artifact list")
    matches = []
    requested_path = Path(args.artifact_manifest).resolve()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"campaign lock artifact {index} must be an object")
        entry_path = _resolve_locked_path(
            lock_path,
            entry.get("manifest_path"),
            name=f"locked artifact {index} path",
        )
        if entry_path == requested_path:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            "artifact manifest must occur exactly once in the campaign lock"
        )
    entry = matches[0]
    if entry.get("manifest_sha256") != args.expected_artifact_manifest_sha256:
        raise ValueError("artifact hash differs from the campaign lock")

    frozen = artifact.manifest
    exact_fields = {
        "artifact_id": frozen["artifact_id"],
        "dataset": frozen["dataset"],
        "condition": frozen["condition"],
        "model": frozen["model"],
        "split": frozen["split"],
        "source_sha256": frozen["source_sha256"],
        "sample_id_sha256": frozen["sample_id_sha256"],
        "num_samples": frozen["num_samples"],
    }
    for field, expected in exact_fields.items():
        if entry.get(field) != expected:
            raise ValueError(f"locked artifact field {field!r} is inconsistent")
    return lock


def _source_fingerprint() -> str:
    """Hash every local source module defining the auxiliary records."""

    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "selectseg/studies/m128.py",
        "selectseg/quadrature.py",
        "selectseg/artifacts.py",
        "selectseg/geometry.py",
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


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _auxiliary_id(
    *,
    campaign_id: str,
    campaign_lock_sha256: str,
    artifact_manifest_sha256: str,
    estimator_spec_sha256: str,
    source_sha256: str,
    gamma: float,
    include_m32_diagnostics: bool,
) -> str:
    identity = {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "campaign_id": campaign_id,
        "campaign_lock_sha256": campaign_lock_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "estimator_spec_sha256": estimator_spec_sha256,
        "source_sha256": source_sha256,
        "gamma_hex": gamma.hex(),
        "m": M128,
        "seed": MIDPOINT_SEED,
        "include_m32_diagnostics": include_m32_diagnostics,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _losses_by_rule(
    probability: np.ndarray,
    prediction: np.ndarray,
    rules: Mapping[int, ThresholdRule],
) -> dict[int, tuple[float, float, float]]:
    """Evaluate the union of all requested grids once for all three losses."""

    all_nodes = np.unique(np.concatenate([rule.nodes for rule in rules.values()]))
    prediction_size = int(prediction.sum())
    boundary_reference = prepare_boundary_reference(prediction)
    dice_losses = np.empty(all_nodes.size, dtype=float)
    nhd_losses = np.empty(all_nodes.size, dtype=float)
    nhd95_losses = np.empty(all_nodes.size, dtype=float)
    for position, node in enumerate(all_nodes):
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

    result = {}
    for count, rule in rules.items():
        positions = np.searchsorted(all_nodes, rule.nodes)
        if not np.array_equal(all_nodes[positions], rule.nodes):
            raise AssertionError("midpoint-node lookup lost exact values")
        result[count] = (
            -float(np.dot(rule.weights, dice_losses[positions])),
            -float(np.dot(rule.weights, nhd_losses[positions])),
            -float(np.dot(rule.weights, nhd95_losses[positions])),
        )
    return result


def score_auxiliary_sample(
    sample,
    *,
    auxiliary_id: str,
    class_index: int,
    class_name: str,
    gamma: float,
    m128_rule: ThresholdRule,
    m32_rule: ThresholdRule | None,
) -> dict[str, object]:
    """Return one M=128 row and optional recomputed M=32 diagnostics."""

    raw_probability = np.asarray(sample.foreground_probability)
    if raw_probability.dtype != np.float32:
        raise TypeError("frozen foreground_probability must have dtype float32")
    if raw_probability.ndim != 2:
        raise ValueError("frozen probability must have one 2D shape")
    if not np.isfinite(raw_probability).all() or np.any(
        (raw_probability < 0) | (raw_probability > 1)
    ):
        raise ValueError("frozen probability must be finite and lie in [0, 1]")
    if m128_rule.estimator_id != "midpoint-v1" or m128_rule.m != M128:
        raise ValueError("M=128 auxiliary scorer requires midpoint-v1 at M=128")
    if m32_rule is not None and (
        m32_rule.estimator_id != "midpoint-v1" or m32_rule.m != M32
    ):
        raise ValueError("M=32 diagnostics require midpoint-v1 at M=32")

    probability = raw_probability.astype(float)
    prediction = probability >= gamma
    rules = {M128: m128_rule}
    if m32_rule is not None:
        rules[M32] = m32_rule
    values = _losses_by_rule(probability, prediction, rules)
    m128_values = values[M128]
    height, width = probability.shape
    row: dict[str, object] = {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "run_id": auxiliary_id,
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "class_index": int(class_index),
        "class_name": str(class_name),
        "height": int(height),
        "width": int(width),
        **dict(zip(M128_SCORE_FIELDS, m128_values, strict=True)),
    }
    if m32_rule is not None:
        m32_values = values[M32]
        diagnostics = (
            *m32_values,
            *(
                left - right
                for left, right in zip(m128_values, m32_values, strict=True)
            ),
        )
        row.update(dict(zip(M32_DIAGNOSTIC_FIELDS, diagnostics, strict=True)))

    expected = set(IDENTITY_FIELDS) | set(M128_SCORE_FIELDS)
    if m32_rule is not None:
        expected |= set(M32_DIAGNOSTIC_FIELDS)
    if set(row) != expected:
        raise RuntimeError("M=128 auxiliary scorer produced an unexpected row schema")
    return row


def _manifest(
    args,
    *,
    artifact,
    estimator_spec,
    m128_rule: ThresholdRule,
    m32_rule: ThresholdRule | None,
    campaign_lock_sha256: str,
    source_sha256: str,
    auxiliary_id: str,
) -> dict[str, object]:
    frozen = artifact.manifest
    quadrature = {
        str(M128): {
            "rule": "midpoint",
            "nodes": m128_rule.nodes.tolist(),
            "weights": m128_rule.weights.tolist(),
        }
    }
    if m32_rule is not None:
        quadrature[str(M32)] = {
            "rule": "midpoint",
            "nodes": m32_rule.nodes.tolist(),
            "weights": m32_rule.weights.tolist(),
        }
    return {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "run_id": auxiliary_id,
        "auxiliary_id": auxiliary_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": frozen["dataset"],
        "condition": frozen["condition"],
        "model": frozen["model"],
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
        "losses": {
            "dice": "negative midpoint expectation of foreground 1-Dice",
            "nhd": "negative midpoint expectation of normalized penalized full HD",
            "nhd95": "negative midpoint expectation of normalized penalized HD95",
        },
        "score_fields": list(M128_SCORE_FIELDS),
        "diagnostic_fields": (
            list(M32_DIAGNOSTIC_FIELDS) if m32_rule is not None else []
        ),
        "quadrature": quadrature,
        "sample_id_sha256": frozen["sample_id_sha256"],
        "provenance": {
            "campaign_id": args.campaign_id,
            "campaign_lock_path": _portable_path(args.campaign_lock),
            "campaign_lock_sha256": campaign_lock_sha256,
            "artifact_manifest_path": _portable_path(artifact.manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "artifact_id": frozen["artifact_id"],
            "artifact_source_sha256": frozen["source_sha256"],
            "estimator_spec_path": _portable_path(estimator_spec.path),
            "estimator_spec_sha256": estimator_spec.sha256,
            "estimator_id": estimator_spec.estimator_id,
            "target_measure": estimator_spec.target_measure,
            "gamma": args.gamma,
            "m": M128,
            "seed": MIDPOINT_SEED,
            "include_m32_diagnostics": m32_rule is not None,
        },
        "canonical_schema_v2_compatible": False,
        "command": [
            "python",
            "-m",
            "selectseg.studies.m128",
            *args.command_arguments,
        ],
    }


def run_auxiliary(args):
    """Run one append-only M=128 artifact scorer."""

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
    m128_rule = build_threshold_rule(estimator_spec, m=M128, seed=MIDPOINT_SEED)
    m32_rule = (
        build_threshold_rule(estimator_spec, m=M32, seed=MIDPOINT_SEED)
        if args.include_m32_diagnostics
        else None
    )

    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise RuntimeError("artifact loader returned an inconsistent manifest hash")
    _validate_campaign_binding(args, artifact=artifact, estimator_spec=estimator_spec)
    frozen = artifact.manifest
    if frozen.get("class_index") != 1:
        raise ValueError("frozen binary artifact must declare class_index=1")
    class_name = frozen.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("frozen binary artifact must declare a class_name")

    source_sha256 = _source_fingerprint()
    auxiliary_id = _auxiliary_id(
        campaign_id=args.campaign_id,
        campaign_lock_sha256=campaign_sha256,
        artifact_manifest_sha256=artifact.manifest_sha256,
        estimator_spec_sha256=estimator_spec.sha256,
        source_sha256=source_sha256,
        gamma=args.gamma,
        include_m32_diagnostics=args.include_m32_diagnostics,
    )
    output_dir = (
        Path(args.output_root) / frozen["dataset"] / frozen["condition"] / auxiliary_id
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"M=128 auxiliary output already exists: {output_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{auxiliary_id}.tmp-", dir=output_dir.parent)
    )
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    manifest = _manifest(
        args,
        artifact=artifact,
        estimator_spec=estimator_spec,
        m128_rule=m128_rule,
        m32_rule=m32_rule,
        campaign_lock_sha256=campaign_sha256,
        source_sha256=source_sha256,
        auxiliary_id=auxiliary_id,
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
                        score_auxiliary_sample,
                        sample,
                        auxiliary_id=auxiliary_id,
                        class_index=frozen["class_index"],
                        class_name=class_name,
                        gamma=args.gamma,
                        m128_rule=m128_rule,
                        m32_rule=m32_rule,
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

        if row_count != frozen["num_samples"]:
            raise RuntimeError(
                f"scored {row_count} rows for {frozen['num_samples']} samples"
            )
        observed_sample_sha256 = hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest()
        if observed_sample_sha256 != frozen["sample_id_sha256"]:
            raise RuntimeError("scored sample identifiers do not match the artifact")
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
    records_path, manifest_path = run_auxiliary(parse_args(argv))
    print(f"saved {records_path}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
