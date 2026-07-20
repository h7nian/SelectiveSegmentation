"""Score one deployment-threshold sensitivity experiment from frozen maps.

This auxiliary workflow is deliberately disjoint from the canonical campaign.
One invocation consumes one lock-listed frozen artifact at exactly one auxiliary
deployment threshold and, in one ordered artifact pass, writes all
action-dependent common quantities together with Dice/nHD/nHD95 M=32 scores.
The auxiliary lock binds the canonical campaign-lock bytes and every frozen
manifest hash.  Outputs are content-addressed, atomically published, and never
replace an existing directory.
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

from scripts.submit_binary_simulations import load_campaign_lock
from selectseg.binary_artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
)
from selectseg.score_binary_common import (
    AUXILIARY_FIELDS,
    BASE_ROW_FIELDS,
    COMMON_SCORE_FIELDS,
    RISK_FIELDS,
    score_binary_common_sample,
)
from selectseg.score_binary_simulation import score_binary_sample
from selectseg.threshold_estimators import (
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


AUXILIARY_SCHEMA_VERSION = 1
AUXILIARY_ARTIFACT_TYPE = "selectseg.binary_gamma_sensitivity"
EXPECTED_GAMMAS = (0.3, 0.7)
EXPECTED_M = 32
EXPECTED_SEED = 0
EXPECTED_ESTIMATOR_ID = "midpoint-v1"
EXPECTED_CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
M32_SCORE_FIELDS = (
    "confidence_dice_m32",
    "confidence_nhd_m32",
    "confidence_nhd95_m32",
)
OUTPUT_ROW_FIELDS = (
    *BASE_ROW_FIELDS,
    *RISK_FIELDS,
    *AUXILIARY_FIELDS,
    *COMMON_SCORE_FIELDS,
    *M32_SCORE_FIELDS,
)

_LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "auxiliary_id",
        "spec",
        "canonical_campaign_lock",
        "protocol",
        "estimator_spec",
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
        "estimator_spec",
        "cpu_partitions",
        "output_root",
    }
)
_FILE_BINDING_FIELDS = frozenset({"path", "sha256"})
_ARTIFACT_BINDING_FIELDS = frozenset({"manifest_path", "manifest_sha256"})
_CAMPAIGN_BINDING_FIELDS = frozenset({"path", "sha256", "campaign_id"})


class _StoreExactlyOnce(argparse.Action):
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
    parser.add_argument("--auxiliary-lock", required=True)
    parser.add_argument("--expected-auxiliary-lock-sha256", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    parser.add_argument(
        "--gamma", required=True, type=float, action=_StoreExactlyOnce
    )
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--max-pending-scores", type=int, default=16)
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _strict_sha256(value, *, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in string.hexdigits for char in value):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return normalized


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path, *, name):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular, non-symlink file: {source}")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {name} {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _project_path(binding_path, value, *, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    repository_candidate = (Path(binding_path).resolve().parents[2] / raw).resolve()
    if cwd_candidate.exists() or not repository_candidate.exists():
        return cwd_candidate
    return repository_candidate


def _validate_file_binding(value, *, location):
    if not isinstance(value, dict) or set(value) != _FILE_BINDING_FIELDS:
        raise ValueError(f"{location} must contain exactly path and sha256")
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError(f"{location}.path must be a non-empty string")
    value["sha256"] = _strict_sha256(value["sha256"], name=f"{location}.sha256")


def _validate_campaign_binding(value, *, location):
    if not isinstance(value, dict) or set(value) != _CAMPAIGN_BINDING_FIELDS:
        raise ValueError(
            f"{location} must contain exactly path, sha256, and campaign_id"
        )
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError(f"{location}.path must be a non-empty string")
    value["sha256"] = _strict_sha256(value["sha256"], name=f"{location}.sha256")
    if not isinstance(value["campaign_id"], str) or not value["campaign_id"].strip():
        raise ValueError(f"{location}.campaign_id must be a non-empty string")


def _validate_protocol(protocol, *, location):
    expected = {
        "gamma_values": list(EXPECTED_GAMMAS),
        "m": EXPECTED_M,
        "quadrature_rule": EXPECTED_ESTIMATOR_ID,
        "seed": EXPECTED_SEED,
    }
    if not isinstance(protocol, dict) or protocol != expected:
        raise ValueError(f"{location} must equal the frozen auxiliary protocol")


def load_auxiliary_lock(path, *, expected_sha256=None):
    """Strictly load and validate the immutable auxiliary and canonical locks."""

    lock_path = Path(path).resolve()
    lock = _load_strict_json(lock_path, name="auxiliary lock")
    if set(lock) != _LOCK_FIELDS or lock.get("lock_schema_version") != 1:
        raise ValueError("auxiliary lock has an unexpected schema")
    lock_sha256 = sha256_file(lock_path)
    if expected_sha256 is not None:
        expected = _strict_sha256(
            expected_sha256, name="expected auxiliary lock SHA-256"
        )
        if lock_sha256 != expected:
            raise ValueError(
                "auxiliary lock SHA-256 mismatch: "
                f"expected {expected}, observed {lock_sha256}"
            )
    auxiliary_id = lock.get("auxiliary_id")
    if not isinstance(auxiliary_id, str) or not auxiliary_id.strip():
        raise ValueError("auxiliary_id must be a non-empty string")
    _validate_file_binding(lock.get("spec"), location="auxiliary lock spec")
    _validate_campaign_binding(
        lock.get("canonical_campaign_lock"),
        location="auxiliary lock canonical_campaign_lock",
    )
    _validate_protocol(lock.get("protocol"), location="auxiliary lock protocol")
    _validate_file_binding(
        lock.get("estimator_spec"), location="auxiliary lock estimator_spec"
    )
    if tuple(lock.get("cpu_partitions", ())) != EXPECTED_CPU_PARTITIONS:
        raise ValueError("auxiliary lock must use the frozen CPU partition list")
    if not isinstance(lock.get("output_root"), str) or not lock["output_root"].strip():
        raise ValueError("auxiliary lock output_root must be a non-empty string")

    spec_path = _project_path(lock_path, lock["spec"]["path"], name="spec path")
    if sha256_file(spec_path) != lock["spec"]["sha256"]:
        raise ValueError("auxiliary spec is missing or its bytes changed")
    spec = _load_strict_json(spec_path, name="auxiliary spec")
    if set(spec) != _SPEC_FIELDS or spec.get("spec_schema_version") != 1:
        raise ValueError("auxiliary spec has an unexpected schema")
    expected_spec = {
        "spec_schema_version": 1,
        "auxiliary_id": lock["auxiliary_id"],
        "canonical_campaign_lock": lock["canonical_campaign_lock"],
        "protocol": lock["protocol"],
        "estimator_spec": lock["estimator_spec"],
        "cpu_partitions": lock["cpu_partitions"],
        "output_root": lock["output_root"],
    }
    if spec != expected_spec:
        raise ValueError("auxiliary spec and lock disagree")

    campaign_binding = lock["canonical_campaign_lock"]
    campaign_path = _project_path(
        lock_path, campaign_binding["path"], name="canonical campaign lock path"
    )
    campaign_path, campaign_sha256, campaign = load_campaign_lock(campaign_path)
    if campaign_sha256 != campaign_binding["sha256"]:
        raise ValueError("canonical campaign lock bytes differ from the auxiliary lock")
    if campaign["campaign_id"] != campaign_binding["campaign_id"]:
        raise ValueError("canonical campaign_id differs from the auxiliary lock")

    estimator_binding = lock["estimator_spec"]
    estimator_path = _project_path(
        lock_path, estimator_binding["path"], name="estimator spec path"
    )
    if sha256_file(estimator_path) != estimator_binding["sha256"]:
        raise ValueError("estimator spec is missing or its bytes changed")
    if (
        campaign["estimator"]["spec_sha256"] != estimator_binding["sha256"]
        or _project_path(
            campaign_path,
            campaign["estimator"]["spec_path"],
            name="canonical estimator path",
        )
        != estimator_path
    ):
        raise ValueError("auxiliary estimator differs from the canonical campaign")

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("auxiliary lock artifacts must be a non-empty list")
    projection = []
    seen = set()
    for index, entry in enumerate(artifacts):
        if not isinstance(entry, dict) or set(entry) != _ARTIFACT_BINDING_FIELDS:
            raise ValueError(
                f"auxiliary lock artifacts[{index}] must contain exactly "
                "manifest_path and manifest_sha256"
            )
        if not isinstance(entry["manifest_path"], str) or not entry[
            "manifest_path"
        ].strip():
            raise ValueError(
                f"auxiliary lock artifacts[{index}].manifest_path must be non-empty"
            )
        entry["manifest_sha256"] = _strict_sha256(
            entry["manifest_sha256"],
            name=f"auxiliary lock artifacts[{index}].manifest_sha256",
        )
        resolved = _project_path(
            lock_path,
            entry["manifest_path"],
            name=f"artifact {index} manifest path",
        )
        key = resolved.as_posix()
        if key in seen:
            raise ValueError("auxiliary lock contains a duplicate artifact path")
        seen.add(key)
        projection.append((resolved, entry["manifest_sha256"]))
    canonical_projection = [
        (
            _project_path(
                campaign_path,
                entry["manifest_path"],
                name=f"canonical artifact {index} path",
            ),
            entry["manifest_sha256"],
        )
        for index, entry in enumerate(campaign["artifacts"])
    ]
    if projection != canonical_projection:
        raise ValueError(
            "auxiliary artifact paths/hashes differ from the canonical campaign"
        )
    return {
        "path": lock_path,
        "sha256": lock_sha256,
        "data": lock,
        "spec_path": spec_path,
        "campaign_path": campaign_path,
        "campaign_sha256": campaign_sha256,
        "campaign": campaign,
        "estimator_path": estimator_path,
        "artifacts": tuple(projection),
    }


def _validate_args(args):
    if not math.isfinite(args.gamma):
        raise ValueError("--gamma must be finite")
    if args.score_workers <= 0:
        raise ValueError("--score-workers must be positive")
    if args.max_pending_scores < args.score_workers:
        raise ValueError("--max-pending-scores must be at least --score-workers")
    args.expected_auxiliary_lock_sha256 = _strict_sha256(
        args.expected_auxiliary_lock_sha256,
        name="--expected-auxiliary-lock-sha256",
    )
    args.expected_artifact_manifest_sha256 = _strict_sha256(
        args.expected_artifact_manifest_sha256,
        name="--expected-artifact-manifest-sha256",
    )


def _portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "selectseg/score_binary_gamma_sensitivity.py",
        "selectseg/score_binary_common.py",
        "selectseg/score_binary_simulation.py",
        "selectseg/threshold_estimators.py",
        "selectseg/binary_artifacts.py",
        "selectseg/binary_baselines.py",
        "selectseg/binary_boundary.py",
        "selectseg/binary_framework.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"source dependency is missing: {source}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
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


def _run_id(*, auxiliary_lock_sha256, artifact_manifest_sha256, source_sha256, gamma):
    identity = {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "auxiliary_lock_sha256": auxiliary_lock_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "source_sha256": source_sha256,
        "gamma_hex": gamma.hex(),
        "m": EXPECTED_M,
        "seed": EXPECTED_SEED,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def score_gamma_sample(
    sample, *, run_id, class_index, class_name, gamma, threshold_rule
):
    """Compute the common and M=32 action-dependent fields for one sample."""

    common = score_binary_common_sample(
        sample,
        common_id=run_id,
        class_index=class_index,
        class_name=class_name,
        gamma=gamma,
    )
    indexed = score_binary_sample(
        sample,
        simulation_id=run_id,
        class_index=class_index,
        class_name=class_name,
        gamma=gamma,
        threshold_rule=threshold_rule,
    )
    for field in (
        "schema_version",
        "run_id",
        "sample_id",
        "image_id",
        "image_index",
        "class_index",
        "class_name",
        "height",
        "width",
    ):
        if common[field] != indexed[field]:
            raise RuntimeError(f"common/indexed identity mismatch for {field}")
    common.update({field: indexed[field] for field in M32_SCORE_FIELDS})
    if set(common) != set(OUTPUT_ROW_FIELDS):
        raise RuntimeError("gamma-sensitivity scorer produced an unexpected row schema")
    return common


def _manifest(
    args,
    *,
    binding,
    artifact,
    estimator_spec,
    threshold_rule,
    source_sha256,
    run_id,
):
    frozen = artifact.manifest
    return {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "run_id": run_id,
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
        "preprocessing": frozen["preprocessing"],
        "losses": {
            "dice": "foreground 1-Dice and negative M32 midpoint expectation",
            "nhd": "normalized penalized full HD and negative M32 expectation",
            "nhd95": "normalized penalized pooled HD95 and negative M32 expectation",
        },
        "risk_fields": list(RISK_FIELDS),
        "auxiliary_fields": list(AUXILIARY_FIELDS),
        "score_fields": [*COMMON_SCORE_FIELDS, *M32_SCORE_FIELDS],
        "quadrature": {
            str(EXPECTED_M): {
                "rule": "midpoint",
                "nodes": threshold_rule.nodes.tolist(),
                "weights": threshold_rule.weights.tolist(),
            }
        },
        "sample_id_sha256": frozen["sample_id_sha256"],
        "provenance": {
            "auxiliary_id": binding["data"]["auxiliary_id"],
            "auxiliary_lock_path": _portable_path(binding["path"]),
            "auxiliary_lock_sha256": binding["sha256"],
            "auxiliary_spec_path": _portable_path(binding["spec_path"]),
            "auxiliary_spec_sha256": binding["data"]["spec"]["sha256"],
            "canonical_campaign_id": binding["campaign"]["campaign_id"],
            "canonical_campaign_lock_path": _portable_path(binding["campaign_path"]),
            "canonical_campaign_lock_sha256": binding["campaign_sha256"],
            "artifact_id": frozen["artifact_id"],
            "artifact_manifest_path": _portable_path(artifact.manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "artifact_source_sha256": frozen["source_sha256"],
            "estimator_spec_path": _portable_path(estimator_spec.path),
            "estimator_spec_sha256": estimator_spec.sha256,
            "estimator_id": estimator_spec.estimator_id,
            "target_measure": estimator_spec.target_measure,
            "gamma": args.gamma,
            "m": EXPECTED_M,
            "seed": EXPECTED_SEED,
            "artifact_passes": 1,
        },
        "canonical_schema_v2_compatible": False,
        "command": [
            "python",
            "-m",
            "selectseg.score_binary_gamma_sensitivity",
            *args.command_arguments,
        ],
    }


def run_gamma_sensitivity(args):
    """Run one append-only ``(condition, gamma)`` auxiliary experiment."""

    _validate_args(args)
    binding = load_auxiliary_lock(
        args.auxiliary_lock,
        expected_sha256=args.expected_auxiliary_lock_sha256,
    )
    gamma_matches = [
        value
        for value in binding["data"]["protocol"]["gamma_values"]
        if float(value).hex() == args.gamma.hex()
    ]
    if len(gamma_matches) != 1:
        raise ValueError("requested gamma is not uniquely predeclared by the lock")
    artifact_path = Path(args.artifact_manifest).resolve()
    matches = [
        (path, sha256)
        for path, sha256 in binding["artifacts"]
        if path == artifact_path
    ]
    if len(matches) != 1:
        raise ValueError("artifact manifest must occur exactly once in the lock")
    if matches[0][1] != args.expected_artifact_manifest_sha256:
        raise ValueError("artifact hash differs from the auxiliary lock")
    if sha256_file(artifact_path) != args.expected_artifact_manifest_sha256:
        raise ValueError("artifact manifest SHA-256 mismatch")

    artifact = load_binary_artifact(artifact_path, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise RuntimeError("artifact loader returned an inconsistent manifest hash")
    frozen = artifact.manifest
    if frozen.get("class_index") != 1:
        raise ValueError("frozen binary artifact must declare class_index=1")
    class_name = frozen.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("frozen binary artifact must declare a class_name")

    estimator_spec = load_estimator_spec(binding["estimator_path"])
    if estimator_spec.sha256 != binding["data"]["estimator_spec"]["sha256"]:
        raise RuntimeError("estimator loader returned an inconsistent SHA-256")
    threshold_rule = build_threshold_rule(
        estimator_spec, m=EXPECTED_M, seed=EXPECTED_SEED
    )
    source_sha256 = _source_fingerprint()
    run_id = _run_id(
        auxiliary_lock_sha256=binding["sha256"],
        artifact_manifest_sha256=artifact.manifest_sha256,
        source_sha256=source_sha256,
        gamma=args.gamma,
    )
    gamma_tag = f"gamma-{args.gamma:.1f}".replace(".", "p")
    output_dir = (
        Path(binding["data"]["output_root"])
        / frozen["dataset"]
        / frozen["condition"]
        / gamma_tag
        / run_id
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"gamma-sensitivity output already exists: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_dir.parent))
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    manifest = _manifest(
        args,
        binding=binding,
        artifact=artifact,
        estimator_spec=estimator_spec,
        threshold_rule=threshold_rule,
        source_sha256=source_sha256,
        run_id=run_id,
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
                        score_gamma_sample,
                        sample,
                        run_id=run_id,
                        class_index=frozen["class_index"],
                        class_name=class_name,
                        gamma=args.gamma,
                        threshold_rule=threshold_rule,
                    )
                )
                if len(pending) < args.max_pending_scores:
                    continue
                output.write(json.dumps(pending.popleft().result(), allow_nan=False) + "\n")
                row_count += 1
            while pending:
                output.write(json.dumps(pending.popleft().result(), allow_nan=False) + "\n")
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
    records_path, manifest_path = run_gamma_sensitivity(parse_args(argv))
    print(f"saved {records_path}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
