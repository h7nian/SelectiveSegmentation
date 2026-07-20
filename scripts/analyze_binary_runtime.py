"""Strictly analyze the sixteen locked confidence-runtime artifacts.

The report summarizes hardware-dependent wall-clock throughput only.  It does
not infer algorithmic complexity and does not mix artifact I/O or model
inference into the timed regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.analyze_binary import EXPECTED_CONDITIONS
from scripts.submit_binary_simulations import _project_path, load_campaign_lock
from selectseg.benchmark_binary_runtime import (
    EXPECTED_ARTIFACT_COUNT,
    METHODS_V1,
    METHODS_V2,
    RECORD_FIELDS,
    RUNTIME_ARTIFACT_TYPE,
    RUNTIME_SCHEMA_VERSION,
    THREAD_VARIABLES,
    _measurement_order,
    _runtime_id,
    _selected_entries,
    load_runtime_ladder_lock,
    runtime_source_fingerprint,
    runtime_protocol_version,
)
from selectseg.binary_artifacts import load_binary_artifact


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_ARTIFACT_TYPE = "selectseg.binary_confidence_runtime_analysis"
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "runtime_id",
        "created_utc",
        "dataset",
        "condition",
        "model",
        "split",
        "benchmark_scope",
        "protocol",
        "execution",
        "selected_samples",
        "num_selected_images",
        "total_selected_pixels",
        "measurements_per_method",
        "record_fields",
        "environment",
        "memory",
        "provenance",
        "command",
        "num_records",
        "records_sha256",
    }
)


@dataclass(frozen=True)
class RuntimeCondition:
    records_path: Path
    manifest_path: Path
    manifest: dict
    records: tuple[dict, ...]

    @property
    def key(self):
        return self.manifest["dataset"], self.manifest["condition"]

    @property
    def protocol_version(self):
        return runtime_protocol_version(self.manifest["protocol"])

    @property
    def methods(self):
        return METHODS_V1 if self.protocol_version == 1 else METHODS_V2


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="RUNTIME_RECORDS_JSONL",
        help="exactly 16 explicit runtime records.jsonl paths",
    )
    parser.add_argument(
        "--output", default="outputs/binary_runtime_analysis/analysis.json"
    )
    parser.add_argument("--benchmark-lock")
    parser.add_argument("--expected-benchmark-lock-sha256")
    return parser.parse_args(argv)


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(raw: str, *, source: str):
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{location} must be a SHA-256 digest")
    return value.lower()


def _finite_positive(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{location} must be finite and positive")
    return result


def _positive_integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _validate_selected(value: Any, *, location: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    expected_fields = {
        "index",
        "sample_id",
        "height",
        "width",
        "pixels",
        "size_quantile_rank",
    }
    indices = []
    ranks = []
    for index, row in enumerate(value):
        row_location = f"{location}[{index}]"
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"{row_location} has an invalid schema")
        if isinstance(row["index"], bool) or not isinstance(row["index"], int) or row["index"] < 0:
            raise ValueError(f"{row_location}.index must be a non-negative integer")
        for field in ("height", "width", "pixels"):
            _positive_integer(row[field], location=f"{row_location}.{field}")
        if (
            isinstance(row["size_quantile_rank"], bool)
            or not isinstance(row["size_quantile_rank"], int)
            or row["size_quantile_rank"] < 0
        ):
            raise ValueError(f"{row_location}.size_quantile_rank must be non-negative")
        if not isinstance(row["sample_id"], str) or not row["sample_id"]:
            raise ValueError(f"{row_location}.sample_id must be non-empty")
        if row["pixels"] != row["height"] * row["width"]:
            raise ValueError(f"{row_location}.pixels is inconsistent")
        indices.append(row["index"])
        ranks.append(row["size_quantile_rank"])
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"{location} must have unique manifest-ordered indices")
    if set(ranks) != set(range(len(value))):
        raise ValueError(f"{location} must contain every size-quantile rank exactly once")


def load_runtime_condition(path: str | os.PathLike[str]) -> RuntimeCondition:
    records_path = Path(path)
    if records_path.name != "records.jsonl" or not records_path.is_file() or records_path.is_symlink():
        raise FileNotFoundError(f"runtime input must be a regular records.jsonl: {records_path}")
    manifest_path = records_path.parent / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"runtime manifest is missing: {manifest_path}")
    manifest = _loads_strict(manifest_path.read_text(encoding="utf-8"), source=str(manifest_path))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError(f"{manifest_path} has an invalid manifest schema")
    if (
        manifest["schema_version"] != RUNTIME_SCHEMA_VERSION
        or manifest["artifact_type"] != RUNTIME_ARTIFACT_TYPE
    ):
        raise ValueError(f"{manifest_path} has an unsupported artifact type/schema")
    run_id = manifest["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) != 16
        or any(character not in "0123456789abcdef" for character in run_id)
        or manifest["runtime_id"] != run_id
    ):
        raise ValueError(f"{manifest_path} has an invalid runtime identity")
    for field in ("dataset", "condition", "model", "split"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"{manifest_path}.{field} must be non-empty")
    scope = manifest["benchmark_scope"]
    if not isinstance(scope, dict) or set(scope) != {"timed", "excluded", "interpretation"}:
        raise ValueError(f"{manifest_path}.benchmark_scope is malformed")
    if "hardware-dependent" not in scope["interpretation"] or "complexity" not in scope["interpretation"]:
        raise ValueError(f"{manifest_path} must distinguish timing from complexity")
    protocol = manifest["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError(f"{manifest_path}.protocol must be an object")
    try:
        protocol_version = runtime_protocol_version(protocol)
    except ValueError as error:
        raise ValueError(f"{manifest_path}.protocol is unsupported") from error
    methods = METHODS_V1 if protocol_version == 1 else METHODS_V2
    execution = manifest["execution"]
    expected_thread_environment = {
        name: "1"
        for name in THREAD_VARIABLES
    }
    if execution != {
        "benchmark_workers": 8,
        "native_threads_per_worker": 1,
        "thread_environment": expected_thread_environment,
    }:
        raise ValueError(f"{manifest_path}.execution must declare eight workers")
    _validate_selected(manifest["selected_samples"], location=f"{manifest_path}.selected_samples")
    num_images = _positive_integer(manifest["num_selected_images"], location=f"{manifest_path}.num_selected_images")
    if num_images != len(manifest["selected_samples"]):
        raise ValueError(f"{manifest_path} selected-image count is inconsistent")
    total_pixels = _positive_integer(manifest["total_selected_pixels"], location=f"{manifest_path}.total_selected_pixels")
    if total_pixels != sum(row["pixels"] for row in manifest["selected_samples"]):
        raise ValueError(f"{manifest_path} selected-pixel count is inconsistent")
    if manifest["measurements_per_method"] != 4:
        raise ValueError(f"{manifest_path} has an unexpected measurement count")
    if manifest["record_fields"] != sorted(RECORD_FIELDS):
        raise ValueError(f"{manifest_path}.record_fields is inconsistent")
    memory = manifest["memory"]
    if not isinstance(memory, dict) or set(memory) != {
        "peak_rss_bytes",
        "scope",
        "method_attribution_available",
    }:
        raise ValueError(f"{manifest_path}.memory is malformed")
    if memory["peak_rss_bytes"] is not None:
        _positive_integer(memory["peak_rss_bytes"], location=f"{manifest_path}.memory.peak_rss_bytes")
    if memory["method_attribution_available"] is not False:
        raise ValueError(f"{manifest_path} must not attribute process RSS to a method")
    environment = manifest["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "packages",
        "platform",
        "machine",
        "cpu_model",
        "logical_cpu_count",
        "affinity_cpu_count",
        "thread_environment",
        "slurm",
    }:
        raise ValueError(f"{manifest_path}.environment is malformed")
    if environment["thread_environment"] != expected_thread_environment:
        raise ValueError(f"{manifest_path}.environment has unfixed native threads")
    packages = environment["packages"]
    if not isinstance(packages, dict) or set(packages) != {
        "python",
        "numpy",
        "scipy",
    }:
        raise ValueError(f"{manifest_path}.environment.packages is malformed")
    for package, version in packages.items():
        if not isinstance(version, str) or not version:
            raise ValueError(
                f"{manifest_path}.environment.packages.{package} is missing"
            )
    for field in ("platform", "machine"):
        if not isinstance(environment[field], str) or not environment[field]:
            raise ValueError(f"{manifest_path}.environment.{field} is missing")
    if environment["cpu_model"] is not None and (
        not isinstance(environment["cpu_model"], str)
        or not environment["cpu_model"]
    ):
        raise ValueError(f"{manifest_path}.environment.cpu_model is malformed")
    if environment["logical_cpu_count"] is not None:
        _positive_integer(
            environment["logical_cpu_count"],
            location=f"{manifest_path}.environment.logical_cpu_count",
        )
    affinity_count = _positive_integer(
        environment["affinity_cpu_count"],
        location=f"{manifest_path}.environment.affinity_cpu_count",
    )
    if affinity_count < execution["benchmark_workers"]:
        raise ValueError(
            f"{manifest_path} exposes fewer CPUs than benchmark workers"
        )
    if not isinstance(environment["slurm"], dict) or set(environment["slurm"]) != {
        "job_id",
        "partition",
        "node_list",
        "cpus_per_task",
    }:
        raise ValueError(f"{manifest_path}.environment.slurm is malformed")
    slurm = environment["slurm"]
    if slurm["partition"] not in {"agsmall", "amdsmall", "msismall"}:
        raise ValueError(f"{manifest_path} was not run on a declared CPU partition")
    if slurm["cpus_per_task"] != "8":
        raise ValueError(f"{manifest_path} must bind SLURM_CPUS_PER_TASK=8")
    for field in ("job_id", "node_list"):
        if not isinstance(slurm[field], str) or not slurm[field]:
            raise ValueError(f"{manifest_path}.environment.slurm.{field} is missing")
    provenance = manifest["provenance"]
    required_provenance = {
        "campaign_id",
        "campaign_lock_path",
        "campaign_lock_sha256",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "artifact_id",
        "artifact_source_sha256",
        "estimator_spec_path",
        "estimator_spec_sha256",
        "estimator_id",
        "benchmark_spec_path",
        "benchmark_spec_sha256",
        "runtime_source_sha256",
    }
    if protocol_version == 2:
        required_provenance |= {
            "benchmark_lock_path",
            "benchmark_lock_sha256",
        }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError(f"{manifest_path}.provenance is malformed")
    for field in (
        "campaign_lock_sha256",
        "artifact_manifest_sha256",
        "artifact_source_sha256",
        "estimator_spec_sha256",
        "benchmark_spec_sha256",
        "runtime_source_sha256",
    ):
        _digest(provenance[field], location=f"{manifest_path}.provenance.{field}")
    if protocol_version == 2:
        _digest(
            provenance["benchmark_lock_sha256"],
            location=f"{manifest_path}.provenance.benchmark_lock_sha256",
        )
        if (
            not isinstance(provenance["benchmark_lock_path"], str)
            or not provenance["benchmark_lock_path"]
        ):
            raise ValueError(
                f"{manifest_path}.provenance.benchmark_lock_path is empty"
            )
    expected_runtime_id = _runtime_id(
        campaign_sha256=provenance["campaign_lock_sha256"],
        artifact_sha256=provenance["artifact_manifest_sha256"],
        estimator_sha256=provenance["estimator_spec_sha256"],
        spec_sha256=provenance["benchmark_spec_sha256"],
        source_sha256=provenance["runtime_source_sha256"],
        benchmark_lock_sha256=(
            provenance["benchmark_lock_sha256"]
            if protocol_version == 2
            else None
        ),
    )
    if run_id != expected_runtime_id or records_path.parent.name != run_id:
        raise ValueError(f"{manifest_path} has a non-content-addressed runtime identity")
    if _digest(manifest["records_sha256"], location=f"{manifest_path}.records_sha256") != _sha256(records_path):
        raise ValueError(f"{records_path} SHA-256 differs from its manifest")

    records = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank runtime row at {records_path}:{line_number}")
        row = _loads_strict(line, source=f"{records_path}:{line_number}")
        if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
            raise ValueError(f"runtime row {records_path}:{line_number} has an invalid schema")
        if row["schema_version"] != RUNTIME_SCHEMA_VERSION or row["run_id"] != run_id:
            raise ValueError(f"runtime row {records_path}:{line_number} has inconsistent identity")
        if row["dataset"] != manifest["dataset"] or row["condition"] != manifest["condition"]:
            raise ValueError(f"runtime row {records_path}:{line_number} has inconsistent condition")
        if row["method"] not in methods:
            raise ValueError(f"runtime row {records_path}:{line_number} has unknown method")
        if (
            isinstance(row["repetition"], bool)
            or isinstance(row["order_position"], bool)
            or row["repetition"] not in range(4)
            or row["order_position"] not in range(len(methods))
        ):
            raise ValueError(f"runtime row {records_path}:{line_number} has invalid trial indices")
        expected_order = _measurement_order(
            methods,
            row["repetition"],
            protocol["measured_order"],
        )
        if expected_order[row["order_position"]] != row["method"]:
            raise ValueError(f"runtime row {records_path}:{line_number} violates alternating order")
        if row["num_images"] != num_images or row["total_pixels"] != total_pixels:
            raise ValueError(f"runtime row {records_path}:{line_number} has inconsistent workload")
        wall = _finite_positive(row["wall_seconds"], location=f"{records_path}:{line_number}.wall_seconds")
        per_image = _finite_positive(row["seconds_per_image"], location=f"{records_path}:{line_number}.seconds_per_image")
        throughput = _finite_positive(row["images_per_second"], location=f"{records_path}:{line_number}.images_per_second")
        if not math.isclose(per_image, wall / num_images, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"runtime row {records_path}:{line_number} has inconsistent per-image time")
        if not math.isclose(throughput, num_images / wall, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"runtime row {records_path}:{line_number} has inconsistent throughput")
        _digest(row["result_sha256"], location=f"{records_path}:{line_number}.result_sha256")
        records.append(row)
    if (
        isinstance(manifest["num_records"], bool)
        or not isinstance(manifest["num_records"], int)
        or len(records) != manifest["num_records"]
        or len(records) != len(methods) * 4
    ):
        raise ValueError(f"{records_path} has an unexpected number of rows")
    identities = {(row["method"], row["repetition"]) for row in records}
    if identities != {
        (method, repetition) for method in methods for repetition in range(4)
    }:
        raise ValueError(f"{records_path} does not contain the complete method/trial grid")
    for method in methods:
        digests = {row["result_sha256"] for row in records if row["method"] == method}
        if len(digests) != 1:
            raise ValueError(f"{records_path} has nondeterministic results for {method}")
    return RuntimeCondition(records_path.resolve(), manifest_path.resolve(), manifest, tuple(records))


def _method_summary(condition: RuntimeCondition, method: str):
    rows = [row for row in condition.records if row["method"] == method]
    wall = np.asarray([row["wall_seconds"] for row in rows], dtype=float)
    per_image = wall / condition.manifest["num_selected_images"]
    megapixels = condition.manifest["total_selected_pixels"] / 1_000_000
    return {
        "method": method,
        "num_trials": len(rows),
        "panel_wall_seconds": {
            "min": float(np.min(wall)),
            "median": float(np.median(wall)),
            "max": float(np.max(wall)),
        },
        "milliseconds_per_image": float(1000 * np.median(per_image)),
        "milliseconds_per_megapixel": float(1000 * np.median(wall) / megapixels),
        "images_per_second": float(condition.manifest["num_selected_images"] / np.median(wall)),
    }


def analyze_condition(condition: RuntimeCondition):
    summaries = {
        method: _method_summary(condition, method) for method in condition.methods
    }
    exact_time = summaries["dice_exact"]["panel_wall_seconds"]["median"]
    joint_time = summaries["m32_joint"]["panel_wall_seconds"]["median"]
    selected = condition.manifest["selected_samples"]
    memory = condition.manifest["memory"]
    environment = condition.manifest["environment"]
    return {
        "dataset": condition.manifest["dataset"],
        "condition": condition.manifest["condition"],
        "model": condition.manifest["model"],
        "is_target_condition": condition.key in TARGET_CONDITIONS,
        "num_selected_images": condition.manifest["num_selected_images"],
        "selected_pixel_count": {
            "min": min(row["pixels"] for row in selected),
            "median": float(median(row["pixels"] for row in selected)),
            "max": max(row["pixels"] for row in selected),
            "total": condition.manifest["total_selected_pixels"],
        },
        "methods": summaries,
        "m32_joint_over_dice_exact_time_ratio": float(joint_time / exact_time),
        "process_peak_rss_bytes": memory["peak_rss_bytes"],
        "process_peak_rss_scope": memory["scope"],
        "hardware": {
            "cpu_model": environment.get("cpu_model"),
            "partition": environment.get("slurm", {}).get("partition"),
            "node_list": environment.get("slurm", {}).get("node_list"),
            "affinity_cpu_count": environment.get("affinity_cpu_count"),
        },
    }


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "selectseg/benchmark_binary_runtime.py",
        root / "scripts/analyze_binary.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_report(
    campaign_lock,
    input_paths,
    *,
    benchmark_lock=None,
    expected_benchmark_lock_sha256=None,
):
    if len(input_paths) != EXPECTED_ARTIFACT_COUNT or len(set(map(str, input_paths))) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError("runtime analysis requires exactly 16 distinct explicit inputs")
    lock_path, lock_sha256, lock = load_campaign_lock(campaign_lock)
    expected = tuple((entry["dataset"], entry["condition"]) for entry in lock["artifacts"])
    if expected != EXPECTED_CONDITIONS:
        raise ValueError("campaign lock conditions differ from the declared 16-condition benchmark")
    by_key = {}
    input_provenance = []
    spec_hashes = set()
    source_hashes = set()
    protocol_versions = set()
    method_sets = set()
    benchmark_lock_hashes = set()
    benchmark_lock_paths = set()
    package_environments = set()
    for path in input_paths:
        condition = load_runtime_condition(path)
        if condition.key in by_key:
            raise ValueError(f"duplicate runtime condition {condition.key}")
        by_key[condition.key] = condition
        entry = lock["artifacts"][expected.index(condition.key)]
        provenance = condition.manifest["provenance"]
        if provenance["campaign_lock_sha256"] != lock_sha256:
            raise ValueError(f"runtime condition {condition.key} binds a different campaign lock")
        if provenance["campaign_id"] != lock["campaign_id"]:
            raise ValueError(f"runtime condition {condition.key} binds a different campaign_id")
        if provenance["artifact_manifest_sha256"] != entry["manifest_sha256"]:
            raise ValueError(f"runtime condition {condition.key} binds a different artifact")
        if provenance["artifact_id"] != entry["artifact_id"]:
            raise ValueError(f"runtime condition {condition.key} binds a different artifact_id")
        if provenance["artifact_source_sha256"] != entry["source_sha256"]:
            raise ValueError(f"runtime condition {condition.key} binds different artifact source bytes")
        for field in ("dataset", "condition", "model", "split"):
            if condition.manifest[field] != entry[field]:
                raise ValueError(
                    f"runtime condition {condition.key} has a different artifact {field}"
                )
        if provenance["estimator_spec_sha256"] != lock["estimator"]["spec_sha256"]:
            raise ValueError(f"runtime condition {condition.key} binds a different estimator")
        if provenance["estimator_id"] != lock["estimator"]["estimator_id"]:
            raise ValueError(f"runtime condition {condition.key} binds a different estimator_id")
        artifact_path = _project_path(lock_path, entry["manifest_path"])
        frozen = load_binary_artifact(artifact_path, validate_payloads=False)
        expected_selected = _selected_entries(frozen.manifest, 16)
        if condition.manifest["selected_samples"] != expected_selected:
            raise ValueError(f"runtime condition {condition.key} used a different sample panel")
        spec_hashes.add(provenance["benchmark_spec_sha256"])
        source_hashes.add(provenance["runtime_source_sha256"])
        package_environments.add(
            tuple(sorted(condition.manifest["environment"]["packages"].items()))
        )
        protocol_versions.add(condition.protocol_version)
        method_sets.add(condition.methods)
        if condition.protocol_version == 2:
            benchmark_lock_hashes.add(provenance["benchmark_lock_sha256"])
            benchmark_lock_paths.add(Path(provenance["benchmark_lock_path"]).resolve())
        input_provenance.append(
            {
                "dataset": condition.key[0],
                "condition": condition.key[1],
                "records_path": condition.records_path.as_posix(),
                "records_sha256": _sha256(condition.records_path),
                "manifest_path": condition.manifest_path.as_posix(),
                "manifest_sha256": _sha256(condition.manifest_path),
            }
        )
    if tuple(by_key) != expected and set(by_key) != set(expected):
        raise ValueError("runtime inputs do not cover the locked 16 conditions")
    if (
        len(spec_hashes) != 1
        or len(source_hashes) != 1
        or len(protocol_versions) != 1
        or len(method_sets) != 1
        or len(package_environments) != 1
    ):
        raise ValueError(
            "runtime conditions were not produced by one spec, source, and "
            "package environment"
        )
    protocol_version = next(iter(protocol_versions))
    methods = next(iter(method_sets))
    if protocol_version == 2:
        if not benchmark_lock or not expected_benchmark_lock_sha256:
            raise ValueError("runtime ladder analysis requires its immutable v2 lock")
        ladder_binding = load_runtime_ladder_lock(
            benchmark_lock,
            expected_sha256=expected_benchmark_lock_sha256,
        )
        locked = ladder_binding["lock"]
        if benchmark_lock_hashes != {ladder_binding["sha256"]}:
            raise ValueError("runtime ladder artifacts bind a different v2 lock")
        if benchmark_lock_paths != {ladder_binding["path"]}:
            raise ValueError("runtime ladder artifacts name a different v2 lock path")
        if locked["canonical_campaign_lock"]["sha256"] != lock_sha256:
            raise ValueError("runtime ladder lock binds a different canonical campaign")
        if locked["canonical_campaign_lock"]["campaign_id"] != lock["campaign_id"]:
            raise ValueError("runtime ladder lock binds a different campaign_id")
        if locked["spec"]["sha256"] != next(iter(spec_hashes)):
            raise ValueError("runtime ladder inputs bind a different benchmark spec")
        if locked["estimator_spec"]["sha256"] != lock["estimator"]["spec_sha256"]:
            raise ValueError("runtime ladder lock binds a different estimator")
        expected_runtime_source_sha256 = runtime_source_fingerprint()
        if source_hashes != {expected_runtime_source_sha256}:
            raise ValueError(
                "runtime ladder inputs bind a different locked runtime source"
            )
    else:
        if benchmark_lock or expected_benchmark_lock_sha256:
            raise ValueError("locked runtime v1 analysis does not consume a v2 lock")
        if benchmark_lock_hashes:
            raise ValueError("locked runtime v1 artifacts unexpectedly name a v2 lock")
        ladder_binding = None
    conditions = [analyze_condition(by_key[key]) for key in EXPECTED_CONDITIONS]
    input_provenance.sort(
        key=lambda row: EXPECTED_CONDITIONS.index(
            (row["dataset"], row["condition"])
        )
    )
    identity = {
        "campaign_lock_sha256": lock_sha256,
        "benchmark_spec_sha256": next(iter(spec_hashes)),
        "runtime_source_sha256": next(iter(source_hashes)),
        "analysis_source_sha256": _source_fingerprint(),
        "input_manifest_sha256": [row["manifest_sha256"] for row in input_provenance],
        "runtime_protocol_version": protocol_version,
    }
    if ladder_binding is not None:
        identity["benchmark_lock_sha256"] = ladder_binding["sha256"]
    analysis_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    targets = [row for row in conditions if row["is_target_condition"]]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_ARTIFACT_TYPE,
        "analysis_id": analysis_id,
        "scope": {
            "timing_status": "descriptive hardware-dependent wall-clock benchmark",
            "timed": "confidence computation on a preloaded deterministic panel",
            "excluded": "model inference and artifact I/O",
            "complexity_status": (
                "separate from algorithmic complexity: Dice-Exact sorts N pixels "
                "in O(N log N); each joint midpoint method evaluates its declared "
                "number of candidate masks and boundary-distance computations"
            ),
        },
        "condition_sets": {
            "all_conditions": [f"{a}/{b}" for a, b in EXPECTED_CONDITIONS],
            "target_conditions": [
                f"{row['dataset']}/{row['condition']}" for row in targets
            ],
            "num_conditions": len(conditions),
            "num_target_conditions": len(targets),
        },
        "provenance": {
            **identity,
            "campaign_lock_path": lock_path.as_posix(),
            "inputs": input_provenance,
        },
        "target_ranges": {
            **{
                f"{method}_milliseconds_per_image": {
                    "min": min(
                        row["methods"][method]["milliseconds_per_image"]
                        for row in targets
                    ),
                    "max": max(
                        row["methods"][method]["milliseconds_per_image"]
                        for row in targets
                    ),
                }
                for method in methods
            },
            "m32_joint_over_dice_exact_time_ratio": {
                "min": min(
                    row["m32_joint_over_dice_exact_time_ratio"] for row in targets
                ),
                "max": max(
                    row["m32_joint_over_dice_exact_time_ratio"] for row in targets
                ),
            },
        },
        "conditions": conditions,
    }


def write_report(report: Mapping[str, Any], output_path: str | os.PathLike[str]):
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime analysis: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    report = build_report(
        args.campaign_lock,
        args.inputs,
        benchmark_lock=args.benchmark_lock,
        expected_benchmark_lock_sha256=args.expected_benchmark_lock_sha256,
    )
    path = write_report(report, args.output)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
