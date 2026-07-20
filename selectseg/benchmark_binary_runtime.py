"""Benchmark confidence computation on one locked frozen-map condition.

This auxiliary workflow times no model inference and no artifact I/O.  It
loads a deterministic size-stratified panel before creating the timer, warms
both confidence implementations, and then measures four whole-panel trials
with eight Python workers and one native thread per worker.  The two timed
workloads are deliberately different outputs:

``m32_joint``
    the production M=32 computation of Dice, nHD, and nHD95 confidence in one
    shared pass;
``dice_exact``
    the exact uniform-threshold Dice confidence alone.

Results are descriptive wall-clock measurements tied to one hardware and
software environment.  They do not replace algorithmic complexity claims.
Outputs are content-addressed, atomically published, and never overwritten.

The immutable v2 ladder keeps the same panel and execution constraints while
timing the joint three-loss workload separately at M=2, M=8, and M=32.  The
locked v1 protocol and its artifacts remain valid and are never rewritten.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import string
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.submit_binary_simulations import _project_path, load_campaign_lock
from selectseg.binary_artifacts import (
    _load_sample,
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
)
from selectseg.binary_baselines import exact_levelset_dice_confidence
from selectseg.score_binary_simulation import score_binary_sample
from selectseg.threshold_estimators import (
    build_threshold_rule,
    load_estimator_spec,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCHEMA_VERSION = 1
RUNTIME_ARTIFACT_TYPE = "selectseg.binary_confidence_runtime"
EXPECTED_ARTIFACT_COUNT = 16
EXPECTED_SPEC_ID = "binary-confidence-runtime-v1"
EXPECTED_LADDER_SPEC_ID = "binary-confidence-runtime-ladder-v2"
METHODS_V1 = ("m32_joint", "dice_exact")
METHODS_V2 = ("m2_joint", "m8_joint", "m32_joint", "dice_exact")
# Public compatibility name used by the locked v1 tests and renderer.
METHODS = METHODS_V1
BENCHMARK_SOURCE_PATHS = (
    "selectseg/benchmark_binary_runtime.py",
    "scripts/submit_binary_simulations.py",
    "selectseg/score_binary_simulation.py",
    "selectseg/score_binary_common.py",
    "selectseg/binary_baselines.py",
    "selectseg/binary_boundary.py",
    "selectseg/binary_framework.py",
    "selectseg/binary_artifacts.py",
    "selectseg/threshold_estimators.py",
)
LADDER_LOCK_SOURCE_PATHS = (
    *BENCHMARK_SOURCE_PATHS,
    "scripts/slurm/benchmark_binary_runtime.sbatch",
    "scripts/slurm/env.sh",
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset",
        "condition",
        "method",
        "repetition",
        "order_position",
        "num_images",
        "total_pixels",
        "wall_seconds",
        "seconds_per_image",
        "images_per_second",
        "result_sha256",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument("--expected-campaign-lock-sha256", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    parser.add_argument("--estimator-spec", required=True)
    parser.add_argument("--expected-estimator-spec-sha256", required=True)
    parser.add_argument("--benchmark-spec", required=True)
    parser.add_argument("--expected-benchmark-spec-sha256", required=True)
    parser.add_argument("--benchmark-lock")
    parser.add_argument("--expected-benchmark-lock-sha256")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _strict_sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in string.hexdigits for character in value):
        raise ValueError(f"{location} must contain 64 hexadecimal characters")
    return normalized


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: str | os.PathLike[str], *, name: str) -> tuple[dict, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular, non-symlink file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {name} {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_benchmark_spec(path: str | os.PathLike[str], expected_sha256: str):
    spec, observed_sha256 = _load_strict_json(path, name="benchmark spec")
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "benchmark spec SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    if set(spec) != {
        "spec_schema_version",
        "benchmark_id",
        "protocol",
        "execution",
        "output_root",
    }:
        raise ValueError("benchmark spec has an unexpected top-level schema")
    schema_version = spec["spec_schema_version"]
    benchmark_id = spec["benchmark_id"]
    if (schema_version, benchmark_id) not in {
        (1, EXPECTED_SPEC_ID),
        (2, EXPECTED_LADDER_SPEC_ID),
    }:
        raise ValueError("unsupported benchmark specification")
    protocol = spec["protocol"]
    if schema_version == 1:
        expected_protocol = {
            "gamma": 0.5,
            "m": 32,
            "seed": 0,
            "methods": list(METHODS_V1),
            "warmup_repetitions": 1,
            "measured_repetitions": 4,
            "measured_order": "alternating",
            "timer": "time.perf_counter_ns",
        }
        expected_protocol_fields = {
            *expected_protocol,
            "sample_selection",
        }
        expected_output_root = "outputs/binary_runtime"
    else:
        expected_protocol = {
            "gamma": 0.5,
            "m_values": [2, 8, 32],
            "seed": 0,
            "methods": list(METHODS_V2),
            "warmup_repetitions": 1,
            "measured_repetitions": 4,
            "measured_order": "williams_latin_v1",
            "timer": "time.perf_counter_ns",
            "cache_policy": (
                "selected arrays are preloaded; no confidence or boundary-distance "
                "result is reused across timed methods"
            ),
        }
        expected_protocol_fields = {
            *expected_protocol,
            "sample_selection",
        }
        expected_output_root = "outputs/binary_runtime_ladder_v2"
    if not isinstance(protocol, dict) or set(protocol) != expected_protocol_fields:
        raise ValueError("benchmark protocol has an unexpected schema")
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError(f"benchmark protocol field {field!r} is not predeclared")
    selection = protocol["sample_selection"]
    if selection != {
        "strategy": "deterministic_pixel_count_quantiles",
        "max_samples": 16,
        "include_size_endpoints": True,
        "execution_order": "manifest_index",
    }:
        raise ValueError("benchmark sample-selection protocol is not predeclared")
    execution = spec["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "benchmark_workers",
        "native_threads_per_worker",
        "thread_environment",
    }:
        raise ValueError("benchmark execution has an unexpected schema")
    if execution["benchmark_workers"] != 8 or execution["native_threads_per_worker"] != 1:
        raise ValueError("benchmark execution must use eight workers and one native thread")
    expected_environment = {variable: "1" for variable in THREAD_VARIABLES}
    if execution["thread_environment"] != expected_environment:
        raise ValueError("benchmark thread environment is not predeclared")
    if spec["output_root"] != expected_output_root:
        raise ValueError("benchmark output root is not predeclared")
    return spec, observed_sha256


def _methods_for_spec(spec: Mapping[str, Any]) -> tuple[str, ...]:
    methods = tuple(spec["protocol"]["methods"])
    expected = METHODS_V1 if spec["spec_schema_version"] == 1 else METHODS_V2
    if methods != expected:
        raise ValueError("benchmark method order differs from its schema")
    return methods


def _m_values_for_spec(spec: Mapping[str, Any]) -> tuple[int, ...]:
    protocol = spec["protocol"]
    if spec["spec_schema_version"] == 1:
        return (int(protocol["m"]),)
    return tuple(int(value) for value in protocol["m_values"])


def runtime_protocol_version(protocol: Mapping[str, Any]) -> int:
    """Return the exact supported runtime protocol version, or reject it."""

    selection = {
        "strategy": "deterministic_pixel_count_quantiles",
        "max_samples": 16,
        "include_size_endpoints": True,
        "execution_order": "manifest_index",
    }
    v1 = {
        "gamma": 0.5,
        "m": 32,
        "seed": 0,
        "methods": list(METHODS_V1),
        "sample_selection": selection,
        "warmup_repetitions": 1,
        "measured_repetitions": 4,
        "measured_order": "alternating",
        "timer": "time.perf_counter_ns",
    }
    v2 = {
        "gamma": 0.5,
        "m_values": [2, 8, 32],
        "seed": 0,
        "methods": list(METHODS_V2),
        "sample_selection": selection,
        "warmup_repetitions": 1,
        "measured_repetitions": 4,
        "measured_order": "williams_latin_v1",
        "timer": "time.perf_counter_ns",
        "cache_policy": (
            "selected arrays are preloaded; no confidence or boundary-distance "
            "result is reused across timed methods"
        ),
    }
    if protocol == v1:
        return 1
    if protocol == v2:
        return 2
    raise ValueError("unsupported runtime protocol")


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_runtime_ladder_lock(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load the immutable v2 benchmark lock and revalidate every bound byte."""

    lock, observed_sha256 = _load_strict_json(path, name="runtime ladder lock")
    expected_sha256 = _strict_sha256(
        expected_sha256, location="expected runtime ladder lock SHA-256"
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("runtime ladder lock SHA-256 mismatch")
    expected_fields = {
        "lock_schema_version",
        "benchmark_id",
        "spec",
        "canonical_campaign_lock",
        "estimator_spec",
        "source_files",
    }
    if set(lock) != expected_fields:
        raise ValueError("runtime ladder lock has an unexpected schema")
    if (
        lock["lock_schema_version"] != 1
        or lock["benchmark_id"] != EXPECTED_LADDER_SPEC_ID
    ):
        raise ValueError("unsupported runtime ladder lock")
    binding_fields = {"path", "sha256"}
    for name in ("spec", "estimator_spec"):
        binding = lock[name]
        if not isinstance(binding, dict) or set(binding) != binding_fields:
            raise ValueError(f"runtime ladder lock {name} binding is malformed")
        source = _resolve_project_path(binding["path"])
        expected = _strict_sha256(binding["sha256"], location=f"lock.{name}.sha256")
        _verify_regular_file(source, expected, name=f"locked {name}")
    campaign = lock["canonical_campaign_lock"]
    if not isinstance(campaign, dict) or set(campaign) != {
        "path",
        "sha256",
        "campaign_id",
    }:
        raise ValueError("runtime ladder campaign-lock binding is malformed")
    campaign_path = _resolve_project_path(campaign["path"])
    campaign_sha = _strict_sha256(
        campaign["sha256"], location="lock.canonical_campaign_lock.sha256"
    )
    _verify_regular_file(campaign_path, campaign_sha, name="locked campaign lock")
    if not isinstance(campaign["campaign_id"], str) or not campaign["campaign_id"]:
        raise ValueError("runtime ladder lock campaign_id is empty")
    sources = lock["source_files"]
    if not isinstance(sources, list) or len(sources) != len(LADDER_LOCK_SOURCE_PATHS):
        raise ValueError("runtime ladder lock source-file set is incomplete")
    observed_paths = []
    for index, binding in enumerate(sources):
        if not isinstance(binding, dict) or set(binding) != binding_fields:
            raise ValueError(f"runtime ladder source binding {index} is malformed")
        source = _resolve_project_path(binding["path"])
        expected = _strict_sha256(
            binding["sha256"], location=f"lock.source_files[{index}].sha256"
        )
        _verify_regular_file(source, expected, name=f"locked source file {index}")
        try:
            observed_paths.append(source.relative_to(PROJECT_ROOT).as_posix())
        except ValueError as error:
            raise ValueError("runtime ladder source path is not project-relative") from error
    if tuple(observed_paths) != LADDER_LOCK_SOURCE_PATHS:
        raise ValueError("runtime ladder lock source paths differ from the fixed set")
    return {
        "path": Path(path).resolve(),
        "sha256": observed_sha256,
        "lock": lock,
    }


def _validate_thread_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    expected = spec["execution"]["thread_environment"]
    observed = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if observed != expected:
        differences = ", ".join(
            f"{name}={observed[name]!r} (expected {expected[name]!r})"
            for name in THREAD_VARIABLES
            if observed[name] != expected[name]
        )
        raise ValueError(f"native thread environment is not fixed: {differences}")
    return {name: str(value) for name, value in observed.items()}


def _verify_regular_file(path: str | os.PathLike[str], expected_sha256: str, *, name: str):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular, non-symlink file: {source}")
    observed = sha256_file(source)
    if observed != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _validate_bindings(args, benchmark_spec):
    lock_path, lock_sha256, lock = load_campaign_lock(args.campaign_lock)
    if lock_sha256 != args.expected_campaign_lock_sha256:
        raise ValueError("campaign lock SHA-256 differs from the requested binding")
    if lock["campaign_id"] != args.campaign_id:
        raise ValueError("campaign_id differs from the campaign lock")
    if len(lock["artifacts"]) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError("runtime campaign requires exactly 16 locked artifacts")
    protocol = benchmark_spec["protocol"]
    if lock["protocol"]["gamma_values"] != [protocol["gamma"]]:
        raise ValueError("benchmark gamma differs from the canonical campaign")
    missing_m = sorted(
        set(_m_values_for_spec(benchmark_spec)) - set(lock["protocol"]["m_values"])
    )
    if missing_m:
        raise ValueError(
            f"benchmark M values are absent from the canonical campaign: {missing_m}"
        )
    if protocol["seed"] not in lock["protocol"]["seeds"]:
        raise ValueError("benchmark seed is absent from the canonical campaign")

    estimator_binding = lock["estimator"]
    locked_estimator = _project_path(lock_path, estimator_binding["spec_path"])
    requested_estimator = Path(args.estimator_spec).resolve()
    if locked_estimator != requested_estimator:
        raise ValueError("benchmark estimator path differs from the campaign lock")
    if estimator_binding["spec_sha256"] != args.expected_estimator_spec_sha256:
        raise ValueError("benchmark estimator hash differs from the campaign lock")

    requested_artifact = Path(args.artifact_manifest).resolve()
    matches = []
    for entry in lock["artifacts"]:
        locked_path = _project_path(lock_path, entry["manifest_path"])
        if locked_path == requested_artifact:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("artifact manifest must occur exactly once in the campaign lock")
    if matches[0]["manifest_sha256"] != args.expected_artifact_manifest_sha256:
        raise ValueError("artifact hash differs from the campaign lock")
    return lock_path, lock_sha256, lock, matches[0]


def _selected_entries(manifest: Mapping[str, Any], max_samples: int):
    entries = manifest["samples"]
    ordered = sorted(
        range(len(entries)),
        key=lambda index: (
            int(entries[index]["height"]) * int(entries[index]["width"]),
            index,
        ),
    )
    count = min(max_samples, len(ordered))
    if count == 1:
        size_positions = [0]
    else:
        size_positions = [
            (rank * (len(ordered) - 1)) // (count - 1) for rank in range(count)
        ]
    selected_indices = sorted(ordered[position] for position in size_positions)
    selected = []
    size_rank = {ordered[position]: rank for rank, position in enumerate(size_positions)}
    for index in selected_indices:
        entry = entries[index]
        selected.append(
            {
                "index": index,
                "sample_id": entry["sample_id"],
                "height": int(entry["height"]),
                "width": int(entry["width"]),
                "pixels": int(entry["height"]) * int(entry["width"]),
                "size_quantile_rank": size_rank[index],
            }
        )
    if len(selected) != count or len({row["index"] for row in selected}) != count:
        raise RuntimeError("deterministic size-quantile selection produced duplicates")
    return selected


def runtime_source_fingerprint() -> str:
    """Hash the fixed local Python dependency set used by the timed process."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in BENCHMARK_SOURCE_PATHS:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"runtime source dependency is missing: {source}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# Preserve the private name used by the immutable v1 implementation.
_source_fingerprint = runtime_source_fingerprint


def _runtime_id(
    *,
    campaign_sha256,
    artifact_sha256,
    estimator_sha256,
    spec_sha256,
    source_sha256,
    benchmark_lock_sha256=None,
):
    identity = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "artifact_type": RUNTIME_ARTIFACT_TYPE,
        "campaign_sha256": campaign_sha256,
        "artifact_sha256": artifact_sha256,
        "estimator_sha256": estimator_sha256,
        "spec_sha256": spec_sha256,
        "source_sha256": source_sha256,
    }
    if benchmark_lock_sha256 is not None:
        identity["benchmark_lock_sha256"] = _strict_sha256(
            benchmark_lock_sha256, location="benchmark lock identity SHA-256"
        )
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _portable_path(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _cpu_model() -> str | None:
    source = Path("/proc/cpuinfo")
    if source.is_file():
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    value = platform.processor().strip()
    return value or None


def _peak_rss_bytes() -> tuple[int | None, str]:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None, "resource.getrusage unavailable"
    if value < 0:
        return None, "resource.getrusage returned a negative value"
    multiplier = 1 if sys.platform == "darwin" else 1024
    return value * multiplier, (
        "process high-water RSS; includes Python, selected decoded maps, warm-up, "
        "and measured confidence computation; excludes model inference and "
        "unselected artifact payloads"
    )


def _score_one(
    method: str,
    sample,
    *,
    runtime_id: str,
    class_index: int,
    class_name: str,
    gamma: float,
    rules: Mapping[int, Any],
):
    if method.startswith("m") and method.endswith("_joint"):
        try:
            m = int(method[1 : -len("_joint")])
        except ValueError as error:
            raise ValueError(f"invalid joint runtime method {method!r}") from error
        if m not in rules:
            raise ValueError(f"runtime method {method!r} lacks a threshold rule")
        row = score_binary_sample(
            sample,
            simulation_id=runtime_id,
            class_index=class_index,
            class_name=class_name,
            gamma=gamma,
            threshold_rule=rules[m],
        )
        result = (
            float(row[f"confidence_dice_m{m}"]),
            float(row[f"confidence_nhd_m{m}"]),
            float(row[f"confidence_nhd95_m{m}"]),
        )
    elif method == "dice_exact":
        probability = np.asarray(sample.foreground_probability)
        prediction = probability >= gamma
        result = (
            float(exact_levelset_dice_confidence(probability, prediction)),
        )
    else:
        raise ValueError(f"unknown runtime method {method!r}")
    values = np.asarray(result, dtype=float)
    if not np.isfinite(values).all() or np.any((values < -1) | (values > 0)):
        raise RuntimeError(f"runtime method {method} produced an invalid confidence")
    return result


def _panel_result(executor, method, samples, **kwargs):
    futures = [
        executor.submit(_score_one, method, sample, **kwargs) for sample in samples
    ]
    return [future.result() for future in futures]


def _result_digest(method: str, selected, values) -> str:
    payload = [
        {
            "sample_id": selected[index]["sample_id"],
            "method": method,
            "values": list(result),
        }
        for index, result in enumerate(values)
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _measurement_order(
    methods: tuple[str, ...], repetition: int, measured_order: str
) -> tuple[str, ...]:
    if measured_order == "alternating":
        return methods if repetition % 2 == 0 else tuple(reversed(methods))
    if measured_order == "williams_latin_v1":
        if len(methods) != 4:
            raise ValueError("Williams runtime order requires exactly four methods")
        # A balanced Williams square: every method occupies every position and
        # every ordered first-order carryover pair occurs exactly once.
        base = (0, 1, 3, 2)
        return tuple(methods[(index + repetition) % 4] for index in base)
    raise ValueError(f"unknown measured-order design {measured_order!r}")


def _write_json(path: Path, value: Mapping[str, Any]):
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def run_benchmark(args):
    """Run one locked, append-only confidence benchmark."""

    for name in (
        "expected_campaign_lock_sha256",
        "expected_artifact_manifest_sha256",
        "expected_estimator_spec_sha256",
        "expected_benchmark_spec_sha256",
    ):
        setattr(args, name, _strict_sha256(getattr(args, name), location=f"--{name}"))
    if not isinstance(args.campaign_id, str) or not args.campaign_id.strip():
        raise ValueError("--campaign-id must be non-empty")
    _verify_regular_file(
        args.campaign_lock, args.expected_campaign_lock_sha256, name="campaign lock"
    )
    _verify_regular_file(
        args.artifact_manifest,
        args.expected_artifact_manifest_sha256,
        name="artifact manifest",
    )
    _verify_regular_file(
        args.estimator_spec,
        args.expected_estimator_spec_sha256,
        name="estimator spec",
    )
    benchmark_spec, benchmark_spec_sha256 = _validate_benchmark_spec(
        args.benchmark_spec, args.expected_benchmark_spec_sha256
    )
    if benchmark_spec["spec_schema_version"] == 2:
        if not args.benchmark_lock or not args.expected_benchmark_lock_sha256:
            raise ValueError(
                "runtime ladder v2 requires --benchmark-lock and its expected SHA-256"
            )
        benchmark_lock = load_runtime_ladder_lock(
            args.benchmark_lock,
            expected_sha256=args.expected_benchmark_lock_sha256,
        )
    else:
        if args.benchmark_lock or args.expected_benchmark_lock_sha256:
            raise ValueError("locked v1 runtime does not consume a v2 benchmark lock")
        benchmark_lock = None
    thread_environment = _validate_thread_environment(benchmark_spec)
    lock_path, campaign_sha256, lock, locked_artifact = _validate_bindings(
        args, benchmark_spec
    )

    if benchmark_lock is not None:
        lock_payload = benchmark_lock["lock"]
        if _resolve_project_path(lock_payload["spec"]["path"]) != Path(
            args.benchmark_spec
        ).resolve() or lock_payload["spec"]["sha256"] != benchmark_spec_sha256:
            raise ValueError("runtime ladder lock binds a different benchmark spec")
        campaign_binding = lock_payload["canonical_campaign_lock"]
        if (
            _resolve_project_path(campaign_binding["path"]) != lock_path.resolve()
            or campaign_binding["sha256"] != campaign_sha256
            or campaign_binding["campaign_id"] != lock["campaign_id"]
        ):
            raise ValueError("runtime ladder lock binds a different campaign")
        estimator_binding = lock_payload["estimator_spec"]
        if (
            _resolve_project_path(estimator_binding["path"])
            != Path(args.estimator_spec).resolve()
            or estimator_binding["sha256"]
            != args.expected_estimator_spec_sha256
        ):
            raise ValueError("runtime ladder lock binds a different estimator")

    estimator = load_estimator_spec(args.estimator_spec)
    if estimator.sha256 != args.expected_estimator_spec_sha256:
        raise RuntimeError("estimator loader returned an inconsistent SHA-256")
    rules = {
        m: build_threshold_rule(
            estimator,
            m=m,
            seed=benchmark_spec["protocol"]["seed"],
        )
        for m in _m_values_for_spec(benchmark_spec)
    }
    if any(rule.estimator_id != "midpoint-v1" for rule in rules.values()):
        raise ValueError("runtime benchmark requires midpoint-v1")
    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise RuntimeError("artifact loader returned an inconsistent manifest hash")
    frozen = artifact.manifest
    for field in (
        "artifact_id",
        "dataset",
        "condition",
        "model",
        "split",
        "source_sha256",
        "sample_id_sha256",
        "num_samples",
    ):
        if frozen[field] != locked_artifact[field]:
            raise ValueError(f"locked artifact field {field!r} is inconsistent")

    source_sha256 = _source_fingerprint()
    runtime_id = _runtime_id(
        campaign_sha256=campaign_sha256,
        artifact_sha256=artifact.manifest_sha256,
        estimator_sha256=estimator.sha256,
        spec_sha256=benchmark_spec_sha256,
        source_sha256=source_sha256,
        benchmark_lock_sha256=(
            None if benchmark_lock is None else benchmark_lock["sha256"]
        ),
    )
    output_root = Path(benchmark_spec["output_root"])
    output_dir = output_root / frozen["dataset"] / frozen["condition"] / runtime_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"runtime benchmark output already exists: {output_dir}")

    selection_spec = benchmark_spec["protocol"]["sample_selection"]
    selected = _selected_entries(frozen, selection_spec["max_samples"])
    samples = []
    for row in selected:
        entry = frozen["samples"][row["index"]]
        sample = _load_sample(artifact.artifact_dir, entry)
        if sample.sample_id != row["sample_id"] or sample.index != row["index"]:
            raise RuntimeError("selected artifact sample identity changed while loading")
        samples.append(sample)
    samples = tuple(samples)
    total_pixels = sum(row["pixels"] for row in selected)
    workers = benchmark_spec["execution"]["benchmark_workers"]
    method_kwargs = {
        "runtime_id": runtime_id,
        "class_index": frozen["class_index"],
        "class_name": frozen["class_name"],
        "gamma": benchmark_spec["protocol"]["gamma"],
        "rules": rules,
    }
    methods = _methods_for_spec(benchmark_spec)
    records = []
    reference_digests: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(benchmark_spec["protocol"]["warmup_repetitions"]):
            for method in methods:
                _panel_result(executor, method, samples, **method_kwargs)
        for repetition in range(benchmark_spec["protocol"]["measured_repetitions"]):
            order = _measurement_order(
                methods,
                repetition,
                benchmark_spec["protocol"]["measured_order"],
            )
            for order_position, method in enumerate(order):
                gc.collect()
                start_ns = time.perf_counter_ns()
                values = _panel_result(executor, method, samples, **method_kwargs)
                end_ns = time.perf_counter_ns()
                wall_seconds = (end_ns - start_ns) / 1_000_000_000
                if not math.isfinite(wall_seconds) or wall_seconds <= 0:
                    raise RuntimeError("runtime timer returned a non-positive duration")
                result_sha256 = _result_digest(method, selected, values)
                previous = reference_digests.setdefault(method, result_sha256)
                if previous != result_sha256:
                    raise RuntimeError(f"{method} results changed across repetitions")
                record = {
                    "schema_version": RUNTIME_SCHEMA_VERSION,
                    "run_id": runtime_id,
                    "dataset": frozen["dataset"],
                    "condition": frozen["condition"],
                    "method": method,
                    "repetition": repetition,
                    "order_position": order_position,
                    "num_images": len(samples),
                    "total_pixels": total_pixels,
                    "wall_seconds": wall_seconds,
                    "seconds_per_image": wall_seconds / len(samples),
                    "images_per_second": len(samples) / wall_seconds,
                    "result_sha256": result_sha256,
                }
                if set(record) != RECORD_FIELDS:
                    raise RuntimeError("runtime benchmark produced an unexpected row schema")
                records.append(record)

    peak_rss_bytes, peak_rss_scope = _peak_rss_bytes()
    packages = _package_versions()
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "artifact_type": RUNTIME_ARTIFACT_TYPE,
        "run_id": runtime_id,
        "runtime_id": runtime_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": frozen["dataset"],
        "condition": frozen["condition"],
        "model": frozen["model"],
        "split": frozen["split"],
        "benchmark_scope": {
            "timed": "confidence computation, task dispatch, and result collection",
            "excluded": [
                "model inference",
                "artifact manifest and payload I/O",
                "selected-panel construction",
                "thread-pool construction",
                "warm-up repetitions",
                "JSON serialization and publication",
            ],
            "interpretation": (
                "descriptive hardware-dependent wall-clock throughput; not an "
                "algorithmic-complexity result"
            ),
        },
        "protocol": benchmark_spec["protocol"],
        "execution": benchmark_spec["execution"],
        "selected_samples": selected,
        "num_selected_images": len(selected),
        "total_selected_pixels": total_pixels,
        "measurements_per_method": benchmark_spec["protocol"]["measured_repetitions"],
        "record_fields": sorted(RECORD_FIELDS),
        "environment": {
            "packages": packages,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "affinity_cpu_count": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "thread_environment": thread_environment,
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "node_list": os.environ.get("SLURM_JOB_NODELIST"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            },
        },
        "memory": {
            "peak_rss_bytes": peak_rss_bytes,
            "scope": peak_rss_scope,
            "method_attribution_available": False,
        },
        "provenance": {
            "campaign_id": lock["campaign_id"],
            "campaign_lock_path": _portable_path(lock_path),
            "campaign_lock_sha256": campaign_sha256,
            "artifact_manifest_path": _portable_path(artifact.manifest_path),
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "artifact_id": frozen["artifact_id"],
            "artifact_source_sha256": frozen["source_sha256"],
            "estimator_spec_path": _portable_path(estimator.path),
            "estimator_spec_sha256": estimator.sha256,
            "estimator_id": estimator.estimator_id,
            "benchmark_spec_path": _portable_path(args.benchmark_spec),
            "benchmark_spec_sha256": benchmark_spec_sha256,
            "runtime_source_sha256": source_sha256,
        },
        "command": [
            "python",
            "-m",
            "selectseg.benchmark_binary_runtime",
            *args.command_arguments,
        ],
    }
    if benchmark_lock is not None:
        manifest["provenance"].update(
            {
                "benchmark_lock_path": _portable_path(benchmark_lock["path"]),
                "benchmark_lock_sha256": benchmark_lock["sha256"],
            }
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{runtime_id}.tmp-", dir=output_dir.parent))
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    try:
        with records_path.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        manifest["num_records"] = len(records)
        manifest["records_sha256"] = sha256_file(records_path)
        _write_json(manifest_path, manifest)
        fsync_directory(staging)
        publish_directory_no_replace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir / "records.jsonl", output_dir / "manifest.json"


def main(argv: Sequence[str] | None = None):
    records_path, manifest_path = run_benchmark(parse_args(argv))
    print(f"saved {records_path}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
