"""Streamed, read-only diagnostics for one frozen binary-map artifact.

Pixelwise Brier score and fixed-bin ECE diagnose marginal foreground
probabilities only.  They cannot identify a joint mask posterior or validate
the shared-threshold coupling.  Label-dependent descriptors are isolated from
prediction-only descriptors and must never be used to tune confidence scores.

Input validation is delegated to the canonical frozen-artifact loader.  This
module processes one native-resolution sample at a time, bounds temporary
pixel memory with chunks, and atomically publishes a no-overwrite result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from selectseg.artifacts import (
    FrozenBinaryArtifact,
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
    sample_id_sha256,
    sha256_file,
)


SCHEMA_VERSION = 1
DESCRIPTOR_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_artifact_diagnostics"
SUMMARY_NAME = "diagnostics.json"
DESCRIPTORS_NAME = "descriptors.jsonl"
DEFAULT_ECE_BINS = 15
DEFAULT_GAMMA = 0.5
DEFAULT_PIXEL_CHUNK_SIZE = 1_048_576
LADDER_M = 32
LADDER_NODES = (np.arange(LADDER_M, dtype=np.float64) + 0.5) / LADDER_M
LADDER_NODES.setflags(write=False)

def _keys(names: str) -> frozenset[str]:
    return frozenset(names.split())


_SUMMARY_KEYS = _keys(
    "schema_version artifact_type diagnostic_id created_utc source_sha256 "
    "environment artifact scope specification counts marginal_calibration "
    "hard_prediction truth shared_threshold_ladder descriptors command"
)
_SCOPE = {
    "purpose": "held-out descriptive diagnostics; not confidence fitting",
    "marginal_calibration_limitation": (
        "pixelwise Brier score and fixed-bin ECE assess marginal calibration "
        "only; they do not identify a joint mask posterior or validate "
        "shared-threshold coupling"
    ),
    "label_use_policy": (
        "label_outcomes may support predeclared descriptive failure-case "
        "selection only and must not tune or define a confidence score"
    ),
}


@dataclass(frozen=True)
class BinaryDiagnostics:
    """A validated summary and the SHA-256 of its exact JSON bytes."""

    summary_path: Path
    summary: Mapping[str, Any]
    summary_sha256: str


@dataclass(frozen=True)
class _SampleStats:
    pixels: int
    brier_sum: float
    bin_counts: np.ndarray
    bin_probability_sums: np.ndarray
    bin_truth_sums: np.ndarray
    probability_sum: float
    truth_foreground: int
    hard_foreground: int
    hard_intersection: int
    hard_errors: int
    ladder_intervals: np.ndarray


class _Aggregate:
    """Small accumulators independent of native image size."""

    def __init__(self, ece_bins: int):
        self.images = self.pixels = 0
        self.brier_sum = 0.0
        self.bin_counts = np.zeros(ece_bins, dtype=np.int64)
        self.bin_probability_sums = np.zeros(ece_bins, dtype=np.float64)
        self.bin_truth_sums = np.zeros(ece_bins, dtype=np.int64)
        self.hard_foreground = self.hard_empty = 0
        self.truth_foreground = self.truth_empty = 0
        self.hard_fraction_sum = self.truth_fraction_sum = 0.0
        self.ladder_intervals = np.zeros(LADDER_M + 1, dtype=np.int64)
        self.ladder_fraction_sums = np.zeros(LADDER_M, dtype=np.float64)
        self.ladder_empty = np.zeros(LADDER_M, dtype=np.int64)
        self.change_fraction_sums = np.zeros(LADDER_M - 1, dtype=np.float64)
        self.change_zero = np.zeros(LADDER_M - 1, dtype=np.int64)
        self.change_max = np.zeros(LADDER_M - 1, dtype=np.float64)
        self.distinct_histogram = np.zeros(LADDER_M + 1, dtype=np.int64)

    def add(self, stats: _SampleStats) -> None:
        foreground = _ladder_foreground(stats.ladder_intervals)
        changes = stats.ladder_intervals[1:LADDER_M]
        foreground_fractions, change_fractions = (
            foreground / stats.pixels, changes / stats.pixels
        )
        self.images += 1
        self.pixels += stats.pixels
        self.brier_sum += stats.brier_sum
        self.bin_counts += stats.bin_counts
        self.bin_probability_sums += stats.bin_probability_sums
        self.bin_truth_sums += stats.bin_truth_sums
        self.hard_foreground += stats.hard_foreground
        self.hard_fraction_sum += stats.hard_foreground / stats.pixels
        self.hard_empty += int(stats.hard_foreground == 0)
        self.truth_foreground += stats.truth_foreground
        self.truth_fraction_sum += stats.truth_foreground / stats.pixels
        self.truth_empty += int(stats.truth_foreground == 0)
        self.ladder_intervals += stats.ladder_intervals
        self.ladder_fraction_sums += foreground_fractions
        self.ladder_empty += foreground == 0
        self.change_fraction_sums += change_fractions
        self.change_zero += changes == 0
        self.change_max = np.maximum(self.change_max, change_fractions)
        self.distinct_histogram[1 + int(np.count_nonzero(changes))] += 1


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    parser.add_argument("--output-root", default="outputs/binary_diagnostics")
    parser.add_argument("--decision-threshold", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    parser.add_argument(
        "--pixel-chunk-size", type=int, default=DEFAULT_PIXEL_CHUNK_SIZE
    )
    parser.add_argument(
        "--write-descriptors",
        action="store_true",
        help="write deterministic, label-separated per-image plot data",
    )
    return parser.parse_args(argv)


def run_diagnostics(args, *, created_utc: str | None = None) -> Path:
    """Compute and atomically publish diagnostics for exactly one artifact."""

    _digest(args.expected_artifact_manifest_sha256,
            "--expected-artifact-manifest-sha256")
    gamma = _number(args.decision_threshold, "--decision-threshold", 0, 1)
    ece_bins = _integer(args.ece_bins, "--ece-bins", 1)
    chunk_size = _integer(args.pixel_chunk_size, "--pixel-chunk-size", 1)
    if type(args.write_descriptors) is not bool:
        raise TypeError("--write-descriptors must be boolean")

    # This is the single authority for manifest, path, payload hash, dtype,
    # shape, finite-domain, and ordered sample validation.
    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise ValueError(
            "artifact manifest SHA-256 mismatch: "
            f"expected {args.expected_artifact_manifest_sha256}, "
            f"got {artifact.manifest_sha256}"
        )
    source_sha256 = _source_fingerprint()
    environment = {
        "packages": {"python": platform.python_version(), "numpy": np.__version__},
        "device": "cpu",
    }
    specification = _specification(gamma, ece_bins, chunk_size)
    diagnostic_id = _diagnostic_id(
        artifact.manifest_sha256, source_sha256, environment, specification,
        args.write_descriptors
    )
    manifest = artifact.manifest
    parent = Path(args.output_root).joinpath(
        str(manifest["dataset"]), str(manifest["condition"]),
        str(manifest["artifact_id"])
    )
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / diagnostic_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"diagnostics already exist: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{diagnostic_id}.tmp-", dir=parent))

    try:
        aggregate = _Aggregate(ece_bins)
        descriptor_path = staging / DESCRIPTORS_NAME
        with ExitStack() as stack:
            descriptor_handle = (
                stack.enter_context(descriptor_path.open("x", encoding="utf-8"))
                if args.write_descriptors else None
            )
            for sample in artifact.iter_samples():
                stats = _sample_stats(
                    sample.foreground_probability, sample.truth, gamma, ece_bins,
                    chunk_size
                )
                aggregate.add(stats)
                if descriptor_handle is not None:
                    row = _descriptor(
                        diagnostic_id, str(manifest["artifact_id"]), sample.sample_id,
                        sample.index, sample.foreground_probability.shape, stats, gamma
                    )
                    descriptor_handle.write(_canonical_json(row) + "\n")
            if descriptor_handle is not None:
                descriptor_handle.flush()
                os.fsync(descriptor_handle.fileno())
        if aggregate.images != int(manifest["num_samples"]):
            raise RuntimeError(
                f"processed {aggregate.images} images; expected {manifest['num_samples']}"
            )
        descriptor_metadata = {
            "included": args.write_descriptors,
            "path": DESCRIPTORS_NAME if args.write_descriptors else None,
            "sha256": sha256_file(descriptor_path) if args.write_descriptors else None,
            "num_rows": aggregate.images if args.write_descriptors else 0,
            "schema_version": DESCRIPTOR_SCHEMA_VERSION,
            "deterministic_given_artifact_and_specification": True,
        }
        timestamp = created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
        summary = _summary(
            artifact, diagnostic_id, _utc(timestamp), source_sha256, environment,
            specification, aggregate, descriptor_metadata, _command(args)
        )
        _validate_summary(summary, "constructed diagnostics")
        summary_path = staging / SUMMARY_NAME
        with summary_path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(staging)
        publish_directory_no_replace(staging, destination)
        fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / SUMMARY_NAME


def load_binary_diagnostics(
    summary_path: str | os.PathLike[str], *, validate_descriptors: bool = True
) -> BinaryDiagnostics:
    """Strictly load a summary and, by default, its descriptor payload."""

    path = Path(summary_path)
    if path.name != SUMMARY_NAME or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected a regular, non-symlink {SUMMARY_NAME}: {path}")
    payload = path.read_bytes()
    summary = _strict_json(payload, str(path))
    _validate_summary(summary, str(path))
    if path.parent.name != summary["diagnostic_id"]:
        raise ValueError("diagnostic directory name does not match diagnostic_id")
    if path.parent.parent.name != summary["artifact"]["artifact_id"]:
        raise ValueError("parent directory name does not match artifact_id")
    if validate_descriptors and summary["descriptors"]["included"]:
        _validate_descriptor_file(path.parent, summary)
    return BinaryDiagnostics(path.resolve(), summary, hashlib.sha256(payload).hexdigest())


def _sample_stats(probability, truth, gamma, ece_bins, chunk_size) -> _SampleStats:
    probability = probability.reshape(-1)
    truth = truth.reshape(-1)
    pixels = int(probability.size)
    bin_counts = np.zeros(ece_bins, dtype=np.int64)
    bin_probability_sums = np.zeros(ece_bins, dtype=np.float64)
    bin_truth_sums = np.zeros(ece_bins, dtype=np.int64)
    ladder_intervals = np.zeros(LADDER_M + 1, dtype=np.int64)
    brier_sum = probability_sum = 0.0
    truth_foreground = hard_foreground = hard_intersection = hard_errors = 0

    for start in range(0, pixels, chunk_size):
        stop = min(start + chunk_size, pixels)
        probabilities = probability[start:stop].astype(np.float64, copy=False)
        labels = truth[start:stop]
        differences = probabilities - labels
        brier_sum += float(np.dot(differences, differences))
        probability_sum += float(np.sum(probabilities, dtype=np.float64))
        truth_foreground += int(np.count_nonzero(labels))
        hard = probabilities >= gamma
        hard_foreground += int(np.count_nonzero(hard))
        hard_intersection += int(np.count_nonzero(hard & (labels == 1)))
        hard_errors += int(np.count_nonzero(hard != (labels == 1)))

        bin_index = np.minimum(
            np.floor(probabilities * ece_bins).astype(np.intp), ece_bins - 1)
        bin_counts += np.bincount(bin_index, minlength=ece_bins)
        bin_probability_sums += np.bincount(
            bin_index, weights=probabilities, minlength=ece_bins)
        bin_truth_sums += np.bincount(
            bin_index, weights=labels, minlength=ece_bins).astype(np.int64)
        ladder_index = np.searchsorted(LADDER_NODES, probabilities, side="right")
        ladder_intervals += np.bincount(ladder_index, minlength=LADDER_M + 1)
    return _SampleStats(
        pixels, brier_sum, bin_counts, bin_probability_sums, bin_truth_sums,
        probability_sum, truth_foreground, hard_foreground, hard_intersection,
        hard_errors, ladder_intervals
    )


def _ladder_foreground(intervals):
    return np.cumsum(intervals[::-1], dtype=np.int64)[::-1][1:]


def _ece(counts, probability_sums, truth_sums):
    total = int(np.sum(counts))
    if total < 1:
        raise ValueError("ECE requires at least one pixel")
    selected = counts > 0
    gaps = np.zeros(len(counts), dtype=np.float64)
    gaps[selected] = np.abs(
        probability_sums[selected] / counts[selected]
        - truth_sums[selected] / counts[selected]
    )
    return float(np.dot(counts / total, gaps))


def _descriptor(diagnostic_id, artifact_id, sample_id, index, shape, stats, gamma):
    foreground = _ladder_foreground(stats.ladder_intervals)
    changes = stats.ladder_intervals[1:LADDER_M]
    denominator = stats.hard_foreground + stats.truth_foreground
    dice_loss = (0.0 if denominator == 0
                 else 1.0 - 2.0 * stats.hard_intersection / denominator)
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "diagnostic_id": diagnostic_id,
        "artifact_id": artifact_id,
        "sample_id": sample_id,
        "image_index": index,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "num_pixels": stats.pixels,
        "prediction_only": {
            "mean_foreground_probability": stats.probability_sum / stats.pixels,
            "decision_threshold": gamma,
            "hard_foreground_fraction": stats.hard_foreground / stats.pixels,
            "hard_empty_mask": stats.hard_foreground == 0,
            "ladder_distinct_mask_count": 1 + int(np.count_nonzero(changes)),
            "ladder_foreground_fractions": (foreground / stats.pixels).tolist(),
            "ladder_adjacent_changed_pixel_fractions": (changes / stats.pixels).tolist(),
        },
        "label_only": {
            "truth_foreground_fraction": stats.truth_foreground / stats.pixels,
            "truth_empty_mask": stats.truth_foreground == 0,
        },
        "label_outcomes": {
            "marginal_brier_score": stats.brier_sum / stats.pixels,
            "fixed_bin_ece": _ece(stats.bin_counts, stats.bin_probability_sums,
                                   stats.bin_truth_sums),
            "hard_pixel_error_rate": stats.hard_errors / stats.pixels,
            "hard_dice_loss": dice_loss,
        },
    }


def _summary(
    artifact: FrozenBinaryArtifact,
    diagnostic_id,
    created_utc,
    source_sha256,
    environment,
    specification,
    aggregate,
    descriptors,
    command,
):
    manifest = artifact.manifest
    images, pixels = aggregate.images, aggregate.pixels
    calibration_bins = []
    for index, count_value in enumerate(aggregate.bin_counts):
        count = int(count_value)
        if count:
            mean_probability = float(aggregate.bin_probability_sums[index]) / count
            foreground_rate = float(aggregate.bin_truth_sums[index]) / count
            gap = abs(mean_probability - foreground_rate)
        else:
            mean_probability = foreground_rate = gap = None
        calibration_bins.append(
            {
                "index": index, "lower": index / len(aggregate.bin_counts),
                "upper": (index + 1) / len(aggregate.bin_counts),
                "upper_inclusive": index == len(aggregate.bin_counts) - 1,
                "num_pixels": count, "mean_probability": mean_probability,
                "empirical_foreground_rate": foreground_rate,
                "absolute_gap": gap,
                "ece_contribution": 0.0 if gap is None else count / pixels * gap,
            }
        )
    global_foreground = _ladder_foreground(aggregate.ladder_intervals)
    global_changes = aggregate.ladder_intervals[1:LADDER_M]
    transitions = [
        {
            "index": index, "lower_threshold": float(LADDER_NODES[index]),
            "upper_threshold": float(LADDER_NODES[index + 1]),
            "total_changed_pixels": int(global_changes[index]),
            "pixel_weighted_changed_fraction": float(global_changes[index]) / pixels,
            "mean_image_changed_fraction": float(
                aggregate.change_fraction_sums[index]) / images,
            "zero_change_image_ratio": int(aggregate.change_zero[index]) / images,
        }
        for index in range(LADDER_M - 1)
    ]
    threshold_masks = [
        {
            "index": index, "threshold": float(LADDER_NODES[index]),
            "total_foreground_pixels": int(global_foreground[index]),
            "pixel_weighted_foreground_fraction": float(
                global_foreground[index]) / pixels,
            "mean_image_foreground_fraction": float(
                aggregate.ladder_fraction_sums[index]) / images,
            "empty_mask_ratio": int(aggregate.ladder_empty[index]) / images,
        }
        for index in range(LADDER_M)
    ]
    observed = np.flatnonzero(aggregate.distinct_histogram)
    distinct_mean = float(np.dot(
        np.arange(LADDER_M + 1), aggregate.distinct_histogram) / images)
    artifact_record = {
        field: manifest[field] for field in
        "artifact_id dataset condition model split class_index class_name num_samples "
        "sample_id_sha256 environment".split()
    }
    artifact_record.update({
        "manifest_path": artifact.manifest_path.as_posix(),
        "manifest_sha256": artifact.manifest_sha256,
        "payload_source_sha256": manifest["source_sha256"],
    })
    return {
        "schema_version": SCHEMA_VERSION, "artifact_type": ARTIFACT_TYPE,
        "diagnostic_id": diagnostic_id, "created_utc": created_utc,
        "source_sha256": source_sha256, "environment": environment,
        "artifact": artifact_record,
        "scope": dict(_SCOPE), "specification": specification,
        "counts": {"num_images": images, "num_pixels": pixels},
        "marginal_calibration": {
            "aggregation": "pixel-weighted over the held-out artifact",
            "brier_score": aggregate.brier_sum / pixels,
            "ece": _ece(aggregate.bin_counts, aggregate.bin_probability_sums,
                         aggregate.bin_truth_sums),
            "bins": calibration_bins,
        },
        "hard_prediction": _mask_summary(
            aggregate.hard_foreground, aggregate.hard_fraction_sum,
            aggregate.hard_empty, pixels, images),
        "truth": _mask_summary(
            aggregate.truth_foreground, aggregate.truth_fraction_sum,
            aggregate.truth_empty, pixels, images),
        "shared_threshold_ladder": {
            "distinct_mask_count": {
                "minimum": int(observed[0]), "maximum": int(observed[-1]),
                "mean": distinct_mean,
                "histogram": [
                    {"distinct_masks": value, "num_images": int(count)}
                    for value, count in enumerate(aggregate.distinct_histogram[1:], 1)
                ],
            },
            "adjacent_changes": {
                "transitions_per_image": LADDER_M - 1,
                "image_transition_pairs": images * (LADDER_M - 1),
                "zero_change_pair_ratio": int(np.sum(aggregate.change_zero))
                / (images * (LADDER_M - 1)),
                "mean_changed_pixel_fraction": float(
                    np.sum(aggregate.change_fraction_sums))
                / (images * (LADDER_M - 1)),
                "maximum_changed_pixel_fraction": float(np.max(aggregate.change_max)),
                "transitions": transitions,
            },
            "threshold_masks": threshold_masks,
        },
        "descriptors": descriptors, "command": command,
    }


def _mask_summary(foreground, fraction_sum, empty, pixels, images):
    return {
        "total_foreground_pixels": foreground,
        "pixel_weighted_foreground_fraction": foreground / pixels,
        "mean_image_foreground_fraction": fraction_sum / images,
        "num_empty_masks": empty, "empty_mask_ratio": empty / images,
    }


def _specification(gamma, ece_bins, chunk_size):
    return {
        "decision_rule": {"form": "foreground_probability >= gamma", "gamma": gamma},
        "ece": {
            "num_equal_width_bins": ece_bins,
            "bin_index": "min(floor(probability * num_bins), num_bins - 1)",
            "aggregation": "pixel-weighted",
        },
        "ladder": {
            "m": LADDER_M, "rule": "uniform midpoint",
            "mask_rule": "foreground_probability >= threshold",
            "nodes": LADDER_NODES.tolist(),
        },
        "pixel_chunk_size": chunk_size,
    }


def _diagnostic_id(manifest_sha, source_sha, environment, specification, descriptors):
    identity = {
        "schema_version": SCHEMA_VERSION, "artifact_type": ARTIFACT_TYPE,
        "artifact_manifest_sha256": manifest_sha,
        "source_sha256": source_sha, "environment": environment,
        "specification": specification,
        "include_descriptors": descriptors,
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:16]


def _source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "selectseg" / "artifacts.py",
        root / "selectseg" / "studies" / "diagnostics.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _command(args):
    arguments = getattr(args, "command_arguments", sys.argv[1:])
    return ["python", "-m", "scripts.diagnose", *list(arguments)]


def _schema(value, location, specification):
    """Validate an exact nested schema made from literals and small rules."""

    if callable(specification):
        return specification(value, location)
    if isinstance(specification, dict):
        value = _object(value, location, specification)
        for field, rule in specification.items():
            _schema(value[field], f"{location}.{field}", rule)
        return value
    return _equals(value, specification, location)


def _int_rule(minimum, maximum=None):
    return lambda value, location: _integer(value, location, minimum, maximum)


def _close_rule(expected):
    return lambda value, location: _close(value, expected, location)


def _validate_summary(summary, source):
    summary = _object(summary, source, _SUMMARY_KEYS)
    _equals(summary["schema_version"], SCHEMA_VERSION, "schema_version")
    _equals(summary["artifact_type"], ARTIFACT_TYPE, "artifact_type")
    _hex(summary["diagnostic_id"], 16, "diagnostic_id")
    _utc(summary["created_utc"])
    _digest(summary["source_sha256"], "source_sha256")
    environment = _schema(
        summary["environment"], "environment",
        {"packages": {"python": _string, "numpy": _string}, "device": "cpu"},
    )
    artifact, images = _validate_artifact_provenance(summary["artifact"])
    _schema(summary["scope"], "scope", _SCOPE)
    spec = _validate_specification(summary["specification"])
    descriptors = _validate_descriptor_metadata(summary["descriptors"], images)
    expected_id = _diagnostic_id(
        artifact["manifest_sha256"], summary["source_sha256"], environment,
        spec, descriptors["included"]
    )
    _equals(summary["diagnostic_id"], expected_id, "diagnostic_id")
    counts = _schema(summary["counts"], "counts", {
        "num_images": images, "num_pixels": _int_rule(images)
    })
    _equals(counts["num_images"], images, "counts.num_images")
    pixels = counts["num_pixels"]
    _validate_calibration(summary["marginal_calibration"], spec, pixels)
    for name in ("hard_prediction", "truth"):
        _validate_mask_summary(summary[name], name, images, pixels)
    _validate_ladder(summary["shared_threshold_ladder"], images, pixels)
    command = _array(summary["command"], "command")
    if not command:
        raise ValueError("command must be nonempty")
    for index, item in enumerate(command):
        _string(item, f"command[{index}]")
def _validate_artifact_provenance(value):
    artifact = _schema(value, "artifact", {
        "artifact_id": lambda item, place: _hex(item, 16, place),
        "manifest_path": _string, "manifest_sha256": _digest,
        "payload_source_sha256": _digest, "dataset": _string,
        "condition": _string, "model": _string, "split": _string,
        "class_index": 1, "class_name": _string, "num_samples": _int_rule(1),
        "sample_id_sha256": _digest, "environment": _validate_artifact_environment,
    })
    return artifact, artifact["num_samples"]


def _validate_artifact_environment(value, location="artifact.environment"):
    value = _object(value, location,
                    "packages device cuda_runtime cuda_device autocast_dtype")
    packages = _object(value["packages"], f"{location}.packages",
                       "python numpy torch torchvision transformers")
    _string(packages["python"], f"{location}.packages.python")
    for name in "numpy torch torchvision transformers".split():
        if packages[name] is not None:
            _string(packages[name], f"{location}.packages.{name}")
    if value["device"] == "cuda":
        for field in ("cuda_runtime", "cuda_device"):
            _string(value[field], f"{location}.{field}")
        expected = "bfloat16"
    elif value["device"] == "cpu":
        for field in ("cuda_runtime", "cuda_device"):
            _equals(value[field], None, f"{location}.{field}")
        expected = "disabled"
    else:
        raise ValueError(f"{location}.device must equal 'cpu' or 'cuda'")
    _equals(value["autocast_dtype"], expected, f"{location}.autocast_dtype")


def _validate_specification(value):
    return _schema(value, "specification", {
        "decision_rule": {
            "form": "foreground_probability >= gamma", "gamma": _fraction,
        },
        "ece": {
            "num_equal_width_bins": _int_rule(1),
            "bin_index": "min(floor(probability * num_bins), num_bins - 1)",
            "aggregation": "pixel-weighted",
        },
        "ladder": {
            "m": LADDER_M, "rule": "uniform midpoint",
            "mask_rule": "foreground_probability >= threshold",
            "nodes": LADDER_NODES.tolist(),
        },
        "pixel_chunk_size": _int_rule(1),
    })


def _validate_calibration(value, spec, pixels):
    calibration = _schema(value, "marginal_calibration", {
        "aggregation": "pixel-weighted over the held-out artifact",
        "brier_score": _fraction, "ece": _fraction,
        "bins": lambda item, place: _array(
            item, place, spec["ece"]["num_equal_width_bins"]),
    })
    num_bins = spec["ece"]["num_equal_width_bins"]
    count = 0
    for index, item in enumerate(_array(calibration["bins"], "bins", num_bins)):
        location = f"bins[{index}]"
        item = _schema(item, location, {
            "index": index, "lower": _close_rule(index / num_bins),
            "upper": _close_rule((index + 1) / num_bins),
            "upper_inclusive": index == num_bins - 1,
            "num_pixels": _int_rule(0, pixels),
            "mean_probability": _nullable_fraction,
            "empirical_foreground_rate": _nullable_fraction,
            "absolute_gap": _nullable_fraction, "ece_contribution": _fraction,
        })
        count += item["num_pixels"]
        empty = item["num_pixels"] == 0
        nullable = "mean_probability empirical_foreground_rate absolute_gap".split()
        if any((item[field] is None) != empty for field in nullable):
            raise ValueError(f"{location} empty-bin fields are inconsistent")
    _equals(count, pixels, "calibration bin pixel count")


def _validate_mask_summary(value, location, images, pixels):
    _schema(value, location, {
        "total_foreground_pixels": _int_rule(0, pixels),
        "pixel_weighted_foreground_fraction": _fraction,
        "mean_image_foreground_fraction": _fraction,
        "num_empty_masks": _int_rule(0, images), "empty_mask_ratio": _fraction,
    })


def _validate_ladder(value, images, pixels):
    ladder = _object(
        value, "shared_threshold_ladder",
        "distinct_mask_count adjacent_changes threshold_masks",
    )
    _validate_distinct_counts(ladder["distinct_mask_count"], images)
    _validate_threshold_masks(ladder["threshold_masks"], images, pixels)
    _validate_adjacent_changes(ladder["adjacent_changes"], images, pixels)


def _validate_distinct_counts(value, images):
    distinct = _schema(value, "distinct_mask_count", {
        "minimum": _int_rule(1, LADDER_M), "maximum": _int_rule(1, LADDER_M),
        "mean": lambda item, place: _number(item, place, 1, LADDER_M),
        "histogram": lambda item, place: _array(item, place, LADDER_M),
    })
    counts = []
    for index, item in enumerate(_array(distinct["histogram"], "histogram", LADDER_M), 1):
        item = _schema(item, f"histogram[{index - 1}]", {
            "distinct_masks": index, "num_images": _int_rule(0, images)
        })
        counts.append(item["num_images"])
    _equals(sum(counts), images, "distinct histogram image count")


def _validate_threshold_masks(value, images, pixels):
    entries = _array(value, "threshold_masks", LADDER_M)
    for index, item in enumerate(entries):
        location = f"threshold_masks[{index}]"
        item = _schema(item, location, {
            "index": index, "threshold": _close_rule(LADDER_NODES[index]),
            "total_foreground_pixels": _int_rule(0, pixels),
            "pixel_weighted_foreground_fraction": _fraction,
            "mean_image_foreground_fraction": _fraction,
            "empty_mask_ratio": _fraction,
        })
    return entries


def _validate_adjacent_changes(value, images, pixels):
    adjacent = _schema(value, "adjacent_changes", {
        "transitions_per_image": LADDER_M - 1,
        "image_transition_pairs": images * (LADDER_M - 1),
        "zero_change_pair_ratio": _fraction,
        "mean_changed_pixel_fraction": _fraction,
        "maximum_changed_pixel_fraction": _fraction,
        "transitions": lambda item, place: _array(item, place, LADDER_M - 1),
    })
    transitions = _array(adjacent["transitions"], "transitions", LADDER_M - 1)
    for index, item in enumerate(transitions):
        location = f"transitions[{index}]"
        item = _schema(item, location, {
            "index": index,
            "lower_threshold": _close_rule(LADDER_NODES[index]),
            "upper_threshold": _close_rule(LADDER_NODES[index + 1]),
            "total_changed_pixels": _int_rule(0, pixels),
            "pixel_weighted_changed_fraction": _fraction,
            "mean_image_changed_fraction": _fraction,
            "zero_change_image_ratio": _fraction,
        })


def _validate_descriptor_metadata(value, images):
    metadata = _object(
        value, "descriptors", "included path sha256 num_rows schema_version "
        "deterministic_given_artifact_and_specification"
    )
    _boolean(metadata["included"], "descriptors.included")
    _equals(metadata["schema_version"], DESCRIPTOR_SCHEMA_VERSION,
            "descriptors.schema_version")
    _equals(metadata["deterministic_given_artifact_and_specification"], True,
            "descriptors.determinism")
    if metadata["included"]:
        _equals(metadata["path"], DESCRIPTORS_NAME, "descriptors.path")
        _digest(metadata["sha256"], "descriptors.sha256")
        _equals(metadata["num_rows"], images, "descriptors.num_rows")
    else:
        for field in ("path", "sha256"):
            _equals(metadata[field], None, f"descriptors.{field}")
        _equals(metadata["num_rows"], 0, "descriptors.num_rows")
    return metadata


def _validate_descriptor_file(directory, summary):
    path = directory / DESCRIPTORS_NAME
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"descriptor file missing or a symlink: {path}")
    if sha256_file(path) != summary["descriptors"]["sha256"]:
        raise ValueError(f"descriptor SHA-256 mismatch: {path}")
    sample_ids, gamma = [], summary["specification"]["decision_rule"]["gamma"]
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            if not line.endswith(b"\n"):
                raise ValueError("every descriptor row must end with newline")
            row = _strict_json(line, f"{path}:{index + 1}")
            sample_ids.append(_validate_descriptor_row(row, index, summary, gamma))
    _equals(len(sample_ids), summary["descriptors"]["num_rows"], "descriptor rows")
    _equals(sample_id_sha256(sample_ids), summary["artifact"]["sample_id_sha256"],
            "descriptor sample order")


def _validate_descriptor_row(row, index, summary, gamma):
    location = f"descriptor[{index}]"
    row = _schema(row, location, {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "diagnostic_id": summary["diagnostic_id"],
        "artifact_id": summary["artifact"]["artifact_id"],
        "sample_id": _string, "image_index": index,
        "height": _int_rule(1), "width": _int_rule(1),
        "num_pixels": _int_rule(1),
        "prediction_only": {
            "mean_foreground_probability": _fraction,
            "decision_threshold": _close_rule(gamma),
            "hard_foreground_fraction": _fraction, "hard_empty_mask": _boolean,
            "ladder_distinct_mask_count": _int_rule(1, LADDER_M),
            "ladder_foreground_fractions": lambda value, place: _fraction_array(
                value, place, LADDER_M),
            "ladder_adjacent_changed_pixel_fractions":
                lambda value, place: _fraction_array(value, place, LADDER_M - 1),
        },
        "label_only": {
            "truth_foreground_fraction": _fraction, "truth_empty_mask": _boolean,
        },
        "label_outcomes": {
            "marginal_brier_score": _fraction, "fixed_bin_ece": _fraction,
            "hard_pixel_error_rate": _fraction, "hard_dice_loss": _fraction,
        },
    })
    _equals(row["num_pixels"], row["height"] * row["width"],
            f"{location}.num_pixels")
    return row["sample_id"]


def _strict_json(payload, source):
    def reject(value):
        raise ValueError(f"{source} contains non-finite JSON constant {value!r}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(payload.decode(), parse_constant=reject,
                          parse_float=finite_float, object_pairs_hook=unique)
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not UTF-8 JSON") from error


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _object(value, location, keys):
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be an object")
    expected = _keys(keys) if isinstance(keys, str) else frozenset(keys)
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{location} schema mismatch: "
                         f"missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")
    return value


def _array(value, location, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise TypeError(f"{location} must be a list{suffix}")
    return value


def _integer(value, location, minimum, maximum=None):
    invalid = type(value) is not int or value < minimum
    if invalid or (maximum is not None and value > maximum):
        interval = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise TypeError(f"{location} must be an integer {interval}")
    return value


def _number(value, location, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{location} must be finite and in [{minimum}, {maximum}]")
    return result


def _fraction(value, location):
    return _number(value, location, 0, 1)


def _nullable_fraction(value, location):
    return None if value is None else _fraction(value, location)


def _fraction_array(value, location, length):
    return [_fraction(item, f"{location}[{index}]") for index, item in
            enumerate(_array(value, location, length))]


def _boolean(value, location):
    if type(value) is not bool:
        raise TypeError(f"{location} must be boolean")
    return value


def _string(value, location):
    invalid = not isinstance(value, str) or not value or value != value.strip()
    if invalid or any(character in value for character in "\0\r\n"):
        raise TypeError(f"{location} must be a nonempty trimmed single-line string")
    return value


def _hex(value, length, location):
    invalid = not isinstance(value, str) or len(value) != length
    if invalid or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{location} must be {length} lowercase hexadecimal digits")
    return value


def _digest(value, location):
    return _hex(value, 64, location)


def _equals(value, expected, location):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{location} must equal {expected!r}")
    return value


def _close(value, expected, location):
    value = _number(value, location, -math.inf, math.inf)
    if not math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{location} is inconsistent with the diagnostic counts")
    return value


def _utc(value):
    value = _string(value, "created_utc")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("created_utc must be ISO-8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_utc must have a UTC offset")
    if timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("created_utc must be expressed in UTC")
    return value


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    args.command_arguments = list(sys.argv[1:] if argv is None else argv)
    path = run_diagnostics(args)
    loaded = load_binary_diagnostics(path)
    print(f"saved {loaded.summary_path}")
    print(f"summary_sha256={loaded.summary_sha256}")
    print(json.dumps({
        "diagnostic_id": loaded.summary["diagnostic_id"],
        "summary_path": loaded.summary_path.as_posix(),
        "summary_sha256": loaded.summary_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = """ARTIFACT_TYPE BinaryDiagnostics DEFAULT_ECE_BINS DESCRIPTORS_NAME
LADDER_M LADDER_NODES SCHEMA_VERSION SUMMARY_NAME load_binary_diagnostics main
parse_args run_diagnostics""".split()
