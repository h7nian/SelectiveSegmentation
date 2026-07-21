"""Analyze explicit exact-cardinality diagnostic artifacts under one lock.

The canonical analysis requires exactly the sixteen lock-listed
``records.jsonl`` paths.  Inputs are never discovered from directories or
globs.  Reported randomized-PIT and reliability summaries are single-label,
aggregate falsification diagnostics: favorable values neither establish
pointwise posterior calibration nor validate the shared-threshold coupling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.analyze.main import EXPECTED_CONDITIONS
from selectseg.artifacts import fsync_directory
from selectseg.studies.cardinality import (
    EXPECTED_DIAGNOSTIC_SEED,
    PROTOCOL,
    CardinalityDiagnostic,
    load_auxiliary_lock,
    load_cardinality_diagnostic,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_cardinality_diagnostics_analysis"
DEFAULT_AUXILIARY_LOCK = (
    "configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json"
)
DEFAULT_OUTPUT = "outputs/binary_cardinality_diagnostics_analysis/analysis.json"
RELIABILITY_BINS = 10
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)
SCOPE = {
    "role": (
        "single-label aggregate falsification and label-proxy diagnostics for "
        "the shared-threshold working posterior's cardinality implications"
    ),
    "interpretation_limit": (
        "flat pooled PIT or favorable mass reliability cannot identify the "
        "pointwise conditional cardinality law, establish posterior calibration, "
        "validate Q_p, or establish Jaccard-Wasserstein agreement"
    ),
    "use_policy": (
        "held-out labels summarize predeclared diagnostics only and must not fit, "
        "tune, or select a confidence score"
    ),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-lock", default=DEFAULT_AUXILIARY_LOCK)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="RECORDS_JSONL",
        help="explicit records.jsonl files; directories/globs are unsupported",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow a nonempty lock-declared subset for unit/smoke tests only",
    )
    return parser.parse_args(argv)


def _source_sha256() -> str:
    source = Path(__file__).resolve()
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64))


def _error_summary(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mean_predicted": _mean(predicted),
        "mean_observed": _mean(observed),
        "signed_bias_predicted_minus_observed": _mean(error),
        "mean_absolute_error": _mean(np.abs(error)),
        "root_mean_squared_error": float(
            np.sqrt(np.mean(np.square(error), dtype=np.float64))
        ),
    }


def _equal_count_reliability(
    predicted: np.ndarray, observed: np.ndarray, sample_indices: np.ndarray
) -> dict[str, Any]:
    if not (predicted.shape == observed.shape == sample_indices.shape):
        raise ValueError("reliability inputs must have matching shapes")
    num_bins = min(RELIABILITY_BINS, int(predicted.size))
    # The index is a deterministic secondary key; no labels enter the binning.
    order = np.lexsort((sample_indices, predicted))
    bins = []
    for index, selected in enumerate(np.array_split(order, num_bins)):
        predicted_mean = _mean(predicted[selected])
        observed_mean = _mean(observed[selected])
        bins.append(
            {
                "bin_index": index,
                "num_images": int(selected.size),
                "predicted_mean_foreground_fraction": predicted_mean,
                "observed_mean_foreground_fraction": observed_mean,
                "signed_gap_predicted_minus_observed": (
                    predicted_mean - observed_mean
                ),
            }
        )
    counts = [row["num_images"] for row in bins]
    if sum(counts) != predicted.size or max(counts) - min(counts) > 1:
        raise RuntimeError("equal-count reliability construction failed")
    return {
        "strategy": "ten_equal_count_bins_sorted_by_predicted_mass",
        "num_bins": num_bins,
        "bins": bins,
    }


def _ks_uniform(pit: np.ndarray) -> float:
    ordered = np.sort(pit, kind="mergesort")
    n = ordered.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    upper = np.max(ranks / n - ordered)
    lower = np.max(ordered - (ranks - 1.0) / n)
    return float(max(upper, lower))


def summarize_condition(diagnostic: CardinalityDiagnostic) -> dict[str, Any]:
    """Summarize one strictly loaded condition without inferential claims."""

    records = diagnostic.records
    predicted_cardinality = np.asarray(
        [row["working_posterior_mean_cardinality"] for row in records],
        dtype=np.float64,
    )
    observed_cardinality = np.asarray(
        [row["truth_cardinality"] for row in records], dtype=np.float64
    )
    predicted_fraction = np.asarray(
        [row["working_posterior_mean_foreground_fraction"] for row in records],
        dtype=np.float64,
    )
    observed_fraction = np.asarray(
        [row["truth_foreground_fraction"] for row in records], dtype=np.float64
    )
    predicted_empty = np.asarray(
        [row["working_posterior_empty_probability"] for row in records],
        dtype=np.float64,
    )
    observed_empty = np.asarray(
        [row["observed_empty_mask"] for row in records], dtype=np.float64
    )
    pit = np.asarray(
        [row["randomized_cardinality_pit"] for row in records], dtype=np.float64
    )
    point_mass = np.asarray(
        [row["observed_cardinality_probability"] for row in records],
        dtype=np.float64,
    )
    sample_indices = np.asarray(
        [row["sample_index"] for row in records], dtype=np.int64
    )
    manifest = diagnostic.manifest
    return {
        "dataset": manifest["dataset"],
        "condition": manifest["condition"],
        "model": manifest["model"],
        "split": manifest["split"],
        "is_target_condition": diagnostic.key in TARGET_CONDITIONS,
        "num_images": len(records),
        "num_pixels": int(sum(row["pixels"] for row in records)),
        "run_id": manifest["run_id"],
        "manifest_path": _portable(diagnostic.manifest_path),
        "manifest_sha256": diagnostic.manifest_sha256,
        "source_artifact": {
            "artifact_id": manifest["provenance"]["artifact_id"],
            "manifest_path": manifest["provenance"]["artifact_manifest_path"],
            "manifest_sha256": manifest["provenance"][
                "artifact_manifest_sha256"
            ],
        },
        "cardinality_error": _error_summary(
            predicted_cardinality, observed_cardinality
        ),
        "foreground_fraction_error": _error_summary(
            predicted_fraction, observed_fraction
        ),
        "empty_mask_identity": {
            **_error_summary(predicted_empty, observed_empty),
            "definition": "Q_p(K=0)=1-max_i p_i",
        },
        "randomized_cardinality_pit": {
            "diagnostic_seed": EXPECTED_DIAGNOSTIC_SEED,
            "mean": _mean(pit),
            "variance": float(np.var(pit, dtype=np.float64)),
            "kolmogorov_smirnov_distance_to_uniform": _ks_uniform(pit),
            "outside_central_90_percent_ratio": _mean(
                ((pit < 0.05) | (pit > 0.95)).astype(np.float64)
            ),
            "zero_observed_point_mass_ratio": _mean(
                (point_mass == 0.0).astype(np.float64)
            ),
            "mean_observed_cardinality_probability": _mean(point_mass),
            "uniform_reference": {
                "mean": 0.5,
                "variance": 1.0 / 12.0,
                "outside_central_90_percent_ratio": 0.1,
            },
            "interpretation": (
                "pooled pseudo-randomized label-proxy falsification diagnostic; "
                "not a pointwise posterior-calibration estimate"
            ),
        },
        "foreground_fraction_reliability": _equal_count_reliability(
            predicted_fraction, observed_fraction, sample_indices
        ),
    }


def build_analysis(
    auxiliary_lock: str | os.PathLike[str],
    inputs: Sequence[str | os.PathLike[str]],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    binding = load_auxiliary_lock(auxiliary_lock)
    if not inputs:
        raise ValueError("at least one explicit cardinality diagnostic is required")
    loaded = [load_cardinality_diagnostic(path) for path in inputs]
    by_key = {}
    canonical_by_key = {
        (entry["dataset"], entry["condition"]): entry
        for entry in binding.campaign["artifacts"]
    }
    for diagnostic in loaded:
        if diagnostic.key in by_key:
            raise ValueError(f"duplicate cardinality condition: {diagnostic.key}")
        if diagnostic.key not in canonical_by_key:
            raise ValueError(f"undeclared cardinality condition: {diagnostic.key}")
        provenance = diagnostic.manifest["provenance"]
        expected = canonical_by_key[diagnostic.key]
        if provenance["auxiliary_lock_sha256"] != binding.sha256:
            raise ValueError("diagnostic auxiliary-lock provenance mismatch")
        if (
            provenance["auxiliary_spec_sha256"]
            != binding.data["spec"]["sha256"]
            or provenance["canonical_campaign_lock_sha256"]
            != binding.campaign_sha256
            or provenance["canonical_campaign_id"]
            != binding.campaign["campaign_id"]
        ):
            raise ValueError("diagnostic spec/campaign provenance mismatch")
        if provenance["artifact_id"] != expected["artifact_id"] or provenance[
            "artifact_manifest_sha256"
        ] != expected["manifest_sha256"]:
            raise ValueError("diagnostic frozen-artifact provenance mismatch")
        if diagnostic.manifest["num_images"] != expected["num_samples"]:
            raise ValueError("diagnostic sample count differs from canonical lock")
        by_key[diagnostic.key] = diagnostic

    expected_keys = tuple(EXPECTED_CONDITIONS)
    observed = set(by_key)
    if allow_incomplete:
        if not observed.issubset(set(expected_keys)):
            raise ValueError("incomplete analysis contains an unknown condition")
    elif observed != set(expected_keys):
        missing = sorted(set(expected_keys) - observed)
        extra = sorted(observed - set(expected_keys))
        raise ValueError(
            f"canonical cardinality analysis requires exactly 16 conditions; "
            f"missing={missing}, extra={extra}"
        )
    ordered_keys = [key for key in expected_keys if key in by_key]
    conditions = [summarize_condition(by_key[key]) for key in ordered_keys]
    source_sha256 = _source_sha256()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "auxiliary_lock_sha256": binding.sha256,
        "analyzer_source_sha256": source_sha256,
        "input_manifests": [
            row["manifest_sha256"] for row in conditions
        ],
        "allow_incomplete": allow_incomplete,
    }
    analysis_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": analysis_id,
        "scope": SCOPE,
        "protocol": PROTOCOL,
        "condition_sets": {
            "complete": not allow_incomplete,
            "num_conditions": len(conditions),
            "num_target_conditions": sum(
                row["is_target_condition"] for row in conditions
            ),
            "conditions": [f"{a}/{b}" for a, b in ordered_keys],
            "target_conditions": [
                f"{a}/{b}" for a, b in ordered_keys if (a, b) in TARGET_CONDITIONS
            ],
        },
        "provenance": {
            "auxiliary_lock_path": _portable(binding.path),
            "auxiliary_lock_sha256": binding.sha256,
            "canonical_campaign_lock_path": _portable(binding.campaign_path),
            "canonical_campaign_lock_sha256": binding.campaign_sha256,
            "analyzer_source_sha256": source_sha256,
            "input_policy": "explicit records.jsonl paths; no discovery or globs",
        },
        "conditions": conditions,
    }


def write_analysis(value: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cardinality analysis already exists: {destination}")
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_analysis(
        args.auxiliary_lock,
        args.inputs,
        allow_incomplete=args.allow_incomplete,
    )
    output = write_analysis(report, args.output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
