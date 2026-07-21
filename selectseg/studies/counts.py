"""Score the predeclared Dice count-posterior experiment for one artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selectseg.artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
    sha256_file,
)
from selectseg.confidence import foreground_dice_loss
from selectseg.counts import (
    action_two_block_dice_confidence,
    count_ladders,
    second_order_dice_similarity,
    shared_threshold_dice_confidence,
)
from selectseg.counts import (
    labels_for_coupling,
    partition_diagnostics,
    partition_dice_confidence,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_dice_count_posterior"
PARTITION_ARTIFACT_TYPE = "selectseg.binary_dice_partition_posterior"
COUPLINGS = (
    "action-two-block",
    "action-components",
    "action-grid",
    "proposal-components",
    "proposal-grid",
)
SCORE_FIELDS = (
    "confidence_dice_shared_m32_recomputed",
    "confidence_dice_action_two_block_m32",
    "confidence_dice_sdc_recomputed",
)
DIAGNOSTIC_FIELDS = (
    "action_pixels",
    "mean_overlap",
    "mean_outside",
    "shared_variance_overlap",
    "shared_variance_outside",
    "shared_covariance",
    "two_block_covariance",
    "shared_second_order_dice_similarity",
    "two_block_second_order_dice_similarity",
    "score_runtime_seconds",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", required=True)
    parser.add_argument("--analysis-contract", required=True)
    parser.add_argument("--expected-analysis-contract-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--coupling", choices=COUPLINGS, default="action-two-block")
    parser.add_argument("--proposal-threshold", type=float)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--master-seed", type=int, default=20260721)
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "selectseg/studies/counts.py",
        "selectseg/counts.py",
        "selectseg/artifacts.py",
        "selectseg/confidence.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_id(args, artifact_manifest_sha256: str, analysis_contract_sha256: str, source_sha256: str) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "analysis_contract_sha256": analysis_contract_sha256,
        "source_sha256": source_sha256,
        "gamma_hex": args.gamma.hex(),
        "m": args.m,
        "coupling": args.coupling,
        "proposal_threshold": args.proposal_threshold,
        "draws": args.draws,
        "repeats": args.repeats,
        "master_seed": args.master_seed,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def score_sample(sample, *, gamma: float, m: int) -> dict:
    probability = np.asarray(sample.foreground_probability)
    truth = np.asarray(sample.truth)
    if probability.dtype != np.float32 or probability.ndim != 2:
        raise TypeError("probability must be one float32 2D array")
    if truth.dtype != np.uint8 or truth.shape != probability.shape:
        raise TypeError("truth must be uint8 and match probability")
    probability = probability.astype(float)
    truth = truth.astype(bool, copy=False)
    action = probability >= gamma

    started = time.perf_counter()
    overlap, outside = count_ladders(probability, action, m=m)
    shared = shared_threshold_dice_confidence(probability, action, m=m)
    two_block = action_two_block_dice_confidence(probability, action, m=m)
    elapsed = time.perf_counter() - started

    action_size = int(action.sum())
    mean_overlap = float(probability[action].sum())
    mean_outside = float(probability[~action].sum())
    denominator = action_size + mean_overlap + mean_outside
    sdc = 1.0 if denominator == 0 else 2.0 * mean_overlap / denominator
    variance_overlap = float(np.var(overlap, ddof=0))
    variance_outside = float(np.var(outside, ddof=0))
    covariance = float(np.mean((overlap - overlap.mean()) * (outside - outside.mean())))

    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "risk_dice": foreground_dice_loss(truth, action),
        "confidence_dice_shared_m32_recomputed": shared,
        "confidence_dice_action_two_block_m32": two_block,
        "confidence_dice_sdc_recomputed": float(sdc),
        "action_pixels": action_size,
        "mean_overlap": mean_overlap,
        "mean_outside": mean_outside,
        "shared_variance_overlap": variance_overlap,
        "shared_variance_outside": variance_outside,
        "shared_covariance": covariance,
        "two_block_covariance": 0.0,
        "shared_second_order_dice_similarity": second_order_dice_similarity(
            action_size,
            mean_overlap,
            mean_outside,
            variance_overlap,
            variance_outside,
            covariance,
        ),
        "two_block_second_order_dice_similarity": second_order_dice_similarity(
            action_size,
            mean_overlap,
            mean_outside,
            variance_overlap,
            variance_outside,
            0.0,
        ),
        "score_runtime_seconds": float(elapsed),
    }
    expected = {
        "schema_version",
        "sample_id",
        "image_index",
        "risk_dice",
        *SCORE_FIELDS,
        *DIAGNOSTIC_FIELDS,
    }
    if set(row) != expected:
        raise RuntimeError("count-posterior scorer emitted an invalid schema")
    if not all(
        np.isfinite(value)
        for key, value in row.items()
        if key not in {"sample_id"}
    ):
        raise RuntimeError("count-posterior scorer emitted a non-finite value")
    return row


def _partition_variant(coupling: str, proposal_threshold: float | None) -> str:
    normalized = coupling.replace("-", "_")
    if coupling.startswith("action-"):
        if proposal_threshold is not None:
            raise ValueError("action coupling cannot use a proposal threshold")
        return normalized
    if proposal_threshold not in {0.05, 0.1, 0.2}:
        raise ValueError("proposal threshold must be one of 0.05, 0.10, or 0.20")
    return f"{normalized}_t{int(round(100 * proposal_threshold)):02d}"


def _partition_seed_family(coupling: str, proposal_threshold: float | None) -> str:
    if coupling.startswith("action-"):
        return "action_component_grid_pair"
    return f"proposal_component_grid_pair_{proposal_threshold:.2f}"


def score_partition_sample(
    sample,
    *,
    coupling: str,
    proposal_threshold: float | None,
    gamma: float,
    draws: int,
    repeats: int,
    master_seed: int,
) -> dict:
    """Score one component or matched-grid coupling via the shared CLI."""

    probability = np.asarray(sample.foreground_probability)
    truth = np.asarray(sample.truth)
    if probability.dtype != np.float32 or probability.ndim != 2:
        raise TypeError("probability must be one float32 2D array")
    if truth.dtype != np.uint8 or truth.shape != probability.shape:
        raise TypeError("truth must be uint8 and match probability")
    probability = probability.astype(float)
    truth = truth.astype(bool, copy=False)
    action = probability >= gamma
    library_coupling = coupling.replace("-", "_")
    started = time.perf_counter()
    labels = labels_for_coupling(
        probability,
        action,
        coupling=library_coupling,
        proposal_threshold=proposal_threshold,
    )
    confidence, estimates = partition_dice_confidence(
        probability,
        action,
        labels,
        draws=draws,
        repeats=repeats,
        master_seed=master_seed,
        sample_id=str(sample.sample_id),
        coupling_id=_partition_seed_family(coupling, proposal_threshold),
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "risk_dice": foreground_dice_loss(truth, action),
        "confidence_dice_partition": confidence,
        "repeat_confidences": estimates.tolist(),
        "monte_carlo_repeat_standard_deviation": float(np.std(estimates, ddof=0)),
        **partition_diagnostics(labels),
        "score_runtime_seconds": float(time.perf_counter() - started),
    }
    scalar_values = [
        value
        for key, value in row.items()
        if key not in {"sample_id", "repeat_confidences"}
    ]
    if not all(np.isfinite(value) for value in scalar_values) or not np.isfinite(estimates).all():
        raise RuntimeError("partition scorer emitted a non-finite value")
    return row


def run(args) -> Path:
    if args.gamma != 0.5 or args.m != 32:
        raise ValueError("the predeclared experiment requires gamma=.5 and M=32")
    is_partition = args.coupling != "action-two-block"
    variant = None
    if is_partition:
        variant = _partition_variant(args.coupling, args.proposal_threshold)
        if args.draws != 256 or args.repeats != 4:
            raise ValueError("partition couplings require 256 draws and four repeats")
    elif args.proposal_threshold is not None:
        raise ValueError("action-two-block does not accept a proposal threshold")
    contract = Path(args.analysis_contract)
    if sha256_file(contract) != args.expected_analysis_contract_sha256:
        raise ValueError("analysis-contract SHA-256 mismatch")
    artifact = load_binary_artifact(args.artifact_manifest, validate_payloads=False)
    if artifact.manifest_sha256 != args.expected_artifact_manifest_sha256:
        raise ValueError("artifact-manifest SHA-256 mismatch")
    source_sha256 = _source_fingerprint()
    run_id = _run_id(args, artifact.manifest_sha256, args.expected_analysis_contract_sha256, source_sha256)
    manifest = artifact.manifest
    parent = Path(args.output_root) / str(manifest["dataset"]) / str(manifest["condition"])
    if variant is not None:
        parent /= variant
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"count-posterior output already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=parent))
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    try:
        row_count = 0
        with records_path.open("x", encoding="utf-8") as output:
            for sample in artifact.iter_samples():
                row = (
                    score_partition_sample(
                        sample,
                        coupling=args.coupling,
                        proposal_threshold=args.proposal_threshold,
                        gamma=args.gamma,
                        draws=args.draws,
                        repeats=args.repeats,
                        master_seed=args.master_seed,
                    )
                    if is_partition
                    else score_sample(sample, gamma=args.gamma, m=args.m)
                )
                output.write(
                    json.dumps(row, allow_nan=False)
                    + "\n"
                )
                row_count += 1
            output.flush()
            os.fsync(output.fileno())
        if row_count != int(manifest["num_samples"]):
            raise RuntimeError("count-posterior scorer emitted the wrong row count")
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": PARTITION_ARTIFACT_TYPE if is_partition else ARTIFACT_TYPE,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": manifest["dataset"],
            "condition": manifest["condition"],
            "model": manifest["model"],
            "num_rows": row_count,
            "score_fields": ["confidence_dice_partition"] if is_partition else list(SCORE_FIELDS),
            "diagnostic_fields": (
                [
                    "repeat_confidences",
                    "monte_carlo_repeat_standard_deviation",
                    "num_blocks",
                    "largest_block_fraction",
                    "score_runtime_seconds",
                ]
                if is_partition
                else list(DIAGNOSTIC_FIELDS)
            ),
            "risk_fields": ["risk_dice"],
            "artifact_manifest_sha256": artifact.manifest_sha256,
            "analysis_contract_sha256": args.expected_analysis_contract_sha256,
            "source_sha256": source_sha256,
            "gamma": args.gamma,
            "m": args.m,
            "coupling": args.coupling,
            "proposal_threshold": args.proposal_threshold,
            "draws": args.draws,
            "repeats": args.repeats,
            "master_seed": args.master_seed,
            "records_sha256": sha256_file(records_path),
            "command": [
                "python",
                "-m",
                "selectseg.studies.counts",
                *args.command_arguments,
            ],
        }
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(output_manifest, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        fsync_directory(staging)
        publish_directory_no_replace(staging, destination)
        fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def main(argv=None):
    destination = run(parse_args(argv))
    print(f"saved {destination / 'records.jsonl'}")
    print(f"saved {destination / 'manifest.json'}")


if __name__ == "__main__":
    main()
