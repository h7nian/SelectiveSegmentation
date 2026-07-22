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
    spatial_copula_dice_confidence,
)
from selectseg.counts import (
    labels_for_coupling,
    partition_diagnostics,
    partition_dice_confidence,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_dice_count_posterior"
PARTITION_ARTIFACT_TYPE = "selectseg.binary_dice_partition_posterior"
COPULA_ARTIFACT_TYPE = "selectseg.binary_dice_spatial_copula"
COUPLINGS = (
    "action-two-block",
    "action-components",
    "action-grid",
    "proposal-components",
    "proposal-grid",
    "spatial-copula",
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
    parser.add_argument("--posterior-draws", type=int)
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--global-variance-weight", type=float)
    parser.add_argument("--spatial-variance-weight", type=float)
    parser.add_argument("--spatial-knot-spacing-diagonal", type=float)
    parser.add_argument("--posterior-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
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
        "coupling": args.coupling,
        "master_seed": args.master_seed,
    }
    if args.coupling == "spatial-copula":
        identity["spatial_copula"] = {
            "posterior_draws": args.posterior_draws,
            "repeat_index": args.repeat_index,
            "global_variance_weight_hex": args.global_variance_weight.hex(),
            "spatial_variance_weight_hex": args.spatial_variance_weight.hex(),
            "spatial_knot_spacing_diagonal_hex": (
                args.spatial_knot_spacing_diagonal.hex()
            ),
            "posterior_batch_size": args.posterior_batch_size,
            "device": args.device,
        }
    else:
        identity.update(
            {
                "m": args.m,
                "proposal_threshold": args.proposal_threshold,
                "draws": args.draws,
                "repeats": args.repeats,
            }
        )
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


def score_spatial_copula_sample(
    sample,
    *,
    gamma: float,
    posterior_draws: int,
    repeat_index: int,
    global_variance_weight: float,
    spatial_variance_weight: float,
    spatial_knot_spacing_diagonal: float,
    posterior_batch_size: int,
    master_seed: int,
    device: str,
) -> dict:
    """Score one image for exactly one spatial-copula Monte Carlo repeat."""

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
    estimate = spatial_copula_dice_confidence(
        probability,
        action,
        posterior_draws=posterior_draws,
        repeat_index=repeat_index,
        global_variance_weight=global_variance_weight,
        spatial_variance_weight=spatial_variance_weight,
        spatial_knot_spacing_diagonal=spatial_knot_spacing_diagonal,
        posterior_batch_size=posterior_batch_size,
        master_seed=master_seed,
        sample_id=str(sample.sample_id),
        device=device,
    )
    grid_shape = estimate.spatial_grid_shape
    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample.sample_id),
        "image_index": int(sample.index),
        "risk_dice": foreground_dice_loss(truth, action),
        "confidence_dice_spatial_copula": estimate.confidence,
        "spatial_grid_height": None if grid_shape is None else grid_shape[0],
        "spatial_grid_width": None if grid_shape is None else grid_shape[1],
        "score_runtime_seconds": float(time.perf_counter() - started),
    }
    numeric = [
        value
        for key, value in row.items()
        if key not in {"sample_id", "spatial_grid_height", "spatial_grid_width"}
    ]
    if not all(np.isfinite(value) for value in numeric):
        raise RuntimeError("spatial-copula scorer emitted a non-finite value")
    if (grid_shape is None) != (spatial_variance_weight == 0):
        raise RuntimeError("spatial-copula scorer emitted inconsistent grid metadata")
    return row


def run(args) -> Path:
    if args.gamma != 0.5 or args.m != 32:
        raise ValueError("the predeclared experiment requires gamma=.5 and M=32")
    is_spatial_copula = args.coupling == "spatial-copula"
    is_partition = args.coupling not in {"action-two-block", "spatial-copula"}
    variant = None
    if is_partition:
        variant = _partition_variant(args.coupling, args.proposal_threshold)
        if args.draws != 256 or args.repeats != 4:
            raise ValueError("partition couplings require 256 draws and four repeats")
    elif not is_spatial_copula and args.proposal_threshold is not None:
        raise ValueError("action-two-block does not accept a proposal threshold")
    if is_spatial_copula:
        required = (
            "posterior_draws",
            "repeat_index",
            "global_variance_weight",
            "spatial_variance_weight",
            "spatial_knot_spacing_diagonal",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "spatial-copula coupling requires: " + ", ".join(missing)
            )
        if args.proposal_threshold is not None:
            raise ValueError("spatial-copula does not accept a proposal threshold")
        if args.draws != 256 or args.repeats != 4:
            raise ValueError(
                "spatial-copula uses --posterior-draws and one --repeat-index; "
                "legacy --draws/--repeats must retain their defaults"
            )
        variant = "spatial_copula"
    else:
        forbidden = (
            "posterior_draws",
            "repeat_index",
            "global_variance_weight",
            "spatial_variance_weight",
            "spatial_knot_spacing_diagonal",
        )
        supplied = [name for name in forbidden if getattr(args, name) is not None]
        if supplied:
            raise ValueError(
                "spatial-copula arguments require --coupling spatial-copula: "
                + ", ".join(supplied)
            )
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
                    score_spatial_copula_sample(
                        sample,
                        gamma=args.gamma,
                        posterior_draws=args.posterior_draws,
                        repeat_index=args.repeat_index,
                        global_variance_weight=args.global_variance_weight,
                        spatial_variance_weight=args.spatial_variance_weight,
                        spatial_knot_spacing_diagonal=(
                            args.spatial_knot_spacing_diagonal
                        ),
                        posterior_batch_size=args.posterior_batch_size,
                        master_seed=args.master_seed,
                        device=args.device,
                    )
                    if is_spatial_copula
                    else score_partition_sample(
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
            "artifact_type": (
                COPULA_ARTIFACT_TYPE
                if is_spatial_copula
                else PARTITION_ARTIFACT_TYPE
                if is_partition
                else ARTIFACT_TYPE
            ),
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": manifest["dataset"],
            "condition": manifest["condition"],
            "model": manifest["model"],
            "num_rows": row_count,
            "score_fields": (
                ["confidence_dice_spatial_copula"]
                if is_spatial_copula
                else ["confidence_dice_partition"]
                if is_partition
                else list(SCORE_FIELDS)
            ),
            "diagnostic_fields": (
                [
                    "spatial_grid_height",
                    "spatial_grid_width",
                    "score_runtime_seconds",
                ]
                if is_spatial_copula
                else [
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
            "m": None if is_spatial_copula else args.m,
            "coupling": args.coupling,
            "proposal_threshold": args.proposal_threshold,
            "draws": None if is_spatial_copula else args.draws,
            "repeats": None if is_spatial_copula else args.repeats,
            "master_seed": args.master_seed,
            "records_sha256": sha256_file(records_path),
            "command": [
                "python",
                "-m",
                "selectseg.studies.counts",
                *args.command_arguments,
            ],
        }
        if is_spatial_copula:
            output_manifest["spatial_copula"] = {
                "posterior_draws": args.posterior_draws,
                "repeat_index": args.repeat_index,
                "global_variance_weight": args.global_variance_weight,
                "spatial_variance_weight": args.spatial_variance_weight,
                "independent_variance_weight": (
                    1
                    - args.global_variance_weight
                    - args.spatial_variance_weight
                ),
                "spatial_knot_spacing_diagonal": (
                    args.spatial_knot_spacing_diagonal
                ),
                "posterior_batch_size": args.posterior_batch_size,
                "device": args.device,
                "marginal_property": "Pr(Y_i=1)=p_i",
                "sampling": "antithetic standard-normal latent pairs",
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
