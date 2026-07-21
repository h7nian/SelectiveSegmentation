"""Score alternative-posterior and stability baselines for one ensemble cell.

The deployed action is always the thresholded arithmetic-mean probability map.
Three independently trained checkpoint maps are used only as an empirical
working posterior over candidate masks.  This deliberately separates the
question "which posterior is used for confidence?" from the deployed action and
realized loss.  The same job also computes inexpensive ensemble dispersion and
a SAM-inspired two-cutoff IoU stability score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selectseg.artifacts import (
    fsync_directory,
    load_binary_artifact,
    publish_directory_no_replace,
    sha256_file,
)
from selectseg.geometry import prepare_boundary_reference
from selectseg.confidence import foreground_dice_loss
from selectseg.ensemble import load_locked_ensemble_sources


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_ensemble_baselines"
SCORE_FIELDS = (
    "confidence_threshold_iou_stability",
    "confidence_ensemble_q_dice",
    "confidence_ensemble_q_nhd",
    "confidence_ensemble_q_nhd95",
    "confidence_ensemble_all_iou",
    "confidence_ensemble_pairwise_dice",
    "confidence_ensemble_negative_mutual_information",
    "confidence_ensemble_negative_probability_variance",
)
RISK_FIELDS = ("risk_dice", "risk_nhd", "risk_nhd95")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--expected-source-lock-sha256", required=True)
    parser.add_argument("--mean-artifact-manifest", required=True)
    parser.add_argument("--expected-mean-artifact-manifest-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--logit-offset", type=float, default=1.0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--max-pending-scores", type=int, default=16)
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    args.command_arguments = arguments
    return args


def _validated_probability(sample) -> np.ndarray:
    probability = np.asarray(sample.foreground_probability)
    if probability.dtype != np.float32 or probability.ndim != 2:
        raise TypeError("frozen probability must be one float32 2D array")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("frozen probability must be finite and lie in [0, 1]")
    return probability.astype(float)


def _entropy_nats(probability: np.ndarray) -> np.ndarray:
    result = np.zeros_like(probability, dtype=float)
    interior = (probability > 0) & (probability < 1)
    p = probability[interior]
    result[interior] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return result


def _dice_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return 1.0 - foreground_dice_loss(left, right)


def threshold_iou_stability(
    probability: np.ndarray, *, logit_offset: float = 1.0
) -> float:
    """IoU of masks at logit cutoffs ``-offset`` and ``+offset``.

    This adapts SAM's two-logit-cutoff stability score to calibrated foreground
    probabilities.  Empty--empty receives one so the score is total; this edge
    convention is stated explicitly because SAM's raw division is undefined
    when both masks are empty.
    """

    if not math.isfinite(logit_offset) or logit_offset <= 0:
        raise ValueError("logit_offset must be finite and positive")
    lower = 1.0 / (1.0 + math.exp(logit_offset))
    upper = 1.0 - lower
    low_mask = probability >= lower
    high_mask = probability >= upper
    union = int(low_mask.sum())
    return 1.0 if union == 0 else float(high_mask.sum() / union)


def score_ensemble_sample(
    mean_sample,
    member_samples,
    *,
    gamma: float = 0.5,
    logit_offset: float = 1.0,
) -> dict:
    """Return realized risks and higher-budget confidence scores for one image."""

    mean_probability = _validated_probability(mean_sample)
    truth = np.asarray(mean_sample.truth)
    if truth.dtype != np.uint8 or truth.shape != mean_probability.shape:
        raise TypeError("mean truth must be uint8 and match the probability shape")
    truth = truth.astype(bool, copy=False)
    member_probabilities = []
    for member in member_samples:
        if member.sample_id != mean_sample.sample_id or member.index != mean_sample.index:
            raise ValueError("ensemble members are not sample-aligned")
        probability = _validated_probability(member)
        member_truth = np.asarray(member.truth)
        if probability.shape != mean_probability.shape or not np.array_equal(
            member_truth, truth
        ):
            raise ValueError("ensemble members disagree on shape or truth")
        member_probabilities.append(probability)
    if len(member_probabilities) != 3:
        raise ValueError("the predeclared empirical posterior requires three members")

    action = mean_probability >= gamma
    member_masks = [probability >= gamma for probability in member_probabilities]
    action_boundary = prepare_boundary_reference(action)
    truth_boundary = prepare_boundary_reference(truth).compare(action)
    member_boundary = [action_boundary.compare(mask) for mask in member_masks]

    stacked_probability = np.stack(member_probabilities)
    predictive_entropy = _entropy_nats(stacked_probability.mean(axis=0))
    expected_entropy = np.mean(_entropy_nats(stacked_probability), axis=0)
    mutual_information = max(
        0.0, float(np.mean(predictive_entropy - expected_entropy))
    )

    all_intersection = np.logical_and.reduce(member_masks)
    all_union = np.logical_or.reduce(member_masks)
    all_iou = 1.0 if not all_union.any() else float(all_intersection.sum() / all_union.sum())
    pairwise_dice = np.mean(
        [
            _dice_similarity(member_masks[left], member_masks[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
    )

    row = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(mean_sample.sample_id),
        "image_index": int(mean_sample.index),
        "risk_dice": foreground_dice_loss(truth, action),
        "risk_nhd": truth_boundary.nhd,
        "risk_nhd95": truth_boundary.nhd95,
        "confidence_threshold_iou_stability": threshold_iou_stability(
            mean_probability, logit_offset=logit_offset
        ),
        "confidence_ensemble_q_dice": -float(
            np.mean([foreground_dice_loss(mask, action) for mask in member_masks])
        ),
        "confidence_ensemble_q_nhd": -float(
            np.mean([boundary.nhd for boundary in member_boundary])
        ),
        "confidence_ensemble_q_nhd95": -float(
            np.mean([boundary.nhd95 for boundary in member_boundary])
        ),
        "confidence_ensemble_all_iou": all_iou,
        "confidence_ensemble_pairwise_dice": float(pairwise_dice),
        "confidence_ensemble_negative_mutual_information": -mutual_information,
        "confidence_ensemble_negative_probability_variance": -float(
            np.var(stacked_probability, axis=0).mean()
        ),
    }
    expected_fields = {
        "schema_version",
        "sample_id",
        "image_index",
        *RISK_FIELDS,
        *SCORE_FIELDS,
    }
    if set(row) != expected_fields:
        raise RuntimeError("ensemble baseline scorer produced an invalid row schema")
    return row


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "selectseg/studies/ensemble.py",
        "selectseg/ensemble.py",
        "selectseg/artifacts.py",
        "selectseg/geometry.py",
        "selectseg/confidence.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_id(args, mean_manifest_sha256: str, source_sha256: str) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_lock_sha256": args.expected_source_lock_sha256,
        "mean_manifest_sha256": mean_manifest_sha256,
        "source_sha256": source_sha256,
        "dataset": args.dataset,
        "condition": args.condition,
        "gamma_hex": args.gamma.hex(),
        "logit_offset_hex": args.logit_offset.hex(),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def run(args):
    if not 0 < args.gamma < 1 or not math.isfinite(args.gamma):
        raise ValueError("gamma must be finite and lie in (0, 1)")
    if args.score_workers <= 0 or args.max_pending_scores < args.score_workers:
        raise ValueError("invalid worker or pending-score count")
    _, cell, members = load_locked_ensemble_sources(
        args.source_lock,
        args.expected_source_lock_sha256,
        args.dataset,
        args.condition,
    )
    mean_manifest = Path(args.mean_artifact_manifest)
    if mean_manifest.is_symlink() or not mean_manifest.is_file():
        raise FileNotFoundError("mean artifact manifest must be a regular file")
    if sha256_file(mean_manifest) != args.expected_mean_artifact_manifest_sha256:
        raise ValueError("mean artifact manifest SHA-256 mismatch")
    mean_artifact = load_binary_artifact(mean_manifest, validate_payloads=False)
    if mean_artifact.manifest_sha256 != args.expected_mean_artifact_manifest_sha256:
        raise RuntimeError("mean artifact changed while loading")
    for field in ("dataset", "condition", "model", "num_samples", "sample_id_sha256"):
        expected = cell[
            "expected_num_samples" if field == "num_samples" else field
        ] if field != "sample_id_sha256" else members[0].manifest[field]
        if mean_artifact.manifest[field] != expected:
            raise ValueError(f"mean artifact has unexpected {field}")

    source_sha256 = _source_fingerprint()
    run_id = _run_id(args, mean_artifact.manifest_sha256, source_sha256)
    output_dir = Path(args.output_root) / args.dataset / args.condition / run_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"ensemble baseline output already exists: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_dir.parent))
    records_path = staging / "records.jsonl"
    manifest_path = staging / "manifest.json"
    try:
        pending = deque()
        row_count = 0
        member_iterators = [artifact.iter_samples() for artifact in members]
        with (
            ThreadPoolExecutor(max_workers=args.score_workers) as pool,
            records_path.open("x", encoding="utf-8") as output,
        ):
            for samples in zip(
                mean_artifact.iter_samples(), *member_iterators, strict=True
            ):
                pending.append(
                    pool.submit(
                        score_ensemble_sample,
                        samples[0],
                        samples[1:],
                        gamma=args.gamma,
                        logit_offset=args.logit_offset,
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
        if row_count != mean_artifact.manifest["num_samples"]:
            raise RuntimeError("ensemble scorer emitted the wrong number of rows")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "condition": args.condition,
            "model": cell["model"],
            "num_rows": row_count,
            "score_fields": list(SCORE_FIELDS),
            "risk_fields": list(RISK_FIELDS),
            "source_lock_sha256": args.expected_source_lock_sha256,
            "mean_artifact_manifest_sha256": mean_artifact.manifest_sha256,
            "member_artifact_manifest_sha256": [
                artifact.manifest_sha256 for artifact in members
            ],
            "source_sha256": source_sha256,
            "gamma": args.gamma,
            "logit_offset": args.logit_offset,
            "empty_stability_convention": "empty-empty IoU equals one",
            "records_sha256": sha256_file(records_path),
            "command": [
                "python",
                "-m",
                "selectseg.studies.ensemble",
                *args.command_arguments,
            ],
        }
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
    return output_dir


def main(argv=None):
    output = run(parse_args(argv))
    print(f"saved {output / 'records.jsonl'}")
    print(f"saved {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
