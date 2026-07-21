"""Focused loss-indexed evaluation for binary segmentation.

This evaluator writes exactly one row per image from a native binary dataset,
including images whose foreground label is empty.  Multiclass one-vs-rest
cohorts are intentionally rejected: selecting queried classes from ground
truth would turn image-level abstention into an oracle-selected class task.

The deployed action is always ``{p_foreground >= gamma}``.  Dice- and
HD95-indexed confidence use the same action, midpoint nodes, weights, and
empty-mask convention; only the loss changes.  The boundary loss is the
pooled bidirectional HD95 divided by the image diagonal, with one-sided empty
masks assigned loss one.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selectseg.baselines import strong_binary_confidences
from selectseg.confidence import (
    foreground_dice_loss,
    midpoint_loss_indexed_confidences,
    midpoint_rule,
    normalized_penalized_hd95,
    soft_dice_confidence,
    validate_quadrature,
)
from selectseg.data import SPECS, SegDataset, eval_collate
from selectseg.models import (
    CLIPSEG_CHECKPOINT,
    CONDITION_NAMES,
    DEEPLABV3_WEIGHTS,
    build_model,
)


SCHEMA_VERSION = 1
DEFAULT_M_VALUES = (2, 8, 32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CONDITION_NAMES))
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/binary")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES)
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--score-workers",
        type=int,
        default=4,
        help="CPU threads used for Dice/nHD95 scoring",
    )
    parser.add_argument(
        "--max-pending-scores",
        type=int,
        default=8,
        help="maximum queued image-scoring tasks",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint():
    """Hash the focused evaluator and every local module defining its output."""

    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "selectseg" / name
        for name in (
            "evaluate.py",
            "baselines.py",
            "confidence.py",
            "data.py",
            "models.py",
        )
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_info(checkpoint):
    if checkpoint is None:
        return None
    path = Path(checkpoint).resolve()
    working_root = Path.cwd().resolve()
    try:
        portable_path = path.relative_to(working_root).as_posix()
    except ValueError:
        portable_path = path.name
    info = {
        "path": portable_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    config_path = path.parent / "train_config.json"
    if config_path.is_file():
        try:
            portable_config_path = config_path.relative_to(working_root).as_posix()
        except ValueError:
            portable_config_path = config_path.name
        info["training_config"] = {
            "path": portable_config_path,
            "sha256": _sha256(config_path),
            "values": json.loads(config_path.read_text()),
        }
    return info


def _package_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "torch", "torchvision", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _binary_entropy(probability):
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def binary_record(
    foreground_probability,
    truth,
    *,
    run_id,
    image_id,
    image_index,
    class_index,
    class_name,
    decision_threshold,
    quadrature_rules,
):
    """Return one strict, label-separated binary experiment row."""

    probability = np.asarray(foreground_probability, dtype=float)
    raw_truth = np.asarray(truth)
    if raw_truth.ndim != 2 or probability.shape != raw_truth.shape:
        raise ValueError("probability and truth must have equal shapes and be 2D")
    if not np.all((raw_truth == 0) | (raw_truth == 1)):
        raise ValueError("truth must be a total binary mask with no void pixels")
    truth = raw_truth.astype(bool, copy=False)
    if not 0 < decision_threshold < 1:
        raise ValueError("decision_threshold must lie strictly inside (0, 1)")
    for count, rule in quadrature_rules.items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError("quadrature rule keys must be positive integer counts")
        nodes, weights = validate_quadrature(*rule)
        expected_nodes, expected_weights = midpoint_rule(int(count))
        if not np.array_equal(nodes, expected_nodes) or not np.array_equal(
            weights, expected_weights
        ):
            raise ValueError(f"quadrature rule m{count} is not its midpoint rule")

    hard_prediction = probability >= decision_threshold
    height, width = probability.shape
    diagonal = math.hypot(height, width)
    risk_nhd95 = normalized_penalized_hd95(truth, hard_prediction)
    loss_indexed = midpoint_loss_indexed_confidences(
        probability, hard_prediction, counts=tuple(quadrature_rules)
    )

    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "sample_id": str(image_id),
        "image_id": str(image_id),
        "image_index": int(image_index),
        "class_index": int(class_index),
        "class_name": str(class_name),
        "height": int(height),
        "width": int(width),
        "image_diagonal": float(diagonal),
        "truth_foreground_fraction": float(truth.mean()),
        "prediction_foreground_fraction": float(hard_prediction.mean()),
        "risk_dice": foreground_dice_loss(truth, hard_prediction),
        "risk_nhd95": risk_nhd95,
        "risk_hd95_pixels": float(risk_nhd95 * diagonal),
        "confidence_sdc": soft_dice_confidence(probability, hard_prediction),
        "confidence_mean_max_probability": float(
            np.maximum(probability, 1 - probability).mean()
        ),
        "confidence_negative_entropy": float(-_binary_entropy(probability).mean()),
    }
    row.update(strong_binary_confidences(probability, hard_prediction))
    for count, scores in loss_indexed.items():
        row[f"confidence_dice_m{count}"] = scores["dice"]
        row[f"confidence_nhd95_m{count}"] = scores["nhd95"]
    return row


def _manifest(
    args,
    *,
    condition,
    spec,
    dataset,
    quadrature_rules,
    sample_ids,
    run_id,
    checkpoint,
    source_sha256,
    device,
):
    score_fields = [
        "confidence_sdc",
        "confidence_mean_max_probability",
        "confidence_negative_entropy",
        "confidence_dice_exact",
        "confidence_qfr_entropy",
        "confidence_plm10_entropy",
        "confidence_mmmc_entropy",
        "confidence_foreground_entropy",
    ]
    for count in quadrature_rules:
        score_fields.extend(
            [f"confidence_dice_m{count}", f"confidence_nhd95_m{count}"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "condition": condition,
        "model": args.model,
        "dataset": args.dataset,
        "split": spec.eval_split,
        "num_images": len(dataset),
        "checkpoint": checkpoint,
        "base_model": {
            "name": args.model,
            "source": (
                CLIPSEG_CHECKPOINT
                if args.model == "clipseg"
                else str(DEEPLABV3_WEIGHTS)
            ),
        },
        "source_sha256": source_sha256,
        "environment": {
            "packages": _package_versions(),
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "cohort": "all images from a native binary segmentation split",
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": args.decision_threshold,
        },
        "preprocessing": {
            "model_input": "square resize with antialiasing",
            "probability_to_native_mask": (
                "bilinear interpolation with align_corners=False"
            ),
        },
        "losses": {
            "dice": "foreground 1-Dice; empty-empty=0; one-sided-empty=1",
            "hd95": (
                "pooled bidirectional HD95 / image diagonal; "
                "empty-empty=0; one-sided-empty=1"
            ),
        },
        "risk_fields": ["risk_dice", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd95_pixels"],
        "score_fields": score_fields,
        "quadrature": {
            str(count): {
                "rule": "midpoint",
                "nodes": nodes.tolist(),
                "weights": weights.tolist(),
            }
            for count, (nodes, weights) in quadrature_rules.items()
        },
        "void_policy": "total binary domain; void labels are forbidden",
        "sdc_empty_convention": (
            "published SDC baseline: confidence 0 when p and hard mask are empty"
        ),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "command": ["python", "-m", "selectseg.evaluate", *sys.argv[1:]],
    }


def _validate_args(args):
    if not 0 < args.decision_threshold < 1:
        raise ValueError("--decision-threshold must lie strictly inside (0, 1)")
    if not args.m_values or any(count <= 0 for count in args.m_values):
        raise ValueError("--m-values must contain positive integers")
    if len(set(args.m_values)) != len(args.m_values):
        raise ValueError("--m-values cannot contain duplicates")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.score_workers <= 0:
        raise ValueError("--score-workers must be positive")
    if args.max_pending_scores < args.score_workers:
        raise ValueError(
            "--max-pending-scores must be at least --score-workers"
        )


def main():
    args = parse_args()
    _validate_args(args)
    finetuned = args.checkpoint is not None
    condition = CONDITION_NAMES[args.model][finetuned]
    spec = SPECS[args.dataset]
    if spec.num_classes != 2:
        raise ValueError(
            "selectseg.evaluate only evaluates native binary datasets; "
            f"{args.dataset!r} has {spec.num_classes} classes"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _checkpoint_info(args.checkpoint)
    source_sha256 = _source_fingerprint()

    model = build_model(
        args.model, spec, finetuned=finetuned, init_weights=not finetuned
    )
    if finetuned:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device).eval()

    full_dataset = SegDataset(
        spec, args.data_root, train=False, image_size=model.image_size
    )
    image_indices = list(range(len(full_dataset)))
    if args.limit is not None:
        image_indices = image_indices[: args.limit]
    sample_ids = [full_dataset.sample_id(index) for index in image_indices]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{args.dataset!r} contains duplicate sample identifiers")
    dataset = Subset(full_dataset, image_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collate,
        pin_memory=device.type == "cuda",
    )
    quadrature_rules = {
        count: midpoint_rule(count) for count in sorted(args.m_values)
    }

    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "dataset": args.dataset,
        "split": spec.eval_split,
        "checkpoint_sha256": None if checkpoint is None else checkpoint["sha256"],
        "source_sha256": source_sha256,
        "decision_threshold": args.decision_threshold,
        "m_values": sorted(args.m_values),
        "sample_ids": sample_ids,
    }
    run_id = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    output_dir = Path(args.output_dir) / args.dataset / condition / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest_temporary_path = manifest_path.with_suffix(".json.tmp")
    if any(
        path.exists()
        for path in (
            output_path,
            manifest_path,
            temporary_path,
            manifest_temporary_path,
        )
    ):
        raise FileExistsError(
            f"run directory already contains output for {run_id}: {output_dir}"
        )

    manifest = _manifest(
        args,
        condition=condition,
        spec=spec,
        dataset=dataset,
        quadrature_rules=quadrature_rules,
        sample_ids=sample_ids,
        run_id=run_id,
        checkpoint=checkpoint,
        source_sha256=source_sha256,
        device=device,
    )
    image_cursor = 0
    row_count = 0
    pending = deque()
    with (
        ThreadPoolExecutor(max_workers=args.score_workers) as score_pool,
        temporary_path.open("x") as output,
        torch.inference_mode(),
    ):
        for images, masks in tqdm(
            loader,
            desc=f"{condition} on {args.dataset}",
            disable=not sys.stderr.isatty(),
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                probabilities = model.predict_probs(
                    images.to(device, non_blocking=True)
                )
            probabilities = probabilities.float()
            for probability, mask in zip(probabilities, masks):
                dataset_index = image_indices[image_cursor]
                image_cursor += 1
                image_id = full_dataset.sample_id(dataset_index)
                upsampled = F.interpolate(
                    probability.unsqueeze(0),
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).cpu()
                unique_labels = set(torch.unique(mask).tolist())
                if not unique_labels <= {0, 1}:
                    raise ValueError(
                        f"{args.dataset}/{image_id} is not a total binary mask: "
                        f"labels={sorted(unique_labels)}"
                    )
                pending.append(
                    score_pool.submit(
                        binary_record,
                        upsampled[1].numpy(),
                        (mask == 1).numpy(),
                        run_id=run_id,
                        image_id=image_id,
                        image_index=dataset_index,
                        class_index=1,
                        class_name=spec.class_names[1],
                        decision_threshold=args.decision_threshold,
                        quadrature_rules=quadrature_rules,
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

    if image_cursor != len(dataset):
        raise RuntimeError(f"processed {image_cursor} of {len(dataset)} images")
    if row_count == 0:
        raise RuntimeError("the native binary split produced no rows")
    temporary_path.replace(output_path)
    manifest["num_rows"] = row_count
    manifest["jsonl_sha256"] = _sha256(output_path)
    manifest_temporary_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    manifest_temporary_path.replace(manifest_path)
    print(f"saved {output_path} ({row_count} rows)")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
