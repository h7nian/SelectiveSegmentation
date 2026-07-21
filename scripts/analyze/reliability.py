"""Strict matched-risk reliability analysis for the canonical binary campaign.

The workflow consumes exactly the sixteen explicit, campaign-bound assembly
``records.jsonl`` files.  It creates ten deterministic equal-count bins for the
three predeclared matched score/loss pairs and attaches pointwise percentile
intervals for each bin's mean observed loss.  Held-out labels enter only this
descriptive analysis; they never fit, select, or tune a score or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.analyze.main import (
    EXPECTED_CONDITIONS,
    ConditionData,
    load_condition,
    validate_campaign_bound_conditions,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.matched_risk_reliability"
OUTPUT_NAME = "analysis.json"
GROUP_BINS = 10
BOOTSTRAP_SEED = 20_260_720
BOOTSTRAP_RESAMPLES = 2_000
CONFIDENCE_LEVEL = 0.95
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)
MATCHED_PAIRS = (
    ("confidence_dice_exact", "risk_dice", "Dice-Exact", "Dice"),
    ("confidence_nhd_m32", "risk_nhd", "nHD-M32", "nHD"),
    ("confidence_nhd95_m32", "risk_nhd95", "nHD95-M32", "nHD95"),
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-lock",
        required=True,
        help="immutable canonical campaign lock",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="RECORDS_JSONL",
        help="the sixteen explicit canonical assembly records.jsonl paths",
    )
    parser.add_argument(
        "--output",
        default="outputs/binary_matched_risk_reliability/analysis.json",
    )
    return parser.parse_args(argv)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "scripts/analyze/main.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _portable(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _finite_unit(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be finite and lie in [0,1]")
    return result


def load_inputs(paths: Sequence[str | os.PathLike[str]]) -> list[ConditionData]:
    if len(paths) != len(EXPECTED_CONDITIONS):
        raise ValueError(
            "matched-risk reliability requires exactly 16 explicit records.jsonl "
            f"inputs; observed {len(paths)}"
        )
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("records.jsonl inputs must be distinct")
    for path in resolved:
        if path.name != "records.jsonl":
            raise ValueError(f"input must be named records.jsonl: {path}")
    loaded = [load_condition(path) for path in resolved]
    by_key: dict[tuple[str, str], ConditionData] = {}
    for data in loaded:
        key = data.dataset, data.condition
        if key not in EXPECTED_CONDITIONS or key in by_key:
            raise ValueError(f"undeclared or duplicate dataset/condition input {key}")
        by_key[key] = data
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("inputs differ from the exact sixteen-condition benchmark")
    return [by_key[key] for key in EXPECTED_CONDITIONS]


def _validate_rows(data: ConditionData) -> None:
    required = {
        "sample_id",
        *(score for score, _, _, _ in MATCHED_PAIRS),
        *(risk for _, risk, _, _ in MATCHED_PAIRS),
    }
    if len(data.rows) < GROUP_BINS:
        raise ValueError(
            f"{data.dataset}/{data.condition} has fewer than {GROUP_BINS} images"
        )
    sample_ids = set()
    for index, row in enumerate(data.rows):
        location = f"{data.jsonl_path}:{index + 1}"
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{location} lacks reliability fields {missing}")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(f"{location}.sample_id must be unique and nonempty")
        sample_ids.add(sample_id)
        for score, risk, _, _ in MATCHED_PAIRS:
            predicted = -float(row[score])
            _finite_unit(predicted, location=f"{location}.-{score}")
            _finite_unit(row[risk], location=f"{location}.{risk}")


def equal_count_bootstrap_reliability(
    predicted_risk: Sequence[float],
    observed_loss: Sequence[float],
    sample_ids: Sequence[str],
    *,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Return the fixed ten-bin curve and within-bin image-bootstrap intervals."""

    predicted = np.asarray(predicted_risk, dtype=np.float64)
    observed = np.asarray(observed_loss, dtype=np.float64)
    ids = list(sample_ids)
    if (
        predicted.ndim != 1
        or observed.ndim != 1
        or predicted.size != observed.size
        or predicted.size != len(ids)
        or predicted.size < GROUP_BINS
    ):
        raise ValueError("aligned predicted/observed/sample arrays need at least 10 rows")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("predicted and observed arrays must be finite")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("predicted risks must lie in [0,1]")
    if np.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError("observed losses must lie in [0,1]")
    if len(set(ids)) != len(ids) or not all(isinstance(item, str) and item for item in ids):
        raise ValueError("sample IDs must be unique nonempty strings")

    order = np.asarray(
        sorted(range(predicted.size), key=lambda index: (predicted[index], ids[index])),
        dtype=np.int64,
    )
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    result = []
    for bin_index, indices in enumerate(np.array_split(order, GROUP_BINS), start=1):
        bin_observed = observed[indices]
        bootstrap_indices = rng.integers(
            0,
            indices.size,
            size=(BOOTSTRAP_RESAMPLES, indices.size),
            endpoint=False,
        )
        bootstrap_means = bin_observed[bootstrap_indices].mean(axis=1)
        lower, upper = np.quantile(
            bootstrap_means,
            [alpha, 1.0 - alpha],
            method="linear",
        )
        result.append(
            {
                "bin_index": bin_index,
                "num_images": int(indices.size),
                "minimum_predicted_risk": float(predicted[indices].min()),
                "maximum_predicted_risk": float(predicted[indices].max()),
                "mean_predicted_risk": float(predicted[indices].mean()),
                "mean_observed_loss": float(bin_observed.mean()),
                "pointwise_ci_lower": float(lower),
                "pointwise_ci_upper": float(upper),
            }
        )
    return result


def _analyze_target_condition(
    data: ConditionData, *, rng: np.random.Generator
) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in data.rows]
    panels = []
    for score, risk, score_label, loss_label in MATCHED_PAIRS:
        predicted = [-float(row[score]) for row in data.rows]
        observed = [float(row[risk]) for row in data.rows]
        panels.append(
            {
                "score_field": score,
                "score_label": score_label,
                "predicted_risk_definition": f"-{score}",
                "observed_loss_field": risk,
                "loss_label": loss_label,
                "bins": equal_count_bootstrap_reliability(
                    predicted,
                    observed,
                    sample_ids,
                    rng=rng,
                ),
            }
        )
    model = data.manifest.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{data.manifest_path}.model must be a nonempty string")
    return {
        "dataset": data.dataset,
        "condition": data.condition,
        "model": model,
        "num_images": len(data.rows),
        "panels": panels,
    }


def _validate_canonical_provenance(
    value: Mapping[str, Any], conditions: Sequence[ConditionData]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("binding") != "campaign-lock":
        raise ValueError("canonical provenance must be campaign-lock bound")
    campaign = value.get("campaign_lock")
    if (
        not isinstance(campaign, Mapping)
        or not isinstance(campaign.get("sha256"), str)
        or len(campaign["sha256"]) != 64
    ):
        raise ValueError("canonical provenance has an invalid campaign lock")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(EXPECTED_CONDITIONS):
        raise ValueError("canonical provenance must retain exactly sixteen inputs")
    observed = {(item.get("dataset"), item.get("condition")) for item in inputs}
    if observed != set(EXPECTED_CONDITIONS):
        raise ValueError("canonical provenance conditions are incomplete")
    condition_keys = {(item.dataset, item.condition) for item in conditions}
    if observed != condition_keys:
        raise ValueError("canonical provenance and loaded conditions differ")
    return dict(value)


def build_report(
    conditions: Sequence[ConditionData], *, canonical_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    by_key = {(item.dataset, item.condition): item for item in conditions}
    if len(by_key) != len(conditions) or set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("analysis requires the exact sixteen declared conditions")
    ordered = [by_key[key] for key in EXPECTED_CONDITIONS]
    for data in ordered:
        _validate_rows(data)
    provenance = _validate_canonical_provenance(canonical_provenance, ordered)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    target_rows = [
        _analyze_target_condition(data, rng=rng)
        for data in ordered
        if (data.dataset, data.condition) in TARGET_CONDITIONS
    ]
    source_sha = _source_sha256()
    protocol = {
        "grouping": (
            "ten equal-count bins ordered by ascending predicted working risk and "
            "then sample_id; exact risk ties may span adjacent bins"
        ),
        "num_bins": GROUP_BINS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "image resampled with replacement within each fixed bin",
        "confidence_level": CONFIDENCE_LEVEL,
        "interval": "pointwise equal-tail percentile interval for mean observed loss",
        "matched_pairs": [
            {
                "score_field": score,
                "predicted_risk_definition": f"-{score}",
                "observed_loss_field": risk,
                "score_label": score_label,
                "loss_label": loss_label,
            }
            for score, risk, score_label, loss_label in MATCHED_PAIRS
        ],
    }
    canonical_inputs = [
        {
            key: item[key]
            for key in (
                "dataset",
                "condition",
                "manifest_sha256",
                "records_sha256",
                "sample_id_sha256",
                "num_samples",
            )
        }
        for item in provenance["inputs"]
    ]
    identity = {
        "artifact_type": ARTIFACT_TYPE,
        "campaign_lock_sha256": provenance["campaign_lock"]["sha256"],
        "canonical_inputs": canonical_inputs,
        "protocol": protocol,
        "workflow_source_sha256": source_sha,
    }
    analysis_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": analysis_id,
        "scope": {
            "status": "single-label descriptive reliability diagnostic",
            "label_use": (
                "held-out labels form fixed-bin means and intervals only; they do not "
                "fit, tune, or select a score, action threshold, bin count, or image"
            ),
            "posterior_limitation": (
                "one reference mask per image cannot establish pointwise conditional-risk "
                "calibration or validate the shared-threshold posterior"
            ),
            "interval_limitation": (
                "intervals are pointwise within-bin image-bootstrap intervals, not "
                "simultaneous confidence bands"
            ),
        },
        "protocol": protocol,
        "condition_sets": {
            "canonical_complete": True,
            "canonical_conditions": [f"{a}/{b}" for a, b in EXPECTED_CONDITIONS],
            "target_conditions": [
                f"{a}/{b}" for a, b in EXPECTED_CONDITIONS if (a, b) in TARGET_CONDITIONS
            ],
            "num_canonical_conditions": 16,
            "num_target_conditions": 10,
        },
        "provenance": {
            "workflow_source_sha256": source_sha,
            "canonical_validation": provenance,
        },
        "conditions": target_rows,
    }
    _canonical_json(report)
    return report


def analyze(
    input_paths: Sequence[str | os.PathLike[str]],
    *,
    campaign_lock: str | os.PathLike[str],
) -> dict[str, Any]:
    conditions = load_inputs(input_paths)
    provenance = validate_campaign_bound_conditions(conditions, campaign_lock)
    return build_report(conditions, canonical_provenance=provenance)


def write_report(report: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite reliability analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze(args.inputs, campaign_lock=args.campaign_lock)
    destination = write_report(report, args.output)
    print(_portable(destination))


if __name__ == "__main__":
    main()
