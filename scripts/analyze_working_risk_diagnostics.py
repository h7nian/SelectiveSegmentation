"""Deterministic grouped diagnostics for the assembled binary benchmark.

This analysis consumes the exact campaign-bound ``records.jsonl`` assemblies.
It describes three different objects without conflating them:

* hard-action quality under Dice, normalized HD, and normalized HD95;
* ranking agreement among the three loss-indexed working-risk scores; and
* population-level grouped agreement between each working-risk proxy and one
  realized held-out loss per image.

The last item is deliberately called a *grouped diagnostic*, not posterior
validation.  With one reference mask per image it neither identifies the true
conditional mask posterior nor verifies the shared-threshold coupling at an
individual covariate value.  No image is selected and no TeX is rendered.

Canonical use requires an explicit immutable campaign lock and exactly the 16
predeclared assembly paths.  ``--allow-incomplete`` exists only for tests and
pipeline smoke checks; it disables campaign-lock claims in the output.  Inputs
are never discovered through directory walks or globs.
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
from scipy.stats import kendalltau, rankdata

from scripts.analyze_binary import (
    EXPECTED_CONDITIONS,
    ConditionData,
    load_condition,
    validate_campaign_bound_conditions,
)
from selectseg.binary_framework import tie_aware_expected_aurc


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "selectseg.working_risk_grouped_diagnostics"
DEFAULT_GROUP_BINS = 10
COVERAGES = (0.25, 0.5, 0.75)

AGREEMENT_SCORES = (
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_nhd_m32", "nHD-M32"),
    ("confidence_nhd95_m32", "nHD95-M32"),
)
RELIABILITY_SCORES = (
    ("confidence_dice_exact", "risk_dice", "Dice-Exact"),
    ("confidence_nhd_m32", "risk_nhd", "nHD-M32"),
    ("confidence_nhd95_m32", "risk_nhd95", "nHD95-M32"),
)
RISK_LABELS = (
    ("risk_dice", "Dice loss"),
    ("risk_nhd", "normalized penalized Hausdorff loss"),
    ("risk_nhd95", "normalized penalized HD95 loss"),
)
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "prediction_foreground_fraction",
        "truth_foreground_fraction",
        "confidence_sdc",
        *(score for score, _ in AGREEMENT_SCORES),
        *(score for score, _, _ in RELIABILITY_SCORES),
        *(risk for risk, _ in RISK_LABELS),
    }
)
MASK_SIZE_STRATA = (
    ("empty", 0.0, 0.0),
    ("very_small", 0.0, 0.01),
    ("small", 0.01, 0.10),
    ("medium", 0.10, 0.50),
    ("large", 0.50, 1.00),
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-lock",
        help="immutable campaign lock; required unless --allow-incomplete is used",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="RECORDS_JSONL",
        help="explicit assembled records.jsonl paths; directories/globs are forbidden",
    )
    parser.add_argument(
        "--output",
        default="outputs/binary_final/diagnostics/working_risk_diagnostics.json",
    )
    parser.add_argument(
        "--group-bins",
        type=int,
        default=DEFAULT_GROUP_BINS,
        help="number of deterministic equal-count working-risk groups",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow a nonempty declared subset for tests/smoke checks only",
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
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("analyze_binary.py"),
        Path(__file__).resolve().parents[1] / "selectseg/binary_framework.py",
    )
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_path(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _finite_number(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
    return result


def _validate_rows(data: ConditionData) -> None:
    """Validate diagnostic-only fields not covered by the canonical loader."""

    for index, row in enumerate(data.rows, start=1):
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"{data.jsonl_path}:{index} lacks diagnostic fields {missing}"
            )
        for field in (
            "prediction_foreground_fraction",
            "truth_foreground_fraction",
            *(risk for risk, _ in RISK_LABELS),
        ):
            _finite_number(
                row[field],
                location=f"{data.jsonl_path}:{index}.{field}",
                minimum=0.0,
                maximum=1.0,
            )
        for field in (
            "confidence_sdc",
            *(score for score, _ in AGREEMENT_SCORES),
            *(score for score, _, _ in RELIABILITY_SCORES),
        ):
            _finite_number(row[field], location=f"{data.jsonl_path}:{index}.{field}")


def load_inputs(
    paths: Sequence[str | os.PathLike[str]], *, allow_incomplete: bool = False
) -> list[ConditionData]:
    if not paths:
        raise ValueError("at least one explicit records.jsonl input is required")
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
        if key in by_key:
            raise ValueError(f"duplicate dataset/condition input {key}")
        if key not in EXPECTED_CONDITIONS:
            raise ValueError(f"undeclared dataset/condition input {key}")
        _validate_rows(data)
        by_key[key] = data

    observed = set(by_key)
    expected = set(EXPECTED_CONDITIONS)
    if allow_incomplete:
        if not observed:
            raise ValueError("incomplete diagnostic input cannot be empty")
    elif observed != expected or len(by_key) != len(EXPECTED_CONDITIONS):
        raise ValueError(
            "canonical diagnostics require exactly the 16 predeclared conditions; "
            f"missing={sorted(expected - observed)}"
        )
    return [by_key[key] for key in EXPECTED_CONDITIONS if key in by_key]


def _array(data: ConditionData, field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in data.rows], dtype=np.float64)


def _correlation(left: np.ndarray, right: np.ndarray, *, method: str) -> dict:
    left_constant = bool(np.all(left == left[0]))
    right_constant = bool(np.all(right == right[0]))
    if left_constant or right_constant:
        sides = []
        if left_constant:
            sides.append("left")
        if right_constant:
            sides.append("right")
        return {
            "defined": False,
            "value": None,
            "undefined_reason": f"constant_{'_and_'.join(sides)}_score",
        }

    if method == "spearman":
        left_rank = rankdata(left, method="average")
        right_rank = rankdata(right, method="average")
        value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    elif method == "kendall_tau_b":
        statistic = kendalltau(left, right, variant="b", nan_policy="raise").statistic
        value = float(statistic)
    else:  # pragma: no cover - private-call programming error
        raise ValueError(f"unknown correlation method {method!r}")
    if not math.isfinite(value):
        raise RuntimeError(f"{method} unexpectedly returned a non-finite value")
    return {"defined": True, "value": value, "undefined_reason": None}


def fractional_acceptance_weights(
    scores: Sequence[float], coverage: float
) -> np.ndarray:
    """Tie-aware selector inclusion weights with exactly ``coverage * n`` mass."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")

    target = float(coverage * values.size)
    weights = np.zeros(values.size, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    start = 0
    remaining = target
    while start < values.size and remaining > 0:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        group_size = stop - start
        inclusion = min(1.0, remaining / group_size)
        weights[order[start:stop]] = inclusion
        remaining -= inclusion * group_size
        start = stop
    if not np.isclose(weights.sum(), target, rtol=1e-12, atol=1e-12):
        raise RuntimeError("fractional selector did not achieve requested coverage")
    return weights


def _fractional_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = float(np.maximum(left, right).sum())
    if union == 0:  # impossible for the positive canonical coverages
        return 1.0
    return float(np.minimum(left, right).sum() / union)


def _score_agreement(data: ConditionData) -> list[dict]:
    result = []
    for left_index, (left_field, left_label) in enumerate(AGREEMENT_SCORES):
        left = _array(data, left_field)
        for right_field, right_label in AGREEMENT_SCORES[left_index + 1 :]:
            right = _array(data, right_field)
            overlaps = []
            for coverage in COVERAGES:
                left_weights = fractional_acceptance_weights(left, coverage)
                right_weights = fractional_acceptance_weights(right, coverage)
                overlaps.append(
                    {
                        "coverage": coverage,
                        "tie_aware_fractional_jaccard": _fractional_jaccard(
                            left_weights, right_weights
                        ),
                    }
                )
            result.append(
                {
                    "left_score": left_field,
                    "left_label": left_label,
                    "right_score": right_field,
                    "right_label": right_label,
                    "spearman_rho": _correlation(left, right, method="spearman"),
                    "kendall_tau_b": _correlation(left, right, method="kendall_tau_b"),
                    "accepted_set_agreement": overlaps,
                }
            )
    return result


def _nullable_distribution_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "n": 0,
            "mean_signed_difference": None,
            "mean_absolute_difference": None,
            "rmse": None,
            "quantile_10": None,
            "median": None,
            "quantile_90": None,
        }
    return {
        "n": int(values.size),
        "mean_signed_difference": float(values.mean()),
        "mean_absolute_difference": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "quantile_10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "quantile_90": float(np.quantile(values, 0.90)),
    }


def _mask_size_membership(fractions: np.ndarray, name: str) -> np.ndarray:
    if name == "empty":
        return fractions == 0.0
    for stratum, lower, upper in MASK_SIZE_STRATA:
        if stratum == name:
            return (fractions > lower) & (fractions <= upper)
    raise KeyError(name)  # pragma: no cover - private-call programming error


def _dice_exact_sdc_diagnostic(data: ConditionData) -> dict:
    # confidence_dice_exact = -E_Q[Dice loss], hence 1 + confidence is E_Q[Dice].
    working_expected_dice = 1.0 + _array(data, "confidence_dice_exact")
    sdc = _array(data, "confidence_sdc")
    difference = working_expected_dice - sdc
    size = _array(data, "prediction_foreground_fraction")
    strata = []
    assigned = np.zeros(size.size, dtype=bool)
    for name, lower, upper in MASK_SIZE_STRATA:
        membership = _mask_size_membership(size, name)
        if np.any(assigned & membership):
            raise AssertionError("prediction mask-size strata overlap")
        assigned |= membership
        summary = _nullable_distribution_summary(difference[membership])
        summary.update(
            {
                "stratum": name,
                "prediction_foreground_fraction_interval": (
                    "{0}" if name == "empty" else f"({lower}, {upper}]"
                ),
            }
        )
        strata.append(summary)
    if not assigned.all():
        raise RuntimeError("a prediction foreground fraction escaped all strata")
    return {
        "quantity": "(1 + confidence_dice_exact) - confidence_sdc",
        "interpretation": (
            "shared-threshold working expected Dice coefficient minus the "
            "published ratio-of-expectations SDC baseline"
        ),
        "overall": _nullable_distribution_summary(difference),
        "by_deployed_prediction_mask_size": strata,
    }


def equal_count_reliability(
    predicted_risk: Sequence[float],
    observed_loss: Sequence[float],
    sample_ids: Sequence[str],
    *,
    bins: int,
) -> dict:
    """Population grouped agreement with deterministic equal-count partitions."""

    predicted = np.asarray(predicted_risk, dtype=np.float64)
    observed = np.asarray(observed_loss, dtype=np.float64)
    ids = list(sample_ids)
    if predicted.ndim != 1 or observed.ndim != 1 or predicted.size == 0:
        raise ValueError("predicted and observed risk must be nonempty 1D arrays")
    if predicted.size != observed.size or len(ids) != predicted.size:
        raise ValueError("predicted risk, observed loss, and sample IDs must align")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("predicted risk and observed loss must be finite")
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("predicted working risks must lie in [0, 1]")
    if np.any((observed < 0) | (observed > 1)):
        raise ValueError("observed losses must lie in [0, 1]")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if len(set(ids)) != len(ids) or not all(
        isinstance(item, str) and item for item in ids
    ):
        raise ValueError("sample IDs must be unique nonempty strings")

    order = np.asarray(
        sorted(range(predicted.size), key=lambda index: (predicted[index], ids[index])),
        dtype=int,
    )
    effective_bins = min(bins, predicted.size)
    groups = []
    ece = 0.0
    for group_index, indices in enumerate(
        np.array_split(order, effective_bins), start=1
    ):
        mean_predicted = float(predicted[indices].mean())
        mean_observed = float(observed[indices].mean())
        gap = mean_predicted - mean_observed
        weight = float(indices.size / predicted.size)
        ece += weight * abs(gap)
        groups.append(
            {
                "group": group_index,
                "n": int(indices.size),
                "weight": weight,
                "minimum_predicted_risk": float(predicted[indices].min()),
                "maximum_predicted_risk": float(predicted[indices].max()),
                "mean_predicted_risk": mean_predicted,
                "mean_observed_loss": mean_observed,
                "signed_group_gap_predicted_minus_observed": gap,
            }
        )

    # Aggregate in the same deterministic order as the grouped summaries so
    # byte-level output does not depend on the caller's row order.
    residual = predicted[order] - observed[order]
    return {
        "n": int(predicted.size),
        "requested_equal_count_groups": int(bins),
        "effective_equal_count_groups": int(effective_bins),
        "bias_predicted_minus_observed": float(residual.mean()),
        "per_image_mae": float(np.abs(residual).mean()),
        "per_image_rmse": float(np.sqrt(np.square(residual).mean())),
        "grouped_ece": float(ece),
        "groups": groups,
    }


def _working_risk_reliability(data: ConditionData, *, bins: int) -> list[dict]:
    sample_ids = [str(row["sample_id"]) for row in data.rows]
    result = []
    for score_field, risk_field, label in RELIABILITY_SCORES:
        predicted = -_array(data, score_field)
        observed = _array(data, risk_field)
        summary = equal_count_reliability(predicted, observed, sample_ids, bins=bins)
        summary.update(
            {
                "score_field": score_field,
                "score_label": label,
                "observed_loss_field": risk_field,
                "predicted_working_risk_definition": f"-{score_field}",
            }
        )
        result.append(summary)
    return result


def _action_quality(data: ConditionData) -> dict:
    dice_loss = _array(data, "risk_dice")
    prediction_fraction = _array(data, "prediction_foreground_fraction")
    truth_fraction = _array(data, "truth_foreground_fraction")
    references = {}
    for risk_field, risk_label in RISK_LABELS:
        risk = _array(data, risk_field)
        references[risk_field] = {
            "risk_label": risk_label,
            "random_ordering_aurc": float(risk.mean()),
            "oracle_aurc": tie_aware_expected_aurc(-risk, risk),
        }
    return {
        "mean_dice_coefficient": float(1.0 - dice_loss.mean()),
        "mean_dice_loss": float(dice_loss.mean()),
        "mean_normalized_penalized_hd_loss": float(_array(data, "risk_nhd").mean()),
        "mean_normalized_penalized_hd95_loss": float(_array(data, "risk_nhd95").mean()),
        "deployed_prediction_empty_rate": float(np.mean(prediction_fraction == 0.0)),
        "reference_truth_empty_rate": float(np.mean(truth_fraction == 0.0)),
        "aurc_references": references,
    }


def analyze_condition(data: ConditionData, *, group_bins: int) -> dict:
    key = data.dataset, data.condition
    model = data.manifest.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{data.manifest_path}.model must be a nonempty string")
    return {
        "dataset": data.dataset,
        "condition": data.condition,
        "model": model,
        "is_target_condition": key in TARGET_CONDITIONS,
        "num_images": len(data.rows),
        "model_action_quality": _action_quality(data),
        "indexed_score_agreement": _score_agreement(data),
        "dice_exact_vs_sdc": _dice_exact_sdc_diagnostic(data),
        "working_risk_grouped_reliability": _working_risk_reliability(
            data, bins=group_bins
        ),
    }


def analyze(
    input_paths: Sequence[str | os.PathLike[str]],
    *,
    campaign_lock: str | os.PathLike[str] | None,
    group_bins: int = DEFAULT_GROUP_BINS,
    allow_incomplete: bool = False,
) -> dict:
    if (
        isinstance(group_bins, bool)
        or not isinstance(group_bins, int)
        or group_bins <= 0
    ):
        raise ValueError("group_bins must be a positive integer")
    data = load_inputs(input_paths, allow_incomplete=allow_incomplete)
    if allow_incomplete:
        if campaign_lock is not None:
            raise ValueError(
                "--allow-incomplete is an explicitly unbound smoke mode; "
                "do not supply --campaign-lock"
            )
        provenance: Mapping[str, Any] = {
            "binding": "unbound-incomplete-smoke-test",
            "analysis_source_sha256": _source_sha256(),
            "inputs": [
                {
                    "dataset": item.dataset,
                    "condition": item.condition,
                    "manifest_sha256": _sha256(item.manifest_path),
                    "records_sha256": _sha256(item.jsonl_path),
                }
                for item in data
            ],
        }
    else:
        if campaign_lock is None:
            raise ValueError("canonical diagnostics require --campaign-lock")
        provenance = dict(validate_campaign_bound_conditions(data, campaign_lock))
        provenance["working_diagnostic_source_sha256"] = _source_sha256()

    conditions = [analyze_condition(item, group_bins=group_bins) for item in data]
    all_keys = [f"{item['dataset']}/{item['condition']}" for item in conditions]
    target_keys = [
        f"{item['dataset']}/{item['condition']}"
        for item in conditions
        if item["is_target_condition"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scope": {
            "analysis_level": "held-out population grouped diagnostic",
            "posterior_limitation": (
                "one reference mask per image does not identify the true conditional "
                "mask posterior, validate the shared-threshold coupling, or establish "
                "pointwise conditional-risk calibration"
            ),
            "label_use": (
                "held-out labels summarize action quality and grouped agreement only; "
                "they do not fit, tune, or select a confidence score or image"
            ),
            "score_orientation": (
                "larger confidence is safer; each matched predicted working risk is "
                "the negative of its confidence"
            ),
            "agreement_score_policy": (
                "ranking and accepted-set agreement compare the three primary M32 "
                "loss-indexed scores"
            ),
            "reliability_score_policy": (
                "matched reliability uses Dice-Exact for Dice and M32 for nHD/nHD95; "
                "the Dice-Exact versus SDC identity remains a separate diagnostic"
            ),
            "accepted_set_definition": (
                "score-tie groups receive the same fractional inclusion probability "
                "needed to achieve the requested coverage; overlap is min-over-max "
                "Jaccard of those inclusion-weight vectors"
            ),
        },
        "specification": {
            "grouping": (
                "deterministic equal-count groups ordered by predicted working risk "
                "and then sample_id; exact score ties may span adjacent groups"
            ),
            "requested_group_bins": group_bins,
            "accepted_set_coverages": list(COVERAGES),
            "agreement_scores": [
                {
                    "score_field": score,
                    "label": label,
                }
                for score, label in AGREEMENT_SCORES
            ],
            "matched_reliability_scores": [
                {
                    "score_field": score,
                    "matched_observed_loss_field": risk,
                    "label": label,
                }
                for score, risk, label in RELIABILITY_SCORES
            ],
            "mask_size_strata": [
                {
                    "name": name,
                    "lower": lower,
                    "upper": upper,
                    "lower_inclusive": name == "empty",
                    "upper_inclusive": True,
                }
                for name, lower, upper in MASK_SIZE_STRATA
            ],
        },
        "condition_sets": {
            "all_analyzed_conditions": all_keys,
            "target_condition_definition": (
                "clipseg-target and deeplabv3-target on each of the five datasets"
            ),
            "target_conditions": target_keys,
            "num_analyzed_conditions": len(all_keys),
            "num_target_conditions": len(target_keys),
        },
        "provenance": provenance,
        "conditions": conditions,
    }
    # Enforce strict standard JSON now, before publishing any bytes.
    _canonical_json(report)
    if not allow_incomplete:
        if len(conditions) != 16 or len(target_keys) != 10:
            raise AssertionError(
                "canonical condition partition must be 16 total / 10 target"
            )
    return report


def write_report(report: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {path}")
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
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The target was checked above and this analysis never intentionally replaces it.
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze(
        args.inputs,
        campaign_lock=args.campaign_lock,
        group_bins=args.group_bins,
        allow_incomplete=args.allow_incomplete,
    )
    destination = write_report(report, args.output)
    print(_portable_path(destination))


if __name__ == "__main__":
    main()
