"""Render one manuscript-ready TeX diagnostic artifact from strict JSON.

The renderer performs no statistical estimation beyond deterministic
minimum--maximum summaries across the ten predeclared target conditions.  It
does not discover inputs, choose examples, or modify manuscript sources.  The
single output contains a target-condition action-quality table and a compact
two-panel range table for score agreement and grouped working-risk reliability.
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

from scripts.analyze.main import EXPECTED_CONDITIONS
from scripts.analyze.working_risk import (
    AGREEMENT_SCORES,
    ARTIFACT_TYPE,
    COVERAGES,
    RELIABILITY_SCORES,
    SCHEMA_VERSION,
    TARGET_CONDITIONS,
)


OUTPUT_NAME = "working_risk_diagnostics.tex"
DATASET_ORDER = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
TARGET_MODEL_ORDER = ("clipseg-target", "deeplabv3-target")
TARGET_MODEL_LABELS = {
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
}
ORDERED_TARGET_CONDITIONS = tuple(
    key for key in EXPECTED_CONDITIONS if key in TARGET_CONDITIONS
)
PAIR_ORDER = (
    ("confidence_dice_m32", "confidence_nhd_m32", "Dice-M32--nHD-M32"),
    ("confidence_nhd_m32", "confidence_nhd95_m32", "nHD-M32--nHD95-M32"),
)
RELIABILITY_ORDER = (
    ("confidence_dice_exact", "risk_dice", r"Dice-Exact $\to$ Dice"),
    ("confidence_nhd_m32", "risk_nhd", r"nHD-M32 $\to$ nHD"),
    ("confidence_nhd95_m32", "risk_nhd95", r"nHD95-M32 $\to$ nHD95"),
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "scope",
        "specification",
        "condition_sets",
        "provenance",
        "conditions",
    }
)
CONDITION_KEYS = frozenset(
    {
        "dataset",
        "condition",
        "model",
        "is_target_condition",
        "num_images",
        "model_action_quality",
        "indexed_score_agreement",
        "dice_exact_vs_sdc",
        "working_risk_grouped_reliability",
    }
)
QUALITY_KEYS = frozenset(
    {
        "mean_dice_coefficient",
        "mean_dice_loss",
        "mean_normalized_penalized_hd_loss",
        "mean_normalized_penalized_hd95_loss",
        "deployed_prediction_empty_rate",
        "reference_truth_empty_rate",
        "aurc_references",
    }
)
SPECIFICATION_KEYS = frozenset(
    {
        "grouping",
        "requested_group_bins",
        "accepted_set_coverages",
        "agreement_scores",
        "matched_reliability_scores",
        "mask_size_strata",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics", required=True, help="strict grouped diagnostics JSON"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/binary_working_risk_diagnostics/rendered",
    )
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


def _assert_finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, location=f"{location}[{index}]")


def load_diagnostics(path: str | os.PathLike[str]) -> tuple[dict, str]:
    """Load duplicate-key-free, finite standard JSON and return its byte hash."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"diagnostics JSON does not exist: {source}")
    raw = source.read_bytes()
    try:
        result = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("diagnostics root must be a JSON object")
    _assert_finite_tree(result, location=str(source))
    return result, hashlib.sha256(raw).hexdigest()


def _exact_mapping(value: Any, keys: frozenset[str], *, location: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{location} must contain exactly {sorted(keys)}; got {observed}"
        )
    return value


def _list(value: Any, *, location: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _number(
    value: Any,
    *,
    location: str,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{location} must be finite and lie in [{minimum}, {maximum}]")
    return result


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _correlation_value(value: Any, *, location: str) -> float:
    if not isinstance(value, dict) or set(value) != {
        "defined",
        "value",
        "undefined_reason",
    }:
        raise ValueError(f"{location} has an invalid correlation schema")
    if value["defined"] is not True or value["undefined_reason"] is not None:
        raise ValueError(
            f"{location} must be defined for every canonical target condition"
        )
    return _number(value["value"], location=f"{location}.value")


def _validate_quality(value: Any, *, location: str) -> dict:
    quality = _exact_mapping(value, QUALITY_KEYS, location=location)
    for field in QUALITY_KEYS - {"aurc_references"}:
        _number(quality[field], location=f"{location}.{field}", minimum=0.0)
    if not math.isclose(
        quality["mean_dice_coefficient"],
        1.0 - quality["mean_dice_loss"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{location} has inconsistent mean Dice coefficient/loss")
    if not isinstance(quality["aurc_references"], dict):
        raise ValueError(f"{location}.aurc_references must be an object")
    return quality


def _validate_specification(value: Any) -> None:
    specification = _exact_mapping(
        value, SPECIFICATION_KEYS, location="diagnostics.specification"
    )
    _string(specification["grouping"], location="diagnostics.specification.grouping")
    group_bins = specification["requested_group_bins"]
    if (
        isinstance(group_bins, bool)
        or not isinstance(group_bins, int)
        or group_bins <= 0
    ):
        raise ValueError(
            "diagnostics.specification.requested_group_bins must be a positive integer"
        )
    if specification["accepted_set_coverages"] != list(COVERAGES):
        raise ValueError("diagnostics.specification.accepted_set_coverages has changed")
    expected_agreement = [
        {"score_field": score, "label": label} for score, label in AGREEMENT_SCORES
    ]
    if specification["agreement_scores"] != expected_agreement:
        raise ValueError(
            "diagnostics.specification.agreement_scores must be the three primary "
            "M32 scores"
        )
    expected_reliability = [
        {
            "score_field": score,
            "matched_observed_loss_field": risk,
            "label": label,
        }
        for score, risk, label in RELIABILITY_SCORES
    ]
    if specification["matched_reliability_scores"] != expected_reliability:
        raise ValueError(
            "diagnostics.specification.matched_reliability_scores must use "
            "Dice-Exact and the two boundary M32 scores"
        )
    _list(
        specification["mask_size_strata"],
        location="diagnostics.specification.mask_size_strata",
    )


def _validate_agreements(value: Any, *, location: str) -> dict[tuple[str, str], dict]:
    rows = _list(value, location=location)
    expected_pairs = {
        (left, right)
        for index, (left, _) in enumerate(AGREEMENT_SCORES)
        for right, _ in AGREEMENT_SCORES[index + 1 :]
    }
    by_pair = {}
    for index, row in enumerate(rows):
        row_location = f"{location}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_location} must be an object")
        required = {
            "left_score",
            "left_label",
            "right_score",
            "right_label",
            "spearman_rho",
            "kendall_tau_b",
            "accepted_set_agreement",
        }
        if set(row) != required:
            raise ValueError(f"{row_location} has an invalid score-agreement schema")
        pair = (
            _string(row["left_score"], location=f"{row_location}.left_score"),
            _string(row["right_score"], location=f"{row_location}.right_score"),
        )
        if pair in by_pair:
            raise ValueError(f"{location} contains duplicate pair {pair}")
        _correlation_value(row["spearman_rho"], location=f"{row_location}.spearman")
        _correlation_value(row["kendall_tau_b"], location=f"{row_location}.kendall")
        overlap_rows = _list(
            row["accepted_set_agreement"],
            location=f"{row_location}.accepted_set_agreement",
        )
        coverage_map = {}
        for overlap_index, overlap in enumerate(overlap_rows):
            overlap_location = f"{row_location}.accepted[{overlap_index}]"
            if not isinstance(overlap, dict) or set(overlap) != {
                "coverage",
                "tie_aware_fractional_jaccard",
            }:
                raise ValueError(f"{overlap_location} has an invalid schema")
            coverage = _number(
                overlap["coverage"],
                location=f"{overlap_location}.coverage",
                minimum=0.0,
            )
            if coverage in coverage_map:
                raise ValueError(
                    f"{row_location} contains duplicate coverage {coverage}"
                )
            coverage_map[coverage] = _number(
                overlap["tie_aware_fractional_jaccard"],
                location=f"{overlap_location}.jaccard",
                minimum=0.0,
            )
        if set(coverage_map) != set(COVERAGES):
            raise ValueError(f"{row_location} must contain coverages {COVERAGES}")
        by_pair[pair] = row
    if set(by_pair) != expected_pairs:
        raise ValueError(f"{location} does not contain exactly the three indexed pairs")
    return by_pair


def _validate_reliability(value: Any, *, location: str) -> dict[str, dict]:
    rows = _list(value, location=location)
    expected = {score: risk for score, risk, _ in RELIABILITY_SCORES}
    by_score = {}
    metric_fields = (
        "bias_predicted_minus_observed",
        "per_image_mae",
        "per_image_rmse",
        "grouped_ece",
    )
    for index, row in enumerate(rows):
        row_location = f"{location}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_location} must be an object")
        score = _string(row.get("score_field"), location=f"{row_location}.score_field")
        risk = _string(
            row.get("observed_loss_field"),
            location=f"{row_location}.observed_loss_field",
        )
        if score in by_score:
            raise ValueError(f"{location} contains duplicate score {score}")
        if expected.get(score) != risk:
            raise ValueError(f"{row_location} is not the declared matched score/risk")
        for field in metric_fields:
            lower = -1.0 if field.startswith("bias_") else 0.0
            _number(row.get(field), location=f"{row_location}.{field}", minimum=lower)
        by_score[score] = row
    if set(by_score) != set(expected):
        raise ValueError(
            f"{location} does not contain exactly the three matched scores"
        )
    return by_score


def validate_diagnostics(value: Any) -> dict:
    root = _exact_mapping(value, TOP_LEVEL_KEYS, location="diagnostics")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"diagnostics.schema_version must equal {SCHEMA_VERSION}")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError(f"diagnostics.artifact_type must equal {ARTIFACT_TYPE!r}")
    scope = root["scope"]
    if not isinstance(scope, dict) or "does not identify" not in str(
        scope.get("posterior_limitation", "")
    ):
        raise ValueError("diagnostics scope must preserve the posterior limitation")
    _validate_specification(root["specification"])

    condition_sets = root["condition_sets"]
    if not isinstance(condition_sets, dict):
        raise ValueError("diagnostics.condition_sets must be an object")
    expected_all = [
        f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
    ]
    expected_target = [
        f"{dataset}/{condition}" for dataset, condition in ORDERED_TARGET_CONDITIONS
    ]
    if condition_sets.get("all_analyzed_conditions") != expected_all:
        raise ValueError("diagnostics must name the exact 16-condition benchmark")
    if condition_sets.get("target_conditions") != expected_target:
        raise ValueError("diagnostics must name the exact ten target conditions")
    if condition_sets.get("num_analyzed_conditions") != 16:
        raise ValueError("diagnostics.num_analyzed_conditions must equal 16")
    if condition_sets.get("num_target_conditions") != 10:
        raise ValueError("diagnostics.num_target_conditions must equal 10")

    conditions = _list(root["conditions"], location="diagnostics.conditions")
    by_key = {}
    for index, raw in enumerate(conditions):
        location = f"diagnostics.conditions[{index}]"
        row = _exact_mapping(raw, CONDITION_KEYS, location=location)
        key = (
            _string(row["dataset"], location=f"{location}.dataset"),
            _string(row["condition"], location=f"{location}.condition"),
        )
        if key in by_key:
            raise ValueError(f"diagnostics contains duplicate condition {key}")
        if key not in EXPECTED_CONDITIONS:
            raise ValueError(f"diagnostics contains undeclared condition {key}")
        if row["is_target_condition"] is not (key in TARGET_CONDITIONS):
            raise ValueError(f"{location}.is_target_condition is inconsistent")
        if isinstance(row["num_images"], bool) or not isinstance(
            row["num_images"], int
        ):
            raise ValueError(f"{location}.num_images must be a positive integer")
        if row["num_images"] <= 0:
            raise ValueError(f"{location}.num_images must be a positive integer")
        expected_model = "clipseg" if key[1].startswith("clipseg") else "deeplabv3"
        if row["model"] != expected_model:
            raise ValueError(f"{location}.model is inconsistent with condition")
        quality = _validate_quality(
            row["model_action_quality"], location=f"{location}.quality"
        )
        agreements = _validate_agreements(
            row["indexed_score_agreement"], location=f"{location}.agreement"
        )
        reliability = _validate_reliability(
            row["working_risk_grouped_reliability"],
            location=f"{location}.reliability",
        )
        by_key[key] = {
            "row": row,
            "quality": quality,
            "agreements": agreements,
            "reliability": reliability,
        }
    if set(by_key) != set(EXPECTED_CONDITIONS) or len(conditions) != 16:
        raise ValueError(
            "diagnostics must contain each declared condition exactly once"
        )
    return by_key


def _format(value: float, *, digits: int = 3) -> str:
    threshold = 0.5 * 10 ** (-digits)
    if abs(value) < threshold:
        value = 0.0
    return f"{value:.{digits}f}"


def _range(values: Sequence[float]) -> str:
    if len(values) != 10:
        raise AssertionError("target-condition ranges require exactly ten values")
    return rf"$[{_format(min(values))},\,{_format(max(values))}]$"


def _preamble(source_hash: str) -> list[str]:
    return [
        "% Generated by scripts/render_working_risk_diagnostics.py.",
        f"% Source diagnostics JSON SHA-256: {source_hash}",
        "% Descriptive diagnostic tables; do not edit values by hand.",
        "",
    ]


def _render_action_quality(by_key: Mapping[tuple[str, str], dict]) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Descriptive action quality across the ten target conditions "
        r"(one cell per condition, rather than a range). Each dataset contributes "
        r"CLIP-T and DL-T. The conditions are not independent replicates, and this "
        r"table is not posterior validation.}",
        r"\label{tab:target-action-quality-diagnostics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{10}{c}}",
        r"\toprule",
    ]
    dataset_header = "Metric"
    for dataset in DATASET_ORDER:
        dataset_header += rf" & \multicolumn{{2}}{{c}}{{{DATASET_LABELS[dataset]}}}"
    lines.append(dataset_header + r" \\")
    lines.append(
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
        r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}"
    )
    model_header = ""
    for _ in DATASET_ORDER:
        for condition in TARGET_MODEL_ORDER:
            model_header += f" & {TARGET_MODEL_LABELS[condition]}"
    lines.extend([model_header + r" \\", r"\midrule"])

    rows = (
        (r"Mean Dice $\uparrow$", "mean_dice_coefficient", False),
        (r"Mean nHD $\downarrow$", "mean_normalized_penalized_hd_loss", False),
        (r"Mean nHD95 $\downarrow$", "mean_normalized_penalized_hd95_loss", False),
        ("Predicted empty", "deployed_prediction_empty_rate", True),
    )
    for label, field, percentage in rows:
        cells = [label]
        for dataset in DATASET_ORDER:
            for condition in TARGET_MODEL_ORDER:
                value = by_key[(dataset, condition)]["quality"][field]
                cells.append(f"{100 * value:.1f}\\%" if percentage else _format(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def _target_items(by_key: Mapping[tuple[str, str], dict]) -> list[dict]:
    return [by_key[key] for key in ORDERED_TARGET_CONDITIONS]


def _render_ranges(by_key: Mapping[tuple[str, str], dict]) -> list[str]:
    targets = _target_items(by_key)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Descriptive minimum--maximum ranges across the ten target "
        r"conditions for indexed-score agreement and matched single-label proxy--outcome agreement. "
        r"Per-image errors combine proxy mismatch with realized-label variation; they are "
        r"not conditional calibration estimates or posterior validation. The ten conditions are not "
        r"independent replicates. Accepted-set agreement is the tie-aware fractional "
        r"Jaccard at the stated coverage.}",
        r"\label{tab:target-working-risk-diagnostic-ranges}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\centering",
        r"\textbf{(a) Indexed-score ranking and accepted-set agreement}\\[2pt]",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Score pair & Spearman $\rho$ & Kendall $\tau_b$ & $J_{0.25}$ & $J_{0.50}$ & $J_{0.75}$ \\",
        r"\midrule",
    ]
    for left, right, label in PAIR_ORDER:
        pair_rows = [item["agreements"][(left, right)] for item in targets]
        spearman = [row["spearman_rho"]["value"] for row in pair_rows]
        kendall = [row["kendall_tau_b"]["value"] for row in pair_rows]
        overlaps = {
            coverage: [
                next(
                    overlap["tie_aware_fractional_jaccard"]
                    for overlap in row["accepted_set_agreement"]
                    if overlap["coverage"] == coverage
                )
                for row in pair_rows
            ]
            for coverage in COVERAGES
        }
        cells = [label, _range(spearman), _range(kendall)]
        cells.extend(_range(overlaps[coverage]) for coverage in COVERAGES)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{5pt}",
            r"\textbf{(b) Descriptive matched proxy--outcome agreement}\\[2pt]",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Working proxy $\to$ observed loss & Bias & Per-image MAE & Per-image RMSE & 10-bin abs. gap \\",
            r"\midrule",
        ]
    )
    metrics = (
        "bias_predicted_minus_observed",
        "per_image_mae",
        "per_image_rmse",
        "grouped_ece",
    )
    for score, _, label in RELIABILITY_ORDER:
        rows = [item["reliability"][score] for item in targets]
        cells = [label]
        cells.extend(_range([row[metric] for row in rows]) for metric in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def render_diagnostics(value: Any, *, source_hash: str) -> str:
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("source_hash must be a lowercase SHA-256 digest")
    by_key = validate_diagnostics(value)
    lines = _preamble(source_hash)
    lines.extend(_render_action_quality(by_key))
    lines.extend(_render_ranges(by_key))
    return "\n".join(lines)


def write_output(tex: str, output_dir: str | os.PathLike[str]) -> Path:
    directory = Path(output_dir)
    destination = directory / OUTPUT_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite rendered diagnostics: {destination}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_NAME}.tmp-", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(tex)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    diagnostics, source_hash = load_diagnostics(args.diagnostics)
    tex = render_diagnostics(diagnostics, source_hash=source_hash)
    destination = write_output(tex, args.output_dir)
    print(destination.as_posix())


if __name__ == "__main__":
    main()
