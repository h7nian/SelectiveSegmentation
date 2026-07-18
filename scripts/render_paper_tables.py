"""Render the four manuscript tables from a validated binary analysis.

This is intentionally a one-way renderer: every displayed number must already
exist in ``analysis.json``.  The default mode accepts only the complete,
predeclared ten-condition experiment with 10,000 bootstrap resamples and
95% percentile intervals. ``--allow-incomplete`` is only for draft smoke tests
and marks every generated table as incomplete.

Example::

    python scripts/render_paper_tables.py \
        --analysis outputs/binary/analysis/analysis.json \
        --output-dir docs/Tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
EXPECTED_RESAMPLES = 10_000
EXPECTED_CONFIDENCE_LEVEL = 0.95

RISKS = {
    "risk_dice": "Dice risk",
    "risk_nhd95": "nHD95 risk",
}
SOURCE_RISK_LABELS = {
    "risk_dice": "Dice risk",
    "risk_nhd95": "Normalized penalized HD95 risk",
}
METHODS = {
    "confidence_sdc": "SDC",
    "confidence_mean_max_probability": "Mean max probability",
    "confidence_negative_entropy": "Negative entropy",
    "confidence_dice_m2": "Dice-M2",
    "confidence_dice_m8": "Dice-M8",
    "confidence_dice_m32": "Dice-M32",
    "confidence_nhd95_m2": "nHD95-M2",
    "confidence_nhd95_m8": "nHD95-M8",
    "confidence_nhd95_m32": "nHD95-M32",
}
MAIN_METHODS = (
    "confidence_sdc",
    "confidence_mean_max_probability",
    "confidence_negative_entropy",
    "confidence_dice_m32",
    "confidence_nhd95_m32",
)
MATCHED_METHODS = {
    "risk_dice": (
        "confidence_dice_m2",
        "confidence_dice_m8",
        "confidence_dice_m32",
    ),
    "risk_nhd95": (
        "confidence_nhd95_m2",
        "confidence_nhd95_m8",
        "confidence_nhd95_m32",
    ),
}
PRIMARY_PAIR = ("confidence_dice_m32", "confidence_nhd95_m32")

# This is the ten-condition matrix declared in Sections/05_experiments.tex.
EXPECTED_CONDITIONS = (
    ("pet", "clipseg-general"),
    ("pet", "clipseg-target"),
    ("pet", "deeplabv3-target"),
    ("pet", "deeplabv3-external"),
    ("kvasir", "clipseg-general"),
    ("kvasir", "clipseg-target"),
    ("kvasir", "deeplabv3-target"),
    ("fives", "clipseg-general"),
    ("fives", "clipseg-target"),
    ("fives", "deeplabv3-target"),
)
CONDITION_ORDER = {key: index for index, key in enumerate(EXPECTED_CONDITIONS)}

OUTPUT_NAMES = (
    "main_results.tex",
    "cross_loss_results.tex",
    "quadrature_ablation.tex",
    "statistical_tests.tex",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="validated analysis.json")
    parser.add_argument("--output-dir", default="docs/Tables")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "allow a nonempty subset of the ten conditions and non-final "
            "resample counts; generated tables are visibly marked INCOMPLETE"
        ),
    )
    return parser.parse_args(argv)


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_analysis(path):
    """Read strict JSON, rejecting duplicate keys and non-finite numbers."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"analysis does not exist: {path}")
    try:
        result = json.loads(
            path.read_text(),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("analysis root must be a JSON object")
    return result


def _mapping(value, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _sequence(value, location):
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _finite(value, location, *, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "a finite number or null" if nullable else "a finite number"
        raise ValueError(f"{location} must be {qualifier}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{location} must be finite")
    return value


def _integer(value, location, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{location} must be positive")
    return value


def _string(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _close(left, right, *, tolerance=1e-10):
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _validate_method(method, location, *, expected_label, oracle, random):
    method = _mapping(method, location)
    required = {
        "label",
        "aurc",
        "oracle_aurc",
        "random_aurc",
        "excess_aurc",
        "normalized_aurc",
    }
    if set(method) != required:
        raise ValueError(
            f"{location} must contain exactly {sorted(required)}; "
            f"got {sorted(method)}"
        )
    if method["label"] != expected_label:
        raise ValueError(
            f"{location}.label must be {expected_label!r}, got {method['label']!r}"
        )
    aurc = _finite(method["aurc"], f"{location}.aurc")
    method_oracle = _finite(method["oracle_aurc"], f"{location}.oracle_aurc")
    method_random = _finite(method["random_aurc"], f"{location}.random_aurc")
    excess = _finite(method["excess_aurc"], f"{location}.excess_aurc")
    normalized = _finite(
        method["normalized_aurc"], f"{location}.normalized_aurc", nullable=True
    )
    for name, value in (
        ("aurc", aurc),
        ("oracle_aurc", method_oracle),
        ("random_aurc", method_random),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{location}.{name} must lie in [0, 1]")
    if not _close(method_oracle, oracle) or not _close(method_random, random):
        raise ValueError(f"{location} disagrees with its risk-level baselines")
    if not _close(excess, aurc - oracle):
        raise ValueError(f"{location}.excess_aurc is inconsistent")
    denominator = random - oracle
    if abs(denominator) <= 1e-15:
        if normalized is not None:
            raise ValueError(
                f"{location}.normalized_aurc must be null for zero denominator"
            )
    else:
        expected = excess / denominator
        if normalized is None or not _close(normalized, expected):
            raise ValueError(f"{location}.normalized_aurc is inconsistent")


def _validate_comparison(
    comparison,
    condition,
    risk_field,
    location,
    *,
    bootstrap_samples,
    confidence_level,
    observations,
    clusters,
):
    comparison = _mapping(comparison, location)
    required = {
        "left",
        "right",
        "difference_left_minus_right",
        "bootstrap",
        "holm_adjusted_p_value",
    }
    if set(comparison) != required:
        raise ValueError(f"{location} has an unexpected comparison schema")
    if (comparison["left"], comparison["right"]) != PRIMARY_PAIR:
        raise ValueError(f"{location} must compare Dice-M32 with nHD95-M32")

    risk = condition["risks"][risk_field]
    methods = risk["methods"]
    expected_difference = (
        methods[PRIMARY_PAIR[0]]["aurc"] - methods[PRIMARY_PAIR[1]]["aurc"]
    )
    difference = _finite(
        comparison["difference_left_minus_right"],
        f"{location}.difference_left_minus_right",
    )
    if not _close(difference, expected_difference):
        raise ValueError(f"{location} difference disagrees with the AURCs")

    bootstrap = _mapping(comparison["bootstrap"], f"{location}.bootstrap")
    bootstrap_required = {
        "difference",
        "ci_low",
        "ci_high",
        "confidence_level",
        "p_value",
        "n_resamples",
        "n_observations",
        "n_clusters",
        "seed",
    }
    if set(bootstrap) != bootstrap_required:
        raise ValueError(f"{location}.bootstrap has an unexpected schema")
    bootstrap_difference = _finite(
        bootstrap["difference"], f"{location}.bootstrap.difference"
    )
    ci_low = _finite(bootstrap["ci_low"], f"{location}.bootstrap.ci_low")
    ci_high = _finite(bootstrap["ci_high"], f"{location}.bootstrap.ci_high")
    bootstrap_confidence_level = _finite(
        bootstrap["confidence_level"], f"{location}.bootstrap.confidence_level"
    )
    p_value = _finite(bootstrap["p_value"], f"{location}.bootstrap.p_value")
    if not _close(bootstrap_difference, difference):
        raise ValueError(f"{location}.bootstrap difference is inconsistent")
    if ci_low > ci_high:
        raise ValueError(f"{location}.bootstrap interval is reversed")
    if not 0 < bootstrap_confidence_level < 1:
        raise ValueError(f"{location}.bootstrap confidence level is invalid")
    if not 0 <= p_value <= 1:
        raise ValueError(f"{location}.bootstrap.p_value must lie in [0, 1]")
    n_resamples = _integer(
        bootstrap["n_resamples"],
        f"{location}.bootstrap.n_resamples",
        positive=True,
    )
    n_observations = _integer(
        bootstrap["n_observations"],
        f"{location}.bootstrap.n_observations",
        positive=True,
    )
    n_clusters = _integer(
        bootstrap["n_clusters"],
        f"{location}.bootstrap.n_clusters",
        positive=True,
    )
    _integer(bootstrap["seed"], f"{location}.bootstrap.seed")
    if (
        n_resamples != bootstrap_samples
        or n_observations != observations
        or n_clusters != clusters
        or not _close(bootstrap_confidence_level, confidence_level)
    ):
        raise ValueError(f"{location}.bootstrap disagrees with the analysis metadata")
    adjusted = _finite(
        comparison["holm_adjusted_p_value"],
        f"{location}.holm_adjusted_p_value",
    )
    if not 0 <= adjusted <= 1:
        raise ValueError(f"{location}.holm_adjusted_p_value must lie in [0, 1]")


def validate_analysis(result, *, allow_incomplete=False):
    """Validate the exact analysis contract consumed by the manuscript."""

    result = _mapping(result, "analysis root")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    analysis = _mapping(result.get("analysis"), "analysis")
    conditions = _sequence(result.get("conditions"), "conditions")
    multiple = _mapping(result.get("multiple_testing"), "multiple_testing")

    condition_count = len(conditions)
    if allow_incomplete:
        if not 0 < condition_count <= len(EXPECTED_CONDITIONS):
            raise ValueError("incomplete analysis must contain 1--10 conditions")
    elif condition_count != len(EXPECTED_CONDITIONS):
        raise ValueError(
            f"complete analysis must contain exactly {len(EXPECTED_CONDITIONS)} "
            f"conditions, got {condition_count}"
        )

    expected_analysis_fields = {
        "tie_policy",
        "normalized_aurc",
        "comparison",
        "bootstrap_samples",
        "confidence_level",
        "bootstrap_p_value",
        "seed",
    }
    if set(analysis) != expected_analysis_fields:
        raise ValueError("analysis metadata has an unexpected schema")
    for field in (
        "tie_policy",
        "normalized_aurc",
        "comparison",
        "bootstrap_p_value",
    ):
        _string(analysis[field], f"analysis.{field}")
    _integer(analysis["seed"], "analysis.seed")
    bootstrap_samples = _integer(
        analysis.get("bootstrap_samples"), "analysis.bootstrap_samples", positive=True
    )
    confidence_level = _finite(
        analysis.get("confidence_level"), "analysis.confidence_level"
    )
    if not allow_incomplete and (
        bootstrap_samples != EXPECTED_RESAMPLES
        or not _close(confidence_level, EXPECTED_CONFIDENCE_LEVEL)
    ):
        raise ValueError(
            "final tables require 10,000 paired bootstrap resamples and "
            "95% percentile intervals"
        )

    by_key = {}
    for index, condition in enumerate(conditions):
        location = f"conditions[{index}]"
        condition = _mapping(condition, location)
        dataset = _string(condition.get("dataset"), f"{location}.dataset")
        name = _string(condition.get("condition"), f"{location}.condition")
        key = (dataset, name)
        if key in by_key:
            raise ValueError(f"duplicate dataset/condition pair {key!r}")
        if key not in CONDITION_ORDER:
            raise ValueError(f"undeclared dataset/condition pair {key!r}")
        by_key[key] = condition
        if condition.get("split") != "test":
            raise ValueError(f"{location}.split must be 'test'")
        observations = _integer(
            condition.get("num_rows"), f"{location}.num_rows", positive=True
        )
        clusters = _integer(
            condition.get("num_image_clusters"),
            f"{location}.num_image_clusters",
            positive=True,
        )

        risks = _mapping(condition.get("risks"), f"{location}.risks")
        comparisons = _mapping(
            condition.get("comparisons"), f"{location}.comparisons"
        )
        if set(risks) != set(RISKS):
            raise ValueError(f"{location} must contain exactly the two risks")
        if set(comparisons) != set(RISKS):
            raise ValueError(f"{location} must compare exactly the two risks")
        for risk_field, expected_label in SOURCE_RISK_LABELS.items():
            risk_location = f"{location}.risks.{risk_field}"
            risk = _mapping(risks[risk_field], risk_location)
            if set(risk) != {"label", "methods", "oracle_aurc", "random_aurc"}:
                raise ValueError(f"{risk_location} has an unexpected schema")
            if risk["label"] != expected_label:
                raise ValueError(f"{risk_location}.label is unexpected")
            oracle = _finite(risk["oracle_aurc"], f"{risk_location}.oracle_aurc")
            random = _finite(risk["random_aurc"], f"{risk_location}.random_aurc")
            if not 0 <= oracle <= random <= 1:
                raise ValueError(f"{risk_location} oracle/random AURCs are invalid")
            methods = _mapping(risk["methods"], f"{risk_location}.methods")
            if set(methods) != set(METHODS):
                raise ValueError(
                    f"{risk_location} must contain exactly the nine methods"
                )
            for method_field, label in METHODS.items():
                _validate_method(
                    methods[method_field],
                    f"{risk_location}.methods.{method_field}",
                    expected_label=label,
                    oracle=oracle,
                    random=random,
                )
            _validate_comparison(
                comparisons[risk_field],
                condition,
                risk_field,
                f"{location}.comparisons.{risk_field}",
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                observations=observations,
                clusters=clusters,
            )

    expected_keys = set(EXPECTED_CONDITIONS)
    if not allow_incomplete and set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        raise ValueError(f"complete analysis is missing declared conditions {missing}")

    expected_multiple_fields = {
        "procedure",
        "family",
        "num_hypotheses",
        "raw_bootstrap_p_values",
        "holm_adjusted_p_values",
        "hypotheses",
        "confidence_intervals",
        "significance_calls",
    }
    if set(multiple) != expected_multiple_fields:
        raise ValueError("multiple_testing has an unexpected schema")
    for field in ("procedure", "family", "confidence_intervals"):
        _string(multiple[field], f"multiple_testing.{field}")
    if multiple["significance_calls"] != "not made by this analysis":
        raise ValueError("multiple_testing.significance_calls is unexpected")

    hypothesis_count = _integer(
        multiple.get("num_hypotheses"), "multiple_testing.num_hypotheses"
    )
    if hypothesis_count != 2 * condition_count:
        raise ValueError("multiple_testing must contain two hypotheses per condition")
    hypotheses = _sequence(
        multiple.get("hypotheses"), "multiple_testing.hypotheses"
    )
    if len(hypotheses) != hypothesis_count:
        raise ValueError("multiple_testing hypothesis count is inconsistent")
    raw_p_values = [
        _finite(value, f"multiple_testing.raw_bootstrap_p_values[{index}]")
        for index, value in enumerate(
            _sequence(
                multiple["raw_bootstrap_p_values"],
                "multiple_testing.raw_bootstrap_p_values",
            )
        )
    ]
    adjusted_p_values = [
        _finite(value, f"multiple_testing.holm_adjusted_p_values[{index}]")
        for index, value in enumerate(
            _sequence(
                multiple["holm_adjusted_p_values"],
                "multiple_testing.holm_adjusted_p_values",
            )
        )
    ]
    if len(raw_p_values) != hypothesis_count or len(adjusted_p_values) != hypothesis_count:
        raise ValueError("multiple_testing p-value array lengths are inconsistent")
    if any(not 0 <= value <= 1 for value in raw_p_values + adjusted_p_values):
        raise ValueError("multiple_testing p-values must lie in [0, 1]")
    expected_adjusted = _holm_adjust(raw_p_values)
    if any(
        not _close(actual, expected)
        for actual, expected in zip(adjusted_p_values, expected_adjusted)
    ):
        raise ValueError("multiple_testing Holm values are inconsistent")

    hypothesis_by_key = {}
    hypothesis_raw = []
    hypothesis_adjusted = []
    for index, hypothesis in enumerate(hypotheses):
        location = f"multiple_testing.hypotheses[{index}]"
        hypothesis = _mapping(hypothesis, location)
        required = {
            "dataset",
            "condition",
            "risk",
            "raw_bootstrap_p_value",
            "holm_adjusted_p_value",
        }
        if set(hypothesis) != required:
            raise ValueError(f"{location} has an unexpected schema")
        key = (
            _string(hypothesis["dataset"], f"{location}.dataset"),
            _string(hypothesis["condition"], f"{location}.condition"),
            _string(hypothesis["risk"], f"{location}.risk"),
        )
        if key in hypothesis_by_key:
            raise ValueError(f"duplicate multiple-testing hypothesis {key!r}")
        hypothesis_by_key[key] = hypothesis
        hypothesis_raw.append(
            _finite(
                hypothesis["raw_bootstrap_p_value"],
                f"{location}.raw_bootstrap_p_value",
            )
        )
        hypothesis_adjusted.append(
            _finite(
                hypothesis["holm_adjusted_p_value"],
                f"{location}.holm_adjusted_p_value",
            )
        )

    if any(
        not _close(left, right)
        for left, right in zip(raw_p_values, hypothesis_raw)
    ) or any(
        not _close(left, right)
        for left, right in zip(adjusted_p_values, hypothesis_adjusted)
    ):
        raise ValueError("multiple_testing arrays disagree with hypotheses")

    for (dataset, name), condition in by_key.items():
        for risk_field in RISKS:
            key = (dataset, name, risk_field)
            if key not in hypothesis_by_key:
                raise ValueError(f"missing multiple-testing hypothesis {key!r}")
            hypothesis = hypothesis_by_key[key]
            comparison = condition["comparisons"][risk_field]
            raw = _finite(
                hypothesis["raw_bootstrap_p_value"],
                f"hypothesis {key}.raw_bootstrap_p_value",
            )
            adjusted = _finite(
                hypothesis["holm_adjusted_p_value"],
                f"hypothesis {key}.holm_adjusted_p_value",
            )
            if not _close(raw, comparison["bootstrap"]["p_value"]):
                raise ValueError(f"hypothesis {key!r} has inconsistent raw p-value")
            if not _close(adjusted, comparison["holm_adjusted_p_value"]):
                raise ValueError(
                    f"hypothesis {key!r} has inconsistent Holm p-value"
                )

    if not allow_incomplete and hypothesis_count != 20:
        raise ValueError("final tables require exactly 20 statistical comparisons")
    return sorted(
        by_key.values(),
        key=lambda item: CONDITION_ORDER[(item["dataset"], item["condition"])],
    )


def _escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _dataset(value):
    return {"pet": "Oxford Pet", "kvasir": "Kvasir-SEG", "fives": "FIVES"}[value]


def _condition(value):
    return {
        "clipseg-general": "CLIPSeg-General",
        "clipseg-target": "CLIPSeg-Target",
        "deeplabv3-target": "DeepLabV3-Target",
        "deeplabv3-external": "DeepLabV3-External",
    }[value]


def _number(value):
    return "--" if value is None else f"{value:.4f}"


def _aurc_cell(method):
    return f"{_number(method['aurc'])} ({_number(method['normalized_aurc'])})"


def _generated_header(source_hash, *, incomplete):
    status = "INCOMPLETE DRAFT SMOKE TEST" if incomplete else "COMPLETE FINAL ANALYSIS"
    return (
        "% AUTO-GENERATED by scripts/render_paper_tables.py; DO NOT EDIT.\n"
        f"% Source analysis.json SHA-256: {source_hash}\n"
        f"% Status: {status}\n"
    )


def render_main_results(conditions, *, header):
    method_headers = ("SDC", "MMP", r"$-H$", "Dice-M32", "nHD95-M32")
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary selective-segmentation results. Each entry is raw AURC "
        r"(nAURC); lower is better for both quantities. AURC is the analytic "
        r"expectation over uniform random order within exact confidence ties. "
        r"MMP denotes mean maximum probability and $-H$ negative mean entropy.}",
        r"\label{tab:main-results}",
        r"{\scriptsize\setlength{\tabcolsep}{3pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        "Dataset & Model condition & " + " & ".join(method_headers) + r" \\",
        r"\midrule",
    ]
    for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
        if risk_index:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{_escape(risk_label)}}}}} \\"
        )
        for condition in conditions:
            methods = condition["risks"][risk_field]["methods"]
            cells = [_aurc_cell(methods[field]) for field in MAIN_METHODS]
            lines.append(
                " & ".join(
                    [
                        _escape(_dataset(condition["dataset"])),
                        _escape(_condition(condition["condition"])),
                        *cells,
                    ]
                )
                + r" \\"
            )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}}", r"}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def render_cross_loss_results(conditions, *, header):
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete cross-loss comparison at $M=32$. Each entry is raw "
        r"AURC (nAURC), and lower is better. Exact confidence ties use their "
        r"analytic random-order expectation; no cell is selected or filled by hand.}",
        r"\label{tab:cross-loss-results}",
        r"{\scriptsize\setlength{\tabcolsep}{4pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Dice risk} & \multicolumn{2}{c}{nHD95 risk} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Dataset & Model condition & Dice-M32 & nHD95-M32 & Dice-M32 & nHD95-M32 \\",
        r"\midrule",
    ]
    for condition in conditions:
        cells = []
        for risk_field in RISKS:
            methods = condition["risks"][risk_field]["methods"]
            cells.extend(_aurc_cell(methods[field]) for field in PRIMARY_PAIR)
        lines.append(
            " & ".join(
                [
                    _escape(_dataset(condition["dataset"])),
                    _escape(_condition(condition["condition"])),
                    *cells,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}}", r"}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def render_quadrature_ablation(conditions, *, header):
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-loss midpoint-quadrature ablation. Reported values are "
        r"raw tie-aware AURC (lower is better). Only $M$ changes within each "
        r"loss-indexed score; the deployed action and evaluation cohort are fixed.}",
        r"\label{tab:quadrature-ablation}",
        r"{\scriptsize\setlength{\tabcolsep}{5pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Dice score / Dice risk} & "
        r"\multicolumn{3}{c}{nHD95 score / nHD95 risk} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Dataset & Model condition & $M=2$ & $M=8$ & $M=32$ & "
        r"$M=2$ & $M=8$ & $M=32$ \\",
        r"\midrule",
    ]
    for condition in conditions:
        cells = []
        for risk_field in RISKS:
            methods = condition["risks"][risk_field]["methods"]
            cells.extend(
                _number(methods[field]["aurc"])
                for field in MATCHED_METHODS[risk_field]
            )
        lines.append(
            " & ".join(
                [
                    _escape(_dataset(condition["dataset"])),
                    _escape(_condition(condition["condition"])),
                    *cells,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}}", r"}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def render_statistical_tests(conditions, *, header):
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{All predeclared paired comparisons. "
        r"$\Delta=\operatorname{AURC}(\mathrm{Dice\text{-}M32})-"
        r"\operatorname{AURC}(\mathrm{nHD95\text{-}M32})$; hence negative "
        r"values favor Dice-M32. CIs are paired image-cluster bootstrap "
        r"percentile intervals and are unadjusted. Each raw two-sided bootstrap "
        r"$p$-value uses the same paired resamples as its interval; Holm "
        r"adjustment covers all 20 comparisons. Lower $p$-values indicate "
        r"stronger evidence, but no result is filtered for significance.}",
        r"\label{tab:statistical-tests}",
        r"{\scriptsize\setlength{\tabcolsep}{4pt}%",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Dataset & Model condition & Evaluation risk & $\Delta$ & 95\% CI & "
        r"Raw $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for condition in conditions:
        for risk_field, risk_label in RISKS.items():
            comparison = condition["comparisons"][risk_field]
            bootstrap = comparison["bootstrap"]
            cells = [
                _escape(_dataset(condition["dataset"])),
                _escape(_condition(condition["condition"])),
                _escape(risk_label),
                _number(comparison["difference_left_minus_right"]),
                f"[{_number(bootstrap['ci_low'])}, {_number(bootstrap['ci_high'])}]",
                _number(bootstrap["p_value"]),
                _number(comparison["holm_adjusted_p_value"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}}", r"}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def render_tables(result, *, source_hash, allow_incomplete=False):
    conditions = validate_analysis(result, allow_incomplete=allow_incomplete)
    header = _generated_header(source_hash, incomplete=allow_incomplete)
    return {
        "main_results.tex": render_main_results(conditions, header=header),
        "cross_loss_results.tex": render_cross_loss_results(conditions, header=header),
        "quadrature_ablation.tex": render_quadrature_ablation(
            conditions, header=header
        ),
        "statistical_tests.tex": render_statistical_tests(conditions, header=header),
    }


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_tables(tables, output_dir):
    if set(tables) != set(OUTPUT_NAMES):
        raise ValueError(f"expected exactly the four table artifacts {OUTPUT_NAMES}")
    output_dir = Path(output_dir)
    for name in OUTPUT_NAMES:
        _atomic_write(output_dir / name, tables[name])
    return tuple(output_dir / name for name in OUTPUT_NAMES)


def main(argv=None):
    args = parse_args(argv)
    analysis_path = Path(args.analysis)
    result = load_analysis(analysis_path)
    source_hash = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    tables = render_tables(
        result,
        source_hash=source_hash,
        allow_incomplete=args.allow_incomplete,
    )
    paths = write_tables(tables, args.output_dir)
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
