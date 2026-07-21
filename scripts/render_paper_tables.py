"""Render manuscript tables from a validated three-loss binary analysis.

The renderer is deliberately one way: every displayed statistic must already
be present in ``analysis.json``.  Final tables require the complete declared
benchmark, 10,000 paired bootstrap resamples, and 95% intervals.  Draft smoke
tests may use ``--allow-incomplete`` and are visibly marked as incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 2
EXPECTED_RESAMPLES = 10_000
EXPECTED_CONFIDENCE_LEVEL = 0.95

RISKS = {
    "risk_dice": "Dice risk",
    "risk_nhd": "Normalized penalized Hausdorff risk",
    "risk_nhd95": "Normalized penalized HD95 risk",
}
METHODS = {
    "confidence_sdc": "SDC",
    "confidence_mean_max_probability": "Mean max probability",
    "confidence_negative_entropy": "Negative entropy",
    "confidence_dice_exact": "Dice-Exact",
    "confidence_qfr_entropy": "QFR-Entropy",
    "confidence_plm10_entropy": "PLM-10/PLA-10-Entropy",
    "confidence_mmmc_entropy": "MMMC-Entropy",
    "confidence_foreground_entropy": "Foreground entropy",
    "confidence_dice_m2": "Dice-M2",
    "confidence_dice_m8": "Dice-M8",
    "confidence_dice_m32": "Dice-M32",
    "confidence_nhd_m2": "nHD-M2",
    "confidence_nhd_m8": "nHD-M8",
    "confidence_nhd_m32": "nHD-M32",
    "confidence_nhd95_m2": "nHD95-M2",
    "confidence_nhd95_m8": "nHD95-M8",
    "confidence_nhd95_m32": "nHD95-M32",
}
MAIN_METHODS_BY_RISK = {
    "risk_dice": (
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_sdc",
    ),
    "risk_nhd": (
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "confidence_sdc",
    ),
    "risk_nhd95": (
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "confidence_sdc",
    ),
}
LOSS_INDEXED_M32 = (
    "confidence_dice_m32",
    "confidence_nhd_m32",
    "confidence_nhd95_m32",
)
MATCHED_METHODS = {
    "risk_dice": (
        "confidence_dice_exact",
        "confidence_dice_m2",
        "confidence_dice_m8",
        "confidence_dice_m32",
    ),
    "risk_nhd": (
        "confidence_nhd_m2",
        "confidence_nhd_m8",
        "confidence_nhd_m32",
    ),
    "risk_nhd95": (
        "confidence_nhd95_m2",
        "confidence_nhd95_m8",
        "confidence_nhd95_m32",
    ),
}
DICE_QUADRATURE_METHODS = {
    "confidence_dice_m2": 2,
    "confidence_dice_m8": 8,
    "confidence_dice_m32": 32,
}
DICE_EXACT_REFERENCE = "confidence_dice_exact"


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    left: str
    right: str
    risk: str


CONTRASTS = (
    ContrastSpec(
        "dice_vs_nhd_under_dice",
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "risk_dice",
    ),
    ContrastSpec(
        "dice_vs_nhd_under_nhd",
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "risk_nhd",
    ),
    ContrastSpec(
        "nhd_vs_nhd95_under_nhd",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "risk_nhd",
    ),
    ContrastSpec(
        "nhd_vs_nhd95_under_nhd95",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "risk_nhd95",
    ),
)
CONTRAST_BY_NAME = {contrast.name: contrast for contrast in CONTRASTS}

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
    ("isic", "clipseg-general"),
    ("isic", "clipseg-target"),
    ("isic", "deeplabv3-target"),
    ("tn3k", "clipseg-general"),
    ("tn3k", "clipseg-target"),
    ("tn3k", "deeplabv3-target"),
)
TARGET_CONDITIONS = tuple(
    key
    for key in EXPECTED_CONDITIONS
    if key[1] in {"clipseg-target", "deeplabv3-target"}
)
CONTROL_CONDITIONS = tuple(
    key for key in EXPECTED_CONDITIONS if key not in TARGET_CONDITIONS
)
HOLM_FAMILY_BY_DATASET = {
    "pet": "core",
    "kvasir": "core",
    "fives": "core",
    "isic": "extension",
    "tn3k": "extension",
}
CORE_CONDITIONS = tuple(
    key for key in EXPECTED_CONDITIONS if HOLM_FAMILY_BY_DATASET[key[0]] == "core"
)
EXTENSION_CONDITIONS = tuple(
    key
    for key in EXPECTED_CONDITIONS
    if HOLM_FAMILY_BY_DATASET[key[0]] == "extension"
)
EXPECTED_HOLM_FAMILY_SIZES = {
    family: len(CONTRASTS)
    * sum(
        HOLM_FAMILY_BY_DATASET[dataset] == family for dataset, _ in EXPECTED_CONDITIONS
    )
    for family in ("core", "extension")
}
CONDITION_ORDER = {key: index for index, key in enumerate(EXPECTED_CONDITIONS)}
CONDITION_ABBREVIATIONS = {
    "clipseg-general": "CLIP-G",
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
    "deeplabv3-external": "DL-E",
}
CONDITION_PANEL_LABELS = {
    "clipseg-general": "CLIPSeg general (CLIP-G)",
    "clipseg-target": "CLIPSeg target (CLIP-T)",
    "deeplabv3-target": "DeepLabV3 target (DL-T)",
    "deeplabv3-external": "DeepLabV3 external (DL-E)",
}
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}

OUTPUT_NAMES = (
    "main_results.tex",
    "full_target_results.tex",
    "complete_results.tex",
    "cross_loss_results.tex",
    "quadrature_ablation.tex",
    "statistical_tests.tex",
)
COMPLETION_MARKER = "results_complete.tex"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="validated analysis.json")
    parser.add_argument("--output-dir", default="docs/Tables")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="render a declared nonempty subset as an INCOMPLETE draft",
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
        suffix = " or null" if nullable else ""
        raise ValueError(f"{location} must be a finite number{suffix}")
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


def _exact_keys(value, expected, location):
    if set(value) != set(expected):
        raise ValueError(
            f"{location} must contain exactly {sorted(expected)}; got {sorted(value)}"
        )


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


def _validate_numerical_validation(value, *, location, observations):
    value = _mapping(value, location)
    _exact_keys(
        value,
        {
            "reference",
            "absolute_error_definition",
            "rank_agreement_definition",
            "exact_match_definition",
            "dice_quadrature",
        },
        location,
    )
    if value["reference"] != DICE_EXACT_REFERENCE:
        raise ValueError(f"{location}.reference is unexpected")
    for field in (
        "absolute_error_definition",
        "rank_agreement_definition",
        "exact_match_definition",
    ):
        _string(value[field], f"{location}.{field}")
    methods = _mapping(value["dice_quadrature"], f"{location}.dice_quadrature")
    _exact_keys(methods, DICE_QUADRATURE_METHODS, f"{location}.dice_quadrature")
    for method, count in DICE_QUADRATURE_METHODS.items():
        method_location = f"{location}.dice_quadrature.{method}"
        summary = _mapping(methods[method], method_location)
        _exact_keys(
            summary,
            {
                "m",
                "num_images",
                "absolute_error",
                "rank_agreement",
                "exact_match_fraction",
            },
            method_location,
        )
        if _integer(summary["m"], f"{method_location}.m", positive=True) != count:
            raise ValueError(f"{method_location}.m is inconsistent")
        if (
            _integer(
                summary["num_images"],
                f"{method_location}.num_images",
                positive=True,
            )
            != observations
        ):
            raise ValueError(f"{method_location}.num_images is inconsistent")
        error_location = f"{method_location}.absolute_error"
        errors = _mapping(summary["absolute_error"], error_location)
        _exact_keys(errors, {"mean", "median", "p95", "max"}, error_location)
        error_values = {
            name: _finite(value, f"{error_location}.{name}")
            for name, value in errors.items()
        }
        if any(value < 0 for value in error_values.values()):
            raise ValueError(f"{error_location} values must be non-negative")
        if not (
            error_values["median"] <= error_values["p95"] <= error_values["max"]
            and error_values["mean"] <= error_values["max"]
        ):
            raise ValueError(f"{error_location} summaries are internally inconsistent")
        rank_location = f"{method_location}.rank_agreement"
        agreement = _mapping(summary["rank_agreement"], rank_location)
        _exact_keys(agreement, {"spearman_rho", "kendall_tau_b"}, rank_location)
        correlations = [
            _finite(agreement[field], f"{rank_location}.{field}", nullable=True)
            for field in ("spearman_rho", "kendall_tau_b")
        ]
        if any(value is not None and not -1 <= value <= 1 for value in correlations):
            raise ValueError(f"{rank_location} values must lie in [-1, 1] or be null")
        exact = _finite(
            summary["exact_match_fraction"],
            f"{method_location}.exact_match_fraction",
        )
        if not 0 <= exact <= 1:
            raise ValueError(
                f"{method_location}.exact_match_fraction must lie in [0, 1]"
            )


def _validate_method(method, location, *, expected_label, oracle, random):
    method = _mapping(method, location)
    _exact_keys(
        method,
        {
            "label",
            "aurc",
            "oracle_aurc",
            "random_aurc",
            "excess_aurc",
            "normalized_aurc",
        },
        location,
    )
    if method["label"] != expected_label:
        raise ValueError(f"{location}.label must be {expected_label!r}")
    aurc = _finite(method["aurc"], f"{location}.aurc")
    method_oracle = _finite(method["oracle_aurc"], f"{location}.oracle_aurc")
    method_random = _finite(method["random_aurc"], f"{location}.random_aurc")
    excess = _finite(method["excess_aurc"], f"{location}.excess_aurc")
    normalized = _finite(
        method["normalized_aurc"], f"{location}.normalized_aurc", nullable=True
    )
    if not all(0 <= value <= 1 for value in (aurc, method_oracle, method_random)):
        raise ValueError(f"{location} AURCs must lie in [0, 1]")
    if not _close(method_oracle, oracle) or not _close(method_random, random):
        raise ValueError(f"{location} disagrees with its risk-level baselines")
    if not _close(excess, aurc - oracle):
        raise ValueError(f"{location}.excess_aurc is inconsistent")
    denominator = random - oracle
    if abs(denominator) <= 1e-15:
        if normalized is not None:
            raise ValueError(f"{location}.normalized_aurc must be null")
    elif normalized is None or not _close(normalized, excess / denominator):
        raise ValueError(f"{location}.normalized_aurc is inconsistent")


def _sha256_string(value, location):
    value = _string(value, location)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{location} must be lowercase SHA-256")
    return value


def _validate_provenance(value, *, allow_incomplete):
    location = "provenance"
    value = _mapping(value, location)
    binding = _string(value.get("binding"), f"{location}.binding")
    if binding == "unbound":
        _exact_keys(value, {"binding", "analysis_source_sha256"}, location)
        if not allow_incomplete:
            raise ValueError("final tables require campaign-bound provenance")
        _sha256_string(
            value["analysis_source_sha256"],
            f"{location}.analysis_source_sha256",
        )
        return {}
    if binding != "campaign-lock":
        raise ValueError(f"{location}.binding is unsupported")
    _exact_keys(
        value,
        {
            "binding",
            "campaign_id",
            "campaign_lock",
            "config_sha256",
            "analysis_source_sha256",
            "inputs",
        },
        location,
    )
    _string(value["campaign_id"], f"{location}.campaign_id")
    _sha256_string(value["config_sha256"], f"{location}.config_sha256")
    _sha256_string(
        value["analysis_source_sha256"], f"{location}.analysis_source_sha256"
    )
    lock = _mapping(value["campaign_lock"], f"{location}.campaign_lock")
    _exact_keys(lock, {"logical_name", "sha256"}, f"{location}.campaign_lock")
    _string(lock["logical_name"], f"{location}.campaign_lock.logical_name")
    _sha256_string(lock["sha256"], f"{location}.campaign_lock.sha256")
    inputs = _sequence(value["inputs"], f"{location}.inputs")
    if not allow_incomplete and len(inputs) != len(EXPECTED_CONDITIONS):
        raise ValueError("final provenance must bind exactly 16 inputs")
    by_key = {}
    for index, item in enumerate(inputs):
        item_location = f"{location}.inputs[{index}]"
        item = _mapping(item, item_location)
        _exact_keys(
            item,
            {
                "logical_id",
                "dataset",
                "condition",
                "assembly_run_id",
                "assembly_source_sha256",
                "artifact_id",
                "manifest_sha256",
                "records_sha256",
                "sample_id_sha256",
                "num_samples",
            },
            item_location,
        )
        for field in (
            "logical_id",
            "dataset",
            "condition",
            "assembly_run_id",
            "artifact_id",
        ):
            _string(item[field], f"{item_location}.{field}")
        for field in (
            "assembly_source_sha256",
            "manifest_sha256",
            "records_sha256",
            "sample_id_sha256",
        ):
            _sha256_string(item[field], f"{item_location}.{field}")
        _integer(item["num_samples"], f"{item_location}.num_samples", positive=True)
        key = (item["dataset"], item["condition"])
        if key in by_key:
            raise ValueError(f"duplicate provenance input {key!r}")
        by_key[key] = item
    return by_key


def _validate_bootstrap(
    bootstrap,
    location,
    *,
    difference,
    samples,
    confidence_level,
    observations,
    clusters,
):
    bootstrap = _mapping(bootstrap, location)
    _exact_keys(
        bootstrap,
        {
            "difference",
            "ci_low",
            "ci_high",
            "confidence_level",
            "p_value",
            "n_resamples",
            "n_observations",
            "n_clusters",
            "seed",
        },
        location,
    )
    observed_difference = _finite(bootstrap["difference"], f"{location}.difference")
    low = _finite(bootstrap["ci_low"], f"{location}.ci_low")
    high = _finite(bootstrap["ci_high"], f"{location}.ci_high")
    level = _finite(bootstrap["confidence_level"], f"{location}.confidence_level")
    p_value = _finite(bootstrap["p_value"], f"{location}.p_value")
    if not _close(observed_difference, difference):
        raise ValueError(f"{location}.difference is inconsistent")
    if low > high or not 0 < level < 1 or not 0 <= p_value <= 1:
        raise ValueError(f"{location} interval or p-value is invalid")
    if (
        _integer(bootstrap["n_resamples"], f"{location}.n_resamples", positive=True)
        != samples
        or _integer(
            bootstrap["n_observations"],
            f"{location}.n_observations",
            positive=True,
        )
        != observations
        or _integer(bootstrap["n_clusters"], f"{location}.n_clusters", positive=True)
        != clusters
        or not _close(level, confidence_level)
    ):
        raise ValueError(f"{location} disagrees with analysis metadata")
    _integer(bootstrap["seed"], f"{location}.seed")


def _validate_comparison(
    comparison,
    condition,
    spec,
    location,
    *,
    samples,
    confidence_level,
    observations,
    clusters,
    family,
    family_size,
):
    comparison = _mapping(comparison, location)
    _exact_keys(
        comparison,
        {
            "name",
            "risk",
            "left",
            "right",
            "difference_left_minus_right",
            "bootstrap",
            "holm_family",
            "holm_family_size",
            "holm_adjusted_p_value",
        },
        location,
    )
    for field in ("name", "risk", "left", "right"):
        if comparison[field] != getattr(spec, field):
            raise ValueError(f"{location}.{field} disagrees with its declaration")
    methods = condition["risks"][spec.risk]["methods"]
    expected = methods[spec.left]["aurc"] - methods[spec.right]["aurc"]
    difference = _finite(
        comparison["difference_left_minus_right"],
        f"{location}.difference_left_minus_right",
    )
    if not _close(difference, expected):
        raise ValueError(f"{location} difference disagrees with the AURCs")
    if comparison["holm_family"] != family:
        raise ValueError(f"{location}.holm_family is inconsistent")
    if (
        _integer(
            comparison["holm_family_size"],
            f"{location}.holm_family_size",
            positive=True,
        )
        != family_size
    ):
        raise ValueError(f"{location}.holm_family_size is inconsistent")
    adjusted = _finite(
        comparison["holm_adjusted_p_value"],
        f"{location}.holm_adjusted_p_value",
    )
    if not 0 <= adjusted <= 1:
        raise ValueError(f"{location}.holm_adjusted_p_value must lie in [0, 1]")
    _validate_bootstrap(
        comparison["bootstrap"],
        f"{location}.bootstrap",
        difference=difference,
        samples=samples,
        confidence_level=confidence_level,
        observations=observations,
        clusters=clusters,
    )


def validate_analysis(result, *, allow_incomplete=False):
    """Validate and canonically order the exact analysis-v2 contract."""

    result = _mapping(result, "analysis root")
    version = result.get("schema_version")
    if version == 1:
        raise ValueError("analysis schema_version 1 is obsolete; expected 2")
    if version != SCHEMA_VERSION:
        raise ValueError(f"analysis schema_version must equal {SCHEMA_VERSION}")
    _exact_keys(
        result,
        {
            "schema_version",
            "provenance",
            "analysis",
            "conditions",
            "multiple_testing",
        },
        "analysis root",
    )
    provenance_by_key = _validate_provenance(
        result["provenance"], allow_incomplete=allow_incomplete
    )
    analysis = _mapping(result["analysis"], "analysis")
    _exact_keys(
        analysis,
        {
            "tie_policy",
            "normalized_aurc",
            "comparisons",
            "contrast_definitions",
            "cross_loss_policy",
            "bootstrap_samples",
            "bootstrap_workers",
            "confidence_level",
            "bootstrap_p_value",
            "seed",
        },
        "analysis",
    )
    for field in (
        "tie_policy",
        "normalized_aurc",
        "comparisons",
        "cross_loss_policy",
        "bootstrap_p_value",
    ):
        _string(analysis[field], f"analysis.{field}")
    declarations = _sequence(
        analysis["contrast_definitions"], "analysis.contrast_definitions"
    )
    if declarations != [
        {
            "name": spec.name,
            "left": spec.left,
            "right": spec.right,
            "risk": spec.risk,
        }
        for spec in CONTRASTS
    ]:
        raise ValueError("analysis.contrast_definitions are unexpected")
    samples = _integer(
        analysis["bootstrap_samples"], "analysis.bootstrap_samples", positive=True
    )
    _integer(analysis["bootstrap_workers"], "analysis.bootstrap_workers", positive=True)
    confidence_level = _finite(
        analysis["confidence_level"], "analysis.confidence_level"
    )
    _integer(analysis["seed"], "analysis.seed")
    if not 0 < confidence_level < 1:
        raise ValueError("analysis.confidence_level must lie in (0, 1)")
    if not allow_incomplete and (
        samples != EXPECTED_RESAMPLES
        or not _close(confidence_level, EXPECTED_CONFIDENCE_LEVEL)
    ):
        raise ValueError("final tables require 10,000 resamples and 95% intervals")

    conditions = _sequence(result["conditions"], "conditions")
    if allow_incomplete:
        if not 0 < len(conditions) <= len(EXPECTED_CONDITIONS):
            raise ValueError("incomplete analysis must contain 1--16 conditions")
    elif len(conditions) != len(EXPECTED_CONDITIONS):
        raise ValueError("complete analysis must contain exactly 16 conditions")
    observed_keys = []
    for index, condition in enumerate(conditions):
        location = f"conditions[{index}]"
        condition = _mapping(condition, location)
        required_condition_fields = {
            "condition",
            "dataset",
            "split",
            "num_rows",
            "num_image_clusters",
            "jsonl",
            "manifest",
            "jsonl_sha256",
            "manifest_sha256",
            "risks",
            "comparisons",
        }
        observed_condition_fields = set(condition)
        allowed_condition_fields = required_condition_fields | {"numerical_validation"}
        if not required_condition_fields <= observed_condition_fields or not (
            observed_condition_fields <= allowed_condition_fields
        ):
            raise ValueError(
                f"{location} must contain required fields "
                f"{sorted(required_condition_fields)} and only optional field "
                "'numerical_validation'"
            )
        dataset = _string(condition["dataset"], f"{location}.dataset")
        name = _string(condition["condition"], f"{location}.condition")
        key = (dataset, name)
        if key in observed_keys:
            raise ValueError(f"duplicate dataset/condition pair {key!r}")
        if key not in CONDITION_ORDER:
            raise ValueError(f"undeclared dataset/condition pair {key!r}")
        observed_keys.append(key)
        if condition["split"] != "test":
            raise ValueError(f"{location}.split must be 'test'")
        _integer(condition["num_rows"], f"{location}.num_rows", positive=True)
        _integer(
            condition["num_image_clusters"],
            f"{location}.num_image_clusters",
            positive=True,
        )
        if not allow_incomplete and "numerical_validation" not in condition:
            raise ValueError(f"{location}.numerical_validation is required")
        if "numerical_validation" in condition:
            _validate_numerical_validation(
                condition["numerical_validation"],
                location=f"{location}.numerical_validation",
                observations=condition["num_rows"],
            )
        _string(condition["jsonl"], f"{location}.jsonl")
        _string(condition["manifest"], f"{location}.manifest")
        digest = _sha256_string(condition["jsonl_sha256"], f"{location}.jsonl_sha256")
        manifest_digest = _sha256_string(
            condition["manifest_sha256"], f"{location}.manifest_sha256"
        )
        if provenance_by_key:
            if key not in provenance_by_key:
                raise ValueError(f"{location} is absent from campaign provenance")
            bound = provenance_by_key[key]
            if bound["records_sha256"] != digest:
                raise ValueError(f"{location}.jsonl_sha256 differs from provenance")
            if bound["manifest_sha256"] != manifest_digest:
                raise ValueError(f"{location}.manifest_sha256 differs from provenance")
            if bound["num_samples"] != condition["num_rows"]:
                raise ValueError(f"{location}.num_rows differs from provenance")

        risks = _mapping(condition["risks"], f"{location}.risks")
        _exact_keys(risks, RISKS, f"{location}.risks")
        for risk_field, risk_label in RISKS.items():
            risk_location = f"{location}.risks.{risk_field}"
            risk = _mapping(risks[risk_field], risk_location)
            _exact_keys(
                risk,
                {"label", "methods", "oracle_aurc", "random_aurc"},
                risk_location,
            )
            if risk["label"] != risk_label:
                raise ValueError(f"{risk_location}.label is unexpected")
            oracle = _finite(risk["oracle_aurc"], f"{risk_location}.oracle_aurc")
            random = _finite(risk["random_aurc"], f"{risk_location}.random_aurc")
            if not 0 <= oracle <= random <= 1:
                raise ValueError(f"{risk_location} oracle/random AURCs are invalid")
            methods = _mapping(risk["methods"], f"{risk_location}.methods")
            _exact_keys(methods, METHODS, f"{risk_location}.methods")
            for method_field, label in METHODS.items():
                _validate_method(
                    methods[method_field],
                    f"{risk_location}.methods.{method_field}",
                    expected_label=label,
                    oracle=oracle,
                    random=random,
                )

    expected_keys = set(EXPECTED_CONDITIONS)
    if not allow_incomplete and set(observed_keys) != expected_keys:
        raise ValueError("complete analysis is missing declared conditions")
    if provenance_by_key and set(provenance_by_key) != set(observed_keys):
        raise ValueError("campaign provenance and analysis conditions differ")
    numerical_presence = ["numerical_validation" in item for item in conditions]
    if any(numerical_presence) and not all(numerical_presence):
        raise ValueError(
            "numerical_validation must be present for every condition or none"
        )
    if not allow_incomplete and not all(numerical_presence):
        raise ValueError("final analysis requires numerical_validation everywhere")
    family_sizes = {}
    for dataset, _ in observed_keys:
        family = HOLM_FAMILY_BY_DATASET[dataset]
        family_sizes[family] = family_sizes.get(family, 0) + len(CONTRASTS)
    by_key = {
        (condition["dataset"], condition["condition"]): condition
        for condition in conditions
    }
    for key, condition in by_key.items():
        comparisons = _mapping(condition["comparisons"], f"condition {key}.comparisons")
        _exact_keys(comparisons, CONTRAST_BY_NAME, f"condition {key}.comparisons")
        for spec in CONTRASTS:
            _validate_comparison(
                comparisons[spec.name],
                condition,
                spec,
                f"condition {key}.comparisons.{spec.name}",
                samples=samples,
                confidence_level=confidence_level,
                observations=condition["num_rows"],
                clusters=condition["num_image_clusters"],
                family=HOLM_FAMILY_BY_DATASET[key[0]],
                family_size=family_sizes[HOLM_FAMILY_BY_DATASET[key[0]]],
            )

    multiple = _mapping(result["multiple_testing"], "multiple_testing")
    _exact_keys(
        multiple,
        {
            "procedure",
            "family_policy",
            "families",
            "total_hypotheses",
            "hypotheses",
            "confidence_intervals",
            "significance_calls",
        },
        "multiple_testing",
    )
    for field in ("procedure", "family_policy", "confidence_intervals"):
        _string(multiple[field], f"multiple_testing.{field}")
    if multiple["significance_calls"] != "not made by this analysis":
        raise ValueError("multiple_testing.significance_calls is unexpected")
    expected_total = len(CONTRASTS) * len(conditions)
    if (
        _integer(multiple["total_hypotheses"], "multiple_testing.total_hypotheses")
        != expected_total
    ):
        raise ValueError("multiple_testing hypothesis count is inconsistent")
    hypotheses = _sequence(multiple["hypotheses"], "multiple_testing.hypotheses")
    if len(hypotheses) != expected_total:
        raise ValueError("multiple_testing hypothesis array length is inconsistent")
    families = _mapping(multiple["families"], "multiple_testing.families")
    _exact_keys(families, family_sizes, "multiple_testing.families")

    family_arrays = {}
    for family, size in family_sizes.items():
        location = f"multiple_testing.families.{family}"
        family_result = _mapping(families[family], location)
        _exact_keys(
            family_result,
            {
                "definition",
                "num_hypotheses",
                "raw_bootstrap_p_values",
                "holm_adjusted_p_values",
            },
            location,
        )
        _string(family_result["definition"], f"{location}.definition")
        if (
            _integer(
                family_result["num_hypotheses"],
                f"{location}.num_hypotheses",
                positive=True,
            )
            != size
        ):
            raise ValueError(f"{location}.num_hypotheses is inconsistent")
        raw = [
            _finite(value, f"{location}.raw_bootstrap_p_values[{index}]")
            for index, value in enumerate(
                _sequence(
                    family_result["raw_bootstrap_p_values"],
                    f"{location}.raw_bootstrap_p_values",
                )
            )
        ]
        adjusted = [
            _finite(value, f"{location}.holm_adjusted_p_values[{index}]")
            for index, value in enumerate(
                _sequence(
                    family_result["holm_adjusted_p_values"],
                    f"{location}.holm_adjusted_p_values",
                )
            )
        ]
        if len(raw) != size or len(adjusted) != size:
            raise ValueError(f"{location} array lengths are inconsistent")
        if any(not 0 <= value <= 1 for value in raw + adjusted):
            raise ValueError(f"{location} values must lie in [0, 1]")
        if any(
            not _close(left, right) for left, right in zip(adjusted, _holm_adjust(raw))
        ):
            raise ValueError(f"{location} Holm values are inconsistent")
        family_arrays[family] = (raw, adjusted)

    hypothesis_by_key = {}
    family_hypotheses = {family: ([], []) for family in families}
    for index, hypothesis in enumerate(hypotheses):
        location = f"multiple_testing.hypotheses[{index}]"
        hypothesis = _mapping(hypothesis, location)
        _exact_keys(
            hypothesis,
            {
                "dataset",
                "condition",
                "contrast",
                "risk",
                "left",
                "right",
                "holm_family",
                "holm_family_size",
                "raw_bootstrap_p_value",
                "holm_adjusted_p_value",
            },
            location,
        )
        key = (hypothesis["dataset"], hypothesis["condition"], hypothesis["contrast"])
        if key in hypothesis_by_key:
            raise ValueError(f"duplicate multiple-testing hypothesis {key!r}")
        condition_key = key[:2]
        if condition_key not in by_key or key[2] not in CONTRAST_BY_NAME:
            raise ValueError(f"{location} identifies an undeclared hypothesis")
        spec = CONTRAST_BY_NAME[key[2]]
        for field in ("risk", "left", "right"):
            if hypothesis[field] != getattr(spec, field):
                raise ValueError(f"{location}.{field} is inconsistent")
        family = HOLM_FAMILY_BY_DATASET[key[0]]
        if hypothesis["holm_family"] != family:
            raise ValueError(f"{location}.holm_family is inconsistent")
        if (
            _integer(
                hypothesis["holm_family_size"],
                f"{location}.holm_family_size",
                positive=True,
            )
            != family_sizes[family]
        ):
            raise ValueError(f"{location}.holm_family_size is inconsistent")
        raw = _finite(
            hypothesis["raw_bootstrap_p_value"], f"{location}.raw_bootstrap_p_value"
        )
        adjusted = _finite(
            hypothesis["holm_adjusted_p_value"], f"{location}.holm_adjusted_p_value"
        )
        if not 0 <= raw <= 1 or not 0 <= adjusted <= 1:
            raise ValueError(f"{location} p-values must lie in [0, 1]")
        hypothesis_by_key[key] = hypothesis
        family_hypotheses[family][0].append(raw)
        family_hypotheses[family][1].append(adjusted)

    for family, arrays in family_arrays.items():
        for source, observed in zip(arrays, family_hypotheses[family]):
            if len(source) != len(observed) or any(
                not _close(left, right) for left, right in zip(source, observed)
            ):
                raise ValueError(
                    f"multiple_testing family {family} arrays disagree with hypotheses"
                )
    for condition_key, condition in by_key.items():
        for spec in CONTRASTS:
            key = (*condition_key, spec.name)
            if key not in hypothesis_by_key:
                raise ValueError(f"missing multiple-testing hypothesis {key!r}")
            hypothesis = hypothesis_by_key[key]
            comparison = condition["comparisons"][spec.name]
            if not _close(
                hypothesis["raw_bootstrap_p_value"], comparison["bootstrap"]["p_value"]
            ):
                raise ValueError(f"hypothesis {key!r} has inconsistent raw p-value")
            if not _close(
                hypothesis["holm_adjusted_p_value"], comparison["holm_adjusted_p_value"]
            ):
                raise ValueError(f"hypothesis {key!r} has inconsistent Holm p-value")

    if not allow_incomplete and family_sizes != EXPECTED_HOLM_FAMILY_SIZES:
        raise ValueError(
            "final tables require separate 40-test core and 24-test extension families"
        )
    return sorted(
        conditions,
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


def _number(value):
    return "--" if value is None else f"{value:.4f}"


def _aurc_number(value):
    """Render raw AURC quantities on the manuscript's x100 display scale."""

    return "--" if value is None else f"{100 * value:.3f}"


def _best_result(value):
    return rf"\bestresult{{{value}}}"


def _is_best(methods, method_field, candidates):
    best = min(methods[field]["aurc"] for field in candidates)
    return methods[method_field]["aurc"] == best


def _groups(conditions, declared):
    by_key = {
        (condition["dataset"], condition["condition"]): condition
        for condition in conditions
    }
    groups = []
    for dataset in DATASET_LABELS:
        selected = [
            by_key[(candidate_dataset, name)]
            for candidate_dataset, name in declared
            if candidate_dataset == dataset and (candidate_dataset, name) in by_key
        ]
        if selected:
            groups.append((dataset, selected))
    return groups


def _condition_panels(conditions, declared):
    """Group declared results into model-condition panels with dataset columns."""

    by_key = {
        (condition["dataset"], condition["condition"]): condition
        for condition in conditions
    }
    declared = tuple(declared)
    declared_set = set(declared)
    condition_order = tuple(dict.fromkeys(name for _, name in declared))
    panels = []
    for condition_name in condition_order:
        groups = tuple(
            (dataset, by_key[(dataset, condition_name)])
            for dataset in DATASET_LABELS
            if (dataset, condition_name) in declared_set
            and (dataset, condition_name) in by_key
        )
        if groups:
            panels.append((condition_name, groups))
    return tuple(panels)


def _tabular_start(groups, *, row_label):
    return [
        rf"\begin{{tabular}}{{l*{{{len(groups)}}}{{c}}}}",
        r"\toprule",
        " & ".join([row_label, *(DATASET_LABELS[dataset] for dataset, _ in groups)])
        + r" \\",
        r"\midrule",
    ]


def _condition_panel_start(condition_name, groups, *, row_label):
    width = r"\textwidth" if len(groups) > 1 else r"0.62\textwidth"
    columns = len(groups)
    return [
        rf"\begin{{tabular*}}{{{width}}}"
        rf"{{@{{\extracolsep{{\fill}}}}l*{{{columns}}}{{c}}@{{}}}}",
        r"\toprule",
        rf"\multicolumn{{{1 + columns}}}{{l}}"
        rf"{{\textit{{{CONDITION_PANEL_LABELS[condition_name]}}}}}" + " \\\\",
        r"\midrule",
        " & ".join(
            [row_label, *(DATASET_LABELS[dataset] for dataset, _ in groups)]
        )
        + " \\\\",
        r"\midrule",
    ]


def _condition_panel_end():
    return [r"\bottomrule", r"\end{tabular*}"]


def _result_number(
    condition,
    risk_field,
    method_field,
    candidates,
    *,
    normalized,
    highlight_best=True,
):
    methods = condition["risks"][risk_field]["methods"]
    method = methods[method_field]
    number = _aurc_number(method["aurc"])
    if normalized:
        number += rf" ({_number(method['normalized_aurc'])})"
    if highlight_best and _is_best(methods, method_field, candidates):
        number = _best_result(number)
    return number


def _result_cell(
    group,
    risk_field,
    method_field,
    candidates,
    *,
    normalized,
    highlight_best=True,
):
    entries = []
    for condition in group:
        number = _result_number(
            condition,
            risk_field,
            method_field,
            candidates,
            normalized=normalized,
            highlight_best=highlight_best,
        )
        entries.append(f"{CONDITION_ABBREVIATIONS[condition['condition']]}: {number}")
    return r"\shortstack{" + r"\\".join(entries) + "}"


def _dice_numerical_pair(group, method_field, getter, *, percentage=False):
    values = []
    for condition in group:
        summary = condition["numerical_validation"]["dice_quadrature"][method_field]
        value = getter(summary)
        rendered = (
            "--"
            if value is None
            else (f"{100 * value:.1f}\\%" if percentage else _number(value))
        )
        if len(group) == 1:
            abbreviation = {"clipseg-target": "C", "deeplabv3-target": "D"}[
                condition["condition"]
            ]
            rendered = f"{abbreviation}: {rendered}"
        values.append(rendered)
    return "/".join(values)


def _generated_header(source_hash, *, incomplete):
    status = "INCOMPLETE DRAFT SMOKE TEST" if incomplete else "COMPLETE FINAL ANALYSIS"
    return (
        "% AUTO-GENERATED by scripts/render_paper_tables.py; DO NOT EDIT.\n"
        f"% Source analysis.json SHA-256: {source_hash}\n"
        f"% Status: {status}\n"
    )


def _table_end(*, resize=True):
    lines = [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines.extend([r"}", r"}"])
    lines.extend([r"\end{table*}", ""])
    return lines


def _baseline_table(conditions, *, header):
    groups = _groups(conditions, TARGET_CONDITIONS)
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary target-adapted results. Methods are rows and datasets "
        r"are columns; each cell gives CLIP-T and DL-T raw AURC $\times100$. The table follows "
        r"the two adjacent geometric steps and includes SDC as the Dice-specific "
        r"reference; the complete 17-score panels are in the appendix. Lower is "
        r"better. Dark blue marks every exactly lowest unrounded AURC among the "
        r"methods shown within each condition and risk block.}",
        r"\label{tab:main-results}",
        r"{\scriptsize\setlength{\tabcolsep}{3pt}%",
        r"\resizebox{\textwidth}{!}{%",
        *_tabular_start(groups, row_label="Confidence method"),
    ]
    for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
        if risk_index:
            lines.append(r"\midrule")
        candidates = MAIN_METHODS_BY_RISK[risk_field]
        lines.append(
            rf"\multicolumn{{{1 + len(groups)}}}{{l}}{{\textit{{{risk_label}}}}} \\"
        )
        for method_field in candidates:
            cells = [
                _result_cell(
                    group, risk_field, method_field, candidates, normalized=False
                )
                for _, group in groups
            ]
            lines.append(" & ".join([METHODS[method_field], *cells]) + r" \\")
    lines.extend(_table_end())
    return lines


def _contrast_label(spec):
    return f"{METHODS[spec.left]} $-$ {METHODS[spec.right]} / {RISKS[spec.risk]}"


def _comparison_cell(group, spec):
    entries = []
    for condition in group:
        comparison = condition["comparisons"][spec.name]
        bootstrap = comparison["bootstrap"]
        entries.append(
            rf"{CONDITION_ABBREVIATIONS[condition['condition']]}: "
            rf"{_aurc_number(comparison['difference_left_minus_right'])} "
            rf"[{_aurc_number(bootstrap['ci_low'])}, {_aurc_number(bootstrap['ci_high'])}]"
        )
    return r"\shortstack{" + r"\\".join(entries) + "}"


def _contrast_table(conditions, *, header, declared, label, caption):
    groups = _groups(conditions, declared)
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        caption,
        rf"\label{{{label}}}",
        r"{\scriptsize\setlength{\tabcolsep}{2pt}%",
        r"\resizebox{\textwidth}{!}{%",
        *_tabular_start(groups, row_label="Adjacent-geometry contrast"),
    ]
    for spec in CONTRASTS:
        cells = [_comparison_cell(group, spec) for _, group in groups]
        lines.append(" & ".join([_contrast_label(spec), *cells]) + r" \\")
    lines.extend(_table_end())
    return lines


def render_main_results(conditions, *, header):
    return "\n".join(_baseline_table(conditions, header=header))


def _complete_results_table(conditions, *, header, declared, label, caption_prefix):
    panels = _condition_panels(conditions, declared)
    lines = [header.rstrip()]
    candidates = tuple(METHODS)
    for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
        panel_label = label if risk_index == 0 else f"{label}-{risk_field[5:]}"
        lines.extend(
            [
                r"\begin{table*}[t]",
                r"\centering",
                rf"\caption{{{caption_prefix} {risk_label}. Methods are rows and "
                r"datasets are columns; each entry is raw AURC $\times100$ (nAURC), with "
                r"conditions identified by the stacked panel headings. Lower is better. "
                r"Dark blue marks every "
                r"lowest unrounded raw AURC within each condition.}",
                rf"\label{{{panel_label}}}",
                r"{\scriptsize\setlength{\tabcolsep}{2pt}%",
            ]
        )
        for panel_index, (condition_name, groups) in enumerate(panels):
            if panel_index:
                lines.append(r"\par\smallskip")
            lines.extend(
                _condition_panel_start(
                    condition_name, groups, row_label="Confidence method"
                )
            )
            for method_field in candidates:
                cells = [
                    _result_number(
                        condition,
                        risk_field,
                        method_field,
                        candidates,
                        normalized=True,
                    )
                    for _, condition in groups
                ]
                lines.append(" & ".join([METHODS[method_field], *cells]) + r" \\")
            lines.extend(_condition_panel_end())
        lines.extend([r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_full_target_results(conditions, *, header):
    return _complete_results_table(
        conditions,
        header=header,
        declared=TARGET_CONDITIONS,
        label="tab:full-target-results",
        caption_prefix=r"Complete target-adapted $17\times3$ results, panel for",
    )


def render_complete_results(conditions, *, header):
    return _complete_results_table(
        conditions,
        header=header,
        declared=CONTROL_CONDITIONS,
        label="tab:complete-results",
        caption_prefix=(
            r"Complete $17\times3$ general and external control results, panel for"
        ),
    )


def render_cross_loss_results(conditions, *, header):
    panels = _condition_panels(conditions, EXPECTED_CONDITIONS)
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full $3\times3$ loss-indexed $M=32$ cross-loss matrix over all "
        r"16 unpooled conditions. Confidence methods are rows and evaluation risks "
        r"form blocks; datasets are columns and conditions are identified by the "
        r"stacked panel headings. Each entry is raw AURC $\times100$ (nAURC). "
        r"Lower is better. Dark blue marks every lowest "
        r"unrounded raw AURC among the three indexed scores within each condition "
        r"and risk block. These cells are descriptive; paired inference is restricted "
        r"to the four fixed adjacent-geometry contrasts.}",
        r"\label{tab:cross-loss-results}",
        r"{\scriptsize\setlength{\tabcolsep}{2pt}%",
    ]
    for panel_index, (condition_name, groups) in enumerate(panels):
        if panel_index:
            lines.append(r"\par\smallskip")
        lines.extend(
            _condition_panel_start(
                condition_name, groups, row_label="Loss-indexed confidence"
            )
        )
        for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
            if risk_index:
                lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{{1 + len(groups)}}}{{l}}{{\textit{{{risk_label}}}}} \\"
            )
            for method_field in LOSS_INDEXED_M32:
                cells = [
                    _result_number(
                        condition,
                        risk_field,
                        method_field,
                        LOSS_INDEXED_M32,
                        normalized=True,
                    )
                    for _, condition in groups
                ]
                lines.append(" & ".join([METHODS[method_field], *cells]) + r" \\")
        lines.extend(_condition_panel_end())
    lines.extend([r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_quadrature_ablation(conditions, *, header):
    panels = _condition_panels(conditions, TARGET_CONDITIONS)
    fidelity_groups = _groups(conditions, TARGET_CONDITIONS)
    lines = [
        header.rstrip(),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-loss quadrature ablation on target-adapted conditions. "
        r"Dice-Exact is the exact level-set oracle; $M=2,8,32$ are midpoint rules. "
        r"Methods are rows, datasets are columns, and CLIP-T and DL-T are shown as "
        r"stacked condition panels. Cells report raw AURC $\times100$. Lower AURC "
        r"is better as a ranking outcome, but a "
        r"coarser rule can win accidentally; numerical fidelity is judged against "
        r"Dice-Exact below and against the high-resolution boundary reference in "
        r"the auxiliary study. AURC need not improve monotonically with $M$. "
        r"Dark blue marks every exactly lowest unrounded AURC within each condition "
        r"and risk block.}",
        r"\label{tab:quadrature-ablation}",
        r"{\scriptsize\setlength{\tabcolsep}{3pt}%",
    ]
    for panel_index, (condition_name, groups) in enumerate(panels):
        if panel_index:
            lines.append(r"\par\smallskip")
        lines.extend(
            _condition_panel_start(condition_name, groups, row_label="Estimator")
        )
        for risk_index, (risk_field, risk_label) in enumerate(RISKS.items()):
            if risk_index:
                lines.append(r"\midrule")
            candidates = MATCHED_METHODS[risk_field]
            lines.append(
                rf"\multicolumn{{{1 + len(groups)}}}{{l}}{{\textit{{{risk_label}}}}} \\"
            )
            for method_field in candidates:
                cells = [
                    _result_number(
                        condition,
                        risk_field,
                        method_field,
                        candidates,
                        normalized=False,
                    )
                    for _, condition in groups
                ]
                lines.append(" & ".join([METHODS[method_field], *cells]) + r" \\")
        lines.extend(_condition_panel_end())
    lines.extend([r"}", r"\end{table*}", ""])
    if all("numerical_validation" in condition for condition in conditions):
        lines.extend(
            [
                "",
                r"\clearpage",
                header.rstrip(),
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Per-image Dice midpoint fidelity to Dice-Exact on "
                r"target-adapted conditions. For each statistic, cells give "
                r"CLIP-T/DL-T in that order. Error rows summarize "
                r"$|C_{\mathrm{Dice},M}-C_{\mathrm{Dice},\mathrm{Exact}}|$; "
                r"rank rows give Spearman $\rho$ and Kendall $\tau_b$. Exact "
                r"floating-point equality is omitted because continuous knot and "
                r"midpoint sums are not expected to coincide bit-for-bit. These "
                r"diagnostics are descriptive "
                r"and do not enter the four fixed contrasts.}",
                r"\label{tab:dice-quadrature-fidelity}",
                r"{\scriptsize\setlength{\tabcolsep}{2pt}%",
                r"\resizebox{\textwidth}{!}{%",
                *_tabular_start(fidelity_groups, row_label="Estimator / statistic"),
            ]
        )
        metrics = (
            (
                "Mean abs. error",
                lambda summary: summary["absolute_error"]["mean"],
                False,
            ),
            (
                "Median abs. error",
                lambda summary: summary["absolute_error"]["median"],
                False,
            ),
            ("P95 abs. error", lambda summary: summary["absolute_error"]["p95"], False),
            ("Max abs. error", lambda summary: summary["absolute_error"]["max"], False),
            (
                "Spearman $\\rho$",
                lambda summary: summary["rank_agreement"]["spearman_rho"],
                False,
            ),
            (
                "Kendall $\\tau_b$",
                lambda summary: summary["rank_agreement"]["kendall_tau_b"],
                False,
            ),
        )
        for method_index, method_field in enumerate(DICE_QUADRATURE_METHODS):
            if method_index:
                lines.append(r"\midrule")
            for statistic, getter, percentage in metrics:
                cells = [
                    _dice_numerical_pair(
                        group,
                        method_field,
                        getter,
                        percentage=percentage,
                    )
                    for _, group in fidelity_groups
                ]
                row = f"{METHODS[method_field]} / {statistic}"
                lines.append(" & ".join([row, *cells]) + r" \\")
        lines.extend(_table_end())
    return "\n".join(lines)


def render_statistical_tests(conditions, *, header):
    core_caption = (
        r"\caption{Pet/Kvasir/FIVES subset of the 64 fixed condition-level "
        r"adjacent-geometry contrasts (40 comparisons). "
        r"Each entry is $100\Delta$, where $\Delta=\operatorname{AURC}(\mathrm{left})-"
        r"\operatorname{AURC}(\mathrm{right})$, and its pointwise 95\% paired "
        r"image-bootstrap interval. Negative values favor the left score. Every "
        r"fixed row is reported without filtering; the intervals condition on one "
        r"fitted checkpoint and are descriptive rather than simultaneous tests.}"
    )
    extension_caption = (
        r"\caption{ISIC/TN3K subset of the 64 fixed condition-level "
        r"adjacent-geometry contrasts (24 comparisons). Entries follow "
        r"Table~\ref{tab:statistical-tests} and use the same pointwise paired "
        r"image-bootstrap convention.}"
    )
    core = _contrast_table(
        conditions,
        header=header,
        declared=CORE_CONDITIONS,
        label="tab:statistical-tests",
        caption=core_caption,
    )
    extension = _contrast_table(
        conditions,
        header=header,
        declared=EXTENSION_CONDITIONS,
        label="tab:statistical-tests-extension",
        caption=extension_caption,
    )
    return "\n\n".join(("\n".join(core), "\n".join(extension)))


def render_tables(result, *, source_hash, allow_incomplete=False):
    conditions = validate_analysis(result, allow_incomplete=allow_incomplete)
    header = _generated_header(source_hash, incomplete=allow_incomplete)
    return {
        "main_results.tex": render_main_results(conditions, header=header),
        "full_target_results.tex": render_full_target_results(
            conditions, header=header
        ),
        "complete_results.tex": render_complete_results(conditions, header=header),
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


def write_tables(tables, output_dir, *, source_hash):
    if set(tables) != set(OUTPUT_NAMES):
        raise ValueError(f"expected exactly the table artifacts {OUTPUT_NAMES}")
    output_dir = Path(output_dir)
    marker = output_dir / COMPLETION_MARKER
    marker.unlink(missing_ok=True)
    for name in OUTPUT_NAMES:
        _atomic_write(output_dir / name, tables[name])
    expected_header = f"% Source analysis.json SHA-256: {source_hash}\n"
    for name in OUTPUT_NAMES:
        if expected_header not in (output_dir / name).read_text():
            raise RuntimeError(f"generated table {name} has a different source hash")
    _atomic_write(
        marker,
        "% AUTO-GENERATED completion sentinel; DO NOT EDIT.\n"
        f"% Source analysis.json SHA-256: {source_hash}\n"
        rf"\def\FinalResultsAnalysisSHA{{{source_hash}}}" + "\n",
    )
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
    for path in write_tables(tables, args.output_dir, source_hash=source_hash):
        print(f"saved {path}")
    print(f"saved {Path(args.output_dir) / COMPLETION_MARKER}")


if __name__ == "__main__":
    main()
