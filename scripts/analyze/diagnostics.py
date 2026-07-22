"""Aggregate explicit held-out binary diagnostics under one campaign lock.

The primary run requires exactly its 16 predeclared ``diagnostics.json`` paths;
``--design extension`` selects the separately locked seven-condition
architecture/domain study. Both consume the immutable campaign lock that names
their source artifacts.
``--allow-incomplete`` permits a nonempty declared subset only for pipeline
smoke tests.  Inputs are never discovered from directories or globs.

This analysis is descriptive.  Brier score and fixed-bin ECE assess marginal
foreground calibration; M=32 ladder summaries assess candidate-mask diversity.
Neither identifies a joint mask posterior nor validates the shared-threshold
coupling.  Held-out labels are never used to fit, tune, or select a confidence
score, and per-image descriptors are not mined for favorable examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze.main import (  # noqa: E402
    EXPECTED_CONDITIONS,
    EXTENSION_CONDITIONS,
)
from scripts.submit.main import load_campaign_lock  # noqa: E402
from selectseg.studies.diagnostics import (  # noqa: E402
    DEFAULT_ECE_BINS,
    LADDER_M,
    BinaryDiagnostics,
    load_binary_diagnostics,
)


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_ARTIFACT_TYPE = "selectseg.diagnostics_analysis"
EXPECTED_CAMPAIGN_IDS = frozenset(
    {
        "binary-midpoint-main-v1",
        "binary-midpoint-main-v2",
    }
)
EXTENSION_CAMPAIGN_IDS = frozenset({"architecture-domain-extension-v1"})
JSON_NAME = "diagnostics_analysis.json"
TEX_NAME = "binary_diagnostics.tex"
EXPECTED_SAMPLE_COUNTS = {
    "pet": 3_669,
    "kvasir": 200,
    "fives": 200,
    "isic": 1_000,
    "tn3k": 614,
}
DATASET_ORDER = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
CONDITION_ORDER = (
    "clipseg-general",
    "clipseg-target",
    "deeplabv3-target",
    "deeplabv3-external",
)
CONDITION_LABELS = {
    "clipseg-general": "CLIP-G",
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
    "deeplabv3-external": "DL-E",
}


@dataclass(frozen=True)
class DiagnosticsDesign:
    expected_conditions: tuple[tuple[str, str], ...]
    campaign_ids: frozenset[str]
    sample_counts: Mapping[str, int]
    dataset_order: tuple[str, ...]
    dataset_labels: Mapping[str, str]
    condition_order: tuple[str, ...]
    condition_labels: Mapping[str, str]
    tex_name: str
    table_label: str
    caption_prefix: str


DESIGNS = {
    "primary": DiagnosticsDesign(
        expected_conditions=EXPECTED_CONDITIONS,
        campaign_ids=EXPECTED_CAMPAIGN_IDS,
        sample_counts=EXPECTED_SAMPLE_COUNTS,
        dataset_order=DATASET_ORDER,
        dataset_labels=DATASET_LABELS,
        condition_order=CONDITION_ORDER,
        condition_labels=CONDITION_LABELS,
        tex_name=TEX_NAME,
        table_label="tab:binary-diagnostics",
        caption_prefix="Fixed held-out diagnostics",
    ),
    "extension": DiagnosticsDesign(
        expected_conditions=EXTENSION_CONDITIONS,
        campaign_ids=EXTENSION_CAMPAIGN_IDS,
        sample_counts={**EXPECTED_SAMPLE_COUNTS, "duts": 5_019},
        dataset_order=(*DATASET_ORDER, "duts"),
        dataset_labels={**DATASET_LABELS, "duts": "DUTS"},
        condition_order=("segformer-target", "deeplabv3-target"),
        condition_labels={
            "segformer-target": "SF-T",
            "deeplabv3-target": "DL-T",
        },
        tex_name="architecture_domain_diagnostics.tex",
        table_label="tab:architecture-domain-diagnostics",
        caption_prefix="Architecture and domain extension diagnostics",
    ),
}


def _sample_counts(design):
    """Keep the primary compatibility constant patchable in contract tests."""

    return (
        EXPECTED_SAMPLE_COUNTS if design is DESIGNS["primary"] else design.sample_counts
    )


METRIC_KEYS = (
    "brier_score",
    "ece",
    "truth_empty_mask_ratio",
    "prediction_empty_mask_ratio",
    "m32_mean_distinct_mask_count",
    "m32_zero_change_pair_ratio",
    "m32_mean_adjacent_changed_pixel_fraction",
)
ANALYSIS_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "campaign",
        "scope",
        "aggregation",
        "conditions",
    }
)
CONDITION_KEYS = frozenset(
    {
        "dataset",
        "condition",
        "model",
        "split",
        "num_images",
        "num_pixels",
        "diagnostic_id",
        "diagnostics_path",
        "diagnostics_sha256",
        "source_artifact",
        "metrics",
    }
)


@dataclass(frozen=True)
class DiagnosticInput:
    loaded: BinaryDiagnostics
    lock_artifact: Mapping

    @property
    def summary(self):
        return self.loaded.summary

    @property
    def key(self):
        artifact = self.summary["artifact"]
        return artifact["dataset"], artifact["condition"]


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        choices=tuple(DESIGNS),
        default="primary",
        help="validate the primary campaign or architecture/domain extension",
    )
    parser.add_argument(
        "--campaign-lock",
        required=True,
        help="one explicit immutable campaign.lock.json",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="DIAGNOSTICS_JSON",
        help="explicit diagnostics.json files; directories/globs are unsupported",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/binary_final/diagnostics",
    )
    parser.add_argument(
        "--paper-table",
        help=(
            "optional second destination for the identical generated TeX table, "
            "for example docs/Tables/binary_diagnostics.tex"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow a nonempty declared subset for smoke testing only",
    )
    return parser.parse_args(argv)


def _portable_path(path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _require_exact_mapping(value, keys, *, location):
    if not isinstance(value, dict) or set(value) != set(keys):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{location} must contain exactly {sorted(keys)}; got {observed}"
        )
    return value


def _require_string(value, *, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _require_digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 digest")
    return value.lower()


def _require_hex(value, length, *, location):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a {length}-character hexadecimal ID")
    return value.lower()


def _require_integer(value, *, location, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _require_number(value, *, location, minimum=0.0, maximum=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{location} must be finite and lie in [{minimum}, {maximum}]")
    return result


def _validate_lock_scope(lock, *, allow_incomplete, design):
    artifacts = lock["artifacts"]
    keys = [(item["dataset"], item["condition"]) for item in artifacts]
    observed = set(keys)
    expected = set(design.expected_conditions)
    if not allow_incomplete and lock["campaign_id"] not in design.campaign_ids:
        raise ValueError(
            "canonical diagnostics require one of the sealed campaign IDs: "
            f"{sorted(design.campaign_ids)}"
        )
    if (
        lock["estimator"]["estimator_id"] != "midpoint-v1"
        or lock["estimator"]["target_measure"] != "uniform-threshold"
    ):
        raise ValueError("diagnostics lock names an unsupported estimator")
    if len(keys) != len(observed):
        raise ValueError("campaign lock contains duplicate dataset/condition entries")
    undeclared = sorted(observed - expected)
    if undeclared:
        raise ValueError(f"campaign lock contains undeclared conditions: {undeclared}")
    if allow_incomplete:
        if not observed:
            raise ValueError("smoke-test campaign lock cannot be empty")
    elif observed != expected or len(artifacts) != len(design.expected_conditions):
        raise ValueError(
            "canonical diagnostics require the exact "
            f"{len(design.expected_conditions)}-condition campaign lock; "
            f"missing={sorted(expected - observed)}"
        )
    if not allow_incomplete:
        sample_counts = _sample_counts(design)
        for artifact in artifacts:
            expected_count = sample_counts[artifact["dataset"]]
            if artifact["num_samples"] != expected_count:
                raise ValueError(
                    "canonical cohort count mismatch for "
                    f"{(artifact['dataset'], artifact['condition'])}: "
                    f"lock={artifact['num_samples']}, expected={expected_count}"
                )
    return {(item["dataset"], item["condition"]): item for item in artifacts}


def _validate_diagnostic_against_lock(loaded, lock_artifact, *, gamma):
    summary = loaded.summary
    artifact = summary["artifact"]
    key = artifact["dataset"], artifact["condition"]
    exact_fields = (
        "artifact_id",
        "dataset",
        "condition",
        "model",
        "split",
        "sample_id_sha256",
        "num_samples",
    )
    for field in exact_fields:
        if artifact[field] != lock_artifact[field]:
            raise ValueError(
                f"diagnostics/lock {field} mismatch for {key}: "
                f"diagnostics={artifact[field]!r}, lock={lock_artifact[field]!r}"
            )
    if artifact["manifest_sha256"].lower() != lock_artifact["manifest_sha256"].lower():
        raise ValueError(f"source artifact manifest SHA-256 mismatch for {key}")
    if (
        artifact["payload_source_sha256"].lower()
        != lock_artifact["source_sha256"].lower()
    ):
        raise ValueError(f"source artifact implementation SHA-256 mismatch for {key}")
    if summary["counts"]["num_images"] != lock_artifact["num_samples"]:
        raise ValueError(
            f"diagnostic image count does not match locked cohort for {key}"
        )

    specification = summary["specification"]
    if specification["decision_rule"]["gamma"] != gamma:
        raise ValueError(f"diagnostic decision threshold differs from lock for {key}")
    if specification["ece"]["num_equal_width_bins"] != DEFAULT_ECE_BINS:
        raise ValueError(
            f"canonical diagnostics require {DEFAULT_ECE_BINS}-bin ECE for {key}"
        )
    if specification["ladder"]["m"] != LADDER_M:
        raise ValueError(f"diagnostic ladder must use M={LADDER_M} for {key}")


def load_inputs(
    campaign_lock,
    diagnostic_paths,
    *,
    allow_incomplete=False,
    design=DESIGNS["primary"],
):
    """Load one lock and explicit, one-per-condition diagnostic summaries."""

    lock_path, lock_sha256, lock = load_campaign_lock(campaign_lock)
    lock_by_key = _validate_lock_scope(
        lock,
        allow_incomplete=allow_incomplete,
        design=design,
    )
    raw_paths = list(diagnostic_paths)
    if not raw_paths:
        raise ValueError("at least one explicit diagnostics.json is required")
    resolved = [Path(path).resolve() for path in raw_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("diagnostics.json inputs must be distinct")
    if not allow_incomplete and len(resolved) != len(design.expected_conditions):
        raise ValueError(
            "canonical diagnostics require exactly "
            f"{len(design.expected_conditions)} "
            f"explicit diagnostics.json inputs; got {len(resolved)}"
        )

    gamma_values = lock["protocol"]["gamma_values"]
    if len(gamma_values) != 1:
        raise ValueError("diagnostics require exactly one lock decision threshold")
    gamma = gamma_values[0]
    by_key = {}
    loaded_inputs = []
    specifications = []
    source_hashes = set()
    for path in resolved:
        loaded = load_binary_diagnostics(path, validate_descriptors=True)
        artifact = loaded.summary["artifact"]
        key = artifact["dataset"], artifact["condition"]
        if key not in lock_by_key:
            raise ValueError(
                f"diagnostics condition is absent from campaign lock: {key}"
            )
        if key in by_key:
            raise ValueError(f"duplicate diagnostics condition: {key}")
        _validate_diagnostic_against_lock(loaded, lock_by_key[key], gamma=gamma)
        record = DiagnosticInput(loaded=loaded, lock_artifact=lock_by_key[key])
        by_key[key] = record
        loaded_inputs.append(record)
        specifications.append(loaded.summary["specification"])
        source_hashes.add(loaded.summary["source_sha256"])

    if not allow_incomplete and set(by_key) != set(lock_by_key):
        raise ValueError(
            "canonical diagnostics do not cover the locked campaign; "
            f"missing={sorted(set(lock_by_key) - set(by_key))}"
        )
    if len(source_hashes) != 1:
        raise ValueError("all diagnostics must use one source fingerprint")
    # Chunk size affects identity but not the aggregate; all other predeclared
    # choices must agree.  Requiring the complete specification is stricter and
    # makes an accidental mixed diagnostics run fail closed.
    canonical_spec = json.dumps(
        specifications[0], sort_keys=True, separators=(",", ":")
    )
    if any(
        json.dumps(item, sort_keys=True, separators=(",", ":")) != canonical_spec
        for item in specifications[1:]
    ):
        raise ValueError("all diagnostics must use the exact same specification")

    loaded_inputs.sort(key=lambda item: (item.key[0], item.key[1]))
    return lock_path, lock_sha256, lock, tuple(loaded_inputs)


def _condition_result(item: DiagnosticInput):
    summary = item.summary
    artifact = summary["artifact"]
    adjacent = summary["shared_threshold_ladder"]["adjacent_changes"]
    metrics = {
        "brier_score": summary["marginal_calibration"]["brier_score"],
        "ece": summary["marginal_calibration"]["ece"],
        "truth_empty_mask_ratio": summary["truth"]["empty_mask_ratio"],
        "prediction_empty_mask_ratio": summary["hard_prediction"]["empty_mask_ratio"],
        "m32_mean_distinct_mask_count": summary["shared_threshold_ladder"][
            "distinct_mask_count"
        ]["mean"],
        "m32_zero_change_pair_ratio": adjacent["zero_change_pair_ratio"],
        "m32_mean_adjacent_changed_pixel_fraction": adjacent[
            "mean_changed_pixel_fraction"
        ],
    }
    return {
        "dataset": artifact["dataset"],
        "condition": artifact["condition"],
        "model": artifact["model"],
        "split": artifact["split"],
        "num_images": summary["counts"]["num_images"],
        "num_pixels": summary["counts"]["num_pixels"],
        "diagnostic_id": summary["diagnostic_id"],
        "diagnostics_path": _portable_path(item.loaded.summary_path),
        "diagnostics_sha256": item.loaded.summary_sha256,
        "source_artifact": {
            "artifact_id": artifact["artifact_id"],
            "manifest_sha256": artifact["manifest_sha256"],
            "sample_id_sha256": artifact["sample_id_sha256"],
            "num_samples": artifact["num_samples"],
        },
        "metrics": metrics,
    }


def analyze(
    campaign_lock,
    diagnostic_paths,
    *,
    allow_incomplete=False,
    design=DESIGNS["primary"],
):
    lock_path, lock_sha256, lock, inputs = load_inputs(
        campaign_lock,
        diagnostic_paths,
        allow_incomplete=allow_incomplete,
        design=design,
    )
    analyzed_keys = {item.key for item in inputs}
    sample_counts = _sample_counts(design)
    complete = analyzed_keys == set(design.expected_conditions) and all(
        item.lock_artifact["num_samples"]
        == sample_counts[item.lock_artifact["dataset"]]
        for item in inputs
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_ARTIFACT_TYPE,
        "campaign": {
            "campaign_id": lock["campaign_id"],
            "lock_path": _portable_path(lock_path),
            "lock_sha256": lock_sha256,
            "num_locked_conditions": len(lock["artifacts"]),
            "num_analyzed_conditions": len(inputs),
            "complete_predeclared_campaign": complete,
            "diagnostics_source_sha256": inputs[0].summary["source_sha256"],
        },
        "scope": {
            "purpose": "held-out descriptive diagnostics; not confidence fitting",
            "label_use": (
                "labels enter only predeclared aggregate Brier/ECE and truth-empty "
                "descriptions; they never fit, tune, or select a confidence score"
            ),
            "posterior_limitation": (
                "marginal calibration and M32 ladder diversity do not identify a "
                "joint mask posterior or validate shared-threshold coupling"
            ),
            "descriptor_policy": (
                "descriptor integrity is validated when present, but no per-image "
                "failure case is selected or reported"
            ),
        },
        "aggregation": {
            "unit": "one locked dataset-condition artifact",
            "pooling": "none across datasets or model conditions",
            "calibration_weighting": "pixel-weighted within each condition",
            "cohort_binding": (
                "source-manifest SHA-256 plus ordered sample-ID SHA-256 and "
                "locked image count"
            ),
            "ladder": f"uniform-midpoint M={LADDER_M}",
            "ece_bins": DEFAULT_ECE_BINS,
            "metric_keys": list(METRIC_KEYS),
        },
        "conditions": [_condition_result(item) for item in inputs],
    }
    validate_analysis(result, design=design)
    return result


def validate_analysis(result, *, design=DESIGNS["primary"]):
    """Validate the exact aggregate schema before JSON or TeX publication."""

    _require_exact_mapping(result, ANALYSIS_KEYS, location="analysis")
    if result["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("analysis.schema_version is unsupported")
    if result["artifact_type"] != ANALYSIS_ARTIFACT_TYPE:
        raise ValueError("analysis.artifact_type is unsupported")
    campaign = _require_exact_mapping(
        result["campaign"],
        {
            "campaign_id",
            "lock_path",
            "lock_sha256",
            "num_locked_conditions",
            "num_analyzed_conditions",
            "complete_predeclared_campaign",
            "diagnostics_source_sha256",
        },
        location="analysis.campaign",
    )
    _require_string(campaign["campaign_id"], location="campaign.campaign_id")
    _require_string(campaign["lock_path"], location="campaign.lock_path")
    _require_digest(campaign["lock_sha256"], location="campaign.lock_sha256")
    _require_digest(
        campaign["diagnostics_source_sha256"],
        location="campaign.diagnostics_source_sha256",
    )
    _require_integer(
        campaign["num_locked_conditions"],
        location="campaign.num_locked_conditions",
        minimum=1,
    )
    analyzed_count = _require_integer(
        campaign["num_analyzed_conditions"],
        location="campaign.num_analyzed_conditions",
        minimum=1,
    )
    if type(campaign["complete_predeclared_campaign"]) is not bool:
        raise ValueError("campaign.complete_predeclared_campaign must be boolean")
    _require_exact_mapping(
        result["scope"],
        {"purpose", "label_use", "posterior_limitation", "descriptor_policy"},
        location="analysis.scope",
    )
    for field, value in result["scope"].items():
        _require_string(value, location=f"analysis.scope.{field}")
    aggregation = _require_exact_mapping(
        result["aggregation"],
        {
            "unit",
            "pooling",
            "calibration_weighting",
            "cohort_binding",
            "ladder",
            "ece_bins",
            "metric_keys",
        },
        location="analysis.aggregation",
    )
    if aggregation["metric_keys"] != list(METRIC_KEYS):
        raise ValueError("analysis.aggregation.metric_keys is inconsistent")
    if aggregation["ece_bins"] != DEFAULT_ECE_BINS:
        raise ValueError("analysis.aggregation.ece_bins is inconsistent")
    for field in (
        "unit",
        "pooling",
        "calibration_weighting",
        "cohort_binding",
        "ladder",
    ):
        _require_string(aggregation[field], location=f"analysis.aggregation.{field}")
    if aggregation["ladder"] != f"uniform-midpoint M={LADDER_M}":
        raise ValueError("analysis.aggregation.ladder is inconsistent")
    conditions = result["conditions"]
    if not isinstance(conditions, list) or len(conditions) != analyzed_count:
        raise ValueError("analysis.conditions count is inconsistent")
    keys = []
    for index, condition in enumerate(conditions):
        location = f"analysis.conditions[{index}]"
        _require_exact_mapping(condition, CONDITION_KEYS, location=location)
        for field in (
            "dataset",
            "condition",
            "model",
            "split",
            "diagnostics_path",
        ):
            _require_string(condition[field], location=f"{location}.{field}")
        _require_hex(
            condition["diagnostic_id"], 16, location=f"{location}.diagnostic_id"
        )
        _require_digest(
            condition["diagnostics_sha256"],
            location=f"{location}.diagnostics_sha256",
        )
        _require_integer(
            condition["num_images"], location=f"{location}.num_images", minimum=1
        )
        _require_integer(
            condition["num_pixels"], location=f"{location}.num_pixels", minimum=1
        )
        source = _require_exact_mapping(
            condition["source_artifact"],
            {"artifact_id", "manifest_sha256", "sample_id_sha256", "num_samples"},
            location=f"{location}.source_artifact",
        )
        _require_hex(
            source["artifact_id"],
            16,
            location=f"{location}.source_artifact.artifact_id",
        )
        for field in ("manifest_sha256", "sample_id_sha256"):
            _require_digest(
                source[field], location=f"{location}.source_artifact.{field}"
            )
        _require_integer(
            source["num_samples"],
            location=f"{location}.source_artifact.num_samples",
            minimum=1,
        )
        if source["num_samples"] != condition["num_images"]:
            raise ValueError(f"{location} source/artifact image counts disagree")
        metrics = _require_exact_mapping(
            condition["metrics"], METRIC_KEYS, location=f"{location}.metrics"
        )
        for field in METRIC_KEYS:
            maximum = (
                float(LADDER_M) if field == "m32_mean_distinct_mask_count" else 1.0
            )
            minimum = 1.0 if field == "m32_mean_distinct_mask_count" else 0.0
            _require_number(
                metrics[field],
                location=f"{location}.metrics.{field}",
                minimum=minimum,
                maximum=maximum,
            )
        keys.append((condition["dataset"], condition["condition"]))
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise ValueError(
            "analysis.conditions must be unique and deterministically sorted"
        )
    undeclared = sorted(set(keys) - set(design.expected_conditions))
    if undeclared:
        raise ValueError(
            f"analysis.conditions contains undeclared entries: {undeclared}"
        )
    if campaign["complete_predeclared_campaign"]:
        if set(keys) != set(design.expected_conditions):
            raise ValueError("complete campaign must contain all declared conditions")
        sample_counts = _sample_counts(design)
        for condition in conditions:
            expected_count = sample_counts[condition["dataset"]]
            if condition["num_images"] != expected_count:
                raise ValueError("complete campaign has a noncanonical cohort count")
    return result


def _json_bytes(result, *, design) -> bytes:
    validate_analysis(result, design=design)
    return (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _latex_cell(entries, metric_key, *, design):
    if not entries:
        return "--"
    lines = []
    for condition in design.condition_order:
        item = entries.get(condition)
        if item is None:
            continue
        value = item["metrics"][metric_key]
        if metric_key in {
            "truth_empty_mask_ratio",
            "prediction_empty_mask_ratio",
            "m32_zero_change_pair_ratio",
            "m32_mean_adjacent_changed_pixel_fraction",
        }:
            formatted = f"{100 * value:.1f}\\%"
        elif metric_key == "m32_mean_distinct_mask_count":
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value:.4f}"
        lines.append(rf"\textsc{{{design.condition_labels[condition]}}}: {formatted}")
    return r"\shortstack[l]{" + r" \\ ".join(lines) + "}"


def render_latex(
    result,
    *,
    analysis_sha256,
    design=DESIGNS["primary"],
):
    validate_analysis(result, design=design)
    observed_datasets = {item["dataset"] for item in result["conditions"]}
    datasets = [
        dataset for dataset in design.dataset_order if dataset in observed_datasets
    ]
    by_dataset = {dataset: {} for dataset in datasets}
    for item in result["conditions"]:
        by_dataset[item["dataset"]][item["condition"]] = item
    metric_labels = {
        "brier_score": "Pixelwise Brier score",
        "ece": f"Fixed-bin ECE ({DEFAULT_ECE_BINS} bins)",
        "truth_empty_mask_ratio": "Truth-empty rate",
        "prediction_empty_mask_ratio": "Prediction-empty rate",
        "m32_mean_distinct_mask_count": "Mean distinct M32 masks",
        "m32_zero_change_pair_ratio": "Zero-change transition ratio",
        "m32_mean_adjacent_changed_pixel_fraction": (
            "Mean adjacent changed-pixel fraction"
        ),
    }
    column_spec = "l" + "c" * len(datasets)
    lines = [
        f"% Generated from diagnostics_analysis.json SHA-256: {analysis_sha256}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            rf"\caption{{{design.caption_prefix}, reported separately for "
            r"each dataset--condition with no cross-condition pooling. Brier score "
            r"and ECE diagnose marginal pixel probabilities; M32 ladder statistics "
            r"describe candidate-mask diversity. Neither identifies a joint mask "
            r"posterior nor validates shared-threshold coupling. Labels are used "
            r"only for the displayed descriptive aggregates, never confidence-score "
            r"fitting or sample selection.}"
        ),
        rf"\label{{{design.table_label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Diagnostic & "
        + " & ".join(design.dataset_labels[dataset] for dataset in datasets)
        + r" \\",
        r"\midrule",
    ]
    for metric_index, metric_key in enumerate(METRIC_KEYS):
        cells = [
            _latex_cell(by_dataset[dataset], metric_key, design=design)
            for dataset in datasets
        ]
        lines.append(metric_labels[metric_key] + " & " + " & ".join(cells) + r" \\")
        if metric_index + 1 < len(METRIC_KEYS):
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path, payload: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(
    result,
    output_dir,
    *,
    paper_table=None,
    design=DESIGNS["primary"],
):
    output_dir = Path(output_dir)
    json_payload = _json_bytes(result, design=design)
    json_sha256 = hashlib.sha256(json_payload).hexdigest()
    latex_payload = render_latex(
        result,
        analysis_sha256=json_sha256,
        design=design,
    ).encode()
    json_path = output_dir / JSON_NAME
    tex_path = output_dir / design.tex_name
    paper_path = None if paper_table is None else Path(paper_table)
    if paper_path is not None and paper_path.resolve() in {
        json_path.resolve(),
        tex_path.resolve(),
    }:
        raise ValueError("--paper-table must be distinct from analysis outputs")
    _atomic_write(json_path, json_payload)
    _atomic_write(tex_path, latex_payload)
    paths = [json_path, tex_path]
    if paper_path is not None:
        _atomic_write(paper_path, latex_payload)
        paths.append(paper_path)
    return tuple(paths)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    design = DESIGNS[args.design]
    result = analyze(
        args.campaign_lock,
        args.inputs,
        allow_incomplete=args.allow_incomplete,
        design=design,
    )
    outputs = write_outputs(
        result,
        args.output_dir,
        paper_table=args.paper_table,
        design=design,
    )
    for path in outputs:
        print(f"saved {path}")
    return outputs


if __name__ == "__main__":
    main()
