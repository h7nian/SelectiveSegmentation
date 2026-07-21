"""Render the complete cardinality/PIT analysis as a compact appendix table."""

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
from scripts.analyze.cardinality import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    SCOPE,
    TARGET_CONDITIONS,
)
from selectseg.artifacts import fsync_directory
from selectseg.studies.cardinality import PROTOCOL


OUTPUT_NAME = "cardinality_diagnostics.tex"
DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
TARGET_MODELS = ("clipseg-target", "deeplabv3-target")
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_id",
        "scope",
        "protocol",
        "condition_sets",
        "provenance",
        "conditions",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/binary_cardinality_diagnostics_analysis/rendered",
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


def _finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, location=f"{location}[{index}]")


def load_analysis(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"cardinality analysis does not exist: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("cardinality analysis must contain one JSON object")
    _finite_tree(value, location=str(source))
    return value, hashlib.sha256(raw).hexdigest()


def _number(value: Any, *, location: str, minimum=-math.inf, maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{location} is outside [{minimum}, {maximum}]")
    return value


def validate_analysis(value: Any) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise ValueError("cardinality analysis has an invalid top-level schema")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_type"] != ARTIFACT_TYPE
    ):
        raise ValueError("cardinality analysis has an unsupported type/schema")
    if (
        not isinstance(value["analysis_id"], str)
        or len(value["analysis_id"]) != 16
    ):
        raise ValueError("cardinality analysis_id must have 16 characters")
    if value["scope"] != SCOPE or value["protocol"] != PROTOCOL:
        raise ValueError("cardinality analysis scope/protocol is not frozen v1")
    condition_sets = value["condition_sets"]
    expected_all = [f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS]
    expected_target = [
        f"{dataset}/{condition}"
        for dataset, condition in EXPECTED_CONDITIONS
        if (dataset, condition) in TARGET_CONDITIONS
    ]
    if condition_sets != {
        "complete": True,
        "num_conditions": 16,
        "num_target_conditions": 10,
        "conditions": expected_all,
        "target_conditions": expected_target,
    }:
        raise ValueError("renderer requires the complete 16-condition analysis")
    conditions = value["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 16:
        raise ValueError("cardinality analysis must contain 16 condition rows")
    by_key = {}
    for index, row in enumerate(conditions):
        if not isinstance(row, dict):
            raise ValueError(f"conditions[{index}] must be an object")
        key = row.get("dataset"), row.get("condition")
        if key not in EXPECTED_CONDITIONS or key in by_key:
            raise ValueError(f"conditions[{index}] has an unknown/duplicate key")
        if row.get("is_target_condition") is not (key in TARGET_CONDITIONS):
            raise ValueError(f"conditions[{index}] has a wrong target flag")
        fraction = row.get("foreground_fraction_error")
        empty = row.get("empty_mask_identity")
        pit = row.get("randomized_cardinality_pit")
        for name, summary in (("fraction", fraction), ("empty", empty)):
            if not isinstance(summary, dict):
                raise ValueError(f"conditions[{index}].{name} is malformed")
            for field in (
                "signed_bias_predicted_minus_observed",
                "mean_absolute_error",
            ):
                _number(summary.get(field), location=f"conditions[{index}].{name}.{field}")
        if (
            not isinstance(pit, dict)
            or pit.get("diagnostic_seed") != PROTOCOL["diagnostic_seed"]
            or "not a pointwise posterior-calibration estimate"
            not in pit.get("interpretation", "")
        ):
            raise ValueError(f"conditions[{index}].randomized PIT is malformed")
        for field in (
            "kolmogorov_smirnov_distance_to_uniform",
            "outside_central_90_percent_ratio",
            "zero_observed_point_mass_ratio",
        ):
            _number(
                pit.get(field),
                location=f"conditions[{index}].pit.{field}",
                minimum=0.0,
                maximum=1.0,
            )
        by_key[key] = row
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("cardinality analysis condition coverage is incomplete")
    return by_key


def _format_signed_pp(value: float) -> str:
    scaled = 100.0 * value
    if abs(scaled) < 0.005:
        scaled = 0.0
    return f"{scaled:+.2f}"


def _format_pp(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _paired_cells(by_key, getter) -> list[str]:
    return [
        " / ".join(getter(by_key[(dataset, condition)]) for condition in TARGET_MODELS)
        for dataset in DATASETS
    ]


def render_analysis(value: Mapping[str, Any], *, source_sha256: str) -> str:
    by_key = validate_analysis(value)
    rows = []
    metric_rows = (
        (
            r"Foreground-mass bias, p.p.",
            lambda row: _format_signed_pp(
                row["foreground_fraction_error"][
                    "signed_bias_predicted_minus_observed"
                ]
            ),
        ),
        (
            r"Foreground-mass MAE, p.p. $\downarrow$",
            lambda row: _format_pp(
                row["foreground_fraction_error"]["mean_absolute_error"]
            ),
        ),
        (
            r"Empty-probability bias, p.p.",
            lambda row: _format_signed_pp(
                row["empty_mask_identity"][
                    "signed_bias_predicted_minus_observed"
                ]
            ),
        ),
        (
            r"Randomized-PIT KS distance $\downarrow$",
            lambda row: f"{row['randomized_cardinality_pit']['kolmogorov_smirnov_distance_to_uniform']:.3f}",
        ),
        (
            r"PIT outside central 90\%, \%",
            lambda row: _format_pp(
                row["randomized_cardinality_pit"][
                    "outside_central_90_percent_ratio"
                ]
            ),
        ),
        (
            r"Unattainable truth cardinality, \% $\downarrow$",
            lambda row: _format_pp(
                row["randomized_cardinality_pit"][
                    "zero_observed_point_mass_ratio"
                ]
            ),
        ),
    )
    for label, getter in metric_rows:
        rows.append(
            label + " & " + " & ".join(_paired_cells(by_key, getter)) + " \\\\"
        )
    header = " & ".join(DATASET_LABELS[dataset] for dataset in DATASETS)
    body = "\n".join(rows)
    return rf"""% Auto-generated by scripts/render_cardinality_diagnostics.py.
% Source analysis SHA-256: {source_sha256}
\begin{{table*}}[t]
\centering
\caption{{Exact shared-threshold cardinality diagnostics on target conditions.
Each cell is \textsc{{CLIP-T}} / \textsc{{DL-T}}.  Foreground-mass and empty-mask
quantities compare the exact $Q_p$ implications
$\mathbb{{E}}_{{Q_p}}[|Y|/|\Omega|]=|\Omega|^{{-1}}\sum_i p_i$ and
$Q_p(Y=\varnothing)=1-\max_i p_i$ with the foreground fraction and empty-mask
indicator of one observed reference. Bias and fixed-seed randomized-PIT
summaries are pooled label proxies; an observed zero-mass cardinality is a
direct support incompatibility. Favorable values do not establish pointwise
posterior calibration or validate $Q_p$.}}
\label{{tab:cardinality-diagnostics}}
\small
\setlength{{\tabcolsep}}{{3.5pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Diagnostic & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""


def write_output(tex: str, output_dir: str | os.PathLike[str]) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / OUTPUT_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cardinality table already exists: {destination}")
    payload = tex.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_NAME}.tmp-", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
        fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    analysis, source_sha256 = load_analysis(args.analysis)
    output = write_output(
        render_analysis(analysis, source_sha256=source_sha256), args.output_dir
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
