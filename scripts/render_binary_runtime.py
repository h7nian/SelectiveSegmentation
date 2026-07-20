"""Render the strict runtime analysis as one compact target-condition table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.analyze_binary import EXPECTED_CONDITIONS
from scripts.analyze_binary_runtime import (
    ANALYSIS_ARTIFACT_TYPE,
    ANALYSIS_SCHEMA_VERSION,
    TARGET_CONDITIONS,
)
from selectseg.benchmark_binary_runtime import METHODS_V1, METHODS_V2


OUTPUT_NAME = "binary_runtime.tex"
DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
TARGET_MODEL_CONDITIONS = ("clipseg-target", "deeplabv3-target")
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_id",
        "scope",
        "condition_sets",
        "provenance",
        "target_ranges",
        "conditions",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/binary_runtime_analysis/rendered"
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


def _finite(value: Any, *, location: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{location} must be finite and positive")
    return result


def _digest(value: Any, *, location: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase hexadecimal digest")
    return value


def load_analysis(path: str | os.PathLike[str]):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"runtime analysis does not exist: {source}")
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
        raise ValueError("runtime analysis must contain one JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def validate_analysis(value: Any):
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise ValueError("runtime analysis has an invalid top-level schema")
    if (
        value["schema_version"] != ANALYSIS_SCHEMA_VERSION
        or value["artifact_type"] != ANALYSIS_ARTIFACT_TYPE
    ):
        raise ValueError("runtime analysis has an unsupported type/schema")
    _digest(value["analysis_id"], location="runtime analysis_id", length=16)
    scope = value["scope"]
    if (
        not isinstance(scope, dict)
        or "hardware-dependent" not in scope.get("timing_status", "")
        or "algorithmic complexity" not in scope.get("complexity_status", "")
    ):
        raise ValueError("runtime analysis must separate timing and complexity")
    condition_sets = value["condition_sets"]
    expected_all = [f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS]
    expected_targets = [
        f"{dataset}/{condition}"
        for dataset, condition in EXPECTED_CONDITIONS
        if (dataset, condition) in TARGET_CONDITIONS
    ]
    if (
        not isinstance(condition_sets, dict)
        or condition_sets.get("all_conditions") != expected_all
        or condition_sets.get("target_conditions") != expected_targets
        or condition_sets.get("num_conditions") != 16
        or condition_sets.get("num_target_conditions") != 10
    ):
        raise ValueError("runtime condition sets are incomplete")
    conditions = value["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 16:
        raise ValueError("runtime analysis must contain exactly 16 conditions")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("runtime analysis provenance must be an object")
    protocol_version = provenance.get("runtime_protocol_version", 1)
    if protocol_version not in {1, 2}:
        raise ValueError("runtime analysis has an unsupported protocol version")
    methods_expected = METHODS_V1 if protocol_version == 1 else METHODS_V2
    required_provenance = {
        "campaign_lock_sha256",
        "benchmark_spec_sha256",
        "runtime_source_sha256",
        "analysis_source_sha256",
        "input_manifest_sha256",
        "campaign_lock_path",
        "inputs",
    }
    optional_provenance = {"runtime_protocol_version"}
    if protocol_version == 2:
        required_provenance.add("benchmark_lock_sha256")
    if not required_provenance <= set(provenance) or (
        set(provenance) - required_provenance - optional_provenance
    ):
        raise ValueError("runtime analysis provenance has an invalid schema")
    for field in (
        "campaign_lock_sha256",
        "benchmark_spec_sha256",
        "runtime_source_sha256",
        "analysis_source_sha256",
    ):
        _digest(provenance[field], location=f"provenance.{field}")
    if protocol_version == 2:
        _digest(
            provenance["benchmark_lock_sha256"],
            location="provenance.benchmark_lock_sha256",
        )
    if (
        not isinstance(provenance["campaign_lock_path"], str)
        or not provenance["campaign_lock_path"]
    ):
        raise ValueError("runtime analysis campaign-lock path is missing")
    inputs = provenance["inputs"]
    manifest_hashes = provenance["input_manifest_sha256"]
    input_fields = {
        "dataset",
        "condition",
        "records_path",
        "records_sha256",
        "manifest_path",
        "manifest_sha256",
    }
    if (
        not isinstance(inputs, list)
        or len(inputs) != 16
        or not isinstance(manifest_hashes, list)
        or len(manifest_hashes) != 16
    ):
        raise ValueError("runtime analysis must bind sixteen inputs")
    input_keys = []
    for index, row in enumerate(inputs):
        if not isinstance(row, dict) or set(row) != input_fields:
            raise ValueError(f"provenance.inputs[{index}] is malformed")
        key = row["dataset"], row["condition"]
        if key not in EXPECTED_CONDITIONS or key in input_keys:
            raise ValueError(f"provenance.inputs[{index}] has an invalid key")
        input_keys.append(key)
        for field in ("records_path", "manifest_path"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"provenance.inputs[{index}].{field} is missing")
        _digest(row["records_sha256"], location=f"inputs[{index}].records_sha256")
        observed_manifest = _digest(
            row["manifest_sha256"], location=f"inputs[{index}].manifest_sha256"
        )
        if observed_manifest != manifest_hashes[index]:
            raise ValueError("runtime input-manifest hash list is inconsistent")
    if set(input_keys) != set(EXPECTED_CONDITIONS):
        raise ValueError("runtime analysis inputs do not cover all conditions")
    identity = {
        key: item
        for key, item in provenance.items()
        if key not in {"campaign_lock_path", "inputs"}
    }
    expected_analysis_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if value["analysis_id"] != expected_analysis_id:
        raise ValueError("runtime analysis_id is inconsistent with provenance")
    by_key = {}
    required_condition = {
        "dataset",
        "condition",
        "model",
        "is_target_condition",
        "num_selected_images",
        "selected_pixel_count",
        "methods",
        "m32_joint_over_dice_exact_time_ratio",
        "process_peak_rss_bytes",
        "process_peak_rss_scope",
        "hardware",
    }
    required_method = {
        "method",
        "num_trials",
        "panel_wall_seconds",
        "milliseconds_per_image",
        "milliseconds_per_megapixel",
        "images_per_second",
    }
    for index, row in enumerate(conditions):
        location = f"conditions[{index}]"
        if not isinstance(row, dict) or set(row) != required_condition:
            raise ValueError(f"{location} has an invalid schema")
        key = row["dataset"], row["condition"]
        if key not in EXPECTED_CONDITIONS or key in by_key:
            raise ValueError(f"{location} has an invalid or duplicate key")
        if row["is_target_condition"] != (key in TARGET_CONDITIONS):
            raise ValueError(f"{location} has an inconsistent target flag")
        if row["num_selected_images"] != 16:
            raise ValueError(f"{location} must use the 16-image panel")
        selected_pixels = row["selected_pixel_count"]
        if not isinstance(selected_pixels, dict) or set(selected_pixels) != {
            "min",
            "median",
            "max",
            "total",
        }:
            raise ValueError(f"{location}.selected_pixel_count is malformed")
        for metric in ("min", "median", "max", "total"):
            _finite(
                selected_pixels[metric],
                location=f"{location}.selected_pixel_count.{metric}",
            )
        if not (
            selected_pixels["min"]
            <= selected_pixels["median"]
            <= selected_pixels["max"]
            <= selected_pixels["total"]
        ):
            raise ValueError(f"{location} has inconsistent selected-pixel counts")
        methods = row["methods"]
        if not isinstance(methods, dict) or set(methods) != set(methods_expected):
            raise ValueError(f"{location}.methods is incomplete")
        for method in methods_expected:
            summary = methods[method]
            if not isinstance(summary, dict) or set(summary) != required_method:
                raise ValueError(f"{location}.methods.{method} is malformed")
            if summary["method"] != method or summary["num_trials"] != 4:
                raise ValueError(f"{location}.methods.{method} has wrong identity")
            for metric in (
                "milliseconds_per_image",
                "milliseconds_per_megapixel",
                "images_per_second",
            ):
                _finite(summary[metric], location=f"{location}.{method}.{metric}")
            wall = summary["panel_wall_seconds"]
            if not isinstance(wall, dict) or set(wall) != {"min", "median", "max"}:
                raise ValueError(f"{location}.{method}.panel_wall_seconds is malformed")
            for metric in ("min", "median", "max"):
                _finite(wall[metric], location=f"{location}.{method}.wall.{metric}")
            if not wall["min"] <= wall["median"] <= wall["max"]:
                raise ValueError(f"{location}.{method} has inconsistent timing order")
            expected_per_image = 1000 * wall["median"] / row["num_selected_images"]
            expected_per_megapixel = (
                1000 * wall["median"] / (selected_pixels["total"] / 1_000_000)
            )
            expected_throughput = row["num_selected_images"] / wall["median"]
            for metric, expected in (
                ("milliseconds_per_image", expected_per_image),
                ("milliseconds_per_megapixel", expected_per_megapixel),
                ("images_per_second", expected_throughput),
            ):
                if not math.isclose(
                    summary[metric], expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{location}.methods.{method}.{metric} is inconsistent"
                    )
        ratio = _finite(
            row["m32_joint_over_dice_exact_time_ratio"],
            location=f"{location}.time_ratio",
        )
        expected_ratio = (
            methods["m32_joint"]["panel_wall_seconds"]["median"]
            / methods["dice_exact"]["panel_wall_seconds"]["median"]
        )
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{location}.time_ratio is inconsistent")
        if row["process_peak_rss_bytes"] is not None:
            _finite(row["process_peak_rss_bytes"], location=f"{location}.peak_rss")
        if (
            not isinstance(row["process_peak_rss_scope"], str)
            or not row["process_peak_rss_scope"]
        ):
            raise ValueError(f"{location}.process_peak_rss_scope is missing")
        hardware = row["hardware"]
        if not isinstance(hardware, dict) or set(hardware) != {
            "cpu_model",
            "partition",
            "node_list",
            "affinity_cpu_count",
        }:
            raise ValueError(f"{location}.hardware is malformed")
        by_key[key] = row
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("runtime conditions do not cover the declared benchmark")
    expected_range_fields = {
        *(f"{method}_milliseconds_per_image" for method in methods_expected),
        "m32_joint_over_dice_exact_time_ratio",
    }
    ranges = value["target_ranges"]
    if not isinstance(ranges, dict) or set(ranges) != expected_range_fields:
        raise ValueError("runtime target ranges have an invalid schema")
    targets = [by_key[key] for key in EXPECTED_CONDITIONS if key in TARGET_CONDITIONS]
    for field in sorted(expected_range_fields):
        summary = ranges[field]
        if not isinstance(summary, dict) or set(summary) != {"min", "max"}:
            raise ValueError(f"target_ranges.{field} is malformed")
        if field == "m32_joint_over_dice_exact_time_ratio":
            values = [row[field] for row in targets]
        else:
            method = field.removesuffix("_milliseconds_per_image")
            values = [row["methods"][method]["milliseconds_per_image"] for row in targets]
        for name, expected in (("min", min(values)), ("max", max(values))):
            observed = _finite(summary[name], location=f"target_ranges.{field}.{name}")
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"target_ranges.{field}.{name} is inconsistent")
    return by_key


def _format_time(value: float) -> str:
    if value < 0.1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _paired_cells(by_key, getter):
    cells = []
    for dataset in DATASETS:
        values = [getter(by_key[(dataset, condition)]) for condition in TARGET_MODEL_CONDITIONS]
        cells.append(" / ".join(values))
    return cells


def render_analysis(value: Mapping[str, Any], *, source_hash: str) -> str:
    by_key = validate_analysis(value)
    protocol_version = value["provenance"].get("runtime_protocol_version", 1)
    methods = METHODS_V1 if protocol_version == 1 else METHODS_V2
    method_labels = {
        "m2_joint": "M2 joint",
        "m8_joint": "M8 joint",
        "m32_joint": "M32 joint",
        "dice_exact": "Dice-Exact",
    }
    rows = []
    for method in methods:
        label = rf"{method_labels[method]}, ms/image $\downarrow$"
        cells = _paired_cells(
            by_key,
            lambda row, method=method: _format_time(
                row["methods"][method]["milliseconds_per_image"]
            ),
        )
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    cells = _paired_cells(
        by_key,
        lambda row: rf"{row['m32_joint_over_dice_exact_time_ratio']:.1f}$\times$",
    )
    rows.append(r"M32 / Exact time ratio" + " & " + " & ".join(cells) + r" \\")
    if protocol_version == 1:
        for method in methods:
            label = rf"{method_labels[method]}, ms/Mpixel $\downarrow$"
            cells = _paired_cells(
                by_key,
                lambda row, method=method: _format_time(
                    row["methods"][method]["milliseconds_per_megapixel"]
                ),
            )
            rows.append(label + " & " + " & ".join(cells) + r" \\")
    else:
        for method in methods:
            label = rf"{method_labels[method]}, images/s $\uparrow$"
            cells = _paired_cells(
                by_key,
                lambda row, method=method: _format_time(
                    row["methods"][method]["images_per_second"]
                ),
            )
            rows.append(label + " & " + " & ".join(cells) + r" \\")
    cells = _paired_cells(
        by_key,
        lambda row: (
            "--"
            if row["process_peak_rss_bytes"] is None
            else f"{row['process_peak_rss_bytes'] / 2**30:.2f}"
        ),
    )
    rows.append(r"Process peak RSS, GiB" + " & " + " & ".join(cells) + r" \\")
    header = " & ".join(DATASET_LABELS[dataset] for dataset in DATASETS)
    body = "\n".join(rows)
    if protocol_version == 1:
        protocol_caption = (
            "M32 joint computes Dice-, nHD-, and nHD95-indexed confidence "
            "together, whereas\nDice-Exact computes only Dice."
        )
        order_caption = "balanced-order"
        complexity_caption = (
            "Algorithmically, Dice-Exact sorts $N$ pixels in $O(N\\log N)$, "
            "while M32\nevaluates 32 candidate masks and their boundary-distance "
            "computations."
        )
    else:
        protocol_caption = (
            "M2, M8, and M32 each compute Dice-, nHD-, and nHD95-indexed "
            "confidence jointly; Dice-Exact computes Dice alone. Selected arrays "
            "are preloaded, and no confidence or boundary-distance result is "
            "reused across timed methods."
        )
        order_caption = "Williams-balanced-order"
        complexity_caption = (
            "The ladder measures empirical scaling with threshold count; it is "
            "not an asymptotic-complexity experiment."
        )
    return rf"""% Auto-generated by scripts/render_binary_runtime.py.
% Source analysis SHA-256: {source_hash}
\begin{{table*}}[t]
\centering
\caption{{Descriptive confidence-computation runtime on the deterministic
16-image pixel-count-quantile panel for each target condition. Each cell is
CLIP-T / DL-T. Values are medians over four {order_caption} whole-panel
measurements after one warm-up, using eight Python workers with one native
numerical thread each.
{protocol_caption} Model inference, artifact I/O, panel assembly,
and serialization are excluded. Peak RSS is a process high-water mark and is
not attributable to either method. Wall-clock values are hardware dependent:
conditions may run on different nodes or CPU models, so the within-condition
method ratio is more interpretable than between-condition timing differences.
{complexity_caption}}}
\label{{tab:binary-runtime}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
Metric & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""


def write_output(tex: str, output_dir: str | os.PathLike[str]):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / OUTPUT_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite rendered runtime table: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{OUTPUT_NAME}.", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(tex, encoding="utf-8")
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    analysis, source_hash = load_analysis(args.analysis)
    tex = render_analysis(analysis, source_hash=source_hash)
    path = write_output(tex, args.output_dir)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
