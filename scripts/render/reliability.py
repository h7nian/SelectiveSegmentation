"""Render five fixed 2x3 matched-risk reliability figures from strict JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze.main import EXPECTED_CONDITIONS
from scripts.analyze.reliability import (
    ARTIFACT_TYPE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    GROUP_BINS,
    MATCHED_PAIRS,
    SCHEMA_VERSION,
    TARGET_CONDITIONS,
)


DATASET_ORDER = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_LABELS = {
    "pet": "Oxford-IIIT Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
MODEL_ORDER = ("clipseg-target", "deeplabv3-target")
MODEL_LABELS = {
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
}
PANEL_TITLES = {
    "confidence_dice_exact": "Dice-Exact → Dice",
    "confidence_nhd_m32": "HD-M32 → HD",
    "confidence_nhd95_m32": "HD95-M32 → HD95",
}
COLORS = ("#1f77b4", "#d95f02", "#2ca02c")
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
CONDITION_FIELDS = frozenset(
    {"dataset", "condition", "model", "num_images", "panels"}
)
PANEL_FIELDS = frozenset(
    {
        "score_field",
        "score_label",
        "predicted_risk_definition",
        "observed_loss_field",
        "loss_label",
        "bins",
    }
)
BIN_FIELDS = frozenset(
    {
        "bin_index",
        "num_images",
        "minimum_predicted_risk",
        "maximum_predicted_risk",
        "mean_predicted_risk",
        "mean_observed_loss",
        "pointwise_ci_lower",
        "pointwise_ci_upper",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/binary_matched_risk_reliability/rendered",
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
        raise ValueError(f"{location} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, location=f"{location}[{index}]")


def load_analysis(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"reliability analysis does not exist: {source}")
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
        raise ValueError("reliability analysis root must be an object")
    _finite_tree(value, location=str(source))
    return value, hashlib.sha256(raw).hexdigest()


def _unit(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must lie in [0,1]")
    return result


def _validate_protocol(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if (
        value.get("num_bins") != GROUP_BINS
        or value.get("bootstrap_seed") != BOOTSTRAP_SEED
        or value.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or value.get("confidence_level") != CONFIDENCE_LEVEL
    ):
        raise ValueError("reliability protocol differs from the fixed 10/20260720/2000 design")
    for field in ("grouping", "bootstrap_unit", "interval"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"protocol.{field} must be nonempty")
    expected_pairs = [
        {
            "score_field": score,
            "predicted_risk_definition": f"-{score}",
            "observed_loss_field": risk,
            "score_label": score_label,
            "loss_label": loss_label,
        }
        for score, risk, score_label, loss_label in MATCHED_PAIRS
    ]
    if value.get("matched_pairs") != expected_pairs:
        raise ValueError("protocol matched pairs have changed")


def validate_analysis(value: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise ValueError("reliability analysis has an invalid top-level schema")
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("reliability analysis type/schema is unsupported")
    analysis_id = value["analysis_id"]
    if (
        not isinstance(analysis_id, str)
        or len(analysis_id) != 16
        or any(character not in "0123456789abcdef" for character in analysis_id)
    ):
        raise ValueError("analysis_id must be sixteen lowercase hex characters")
    scope = value["scope"]
    if (
        not isinstance(scope, dict)
        or "single-label descriptive" not in scope.get("status", "")
        or "pointwise" not in scope.get("interval_limitation", "")
        or "not simultaneous" not in scope.get("interval_limitation", "")
        or "cannot establish pointwise" not in scope.get("posterior_limitation", "")
    ):
        raise ValueError("scope must preserve descriptive and interval limitations")
    _validate_protocol(value["protocol"])

    expected_all = [f"{a}/{b}" for a, b in EXPECTED_CONDITIONS]
    expected_targets = [
        f"{a}/{b}" for a, b in EXPECTED_CONDITIONS if (a, b) in TARGET_CONDITIONS
    ]
    if value["condition_sets"] != {
        "canonical_complete": True,
        "canonical_conditions": expected_all,
        "target_conditions": expected_targets,
        "num_canonical_conditions": 16,
        "num_target_conditions": 10,
    }:
        raise ValueError("condition_sets differs from the fixed 16/10 design")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "workflow_source_sha256",
        "canonical_validation",
    }:
        raise ValueError("provenance has an invalid schema")
    source_sha = provenance["workflow_source_sha256"]
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("workflow source SHA-256 is invalid")
    canonical = provenance["canonical_validation"]
    if (
        not isinstance(canonical, dict)
        or canonical.get("binding") != "campaign-lock"
        or len(canonical.get("inputs", [])) != 16
    ):
        raise ValueError("canonical validation provenance is incomplete")

    rows = value["conditions"]
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("reliability analysis must contain ten target conditions")
    by_key = {}
    expected_panel_identity = [
        (score, risk, score_label, loss_label)
        for score, risk, score_label, loss_label in MATCHED_PAIRS
    ]
    for row_index, row in enumerate(rows):
        location = f"conditions[{row_index}]"
        if not isinstance(row, dict) or set(row) != CONDITION_FIELDS:
            raise ValueError(f"{location} has an invalid schema")
        key = row["dataset"], row["condition"]
        if key not in TARGET_CONDITIONS or key in by_key:
            raise ValueError(f"{location} has an invalid or duplicate target key")
        expected_model = "clipseg" if key[1] == "clipseg-target" else "deeplabv3"
        if row["model"] != expected_model:
            raise ValueError(f"{location}.model is inconsistent")
        num_images = row["num_images"]
        if isinstance(num_images, bool) or not isinstance(num_images, int) or num_images < 10:
            raise ValueError(f"{location}.num_images must be an integer >= 10")
        panels = row["panels"]
        if not isinstance(panels, list) or len(panels) != len(MATCHED_PAIRS):
            raise ValueError(f"{location}.panels must contain three entries")
        observed_identity = []
        for panel_index, panel in enumerate(panels):
            panel_location = f"{location}.panels[{panel_index}]"
            if not isinstance(panel, dict) or set(panel) != PANEL_FIELDS:
                raise ValueError(f"{panel_location} has an invalid schema")
            identity = (
                panel["score_field"],
                panel["observed_loss_field"],
                panel["score_label"],
                panel["loss_label"],
            )
            observed_identity.append(identity)
            if panel["predicted_risk_definition"] != f"-{panel['score_field']}":
                raise ValueError(f"{panel_location} has a wrong risk definition")
            bins = panel["bins"]
            if not isinstance(bins, list) or len(bins) != GROUP_BINS:
                raise ValueError(f"{panel_location} must contain ten bins")
            counts = []
            for bin_offset, item in enumerate(bins, start=1):
                bin_location = f"{panel_location}.bins[{bin_offset - 1}]"
                if not isinstance(item, dict) or set(item) != BIN_FIELDS:
                    raise ValueError(f"{bin_location} has an invalid schema")
                if item["bin_index"] != bin_offset:
                    raise ValueError(f"{bin_location}.bin_index is inconsistent")
                count = item["num_images"]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{bin_location}.num_images must be positive")
                counts.append(count)
                low_x = _unit(
                    item["minimum_predicted_risk"], location=f"{bin_location}.min_x"
                )
                mean_x = _unit(
                    item["mean_predicted_risk"], location=f"{bin_location}.mean_x"
                )
                high_x = _unit(
                    item["maximum_predicted_risk"], location=f"{bin_location}.max_x"
                )
                _unit(
                    item["mean_observed_loss"], location=f"{bin_location}.mean_y"
                )
                low_y = _unit(
                    item["pointwise_ci_lower"], location=f"{bin_location}.low_y"
                )
                high_y = _unit(
                    item["pointwise_ci_upper"], location=f"{bin_location}.high_y"
                )
                if not low_x <= mean_x <= high_x or not low_y <= high_y:
                    raise ValueError(f"{bin_location} has inconsistent bounds")
            if sum(counts) != num_images or max(counts) - min(counts) > 1:
                raise ValueError(f"{panel_location} bins are not equal-count")
        if observed_identity != expected_panel_identity:
            raise ValueError(f"{location}.panels do not match the fixed pairs")
        by_key[key] = row
    if set(by_key) != set(TARGET_CONDITIONS):
        raise ValueError("target condition coverage is incomplete")
    return by_key


def _panel_by_score(condition: Mapping[str, Any], score_field: str) -> Mapping[str, Any]:
    return next(panel for panel in condition["panels"] if panel["score_field"] == score_field)


def render_dataset_figure(
    by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    dataset: str,
    destination: Path,
    source_hash: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), squeeze=False)
    for column, (score, _, _, _) in enumerate(MATCHED_PAIRS):
        column_panels = [
            _panel_by_score(by_key[(dataset, condition)], score)
            for condition in MODEL_ORDER
        ]
        maximum = max(
            value
            for panel in column_panels
            for item in panel["bins"]
            for value in (
                item["maximum_predicted_risk"],
                item["pointwise_ci_upper"],
                item["mean_observed_loss"],
            )
        )
        limit = min(1.0, max(0.08, 1.08 * maximum))
        for row_index, condition in enumerate(MODEL_ORDER):
            panel = column_panels[row_index]
            bins = panel["bins"]
            x = np.asarray([item["mean_predicted_risk"] for item in bins])
            y = np.asarray([item["mean_observed_loss"] for item in bins])
            low = np.asarray([item["pointwise_ci_lower"] for item in bins])
            high = np.asarray([item["pointwise_ci_upper"] for item in bins])
            axis = axes[row_index, column]
            axis.plot(
                [0.0, limit],
                [0.0, limit],
                linestyle="--",
                linewidth=1.0,
                color="#666666",
                label="identity",
                zorder=1,
            )
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                color=COLORS[column],
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                capsize=2.2,
                label="equal-count bin",
                zorder=2,
            )
            axis.set_xlim(0.0, limit)
            axis.set_ylim(0.0, limit)
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")
            if row_index == 0:
                axis.set_title(PANEL_TITLES[score], fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{MODEL_LABELS[condition]}\nMean observed loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(DATASET_LABELS[dataset], y=0.915, fontsize=12)
    figure.supxlabel("Mean predicted working risk", y=0.073)
    figure.text(
        0.5,
        0.018,
        "Single-label descriptive diagnostic; bars are pointwise 95% within-bin "
        "image-bootstrap CIs (2,000 resamples), not simultaneous bands.",
        ha="center",
        fontsize=7.5,
    )
    figure.subplots_adjust(
        top=0.81,
        bottom=0.16,
        left=0.08,
        right=0.99,
        hspace=0.34,
        wspace=0.25,
    )
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={
            "Title": f"Matched-risk reliability: {DATASET_LABELS[dataset]}",
            "Subject": f"Source analysis SHA-256 {source_hash}",
            "Creator": "scripts.render_matched_risk_reliability",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def render_figures(
    analysis: Mapping[str, Any],
    *,
    source_hash: str,
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("source_hash must be a lowercase SHA-256 digest")
    by_key = validate_analysis(analysis)
    directory = Path(output_dir)
    if directory.is_symlink():
        raise ValueError(f"output directory may not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    destinations = [
        directory / f"matched_risk_reliability_{dataset}.pdf"
        for dataset in DATASET_ORDER
    ]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("refusing to overwrite matched-risk reliability figures")

    created = []
    try:
        with tempfile.TemporaryDirectory(prefix=".matched-risk-", dir=directory) as staging:
            staging_dir = Path(staging)
            for dataset, destination in zip(DATASET_ORDER, destinations, strict=True):
                temporary = staging_dir / destination.name
                render_dataset_figure(
                    by_key,
                    dataset=dataset,
                    destination=temporary,
                    source_hash=source_hash,
                )
                os.link(temporary, destination)
                created.append(destination)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return destinations


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    analysis, source_hash = load_analysis(args.analysis)
    for destination in render_figures(
        analysis,
        source_hash=source_hash,
        output_dir=args.output_dir,
    ):
        print(destination.as_posix())


if __name__ == "__main__":
    main()
