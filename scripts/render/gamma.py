"""Render compact manuscript TeX from strict gamma-sensitivity analysis JSON.

The renderer performs no input discovery and no new statistical estimation. It
validates the complete 16-condition report, recomputes the ten-target headline
summary, embeds the source JSON hash, writes to a content-addressed directory,
and refuses to replace an existing artifact.
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

from scripts.analyze.main import CONTRASTS, EXPECTED_CONDITIONS
from scripts.analyze.gamma import (
    ALL_GAMMAS,
    ARTIFACT_TYPE,
    AUXILIARY_GAMMAS,
    COVERAGES,
    GAMMA_PAIRS,
    INDEXED_SCORES,
    RISK_FIELDS,
    SCHEMA_VERSION,
    TARGET_CONDITIONS,
    aggregate_targets,
)


OUTPUT_NAME = "gamma_sensitivity.tex"
DEFAULT_OUTPUT_ROOT = "outputs/binary_gamma_sensitivity_analysis/rendered_v3"
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC 2018",
    "tn3k": "TN3K",
}
MODEL_LABELS = {
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
}
ORDERED_TARGET_CONDITIONS = tuple(
    key for key in EXPECTED_CONDITIONS if key in TARGET_CONDITIONS
)
CONTRAST_LABELS = {
    "dice_vs_nhd_under_dice": r"Dice--HD $\mid$ Dice",
    "dice_vs_nhd_under_nhd": r"Dice--HD $\mid$ HD",
    "nhd_vs_nhd95_under_nhd": r"HD--HD95 $\mid$ HD",
    "nhd_vs_nhd95_under_nhd95": r"HD--HD95 $\mid$ HD95",
}
PLOT_CONTRAST_LABELS = {
    "dice_vs_nhd_under_dice": "Dice - HD | Dice risk",
    "dice_vs_nhd_under_nhd": "Dice - HD | HD risk",
    "nhd_vs_nhd95_under_nhd": "HD - HD95 | HD risk",
    "nhd_vs_nhd95_under_nhd95": "HD - HD95 | HD95 risk",
}
RISK_LABELS = {
    "risk_dice": "Mean Dice loss",
    "risk_nhd": "Mean HD loss",
    "risk_nhd95": "Mean HD95 loss",
}


def _display_label(label: str) -> str:
    """Map legacy normalized-coordinate labels to manuscript terminology."""

    return label.replace("nHD95", "HD95").replace("nHD", "HD")
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_id",
        "scope",
        "specification",
        "condition_sets",
        "provenance",
        "target_headline",
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
        "action_quality_by_gamma",
        "contrasts",
        "indexed_score_stability",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-output")
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


def load_analysis(path: str | os.PathLike[str]) -> tuple[dict, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"gamma analysis JSON does not exist: {source}")
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
        raise ValueError("gamma analysis root must be an object")
    _assert_finite_tree(result, location=str(source))
    return result, hashlib.sha256(raw).hexdigest()


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
        raise ValueError(f"{location} must lie in [{minimum},{maximum}]")
    return result


def _validate_range(value: Any, *, location: str, minimum: float = -1.0) -> None:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ValueError(f"{location} has an invalid range schema")
    low = _number(value["min"], location=f"{location}.min", minimum=minimum)
    high = _number(value["max"], location=f"{location}.max", minimum=minimum)
    if low > high:
        raise ValueError(f"{location}.min exceeds max")


def _validate_correlation(value: Any, *, location: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "defined",
        "value",
        "undefined_reason",
    }:
        raise ValueError(f"{location} has an invalid correlation schema")
    if value["defined"]:
        _number(value["value"], location=f"{location}.value")
        if value["undefined_reason"] is not None:
            raise ValueError(f"{location} has inconsistent defined state")
    elif value["value"] is not None or not isinstance(value["undefined_reason"], str):
        raise ValueError(f"{location} has inconsistent undefined state")


def _gamma_key(gamma: float) -> str:
    return f"{gamma:.1f}"


def _pair_key(left: float, right: float) -> str:
    return f"{left:.1f}_vs_{right:.1f}"


def _expected_direction(value: float) -> str:
    if value < 0:
        return "left_lower_aurc"
    if value > 0:
        return "right_lower_aurc"
    return "exact_tie"


def _validate_action(value: Any, *, location: str) -> None:
    expected = {
        "mean_matched_losses",
        "deployed_prediction_empty_rate",
        "mean_prediction_foreground_fraction",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} has an invalid action-quality schema")
    losses = value["mean_matched_losses"]
    if not isinstance(losses, dict) or set(losses) != set(RISK_FIELDS):
        raise ValueError(f"{location}.mean_matched_losses is incomplete")
    for risk in RISK_FIELDS:
        _number(losses[risk], location=f"{location}.{risk}", minimum=0.0)
    for field in (
        "deployed_prediction_empty_rate",
        "mean_prediction_foreground_fraction",
    ):
        _number(value[field], location=f"{location}.{field}", minimum=0.0)


def _validate_contrast(value: Any, *, spec, location: str) -> None:
    expected = {
        "name",
        "left",
        "right",
        "risk",
        "by_gamma",
        "sensitivity_vs_primary",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} has an invalid contrast schema")
    for field in ("name", "left", "right", "risk"):
        if value[field] != getattr(spec, field):
            raise ValueError(f"{location}.{field} differs from the declaration")
    by_gamma = value["by_gamma"]
    if not isinstance(by_gamma, dict) or set(by_gamma) != {
        _gamma_key(gamma) for gamma in ALL_GAMMAS
    }:
        raise ValueError(f"{location}.by_gamma is incomplete")
    for gamma in ALL_GAMMAS:
        gamma_key = _gamma_key(gamma)
        item = by_gamma[gamma_key]
        if not isinstance(item, dict) or set(item) != {
            "left_aurc",
            "right_aurc",
            "difference_left_minus_right",
            "direction",
        }:
            raise ValueError(f"{location}.by_gamma.{gamma_key} is malformed")
        left = _number(
            item["left_aurc"], location=f"{location}.{gamma_key}.left", minimum=0.0
        )
        right = _number(
            item["right_aurc"], location=f"{location}.{gamma_key}.right", minimum=0.0
        )
        difference = _number(
            item["difference_left_minus_right"],
            location=f"{location}.{gamma_key}.difference",
        )
        if not math.isclose(difference, left - right, rel_tol=1e-13, abs_tol=1e-14):
            raise ValueError(f"{location}.{gamma_key} has inconsistent AURCs")
        if item["direction"] != _expected_direction(difference):
            raise ValueError(f"{location}.{gamma_key}.direction is inconsistent")
    sensitivity = value["sensitivity_vs_primary"]
    if not isinstance(sensitivity, dict) or set(sensitivity) != {
        _gamma_key(gamma) for gamma in AUXILIARY_GAMMAS
    }:
        raise ValueError(f"{location}.sensitivity_vs_primary is incomplete")
    primary = by_gamma["0.5"]
    for gamma in AUXILIARY_GAMMAS:
        gamma_key = _gamma_key(gamma)
        item = sensitivity[gamma_key]
        if not isinstance(item, dict) or set(item) != {
            "paired_change_from_gamma_0.5",
            "primary_direction",
            "auxiliary_direction",
            "direction_retained",
            "strict_reversal",
            "tie_transition",
        }:
            raise ValueError(f"{location}.sensitivity.{gamma_key} is malformed")
        expected_change = (
            by_gamma[gamma_key]["difference_left_minus_right"]
            - primary["difference_left_minus_right"]
        )
        change = _number(
            item["paired_change_from_gamma_0.5"],
            location=f"{location}.sensitivity.{gamma_key}.change",
        )
        if not math.isclose(change, expected_change, rel_tol=1e-13, abs_tol=1e-14):
            raise ValueError(
                f"{location}.sensitivity.{gamma_key} change is inconsistent"
            )
        primary_direction = primary["direction"]
        auxiliary_direction = by_gamma[gamma_key]["direction"]
        reversal = {primary_direction, auxiliary_direction} == {
            "left_lower_aurc",
            "right_lower_aurc",
        }
        retained = primary_direction == auxiliary_direction
        if (
            item["primary_direction"] != primary_direction
            or item["auxiliary_direction"] != auxiliary_direction
            or item["direction_retained"] is not retained
            or item["strict_reversal"] is not reversal
            or item["tie_transition"] is not (not retained and not reversal)
        ):
            raise ValueError(
                f"{location}.sensitivity.{gamma_key} direction is inconsistent"
            )


def _validate_score_stability(
    value: Any, *, score: str, label: str, location: str
) -> None:
    if not isinstance(value, dict) or set(value) != {"label", "gamma_pairs"}:
        raise ValueError(f"{location} has an invalid score-stability schema")
    if value["label"] != label:
        raise ValueError(f"{location}.label differs from the declaration")
    pairs = value["gamma_pairs"]
    expected_pairs = {_pair_key(left, right) for left, right in GAMMA_PAIRS}
    if not isinstance(pairs, dict) or set(pairs) != expected_pairs:
        raise ValueError(f"{location}.gamma_pairs is incomplete")
    for left_gamma, right_gamma in GAMMA_PAIRS:
        pair_key = _pair_key(left_gamma, right_gamma)
        item = pairs[pair_key]
        if not isinstance(item, dict) or set(item) != {
            "left_gamma",
            "right_gamma",
            "spearman_rho",
            "accepted_set_agreement",
        }:
            raise ValueError(f"{location}.{pair_key} is malformed")
        if item["left_gamma"] != left_gamma or item["right_gamma"] != right_gamma:
            raise ValueError(f"{location}.{pair_key} has inconsistent gamma labels")
        _validate_correlation(
            item["spearman_rho"], location=f"{location}.{pair_key}.spearman"
        )
        accepted = item["accepted_set_agreement"]
        if not isinstance(accepted, list) or len(accepted) != len(COVERAGES):
            raise ValueError(
                f"{location}.{pair_key}.accepted_set_agreement is incomplete"
            )
        by_coverage = {}
        for row in accepted:
            if not isinstance(row, dict) or set(row) != {
                "coverage",
                "tie_aware_fractional_jaccard",
            }:
                raise ValueError(f"{location}.{pair_key} has malformed agreement")
            coverage = _number(
                row["coverage"], location=f"{location}.{pair_key}.coverage", minimum=0.0
            )
            if coverage in by_coverage:
                raise ValueError(f"{location}.{pair_key} has duplicate coverage")
            by_coverage[coverage] = _number(
                row["tie_aware_fractional_jaccard"],
                location=f"{location}.{pair_key}.jaccard",
                minimum=0.0,
            )
        if set(by_coverage) != set(COVERAGES):
            raise ValueError(f"{location}.{pair_key} has unexpected coverages")


def validate_analysis(value: Any) -> dict[tuple[str, str], dict]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_KEYS:
        raise ValueError("gamma analysis has an invalid top-level schema")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_type"] != ARTIFACT_TYPE
    ):
        raise ValueError("gamma analysis type/schema is unsupported")
    if not isinstance(value["analysis_id"], str) or len(value["analysis_id"]) != 16:
        raise ValueError("analysis_id must be a 16-character content identity")
    scope = value["scope"]
    if not isinstance(scope, dict) or "neither" not in scope.get("status", ""):
        raise ValueError(
            "analysis must preserve the non-tuning/non-guarantee limitation"
        )
    sets = value["condition_sets"]
    expected_all = [
        f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
    ]
    expected_targets = [
        f"{dataset}/{condition}" for dataset, condition in ORDERED_TARGET_CONDITIONS
    ]
    if (
        not isinstance(sets, dict)
        or sets.get("all_conditions") != expected_all
        or sets.get("target_conditions") != expected_targets
        or sets.get("num_conditions") != 16
        or sets.get("num_target_conditions") != 10
        or sets.get("num_auxiliary_experiments") != 32
    ):
        raise ValueError("condition_sets does not contain the exact 16/10/32 design")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    for digest in (
        provenance.get("analysis_source_sha256"),
        provenance.get("auxiliary_lock", {}).get("sha256"),
        provenance.get("canonical_primary_analysis", {}).get("sha256"),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("provenance contains an invalid SHA-256 digest")
    auxiliary_inputs = provenance.get("auxiliary_inputs")
    if not isinstance(auxiliary_inputs, list) or len(auxiliary_inputs) != 32:
        raise ValueError("provenance must retain exactly 32 auxiliary inputs")

    rows = value["conditions"]
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("conditions must contain exactly 16 rows")
    by_key = {}
    for index, row in enumerate(rows):
        location = f"conditions[{index}]"
        if not isinstance(row, dict) or set(row) != CONDITION_KEYS:
            raise ValueError(f"{location} has an invalid schema")
        key = row["dataset"], row["condition"]
        if key not in EXPECTED_CONDITIONS or key in by_key:
            raise ValueError(f"{location} has an unknown or duplicate condition")
        if row["is_target_condition"] is not (key in TARGET_CONDITIONS):
            raise ValueError(f"{location}.is_target_condition is inconsistent")
        expected_model = "clipseg" if key[1].startswith("clipseg") else "deeplabv3"
        if row["model"] != expected_model:
            raise ValueError(f"{location}.model is inconsistent")
        if isinstance(row["num_images"], bool) or not isinstance(
            row["num_images"], int
        ):
            raise ValueError(f"{location}.num_images must be a positive integer")
        if row["num_images"] <= 0:
            raise ValueError(f"{location}.num_images must be a positive integer")
        action = row["action_quality_by_gamma"]
        if not isinstance(action, dict) or set(action) != {
            _gamma_key(gamma) for gamma in ALL_GAMMAS
        }:
            raise ValueError(f"{location}.action_quality_by_gamma is incomplete")
        for gamma in ALL_GAMMAS:
            _validate_action(action[_gamma_key(gamma)], location=f"{location}.action")
        contrasts = row["contrasts"]
        if not isinstance(contrasts, dict) or set(contrasts) != {
            spec.name for spec in CONTRASTS
        }:
            raise ValueError(f"{location}.contrasts is incomplete")
        for spec in CONTRASTS:
            _validate_contrast(
                contrasts[spec.name], spec=spec, location=f"{location}.{spec.name}"
            )
        stability = row["indexed_score_stability"]
        if not isinstance(stability, dict) or set(stability) != {
            score for score, _ in INDEXED_SCORES
        }:
            raise ValueError(f"{location}.indexed_score_stability is incomplete")
        for score, label in INDEXED_SCORES:
            _validate_score_stability(
                stability[score],
                score=score,
                label=label,
                location=f"{location}.{score}",
            )
        by_key[key] = row
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("condition rows differ from the exact benchmark")
    recomputed = aggregate_targets([by_key[key] for key in EXPECTED_CONDITIONS])
    if recomputed != value["target_headline"]:
        raise ValueError("target headline differs from the ten condition statistics")
    return by_key


def _format(value: float, digits: int = 3) -> str:
    threshold = 0.5 * 10 ** (-digits)
    if abs(value) < threshold:
        value = 0.0
    return f"{value:.{digits}f}"


def _range(
    value: Mapping[str, float], *, digits: int = 3, percent: bool = False
) -> str:
    low = float(value["min"])
    high = float(value["max"])
    if percent:
        return rf"$[{100 * low:.1f},\,{100 * high:.1f}]\%$"
    return rf"$[{_format(low, digits)},\,{_format(high, digits)}]$"


def _triple(contrast: Mapping[str, Any]) -> str:
    values = [
        contrast["by_gamma"][_gamma_key(gamma)]["difference_left_minus_right"]
        for gamma in ALL_GAMMAS
    ]
    marker = ""
    if any(
        contrast["sensitivity_vs_primary"][_gamma_key(gamma)]["strict_reversal"]
        for gamma in AUXILIARY_GAMMAS
    ):
        marker = r"\textsuperscript{R}"
    return "/".join(_format(100 * value) for value in values) + marker


def _macro_triple(by_key: Mapping[tuple[str, str], dict], *, contrast_name: str) -> str:
    values = []
    for gamma in ALL_GAMMAS:
        gamma_key = _gamma_key(gamma)
        differences = [
            by_key[key]["contrasts"][contrast_name]["by_gamma"][gamma_key][
                "difference_left_minus_right"
            ]
            for key in ORDERED_TARGET_CONDITIONS
        ]
        values.append(math.fsum(differences) / len(differences))
    primary_direction = _expected_direction(values[ALL_GAMMAS.index(0.5)])
    has_reversal = any(
        {
            primary_direction,
            _expected_direction(values[ALL_GAMMAS.index(gamma)]),
        }
        == {"left_lower_aurc", "right_lower_aurc"}
        for gamma in AUXILIARY_GAMMAS
    )
    marker = r"\textsuperscript{R}" if has_reversal else ""
    return "/".join(_format(100 * value) for value in values) + marker


def _aurc_range(value: Mapping[str, float]) -> str:
    return rf"$[{_format(100 * float(value['min']))},\,{_format(100 * float(value['max']))}]$"


def _render_contrast_table(by_key: Mapping[tuple[str, str], dict]) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Deployment-threshold sensitivity of the four adjacent-geometry "
        r"AURC contrasts in the ten target-adapted conditions. Each cell reports "
        r"$100\Delta_{.3}/100\Delta_{.5}/100\Delta_{.7}$, where $\Delta=\mathrm{AURC}(\text{left})"
        r"-\mathrm{AURC}(\text{right})$ and lower AURC is better. Negative values "
        r"favor the left score. In a condition row, superscript R marks a strict sign "
        r"reversal relative to the locked $\gamma=0.5$ action; in the macro-mean row, "
        r"it marks a reversal of the ten-condition macro mean. These are descriptive action "
        r"sensitivities, not threshold tuning or a robustness guarantee.}",
        r"\label{tab:gamma-contrast-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Model & Dice--HD $\mid$ Dice & Dice--HD $\mid$ HD & HD--HD95 $\mid$ HD & HD--HD95 $\mid$ HD95 \\",
        r"\midrule",
    ]
    for dataset, condition in ORDERED_TARGET_CONDITIONS:
        row = by_key[(dataset, condition)]
        cells = [DATASET_LABELS[dataset], MODEL_LABELS[condition]]
        cells.extend(_triple(row["contrasts"][spec.name]) for spec in CONTRASTS)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    macro_cells = [r"\textbf{Macro mean}", "--"]
    macro_cells.extend(
        _macro_triple(by_key, contrast_name=spec.name) for spec in CONTRASTS
    )
    lines.append(" & ".join(macro_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return lines


def _render_action_panel(headline: Mapping[str, Any]) -> list[str]:
    action = headline["action_quality"]
    lines = [
        r"\textbf{(a) Action-quality ranges across ten target conditions}\\[2pt]",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & $\gamma=0.3$ & $\gamma=0.5$ & $\gamma=0.7$ \\",
        r"\midrule",
    ]
    for risk in RISK_FIELDS:
        cells = [RISK_LABELS[risk]]
        cells.extend(
            _range(action[_gamma_key(gamma)]["mean_matched_loss_ranges"][risk])
            for gamma in ALL_GAMMAS
        )
        lines.append(" & ".join(cells) + r" \\")
    cells = ["Predicted-empty rate"]
    cells.extend(
        _range(
            action[_gamma_key(gamma)]["deployed_prediction_empty_rate_range"],
            percent=True,
        )
        for gamma in ALL_GAMMAS
    )
    lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\vspace{5pt}"])
    return lines


def _render_stability_panel(headline: Mapping[str, Any]) -> list[str]:
    stability = headline["indexed_score_stability"]
    lines = [
        r"\textbf{(b) Score-ranking and accepted-set stability versus the primary action}\\[2pt]",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Score & Comparison & Spearman $\rho$ & $J_{0.25}$ & $J_{0.50}$ & $J_{0.75}$ \\",
        r"\midrule",
    ]
    for score, label in INDEXED_SCORES:
        for gamma in AUXILIARY_GAMMAS:
            pair_key = _pair_key(gamma, 0.5)
            item = stability[score]["gamma_pairs"][pair_key]
            correlation = item["spearman_rho"]
            if correlation["range"] is None:
                rho = "n/a"
            else:
                rho = _range(correlation["range"])
                if correlation["num_undefined"]:
                    rho += rf"$^{{\dagger {correlation['num_undefined']}}}$"
            cells = [_display_label(label), rf"${gamma:.1f}$ vs $.5$", rho]
            cells.extend(
                _range(item["accepted_set_jaccard_ranges"][f"{coverage:.2f}"])
                for coverage in COVERAGES
            )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\vspace{5pt}"])
    return lines


def _render_direction_panel(headline: Mapping[str, Any]) -> list[str]:
    contrasts = headline["contrasts"]
    lines = [
        r"\textbf{(c) Direction retention and paired contrast change from $\gamma=0.5$}\\[2pt]",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Contrast & $\gamma$ & Retained/10 & Reversed/10 & Tie transitions/10 & Range of $100(\Delta_\gamma-\Delta_{.5})$ \\",
        r"\midrule",
    ]
    for spec in CONTRASTS:
        for gamma in AUXILIARY_GAMMAS:
            item = contrasts[spec.name]["sensitivity_vs_primary"][_gamma_key(gamma)]
            cells = [
                CONTRAST_LABELS[spec.name],
                f"{gamma:.1f}",
                str(item["num_direction_retained"]),
                str(item["num_strict_reversals"]),
                str(item["num_tie_transitions"]),
                _aurc_range(item["paired_change_from_gamma_0.5"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return lines


def render_analysis(value: Any, *, source_hash: str) -> str:
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("source_hash must be a lowercase SHA-256 digest")
    by_key = validate_analysis(value)
    lines = [
        "% Auto-generated by scripts/render_gamma_sensitivity.py; do not edit.",
        f"% Source gamma analysis JSON SHA-256: {source_hash}",
    ]
    lines.extend(_render_contrast_table(by_key))
    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Descriptive deployment-action sensitivity summaries over the ten "
            r"target-adapted conditions. Ranges are minimum--maximum across conditions, "
            r"which are not independent replicates. Accepted-set agreement is the "
            r"tie-aware fractional Jaccard. Direction is defined by the exact sign of "
            r"the signed AURC contrast; displayed contrasts and changes are on the "
            r"$\times100$ scale and no data-dependent tolerance is used. Action "
            r"quality is reported so that stable rankings cannot conceal a degraded "
            r"deployed mask. This analysis does not choose a threshold and does not "
            r"establish a robustness guarantee.}",
            r"\label{tab:gamma-action-and-ranking-sensitivity}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\centering",
        ]
    )
    lines.extend(_render_action_panel(value["target_headline"]))
    lines.extend(_render_stability_panel(value["target_headline"]))
    lines.extend(_render_direction_panel(value["target_headline"]))
    lines.extend([r"\end{minipage}", r"\end{table*}", ""])
    return "\n".join(lines)


def macro_contrast_values(
    by_key: Mapping[tuple[str, str], dict], *, contrast_name: str
) -> list[float]:
    """Return target-condition macro contrasts on the manuscript display scale."""
    return [
        100
        * math.fsum(
            by_key[key]["contrasts"][contrast_name]["by_gamma"][_gamma_key(gamma)][
                "difference_left_minus_right"
            ]
            for key in ORDERED_TARGET_CONDITIONS
        )
        / len(ORDERED_TARGET_CONDITIONS)
        for gamma in ALL_GAMMAS
    ]


def render_figure(
    by_key: Mapping[tuple[str, str], dict], output: str | os.PathLike[str]
) -> Path:
    """Render the four predeclared macro contrasts across deployment thresholds."""
    import matplotlib.pyplot as plt

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite gamma figure: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    colors = ("#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf")
    markers = ("o", "s", "^", "D")
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.axhline(0, color="0.45", linewidth=1, linestyle="--")
    for (contrast_name, label), color, marker in zip(
        PLOT_CONTRAST_LABELS.items(), colors, markers, strict=True
    ):
        axis.plot(
            ALL_GAMMAS,
            macro_contrast_values(by_key, contrast_name=contrast_name),
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label,
        )
    axis.set_xticks(ALL_GAMMAS)
    axis.set_xlabel(r"Deployed action threshold $\gamma$")
    axis.set_ylabel(r"Macro AURC contrast $\times 100$")
    axis.set_title("Action-threshold sensitivity (10 target conditions)")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    figure.tight_layout()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, bbox_inches="tight", metadata={"Creator": __name__})
        os.link(temporary, destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return destination


def write_output(
    tex: str, output_root: str | os.PathLike[str], *, source_hash: str
) -> Path:
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("source_hash must be a lowercase SHA-256 digest")
    directory = Path(output_root) / source_hash[:16]
    destination = directory / OUTPUT_NAME
    if destination.exists() or destination.is_symlink() or directory.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite gamma TeX artifact: {destination}"
        )
    directory.mkdir(parents=True, exist_ok=False)
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
    analysis, source_hash = load_analysis(args.analysis)
    tex = render_analysis(analysis, source_hash=source_hash)
    destination = write_output(tex, args.output_root, source_hash=source_hash)
    print(destination.as_posix())
    if args.figure_output:
        figure = render_figure(validate_analysis(analysis), args.figure_output)
        print(figure.as_posix())


if __name__ == "__main__":
    main()
