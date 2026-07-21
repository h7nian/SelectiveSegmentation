"""Render a concise appendix table from strict M=128 analysis JSON.

The renderer performs no analysis and never discovers inputs.  It validates
the complete 16-condition report, recomputes the ten-target aggregate ranges,
embeds the source JSON hash, and refuses to replace an existing TeX artifact.
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
from scripts.analyze.m128 import (
    ARTIFACT_TYPE,
    COMPARISONS,
    SCHEMA_VERSION,
    TARGET_CONDITIONS,
    _aggregate_target_ranges,
)


OUTPUT_NAME = "m128_numerical_reference.tex"
DEFAULT_OUTPUT_DIR = "outputs/binary_m128_auxiliary_analysis/rendered_v2"
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
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_id",
        "scope",
        "specification",
        "condition_sets",
        "provenance",
        "target_aggregate_ranges",
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
        "comparisons",
    }
)
COMPARISON_KEYS = frozenset(
    {
        "name",
        "label",
        "candidate_score",
        "reference_score",
        "matched_risk",
        "reference_interpretation",
        "num_images",
        "per_image_absolute_score_error",
        "rank_agreement",
        "matched_risk_aurc",
    }
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
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


def load_analysis(path: str | os.PathLike[str]) -> tuple[dict, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"analysis JSON does not exist: {source}")
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
        raise ValueError("analysis root must be an object")
    _assert_finite_tree(result, location=str(source))
    return result, hashlib.sha256(raw).hexdigest()


def _number(
    value: Any,
    *,
    location: str,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{location} must lie in [{minimum}, {maximum}]")
    return result


def _correlation(value: Any, *, location: str) -> float:
    if not isinstance(value, dict) or set(value) != {
        "defined",
        "value",
        "undefined_reason",
    }:
        raise ValueError(f"{location} has an invalid correlation schema")
    if value["defined"] is not True or value["undefined_reason"] is not None:
        raise ValueError(f"{location} must be defined for every target condition")
    return _number(value["value"], location=f"{location}.value", minimum=-1.0)


def _validate_comparison(value: Any, *, spec, location: str, num_images: int) -> dict:
    if not isinstance(value, dict) or set(value) != COMPARISON_KEYS:
        raise ValueError(f"{location} has an invalid comparison schema")
    expected_strings = {
        "name": spec.name,
        "label": spec.label,
        "candidate_score": spec.candidate_score,
        "reference_score": spec.reference_score,
        "matched_risk": spec.matched_risk,
        "reference_interpretation": spec.interpretation,
    }
    for field, expected in expected_strings.items():
        if value[field] != expected:
            raise ValueError(f"{location}.{field} differs from the declared comparison")
    if value["num_images"] != num_images:
        raise ValueError(f"{location}.num_images differs from its condition")
    errors = value["per_image_absolute_score_error"]
    if not isinstance(errors, dict) or set(errors) != {"mean", "median", "p95", "max"}:
        raise ValueError(f"{location}.per_image_absolute_score_error is malformed")
    for metric in ("mean", "median", "p95", "max"):
        _number(errors[metric], location=f"{location}.errors.{metric}")
    if errors["max"] < max(errors["mean"], errors["median"], errors["p95"]):
        raise ValueError(f"{location} has inconsistent absolute-error summaries")
    ranks = value["rank_agreement"]
    if not isinstance(ranks, dict) or set(ranks) != {"spearman_rho", "kendall_tau_b"}:
        raise ValueError(f"{location}.rank_agreement is malformed")
    _correlation(ranks["spearman_rho"], location=f"{location}.spearman_rho")
    _correlation(ranks["kendall_tau_b"], location=f"{location}.kendall_tau_b")
    aurc = value["matched_risk_aurc"]
    required_aurc = {
        "candidate",
        "reference",
        "signed_candidate_minus_reference",
        "absolute_gap",
        "tie_policy",
    }
    if not isinstance(aurc, dict) or set(aurc) != required_aurc:
        raise ValueError(f"{location}.matched_risk_aurc is malformed")
    candidate = _number(aurc["candidate"], location=f"{location}.aurc.candidate")
    reference = _number(aurc["reference"], location=f"{location}.aurc.reference")
    signed = _number(
        aurc["signed_candidate_minus_reference"],
        location=f"{location}.aurc.signed",
        minimum=-1.0,
    )
    gap = _number(aurc["absolute_gap"], location=f"{location}.aurc.absolute_gap")
    if not math.isclose(signed, candidate - reference, abs_tol=1e-14):
        raise ValueError(f"{location} has inconsistent signed AURC gap")
    if not math.isclose(gap, abs(signed), abs_tol=1e-14):
        raise ValueError(f"{location} has inconsistent absolute AURC gap")
    return value


def validate_analysis(value: Any) -> dict[tuple[str, str], dict]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_KEYS:
        raise ValueError("analysis has an invalid top-level schema")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_type"] != ARTIFACT_TYPE
    ):
        raise ValueError("analysis type/schema is unsupported")
    if not isinstance(value["analysis_id"], str) or len(value["analysis_id"]) != 16:
        raise ValueError("analysis_id must be a 16-character content identity")
    scope = value["scope"]
    if not isinstance(scope, dict) or "not an exact integral" not in scope.get(
        "m128_status", ""
    ):
        raise ValueError("analysis must explicitly identify M128 as non-exact")
    sets = value["condition_sets"]
    if not isinstance(sets, dict):
        raise ValueError("condition_sets must be an object")
    expected_all = [
        f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
    ]
    expected_target = [
        f"{dataset}/{condition}" for dataset, condition in ORDERED_TARGET_CONDITIONS
    ]
    if (
        sets.get("all_conditions") != expected_all
        or sets.get("target_conditions") != expected_target
        or sets.get("num_conditions") != 16
        or sets.get("num_target_conditions") != 10
    ):
        raise ValueError("condition_sets does not contain the exact 16/10 benchmark")
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
        if row["model"] not in {"clipseg", "deeplabv3"}:
            raise ValueError(f"{location}.model is unsupported")
        num_images = row["num_images"]
        if (
            isinstance(num_images, bool)
            or not isinstance(num_images, int)
            or num_images <= 0
        ):
            raise ValueError(f"{location}.num_images must be positive")
        comparisons = row["comparisons"]
        if not isinstance(comparisons, dict) or set(comparisons) != {
            spec.name for spec in COMPARISONS
        }:
            raise ValueError(f"{location}.comparisons is incomplete")
        for spec in COMPARISONS:
            _validate_comparison(
                comparisons[spec.name],
                spec=spec,
                location=f"{location}.comparisons.{spec.name}",
                num_images=num_images,
            )
        by_key[key] = row
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("condition rows differ from the exact benchmark")
    recomputed = _aggregate_target_ranges([by_key[key] for key in EXPECTED_CONDITIONS])
    if recomputed != value["target_aggregate_ranges"]:
        raise ValueError("target aggregate ranges do not match condition statistics")
    return by_key


def _format(value: float, digits: int = 4, *, embedded_math: bool = False) -> str:
    if value != 0.0 and abs(value) < 10 ** (-digits):
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10**exponent)
        formatted = rf"{mantissa:.1f}\!\times\!10^{{{exponent}}}"
        return formatted if embedded_math else f"${formatted}$"
    return f"{value:.{digits}f}"


def _range(value: Mapping[str, float], digits: int = 4) -> str:
    lower = _format(value["min"], digits, embedded_math=True)
    upper = _format(value["max"], digits, embedded_math=True)
    return rf"$[{lower},\,{upper}]$"


def _aurc_format(value: float) -> str:
    return _format(100 * value, 3)


def _aurc_range(value: Mapping[str, float]) -> str:
    return _range({"min": 100 * value["min"], "max": 100 * value["max"]}, 3)


def _target_table(by_key: Mapping[tuple[str, str], dict]) -> list[str]:
    lines = [
        r"\textbf{(a) Per-condition boundary-score fidelity}\\[2pt]",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{nHD: M32 vs M128} & \multicolumn{4}{c}{nHD95: M32 vs M128} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"Dataset & Model & Mean/P95 $|\Delta C|$ & $\rho$ & $\tau_b$ & $100|\Delta\mathrm{AURC}|$ & Mean/P95 $|\Delta C|$ & $\rho$ & $\tau_b$ & $100|\Delta\mathrm{AURC}|$ \\",
        r"\midrule",
    ]
    for dataset, condition in ORDERED_TARGET_CONDITIONS:
        row = by_key[(dataset, condition)]
        cells = [DATASET_LABELS[dataset], MODEL_LABELS[condition]]
        for name in ("nhd_m32_vs_m128", "nhd95_m32_vs_m128"):
            comparison = row["comparisons"][name]
            error = comparison["per_image_absolute_score_error"]
            rank = comparison["rank_agreement"]
            cells.extend(
                [
                    f"{_format(error['mean'])}/{_format(error['p95'])}",
                    _format(rank["spearman_rho"]["value"], 3),
                    _format(rank["kendall_tau_b"]["value"], 3),
                    _aurc_format(comparison["matched_risk_aurc"]["absolute_gap"]),
                ]
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\vspace{5pt}"])
    return lines


def _range_table(ranges: Mapping[str, Any]) -> list[str]:
    lines = [
        r"\textbf{(b) Minimum--maximum ranges across the ten target conditions}\\[2pt]",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Comparison & Mean $|\Delta C|$ & Median $|\Delta C|$ & P95 $|\Delta C|$ & Max $|\Delta C|$ & Spearman $\rho$ & Kendall $\tau_b$ & $100|\Delta\mathrm{AURC}|$ \\",
        r"\midrule",
    ]
    labels = {
        "dice_m128_vs_exact": "Dice: M128 vs Exact",
        "nhd_m32_vs_m128": "nHD: M32 vs M128",
        "nhd95_m32_vs_m128": "nHD95: M32 vs M128",
    }
    for spec in COMPARISONS:
        row = ranges[spec.name]
        errors = row["per_image_absolute_score_error"]
        ranks = row["rank_agreement"]
        cells = [labels[spec.name]]
        cells.extend(
            _range(errors[metric]) for metric in ("mean", "median", "p95", "max")
        )
        cells.extend(
            [
                _range(ranks["spearman_rho"], 3),
                _range(ranks["kendall_tau_b"], 3),
                _aurc_range(row["matched_risk_aurc_absolute_gap"]),
            ]
        )
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
        "% Auto-generated by scripts/render_m128_auxiliary.py; do not edit.",
        f"% Source analysis JSON SHA-256: {source_hash}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{High-resolution midpoint fidelity on the ten target-adapted conditions. "
        r"M128 is a numerical reference for nHD and nHD95, not an exact integral; "
        r"only the Dice level-set calculation is exact. Absolute score errors and rank "
        r"agreement are computed per condition on the same frozen maps. AURC gaps are "
        r"displayed on the $\times100$ scale and use "
        r"the matched deployment risk and analytic averaging within score ties. Panel~(a) "
        r"shows the boundary comparisons; panel~(b) summarizes all metrics, including the "
        r"Dice numerical check. Ranges are descriptive and conditions are not independent "
        r"replicates.}",
        r"\label{tab:m128-numerical-reference}",
        r"\begin{minipage}{0.99\textwidth}",
        r"\centering",
    ]
    lines.extend(_target_table(by_key))
    lines.extend(_range_table(value["target_aggregate_ranges"]))
    lines.extend([r"\end{minipage}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_output(tex: str, output_dir: str | os.PathLike[str]) -> Path:
    directory = Path(output_dir)
    destination = directory / OUTPUT_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite rendered M128 table: {destination}"
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
    analysis, source_hash = load_analysis(args.analysis)
    tex = render_analysis(analysis, source_hash=source_hash)
    destination = write_output(tex, args.output_dir)
    print(destination.as_posix())


if __name__ == "__main__":
    main()
