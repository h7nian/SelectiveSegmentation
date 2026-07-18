"""Strict analysis for the loss-indexed binary segmentation experiment.

The script consumes focused ``*.jsonl`` files written by
``selectseg.binary_eval`` and the matching ``*.manifest.json`` files.  It does
not import or reuse the legacy multiclass analysis.

Example::

    python scripts/analyze_binary.py --input-dir outputs/binary \
        --output-dir outputs/binary/analysis

Use ``--inputs`` to analyze an explicit, reproducible list of JSONL files.
The outputs are a machine-readable JSON summary, a long-form CSV, and a LaTeX
main table.  All AURCs use analytic random ordering within exact confidence
ties.  The predeclared Dice-M32 versus nHD95-M32 comparisons report paired
image-cluster percentile-bootstrap intervals and two-sided bootstrap p-values
computed from the same resamples. Holm adjustment is applied to those p-values
without making significance calls.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.binary_framework import (  # noqa: E402
    summarize_aurc,
    tie_aware_expected_aurc,
)


SUPPORTED_SCHEMA_VERSION = 1
JSON_NAME = "analysis.json"
CSV_NAME = "main_table.csv"
LATEX_NAME = "main_table.tex"

METHODS = (
    ("confidence_sdc", "SDC"),
    ("confidence_mean_max_probability", "Mean max probability"),
    ("confidence_negative_entropy", "Negative entropy"),
    ("confidence_dice_m2", "Dice-M2"),
    ("confidence_dice_m8", "Dice-M8"),
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_nhd95_m2", "nHD95-M2"),
    ("confidence_nhd95_m8", "nHD95-M8"),
    ("confidence_nhd95_m32", "nHD95-M32"),
)
RISKS = (
    ("risk_dice", "Dice risk"),
    ("risk_nhd95", "Normalized penalized HD95 risk"),
)
METHOD_LABELS = dict(METHODS)
RISK_LABELS = dict(RISKS)
REQUIRED_SCORE_FIELDS = frozenset(METHOD_LABELS)
REQUIRED_RISK_FIELDS = frozenset(RISK_LABELS)
REQUIRED_ROW_FIELDS = frozenset(
    {"schema_version", "run_id", "sample_id", "image_id", "risk_hd95_pixels"}
)
AUXILIARY_RISK_FIELDS = frozenset({"risk_hd95_pixels"})
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "condition",
        "dataset",
        "split",
        "num_images",
        "num_rows",
        "jsonl_sha256",
        "sample_id_sha256",
        "risk_fields",
        "auxiliary_fields",
        "score_fields",
    }
)


@dataclass(frozen=True)
class ConditionData:
    jsonl_path: Path
    manifest_path: Path
    manifest: dict
    rows: tuple[dict, ...]

    @property
    def condition(self):
        return self.manifest["condition"]

    @property
    def dataset(self):
        return self.manifest["dataset"]


@dataclass(frozen=True)
class PairedBootstrapResult:
    difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    p_value: float
    n_resamples: int
    n_observations: int
    n_clusters: int
    seed: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="outputs/binary",
        help="directory recursively searched for *.jsonl when --inputs is omitted",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="explicit JSONL files; each requires a matching .manifest.json",
    )
    parser.add_argument("--output-dir", default="outputs/binary/analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text: str, *, source: str):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite_tree(value, *, location: str):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, location=f"{location}[{index}]")


def _required_string(mapping, field, *, location):
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string")
    return value


def _field_list(manifest, name, *, location):
    value = manifest.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{name} must be a non-empty list")
    if not all(isinstance(field, str) and field for field in value):
        raise ValueError(f"{location}.{name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{location}.{name} contains duplicate fields")
    return set(value)


def manifest_path_for(jsonl_path: Path) -> Path:
    jsonl_path = Path(jsonl_path)
    if jsonl_path.name == "records.jsonl":
        return jsonl_path.parent / "manifest.json"
    return jsonl_path.with_suffix(".manifest.json")


def load_condition(jsonl_path) -> ConditionData:
    """Load and fully validate one JSONL/manifest pair."""

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL input does not exist: {jsonl_path}")
    manifest_path = manifest_path_for(jsonl_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"matching manifest does not exist: {manifest_path}")

    manifest = _loads_strict(
        manifest_path.read_text(), source=str(manifest_path)
    )
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain one JSON object: {manifest_path}")
    _assert_finite_tree(manifest, location=str(manifest_path))
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"manifest {manifest_path} is missing required fields {missing}")
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported schema_version {manifest['schema_version']!r} in "
            f"{manifest_path}; expected {SUPPORTED_SCHEMA_VERSION}"
        )
    for field in (
        "run_id",
        "condition",
        "dataset",
        "split",
        "jsonl_sha256",
        "sample_id_sha256",
    ):
        _required_string(manifest, field, location=str(manifest_path))
    for count_field in ("num_rows", "num_images"):
        if isinstance(manifest[count_field], bool) or not isinstance(
            manifest[count_field], int
        ):
            raise ValueError(
                f"{manifest_path}.{count_field} must be a positive integer"
            )
        if manifest[count_field] <= 0:
            raise ValueError(
                f"{manifest_path}.{count_field} must be a positive integer"
            )
    if manifest["num_rows"] != manifest["num_images"]:
        raise ValueError(
            f"native binary runs require one row per image, but {manifest_path} "
            f"declares num_rows={manifest['num_rows']} and "
            f"num_images={manifest['num_images']}"
        )

    expected_hash = manifest["jsonl_sha256"].lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise ValueError(f"{manifest_path}.jsonl_sha256 is not a SHA-256 hex digest")
    expected_sample_hash = manifest["sample_id_sha256"].lower()
    if len(expected_sample_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sample_hash
    ):
        raise ValueError(
            f"{manifest_path}.sample_id_sha256 is not a SHA-256 hex digest"
        )
    actual_hash = _sha256(jsonl_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {jsonl_path}: manifest={expected_hash}, "
            f"actual={actual_hash}"
        )

    score_fields = _field_list(manifest, "score_fields", location=str(manifest_path))
    risk_fields = _field_list(manifest, "risk_fields", location=str(manifest_path))
    auxiliary_fields = _field_list(
        manifest, "auxiliary_fields", location=str(manifest_path)
    )
    missing_scores = sorted(REQUIRED_SCORE_FIELDS - score_fields)
    missing_risks = sorted(REQUIRED_RISK_FIELDS - risk_fields)
    if missing_scores or missing_risks:
        raise ValueError(
            f"manifest {manifest_path} lacks required score/risk fields: "
            f"scores={missing_scores}, risks={missing_risks}"
        )
    if risk_fields != REQUIRED_RISK_FIELDS:
        raise ValueError(
            f"manifest {manifest_path} must list exactly the two main risks "
            f"{sorted(REQUIRED_RISK_FIELDS)}; got {sorted(risk_fields)}"
        )
    if auxiliary_fields != AUXILIARY_RISK_FIELDS:
        raise ValueError(
            f"manifest {manifest_path} must list exactly the auxiliary fields "
            f"{sorted(AUXILIARY_RISK_FIELDS)}; got {sorted(auxiliary_fields)}"
        )

    rows = []
    with jsonl_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {jsonl_path}:{line_number}")
            row = _loads_strict(line, source=f"{jsonl_path}:{line_number}")
            if not isinstance(row, dict):
                raise ValueError(f"row {jsonl_path}:{line_number} is not an object")
            _assert_finite_tree(row, location=f"{jsonl_path}:{line_number}")
            rows.append(row)
    if len(rows) != manifest["num_rows"]:
        raise ValueError(
            f"row-count mismatch for {jsonl_path}: manifest={manifest['num_rows']}, "
            f"actual={len(rows)}"
        )
    if not rows:
        raise ValueError(f"JSONL input has no rows: {jsonl_path}")

    row_fields = set(rows[0])
    missing_row_fields = sorted(
        (REQUIRED_ROW_FIELDS | score_fields | risk_fields) - row_fields
    )
    if missing_row_fields:
        raise ValueError(f"{jsonl_path} rows lack required fields {missing_row_fields}")
    manifested_scores = {field for field in row_fields if field.startswith("confidence_")}
    row_risks = {field for field in row_fields if field.startswith("risk_")}
    unregistered_risks = row_risks - risk_fields - AUXILIARY_RISK_FIELDS
    if (
        manifested_scores != score_fields
        or not risk_fields.issubset(row_risks)
        or unregistered_risks
    ):
        raise ValueError(
            f"manifest/row score-risk schema mismatch in {jsonl_path}: "
            f"score_fields={sorted(score_fields)} row_scores={sorted(manifested_scores)}; "
            f"risk_fields={sorted(risk_fields)} row_risks={sorted(row_risks)}"
        )

    sample_ids = set()
    ordered_sample_ids = []
    image_ids = set()
    for line_number, row in enumerate(rows, start=1):
        if set(row) != row_fields:
            raise ValueError(f"inconsistent row schema at {jsonl_path}:{line_number}")
        sample_id = _required_string(
            row, "sample_id", location=f"{jsonl_path}:{line_number}"
        )
        _required_string(row, "image_id", location=f"{jsonl_path}:{line_number}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {jsonl_path}")
        sample_ids.add(sample_id)
        ordered_sample_ids.append(sample_id)
        image_id = row["image_id"]
        if image_id in image_ids:
            raise ValueError(f"duplicate image_id {image_id!r} in {jsonl_path}")
        image_ids.add(image_id)
        if row.get("schema_version") != manifest["schema_version"]:
            raise ValueError(
                f"schema_version mismatch at {jsonl_path}:{line_number}"
            )
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"run_id mismatch at {jsonl_path}:{line_number}")
        for field in score_fields | risk_fields | AUXILIARY_RISK_FIELDS:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{jsonl_path}:{line_number}.{field} must be numeric"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"{jsonl_path}:{line_number}.{field} must be finite"
                )
        for field in risk_fields:
            if not 0 <= row[field] <= 1:
                raise ValueError(
                    f"{jsonl_path}:{line_number}.{field} must lie in [0, 1]"
                )
        if row["risk_hd95_pixels"] < 0:
            raise ValueError(
                f"{jsonl_path}:{line_number}.risk_hd95_pixels must be non-negative"
            )

    sample_id_hash = hashlib.sha256(
        "\n".join(ordered_sample_ids).encode("utf-8")
    ).hexdigest()
    if sample_id_hash != expected_sample_hash:
        raise ValueError(
            f"sample_id_sha256 mismatch for {jsonl_path}: "
            f"manifest={manifest['sample_id_sha256']}, actual={sample_id_hash}"
        )

    return ConditionData(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        manifest=manifest,
        rows=tuple(rows),
    )


def holm_adjust(p_values):
    """Holm step-down family-wise-error adjustment, in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    factors = values.size - np.arange(values.size)
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(ranked * factors))
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def _cluster_groups(cluster_ids, count):
    ids = list(cluster_ids)
    if len(ids) != count:
        raise ValueError("cluster_ids must match the number of observations")
    locations = {}
    for index, key in enumerate(ids):
        if key is None:
            raise ValueError("cluster_ids cannot contain missing values")
        try:
            hash(key)
        except TypeError as error:
            raise TypeError("cluster_ids must be hashable") from error
        locations.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=int) for indices in locations.values()]


def paired_cluster_bootstrap_aurc_test(
    left_confidences,
    right_confidences,
    risks,
    *,
    cluster_ids,
    n_resamples=10_000,
    confidence_level=0.95,
    seed=0,
):
    """Paired cluster bootstrap interval and equal-tail p-value for AURC.

    Each resample draws image clusters with replacement and uses the identical
    sampled rows for both scores and the risk. The percentile interval and the
    two-sided p-value are computed from that one array of AURC differences.
    For ``B`` draws, the p-value is::

        min(1, 2 * min((1 + #{delta* <= 0}) / (B + 1),
                       (1 + #{delta* >= 0}) / (B + 1)))

    This confidence-curve p-value is aligned with the equal-tail percentile
    interval. Both are invariant to separate strictly increasing transforms
    of the two confidence scores because every statistic is rank based.
    """

    left = np.asarray(left_confidences, dtype=float)
    right = np.asarray(right_confidences, dtype=float)
    risk = np.asarray(risks, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or risk.ndim != 1:
        raise ValueError("confidences and risks must be one-dimensional")
    if not left.size or left.size != right.size or left.size != risk.size:
        raise ValueError("paired confidences and risks must have one non-empty length")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(risk).all():
        raise ValueError("confidences and risks must be finite")
    if isinstance(n_resamples, bool) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise TypeError("n_resamples must be a positive integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    groups = _cluster_groups(cluster_ids, left.size)
    singleton_clusters = len(groups) == left.size and all(
        group.size == 1 for group in groups
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for replicate in range(n_resamples):
        if singleton_clusters:
            indices = rng.integers(0, len(groups), size=len(groups))
        else:
            sampled = rng.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[position] for position in sampled])
        draws[replicate] = tie_aware_expected_aurc(
            left[indices], risk[indices]
        ) - tie_aware_expected_aurc(right[indices], risk[indices])

    tail = (1 - confidence_level) / 2
    ci_low, ci_high = np.quantile(draws, [tail, 1 - tail])
    lower_tail = (1 + np.count_nonzero(draws <= 0)) / (n_resamples + 1)
    upper_tail = (1 + np.count_nonzero(draws >= 0)) / (n_resamples + 1)
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))
    observed = tie_aware_expected_aurc(left, risk) - tie_aware_expected_aurc(
        right, risk
    )
    return PairedBootstrapResult(
        difference=float(observed),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=float(confidence_level),
        p_value=float(p_value),
        n_resamples=int(n_resamples),
        n_observations=int(left.size),
        n_clusters=len(groups),
        seed=int(seed),
    )


def _derived_seed(base_seed, *parts):
    payload = "|".join([str(base_seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def analyze_conditions(
    conditions,
    *,
    bootstrap_samples=10_000,
    confidence_level=0.95,
    seed=0,
):
    if not conditions:
        raise ValueError("at least one condition is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    conditions = sorted(
        conditions, key=lambda item: (item.dataset, item.condition, str(item.jsonl_path))
    )
    identifiers = [(item.dataset, item.condition) for item in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("each dataset/condition pair must appear exactly once")

    result = {
        "schema_version": 1,
        "analysis": {
            "tie_policy": "analytic expectation over random within-tie order",
            "normalized_aurc": (
                "(AURC - oracle AURC) / (random AURC - oracle AURC)"
            ),
            "comparison": (
                "confidence_dice_m32 versus confidence_nhd95_m32; reported "
                "difference is AURC(Dice-M32) - AURC(nHD95-M32)"
            ),
            "bootstrap_samples": int(bootstrap_samples),
            "confidence_level": float(confidence_level),
            "bootstrap_p_value": (
                "two-sided equal-tail confidence-curve p-value from the same "
                "paired image-cluster bootstrap draws as the percentile interval"
            ),
            "seed": int(seed),
        },
        "conditions": [],
        "multiple_testing": {},
    }
    comparison_records = []
    for data in conditions:
        rows = data.rows
        image_ids = [row["image_id"] for row in rows]
        condition_result = {
            "condition": data.condition,
            "dataset": data.dataset,
            "split": data.manifest["split"],
            "num_rows": len(rows),
            "num_image_clusters": len(set(image_ids)),
            "jsonl": str(data.jsonl_path),
            "manifest": str(data.manifest_path),
            "jsonl_sha256": data.manifest["jsonl_sha256"],
            "risks": {},
            "comparisons": {},
        }
        for risk_field, risk_label in RISKS:
            risks = np.asarray([row[risk_field] for row in rows], dtype=float)
            risk_result = {
                "label": risk_label,
                "methods": {},
            }
            for score_field, method_label in METHODS:
                confidences = np.asarray(
                    [row[score_field] for row in rows], dtype=float
                )
                summary = summarize_aurc(confidences, risks)
                values = asdict(summary)
                values["label"] = method_label
                risk_result["methods"][score_field] = values
            first = risk_result["methods"][METHODS[0][0]]
            risk_result["oracle_aurc"] = first["oracle_aurc"]
            risk_result["random_aurc"] = first["random_aurc"]
            condition_result["risks"][risk_field] = risk_result

            left_field = "confidence_dice_m32"
            right_field = "confidence_nhd95_m32"
            left = np.asarray([row[left_field] for row in rows], dtype=float)
            right = np.asarray([row[right_field] for row in rows], dtype=float)
            bootstrap_seed = _derived_seed(
                seed, data.dataset, data.condition, risk_field, "bootstrap"
            )
            bootstrap = paired_cluster_bootstrap_aurc_test(
                left,
                right,
                risks,
                cluster_ids=image_ids,
                n_resamples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            comparison = {
                "left": left_field,
                "right": right_field,
                "difference_left_minus_right": bootstrap.difference,
                "bootstrap": asdict(bootstrap),
                "holm_adjusted_p_value": None,
            }
            condition_result["comparisons"][risk_field] = comparison
            comparison_records.append(
                (data.dataset, data.condition, risk_field, comparison)
            )
        result["conditions"].append(condition_result)

    raw_p_values = [
        record["bootstrap"]["p_value"]
        for _, _, _, record in comparison_records
    ]
    adjusted = holm_adjust(raw_p_values)
    hypotheses = []
    for (dataset, condition, risk_field, record), adjusted_p in zip(
        comparison_records, adjusted
    ):
        record["holm_adjusted_p_value"] = adjusted_p
        hypotheses.append(
            {
                "dataset": dataset,
                "condition": condition,
                "risk": risk_field,
                "raw_bootstrap_p_value": record["bootstrap"]["p_value"],
                "holm_adjusted_p_value": adjusted_p,
            }
        )
    result["multiple_testing"] = {
        "procedure": "Holm step-down family-wise-error adjustment",
        "family": (
            "all predeclared Dice-M32 versus nHD95-M32 paired cluster "
            "bootstrap tests across supplied conditions and both risks"
        ),
        "num_hypotheses": len(raw_p_values),
        "raw_bootstrap_p_values": raw_p_values,
        "holm_adjusted_p_values": adjusted,
        "hypotheses": hypotheses,
        "confidence_intervals": (
            "unadjusted equal-tail percentile intervals from the same paired "
            "bootstrap draws; Holm adjustment applies only to the 20 p-values"
        ),
        "significance_calls": "not made by this analysis",
    }
    return result


def _format_csv_number(value):
    return "" if value is None else format(value, ".12g")


def write_csv(result, path):
    path = Path(path)
    columns = [
        "dataset",
        "condition",
        "split",
        "num_rows",
        "num_image_clusters",
        "risk",
        "risk_label",
        "method",
        "method_label",
        "aurc",
        "oracle_aurc",
        "random_aurc",
        "excess_aurc",
        "normalized_aurc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for condition in result["conditions"]:
            for risk_field, _ in RISKS:
                risk = condition["risks"][risk_field]
                for score_field, _ in METHODS:
                    method = risk["methods"][score_field]
                    writer.writerow(
                        {
                            "dataset": condition["dataset"],
                            "condition": condition["condition"],
                            "split": condition["split"],
                            "num_rows": condition["num_rows"],
                            "num_image_clusters": condition["num_image_clusters"],
                            "risk": risk_field,
                            "risk_label": risk["label"],
                            "method": score_field,
                            "method_label": method["label"],
                            "aurc": _format_csv_number(method["aurc"]),
                            "oracle_aurc": _format_csv_number(method["oracle_aurc"]),
                            "random_aurc": _format_csv_number(method["random_aurc"]),
                            "excess_aurc": _format_csv_number(method["excess_aurc"]),
                            "normalized_aurc": _format_csv_number(
                                method["normalized_aurc"]
                            ),
                        }
                    )


def _latex_escape(value):
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


def _latex_number(value):
    return "--" if value is None else f"{value:.4f}"


def write_latex(result, path):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Loss-indexed confidence for binary selective segmentation. "
        r"AURC and excess AURC are in the risk's normalized units; normalized "
        r"AURC is zero for the oracle and one for random ordering.}",
        r"\label{tab:binary-main}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Condition & Confidence & AURC & E-AURC & nAURC \\",
        r"\midrule",
    ]
    for risk_position, (risk_field, risk_label) in enumerate(RISKS):
        if risk_position:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{{_latex_escape(risk_label)}}}}} \\"
        )
        for condition in result["conditions"]:
            risk = condition["risks"][risk_field]
            for score_field, _ in METHODS:
                method = risk["methods"][score_field]
                lines.append(
                    " & ".join(
                        [
                            _latex_escape(condition["dataset"]),
                            _latex_escape(condition["condition"]),
                            _latex_escape(method["label"]),
                            _latex_number(method["aurc"]),
                            _latex_number(method["excess_aurc"]),
                            _latex_number(method["normalized_aurc"]),
                        ]
                    )
                    + r" \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    Path(path).write_text("\n".join(lines))


def write_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_NAME
    csv_path = output_dir / CSV_NAME
    latex_path = output_dir / LATEX_NAME
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_csv(result, csv_path)
    write_latex(result, latex_path)
    return json_path, csv_path, latex_path


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie strictly between 0 and 1")
    if args.inputs is None:
        inputs = sorted(Path(args.input_dir).rglob("*.jsonl"))
    else:
        inputs = sorted({Path(path) for path in args.inputs}, key=str)
    if not inputs:
        raise FileNotFoundError("no binary JSONL inputs were selected")
    conditions = [load_condition(path) for path in inputs]
    result = analyze_conditions(
        conditions,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    paths = write_outputs(result, args.output_dir)
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
