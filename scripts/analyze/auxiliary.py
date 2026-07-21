"""Analyze deployment-threshold and M=128 binary-segmentation runs.

This script is deliberately separate from the primary analysis.  It joins every
auxiliary run to the fixed ``gamma=0.5, M=32`` cohort by ``sample_id`` and fails
closed when a condition, sample, invariant probability-only baseline, or common
M=128 risk/baseline differs.

Example::

    python scripts/analyze/auxiliary.py \
        --primary-root outputs/binary_final \
        --gamma-root 0.3=outputs/binary_thresholds_matched/gamma03 \
        --gamma-root 0.7=outputs/binary_thresholds_matched/gamma07 \
        --m128-root outputs/binary_m128_matched \
        --output-dir outputs/binary_auxiliary_analysis

The JSON artifact records all validated sources and AURC summaries.  The TeX
artifact contains two manuscript-ready tables with methods as rows and the
unpooled dataset--condition pairs as columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.confidence import summarize_aurc  # noqa: E402


SCHEMA_VERSION = 1
JSON_NAME = "auxiliary_analysis.json"
LATEX_NAME = "auxiliary_tables.tex"

RISKS = (
    ("risk_dice", "Dice risk"),
    ("risk_nhd95", "nHD95 risk"),
)
RISK_FIELDS = frozenset(field for field, _ in RISKS)
AUXILIARY_RISK_FIELDS = frozenset({"risk_hd95_pixels"})
BASELINES = (
    ("confidence_sdc", "SDC"),
    ("confidence_mean_max_probability", "Mean max probability"),
    ("confidence_negative_entropy", "Negative entropy"),
)
PROBABILITY_ONLY_BASELINES = frozenset(
    {"confidence_mean_max_probability", "confidence_negative_entropy"}
)
PRIMARY_METHODS = (
    *BASELINES,
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_nhd95_m32", "nHD95-M32"),
)
PRIMARY_SCORE_FIELDS = frozenset(
    field
    for field, _ in (
        *BASELINES,
        ("confidence_dice_m2", "Dice-M2"),
        ("confidence_nhd95_m2", "nHD95-M2"),
        ("confidence_dice_m8", "Dice-M8"),
        ("confidence_nhd95_m8", "nHD95-M8"),
        ("confidence_dice_m32", "Dice-M32"),
        ("confidence_nhd95_m32", "nHD95-M32"),
    )
)
M128_SCORE_FIELDS = frozenset(
    {
        *(field for field, _ in BASELINES),
        "confidence_dice_m128",
        "confidence_nhd95_m128",
    }
)
MATCHED_M_METHODS = {
    "risk_dice": (
        ("confidence_dice_m32", "Dice-M32", "primary"),
        ("confidence_dice_m128", "Dice-M128", "m128"),
    ),
    "risk_nhd95": (
        ("confidence_nhd95_m32", "nHD95-M32", "primary"),
        ("confidence_nhd95_m128", "nHD95-M128", "m128"),
    ),
}

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
        "decision_rule",
        "quadrature",
    }
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "sample_id",
        "image_id",
        "image_index",
        "class_index",
        "class_name",
        "height",
        "width",
        "image_diagonal",
        "truth_foreground_fraction",
        "prediction_foreground_fraction",
    }
)
GAMMA_INVARIANT_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "image_id",
        "image_index",
        "class_index",
        "class_name",
        "height",
        "width",
        "image_diagonal",
        "truth_foreground_fraction",
        *PROBABILITY_ONLY_BASELINES,
    }
)
MANIFEST_INVARIANT_FIELDS = (
    "model",
    "dataset",
    "condition",
    "split",
    "num_images",
    "num_rows",
    "cohort",
    "checkpoint",
    "base_model",
    "source_sha256",
    "preprocessing",
    "losses",
    "void_policy",
    "sdc_empty_convention",
)

DATASET_ORDER = {"pet": 0, "kvasir": 1, "fives": 2}
CONDITION_ORDER = {
    "clipseg-general": 0,
    "clipseg-target": 1,
    "deeplabv3-target": 2,
    "deeplabv3-external": 3,
}
DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
}
CONDITION_LABELS = {
    "clipseg-general": "CLIP-G",
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
    "deeplabv3-external": "DL-E",
}


@dataclass(frozen=True)
class RunData:
    records_path: Path
    manifest_path: Path
    manifest: dict
    rows: dict[str, dict]
    row_fields: frozenset[str]

    @property
    def key(self) -> tuple[str, str]:
        return self.manifest["dataset"], self.manifest["condition"]

    @property
    def gamma(self) -> float:
        return float(self.manifest["decision_rule"]["gamma"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-root",
        default="outputs/binary_final",
        help="gamma=0.5 root containing the primary M=2,8,32 runs",
    )
    parser.add_argument(
        "--gamma-root",
        action="append",
        required=True,
        metavar="GAMMA=PATH",
        help="auxiliary deployment-threshold root; repeat once per gamma",
    )
    parser.add_argument(
        "--m128-root",
        required=True,
        help="gamma=0.5 root containing the matched M=128 runs",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/binary_auxiliary_analysis",
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


def _nonempty_string(value, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _positive_integer(value, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _field_set(manifest, name, *, location):
    value = manifest[name]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(field, str) and field for field in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{location}.{name} must be a nonempty unique field list")
    return frozenset(value)


def _validate_digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _validate_quadrature(manifest, expected_m_values, *, location):
    quadrature = manifest["quadrature"]
    if not isinstance(quadrature, dict):
        raise ValueError(f"{location}.quadrature must be an object")
    expected_keys = {str(count) for count in expected_m_values}
    if set(quadrature) != expected_keys:
        raise ValueError(
            f"{location}.quadrature must contain exactly {sorted(expected_keys)}"
        )
    for count in expected_m_values:
        rule = quadrature[str(count)]
        if not isinstance(rule, dict) or set(rule) != {"rule", "nodes", "weights"}:
            raise ValueError(f"{location}.quadrature.{count} has invalid schema")
        expected_nodes = [(index + 0.5) / count for index in range(count)]
        expected_weights = [1 / count] * count
        if rule["rule"] != "midpoint":
            raise ValueError(f"{location}.quadrature.{count} must use midpoint")
        if rule["nodes"] != expected_nodes or rule["weights"] != expected_weights:
            raise ValueError(
                f"{location}.quadrature.{count} has unexpected nodes or weights"
            )


def load_run(
    records_path,
    *,
    expected_score_fields,
    expected_m_values,
    expected_gamma=None,
) -> RunData:
    """Load and strictly validate one canonical records/manifest pair."""

    records_path = Path(records_path)
    if records_path.name != "records.jsonl" or not records_path.is_file():
        raise FileNotFoundError(
            f"canonical records.jsonl does not exist: {records_path}"
        )
    manifest_path = records_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"matching manifest.json does not exist: {manifest_path}"
        )

    manifest = _loads_strict(manifest_path.read_text(), source=str(manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain one JSON object: {manifest_path}")
    _assert_finite_tree(manifest, location=str(manifest_path))
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"{manifest_path} is missing fields {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION or isinstance(
        manifest["schema_version"], bool
    ):
        raise ValueError(f"{manifest_path}.schema_version must equal {SCHEMA_VERSION}")
    for field in ("run_id", "condition", "dataset", "split"):
        _nonempty_string(manifest[field], location=f"{manifest_path}.{field}")
    num_rows = _positive_integer(
        manifest["num_rows"], location=f"{manifest_path}.num_rows"
    )
    num_images = _positive_integer(
        manifest["num_images"], location=f"{manifest_path}.num_images"
    )
    if num_rows != num_images:
        raise ValueError(f"{manifest_path} must declare one row per image")

    expected_hash = _validate_digest(
        manifest["jsonl_sha256"], location=f"{manifest_path}.jsonl_sha256"
    )
    if _sha256(records_path) != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {records_path}")
    expected_sample_hash = _validate_digest(
        manifest["sample_id_sha256"],
        location=f"{manifest_path}.sample_id_sha256",
    )

    score_fields = _field_set(manifest, "score_fields", location=manifest_path)
    risk_fields = _field_set(manifest, "risk_fields", location=manifest_path)
    auxiliary_fields = _field_set(manifest, "auxiliary_fields", location=manifest_path)
    if score_fields != frozenset(expected_score_fields):
        raise ValueError(
            f"{manifest_path}.score_fields must equal "
            f"{sorted(expected_score_fields)}; got {sorted(score_fields)}"
        )
    if risk_fields != RISK_FIELDS or auxiliary_fields != AUXILIARY_RISK_FIELDS:
        raise ValueError(f"{manifest_path} has an unexpected risk schema")

    decision_rule = manifest["decision_rule"]
    if not isinstance(decision_rule, dict):
        raise ValueError(f"{manifest_path}.decision_rule must be an object")
    if decision_rule.get("form") != "foreground_probability >= gamma":
        raise ValueError(f"{manifest_path} has an unexpected decision-rule form")
    gamma = decision_rule.get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError(f"{manifest_path}.decision_rule.gamma must be numeric")
    gamma = float(gamma)
    if not 0 < gamma < 1:
        raise ValueError(f"{manifest_path}.decision_rule.gamma must lie in (0, 1)")
    if expected_gamma is not None and gamma != float(expected_gamma):
        raise ValueError(
            f"{manifest_path} declares gamma={gamma}, expected {expected_gamma}"
        )
    _validate_quadrature(manifest, expected_m_values, location=manifest_path)

    rows: dict[str, dict] = {}
    row_fields = None
    ordered_sample_ids = []
    image_ids = set()
    with records_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {records_path}:{line_number}")
            row = _loads_strict(line, source=f"{records_path}:{line_number}")
            if not isinstance(row, dict):
                raise ValueError(f"{records_path}:{line_number} is not an object")
            _assert_finite_tree(row, location=f"{records_path}:{line_number}")
            current_fields = frozenset(row)
            if row_fields is None:
                row_fields = current_fields
                required = (
                    REQUIRED_ROW_FIELDS | score_fields | risk_fields | auxiliary_fields
                )
                if not required.issubset(row_fields):
                    raise ValueError(
                        f"{records_path} rows lack fields {sorted(required - row_fields)}"
                    )
                row_scores = {
                    field for field in row_fields if field.startswith("confidence_")
                }
                row_risks = {field for field in row_fields if field.startswith("risk_")}
                if (
                    row_scores != score_fields
                    or row_risks != risk_fields | auxiliary_fields
                ):
                    raise ValueError(f"{records_path} manifest/row schema mismatch")
            elif current_fields != row_fields:
                raise ValueError(
                    f"inconsistent row schema at {records_path}:{line_number}"
                )

            sample_id = _nonempty_string(
                row.get("sample_id"),
                location=f"{records_path}:{line_number}.sample_id",
            )
            image_id = _nonempty_string(
                row.get("image_id"),
                location=f"{records_path}:{line_number}.image_id",
            )
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {records_path}")
            if image_id in image_ids:
                raise ValueError(f"duplicate image_id {image_id!r} in {records_path}")
            if row["schema_version"] != manifest["schema_version"]:
                raise ValueError(f"schema version mismatch in {records_path}")
            if row["run_id"] != manifest["run_id"]:
                raise ValueError(f"run_id mismatch in {records_path}")
            for field in score_fields | risk_fields | auxiliary_fields:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"{records_path}:{line_number}.{field} must be finite numeric"
                    )
            for field in risk_fields:
                if not 0 <= row[field] <= 1:
                    raise ValueError(
                        f"{records_path}:{line_number}.{field} must lie in [0, 1]"
                    )
            if row["risk_hd95_pixels"] < 0:
                raise ValueError(
                    f"{records_path}:{line_number}.risk_hd95_pixels must be nonnegative"
                )
            rows[sample_id] = row
            ordered_sample_ids.append(sample_id)
            image_ids.add(image_id)

    if len(rows) != num_rows:
        raise ValueError(
            f"row-count mismatch for {records_path}: manifest={num_rows}, "
            f"actual={len(rows)}"
        )
    actual_sample_hash = hashlib.sha256(
        "\n".join(ordered_sample_ids).encode("utf-8")
    ).hexdigest()
    if actual_sample_hash != expected_sample_hash:
        raise ValueError(f"sample_id_sha256 mismatch for {records_path}")
    assert row_fields is not None
    return RunData(
        records_path=records_path,
        manifest_path=manifest_path,
        manifest=manifest,
        rows=rows,
        row_fields=row_fields,
    )


def load_root(
    root,
    *,
    expected_score_fields,
    expected_m_values,
    expected_gamma=None,
):
    """Load one experiment root and reject missing or duplicate conditions."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    records_paths = sorted(root.rglob("records.jsonl"))
    manifest_parents = {path.parent for path in root.rglob("manifest.json")}
    records_parents = {path.parent for path in records_paths}
    if manifest_parents != records_parents:
        missing_records = sorted(
            str(path) for path in manifest_parents - records_parents
        )
        missing_manifests = sorted(
            str(path) for path in records_parents - manifest_parents
        )
        raise ValueError(
            f"incomplete records/manifest pairs under {root}: "
            f"missing_records={missing_records}, missing_manifests={missing_manifests}"
        )
    if not records_paths:
        raise ValueError(f"no completed records.jsonl runs under {root}")

    result = {}
    for records_path in records_paths:
        run = load_run(
            records_path,
            expected_score_fields=expected_score_fields,
            expected_m_values=expected_m_values,
            expected_gamma=expected_gamma,
        )
        if run.key in result:
            raise ValueError(
                f"duplicate dataset/condition {run.key!r} under experiment root {root}"
            )
        result[run.key] = run
    return result


def _parse_gamma_roots(specifications):
    result = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"--gamma-root must have GAMMA=PATH form, got {specification!r}"
            )
        gamma_text, path_text = specification.split("=", 1)
        try:
            gamma = float(gamma_text)
        except ValueError as error:
            raise ValueError(f"invalid gamma in {specification!r}") from error
        if not math.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError(f"gamma must lie in (0, 1), got {gamma_text!r}")
        if not path_text:
            raise ValueError(f"gamma root path is empty in {specification!r}")
        if gamma in result:
            raise ValueError(f"duplicate --gamma-root for gamma={gamma}")
        result[gamma] = Path(path_text)
    return result


def _condition_sort_key(key):
    dataset, condition = key
    return (
        DATASET_ORDER.get(dataset, len(DATASET_ORDER)),
        dataset,
        CONDITION_ORDER.get(condition, len(CONDITION_ORDER)),
        condition,
    )


def _compare_manifest_invariants(reference, candidate, *, context):
    for field in MANIFEST_INVARIANT_FIELDS:
        if field not in reference.manifest or field not in candidate.manifest:
            raise ValueError(f"{context}: manifest invariant {field!r} is missing")
        if reference.manifest[field] != candidate.manifest[field]:
            raise ValueError(f"{context}: manifest invariant {field!r} differs")


def _strict_join(reference, candidate, *, exact_fields, context):
    reference_ids = set(reference.rows)
    candidate_ids = set(candidate.rows)
    if reference_ids != candidate_ids:
        missing = sorted(reference_ids - candidate_ids)
        extra = sorted(candidate_ids - reference_ids)
        raise ValueError(
            f"{context}: sample_id join mismatch; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    for sample_id in sorted(reference_ids):
        reference_row = reference.rows[sample_id]
        candidate_row = candidate.rows[sample_id]
        for field in exact_fields:
            if field not in reference_row or field not in candidate_row:
                raise ValueError(
                    f"{context}: join field {field!r} is missing for {sample_id!r}"
                )
            if reference_row[field] != candidate_row[field]:
                raise ValueError(
                    f"{context}: exact join field {field!r} differs for "
                    f"sample_id={sample_id!r}"
                )


def _require_same_condition_set(reference, candidate, *, context):
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys, key=_condition_sort_key)
        extra = sorted(candidate_keys - reference_keys, key=_condition_sort_key)
        raise ValueError(
            f"{context}: condition-set mismatch; missing={missing}, extra={extra}"
        )


def _summary(run, score_field, risk_field, label):
    sample_ids = sorted(run.rows)
    summary = summarize_aurc(
        [run.rows[sample_id][score_field] for sample_id in sample_ids],
        [run.rows[sample_id][risk_field] for sample_id in sample_ids],
    )
    result = asdict(summary)
    result["label"] = label
    return result


def analyze_auxiliary(primary_root, gamma_roots, m128_root):
    """Validate, join, and summarize the two auxiliary experiments."""

    primary = load_root(
        primary_root,
        expected_score_fields=PRIMARY_SCORE_FIELDS,
        expected_m_values=(2, 8, 32),
        expected_gamma=0.5,
    )
    gamma_runs = {0.5: primary}
    for gamma, root in sorted(gamma_roots.items()):
        if gamma == 0.5:
            raise ValueError(
                "gamma=0.5 is supplied by --primary-root and cannot repeat"
            )
        runs = load_root(
            root,
            expected_score_fields=PRIMARY_SCORE_FIELDS,
            expected_m_values=(2, 8, 32),
            expected_gamma=gamma,
        )
        _require_same_condition_set(
            primary, runs, context=f"deployment-threshold gamma={gamma}"
        )
        gamma_runs[gamma] = runs

    m128 = load_root(
        m128_root,
        expected_score_fields=M128_SCORE_FIELDS,
        expected_m_values=(128,),
        expected_gamma=0.5,
    )
    _require_same_condition_set(primary, m128, context="M=128 experiment")

    condition_keys = sorted(primary, key=_condition_sort_key)
    gamma_values = sorted(gamma_runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "tie_policy": "analytic expectation over random within-tie order",
            "primary_gamma": 0.5,
            "gamma_values": gamma_values,
            "gamma_join": (
                "exact sample_id join; manifest/model invariants, sample metadata, "
                "and probability-only baselines are equal"
            ),
            "m128_join": (
                "exact sample_id join; all common row fields except run_id, "
                "including both risks and all three baselines, are equal"
            ),
            "gamma_invariant_row_fields": sorted(GAMMA_INVARIANT_ROW_FIELDS),
            "m128_required_common_fields": sorted(
                RISK_FIELDS | AUXILIARY_RISK_FIELDS | {field for field, _ in BASELINES}
            ),
            "condition_order": [list(key) for key in condition_keys],
        },
        "conditions": [],
    }

    for key in condition_keys:
        reference = primary[key]
        condition = {
            "dataset": key[0],
            "condition": key[1],
            "split": reference.manifest["split"],
            "num_rows": len(reference.rows),
            "sources": {
                "gamma": {},
                "m128": {
                    "records": str(m128[key].records_path),
                    "manifest": str(m128[key].manifest_path),
                    "run_id": m128[key].manifest["run_id"],
                    "jsonl_sha256": m128[key].manifest["jsonl_sha256"],
                },
            },
            "threshold_robustness": {},
            "m128_ablation": {},
        }

        for gamma in gamma_values:
            candidate = gamma_runs[gamma][key]
            _strict_join(
                reference,
                candidate,
                exact_fields=GAMMA_INVARIANT_ROW_FIELDS,
                context=f"gamma={gamma} invariant join, condition={key}",
            )
            _compare_manifest_invariants(
                reference,
                candidate,
                context=f"gamma={gamma}, condition={key}",
            )
            gamma_key = format(gamma, ".12g")
            condition["sources"]["gamma"][gamma_key] = {
                "records": str(candidate.records_path),
                "manifest": str(candidate.manifest_path),
                "run_id": candidate.manifest["run_id"],
                "jsonl_sha256": candidate.manifest["jsonl_sha256"],
            }
            gamma_result = {}
            for risk_field, risk_label in RISKS:
                gamma_result[risk_field] = {
                    "label": risk_label,
                    "methods": {
                        score_field: _summary(
                            candidate, score_field, risk_field, method_label
                        )
                        for score_field, method_label in PRIMARY_METHODS
                    },
                }
            condition["threshold_robustness"][gamma_key] = gamma_result

        m128_candidate = m128[key]
        common_fields = (reference.row_fields & m128_candidate.row_fields) - {"run_id"}
        required_common = (
            RISK_FIELDS | AUXILIARY_RISK_FIELDS | {field for field, _ in BASELINES}
        )
        if not required_common.issubset(common_fields):
            raise ValueError(
                f"M=128, condition={key}: missing required common fields "
                f"{sorted(required_common - common_fields)}"
            )
        _strict_join(
            reference,
            m128_candidate,
            exact_fields=common_fields,
            context=f"M=128 common-field join, condition={key}",
        )
        _compare_manifest_invariants(
            reference, m128_candidate, context=f"M=128, condition={key}"
        )
        for risk_field, risk_label in RISKS:
            methods = {}
            for score_field, label, source in MATCHED_M_METHODS[risk_field]:
                run = reference if source == "primary" else m128_candidate
                methods[score_field] = _summary(run, score_field, risk_field, label)
            m32_field, m128_field = [
                field for field, _, _ in MATCHED_M_METHODS[risk_field]
            ]
            condition["m128_ablation"][risk_field] = {
                "label": risk_label,
                "methods": methods,
                "aurc_difference_m128_minus_m32": (
                    methods[m128_field]["aurc"] - methods[m32_field]["aurc"]
                ),
            }
        result["conditions"].append(condition)

    return result


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


def _latex_aurc(value):
    """Render an AURC-derived value on the manuscript-only x100 scale."""

    return "--" if value is None else f"{100 * value:.4f}"


def _header_lines(conditions, *, first_column):
    count = len(conditions)
    lines = [rf"\begin{{tabular}}{{l*{{{count}}}{{c}}}}", r"\toprule"]
    groups = []
    start = 2
    position = 0
    while position < count:
        dataset = conditions[position]["dataset"]
        end = position + 1
        while end < count and conditions[end]["dataset"] == dataset:
            end += 1
        groups.append((dataset, start, start + end - position - 1, end - position))
        start += end - position
        position = end
    group_cells = [""] + [
        rf"\multicolumn{{{width}}}{{c}}{{{_latex_escape(DATASET_LABELS.get(dataset, dataset))}}}"
        for dataset, _, _, width in groups
    ]
    lines.append(" & ".join(group_cells) + r" \\")
    lines.append(
        "".join(rf"\cmidrule(lr){{{left}-{right}}}" for _, left, right, _ in groups)
    )
    labels = [
        _latex_escape(CONDITION_LABELS.get(item["condition"], item["condition"]))
        for item in conditions
    ]
    lines.append(" & ".join([first_column, *labels]) + r" \\")
    lines.append(r"\midrule")
    return lines


def _write_threshold_table(result):
    conditions = result["conditions"]
    gamma_keys = [format(value, ".12g") for value in result["analysis"]["gamma_values"]]
    gamma_order = "/".join(gamma_keys)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Deployment-threshold robustness. Each cell lists tie-aware "
            rf"$100\times\mathrm{{AURC}}$ from top to bottom for $\gamma={gamma_order}$; lower "
            r"is better. This is a display-only transformation; the JSON retains raw "
            r"AURC. Methods are rows and columns are unpooled "
            r"dataset--condition pairs. The probability maps and held-out cohorts "
            r"are exactly joined across thresholds.}"
        ),
        r"\label{tab:threshold-robustness}",
        r"{\scriptsize\setlength{\tabcolsep}{2pt}%",
        r"\resizebox{\textwidth}{!}{%",
        *_header_lines(conditions, first_column="Confidence method"),
    ]
    for risk_position, (risk_field, risk_label) in enumerate(RISKS):
        if risk_position:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{len(conditions) + 1}}}{{l}}{{\textit{{{risk_label}; "
            r"displayed as $100\times\mathrm{AURC}$}} \\"
        )
        for score_field, method_label in PRIMARY_METHODS:
            cells = []
            for condition in conditions:
                values = [
                    condition["threshold_robustness"][gamma][risk_field]["methods"][
                        score_field
                    ]["aurc"]
                    for gamma in gamma_keys
                ]
                cells.append(
                    r"\shortstack{" + r"\\".join(map(_latex_aurc, values)) + "}"
                )
            lines.append(" & ".join([method_label, *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", "}", r"\end{table*}"])
    return lines


def _write_m128_table(result):
    conditions = result["conditions"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{High-resolution midpoint check. Entries are tie-aware "
            r"$100\times\mathrm{AURC}$ (lower is better). This is a display-only "
            r"transformation; the JSON retains raw AURC. Methods are rows and columns are unpooled "
            r"dataset--condition pairs. Within every column, $M=32$ and $M=128$ "
            r"use exactly joined samples, actions, risks, and baseline scores.}"
        ),
        r"\label{tab:m128-ablation}",
        r"{\scriptsize\setlength{\tabcolsep}{3pt}%",
        r"\resizebox{\textwidth}{!}{%",
        *_header_lines(conditions, first_column="Loss-indexed method"),
    ]
    for risk_position, (risk_field, risk_label) in enumerate(RISKS):
        if risk_position:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{len(conditions) + 1}}}{{l}}{{\textit{{{risk_label}; "
            r"displayed as $100\times\mathrm{AURC}$}} \\"
        )
        for score_field, method_label, _ in MATCHED_M_METHODS[risk_field]:
            cells = [
                _latex_aurc(
                    condition["m128_ablation"][risk_field]["methods"][score_field][
                        "aurc"
                    ]
                )
                for condition in conditions
            ]
            lines.append(" & ".join([method_label, *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", "}", r"\end{table*}"])
    return lines


def write_latex(result, path, *, json_sha256):
    lines = [
        "% AUTO-GENERATED by scripts/analyze/auxiliary.py; DO NOT EDIT.",
        f"% Source auxiliary_analysis.json SHA-256: {json_sha256}",
        *_write_threshold_table(result),
        "",
        *_write_m128_table(result),
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    gamma_roots = _parse_gamma_roots(args.gamma_root)
    result = analyze_auxiliary(args.primary_root, gamma_roots, args.m128_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_NAME
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    latex_path = output_dir / LATEX_NAME
    write_latex(result, latex_path, json_sha256=_sha256(json_path))
    print(f"saved {json_path}")
    print(f"saved {latex_path}")


if __name__ == "__main__":
    main()
