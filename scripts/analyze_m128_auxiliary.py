"""Analyze the locked M=128 numerical-reference campaign.

The canonical boundary scores use 32 uniform midpoint thresholds.  This
analysis compares them with a separately generated 128-midpoint calculation
on the *same frozen probability maps*.  M=128 is a high-resolution numerical
reference, not an exact integral.  Dice-M128 is additionally compared with the
available exact level-set Dice calculation.

Canonical execution requires exactly sixteen explicit M=128 JSONL inputs,
exactly sixteen explicit canonical assembly JSONLs, and the immutable campaign
lock.  Inputs are never selected through a glob or directory walk.  Every
manifest, payload hash, campaign binding, schema, sample identifier, and join
identity is validated before statistics are computed.
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
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata

from scripts.analyze_binary import (
    EXPECTED_CONDITIONS,
    ConditionData,
    load_analysis_campaign_lock,
    load_condition,
    validate_campaign_bound_conditions,
)
from selectseg.binary_framework import tie_aware_expected_aurc
from selectseg.score_binary_m128_auxiliary import (
    AUXILIARY_ARTIFACT_TYPE,
    AUXILIARY_SCHEMA_VERSION,
    IDENTITY_FIELDS,
    M128_SCORE_FIELDS,
    _auxiliary_id,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_m128_numerical_reference_analysis"
M128_ARTIFACT_COUNT = len(EXPECTED_CONDITIONS)
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)


class ComparisonSpec(NamedTuple):
    name: str
    candidate_score: str
    reference_score: str
    matched_risk: str
    label: str
    interpretation: str


COMPARISONS = (
    ComparisonSpec(
        "dice_m128_vs_exact",
        "confidence_dice_m128_aux",
        "confidence_dice_exact",
        "risk_dice",
        "Dice: M128 vs Exact",
        "exact level-set Dice is the numerical reference",
    ),
    ComparisonSpec(
        "nhd_m32_vs_m128",
        "confidence_nhd_m32",
        "confidence_nhd_m128_aux",
        "risk_nhd",
        "nHD: M32 vs M128",
        "M128 is a high-resolution numerical reference, not exact",
    ),
    ComparisonSpec(
        "nhd95_m32_vs_m128",
        "confidence_nhd95_m32",
        "confidence_nhd95_m128_aux",
        "risk_nhd95",
        "nHD95: M32 vs M128",
        "M128 is a high-resolution numerical reference, not exact",
    ),
)
COMPARISON_BY_NAME = {spec.name: spec for spec in COMPARISONS}
EXPECTED_AUXILIARY_ROW_FIELDS = frozenset(IDENTITY_FIELDS) | frozenset(
    M128_SCORE_FIELDS
)
JOIN_FIELDS = (
    "sample_id",
    "image_id",
    "image_index",
    "class_index",
    "class_name",
    "height",
    "width",
)


@dataclass(frozen=True)
class M128ConditionData:
    jsonl_path: Path
    manifest_path: Path
    manifest: dict
    rows: tuple[dict, ...]

    @property
    def dataset(self) -> str:
        return self.manifest["dataset"]

    @property
    def condition(self) -> str:
        return self.manifest["condition"]


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument(
        "--m128-inputs",
        nargs="+",
        required=True,
        metavar="M128_RECORDS_JSONL",
        help="exactly 16 explicit M128 auxiliary records.jsonl paths",
    )
    parser.add_argument(
        "--canonical-inputs",
        nargs="+",
        required=True,
        metavar="CANONICAL_RECORDS_JSONL",
        help="exactly 16 explicit canonical assembly records.jsonl paths",
    )
    parser.add_argument(
        "--output",
        default="outputs/binary_m128_auxiliary_analysis/analysis.json",
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


def _loads_strict(raw: str, *, source: str):
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _assert_finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, location=f"{location}[{index}]")


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{location} must be a SHA-256 digest")
    return value.lower()


def _required_string(mapping: Mapping[str, Any], field: str, *, location: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{field} must be a nonempty string")
    return value


def _positive_integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _finite_score(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 0.0:
        raise ValueError(f"{location} must be finite and lie in [-1, 0]")
    return result


def _portable_path(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "scripts/analyze_binary.py",
        root / "selectseg/binary_framework.py",
        root / "selectseg/score_binary_m128_auxiliary.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_quadrature(manifest: Mapping[str, Any], *, location: str) -> None:
    quadrature = manifest.get("quadrature")
    if not isinstance(quadrature, dict) or set(quadrature) != {"128"}:
        raise ValueError(f"{location}.quadrature must contain exactly M=128")
    rule = quadrature["128"]
    if not isinstance(rule, dict) or set(rule) != {"rule", "nodes", "weights"}:
        raise ValueError(f"{location}.quadrature.128 has an invalid schema")
    if rule["rule"] != "midpoint":
        raise ValueError(f"{location}.quadrature.128.rule must be midpoint")
    nodes = rule["nodes"]
    weights = rule["weights"]
    expected_nodes = [(index + 0.5) / 128 for index in range(128)]
    expected_weights = [1.0 / 128] * 128
    if nodes != expected_nodes or weights != expected_weights:
        raise ValueError(f"{location}.quadrature.128 is not the locked midpoint rule")


def load_m128_condition(path: str | os.PathLike[str]) -> M128ConditionData:
    """Load and fully validate one M=128 JSONL/manifest pair."""

    jsonl_path = Path(path)
    if jsonl_path.name != "records.jsonl" or not jsonl_path.is_file():
        raise FileNotFoundError(f"M128 input must be an existing records.jsonl: {path}")
    if jsonl_path.is_symlink():
        raise ValueError(f"M128 records input must not be a symlink: {jsonl_path}")
    manifest_path = jsonl_path.parent / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"M128 manifest does not exist: {manifest_path}")
    manifest = _loads_strict(
        manifest_path.read_text(encoding="utf-8"), source=str(manifest_path)
    )
    if not isinstance(manifest, dict):
        raise ValueError(f"M128 manifest must contain one object: {manifest_path}")
    _assert_finite_tree(manifest, location=str(manifest_path))
    location = str(manifest_path)
    required = {
        "schema_version",
        "artifact_type",
        "run_id",
        "auxiliary_id",
        "dataset",
        "condition",
        "model",
        "split",
        "num_images",
        "num_rows",
        "sample_id_sha256",
        "source_sha256",
        "score_fields",
        "diagnostic_fields",
        "quadrature",
        "decision_rule",
        "provenance",
        "canonical_schema_v2_compatible",
        "jsonl_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"{manifest_path} is missing required fields {missing}")
    if manifest["schema_version"] != AUXILIARY_SCHEMA_VERSION:
        raise ValueError(f"{location}.schema_version is unsupported")
    if manifest["artifact_type"] != AUXILIARY_ARTIFACT_TYPE:
        raise ValueError(f"{location}.artifact_type is not the M128 auxiliary type")
    run_id = _required_string(manifest, "run_id", location=location)
    if manifest.get("auxiliary_id") != run_id:
        raise ValueError(f"{location}.auxiliary_id must equal run_id")
    for field in ("dataset", "condition", "model", "split"):
        _required_string(manifest, field, location=location)
    if manifest.get("canonical_schema_v2_compatible") is not False:
        raise ValueError(f"{location} must remain separate from canonical schema v2")
    if manifest.get("score_fields") != list(M128_SCORE_FIELDS):
        raise ValueError(f"{location}.score_fields differs from the M128 schema")
    if manifest.get("diagnostic_fields") != []:
        raise ValueError(
            f"{location}.diagnostic_fields must be empty for the locked campaign"
        )
    num_rows = _positive_integer(manifest["num_rows"], location=f"{location}.num_rows")
    if _positive_integer(
        manifest["num_images"], location=f"{location}.num_images"
    ) != num_rows:
        raise ValueError(f"{location} must contain one row per image")
    expected_jsonl_sha = _digest(
        manifest["jsonl_sha256"], location=f"{location}.jsonl_sha256"
    )
    if _sha256(jsonl_path) != expected_jsonl_sha:
        raise ValueError(f"SHA-256 mismatch for {jsonl_path}")
    expected_sample_sha = _digest(
        manifest["sample_id_sha256"], location=f"{location}.sample_id_sha256"
    )
    _digest(manifest["source_sha256"], location=f"{location}.source_sha256")
    decision = manifest.get("decision_rule")
    if decision != {"form": "foreground_probability >= gamma", "gamma": 0.5}:
        raise ValueError(f"{location}.decision_rule differs from gamma=0.5")
    _validate_quadrature(manifest, location=location)

    rows = []
    ordered_sample_ids = []
    sample_ids = set()
    image_ids = set()
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {jsonl_path}:{line_number}")
            row = _loads_strict(line, source=f"{jsonl_path}:{line_number}")
            if not isinstance(row, dict) or set(row) != EXPECTED_AUXILIARY_ROW_FIELDS:
                raise ValueError(f"invalid M128 row schema at {jsonl_path}:{line_number}")
            _assert_finite_tree(row, location=f"{jsonl_path}:{line_number}")
            if row["schema_version"] != AUXILIARY_SCHEMA_VERSION:
                raise ValueError(f"schema mismatch at {jsonl_path}:{line_number}")
            if row["run_id"] != run_id:
                raise ValueError(f"run_id mismatch at {jsonl_path}:{line_number}")
            sample_id = _required_string(
                row, "sample_id", location=f"{jsonl_path}:{line_number}"
            )
            image_id = _required_string(
                row, "image_id", location=f"{jsonl_path}:{line_number}"
            )
            if sample_id in sample_ids or image_id in image_ids:
                raise ValueError(f"duplicate image identity at {jsonl_path}:{line_number}")
            sample_ids.add(sample_id)
            image_ids.add(image_id)
            ordered_sample_ids.append(sample_id)
            if row["image_index"] != line_number - 1:
                raise ValueError(f"non-contiguous image_index at {jsonl_path}:{line_number}")
            if row["class_index"] != 1:
                raise ValueError(f"class_index must equal 1 at {jsonl_path}:{line_number}")
            _required_string(row, "class_name", location=f"{jsonl_path}:{line_number}")
            _positive_integer(row["height"], location=f"{jsonl_path}:{line_number}.height")
            _positive_integer(row["width"], location=f"{jsonl_path}:{line_number}.width")
            for score in M128_SCORE_FIELDS:
                _finite_score(row[score], location=f"{jsonl_path}:{line_number}.{score}")
            rows.append(row)
    if len(rows) != num_rows:
        raise ValueError(f"row-count mismatch for {jsonl_path}")
    observed_sample_sha = hashlib.sha256(
        "\n".join(ordered_sample_ids).encode("utf-8")
    ).hexdigest()
    if observed_sample_sha != expected_sample_sha:
        raise ValueError(f"sample_id_sha256 mismatch for {jsonl_path}")
    return M128ConditionData(jsonl_path, manifest_path, manifest, tuple(rows))


def load_m128_inputs(paths: Sequence[str | os.PathLike[str]]) -> list[M128ConditionData]:
    if len(paths) != M128_ARTIFACT_COUNT:
        raise ValueError("canonical M128 analysis requires exactly 16 explicit inputs")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("M128 input paths must be distinct")
    loaded = [load_m128_condition(path) for path in resolved]
    by_key = {}
    for data in loaded:
        key = data.dataset, data.condition
        if key in by_key:
            raise ValueError(f"duplicate M128 condition {key}")
        by_key[key] = data
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("M128 conditions differ from the locked benchmark")
    return [by_key[key] for key in EXPECTED_CONDITIONS]


def load_canonical_inputs(paths: Sequence[str | os.PathLike[str]]) -> list[ConditionData]:
    if len(paths) != M128_ARTIFACT_COUNT:
        raise ValueError("M128 analysis requires exactly 16 canonical inputs")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("canonical input paths must be distinct")
    loaded = [load_condition(path) for path in resolved]
    by_key = {}
    for data in loaded:
        key = data.dataset, data.condition
        if key in by_key:
            raise ValueError(f"duplicate canonical condition {key}")
        by_key[key] = data
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("canonical conditions differ from the locked benchmark")
    return [by_key[key] for key in EXPECTED_CONDITIONS]


def validate_m128_campaign_binding(
    conditions: Sequence[M128ConditionData], campaign_lock: str | os.PathLike[str]
) -> dict:
    lock_path, lock_sha, campaign_id, config, locked = load_analysis_campaign_lock(
        campaign_lock
    )
    lock_payload = _loads_strict(
        lock_path.read_text(encoding="utf-8"), source=str(lock_path)
    )
    estimator = lock_payload["estimator"]
    inputs = []
    if {(item.dataset, item.condition) for item in conditions} != set(locked):
        raise ValueError("M128 inputs do not match the campaign lock")
    for data in conditions:
        key = data.dataset, data.condition
        artifact = locked[key]
        manifest = data.manifest
        location = str(data.manifest_path)
        expected_top = {
            "model": artifact["model"],
            "split": artifact["split"],
            "num_images": artifact["num_samples"],
            "sample_id_sha256": artifact["sample_id_sha256"],
        }
        for field, expected in expected_top.items():
            observed = manifest.get(field)
            if field.endswith("sha256") and isinstance(observed, str):
                observed = observed.lower()
            if observed != expected:
                raise ValueError(f"{location}.{field} differs from the campaign lock")
        checkpoint = manifest.get("checkpoint")
        checkpoint_sha = None if checkpoint is None else checkpoint.get("sha256")
        if checkpoint_sha != artifact["checkpoint_sha256"]:
            raise ValueError(f"{location}.checkpoint differs from the campaign lock")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{location}.provenance must be an object")
        expected_provenance = {
            "campaign_id": campaign_id,
            "campaign_lock_sha256": lock_sha,
            "artifact_id": artifact["artifact_id"],
            "artifact_manifest_sha256": artifact["manifest_sha256"],
            "artifact_source_sha256": artifact["source_sha256"],
            "estimator_spec_sha256": estimator["spec_sha256"],
            "estimator_id": estimator["estimator_id"],
            "target_measure": estimator["target_measure"],
            "gamma": 0.5,
            "m": 128,
            "seed": 0,
            "include_m32_diagnostics": False,
        }
        for field, expected in expected_provenance.items():
            observed = provenance.get(field)
            if field.endswith("sha256") and isinstance(observed, str):
                observed = observed.lower()
            if observed != expected:
                raise ValueError(f"{location}.provenance.{field} differs from the lock")
        expected_run_id = _auxiliary_id(
            campaign_id=campaign_id,
            campaign_lock_sha256=lock_sha,
            artifact_manifest_sha256=artifact["manifest_sha256"],
            estimator_spec_sha256=estimator["spec_sha256"],
            source_sha256=manifest["source_sha256"],
            gamma=0.5,
            include_m32_diagnostics=False,
        )
        if manifest["run_id"] != expected_run_id:
            raise ValueError(f"{location}.run_id is not its content-derived identity")
        inputs.append(
            {
                "logical_id": f"{data.dataset}/{data.condition}/{manifest['run_id']}",
                "dataset": data.dataset,
                "condition": data.condition,
                "run_id": manifest["run_id"],
                "manifest_path": _portable_path(data.manifest_path),
                "manifest_sha256": _sha256(data.manifest_path),
                "records_path": _portable_path(data.jsonl_path),
                "records_sha256": manifest["jsonl_sha256"].lower(),
                "artifact_id": artifact["artifact_id"],
                "num_images": manifest["num_images"],
                "sample_id_sha256": manifest["sample_id_sha256"].lower(),
            }
        )
    return {
        "binding": "campaign-lock",
        "campaign_id": campaign_id,
        "campaign_lock": {
            "logical_name": lock_path.name,
            "sha256": lock_sha,
        },
        "config_sha256": config["sha256"].lower(),
        "inputs": inputs,
    }


def _correlation(left: np.ndarray, right: np.ndarray, *, method: str) -> dict:
    left_constant = bool(np.all(left == left[0]))
    right_constant = bool(np.all(right == right[0]))
    if left_constant or right_constant:
        sides = []
        if left_constant:
            sides.append("candidate")
        if right_constant:
            sides.append("reference")
        return {
            "defined": False,
            "value": None,
            "undefined_reason": f"constant_{'_and_'.join(sides)}_score",
        }
    if method == "spearman":
        value = float(
            np.corrcoef(
                rankdata(left, method="average"),
                rankdata(right, method="average"),
            )[0, 1]
        )
    elif method == "kendall_tau_b":
        value = float(kendalltau(left, right, variant="b", nan_policy="raise").statistic)
    else:
        raise ValueError(f"unsupported correlation method {method}")
    if not math.isfinite(value):
        raise RuntimeError(f"{method} returned a non-finite value")
    return {"defined": True, "value": value, "undefined_reason": None}


def _comparison_summary(
    canonical_rows: Sequence[Mapping[str, Any]],
    auxiliary_rows: Sequence[Mapping[str, Any]],
    spec: ComparisonSpec,
) -> dict:
    candidate = []
    reference = []
    risks = []
    for canonical, auxiliary in zip(canonical_rows, auxiliary_rows, strict=True):
        candidate_source = auxiliary if spec.candidate_score in auxiliary else canonical
        reference_source = auxiliary if spec.reference_score in auxiliary else canonical
        candidate.append(float(candidate_source[spec.candidate_score]))
        reference.append(float(reference_source[spec.reference_score]))
        risks.append(float(canonical[spec.matched_risk]))
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    risk_array = np.asarray(risks, dtype=np.float64)
    absolute_error = np.abs(candidate_array - reference_array)
    candidate_aurc = tie_aware_expected_aurc(candidate_array, risk_array)
    reference_aurc = tie_aware_expected_aurc(reference_array, risk_array)
    return {
        "name": spec.name,
        "label": spec.label,
        "candidate_score": spec.candidate_score,
        "reference_score": spec.reference_score,
        "matched_risk": spec.matched_risk,
        "reference_interpretation": spec.interpretation,
        "num_images": len(candidate_array),
        "per_image_absolute_score_error": {
            "mean": float(absolute_error.mean()),
            "median": float(np.median(absolute_error)),
            "p95": float(np.quantile(absolute_error, 0.95)),
            "max": float(absolute_error.max()),
        },
        "rank_agreement": {
            "spearman_rho": _correlation(
                candidate_array, reference_array, method="spearman"
            ),
            "kendall_tau_b": _correlation(
                candidate_array, reference_array, method="kendall_tau_b"
            ),
        },
        "matched_risk_aurc": {
            "candidate": candidate_aurc,
            "reference": reference_aurc,
            "signed_candidate_minus_reference": candidate_aurc - reference_aurc,
            "absolute_gap": abs(candidate_aurc - reference_aurc),
            "tie_policy": "analytic expectation over random within-score-tie order",
        },
    }


def analyze_condition(canonical: ConditionData, auxiliary: M128ConditionData) -> dict:
    if (canonical.dataset, canonical.condition) != (
        auxiliary.dataset,
        auxiliary.condition,
    ):
        raise ValueError("canonical/M128 condition keys differ")
    if len(canonical.rows) != len(auxiliary.rows):
        raise ValueError("canonical/M128 row counts differ")
    canonical_by_sample = {row["sample_id"]: row for row in canonical.rows}
    if len(canonical_by_sample) != len(canonical.rows):
        raise ValueError("canonical sample IDs are not unique")
    joined_canonical = []
    joined_auxiliary = []
    for auxiliary_row in auxiliary.rows:
        sample_id = auxiliary_row["sample_id"]
        if sample_id not in canonical_by_sample:
            raise ValueError(f"M128 sample {sample_id!r} is absent from canonical rows")
        canonical_row = canonical_by_sample[sample_id]
        for field in JOIN_FIELDS:
            if canonical_row.get(field) != auxiliary_row.get(field):
                raise ValueError(
                    f"join identity mismatch for {sample_id!r} in field {field}"
                )
        joined_canonical.append(canonical_row)
        joined_auxiliary.append(auxiliary_row)
    if len(joined_canonical) != len(canonical_by_sample):
        raise ValueError("canonical/M128 sample sets differ")
    key = canonical.dataset, canonical.condition
    return {
        "dataset": canonical.dataset,
        "condition": canonical.condition,
        "model": canonical.manifest["model"],
        "is_target_condition": key in TARGET_CONDITIONS,
        "num_images": len(joined_canonical),
        "comparisons": {
            spec.name: _comparison_summary(
                joined_canonical, joined_auxiliary, spec
            )
            for spec in COMPARISONS
        },
    }


def _range(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate range requires finite nonempty values")
    return {"min": float(array.min()), "max": float(array.max())}


def _aggregate_target_ranges(conditions: Sequence[Mapping[str, Any]]) -> dict:
    targets = [item for item in conditions if item["is_target_condition"]]
    result = {}
    for spec in COMPARISONS:
        rows = [item["comparisons"][spec.name] for item in targets]
        error_metrics = {}
        for metric in ("mean", "median", "p95", "max"):
            error_metrics[metric] = _range(
                [row["per_image_absolute_score_error"][metric] for row in rows]
            )
        rank_metrics = {}
        for metric in ("spearman_rho", "kendall_tau_b"):
            values = [row["rank_agreement"][metric] for row in rows]
            if not all(value["defined"] for value in values):
                raise ValueError(
                    f"{spec.name}.{metric} must be defined in every target condition"
                )
            rank_metrics[metric] = _range([value["value"] for value in values])
        result[spec.name] = {
            "label": spec.label,
            "num_target_conditions": len(rows),
            "per_image_absolute_score_error": error_metrics,
            "rank_agreement": rank_metrics,
            "matched_risk_aurc_absolute_gap": _range(
                [row["matched_risk_aurc"]["absolute_gap"] for row in rows]
            ),
        }
    return result


def analyze(
    m128_paths: Sequence[str | os.PathLike[str]],
    canonical_paths: Sequence[str | os.PathLike[str]],
    *,
    campaign_lock: str | os.PathLike[str],
) -> dict:
    auxiliary = load_m128_inputs(m128_paths)
    canonical = load_canonical_inputs(canonical_paths)
    canonical_provenance = validate_campaign_bound_conditions(canonical, campaign_lock)
    auxiliary_provenance = validate_m128_campaign_binding(auxiliary, campaign_lock)
    if (
        canonical_provenance["campaign_lock"]["sha256"]
        != auxiliary_provenance["campaign_lock"]["sha256"]
    ):
        raise ValueError("canonical and M128 inputs bind different campaign locks")
    for canonical_item, auxiliary_item in zip(canonical, auxiliary, strict=True):
        if canonical_item.manifest["sample_id_sha256"].lower() != auxiliary_item.manifest[
            "sample_id_sha256"
        ].lower():
            raise ValueError("canonical and M128 sample cohorts differ")
    conditions = [
        analyze_condition(canonical_item, auxiliary_item)
        for canonical_item, auxiliary_item in zip(canonical, auxiliary, strict=True)
    ]
    target_keys = [
        f"{item['dataset']}/{item['condition']}"
        for item in conditions
        if item["is_target_condition"]
    ]
    source_sha = _source_sha256()
    identity_payload = {
        "campaign_lock_sha256": canonical_provenance["campaign_lock"]["sha256"],
        "analysis_source_sha256": source_sha,
        "canonical_manifests": [
            item["manifest_sha256"] for item in canonical_provenance["inputs"]
        ],
        "m128_manifests": [
            item["manifest_sha256"] for item in auxiliary_provenance["inputs"]
        ],
    }
    analysis_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": analysis_id,
        "scope": {
            "purpose": (
                "numerical fidelity of M32 boundary quadrature against M128 and "
                "of Dice-M128 against the exact level-set Dice calculation"
            ),
            "m128_status": (
                "M128 is a high-resolution uniform-midpoint numerical reference; "
                "it is not an exact integral"
            ),
            "statistical_status": (
                "descriptive numerical diagnostics on fixed held-out images; no "
                "hypothesis tests or condition-independence claims"
            ),
        },
        "specification": {
            "score_error": "per-image absolute confidence difference",
            "rank_agreement": "Spearman average-rank correlation and Kendall tau-b",
            "aurc_gap": (
                "absolute difference in tie-aware matched-risk AURC; within-score "
                "ties are averaged analytically"
            ),
            "comparisons": [spec._asdict() for spec in COMPARISONS],
        },
        "condition_sets": {
            "all_conditions": [
                f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
            ],
            "target_condition_definition": (
                "clipseg-target and deeplabv3-target on each of five datasets"
            ),
            "target_conditions": target_keys,
            "num_conditions": len(conditions),
            "num_target_conditions": len(target_keys),
        },
        "provenance": {
            "analysis_source_sha256": source_sha,
            "canonical": canonical_provenance,
            "m128_auxiliary": auxiliary_provenance,
        },
        "target_aggregate_ranges": _aggregate_target_ranges(conditions),
        "conditions": conditions,
    }
    if len(conditions) != 16 or len(target_keys) != 10:
        raise AssertionError("canonical analysis must contain 16/10 total/target conditions")
    json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return report


def write_report(report: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite M128 analysis report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze(
        args.m128_inputs,
        args.canonical_inputs,
        campaign_lock=args.campaign_lock,
    )
    destination = write_report(report, args.output)
    print(_portable_path(destination))


if __name__ == "__main__":
    main()
