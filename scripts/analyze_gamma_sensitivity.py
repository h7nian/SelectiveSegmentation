r"""Analyze the locked deployment-threshold sensitivity campaign.

The primary action threshold remains :math:`\gamma=0.5`.  This auxiliary
analysis joins the separately scored :math:`\gamma\in\{0.3,0.7\}` actions to
the same frozen-image cohorts and reports how action quality, adjacent-geometry
AURC contrasts, and the selected image sets change.  It is a deployment-action
sensitivity analysis, not threshold tuning and not a robustness guarantee.

Canonical execution requires exactly 32 explicit auxiliary ``records.jsonl``
paths (16 conditions times two auxiliary thresholds), exactly 16 explicit
canonical assembly paths, the immutable auxiliary lock, and the locked primary
analysis JSON.  The script never discovers inputs with a glob.  It validates
all hashes, schemas, campaign bindings, condition keys, and sample-level joins
before computing any statistic.
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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from scripts.analyze_binary import (
    ANALYSIS_SCHEMA_VERSION,
    CONTRASTS,
    EXPECTED_CONDITIONS,
    ConditionData,
    load_condition,
    validate_campaign_bound_conditions,
)
from selectseg.binary_framework import tie_aware_expected_aurc
from selectseg.score_binary_common import (
    AUXILIARY_FIELDS,
    COMMON_SCORE_FIELDS,
    RISK_FIELDS,
)
from selectseg.score_binary_gamma_sensitivity import (
    AUXILIARY_ARTIFACT_TYPE,
    AUXILIARY_SCHEMA_VERSION,
    EXPECTED_GAMMAS,
    EXPECTED_M,
    EXPECTED_SEED,
    M32_SCORE_FIELDS,
    OUTPUT_ROW_FIELDS,
    load_auxiliary_lock,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_gamma_sensitivity_analysis"
AUXILIARY_GAMMAS = tuple(float(value) for value in EXPECTED_GAMMAS)
PRIMARY_GAMMA = 0.5
ALL_GAMMAS = (0.3, 0.5, 0.7)
COVERAGES = (0.25, 0.50, 0.75)
TARGET_CONDITIONS = frozenset(
    (dataset, condition)
    for dataset, condition in EXPECTED_CONDITIONS
    if condition in {"clipseg-target", "deeplabv3-target"}
)
INDEXED_SCORES = (
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_nhd_m32", "nHD-M32"),
    ("confidence_nhd95_m32", "nHD95-M32"),
)
RISK_LABELS = {
    "risk_dice": "Dice loss",
    "risk_nhd": "normalized penalized Hausdorff loss",
    "risk_nhd95": "normalized penalized HD95 loss",
}
GAMMA_PAIRS = ((0.3, 0.5), (0.7, 0.5), (0.3, 0.7))
JOIN_FIELDS = (
    "sample_id",
    "image_id",
    "image_index",
    "class_index",
    "class_name",
    "height",
    "width",
)
# These quantities depend only on the frozen probability map/truth, not on the
# deployed hard action.  Identifier fields must agree exactly.  Floating-point
# fields use a near-machine-precision tolerance because jobs may run on
# different CPU partitions whose reduction kernels can differ by one ULP.
# QFR, foreground entropy, SDC, and exact Dice are intentionally excluded:
# each uses the deployed mask and therefore changes with gamma.
GAMMA_INVARIANT_FIELDS = (
    *JOIN_FIELDS,
    "image_diagonal",
    "truth_foreground_fraction",
    "confidence_mean_max_probability",
    "confidence_negative_entropy",
    "confidence_plm10_entropy",
    "confidence_mmmc_entropy",
)
FLOAT_GAMMA_INVARIANT_FIELDS = frozenset(
    {
        "image_diagonal",
        "truth_foreground_fraction",
        "confidence_mean_max_probability",
        "confidence_negative_entropy",
        "confidence_plm10_entropy",
        "confidence_mmmc_entropy",
    }
)
# The observed cross-partition maximum is 1.17e-12 for a patch-sum baseline;
# keep the acceptance envelope close to that numerical floor and many orders
# below any reported score difference.
INVARIANT_REL_TOL = 2e-12
INVARIANT_ABS_TOL = 2e-12
EXPECTED_AUXILIARY_ROW_FIELDS = frozenset(OUTPUT_ROW_FIELDS)
EXPECTED_SCORE_FIELDS = [*COMMON_SCORE_FIELDS, *M32_SCORE_FIELDS]


@dataclass(frozen=True)
class GammaConditionData:
    records_path: Path
    manifest_path: Path
    manifest: dict
    rows: tuple[dict, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.manifest["dataset"], self.manifest["condition"]

    @property
    def gamma(self) -> float:
        return float(self.manifest["decision_rule"]["gamma"])


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-lock", required=True)
    parser.add_argument(
        "--auxiliary-inputs",
        nargs="+",
        required=True,
        metavar="AUXILIARY_RECORDS_JSONL",
        help="exactly 32 explicit gamma=0.3/0.7 records.jsonl paths",
    )
    parser.add_argument(
        "--canonical-inputs",
        nargs="+",
        required=True,
        metavar="CANONICAL_RECORDS_JSONL",
        help="exactly 16 explicit locked gamma=0.5 assembly records.jsonl paths",
    )
    parser.add_argument("--canonical-analysis", required=True)
    parser.add_argument(
        "--output-root", default="outputs/binary_gamma_sensitivity_analysis"
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


def _loads_strict(raw: str, *, source: str) -> Any:
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


def _finite_number(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{location} must be <= {maximum}")
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
        root / "selectseg/score_binary_gamma_sensitivity.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_midpoint_rule(manifest: Mapping[str, Any], *, location: str) -> None:
    quadrature = manifest.get("quadrature")
    if not isinstance(quadrature, dict) or set(quadrature) != {str(EXPECTED_M)}:
        raise ValueError(f"{location}.quadrature must contain exactly M={EXPECTED_M}")
    rule = quadrature[str(EXPECTED_M)]
    if not isinstance(rule, dict) or set(rule) != {"rule", "nodes", "weights"}:
        raise ValueError(f"{location}.quadrature.{EXPECTED_M} has an invalid schema")
    expected_nodes = [(index + 0.5) / EXPECTED_M for index in range(EXPECTED_M)]
    expected_weights = [1.0 / EXPECTED_M] * EXPECTED_M
    if (
        rule["rule"] != "midpoint"
        or rule["nodes"] != expected_nodes
        or rule["weights"] != expected_weights
    ):
        raise ValueError(f"{location} does not contain the locked M32 midpoint rule")


def load_gamma_condition(path: str | os.PathLike[str]) -> GammaConditionData:
    """Load and fully validate one auxiliary gamma records/manifest pair."""

    records_path = Path(path)
    if (
        records_path.name != "records.jsonl"
        or not records_path.is_file()
        or records_path.is_symlink()
    ):
        raise FileNotFoundError(
            f"gamma auxiliary input must be a regular records.jsonl: {records_path}"
        )
    manifest_path = records_path.parent / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"gamma auxiliary manifest is missing: {manifest_path}")
    manifest = _loads_strict(
        manifest_path.read_text(encoding="utf-8"), source=str(manifest_path)
    )
    if not isinstance(manifest, dict):
        raise ValueError(f"gamma manifest must contain one object: {manifest_path}")
    _assert_finite_tree(manifest, location=str(manifest_path))
    location = str(manifest_path)
    required = {
        "schema_version",
        "artifact_type",
        "run_id",
        "dataset",
        "condition",
        "model",
        "split",
        "num_images",
        "num_rows",
        "sample_id_sha256",
        "source_sha256",
        "score_fields",
        "risk_fields",
        "auxiliary_fields",
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
        raise ValueError(f"{location}.artifact_type is not gamma sensitivity")
    run_id = _required_string(manifest, "run_id", location=location)
    for field in ("dataset", "condition", "model", "split"):
        _required_string(manifest, field, location=location)
    if manifest.get("canonical_schema_v2_compatible") is not False:
        raise ValueError(f"{location} must remain separate from canonical schema v2")
    if manifest.get("score_fields") != EXPECTED_SCORE_FIELDS:
        raise ValueError(f"{location}.score_fields differs from the locked schema")
    if manifest.get("risk_fields") != list(RISK_FIELDS):
        raise ValueError(f"{location}.risk_fields differs from the locked schema")
    if manifest.get("auxiliary_fields") != list(AUXILIARY_FIELDS):
        raise ValueError(f"{location}.auxiliary_fields differs from the locked schema")
    num_rows = _positive_integer(manifest["num_rows"], location=f"{location}.num_rows")
    if _positive_integer(
        manifest["num_images"], location=f"{location}.num_images"
    ) != num_rows:
        raise ValueError(f"{location} must contain one row per image")
    if _sha256(records_path) != _digest(
        manifest["jsonl_sha256"], location=f"{location}.jsonl_sha256"
    ):
        raise ValueError(f"SHA-256 mismatch for {records_path}")
    sample_sha = _digest(
        manifest["sample_id_sha256"], location=f"{location}.sample_id_sha256"
    )
    _digest(manifest["source_sha256"], location=f"{location}.source_sha256")
    decision = manifest.get("decision_rule")
    if not isinstance(decision, dict) or set(decision) != {"form", "gamma"}:
        raise ValueError(f"{location}.decision_rule has an invalid schema")
    gamma = _finite_number(
        decision["gamma"], location=f"{location}.decision_rule.gamma"
    )
    if decision["form"] != "foreground_probability >= gamma" or gamma not in AUXILIARY_GAMMAS:
        raise ValueError(f"{location}.decision_rule is not a locked auxiliary gamma")
    _validate_midpoint_rule(manifest, location=location)

    rows = []
    sample_ids = set()
    image_ids = set()
    ordered_sample_ids = []
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row_location = f"{records_path}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank JSONL row at {row_location}")
            row = _loads_strict(line, source=row_location)
            if not isinstance(row, dict) or set(row) != EXPECTED_AUXILIARY_ROW_FIELDS:
                raise ValueError(f"invalid gamma row schema at {row_location}")
            _assert_finite_tree(row, location=row_location)
            if row["schema_version"] != 2 or row["run_id"] != run_id:
                raise ValueError(f"schema/run identity mismatch at {row_location}")
            sample_id = _required_string(row, "sample_id", location=row_location)
            image_id = _required_string(row, "image_id", location=row_location)
            if sample_id in sample_ids or image_id in image_ids:
                raise ValueError(f"duplicate image identity at {row_location}")
            sample_ids.add(sample_id)
            image_ids.add(image_id)
            ordered_sample_ids.append(sample_id)
            if row["image_index"] != line_number - 1:
                raise ValueError(f"non-contiguous image_index at {row_location}")
            if row["class_index"] != 1:
                raise ValueError(f"class_index must equal 1 at {row_location}")
            _required_string(row, "class_name", location=row_location)
            _positive_integer(row["height"], location=f"{row_location}.height")
            _positive_integer(row["width"], location=f"{row_location}.width")
            for field in (*RISK_FIELDS, "truth_foreground_fraction", "prediction_foreground_fraction"):
                _finite_number(
                    row[field], location=f"{row_location}.{field}", minimum=0.0, maximum=1.0
                )
            for field in M32_SCORE_FIELDS:
                _finite_number(
                    row[field], location=f"{row_location}.{field}", minimum=-1.0, maximum=0.0
                )
            for field in EXPECTED_SCORE_FIELDS:
                _finite_number(row[field], location=f"{row_location}.{field}")
            rows.append(row)
    if len(rows) != num_rows:
        raise ValueError(f"row-count mismatch for {records_path}")
    observed_sample_sha = hashlib.sha256(
        "\n".join(ordered_sample_ids).encode("utf-8")
    ).hexdigest()
    if observed_sample_sha != sample_sha:
        raise ValueError(f"sample_id_sha256 mismatch for {records_path}")
    return GammaConditionData(records_path, manifest_path, manifest, tuple(rows))


def _locked_artifacts(binding: Mapping[str, Any]) -> dict[tuple[str, str], dict]:
    result = {}
    for artifact in binding["campaign"]["artifacts"]:
        key = artifact["dataset"], artifact["condition"]
        if key in result:
            raise ValueError(f"canonical campaign has duplicate condition {key}")
        result[key] = artifact
    if set(result) != set(EXPECTED_CONDITIONS):
        raise ValueError("auxiliary lock does not bind the exact 16-condition benchmark")
    return result


def _validate_auxiliary_binding(
    data: GammaConditionData,
    *,
    binding: Mapping[str, Any],
    locked: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict:
    key = data.key
    if key not in locked:
        raise ValueError(f"undeclared gamma auxiliary condition {key}")
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
            raise ValueError(f"{location}.{field} differs from the auxiliary lock")
    checkpoint = manifest.get("checkpoint")
    checkpoint_sha = None if checkpoint is None else checkpoint.get("sha256")
    if checkpoint_sha != artifact["checkpoint_sha256"]:
        raise ValueError(f"{location}.checkpoint differs from the auxiliary lock")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{location}.provenance must be an object")
    expected_provenance = {
        "auxiliary_id": binding["data"]["auxiliary_id"],
        "auxiliary_lock_sha256": binding["sha256"],
        "auxiliary_spec_sha256": binding["data"]["spec"]["sha256"],
        "canonical_campaign_id": binding["campaign"]["campaign_id"],
        "canonical_campaign_lock_sha256": binding["campaign_sha256"],
        "artifact_id": artifact["artifact_id"],
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "artifact_source_sha256": artifact["source_sha256"],
        "estimator_spec_sha256": binding["data"]["estimator_spec"]["sha256"],
        "estimator_id": "midpoint-v1",
        "target_measure": "uniform-threshold",
        "gamma": data.gamma,
        "m": EXPECTED_M,
        "seed": EXPECTED_SEED,
        "artifact_passes": 1,
    }
    for field, expected in expected_provenance.items():
        observed = provenance.get(field)
        if field.endswith("sha256") and isinstance(observed, str):
            observed = observed.lower()
        if observed != expected:
            raise ValueError(f"{location}.provenance.{field} differs from the lock")
    return {
        "dataset": data.key[0],
        "condition": data.key[1],
        "gamma": data.gamma,
        "run_id": manifest["run_id"],
        "manifest_path": _portable_path(data.manifest_path),
        "manifest_sha256": _sha256(data.manifest_path),
        "records_path": _portable_path(data.records_path),
        "records_sha256": manifest["jsonl_sha256"].lower(),
        "sample_id_sha256": manifest["sample_id_sha256"].lower(),
        "num_images": manifest["num_images"],
        "artifact_id": artifact["artifact_id"],
    }


def load_gamma_inputs(
    paths: Sequence[str | os.PathLike[str]], *, binding: Mapping[str, Any]
) -> tuple[list[GammaConditionData], list[dict]]:
    expected_count = len(EXPECTED_CONDITIONS) * len(AUXILIARY_GAMMAS)
    if len(paths) != expected_count:
        raise ValueError(
            "gamma sensitivity analysis is incomplete: expected exactly "
            f"{expected_count} explicit auxiliary records.jsonl inputs "
            "(16 conditions x gamma={0.3,0.7}); "
            f"observed {len(paths)}"
        )
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("gamma auxiliary input paths must be distinct")
    locked = _locked_artifacts(binding)
    loaded = [load_gamma_condition(path) for path in resolved]
    by_key = {}
    provenance = []
    for data in loaded:
        key = (*data.key, data.gamma)
        if key in by_key:
            raise ValueError(f"duplicate gamma auxiliary experiment {key}")
        by_key[key] = data
        provenance.append(
            _validate_auxiliary_binding(data, binding=binding, locked=locked)
        )
    expected_grid = {
        (dataset, condition, gamma)
        for dataset, condition in EXPECTED_CONDITIONS
        for gamma in AUXILIARY_GAMMAS
    }
    if set(by_key) != expected_grid:
        missing = sorted(expected_grid - set(by_key))
        unexpected = sorted(set(by_key) - expected_grid)
        raise ValueError(
            "gamma sensitivity analysis is incomplete or off-grid: "
            f"missing={missing}; unexpected={unexpected}"
        )
    ordered = [
        by_key[(dataset, condition, gamma)]
        for dataset, condition in EXPECTED_CONDITIONS
        for gamma in AUXILIARY_GAMMAS
    ]
    provenance.sort(key=lambda row: (row["dataset"], row["condition"], row["gamma"]))
    return ordered, provenance


def load_canonical_inputs(paths: Sequence[str | os.PathLike[str]]) -> list[ConditionData]:
    if len(paths) != len(EXPECTED_CONDITIONS):
        raise ValueError(
            "gamma sensitivity analysis requires exactly 16 explicit canonical inputs"
        )
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("canonical input paths must be distinct")
    loaded = [load_condition(path) for path in resolved]
    by_key = {}
    for data in loaded:
        key = data.dataset, data.condition
        if key in by_key:
            raise ValueError(f"duplicate canonical condition {key}")
        if data.manifest.get("decision_rule") != {
            "form": "foreground_probability >= gamma",
            "gamma": PRIMARY_GAMMA,
        }:
            raise ValueError(f"canonical condition {key} is not gamma=0.5")
        by_key[key] = data
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("canonical inputs differ from the exact 16-condition benchmark")
    return [by_key[key] for key in EXPECTED_CONDITIONS]


def load_primary_analysis(path: str | os.PathLike[str]) -> tuple[dict, str, Path]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"locked primary analysis does not exist: {source}")
    raw = source.read_bytes()
    value = _loads_strict(raw.decode("utf-8"), source=str(source))
    if not isinstance(value, dict):
        raise ValueError("primary analysis must contain one JSON object")
    _assert_finite_tree(value, location=str(source))
    return value, hashlib.sha256(raw).hexdigest(), source


def validate_primary_analysis(
    value: Mapping[str, Any],
    canonical: Sequence[ConditionData],
    *,
    campaign_lock_sha256: str,
) -> None:
    """Bind the supplied primary JSON to the same canonical records and values."""

    if value.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("primary analysis has an unsupported schema")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("binding") != "campaign-lock":
        raise ValueError("primary analysis is not campaign-lock bound")
    lock = provenance.get("campaign_lock")
    if not isinstance(lock, dict) or lock.get("sha256") != campaign_lock_sha256:
        raise ValueError("primary analysis binds a different campaign lock")
    definitions = value.get("analysis", {}).get("contrast_definitions")
    expected_definitions = [
        {
            "name": spec.name,
            "left": spec.left,
            "right": spec.right,
            "risk": spec.risk,
        }
        for spec in CONTRASTS
    ]
    if definitions != expected_definitions:
        raise ValueError("primary analysis contrast definitions changed")
    condition_rows = value.get("conditions")
    if not isinstance(condition_rows, list) or len(condition_rows) != 16:
        raise ValueError("primary analysis must contain exactly 16 conditions")
    by_key = {}
    for row in condition_rows:
        if not isinstance(row, dict):
            raise ValueError("primary analysis condition must be an object")
        key = row.get("dataset"), row.get("condition")
        if key in by_key or key not in EXPECTED_CONDITIONS:
            raise ValueError("primary analysis has an unknown or duplicate condition")
        by_key[key] = row
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("primary analysis conditions differ from the benchmark")
    for data in canonical:
        key = data.dataset, data.condition
        row = by_key[key]
        if (
            row.get("jsonl_sha256") != data.manifest["jsonl_sha256"]
            or row.get("manifest_sha256") != _sha256(data.manifest_path)
            or row.get("num_rows") != len(data.rows)
        ):
            raise ValueError(f"primary analysis source differs for condition {key}")
        comparisons = row.get("comparisons")
        risks = row.get("risks")
        if not isinstance(comparisons, dict) or not isinstance(risks, dict):
            raise ValueError(f"primary analysis is incomplete for condition {key}")
        for spec in CONTRASTS:
            left = np.asarray([item[spec.left] for item in data.rows], dtype=float)
            right = np.asarray([item[spec.right] for item in data.rows], dtype=float)
            observed_risk = np.asarray(
                [item[spec.risk] for item in data.rows], dtype=float
            )
            left_aurc = tie_aware_expected_aurc(left, observed_risk)
            right_aurc = tie_aware_expected_aurc(right, observed_risk)
            primary = comparisons.get(spec.name)
            if not isinstance(primary, dict):
                raise ValueError(f"primary analysis lacks {key}/{spec.name}")
            expected_difference = left_aurc - right_aurc
            if not math.isclose(
                primary.get("difference_left_minus_right", math.nan),
                expected_difference,
                rel_tol=1e-13,
                abs_tol=1e-14,
            ):
                raise ValueError(f"primary contrast value differs for {key}/{spec.name}")
            for score, aurc in ((spec.left, left_aurc), (spec.right, right_aurc)):
                reported = risks.get(spec.risk, {}).get("methods", {}).get(score, {}).get(
                    "aurc"
                )
                if not math.isclose(
                    reported if isinstance(reported, (int, float)) else math.nan,
                    aurc,
                    rel_tol=1e-13,
                    abs_tol=1e-14,
                ):
                    raise ValueError(
                        f"primary AURC value differs for {key}/{spec.risk}/{score}"
                    )


def fractional_acceptance_weights(scores: Sequence[float], coverage: float) -> np.ndarray:
    """Tie-aware inclusion weights with exactly ``coverage * n`` accepted mass."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0,1]")
    target = float(coverage * values.size)
    weights = np.zeros(values.size, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    remaining = target
    start = 0
    while start < values.size and remaining > 0:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        group_size = stop - start
        inclusion = min(1.0, remaining / group_size)
        weights[order[start:stop]] = inclusion
        remaining -= inclusion * group_size
        start = stop
    if not np.isclose(weights.sum(), target, rtol=1e-12, atol=1e-12):
        raise RuntimeError("fractional selector did not achieve requested coverage")
    return weights


def _fractional_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.maximum(left, right).sum())
    if denominator <= 0:
        return 1.0
    return float(np.minimum(left, right).sum() / denominator)


def _correlation(left: np.ndarray, right: np.ndarray) -> dict:
    left_constant = bool(np.all(left == left[0]))
    right_constant = bool(np.all(right == right[0]))
    if left_constant or right_constant:
        sides = []
        if left_constant:
            sides.append("left")
        if right_constant:
            sides.append("right")
        return {
            "defined": False,
            "value": None,
            "undefined_reason": f"constant_{'_and_'.join(sides)}_score",
        }
    value = float(
        np.corrcoef(rankdata(left, method="average"), rankdata(right, method="average"))[0, 1]
    )
    if not math.isfinite(value):
        raise RuntimeError("Spearman correlation unexpectedly became non-finite")
    return {"defined": True, "value": value, "undefined_reason": None}


def _gamma_key(gamma: float) -> str:
    return f"{gamma:.1f}"


def _pair_key(left: float, right: float) -> str:
    return f"{left:.1f}_vs_{right:.1f}"


def _direction(value: float) -> str:
    if value < 0:
        return "left_lower_aurc"
    if value > 0:
        return "right_lower_aurc"
    return "exact_tie"


def _action_quality(rows: Sequence[Mapping[str, Any]]) -> dict:
    result = {
        "mean_matched_losses": {
            field: float(np.mean([float(row[field]) for row in rows]))
            for field in RISK_FIELDS
        },
        "deployed_prediction_empty_rate": float(
            np.mean([float(row["prediction_foreground_fraction"]) == 0.0 for row in rows])
        ),
        "mean_prediction_foreground_fraction": float(
            np.mean([float(row["prediction_foreground_fraction"]) for row in rows])
        ),
    }
    return result


def _contrast_summary(
    rows_by_gamma: Mapping[float, Sequence[Mapping[str, Any]]], spec
) -> dict:
    by_gamma = {}
    for gamma in ALL_GAMMAS:
        rows = rows_by_gamma[gamma]
        left = np.asarray([row[spec.left] for row in rows], dtype=np.float64)
        right = np.asarray([row[spec.right] for row in rows], dtype=np.float64)
        risk = np.asarray([row[spec.risk] for row in rows], dtype=np.float64)
        left_aurc = tie_aware_expected_aurc(left, risk)
        right_aurc = tie_aware_expected_aurc(right, risk)
        difference = left_aurc - right_aurc
        by_gamma[_gamma_key(gamma)] = {
            "left_aurc": left_aurc,
            "right_aurc": right_aurc,
            "difference_left_minus_right": difference,
            "direction": _direction(difference),
        }
    primary = by_gamma[_gamma_key(PRIMARY_GAMMA)]
    sensitivity = {}
    for gamma in AUXILIARY_GAMMAS:
        auxiliary = by_gamma[_gamma_key(gamma)]
        primary_direction = primary["direction"]
        auxiliary_direction = auxiliary["direction"]
        strict_reversal = {
            primary_direction,
            auxiliary_direction,
        } == {"left_lower_aurc", "right_lower_aurc"}
        sensitivity[_gamma_key(gamma)] = {
            "paired_change_from_gamma_0.5": (
                auxiliary["difference_left_minus_right"]
                - primary["difference_left_minus_right"]
            ),
            "primary_direction": primary_direction,
            "auxiliary_direction": auxiliary_direction,
            "direction_retained": auxiliary_direction == primary_direction,
            "strict_reversal": strict_reversal,
            "tie_transition": (
                auxiliary_direction != primary_direction and not strict_reversal
            ),
        }
    return {
        "name": spec.name,
        "left": spec.left,
        "right": spec.right,
        "risk": spec.risk,
        "by_gamma": by_gamma,
        "sensitivity_vs_primary": sensitivity,
    }


def _score_stability(
    rows_by_gamma: Mapping[float, Sequence[Mapping[str, Any]]], score: str
) -> dict:
    result = {}
    arrays = {
        gamma: np.asarray([row[score] for row in rows_by_gamma[gamma]], dtype=np.float64)
        for gamma in ALL_GAMMAS
    }
    for left_gamma, right_gamma in GAMMA_PAIRS:
        left = arrays[left_gamma]
        right = arrays[right_gamma]
        accepted = []
        for coverage in COVERAGES:
            left_weights = fractional_acceptance_weights(left, coverage)
            right_weights = fractional_acceptance_weights(right, coverage)
            accepted.append(
                {
                    "coverage": coverage,
                    "tie_aware_fractional_jaccard": _fractional_jaccard(
                        left_weights, right_weights
                    ),
                }
            )
        result[_pair_key(left_gamma, right_gamma)] = {
            "left_gamma": left_gamma,
            "right_gamma": right_gamma,
            "spearman_rho": _correlation(left, right),
            "accepted_set_agreement": accepted,
        }
    return result


def analyze_condition(
    canonical: ConditionData,
    auxiliary_by_gamma: Mapping[float, GammaConditionData],
) -> dict:
    """Strictly join one condition and compute all threshold diagnostics."""

    key = canonical.dataset, canonical.condition
    if set(auxiliary_by_gamma) != set(AUXILIARY_GAMMAS):
        raise ValueError(f"condition {key} lacks the two auxiliary gamma runs")
    canonical_by_sample = {row["sample_id"]: row for row in canonical.rows}
    if len(canonical_by_sample) != len(canonical.rows):
        raise ValueError(f"canonical sample IDs are not unique for {key}")
    rows_by_gamma: dict[float, tuple[dict, ...]] = {
        PRIMARY_GAMMA: tuple(canonical.rows)
    }
    for gamma in AUXILIARY_GAMMAS:
        auxiliary = auxiliary_by_gamma[gamma]
        if auxiliary.key != key:
            raise ValueError(f"canonical/auxiliary condition keys differ for {key}")
        if len(auxiliary.rows) != len(canonical.rows):
            raise ValueError(f"canonical/auxiliary row counts differ for {key}/gamma={gamma}")
        if auxiliary.manifest["model"] != canonical.manifest["model"]:
            raise ValueError(f"canonical/auxiliary model differs for {key}/gamma={gamma}")
        joined = []
        seen = set()
        for row in auxiliary.rows:
            sample_id = row["sample_id"]
            if sample_id not in canonical_by_sample or sample_id in seen:
                raise ValueError(f"sample-set mismatch for {key}/gamma={gamma}")
            reference = canonical_by_sample[sample_id]
            for field in GAMMA_INVARIANT_FIELDS:
                observed = row.get(field)
                expected = reference.get(field)
                if field in FLOAT_GAMMA_INVARIANT_FIELDS:
                    agrees = isinstance(observed, (int, float)) and isinstance(
                        expected, (int, float)
                    ) and math.isclose(
                        float(observed),
                        float(expected),
                        rel_tol=INVARIANT_REL_TOL,
                        abs_tol=INVARIANT_ABS_TOL,
                    )
                else:
                    agrees = observed == expected
                if not agrees:
                    raise ValueError(
                        f"gamma-invariant join mismatch for {key}/{sample_id!r} "
                        f"at gamma={gamma} in field {field}"
                    )
            seen.add(sample_id)
            joined.append(row)
        if seen != set(canonical_by_sample):
            raise ValueError(f"sample-set mismatch for {key}/gamma={gamma}")
        rows_by_gamma[gamma] = tuple(joined)

    # Align gamma=0.5 to auxiliary iteration order.  The auxiliary scorers are
    # contiguous, but the explicit sample-key join is the binding contract.
    order = [row["sample_id"] for row in rows_by_gamma[AUXILIARY_GAMMAS[0]]]
    rows_by_gamma[PRIMARY_GAMMA] = tuple(canonical_by_sample[item] for item in order)
    other_by_sample = {
        row["sample_id"]: row for row in rows_by_gamma[AUXILIARY_GAMMAS[1]]
    }
    rows_by_gamma[AUXILIARY_GAMMAS[1]] = tuple(other_by_sample[item] for item in order)

    return {
        "dataset": canonical.dataset,
        "condition": canonical.condition,
        "model": canonical.manifest["model"],
        "is_target_condition": key in TARGET_CONDITIONS,
        "num_images": len(order),
        "action_quality_by_gamma": {
            _gamma_key(gamma): _action_quality(rows_by_gamma[gamma])
            for gamma in ALL_GAMMAS
        },
        "contrasts": {
            spec.name: _contrast_summary(rows_by_gamma, spec) for spec in CONTRASTS
        },
        "indexed_score_stability": {
            score: {
                "label": label,
                "gamma_pairs": _score_stability(rows_by_gamma, score),
            }
            for score, label in INDEXED_SCORES
        },
    }


def _range(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("range requires a finite nonempty sequence")
    return {"min": float(array.min()), "max": float(array.max())}


def _distribution(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution summary requires finite values")
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "mean_absolute": float(np.abs(array).mean()),
        "max_absolute": float(np.abs(array).max()),
    }


def aggregate_targets(conditions: Sequence[Mapping[str, Any]]) -> dict:
    targets = [row for row in conditions if row["is_target_condition"]]
    if len(targets) != 10:
        raise ValueError("target headline requires exactly ten target-adapted conditions")
    action = {}
    for gamma in ALL_GAMMAS:
        gamma_key = _gamma_key(gamma)
        action[gamma_key] = {
            "mean_matched_loss_ranges": {
                risk: _range(
                    [
                        row["action_quality_by_gamma"][gamma_key][
                            "mean_matched_losses"
                        ][risk]
                        for row in targets
                    ]
                )
                for risk in RISK_FIELDS
            },
            "deployed_prediction_empty_rate_range": _range(
                [
                    row["action_quality_by_gamma"][gamma_key][
                        "deployed_prediction_empty_rate"
                    ]
                    for row in targets
                ]
            ),
            "mean_prediction_foreground_fraction_range": _range(
                [
                    row["action_quality_by_gamma"][gamma_key][
                        "mean_prediction_foreground_fraction"
                    ]
                    for row in targets
                ]
            ),
        }

    contrasts = {}
    for spec in CONTRASTS:
        rows = [row["contrasts"][spec.name] for row in targets]
        by_gamma = {
            _gamma_key(gamma): _range(
                [
                    row["by_gamma"][_gamma_key(gamma)][
                        "difference_left_minus_right"
                    ]
                    for row in rows
                ]
            )
            for gamma in ALL_GAMMAS
        }
        sensitivity = {}
        for gamma in AUXILIARY_GAMMAS:
            gamma_key = _gamma_key(gamma)
            items = [row["sensitivity_vs_primary"][gamma_key] for row in rows]
            changes = [item["paired_change_from_gamma_0.5"] for item in items]
            sensitivity[gamma_key] = {
                "num_target_conditions": len(items),
                "num_direction_retained": sum(
                    bool(item["direction_retained"]) for item in items
                ),
                "num_strict_reversals": sum(
                    bool(item["strict_reversal"]) for item in items
                ),
                "num_tie_transitions": sum(
                    bool(item["tie_transition"]) for item in items
                ),
                "strict_reversal_conditions": [
                    f"{target['dataset']}/{target['condition']}"
                    for target, item in zip(targets, items, strict=True)
                    if item["strict_reversal"]
                ],
                "paired_change_from_gamma_0.5": _distribution(changes),
            }
        contrasts[spec.name] = {
            "left": spec.left,
            "right": spec.right,
            "risk": spec.risk,
            "difference_ranges_by_gamma": by_gamma,
            "sensitivity_vs_primary": sensitivity,
        }

    stability = {}
    for score, label in INDEXED_SCORES:
        stability[score] = {"label": label, "gamma_pairs": {}}
        for left_gamma, right_gamma in GAMMA_PAIRS:
            pair_key = _pair_key(left_gamma, right_gamma)
            items = [
                row["indexed_score_stability"][score]["gamma_pairs"][pair_key]
                for row in targets
            ]
            correlation_values = [
                item["spearman_rho"]["value"]
                for item in items
                if item["spearman_rho"]["defined"]
            ]
            correlations = {
                "num_defined": len(correlation_values),
                "num_undefined": len(items) - len(correlation_values),
                "range": _range(correlation_values) if correlation_values else None,
            }
            agreements = {}
            for coverage in COVERAGES:
                values = [
                    next(
                        entry["tie_aware_fractional_jaccard"]
                        for entry in item["accepted_set_agreement"]
                        if entry["coverage"] == coverage
                    )
                    for item in items
                ]
                agreements[f"{coverage:.2f}"] = _range(values)
            stability[score]["gamma_pairs"][pair_key] = {
                "spearman_rho": correlations,
                "accepted_set_jaccard_ranges": agreements,
            }
    return {
        "num_target_conditions": 10,
        "action_quality": action,
        "contrasts": contrasts,
        "indexed_score_stability": stability,
    }


def analyze(
    auxiliary_paths: Sequence[str | os.PathLike[str]],
    canonical_paths: Sequence[str | os.PathLike[str]],
    *,
    auxiliary_lock: str | os.PathLike[str],
    canonical_analysis_path: str | os.PathLike[str],
) -> dict:
    binding = load_auxiliary_lock(auxiliary_lock)
    canonical = load_canonical_inputs(canonical_paths)
    canonical_provenance = validate_campaign_bound_conditions(
        canonical, binding["campaign_path"]
    )
    auxiliary, auxiliary_inputs = load_gamma_inputs(
        auxiliary_paths, binding=binding
    )
    primary_analysis, primary_hash, primary_path = load_primary_analysis(
        canonical_analysis_path
    )
    validate_primary_analysis(
        primary_analysis,
        canonical,
        campaign_lock_sha256=binding["campaign_sha256"],
    )
    by_auxiliary = {(item.key, item.gamma): item for item in auxiliary}
    conditions = []
    for canonical_item in canonical:
        key = canonical_item.dataset, canonical_item.condition
        pair = {
            gamma: by_auxiliary[(key, gamma)] for gamma in AUXILIARY_GAMMAS
        }
        for auxiliary_item in pair.values():
            if (
                auxiliary_item.manifest["sample_id_sha256"].lower()
                != canonical_item.manifest["sample_id_sha256"].lower()
            ):
                raise ValueError(f"canonical/auxiliary sample cohort differs for {key}")
        conditions.append(analyze_condition(canonical_item, pair))

    source_sha = _source_sha256()
    identity = {
        "analysis_source_sha256": source_sha,
        "auxiliary_lock_sha256": binding["sha256"],
        "canonical_analysis_sha256": primary_hash,
        "canonical_manifest_sha256": [
            row["manifest_sha256"] for row in canonical_provenance["inputs"]
        ],
        "auxiliary_manifest_sha256": [
            row["manifest_sha256"] for row in auxiliary_inputs
        ],
    }
    analysis_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    target_keys = [
        f"{row['dataset']}/{row['condition']}"
        for row in conditions
        if row["is_target_condition"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": analysis_id,
        "scope": {
            "purpose": (
                "deployment-action sensitivity at fixed predeclared thresholds "
                "gamma={0.3,0.5,0.7} on the same frozen probability maps"
            ),
            "primary_threshold": 0.5,
            "status": (
                "descriptive deployment-action sensitivity; this is neither "
                "test-set threshold tuning nor a robustness guarantee"
            ),
            "scientific_headline": (
                "the ten target-adapted conditions; six general/external conditions "
                "remain fixed stress controls"
            ),
        },
        "specification": {
            "gamma_values": list(ALL_GAMMAS),
            "indexed_scores": [
                {"field": field, "label": label} for field, label in INDEXED_SCORES
            ],
            "matched_losses": [
                {"field": field, "label": RISK_LABELS[field]} for field in RISK_FIELDS
            ],
            "contrasts": [
                {
                    "name": spec.name,
                    "left": spec.left,
                    "right": spec.right,
                    "risk": spec.risk,
                    "difference": "AURC(left) - AURC(right); lower AURC is better",
                }
                for spec in CONTRASTS
            ],
            "tie_policy": "analytic expectation over random within-score-tie order",
            "direction_rule": (
                "negative favors left, positive favors right, and exactly zero is a tie; "
                "no data-dependent tolerance is used"
            ),
            "accepted_set_agreement": (
                "tie-aware fractional Jaccard at coverage 0.25, 0.50, and 0.75"
            ),
        },
        "condition_sets": {
            "all_conditions": [
                f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
            ],
            "target_conditions": target_keys,
            "num_conditions": len(conditions),
            "num_target_conditions": len(target_keys),
            "num_auxiliary_experiments": len(auxiliary),
        },
        "provenance": {
            "analysis_source_sha256": source_sha,
            "auxiliary_lock": {
                "logical_name": binding["path"].name,
                "sha256": binding["sha256"],
            },
            "canonical_campaign": canonical_provenance,
            "canonical_primary_analysis": {
                "path": _portable_path(primary_path),
                "sha256": primary_hash,
            },
            "auxiliary_inputs": auxiliary_inputs,
        },
        "target_headline": aggregate_targets(conditions),
        "conditions": conditions,
    }
    if len(conditions) != 16 or len(target_keys) != 10 or len(auxiliary) != 32:
        raise AssertionError("gamma analysis did not preserve the 16/10/32 design")
    json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return report


def write_report(report: Mapping[str, Any], output_root: str | os.PathLike[str]) -> Path:
    analysis_id = report.get("analysis_id")
    if not isinstance(analysis_id, str) or len(analysis_id) != 16:
        raise ValueError("report.analysis_id must be a 16-character content identity")
    directory = Path(output_root) / analysis_id
    destination = directory / "analysis.json"
    if destination.exists() or destination.is_symlink() or directory.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite gamma sensitivity analysis: {destination}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".analysis.json.tmp-", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
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
    report = analyze(
        args.auxiliary_inputs,
        args.canonical_inputs,
        auxiliary_lock=args.auxiliary_lock,
        canonical_analysis_path=args.canonical_analysis,
    )
    destination = write_report(report, args.output_root)
    print(_portable_path(destination))


if __name__ == "__main__":
    main()
