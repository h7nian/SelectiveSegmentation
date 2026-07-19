"""Strict analysis for the loss-indexed binary segmentation experiment.

The script consumes focused ``*.jsonl`` files written by
``selectseg.binary_eval`` and the matching ``*.manifest.json`` files.  It does
not import or reuse the legacy multiclass analysis.

Example::

    python scripts/analyze_binary.py --input-dir outputs/binary \
        --output-dir outputs/binary/analysis

Use ``--inputs`` to analyze an explicit, reproducible list of JSONL files.
Complete analysis additionally requires ``--campaign-lock`` and verifies that
every input is a final assembly bound to that exact immutable campaign.
The outputs are a machine-readable JSON summary, a long-form CSV, and a LaTeX
main table.  All AURCs use analytic random ordering within exact confidence
ties.  Four predeclared adjacent-geometry comparisons report paired
image-cluster percentile-bootstrap intervals and two-sided approximate
bootstrap tail probabilities computed from the same resamples. A Holm step-down
transform is applied separately to the core and extension families, without
making significance or exact finite-sample error-control claims. All other
score--risk combinations remain descriptive.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.binary_framework import (  # noqa: E402
    summarize_aurc,
    tie_aware_expected_aurc,
)


SUPPORTED_SCHEMA_VERSION = 2
ANALYSIS_SCHEMA_VERSION = 2
JSON_NAME = "analysis.json"
CSV_NAME = "main_table.csv"
LATEX_NAME = "main_table.tex"

METHODS = (
    ("confidence_sdc", "SDC"),
    ("confidence_mean_max_probability", "Mean max probability"),
    ("confidence_negative_entropy", "Negative entropy"),
    ("confidence_dice_exact", "Dice-Exact"),
    ("confidence_qfr_entropy", "QFR-Entropy"),
    ("confidence_plm10_entropy", "PLM-10/PLA-10-Entropy"),
    ("confidence_mmmc_entropy", "MMMC-Entropy"),
    ("confidence_foreground_entropy", "Foreground entropy"),
    ("confidence_dice_m2", "Dice-M2"),
    ("confidence_dice_m8", "Dice-M8"),
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_nhd_m2", "nHD-M2"),
    ("confidence_nhd_m8", "nHD-M8"),
    ("confidence_nhd_m32", "nHD-M32"),
    ("confidence_nhd95_m2", "nHD95-M2"),
    ("confidence_nhd95_m8", "nHD95-M8"),
    ("confidence_nhd95_m32", "nHD95-M32"),
)
EXPECTED_CONDITIONS = (
    ("pet", "clipseg-general"),
    ("pet", "clipseg-target"),
    ("pet", "deeplabv3-target"),
    ("pet", "deeplabv3-external"),
    ("kvasir", "clipseg-general"),
    ("kvasir", "clipseg-target"),
    ("kvasir", "deeplabv3-target"),
    ("fives", "clipseg-general"),
    ("fives", "clipseg-target"),
    ("fives", "deeplabv3-target"),
    ("isic", "clipseg-general"),
    ("isic", "clipseg-target"),
    ("isic", "deeplabv3-target"),
    ("tn3k", "clipseg-general"),
    ("tn3k", "clipseg-target"),
    ("tn3k", "deeplabv3-target"),
)
HOLM_FAMILY_BY_DATASET = {
    "pet": "core",
    "kvasir": "core",
    "fives": "core",
    "isic": "extension",
    "tn3k": "extension",
}
HOLM_FAMILY_DEFINITIONS = {
    "core": (
        "Oxford Pet, Kvasir-SEG, and FIVES conditions across the four "
        "predeclared adjacent-geometry contrasts"
    ),
    "extension": (
        "ISIC 2018 and TN3K conditions across the four predeclared "
        "adjacent-geometry contrasts"
    ),
}
RISKS = (
    ("risk_dice", "Dice risk"),
    ("risk_nhd", "Normalized penalized Hausdorff risk"),
    ("risk_nhd95", "Normalized penalized HD95 risk"),
)
DICE_QUADRATURE_METHODS = (
    ("confidence_dice_m2", 2),
    ("confidence_dice_m8", 8),
    ("confidence_dice_m32", 32),
)
DICE_EXACT_REFERENCE = "confidence_dice_exact"


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    left: str
    right: str
    risk: str


CONTRASTS = (
    ContrastSpec(
        "dice_vs_nhd_under_dice",
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "risk_dice",
    ),
    ContrastSpec(
        "dice_vs_nhd_under_nhd",
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "risk_nhd",
    ),
    ContrastSpec(
        "nhd_vs_nhd95_under_nhd",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "risk_nhd",
    ),
    ContrastSpec(
        "nhd_vs_nhd95_under_nhd95",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
        "risk_nhd95",
    ),
)
METHOD_LABELS = dict(METHODS)
RISK_LABELS = dict(RISKS)
REQUIRED_SCORE_FIELDS = frozenset(METHOD_LABELS)
REQUIRED_RISK_FIELDS = frozenset(RISK_LABELS)
REQUIRED_ROW_FIELDS = frozenset({"schema_version", "run_id", "sample_id", "image_id"})
OPTIONAL_AUXILIARY_FIELDS = frozenset({"risk_hd_pixels", "risk_hd95_pixels"})
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
ASSEMBLY_ARTIFACT_TYPE = "selectseg.binary_simulation_assembly"
LOCK_SCHEMA_VERSION = 1
LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "campaign_id",
        "config",
        "protocol",
        "estimator",
        "paths",
        "artifacts",
    }
)
LOCK_ARTIFACT_FIELDS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "artifact_id",
        "dataset",
        "condition",
        "model",
        "split",
        "checkpoint_sha256",
        "source_sha256",
        "sample_id_sha256",
        "num_samples",
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
        dest="input_dirs",
        action="append",
        default=None,
        help=(
            "directory recursively searched for *.jsonl when --inputs is omitted; "
            "repeat to combine disjoint roots (default: outputs/binary)"
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="explicit JSONL files; each requires a matching .manifest.json",
    )
    parser.add_argument(
        "--campaign-lock",
        default=None,
        help=(
            "immutable campaign lock required for complete analysis; every "
            "assembly is checked against its SHA-256 and locked cohort"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/binary/analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=4,
        help="parallel adjacent contrasts per condition (default: 4)",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "allow a nonempty declared subset for draft analysis; by default "
            "the exact 16-condition benchmark is required"
        ),
    )
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


def _analysis_source_sha256() -> str:
    """Bind the analysis result to the code that defines its statistics."""

    paths = (Path(__file__).resolve(), REPO_ROOT / "selectseg/binary_framework.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _digest(value, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    value = value.lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


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


def _field_list(manifest, name, *, location, allow_empty=False):
    value = manifest.get(name)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{location}.{name} must be {qualifier}")
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

    manifest = _loads_strict(manifest_path.read_text(), source=str(manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain one JSON object: {manifest_path}")
    _assert_finite_tree(manifest, location=str(manifest_path))
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(
            f"manifest {manifest_path} is missing required fields {missing}"
        )
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
        manifest,
        "auxiliary_fields",
        location=str(manifest_path),
        allow_empty=True,
    )
    missing_scores = sorted(REQUIRED_SCORE_FIELDS - score_fields)
    missing_risks = sorted(REQUIRED_RISK_FIELDS - risk_fields)
    if missing_scores or missing_risks:
        raise ValueError(
            f"manifest {manifest_path} lacks required score/risk fields: "
            f"scores={missing_scores}, risks={missing_risks}"
        )
    if score_fields != REQUIRED_SCORE_FIELDS:
        raise ValueError(
            f"manifest {manifest_path} must list exactly the predeclared scores "
            f"{sorted(REQUIRED_SCORE_FIELDS)}; got {sorted(score_fields)}"
        )
    if risk_fields != REQUIRED_RISK_FIELDS:
        raise ValueError(
            f"manifest {manifest_path} must list exactly the three main risks "
            f"{sorted(REQUIRED_RISK_FIELDS)}; got {sorted(risk_fields)}"
        )
    undeclared_auxiliary = auxiliary_fields - OPTIONAL_AUXILIARY_FIELDS
    if undeclared_auxiliary:
        raise ValueError(
            f"manifest {manifest_path} contains unsupported auxiliary fields "
            f"{sorted(undeclared_auxiliary)}; allowed optional fields are "
            f"{sorted(OPTIONAL_AUXILIARY_FIELDS)}"
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
        (REQUIRED_ROW_FIELDS | score_fields | risk_fields | auxiliary_fields)
        - row_fields
    )
    if missing_row_fields:
        raise ValueError(f"{jsonl_path} rows lack required fields {missing_row_fields}")
    manifested_scores = {
        field for field in row_fields if field.startswith("confidence_")
    }
    row_risks = {field for field in row_fields if field.startswith("risk_")}
    if manifested_scores != score_fields or row_risks != risk_fields | auxiliary_fields:
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
            raise ValueError(f"schema_version mismatch at {jsonl_path}:{line_number}")
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"run_id mismatch at {jsonl_path}:{line_number}")
        for field in score_fields | risk_fields | auxiliary_fields:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{jsonl_path}:{line_number}.{field} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{jsonl_path}:{line_number}.{field} must be finite")
        for field in risk_fields:
            if not 0 <= row[field] <= 1:
                raise ValueError(
                    f"{jsonl_path}:{line_number}.{field} must lie in [0, 1]"
                )
        for field in auxiliary_fields:
            if row[field] < 0:
                raise ValueError(
                    f"{jsonl_path}:{line_number}.{field} must be non-negative"
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


def load_analysis_campaign_lock(path):
    """Load the portable immutable lock without requiring frozen payloads."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"campaign lock does not exist: {path}")
    raw = path.read_bytes()
    lock = _loads_strict(raw.decode("utf-8"), source=str(path))
    if not isinstance(lock, dict) or set(lock) != LOCK_FIELDS:
        raise ValueError(f"campaign lock must contain exactly {sorted(LOCK_FIELDS)}")
    _assert_finite_tree(lock, location=str(path))
    if lock["lock_schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError(f"{path}.lock_schema_version must equal {LOCK_SCHEMA_VERSION}")
    campaign_id = _required_string(lock, "campaign_id", location=str(path))
    protocol = lock.get("protocol")
    expected_protocol = {
        "gamma_values": [0.5],
        "m_values": [2, 8, 32],
        "quadrature_rule": "midpoint-v1",
        "seeds": [0],
    }
    if protocol != expected_protocol:
        raise ValueError("campaign lock does not contain the final midpoint protocol")
    config = lock.get("config")
    if not isinstance(config, dict) or set(config) != {"path", "sha256"}:
        raise ValueError("campaign lock config provenance is malformed")
    _required_string(config, "path", location=f"{path}.config")
    _digest(config["sha256"], location=f"{path}.config.sha256")
    estimator = lock.get("estimator")
    if not isinstance(estimator, dict):
        raise ValueError("campaign lock estimator provenance is malformed")
    if (
        estimator.get("estimator_id") != "midpoint-v1"
        or estimator.get("target_measure") != "uniform-threshold"
    ):
        raise ValueError("campaign lock estimator is not midpoint-v1")
    _digest(estimator.get("spec_sha256"), location=f"{path}.estimator.spec_sha256")

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED_CONDITIONS):
        raise ValueError("campaign lock must contain exactly 16 artifacts")
    by_key = {}
    for index, artifact in enumerate(artifacts):
        location = f"{path}.artifacts[{index}]"
        if not isinstance(artifact, dict) or set(artifact) != LOCK_ARTIFACT_FIELDS:
            raise ValueError(
                f"{location} must contain exactly {sorted(LOCK_ARTIFACT_FIELDS)}"
            )
        dataset = _required_string(artifact, "dataset", location=location)
        condition = _required_string(artifact, "condition", location=location)
        key = (dataset, condition)
        if key in by_key:
            raise ValueError(f"campaign lock contains duplicate condition {key}")
        for field in ("manifest_sha256", "source_sha256", "sample_id_sha256"):
            _digest(artifact[field], location=f"{location}.{field}")
        checkpoint_sha = artifact["checkpoint_sha256"]
        if checkpoint_sha is not None:
            _digest(checkpoint_sha, location=f"{location}.checkpoint_sha256")
        if (
            isinstance(artifact["num_samples"], bool)
            or not isinstance(artifact["num_samples"], int)
            or artifact["num_samples"] <= 0
        ):
            raise ValueError(f"{location}.num_samples must be a positive integer")
        for field in ("manifest_path", "artifact_id", "model", "split"):
            _required_string(artifact, field, location=location)
        by_key[key] = artifact
    if set(by_key) != set(EXPECTED_CONDITIONS):
        raise ValueError("campaign lock conditions differ from the declared benchmark")
    return path, _sha256(path), campaign_id, config, by_key


def validate_campaign_bound_conditions(conditions, campaign_lock):
    """Require final assemblies and bind each one to its locked source artifact."""

    lock_path, lock_sha, campaign_id, config, locked = load_analysis_campaign_lock(
        campaign_lock
    )
    observed = {(item.dataset, item.condition) for item in conditions}
    if observed != set(locked) or len(conditions) != len(locked):
        raise ValueError(
            "analysis inputs must contain each locked condition exactly once"
        )

    inputs = []
    for data in sorted(conditions, key=lambda item: (item.dataset, item.condition)):
        manifest = data.manifest
        location = str(data.manifest_path)
        if manifest.get("artifact_type") != ASSEMBLY_ARTIFACT_TYPE:
            raise ValueError(
                f"{location}.artifact_type must equal {ASSEMBLY_ARTIFACT_TYPE!r}"
            )
        assembly = manifest.get("assembly")
        if not isinstance(assembly, dict):
            raise ValueError(f"{location}.assembly must be an object")
        if assembly.get("assembly_schema_version") != 2:
            raise ValueError(f"{location}.assembly has an unsupported schema")
        assembly_source_sha = _digest(
            assembly.get("assembly_source_sha256"),
            location=f"{location}.assembly.assembly_source_sha256",
        )
        artifact = locked[(data.dataset, data.condition)]
        expected = {
            "campaign_id": campaign_id,
            "campaign_lock_sha256": lock_sha,
            "artifact_id": artifact["artifact_id"],
            "artifact_manifest_sha256": artifact["manifest_sha256"].lower(),
            "artifact_source_sha256": artifact["source_sha256"].lower(),
        }
        for field, value in expected.items():
            observed_value = assembly.get(field)
            if field.endswith("sha256") and isinstance(observed_value, str):
                observed_value = observed_value.lower()
            if observed_value != value:
                raise ValueError(f"{location}.assembly.{field} differs from the lock")
        if manifest["split"] != artifact["split"]:
            raise ValueError(f"{location}.split differs from the lock")
        if manifest.get("model") != artifact["model"]:
            raise ValueError(f"{location}.model differs from the lock")
        if manifest["num_images"] != artifact["num_samples"]:
            raise ValueError(f"{location}.num_images differs from the lock")
        if manifest["sample_id_sha256"].lower() != artifact["sample_id_sha256"].lower():
            raise ValueError(f"{location}.sample_id_sha256 differs from the lock")
        checkpoint = manifest.get("checkpoint")
        checkpoint_sha = None if checkpoint is None else checkpoint.get("sha256")
        if checkpoint_sha != artifact["checkpoint_sha256"]:
            raise ValueError(f"{location}.checkpoint differs from the lock")
        manifest_sha = _sha256(data.manifest_path)
        logical_id = f"{data.dataset}/{data.condition}/{manifest['run_id']}"
        inputs.append(
            {
                "logical_id": logical_id,
                "dataset": data.dataset,
                "condition": data.condition,
                "assembly_run_id": manifest["run_id"],
                "assembly_source_sha256": assembly_source_sha,
                "artifact_id": artifact["artifact_id"],
                "manifest_sha256": manifest_sha,
                "records_sha256": manifest["jsonl_sha256"].lower(),
                "sample_id_sha256": manifest["sample_id_sha256"].lower(),
                "num_samples": manifest["num_images"],
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
        "analysis_source_sha256": _analysis_source_sha256(),
        "inputs": inputs,
    }


def holm_adjust(p_values):
    """Apply Holm's step-down multiplicity adjustment in original order."""

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
    """Paired cluster bootstrap interval and equal-tail bootstrap tail area.

    Each resample draws image clusters with replacement and uses the identical
    sampled rows for both scores and the risk. The percentile interval and the
    two-sided approximate tail probability are computed from that one array of
    AURC differences. For ``B`` draws, the tail probability is::

        min(1, 2 * min((1 + #{delta* <= 0}) / (B + 1),
                       (1 + #{delta* >= 0}) / (B + 1)))

    This confidence-curve tail area is aligned with the equal-tail percentile
    interval. Both are invariant to separate strictly increasing transforms of
    the two confidence scores because every statistic is rank based. It is not
    asserted to be an exact finite-sample null-calibrated p-value.
    """

    left = np.asarray(left_confidences, dtype=float)
    right = np.asarray(right_confidences, dtype=float)
    risk = np.asarray(risks, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or risk.ndim != 1:
        raise ValueError("confidences and risks must be one-dimensional")
    if not left.size or left.size != right.size or left.size != risk.size:
        raise ValueError("paired confidences and risks must have one non-empty length")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.isfinite(risk).all()
    ):
        raise ValueError("confidences and risks must be finite")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)):
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


def _rank_agreement(approximation, reference):
    """Return Spearman and Kendall tau-b, or ``None`` when undefined."""

    approximation = np.asarray(approximation, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if np.unique(approximation).size < 2 or np.unique(reference).size < 2:
        return {"spearman_rho": None, "kendall_tau_b": None}
    approximation_ranks = rankdata(approximation, method="average")
    reference_ranks = rankdata(reference, method="average")
    spearman = float(
        np.clip(np.corrcoef(approximation_ranks, reference_ranks)[0, 1], -1, 1)
    )
    kendall = float(
        np.clip(
            kendalltau(approximation, reference, variant="b", method="auto").statistic,
            -1,
            1,
        )
    )
    if not math.isfinite(spearman) or not math.isfinite(kendall):
        raise RuntimeError("rank agreement unexpectedly became non-finite")
    return {"spearman_rho": spearman, "kendall_tau_b": kendall}


def _dice_quadrature_validation(rows):
    reference = np.asarray([row[DICE_EXACT_REFERENCE] for row in rows], dtype=float)
    methods = {}
    for field, count in DICE_QUADRATURE_METHODS:
        approximation = np.asarray([row[field] for row in rows], dtype=float)
        absolute_error = np.abs(approximation - reference)
        methods[field] = {
            "m": count,
            "num_images": int(reference.size),
            "absolute_error": {
                "mean": float(absolute_error.mean()),
                "median": float(np.median(absolute_error)),
                "p95": float(np.quantile(absolute_error, 0.95, method="linear")),
                "max": float(absolute_error.max()),
            },
            "rank_agreement": _rank_agreement(approximation, reference),
            "exact_match_fraction": float(np.mean(approximation == reference)),
        }
    return {
        "reference": DICE_EXACT_REFERENCE,
        "absolute_error_definition": "per-image |C_Dice,M - C_Dice,Exact|",
        "rank_agreement_definition": (
            "Spearman correlation of average ranks and Kendall tau-b over "
            "image-level confidences; null when either ranking is constant"
        ),
        "exact_match_definition": (
            "fraction with exact floating-point equality and no tolerance"
        ),
        "dice_quadrature": methods,
    }


def analyze_conditions(
    conditions,
    *,
    bootstrap_samples=10_000,
    confidence_level=0.95,
    seed=0,
    bootstrap_workers=4,
    allow_incomplete=False,
):
    if not conditions:
        raise ValueError("at least one condition is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if (
        isinstance(bootstrap_workers, bool)
        or not isinstance(bootstrap_workers, int)
        or bootstrap_workers <= 0
    ):
        raise ValueError("bootstrap_workers must be a positive integer")

    conditions = sorted(
        conditions,
        key=lambda item: (item.dataset, item.condition, str(item.jsonl_path)),
    )
    identifiers = [(item.dataset, item.condition) for item in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("each dataset/condition pair must appear exactly once")
    observed = set(identifiers)
    expected = set(EXPECTED_CONDITIONS)
    undeclared = sorted(observed - expected)
    if undeclared:
        raise ValueError(f"undeclared dataset/condition pairs: {undeclared}")
    if allow_incomplete:
        if len(observed) > len(expected):
            raise ValueError("incomplete analysis exceeds the declared benchmark")
    elif observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(
            "complete analysis requires exactly the 16 declared conditions; "
            f"missing={missing}"
        )

    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "provenance": {
            "binding": "unbound",
            "analysis_source_sha256": _analysis_source_sha256(),
        },
        "analysis": {
            "tie_policy": "analytic expectation over random within-tie order",
            "normalized_aurc": ("(AURC - oracle AURC) / (random AURC - oracle AURC)"),
            "comparisons": (
                "four predeclared adjacent-geometry contrasts; every reported "
                "difference is AURC(left) - AURC(right)"
            ),
            "contrast_definitions": [asdict(contrast) for contrast in CONTRASTS],
            "cross_loss_policy": (
                "all method-by-risk AURCs are reported descriptively; only the "
                "four declared adjacent-geometry contrasts receive bootstrap "
                "intervals and the multiplicity-aware Holm transform"
            ),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_workers": int(bootstrap_workers),
            "confidence_level": float(confidence_level),
            "bootstrap_p_value": (
                "two-sided approximate equal-tail confidence-curve tail "
                "probability from the same paired image-cluster bootstrap draws "
                "as the percentile interval; despite the compatibility field "
                "name, this is not claimed to be an exact null-calibrated p-value"
            ),
            "seed": int(seed),
        },
        "conditions": [],
        "multiple_testing": {},
    }
    family_sizes = {}
    for dataset, _ in identifiers:
        family = HOLM_FAMILY_BY_DATASET[dataset]
        family_sizes[family] = family_sizes.get(family, 0) + len(CONTRASTS)
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
            "jsonl": (
                f"{data.dataset}/{data.condition}/{data.manifest['run_id']}/records.jsonl"
            ),
            "manifest": (
                f"{data.dataset}/{data.condition}/{data.manifest['run_id']}/manifest.json"
            ),
            "jsonl_sha256": data.manifest["jsonl_sha256"],
            "manifest_sha256": _sha256(data.manifest_path),
            "risks": {},
            "comparisons": {},
            "numerical_validation": _dice_quadrature_validation(rows),
        }
        risk_arrays = {}
        for risk_field, risk_label in RISKS:
            risks = np.asarray([row[risk_field] for row in rows], dtype=float)
            risk_arrays[risk_field] = risks
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

        def compute_contrast(contrast):
            risks = risk_arrays[contrast.risk]
            left = np.asarray([row[contrast.left] for row in rows], dtype=float)
            right = np.asarray([row[contrast.right] for row in rows], dtype=float)
            bootstrap_seed = _derived_seed(
                seed, data.dataset, data.condition, contrast.name, "bootstrap"
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
            return contrast, bootstrap

        worker_count = min(bootstrap_workers, len(CONTRASTS))
        if worker_count == 1:
            contrast_results = [compute_contrast(contrast) for contrast in CONTRASTS]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                contrast_results = list(executor.map(compute_contrast, CONTRASTS))
        for contrast, bootstrap in contrast_results:
            holm_family = HOLM_FAMILY_BY_DATASET[data.dataset]
            comparison = {
                "name": contrast.name,
                "risk": contrast.risk,
                "left": contrast.left,
                "right": contrast.right,
                "difference_left_minus_right": bootstrap.difference,
                "bootstrap": asdict(bootstrap),
                "holm_family": holm_family,
                "holm_family_size": family_sizes[holm_family],
                "holm_adjusted_p_value": None,
            }
            condition_result["comparisons"][contrast.name] = comparison
            comparison_records.append(
                (
                    data.dataset,
                    data.condition,
                    contrast,
                    holm_family,
                    comparison,
                )
            )
        result["conditions"].append(condition_result)

    families = {}
    for family in HOLM_FAMILY_DEFINITIONS:
        records = [
            record
            for _, _, _, record_family, record in comparison_records
            if record_family == family
        ]
        if not records:
            continue
        raw_p_values = [record["bootstrap"]["p_value"] for record in records]
        adjusted = holm_adjust(raw_p_values)
        for record, adjusted_p in zip(records, adjusted):
            record["holm_adjusted_p_value"] = adjusted_p
        families[family] = {
            "definition": HOLM_FAMILY_DEFINITIONS[family],
            "num_hypotheses": len(records),
            "raw_bootstrap_p_values": raw_p_values,
            "holm_adjusted_p_values": adjusted,
        }

    hypotheses = []
    for dataset, condition, contrast, family, record in comparison_records:
        hypotheses.append(
            {
                "dataset": dataset,
                "condition": condition,
                "contrast": contrast.name,
                "risk": contrast.risk,
                "left": contrast.left,
                "right": contrast.right,
                "holm_family": family,
                "holm_family_size": family_sizes[family],
                "raw_bootstrap_p_value": record["bootstrap"]["p_value"],
                "holm_adjusted_p_value": record["holm_adjusted_p_value"],
            }
        )
    result["multiple_testing"] = {
        "procedure": (
            "Holm step-down transform of approximate bootstrap tail probabilities, "
            "applied separately within each family; no exact finite-sample FWER "
            "control is claimed"
        ),
        "family_policy": (
            "core and extension experiments are distinct predeclared families; "
            "approximate tail probabilities are never pooled across them"
        ),
        "families": families,
        "total_hypotheses": len(comparison_records),
        "hypotheses": hypotheses,
        "confidence_intervals": (
            "unadjusted equal-tail percentile intervals from the same paired "
            "bootstrap draws; the Holm transform applies only within the named family"
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
        "numerical_reference",
        "absolute_error_mean",
        "absolute_error_median",
        "absolute_error_p95",
        "absolute_error_max",
        "spearman_rho",
        "kendall_tau_b",
        "exact_match_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for condition in result["conditions"]:
            for risk_field, _ in RISKS:
                risk = condition["risks"][risk_field]
                for score_field, _ in METHODS:
                    method = risk["methods"][score_field]
                    numerical = None
                    if risk_field == "risk_dice":
                        numerical = condition["numerical_validation"][
                            "dice_quadrature"
                        ].get(score_field)
                    absolute_error = (
                        None if numerical is None else numerical["absolute_error"]
                    )
                    rank_agreement = (
                        None if numerical is None else numerical["rank_agreement"]
                    )
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
                            "numerical_reference": (
                                ""
                                if numerical is None
                                else condition["numerical_validation"]["reference"]
                            ),
                            "absolute_error_mean": _format_csv_number(
                                None
                                if absolute_error is None
                                else absolute_error["mean"]
                            ),
                            "absolute_error_median": _format_csv_number(
                                None
                                if absolute_error is None
                                else absolute_error["median"]
                            ),
                            "absolute_error_p95": _format_csv_number(
                                None
                                if absolute_error is None
                                else absolute_error["p95"]
                            ),
                            "absolute_error_max": _format_csv_number(
                                None
                                if absolute_error is None
                                else absolute_error["max"]
                            ),
                            "spearman_rho": _format_csv_number(
                                None
                                if rank_agreement is None
                                else rank_agreement["spearman_rho"]
                            ),
                            "kendall_tau_b": _format_csv_number(
                                None
                                if rank_agreement is None
                                else rank_agreement["kendall_tau_b"]
                            ),
                            "exact_match_fraction": _format_csv_number(
                                None
                                if numerical is None
                                else numerical["exact_match_fraction"]
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
    if args.bootstrap_workers <= 0:
        raise ValueError("--bootstrap-workers must be positive")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie strictly between 0 and 1")
    if not args.allow_incomplete and args.inputs is None:
        raise ValueError("complete analysis requires 16 explicit --inputs")
    if not args.allow_incomplete and args.campaign_lock is None:
        raise ValueError("complete analysis requires --campaign-lock")
    if args.inputs is None:
        roots = args.input_dirs or ["outputs/binary"]
        inputs = sorted(
            {path for root in roots for path in Path(root).rglob("*.jsonl")},
            key=str,
        )
    else:
        inputs = [Path(path) for path in args.inputs]
        resolved = [path.resolve() for path in inputs]
        if len(resolved) != len(set(resolved)):
            raise ValueError("--inputs must not contain duplicate paths")
        inputs = sorted(inputs, key=str)
    if not inputs:
        raise FileNotFoundError("no binary JSONL inputs were selected")
    conditions = [load_condition(path) for path in inputs]
    provenance = None
    if args.campaign_lock is not None:
        provenance = validate_campaign_bound_conditions(conditions, args.campaign_lock)
    result = analyze_conditions(
        conditions,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        bootstrap_workers=args.bootstrap_workers,
        allow_incomplete=args.allow_incomplete,
    )
    if provenance is not None:
        result["provenance"] = provenance
    paths = write_outputs(result, args.output_dir)
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
