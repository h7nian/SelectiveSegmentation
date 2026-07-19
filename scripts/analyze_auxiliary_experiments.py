"""Strict threshold-robustness and quadrature-convergence analysis.

The canonical M=2,8,32 run supplies gamma=0.5.  Matched gamma=0.3/0.7
runs measure deployment-threshold robustness, while a separately evaluated
M=128 run is the dense midpoint reference.  Every comparison is an exact
sample-id join and fails closed on cohort, prediction, or provenance drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.binary_framework import summarize_aurc  # noqa: E402


SCHEMA_VERSION = 1
EXPECTED_SOURCE_SHA256 = (
    "0c44f7f9a2b483843069add08f80823ea8b2a01814914a9dd0acc71d6b7440d7"
)
JSON_NAME = "auxiliary_experiments.json"
LATEX_NAME = "auxiliary_experiments.tex"

PRIMARY_M_VALUES = (2, 8, 32)
REFERENCE_M = 128
GAMMAS = (0.3, 0.5, 0.7)
RISKS = frozenset({"risk_dice", "risk_nhd95"})
AUXILIARY_RISKS = frozenset({"risk_hd95_pixels"})
BASELINES = frozenset(
    {
        "confidence_sdc",
        "confidence_mean_max_probability",
        "confidence_negative_entropy",
    }
)
PROBABILITY_ONLY_SCORES = frozenset(
    {"confidence_mean_max_probability", "confidence_negative_entropy"}
)
LOSS_SPECS = {
    "dice": ("risk_dice", "confidence_dice_m{}", "Dice"),
    "nhd95": ("risk_nhd95", "confidence_nhd95_m{}", "nHD95"),
}
PRIMARY_SCORES = BASELINES | frozenset(
    f"confidence_{loss}_m{count}"
    for loss in LOSS_SPECS
    for count in PRIMARY_M_VALUES
)
REFERENCE_SCORES = BASELINES | frozenset(
    f"confidence_{loss}_m{REFERENCE_M}" for loss in LOSS_SPECS
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "condition",
        "model",
        "dataset",
        "split",
        "num_images",
        "num_rows",
        "source_sha256",
        "decision_rule",
        "risk_fields",
        "auxiliary_fields",
        "score_fields",
        "quadrature",
        "sample_id_sha256",
        "jsonl_sha256",
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
GAMMA_JOIN_FIELDS = frozenset(
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
        *PROBABILITY_ONLY_SCORES,
    }
)
M128_REQUIRED_COMMON_FIELDS = (
    (REQUIRED_ROW_FIELDS - {"run_id"})
    | RISKS
    | AUXILIARY_RISKS
    | BASELINES
)
MANIFEST_JOIN_FIELDS = (
    "model",
    "dataset",
    "condition",
    "split",
    "num_images",
    "num_rows",
    "checkpoint",
    "base_model",
    "source_sha256",
    "cohort",
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


@dataclass(frozen=True)
class Run:
    records_path: Path
    manifest_path: Path
    manifest: dict
    rows: dict[str, dict]
    row_fields: frozenset[str]

    @property
    def key(self) -> tuple[str, str]:
        return self.manifest["dataset"], self.manifest["condition"]


@dataclass(frozen=True)
class RootRuns:
    runs: dict[tuple[str, str], Run]
    rejected_fingerprints: tuple[dict, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", default="outputs/binary_final")
    parser.add_argument(
        "--gamma03-root",
        default="outputs/binary_thresholds_matched/gamma03",
    )
    parser.add_argument(
        "--gamma07-root",
        default="outputs/binary_thresholds_matched/gamma07",
    )
    parser.add_argument("--m128-root", default="outputs/binary_m128_matched")
    parser.add_argument(
        "--output-dir", default="outputs/binary_auxiliary_analysis"
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


def _load_json(text, *, source):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value, *, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, location=f"{location}[{index}]")


def _digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _nonempty_string(value, *, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _field_set(manifest, field, *, location):
    values = manifest[field]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{location}.{field} must be a nonempty unique list")
    return frozenset(values)


def _validate_quadrature(manifest, expected_m_values, *, location):
    quadrature = manifest["quadrature"]
    expected_keys = {str(value) for value in expected_m_values}
    if not isinstance(quadrature, dict) or set(quadrature) != expected_keys:
        raise ValueError(
            f"{location}.quadrature must contain exactly {sorted(expected_keys)}"
        )
    for count in expected_m_values:
        rule = quadrature[str(count)]
        nodes = [(index + 0.5) / count for index in range(count)]
        weights = [1 / count] * count
        if not isinstance(rule, dict) or rule.get("rule") != "midpoint":
            raise ValueError(f"{location}.quadrature.{count} must use midpoint")
        if set(rule) != {"rule", "nodes", "weights"}:
            raise ValueError(f"{location}.quadrature.{count} has invalid schema")
        if rule["nodes"] != nodes or rule["weights"] != weights:
            raise ValueError(
                f"{location}.quadrature.{count} has unexpected nodes or weights"
            )


def _read_manifest(path):
    manifest = _load_json(Path(path).read_text(), source=str(path))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain one object: {path}")
    _assert_finite(manifest, location=str(path))
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"{path} is missing fields {sorted(missing)}")
    return manifest


def _load_run(
    manifest_path,
    manifest,
    *,
    expected_gamma,
    expected_m_values,
    expected_scores,
):
    manifest_path = Path(manifest_path)
    records_path = manifest_path.with_name("records.jsonl")
    if not records_path.is_file():
        raise FileNotFoundError(f"matching records.jsonl is missing: {manifest_path}")
    if manifest["schema_version"] != SCHEMA_VERSION or isinstance(
        manifest["schema_version"], bool
    ):
        raise ValueError(f"{manifest_path}.schema_version must equal 1")
    for field in ("run_id", "condition", "model", "dataset", "split"):
        _nonempty_string(manifest[field], location=f"{manifest_path}.{field}")
    for field in ("num_images", "num_rows"):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{manifest_path}.{field} must be a positive integer")
    if manifest["num_images"] != manifest["num_rows"]:
        raise ValueError(f"{manifest_path} must declare one row per image")

    if _field_set(manifest, "score_fields", location=manifest_path) != expected_scores:
        raise ValueError(f"{manifest_path}.score_fields has an unexpected schema")
    if _field_set(manifest, "risk_fields", location=manifest_path) != RISKS:
        raise ValueError(f"{manifest_path}.risk_fields has an unexpected schema")
    if (
        _field_set(manifest, "auxiliary_fields", location=manifest_path)
        != AUXILIARY_RISKS
    ):
        raise ValueError(f"{manifest_path}.auxiliary_fields has unexpected schema")

    decision = manifest["decision_rule"]
    if not isinstance(decision, dict) or decision.get("form") != (
        "foreground_probability >= gamma"
    ):
        raise ValueError(f"{manifest_path} has an unexpected decision rule")
    gamma = decision.get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError(f"{manifest_path} has a nonnumeric gamma")
    if float(gamma) != float(expected_gamma):
        raise ValueError(
            f"{manifest_path} declares gamma={gamma}, expected {expected_gamma}"
        )
    _validate_quadrature(manifest, expected_m_values, location=manifest_path)

    expected_jsonl_hash = _digest(
        manifest["jsonl_sha256"], location=f"{manifest_path}.jsonl_sha256"
    )
    if _sha256(records_path) != expected_jsonl_hash:
        raise ValueError(f"SHA-256 mismatch for {records_path}")
    expected_sample_hash = _digest(
        manifest["sample_id_sha256"],
        location=f"{manifest_path}.sample_id_sha256",
    )

    rows = {}
    image_ids = set()
    sample_ids = []
    row_fields = None
    with records_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {records_path}:{line_number}")
            row = _load_json(line, source=f"{records_path}:{line_number}")
            if not isinstance(row, dict):
                raise ValueError(f"{records_path}:{line_number} is not an object")
            _assert_finite(row, location=f"{records_path}:{line_number}")
            current_fields = frozenset(row)
            required = REQUIRED_ROW_FIELDS | RISKS | AUXILIARY_RISKS | expected_scores
            if not required.issubset(current_fields):
                raise ValueError(
                    f"{records_path}:{line_number} lacks fields "
                    f"{sorted(required - current_fields)}"
                )
            if row_fields is None:
                row_fields = current_fields
                row_scores = {
                    field for field in row_fields if field.startswith("confidence_")
                }
                row_risks = {
                    field for field in row_fields if field.startswith("risk_")
                }
                if row_scores != expected_scores or row_risks != (
                    RISKS | AUXILIARY_RISKS
                ):
                    raise ValueError(f"{records_path} manifest/row schema mismatch")
            elif current_fields != row_fields:
                raise ValueError(
                    f"inconsistent row schema at {records_path}:{line_number}"
                )

            sample_id = _nonempty_string(
                row["sample_id"], location=f"{records_path}:{line_number}.sample_id"
            )
            image_id = _nonempty_string(
                row["image_id"], location=f"{records_path}:{line_number}.image_id"
            )
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {records_path}")
            if image_id in image_ids:
                raise ValueError(f"duplicate image_id {image_id!r} in {records_path}")
            if row["schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"row schema mismatch in {records_path}")
            if row["run_id"] != manifest["run_id"]:
                raise ValueError(f"row run_id mismatch in {records_path}")
            for field in RISKS | AUXILIARY_RISKS | expected_scores:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"{records_path}:{line_number}.{field} must be finite numeric"
                    )
            for field in RISKS:
                if not 0 <= row[field] <= 1:
                    raise ValueError(
                        f"{records_path}:{line_number}.{field} must lie in [0, 1]"
                    )
            if row["risk_hd95_pixels"] < 0:
                raise ValueError(f"negative HD95 risk in {records_path}:{line_number}")
            rows[sample_id] = row
            sample_ids.append(sample_id)
            image_ids.add(image_id)

    if len(rows) != manifest["num_rows"]:
        raise ValueError(
            f"row-count mismatch for {records_path}: "
            f"{len(rows)} != {manifest['num_rows']}"
        )
    actual_sample_hash = hashlib.sha256(
        "\n".join(sample_ids).encode("utf-8")
    ).hexdigest()
    if actual_sample_hash != expected_sample_hash:
        raise ValueError(f"sample_id_sha256 mismatch for {records_path}")
    assert row_fields is not None
    return Run(records_path, manifest_path, manifest, rows, row_fields)


def load_root(
    root,
    *,
    expected_gamma,
    expected_m_values,
    expected_scores,
):
    """Load one root, rejecting stale fingerprints before duplicate selection."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    manifest_paths = sorted(root.rglob("manifest.json"))
    records_parents = {path.parent for path in root.rglob("records.jsonl")}
    manifest_parents = {path.parent for path in manifest_paths}
    if records_parents != manifest_parents:
        raise ValueError(f"incomplete records/manifest pairs under {root}")
    if not manifest_paths:
        raise ValueError(f"no completed runs under {root}")

    runs = {}
    rejected = []
    for manifest_path in manifest_paths:
        manifest = _read_manifest(manifest_path)
        observed_source = _digest(
            manifest["source_sha256"],
            location=f"{manifest_path}.source_sha256",
        )
        if observed_source != EXPECTED_SOURCE_SHA256:
            rejected.append(
                {
                    "manifest": str(manifest_path),
                    "dataset": manifest.get("dataset"),
                    "condition": manifest.get("condition"),
                    "run_id": manifest.get("run_id"),
                    "observed_source_sha256": observed_source,
                    "required_source_sha256": EXPECTED_SOURCE_SHA256,
                }
            )
            continue
        run = _load_run(
            manifest_path,
            manifest,
            expected_gamma=expected_gamma,
            expected_m_values=expected_m_values,
            expected_scores=expected_scores,
        )
        if run.key in runs:
            raise ValueError(f"duplicate accepted dataset/condition {run.key!r} in {root}")
        runs[run.key] = run
    if not runs:
        sources = sorted(item["observed_source_sha256"] for item in rejected)
        raise ValueError(
            f"no runs under {root} match required source_sha256 "
            f"{EXPECTED_SOURCE_SHA256}; rejected={sources}"
        )
    return RootRuns(runs, tuple(rejected))


def _condition_sort_key(key):
    dataset, condition = key
    return (
        DATASET_ORDER.get(dataset, len(DATASET_ORDER)),
        dataset,
        CONDITION_ORDER.get(condition, len(CONDITION_ORDER)),
        condition,
    )


def _require_condition_set(reference, candidate, *, context):
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate), key=_condition_sort_key)
        extra = sorted(set(candidate) - set(reference), key=_condition_sort_key)
        raise ValueError(
            f"{context} condition-set mismatch: missing={missing}, extra={extra}"
        )


def _join_rows(reference, candidate, *, fields, context):
    if set(reference.rows) != set(candidate.rows):
        missing = sorted(set(reference.rows) - set(candidate.rows))
        extra = sorted(set(candidate.rows) - set(reference.rows))
        raise ValueError(
            f"{context} sample_id mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for sample_id in sorted(reference.rows):
        for field in sorted(fields):
            if field not in reference.rows[sample_id] or field not in (
                candidate.rows[sample_id]
            ):
                raise ValueError(f"{context} is missing join field {field!r}")
            if reference.rows[sample_id][field] != candidate.rows[sample_id][field]:
                raise ValueError(
                    f"{context} field {field!r} differs for sample_id={sample_id!r}"
                )


def _join_manifests(reference, candidate, *, context):
    for field in MANIFEST_JOIN_FIELDS:
        if field not in reference.manifest or field not in candidate.manifest:
            raise ValueError(f"{context} is missing manifest field {field!r}")
        if reference.manifest[field] != candidate.manifest[field]:
            raise ValueError(f"{context} manifest field {field!r} differs")


def _aurc(run, *, score_field, risk_field):
    sample_ids = sorted(run.rows)
    values = summarize_aurc(
        [run.rows[sample_id][score_field] for sample_id in sample_ids],
        [run.rows[sample_id][risk_field] for sample_id in sample_ids],
    )
    return asdict(values)


def _spearman(left, right):
    left_rank = rankdata(np.asarray(left, dtype=float), method="average")
    right_rank = rankdata(np.asarray(right, dtype=float), method="average")
    if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _mean(values):
    clean = [value for value in values if value is not None]
    return None if not clean else float(np.mean(clean))


def _condition_analysis(key, gamma_runs, reference_run):
    primary = gamma_runs[0.5]
    sample_ids = sorted(primary.rows)
    result = {
        "dataset": key[0],
        "condition": key[1],
        "num_samples": len(sample_ids),
        "sources": {
            "gamma": {
                format(gamma, ".1f"): str(run.manifest_path)
                for gamma, run in sorted(gamma_runs.items())
            },
            "m128": str(reference_run.manifest_path),
        },
        "threshold_robustness": {},
        "convergence_to_m128": {},
    }

    for loss, (risk_field, score_template, label) in LOSS_SPECS.items():
        score_field = score_template.format(32)
        gamma_summaries = {}
        score_vectors = {}
        for gamma, run in sorted(gamma_runs.items()):
            gamma_key = format(gamma, ".1f")
            risks = [run.rows[sample_id][risk_field] for sample_id in sample_ids]
            scores = [run.rows[sample_id][score_field] for sample_id in sample_ids]
            gamma_summaries[gamma_key] = {
                **_aurc(run, score_field=score_field, risk_field=risk_field),
                "mean_risk": float(np.mean(risks)),
            }
            score_vectors[gamma_key] = scores
        pairwise = {}
        gamma_keys = sorted(score_vectors, key=float)
        for left_index, left in enumerate(gamma_keys):
            for right in gamma_keys[left_index + 1 :]:
                pairwise[f"{left}_vs_{right}"] = _spearman(
                    score_vectors[left], score_vectors[right]
                )
        result["threshold_robustness"][loss] = {
            "label": label,
            "risk_field": risk_field,
            "score_field": score_field,
            "gammas": gamma_summaries,
            "pairwise_spearman": pairwise,
            "mean_pairwise_spearman": _mean(pairwise.values()),
            "min_pairwise_spearman": (
                None
                if all(value is None for value in pairwise.values())
                else min(value for value in pairwise.values() if value is not None)
            ),
        }

        reference_field = score_template.format(REFERENCE_M)
        reference_scores = np.asarray(
            [reference_run.rows[sample_id][reference_field] for sample_id in sample_ids]
        )
        reference_aurc = _aurc(
            reference_run,
            score_field=reference_field,
            risk_field=risk_field,
        )
        approximations = {}
        for count in PRIMARY_M_VALUES:
            score_field = score_template.format(count)
            scores = np.asarray(
                [primary.rows[sample_id][score_field] for sample_id in sample_ids]
            )
            errors = np.abs(scores - reference_scores)
            summary = _aurc(primary, score_field=score_field, risk_field=risk_field)
            signed_gap = summary["aurc"] - reference_aurc["aurc"]
            approximations[str(count)] = {
                "score_field": score_field,
                "mae": float(np.mean(errors)),
                "q95_absolute_error": float(np.quantile(errors, 0.95)),
                "spearman": _spearman(scores, reference_scores),
                "aurc": summary["aurc"],
                "m128_aurc": reference_aurc["aurc"],
                "aurc_minus_m128": signed_gap,
                "absolute_aurc_gap": abs(signed_gap),
            }
        result["convergence_to_m128"][loss] = {
            "label": label,
            "risk_field": risk_field,
            "reference_score_field": reference_field,
            "reference_aurc": reference_aurc,
            "approximations": approximations,
        }
    return result


def _aggregate_by_dataset(conditions):
    result = {}
    for dataset in sorted({item["dataset"] for item in conditions}, key=lambda value: (
        DATASET_ORDER.get(value, len(DATASET_ORDER)),
        value,
    )):
        items = [item for item in conditions if item["dataset"] == dataset]
        aggregate = {
            "num_conditions": len(items),
            "conditions": [item["condition"] for item in items],
            "aggregation": "unweighted macro mean of condition-level metrics",
            "threshold_robustness": {},
            "convergence_to_m128": {},
        }
        for loss in LOSS_SPECS:
            threshold = {"gammas": {}}
            for gamma in map(lambda value: format(value, ".1f"), GAMMAS):
                entries = [
                    item["threshold_robustness"][loss]["gammas"][gamma]
                    for item in items
                ]
                threshold["gammas"][gamma] = {
                    "mean_aurc": _mean(entry["aurc"] for entry in entries),
                    "mean_risk": _mean(entry["mean_risk"] for entry in entries),
                    "mean_normalized_aurc": _mean(
                        entry["normalized_aurc"] for entry in entries
                    ),
                }
            threshold["mean_pairwise_spearman"] = _mean(
                item["threshold_robustness"][loss]["mean_pairwise_spearman"]
                for item in items
            )
            threshold["mean_min_pairwise_spearman"] = _mean(
                item["threshold_robustness"][loss]["min_pairwise_spearman"]
                for item in items
            )
            aggregate["threshold_robustness"][loss] = threshold

            convergence = {"approximations": {}}
            for count in map(str, PRIMARY_M_VALUES):
                entries = [
                    item["convergence_to_m128"][loss]["approximations"][count]
                    for item in items
                ]
                convergence["approximations"][count] = {
                    "mean_mae": _mean(entry["mae"] for entry in entries),
                    "mean_q95_absolute_error": _mean(
                        entry["q95_absolute_error"] for entry in entries
                    ),
                    "mean_spearman": _mean(entry["spearman"] for entry in entries),
                    "mean_aurc_minus_m128": _mean(
                        entry["aurc_minus_m128"] for entry in entries
                    ),
                    "mean_absolute_aurc_gap": _mean(
                        entry["absolute_aurc_gap"] for entry in entries
                    ),
                    "num_defined_spearman": sum(
                        entry["spearman"] is not None for entry in entries
                    ),
                }
            aggregate["convergence_to_m128"][loss] = convergence
        result[dataset] = aggregate
    return result


def analyze(primary_root, gamma03_root, gamma07_root, m128_root):
    primary_bundle = load_root(
        primary_root,
        expected_gamma=0.5,
        expected_m_values=PRIMARY_M_VALUES,
        expected_scores=PRIMARY_SCORES,
    )
    gamma03_bundle = load_root(
        gamma03_root,
        expected_gamma=0.3,
        expected_m_values=PRIMARY_M_VALUES,
        expected_scores=PRIMARY_SCORES,
    )
    gamma07_bundle = load_root(
        gamma07_root,
        expected_gamma=0.7,
        expected_m_values=PRIMARY_M_VALUES,
        expected_scores=PRIMARY_SCORES,
    )
    m128_bundle = load_root(
        m128_root,
        expected_gamma=0.5,
        expected_m_values=(REFERENCE_M,),
        expected_scores=REFERENCE_SCORES,
    )
    primary = primary_bundle.runs
    bundles = {
        "gamma03": gamma03_bundle,
        "gamma07": gamma07_bundle,
        "m128": m128_bundle,
    }
    for label, bundle in bundles.items():
        _require_condition_set(primary, bundle.runs, context=label)

    conditions = []
    for key in sorted(primary, key=_condition_sort_key):
        reference = primary[key]
        gamma_runs = {
            0.3: gamma03_bundle.runs[key],
            0.5: reference,
            0.7: gamma07_bundle.runs[key],
        }
        for gamma, candidate in gamma_runs.items():
            _join_manifests(
                reference,
                candidate,
                context=f"gamma={gamma}, condition={key}",
            )
            _join_rows(
                reference,
                candidate,
                fields=GAMMA_JOIN_FIELDS,
                context=f"gamma={gamma}, condition={key}",
            )

        m128 = m128_bundle.runs[key]
        _join_manifests(reference, m128, context=f"M=128, condition={key}")
        common_fields = (reference.row_fields & m128.row_fields) - {"run_id"}
        if not M128_REQUIRED_COMMON_FIELDS.issubset(common_fields):
            raise ValueError(
                f"M=128, condition={key} lacks common fields "
                f"{sorted(M128_REQUIRED_COMMON_FIELDS - common_fields)}"
            )
        _join_rows(
            reference,
            m128,
            fields=common_fields,
            context=f"M=128, condition={key}",
        )
        conditions.append(_condition_analysis(key, gamma_runs, m128))

    all_bundles = {"primary": primary_bundle, **bundles}
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "expected_source_sha256": EXPECTED_SOURCE_SHA256,
            "gammas": list(GAMMAS),
            "primary_m_values": list(PRIMARY_M_VALUES),
            "reference_m": REFERENCE_M,
            "tie_policy": "analytic expectation over random within-score-tie order",
            "dataset_aggregation": (
                "unweighted macro mean of condition-level metrics; "
                "per-condition values are retained"
            ),
            "m128_join": (
                "exact sample_id join and exact equality of every common row field "
                "except run_id"
            ),
        },
        "rejected_fingerprint_candidates": {
            label: list(bundle.rejected_fingerprints)
            for label, bundle in all_bundles.items()
        },
        "conditions": conditions,
    }
    result["datasets"] = _aggregate_by_dataset(conditions)
    return result


def _latex_number(value, digits=4):
    return "--" if value is None else f"{value:.{digits}f}"


def _latex_header(datasets):
    labels = [
        f"{DATASET_LABELS.get(dataset, dataset)} ({datasets[dataset]['num_conditions']})"
        for dataset in datasets
    ]
    return [
        rf"\begin{{tabular}}{{l*{{{len(datasets)}}}{{c}}}}",
        r"\toprule",
        "Metric & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]


def _threshold_table(result):
    datasets = result["datasets"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Deployment-threshold robustness. Entries are condition-macro "
            r"means; the number of conditions appears in parentheses. Lower AURC "
            r"and risk are better, whereas higher Spearman $\rho$ is better.}"
        ),
        r"\label{tab:aux-threshold-robustness}",
        r"{\small",
        *_latex_header(datasets),
    ]
    for loss_index, (loss, (_, _, label)) in enumerate(LOSS_SPECS.items()):
        if loss_index:
            lines.append(r"\midrule")
        for gamma in map(lambda value: format(value, ".1f"), GAMMAS):
            aurc = [
                datasets[dataset]["threshold_robustness"][loss]["gammas"][gamma][
                    "mean_aurc"
                ]
                for dataset in datasets
            ]
            risk = [
                datasets[dataset]["threshold_robustness"][loss]["gammas"][gamma][
                    "mean_risk"
                ]
                for dataset in datasets
            ]
            lines.append(
                f"{label}-M32 AURC, $\\gamma={gamma}$ & "
                + " & ".join(map(_latex_number, aurc))
                + r" \\"
            )
            lines.append(
                f"Mean {label} risk, $\\gamma={gamma}$ & "
                + " & ".join(map(_latex_number, risk))
                + r" \\"
            )
        rho = [
            datasets[dataset]["threshold_robustness"][loss][
                "mean_min_pairwise_spearman"
            ]
            for dataset in datasets
        ]
        lines.append(
            f"{label}-M32 minimum pairwise $\\rho$ & "
            + " & ".join(map(_latex_number, rho))
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return lines


def _convergence_table(result):
    datasets = result["datasets"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Convergence to the matched $M=128$ midpoint reference. "
            r"Entries are condition-macro means. Lower MAE, 95th-percentile "
            r"absolute error (Q95), and absolute AURC gap are better; higher "
            r"Spearman $\rho$ is better.}"
        ),
        r"\label{tab:aux-m128-convergence}",
        r"{\small",
        *_latex_header(datasets),
    ]
    metrics = (
        ("mean_mae", "MAE"),
        ("mean_q95_absolute_error", "Q95"),
        ("mean_spearman", r"Spearman $\rho$"),
        ("mean_absolute_aurc_gap", r"$|\Delta\mathrm{AURC}|$"),
    )
    for loss_index, (loss, (_, _, label)) in enumerate(LOSS_SPECS.items()):
        if loss_index:
            lines.append(r"\midrule")
        for count in map(str, PRIMARY_M_VALUES):
            for field, metric_label in metrics:
                values = [
                    datasets[dataset]["convergence_to_m128"][loss][
                        "approximations"
                    ][count][field]
                    for dataset in datasets
                ]
                lines.append(
                    f"{label}-M{count}: {metric_label} & "
                    + " & ".join(map(_latex_number, values))
                    + r" \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return lines


def write_latex(result, path, *, json_sha256):
    lines = [
        "% AUTO-GENERATED by scripts/analyze_auxiliary_experiments.py.",
        f"% Source {JSON_NAME} SHA-256: {json_sha256}",
        *_threshold_table(result),
        "",
        *_convergence_table(result),
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    result = analyze(
        args.primary_root,
        args.gamma03_root,
        args.gamma07_root,
        args.m128_root,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_NAME
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    latex_path = output_dir / LATEX_NAME
    write_latex(result, latex_path, json_sha256=_sha256(json_path))
    print(f"saved {json_path}")
    print(f"saved {latex_path}")


if __name__ == "__main__":
    main()
