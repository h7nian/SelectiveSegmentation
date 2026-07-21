"""Rebuild and verify the public three-seed analysis from portable records.

The replay bundle contains exactly thirty condition records: five datasets,
two target models, and training seeds 0, 1, and 2.  Its manifests expose only
scientific identifiers, cohort hashes, field declarations, and content
digests.  This module validates every byte, rejoins the held-out cohorts,
recomputes all AURCs and contrasts, renders both seed tables, and requires all
three reconstructed outputs to equal the released references byte for byte.

Run ``python -m scripts.maintenance.replay_seed`` from an extracted anonymous
artifact.  No arguments are needed for the fixed release layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from scripts.analyze.main import CONTRASTS, METHODS, RISKS
from selectseg.confidence import summarize_aurc


LOCK_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 2
LOCK_ARTIFACT_TYPE = "selectseg.portable_seed_replay_lock"
MANIFEST_ARTIFACT_TYPE = "selectseg.portable_seed_condition"
PUBLIC_ANALYSIS_ARTIFACT_TYPE = "selectseg.binary_seed_public_analysis"
DEFAULT_LOCK = "results/seed_replay.lock.json"
DEFAULT_OUTPUT_DIR = "rebuild/seed_replay"
TARGET_DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
TARGET_CONDITIONS = ("clipseg-target", "deeplabv3-target")
TARGET_MODELS = {
    "clipseg-target": "clipseg",
    "deeplabv3-target": "deeplabv3",
}
TRAINING_SEEDS = (0, 1, 2)
COHORT_JOIN_FIELDS = (
    "sample_id",
    "image_id",
    "image_index",
    "class_index",
    "class_name",
    "height",
    "width",
    "image_diagonal",
    "truth_foreground_fraction",
)
COHORT_FLOAT_FIELDS = frozenset({"image_diagonal", "truth_foreground_fraction"})
COHORT_REL_TOL = 2e-12
COHORT_ABS_TOL = 2e-12
REQUIRED_ROW_FIELDS = frozenset({"schema_version", "run_id", "sample_id", "image_id"})
OPTIONAL_AUXILIARY_FIELDS = frozenset({"risk_hd_pixels", "risk_hd95_pixels"})
SCORE_FIELDS = frozenset(method for method, _ in METHODS)
RISK_FIELDS = frozenset(risk for risk, _ in RISKS)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]*\Z")

_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_provenance",
        "conditions",
        "expected_outputs",
    }
)
_LOCK_CONDITION_FIELDS = frozenset(
    {
        "training_seed",
        "dataset",
        "condition",
        "model",
        "logical_id",
        "manifest_path",
        "manifest_sha256",
        "records_path",
        "records_sha256",
        "source_assembly_manifest_sha256",
        "num_images",
        "sample_id_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "manifest_schema_version",
        "artifact_type",
        "record_schema_version",
        "training_seed",
        "logical_id",
        "dataset",
        "condition",
        "model",
        "split",
        "run_id",
        "num_images",
        "num_rows",
        "records_sha256",
        "sample_id_sha256",
        "score_fields",
        "risk_fields",
        "auxiliary_fields",
        "source_assembly_manifest_sha256",
    }
)
_ANALYSIS_PROVENANCE_FIELDS = frozenset(
    {
        "source_analysis_sha256",
        "downstream_lock_sha256",
        "canonical_seed0",
        "analysis_source_sha256",
    }
)
_CANONICAL_FIELDS = frozenset({"analysis_sha256", "campaign_lock_sha256"})
_EXPECTED_OUTPUT_KEYS = (
    "analysis",
    "robustness_table",
    "gate_table",
)
_EXPECTED_OUTPUT_PATHS = {
    "analysis": "results/seed_robustness_analysis.json",
    "robustness_table": "tables/seed_robustness.tex",
    "gate_table": "tables/seed_sensitivity_main.tex",
}

DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC",
    "tn3k": "TN3K",
}
CONDITION_LABELS = {
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
}
CONTRAST_LABELS = {
    "dice_vs_nhd_under_dice": r"Dice $-$ nHD (Dice risk)",
    "dice_vs_nhd_under_nhd": r"Dice $-$ nHD (nHD risk)",
    "nhd_vs_nhd95_under_nhd": r"nHD $-$ nHD95 (nHD risk)",
    "nhd_vs_nhd95_under_nhd95": r"nHD $-$ nHD95 (nHD95 risk)",
}
GATE_CONTRAST = "nhd_vs_nhd95_under_nhd95"
EXPECTED_REVERSAL_CELLS = frozenset(
    {
        ("clipseg-target", "isic"),
        ("clipseg-target", "tn3k"),
        ("deeplabv3-target", "kvasir"),
        ("deeplabv3-target", "isic"),
        ("deeplabv3-target", "pet"),
    }
)
EXPECTED_DAGGER_CELLS = frozenset(
    {
        ("clipseg-target", "tn3k"),
        ("deeplabv3-target", "isic"),
        ("deeplabv3-target", "pet"),
    }
)


class ReplayValidationError(RuntimeError):
    """Raised when the portable replay closure is incomplete or inconsistent."""


@dataclass(frozen=True)
class PortableCondition:
    manifest: Mapping[str, object]
    rows: tuple[dict, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value, *, location: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value.lower()) is None:
        raise ReplayValidationError(f"{location} must be a SHA-256 digest")
    return value.lower()


def _positive_int(value, *, location: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ReplayValidationError(f"{location} must be a {qualifier} integer")
    return value


def _safe_identifier(value, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_IDENTIFIER.fullmatch(value) is None
        or value.startswith("/")
        or ".." in PurePosixPath(value).parts
        or "\\" in value
    ):
        raise ReplayValidationError(f"{location} must be a safe portable identifier")
    return value


def _safe_path(value, *, location: str) -> str:
    _safe_identifier(value, location=location)
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ReplayValidationError(f"{location} must be normalized")
    return path.as_posix()


def _exact_fields(value, expected, *, location: str):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ReplayValidationError(
            f"{location} must contain exactly {sorted(expected)}"
        )
    return value


def _finite_tree(value, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReplayValidationError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, location=f"{location}[{index}]")


def _strict_json(payload: bytes, *, source: str):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReplayValidationError(f"invalid strict JSON in {source}: {error}") from error
    _finite_tree(value, location=source)
    return value


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _expected_analysis_metadata() -> dict:
    return {
        "estimand": "descriptive target-model training-seed variation over seeds 0,1,2",
        "replication_unit": "one independently trained checkpoint",
        "inference": "none; no image pooling and no seed-level hypothesis test",
        "statistics": "three values, mean, range, and sample standard deviation",
        "aurc_scale": "raw [0,1]; renderers may display 100 x AURC",
        "contrast_definition": "AURC(left score) - AURC(right score)",
        "contrast_definitions": [asdict(contrast) for contrast in CONTRASTS],
        "cohort_join_fields": list(COHORT_JOIN_FIELDS),
    }


def _validate_analysis_provenance(value) -> dict:
    provenance = _exact_fields(
        value, _ANALYSIS_PROVENANCE_FIELDS, location="analysis_provenance"
    )
    for field in (
        "source_analysis_sha256",
        "downstream_lock_sha256",
        "analysis_source_sha256",
    ):
        _digest(provenance[field], location=f"analysis_provenance.{field}")
    canonical = _exact_fields(
        provenance["canonical_seed0"],
        _CANONICAL_FIELDS,
        location="analysis_provenance.canonical_seed0",
    )
    for field in _CANONICAL_FIELDS:
        _digest(canonical[field], location=f"analysis_provenance.canonical_seed0.{field}")
    return provenance


def load_replay_lock(payload: bytes) -> dict:
    lock = _strict_json(payload, source=DEFAULT_LOCK)
    _exact_fields(lock, _LOCK_FIELDS, location="seed replay lock")
    if lock["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ReplayValidationError("unsupported seed replay lock schema")
    if lock["artifact_type"] != LOCK_ARTIFACT_TYPE:
        raise ReplayValidationError("unexpected seed replay lock artifact type")
    _validate_analysis_provenance(lock["analysis_provenance"])

    conditions = lock["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 30:
        raise ReplayValidationError("seed replay lock must contain exactly 30 conditions")
    expected_identities = sorted(
        (seed, dataset, condition)
        for seed in TRAINING_SEEDS
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    )
    observed_identities = []
    paths = set()
    logical_ids = set()
    for index, row in enumerate(conditions):
        location = f"conditions[{index}]"
        _exact_fields(row, _LOCK_CONDITION_FIELDS, location=location)
        seed = _positive_int(
            row["training_seed"], location=f"{location}.training_seed", allow_zero=True
        )
        dataset = row["dataset"]
        condition = row["condition"]
        if seed not in TRAINING_SEEDS or dataset not in TARGET_DATASETS:
            raise ReplayValidationError(f"{location} is outside the fixed seed grid")
        if condition not in TARGET_CONDITIONS or row["model"] != TARGET_MODELS[condition]:
            raise ReplayValidationError(f"{location} has an invalid target condition")
        observed_identities.append((seed, dataset, condition))
        logical_id = _safe_identifier(row["logical_id"], location=f"{location}.logical_id")
        parts = PurePosixPath(logical_id).parts
        if len(parts) != 4 or parts[:3] != (f"seed-{seed}", dataset, condition):
            raise ReplayValidationError(f"{location}.logical_id differs from its identity")
        if logical_id in logical_ids:
            raise ReplayValidationError("seed replay lock reuses a logical identifier")
        logical_ids.add(logical_id)
        for field in ("manifest_path", "records_path"):
            path = _safe_path(row[field], location=f"{location}.{field}")
            if path in paths:
                raise ReplayValidationError("seed replay lock reuses an input path")
            paths.add(path)
        expected_parent = f"results/seed_records/{logical_id}"
        if row["manifest_path"] != f"{expected_parent}/manifest.json":
            raise ReplayValidationError(f"{location}.manifest_path is not canonical")
        if row["records_path"] != f"{expected_parent}/records.jsonl":
            raise ReplayValidationError(f"{location}.records_path is not canonical")
        for field in (
            "manifest_sha256",
            "records_sha256",
            "source_assembly_manifest_sha256",
            "sample_id_sha256",
        ):
            _digest(row[field], location=f"{location}.{field}")
        _positive_int(row["num_images"], location=f"{location}.num_images")
    if observed_identities != expected_identities:
        raise ReplayValidationError("seed replay conditions are not in canonical order")

    outputs = _exact_fields(
        lock["expected_outputs"], _EXPECTED_OUTPUT_KEYS, location="expected_outputs"
    )
    for name in _EXPECTED_OUTPUT_KEYS:
        binding = _exact_fields(
            outputs[name], {"path", "sha256"}, location=f"expected_outputs.{name}"
        )
        if binding["path"] != _EXPECTED_OUTPUT_PATHS[name]:
            raise ReplayValidationError(f"expected_outputs.{name}.path is not canonical")
        _digest(binding["sha256"], location=f"expected_outputs.{name}.sha256")
    return lock


def validate_portable_condition(
    manifest_payload: bytes,
    records_payload: bytes,
    *,
    binding: Mapping[str, object] | None = None,
) -> PortableCondition:
    manifest = _strict_json(manifest_payload, source="portable seed manifest")
    _exact_fields(manifest, _MANIFEST_FIELDS, location="portable seed manifest")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ReplayValidationError("unsupported portable seed manifest schema")
    if manifest["artifact_type"] != MANIFEST_ARTIFACT_TYPE:
        raise ReplayValidationError("unexpected portable seed manifest artifact type")
    if manifest["record_schema_version"] != RECORD_SCHEMA_VERSION:
        raise ReplayValidationError("unsupported portable seed record schema")
    seed = _positive_int(
        manifest["training_seed"], location="manifest.training_seed", allow_zero=True
    )
    dataset = manifest["dataset"]
    condition = manifest["condition"]
    if seed not in TRAINING_SEEDS or dataset not in TARGET_DATASETS:
        raise ReplayValidationError("portable manifest is outside the fixed seed grid")
    if condition not in TARGET_CONDITIONS or manifest["model"] != TARGET_MODELS[condition]:
        raise ReplayValidationError("portable manifest has an invalid target condition")
    if manifest["split"] != "test":
        raise ReplayValidationError("portable seed condition must use the test split")
    logical_id = _safe_identifier(manifest["logical_id"], location="manifest.logical_id")
    run_id = _safe_identifier(manifest["run_id"], location="manifest.run_id")
    if logical_id != f"seed-{seed}/{dataset}/{condition}/{run_id}":
        raise ReplayValidationError("portable manifest logical_id is inconsistent")
    num_images = _positive_int(manifest["num_images"], location="manifest.num_images")
    if manifest["num_rows"] != num_images:
        raise ReplayValidationError("portable manifest row/image counts differ")
    records_sha = _digest(manifest["records_sha256"], location="manifest.records_sha256")
    if _sha256_bytes(records_payload) != records_sha:
        raise ReplayValidationError("portable records SHA-256 mismatch")
    sample_sha = _digest(
        manifest["sample_id_sha256"], location="manifest.sample_id_sha256"
    )
    _digest(
        manifest["source_assembly_manifest_sha256"],
        location="manifest.source_assembly_manifest_sha256",
    )

    declared_scores = manifest["score_fields"]
    declared_risks = manifest["risk_fields"]
    declared_auxiliary = manifest["auxiliary_fields"]
    for field, value in (
        ("score_fields", declared_scores),
        ("risk_fields", declared_risks),
        ("auxiliary_fields", declared_auxiliary),
    ):
        if not isinstance(value, list) or len(value) != len(set(value)):
            raise ReplayValidationError(f"manifest.{field} must be a unique list")
    if set(declared_scores) != SCORE_FIELDS or set(declared_risks) != RISK_FIELDS:
        raise ReplayValidationError("portable manifest score/risk declarations differ")
    if set(declared_auxiliary) - OPTIONAL_AUXILIARY_FIELDS:
        raise ReplayValidationError("portable manifest has unsupported auxiliary fields")

    rows = []
    row_fields = None
    sample_ids = set()
    image_ids = set()
    ordered_sample_ids = []
    try:
        lines = records_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ReplayValidationError("portable records are not UTF-8") from error
    if len(lines) != num_images or any(not line for line in lines):
        raise ReplayValidationError("portable record count or blank-line policy differs")
    required = (
        REQUIRED_ROW_FIELDS
        | SCORE_FIELDS
        | RISK_FIELDS
        | set(declared_auxiliary)
        | set(COHORT_JOIN_FIELDS)
    )
    for line_number, line in enumerate(lines, start=1):
        row = _strict_json(line.encode("utf-8"), source=f"records row {line_number}")
        if not isinstance(row, dict):
            raise ReplayValidationError(f"records row {line_number} is not an object")
        if row_fields is None:
            row_fields = set(row)
            if not required <= row_fields:
                raise ReplayValidationError("portable records lack required fields")
            row_scores = {field for field in row_fields if field.startswith("confidence_")}
            row_risks = {field for field in row_fields if field.startswith("risk_")}
            if row_scores != SCORE_FIELDS or row_risks != RISK_FIELDS | set(declared_auxiliary):
                raise ReplayValidationError("portable record score/risk schema differs")
        elif set(row) != row_fields:
            raise ReplayValidationError("portable records have inconsistent row schemas")
        if row["schema_version"] != RECORD_SCHEMA_VERSION or row["run_id"] != run_id:
            raise ReplayValidationError("portable record envelope differs from its manifest")
        sample_id = row["sample_id"]
        image_id = row["image_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ReplayValidationError("portable records contain an invalid sample_id")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ReplayValidationError("portable records contain an invalid image_id")
        sample_ids.add(sample_id)
        image_ids.add(image_id)
        ordered_sample_ids.append(sample_id)
        for field in SCORE_FIELDS | RISK_FIELDS | set(declared_auxiliary):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReplayValidationError(f"records row {line_number}.{field} is not numeric")
            if not math.isfinite(float(value)):
                raise ReplayValidationError(f"records row {line_number}.{field} is not finite")
        for field in RISK_FIELDS:
            if not 0 <= row[field] <= 1:
                raise ReplayValidationError(f"records row {line_number}.{field} is outside [0,1]")
        for field in declared_auxiliary:
            if row[field] < 0:
                raise ReplayValidationError(f"records row {line_number}.{field} is negative")
        rows.append(row)
    observed_sample_sha = _sha256_bytes("\n".join(ordered_sample_ids).encode("utf-8"))
    if observed_sample_sha != sample_sha:
        raise ReplayValidationError("portable record sample-order digest mismatch")

    if binding is not None:
        comparisons = {
            "training_seed": seed,
            "dataset": dataset,
            "condition": condition,
            "model": manifest["model"],
            "logical_id": logical_id,
            "records_sha256": records_sha,
            "source_assembly_manifest_sha256": manifest[
                "source_assembly_manifest_sha256"
            ],
            "num_images": num_images,
            "sample_id_sha256": sample_sha,
        }
        for field, observed in comparisons.items():
            if binding[field] != observed:
                raise ReplayValidationError(f"portable manifest differs from lock field {field}")
        if _sha256_bytes(manifest_payload) != binding["manifest_sha256"]:
            raise ReplayValidationError("portable manifest SHA-256 differs from lock")
    return PortableCondition(manifest=manifest, rows=tuple(rows))


def _strict_cohort_join(reference: PortableCondition, candidate: PortableCondition, *, context: str) -> None:
    if reference.manifest["sample_id_sha256"] != candidate.manifest["sample_id_sha256"]:
        raise ReplayValidationError(f"{context}: sample-order digest differs")
    if len(reference.rows) != len(candidate.rows):
        raise ReplayValidationError(f"{context}: cohort size differs")
    for left, right in zip(reference.rows, candidate.rows, strict=True):
        for field in COHORT_JOIN_FIELDS:
            if field in COHORT_FLOAT_FIELDS:
                agrees = math.isclose(
                    float(left[field]),
                    float(right[field]),
                    rel_tol=COHORT_REL_TOL,
                    abs_tol=COHORT_ABS_TOL,
                )
            else:
                agrees = left[field] == right[field]
            if not agrees:
                raise ReplayValidationError(f"{context}: cohort field {field!r} differs")


def _summarize_condition(run: PortableCondition) -> dict:
    raw_aurc = {}
    for risk, _ in RISKS:
        risks = [row[risk] for row in run.rows]
        raw_aurc[risk] = {
            method: asdict(summarize_aurc([row[method] for row in run.rows], risks))["aurc"]
            for method, _ in METHODS
        }
    contrasts = {
        contrast.name: raw_aurc[contrast.risk][contrast.left]
        - raw_aurc[contrast.risk][contrast.right]
        for contrast in CONTRASTS
    }
    return {"raw_aurc": raw_aurc, "contrasts": contrasts}


def _direction(value: float) -> int:
    if not math.isfinite(value):
        raise ReplayValidationError("seed summary contains a non-finite value")
    return 1 if value > 0 else -1 if value < 0 else 0


def _three_seed_summary(values: Mapping[int, float]) -> dict:
    if set(values) != set(TRAINING_SEEDS):
        raise ReplayValidationError("seed summary requires exactly seeds 0, 1, and 2")
    ordered = [float(values[seed]) for seed in TRAINING_SEEDS]
    if not all(math.isfinite(value) for value in ordered):
        raise ReplayValidationError("seed summary contains a non-finite value")
    return {
        "values": {str(seed): ordered[seed] for seed in TRAINING_SEEDS},
        "mean": float(statistics.fmean(ordered)),
        "minimum": min(ordered),
        "maximum": max(ordered),
        "range": max(ordered) - min(ordered),
        "sample_standard_deviation": float(statistics.stdev(ordered)),
    }


def _contrast_seed_summary(values: Mapping[int, float]) -> dict:
    summary = _three_seed_summary(values)
    directions = {str(seed): _direction(values[seed]) for seed in TRAINING_SEEDS}
    counts = {
        direction: tuple(directions.values()).count(direction)
        for direction in (-1, 0, 1)
    }
    majority = [direction for direction, count in counts.items() if count >= 2]
    majority_direction = majority[0] if len(majority) == 1 else None
    summary.update(
        {
            "directions": directions,
            "majority_direction": majority_direction,
            "seed0_is_majority_direction": (
                majority_direction is not None and directions["0"] == majority_direction
            ),
            "direction_reversal": -1 in directions.values() and 1 in directions.values(),
        }
    )
    return summary


def _gate_c(cells: Sequence[dict]) -> dict:
    seed0_not_majority = []
    reversal_counts = {contrast.name: 0 for contrast in CONTRASTS}
    for cell in cells:
        for name, summary in cell["summary"]["contrasts"].items():
            if not summary["seed0_is_majority_direction"]:
                seed0_not_majority.append(
                    {
                        "dataset": cell["dataset"],
                        "condition": cell["condition"],
                        "contrast": name,
                    }
                )
            if summary["direction_reversal"]:
                reversal_counts[name] += 1
    reversal_threshold = [name for name, count in reversal_counts.items() if count >= 3]
    fired = bool(seed0_not_majority or reversal_threshold)
    reasons = []
    if seed0_not_majority:
        reasons.append("seed0_not_majority_direction")
    if reversal_threshold:
        reasons.append("at_least_three_conditions_reverse_for_one_contrast")
    return {
        "fired": fired,
        "decision": (
            "move three-seed table to the main results and call the affected "
            "comparison training-sensitive"
            if fired
            else "report direction retention in the main text; keep full values in appendix"
        ),
        "seed0_not_majority_cells": seed0_not_majority,
        "direction_reversal_counts": reversal_counts,
        "contrasts_with_at_least_three_reversals": reversal_threshold,
        "reasons": reasons,
    }


def rebuild_analysis(lock: Mapping[str, object], runs: Mapping[tuple[int, str, str], PortableCondition]) -> dict:
    expected = {
        (seed, dataset, condition)
        for seed in TRAINING_SEEDS
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    }
    if set(runs) != expected:
        raise ReplayValidationError("portable run grid is incomplete")
    for dataset in TARGET_DATASETS:
        for condition in TARGET_CONDITIONS:
            reference = runs[(0, dataset, condition)]
            for seed in (1, 2):
                _strict_cohort_join(
                    reference,
                    runs[(seed, dataset, condition)],
                    context=f"{dataset}/{condition}, seed 0 vs {seed}",
                )

    cells = []
    binding_by_identity = {
        (row["training_seed"], row["dataset"], row["condition"]): row
        for row in lock["conditions"]
    }
    for dataset, condition in sorted(
        (dataset, condition)
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    ):
        summaries = {
            seed: _summarize_condition(runs[(seed, dataset, condition)])
            for seed in TRAINING_SEEDS
        }
        raw_summary = {
            risk: {
                method: _three_seed_summary(
                    {
                        seed: summaries[seed]["raw_aurc"][risk][method]
                        for seed in TRAINING_SEEDS
                    }
                )
                for method, _ in METHODS
            }
            for risk, _ in RISKS
        }
        contrast_summary = {
            contrast.name: _contrast_seed_summary(
                {
                    seed: summaries[seed]["contrasts"][contrast.name]
                    for seed in TRAINING_SEEDS
                }
            )
            for contrast in CONTRASTS
        }
        sources = {}
        for seed in TRAINING_SEEDS:
            binding = binding_by_identity[(seed, dataset, condition)]
            sources[str(seed)] = {
                "logical_id": binding["logical_id"],
                "records_sha256": binding["records_sha256"],
                "manifest_sha256": binding["source_assembly_manifest_sha256"],
            }
        cells.append(
            {
                "dataset": dataset,
                "condition": condition,
                "model": TARGET_MODELS[condition],
                "num_images_per_seed": len(runs[(0, dataset, condition)].rows),
                "sources": sources,
                "summary": {"raw_aurc": raw_summary, "contrasts": contrast_summary},
            }
        )
    result = {
        "schema_version": 1,
        "artifact_type": PUBLIC_ANALYSIS_ARTIFACT_TYPE,
        "analysis": _expected_analysis_metadata(),
        "provenance": lock["analysis_provenance"],
        "cells": cells,
    }
    result["gate_c"] = _gate_c(cells)
    return result


def _robustness_cell(summary: Mapping[str, object]) -> str:
    values = summary["values"]
    return " / ".join(f"{100 * float(values[str(seed)]):+.2f}" for seed in TRAINING_SEEDS)


def _dispersion_cell(summary: Mapping[str, object]) -> str:
    mean = 100 * float(summary["mean"])
    sample_sd = 100 * float(summary["sample_standard_deviation"])
    value_range = 100 * float(summary["range"])
    return rf"{mean:+.2f} $\pm$ {sample_sd:.2f} [{value_range:.2f}]"


def render_robustness_table(analysis: Mapping[str, object]) -> bytes:
    by_key = {(cell["dataset"], cell["condition"]): cell for cell in analysis["cells"]}
    gate = "fired" if analysis["gate_c"]["fired"] else "not fired"
    source_sha = analysis["provenance"]["source_analysis_sha256"]
    lines = [
        "% AUTO-GENERATED by scripts/render_binary_seed_extension.py; DO NOT EDIT.",
        f"% Source analysis.json SHA-256: {source_sha}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Target-model training-seed sensitivity. Panel (a) reports each signed "
        r"AURC contrast for seeds 0 / 1 / 2; panel (b) reports mean $\pm$ sample SD "
        r"[range]. All entries are multiplied by 100 for display only, and negative "
        r"values favor the left score. Seeds are checkpoint-level descriptive replicates "
        r"and are not pooled as images; the machine-readable source also retains the "
        rf"same summaries for every raw AURC. Gate C was {gate}.}}",
        r"\label{tab:seed-robustness}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textit{(a) Seed-specific contrast: seed 0 / seed 1 / seed 2}} \\",
        "Model & Contrast & "
        + " & ".join(DATASET_LABELS[dataset] for dataset in TARGET_DATASETS)
        + r" \\",
        r"\midrule",
    ]
    for condition_index, condition in enumerate(TARGET_CONDITIONS):
        for contrast in CONTRASTS:
            cells = [
                _robustness_cell(
                    by_key[(dataset, condition)]["summary"]["contrasts"][contrast.name]
                )
                for dataset in TARGET_DATASETS
            ]
            lines.append(
                " & ".join(
                    [CONDITION_LABELS[condition], CONTRAST_LABELS[contrast.name], *cells]
                )
                + r" \\"
            )
        if condition_index + 1 < len(TARGET_CONDITIONS):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\textit{(b) Across-seed summary: mean $\pm$ sample SD [range]}} \\",
            r"\midrule",
        ]
    )
    for condition_index, condition in enumerate(TARGET_CONDITIONS):
        for contrast in CONTRASTS:
            cells = [
                _dispersion_cell(
                    by_key[(dataset, condition)]["summary"]["contrasts"][contrast.name]
                )
                for dataset in TARGET_DATASETS
            ]
            lines.append(
                " & ".join(
                    [CONDITION_LABELS[condition], CONTRAST_LABELS[contrast.name], *cells]
                )
                + r" \\"
            )
        if condition_index + 1 < len(TARGET_CONDITIONS):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _gate_cells(analysis: Mapping[str, object], by_key: Mapping[tuple[str, str], dict]):
    reversal_cells = set()
    dagger_cells = set()
    for condition in TARGET_CONDITIONS:
        for dataset in TARGET_DATASETS:
            summary = by_key[(dataset, condition)]["summary"]["contrasts"][GATE_CONTRAST]
            if summary["direction_reversal"]:
                reversal_cells.add((condition, dataset))
                if not summary["seed0_is_majority_direction"]:
                    dagger_cells.add((condition, dataset))
    listed_nonmajority = {
        (entry["condition"], entry["dataset"], entry["contrast"])
        for entry in analysis["gate_c"]["seed0_not_majority_cells"]
    }
    expected_nonmajority = {
        (condition, dataset, GATE_CONTRAST)
        for condition, dataset in EXPECTED_DAGGER_CELLS
    }
    gate = analysis["gate_c"]
    if (
        gate["fired"] is not True
        or gate["direction_reversal_counts"].get(GATE_CONTRAST) != 5
        or gate["contrasts_with_at_least_three_reversals"] != [GATE_CONTRAST]
        or reversal_cells != EXPECTED_REVERSAL_CELLS
        or dagger_cells != EXPECTED_DAGGER_CELLS
        or listed_nonmajority != expected_nonmajority
    ):
        raise ReplayValidationError("recomputed analysis differs from fixed Gate C")
    return reversal_cells, dagger_cells


def _gate_contrast_cell(summary: Mapping[str, object], *, dagger: bool) -> str:
    values = summary["values"]
    displayed = "/".join(
        f"{100 * float(values[str(seed)]):+.3f}" for seed in TRAINING_SEEDS
    )
    marker = r"^{\dagger}" if dagger else ""
    return rf"$\bigl({displayed}\bigr){marker}$"


def render_gate_table(analysis: Mapping[str, object]) -> bytes:
    by_key = {(cell["dataset"], cell["condition"]): cell for cell in analysis["cells"]}
    reversal_cells, dagger_cells = _gate_cells(analysis, by_key)
    source_sha = analysis["provenance"]["source_analysis_sha256"]
    lines = [
        "% AUTO-GENERATED by scripts/render_seed_gate_table.py; DO NOT EDIT.",
        f"% Source seed analysis SHA-256: {source_sha}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Checkpoint-level direction reversals for nHD versus nHD95 under "
        r"nHD95 risk. Each populated cell reports "
        r"$100\!\times\![\operatorname{AURC}(C_{\mathrm{nHD}})-"
        r"\operatorname{AURC}(C_{\mathrm{nHD95}})]$ for seeds 0/1/2; negative "
        r"values favor nHD. Only the five cells that reverse direction are shown; "
        r"all other cells are dashes. $\dagger$ marks the three cells for which "
        r"seed 0 disagrees with the majority direction. These are descriptive "
        r"results from three independently trained checkpoints, not seed-level "
        r"inference.}",
        r"\label{tab:seed-sensitivity-main}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Model & "
        + " & ".join(DATASET_LABELS[dataset] for dataset in TARGET_DATASETS)
        + r" \\",
        r"\midrule",
    ]
    for condition in TARGET_CONDITIONS:
        cells = []
        for dataset in TARGET_DATASETS:
            key = (condition, dataset)
            if key not in reversal_cells:
                cells.append(r"--")
                continue
            summary = by_key[(dataset, condition)]["summary"]["contrasts"][GATE_CONTRAST]
            cells.append(_gate_contrast_cell(summary, dagger=key in dagger_cells))
        lines.append(" & ".join([CONDITION_LABELS[condition], *cells]) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def verify_replay_payloads(
    lock_payload: bytes,
    read_payload: Callable[[str], bytes],
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Recompute all release outputs from a strict payload reader."""

    lock = load_replay_lock(lock_payload)
    runs = {}
    for binding in lock["conditions"]:
        manifest_payload = read_payload(binding["manifest_path"])
        records_payload = read_payload(binding["records_path"])
        run = validate_portable_condition(
            manifest_payload, records_payload, binding=binding
        )
        identity = (
            binding["training_seed"],
            binding["dataset"],
            binding["condition"],
        )
        if identity in runs:
            raise ReplayValidationError("seed replay lock contains duplicate identities")
        runs[identity] = run
    analysis = rebuild_analysis(lock, runs)
    rebuilt = {
        "analysis": _json_bytes(analysis),
        "robustness_table": render_robustness_table(analysis),
        "gate_table": render_gate_table(analysis),
    }
    for name in _EXPECTED_OUTPUT_KEYS:
        binding = lock["expected_outputs"][name]
        expected = read_payload(binding["path"])
        if _sha256_bytes(expected) != binding["sha256"]:
            raise ReplayValidationError(f"released {name} SHA-256 differs from lock")
        if rebuilt[name] != expected:
            raise ReplayValidationError(f"reconstructed {name} is not byte-exact")
    report = {
        "verified": True,
        "condition_count": len(runs),
        "seed_count": len(TRAINING_SEEDS),
        "output_sha256": {
            name: _sha256_bytes(payload) for name, payload in rebuilt.items()
        },
    }
    return rebuilt, report


def _read_regular(root: Path, relative: str) -> bytes:
    path = PurePosixPath(_safe_path(relative, location="replay input path"))
    current = root
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"replay root does not exist: {root}") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ReplayValidationError("replay root must be a real directory")
    for index, part in enumerate(path.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise FileNotFoundError(f"replay input is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ReplayValidationError(f"replay input traverses a link: {relative}")
        last = index == len(path.parts) - 1
        if last and not stat.S_ISREG(mode):
            raise ReplayValidationError(f"replay input is not a regular file: {relative}")
        if not last and not stat.S_ISDIR(mode):
            raise ReplayValidationError(f"replay input parent is not a directory: {relative}")
    return current.read_bytes()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replay_release(
    root: str | os.PathLike[str] = ".",
    *,
    lock_path: str = DEFAULT_LOCK,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    root_path = Path(root).resolve(strict=True)
    lock_payload = _read_regular(root_path, lock_path)

    def read_release_payload(relative: str) -> bytes:
        try:
            return _read_regular(root_path, relative)
        except FileNotFoundError as original_error:
            public_clone_aliases = {
                "tables/seed_robustness.tex": "docs/Tables/seed_robustness.tex",
                "tables/seed_sensitivity_main.tex": (
                    "docs/Tables/seed_sensitivity_main.tex"
                ),
            }
            try:
                alias = public_clone_aliases[relative]
            except KeyError:
                raise original_error
            return _read_regular(root_path, alias)

    rebuilt, report = verify_replay_payloads(
        lock_payload, read_release_payload
    )
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root_path / destination
    output_names = {
        "analysis": "seed_robustness_analysis.json",
        "robustness_table": "seed_robustness.tex",
        "gate_table": "seed_sensitivity_main.tex",
    }
    for name, filename in output_names.items():
        _write_new(destination / filename, rebuilt[name])
    report["output_dir"] = destination.as_posix()
    return report


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="extracted artifact root")
    parser.add_argument("--lock", default=DEFAULT_LOCK, help="portable replay lock")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = replay_release(args.root, lock_path=args.lock, output_dir=args.output_dir)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
