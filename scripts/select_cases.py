"""Predeclare and select qualitative binary-segmentation cases without pixels.

This stage validates the immutable main campaign, all 16 explicit assembled
``records.jsonl`` files, and the 16 lock-listed frozen-artifact manifests.  It
then selects cases from only the ten target-adapted conditions using numerical
rules declared in :data:`SELECTION_RULES`.  Crucially, it never opens an NPZ
payload: visual arrays are read only by the separate renderer after the sample
identities have been committed to ``selection.json``.

The output is content addressed and publication is atomic.  Existing output
directories are never reused or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from scripts.analyze.main import (
    EXPECTED_CONDITIONS,
    ConditionData,
    load_analysis_campaign_lock,
    load_condition,
    validate_campaign_bound_conditions,
)
from selectseg.artifacts import load_binary_artifact


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.binary_qualitative_case_selection"
TARGET_CONDITIONS = ("clipseg-target", "deeplabv3-target")
DATASET_ORDER = ("pet", "kvasir", "fives", "isic", "tn3k")
CASE_ORDER = (
    "dice_vs_nhd_rank_disagreement",
    "nhd_vs_nhd95_rank_disagreement",
    "empty_action",
    "confident_failure",
)
CONDITION_PRIORITY = {name: index for index, name in enumerate(TARGET_CONDITIONS)}
MATCHED_LOSSES = (
    ("dice", "risk_dice", "confidence_dice_m32"),
    ("nhd", "risk_nhd", "confidence_nhd_m32"),
    ("nhd95", "risk_nhd95", "confidence_nhd95_m32"),
)
LOSS_PRIORITY = {name: index for index, (name, _, _) in enumerate(MATCHED_LOSSES)}

SELECTION_RULES = {
    "rule_version": "qualitative-target-v1",
    "condition_scope": (
        "validate all 16 declared benchmark conditions; select and headline only "
        "clipseg-target and deeplabv3-target on each dataset"
    ),
    "rank_definition": (
        "within each target condition, rank confidence from safest to riskiest "
        "using average ranks for exact ties, then normalize (rank-1)/(n-1)"
    ),
    "case_priority": list(CASE_ORDER),
    "case_rules": {
        "dice_vs_nhd_rank_disagreement": (
            "maximize the absolute difference between condition-local normalized "
            "safety ranks of confidence_dice_m32 and confidence_nhd_m32"
        ),
        "nhd_vs_nhd95_rank_disagreement": (
            "maximize the absolute difference between condition-local normalized "
            "safety ranks of confidence_nhd_m32 and confidence_nhd95_m32"
        ),
        "empty_action": (
            "among rows with prediction_foreground_fraction exactly zero, maximize "
            "the realized matched loss over Dice, nHD, and nHD95"
        ),
        "confident_failure": (
            "maximize observed matched loss minus predicted working risk over the "
            "three M32 loss-indexed scores, where predicted risk=-confidence"
        ),
    },
    "duplicate_policy": (
        "process case types in case_priority order and take the highest-ranked "
        "candidate whose underlying sample_id has not yet been used in that "
        "dataset; if every eligible candidate is already used, take the original "
        "top candidate and mark duplicate_fallback_used=true"
    ),
    "tie_breaks": (
        "after the descending selection objective: condition priority "
        "clipseg-target then deeplabv3-target, sample_id lexicographic order, "
        "then matched-loss priority Dice, nHD, nHD95 when applicable"
    ),
    "missing_empty_action_policy": (
        "emit an unavailable case entry; never substitute a nonempty prediction"
    ),
    "pixel_access_policy": (
        "selection reads assembled scalar records and frozen-artifact manifests "
        "only; it does not open probability/truth NPZ payloads"
    ),
    "visual_selection_policy": (
        "no image is selected by visual appeal, appearance, or manual review"
    ),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="RECORDS_JSONL",
        help="exactly 16 explicit assembled records.jsonl paths",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/binary_qualitative_cases",
    )
    return parser.parse_args(argv)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        root / "scripts/analyze/main.py",
        root / "selectseg/artifacts.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_all_conditions(paths: Sequence[str | os.PathLike[str]]) -> list[ConditionData]:
    if len(paths) != len(EXPECTED_CONDITIONS):
        raise ValueError("qualitative selection requires exactly 16 explicit inputs")
    if any(any(token in str(path) for token in "*?[]") for path in paths):
        raise ValueError("input paths must be explicit; glob expressions are forbidden")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("records.jsonl inputs must be distinct")
    if any(path.name != "records.jsonl" for path in resolved):
        raise ValueError("every input must be named records.jsonl")

    loaded = [load_condition(path) for path in resolved]
    by_key: dict[tuple[str, str], ConditionData] = {}
    for item in loaded:
        key = item.dataset, item.condition
        if key in by_key:
            raise ValueError(f"duplicate dataset/condition input {key}")
        by_key[key] = item
    if set(by_key) != set(EXPECTED_CONDITIONS):
        missing = sorted(set(EXPECTED_CONDITIONS) - set(by_key))
        extra = sorted(set(by_key) - set(EXPECTED_CONDITIONS))
        raise ValueError(f"condition set mismatch: missing={missing}, extra={extra}")
    return [by_key[key] for key in EXPECTED_CONDITIONS]


def normalized_safety_ranks(scores: Sequence[float]) -> np.ndarray:
    """Return condition-local safest-to-riskiest average ranks in ``[0, 1]``."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if values.size == 1:
        return np.zeros(1, dtype=np.float64)
    return (rankdata(-values, method="average") - 1.0) / (values.size - 1.0)


def _finite_float(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _condition_candidates(data: ConditionData) -> list[dict[str, Any]]:
    required = {
        "sample_id",
        "image_id",
        "image_index",
        "height",
        "width",
        "prediction_foreground_fraction",
        "truth_foreground_fraction",
        "risk_dice",
        "risk_nhd",
        "risk_nhd95",
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    }
    for index, row in enumerate(data.rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{data.jsonl_path}:{index + 1} lacks {missing}")

    score_fields = (
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    )
    ranks = {
        field: normalized_safety_ranks(
            [_finite_float(row[field], location=f"row.{field}") for row in data.rows]
        )
        for field in score_fields
    }
    result = []
    for index, row in enumerate(data.rows):
        prediction_fraction = _finite_float(
            row["prediction_foreground_fraction"],
            location="row.prediction_foreground_fraction",
        )
        truth_fraction = _finite_float(
            row["truth_foreground_fraction"],
            location="row.truth_foreground_fraction",
        )
        if not 0.0 <= prediction_fraction <= 1.0:
            raise ValueError("prediction_foreground_fraction must lie in [0, 1]")
        if not 0.0 <= truth_fraction <= 1.0:
            raise ValueError("truth_foreground_fraction must lie in [0, 1]")
        risks = {
            field: _finite_float(row[field], location=f"row.{field}")
            for _, field, _ in MATCHED_LOSSES
        }
        scores = {
            field: _finite_float(row[field], location=f"row.{field}")
            for _, _, field in MATCHED_LOSSES
        }
        result.append(
            {
                "dataset": data.dataset,
                "condition": data.condition,
                "model": data.manifest["model"],
                "sample_id": str(row["sample_id"]),
                "image_id": str(row["image_id"]),
                "image_index": int(row["image_index"]),
                "height": int(row["height"]),
                "width": int(row["width"]),
                "prediction_foreground_fraction": prediction_fraction,
                "truth_foreground_fraction": truth_fraction,
                "scores": scores,
                "risks": risks,
                "normalized_safety_ranks": {
                    field: float(values[index]) for field, values in ranks.items()
                },
            }
        )
    return result


def _rank_disagreement_candidates(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        left_rank = float(row["normalized_safety_ranks"][left])
        right_rank = float(row["normalized_safety_ranks"][right])
        candidates.append(
            {
                "row": row,
                "objective": abs(left_rank - right_rank),
                "objective_details": {
                    "left_score_field": left,
                    "right_score_field": right,
                    "left_normalized_safety_rank": left_rank,
                    "right_normalized_safety_rank": right_rank,
                    "absolute_normalized_rank_difference": abs(left_rank - right_rank),
                },
                "matched_loss": None,
            }
        )
    return candidates


def build_case_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Build predeclared candidate lists without accessing any pixel arrays."""

    candidates: dict[str, list[dict]] = {
        "dice_vs_nhd_rank_disagreement": _rank_disagreement_candidates(
            rows, "confidence_dice_m32", "confidence_nhd_m32"
        ),
        "nhd_vs_nhd95_rank_disagreement": _rank_disagreement_candidates(
            rows, "confidence_nhd_m32", "confidence_nhd95_m32"
        ),
        "empty_action": [],
        "confident_failure": [],
    }
    for row in rows:
        if float(row["prediction_foreground_fraction"]) == 0.0:
            for loss, risk_field, score_field in MATCHED_LOSSES:
                risk = float(row["risks"][risk_field])
                candidates["empty_action"].append(
                    {
                        "row": row,
                        "objective": risk,
                        "objective_details": {
                            "matched_loss": loss,
                            "observed_loss_field": risk_field,
                            "observed_matched_loss": risk,
                        },
                        "matched_loss": loss,
                    }
                )
        for loss, risk_field, score_field in MATCHED_LOSSES:
            risk = float(row["risks"][risk_field])
            score = float(row["scores"][score_field])
            predicted_risk = -score
            gap = risk - predicted_risk
            candidates["confident_failure"].append(
                {
                    "row": row,
                    "objective": gap,
                    "objective_details": {
                        "matched_loss": loss,
                        "observed_loss_field": risk_field,
                        "score_field": score_field,
                        "observed_matched_loss": risk,
                        "predicted_working_risk": predicted_risk,
                        "observed_minus_predicted_risk_gap": gap,
                    },
                    "matched_loss": loss,
                }
            )
    return candidates


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple:
    row = candidate["row"]
    matched_loss = candidate.get("matched_loss")
    loss_order = -1 if matched_loss is None else LOSS_PRIORITY[matched_loss]
    return (
        -float(candidate["objective"]),
        CONDITION_PRIORITY[str(row["condition"])],
        str(row["sample_id"]),
        loss_order,
    )


def choose_dataset_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose four case types sequentially, avoiding repeated images if possible."""

    candidates = build_case_candidates(rows)
    used_sample_ids: set[str] = set()
    selected = []
    for case_type in CASE_ORDER:
        ordered = sorted(candidates[case_type], key=_candidate_sort_key)
        if not ordered:
            selected.append(
                {
                    "case_type": case_type,
                    "status": "unavailable",
                    "reason": "no deployed empty action exists in either target condition",
                    "duplicate_fallback_used": False,
                }
            )
            continue
        choice = next(
            (
                candidate
                for candidate in ordered
                if str(candidate["row"]["sample_id"]) not in used_sample_ids
            ),
            None,
        )
        duplicate_fallback = choice is None
        if choice is None:
            choice = ordered[0]
        row = choice["row"]
        used_sample_ids.add(str(row["sample_id"]))
        selected.append(
            {
                "case_type": case_type,
                "status": "selected",
                "duplicate_fallback_used": duplicate_fallback,
                "selection_objective": float(choice["objective"]),
                "selection_objective_details": choice["objective_details"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "model": row["model"],
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "image_index": row["image_index"],
                "height": row["height"],
                "width": row["width"],
                "prediction_foreground_fraction": row[
                    "prediction_foreground_fraction"
                ],
                "truth_foreground_fraction": row["truth_foreground_fraction"],
                "scores": dict(row["scores"]),
                "risks": dict(row["risks"]),
                "normalized_safety_ranks": dict(
                    row["normalized_safety_ranks"]
                ),
            }
        )
    return selected


def _validate_artifact_manifests(
    conditions: Sequence[ConditionData], campaign_lock: str | os.PathLike[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    _, _, _, _, locked = load_analysis_campaign_lock(campaign_lock)
    condition_by_key = {(item.dataset, item.condition): item for item in conditions}
    validated: dict[tuple[str, str], dict[str, Any]] = {}
    for key in EXPECTED_CONDITIONS:
        lock_entry = locked[key]
        manifest_path = Path(lock_entry["manifest_path"])
        artifact = load_binary_artifact(manifest_path, validate_payloads=False)
        if artifact.manifest_sha256 != str(lock_entry["manifest_sha256"]).lower():
            raise ValueError(f"frozen artifact manifest hash differs from lock for {key}")
        manifest = artifact.manifest
        expected = {
            "artifact_id": lock_entry["artifact_id"],
            "dataset": key[0],
            "condition": key[1],
            "model": lock_entry["model"],
            "split": lock_entry["split"],
            "num_samples": lock_entry["num_samples"],
            "sample_id_sha256": lock_entry["sample_id_sha256"],
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"artifact {key} field {field} differs from the lock")
        entries = manifest["samples"]
        row_ids = [str(row["sample_id"]) for row in condition_by_key[key].rows]
        artifact_ids = [str(entry["sample_id"]) for entry in entries]
        if row_ids != artifact_ids:
            raise ValueError(f"artifact/assembly sample order differs for {key}")
        validated[key] = {
            "manifest_path": _portable_path(artifact.manifest_path),
            "manifest_sha256": artifact.manifest_sha256,
            "artifact_id": manifest["artifact_id"],
            "samples": {str(entry["sample_id"]): dict(entry) for entry in entries},
        }
    return validated


def build_selection(
    input_paths: Sequence[str | os.PathLike[str]],
    *,
    campaign_lock: str | os.PathLike[str],
) -> dict[str, Any]:
    conditions = _load_all_conditions(input_paths)
    campaign_provenance = validate_campaign_bound_conditions(
        conditions, campaign_lock
    )
    artifact_by_key = _validate_artifact_manifests(conditions, campaign_lock)
    data_by_key = {(item.dataset, item.condition): item for item in conditions}
    assembly_provenance = {
        (entry["dataset"], entry["condition"]): entry
        for entry in campaign_provenance["inputs"]
    }

    datasets = []
    for dataset in DATASET_ORDER:
        numerical_rows = []
        for condition in TARGET_CONDITIONS:
            numerical_rows.extend(_condition_candidates(data_by_key[(dataset, condition)]))
        selected_cases = choose_dataset_cases(numerical_rows)
        for case in selected_cases:
            if case["status"] != "selected":
                continue
            key = dataset, str(case["condition"])
            artifact = artifact_by_key[key]
            sample = artifact["samples"][str(case["sample_id"])]
            assembly = assembly_provenance[key]
            case["provenance"] = {
                "assembly_manifest_sha256": assembly["manifest_sha256"],
                "assembly_records_sha256": assembly["records_sha256"],
                "artifact_id": artifact["artifact_id"],
                "artifact_manifest_path": artifact["manifest_path"],
                "artifact_manifest_sha256": artifact["manifest_sha256"],
                "sample_payload_path": str(sample["path"]),
                "sample_payload_sha256": str(sample["sha256"]),
            }
        datasets.append(
            {
                "dataset": dataset,
                "target_conditions": list(TARGET_CONDITIONS),
                "cases": selected_cases,
            }
        )

    base = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "scope": {
            "interpretation": (
                "mechanically selected diagnostic examples; they are not a random "
                "sample and must not be described as representative"
            ),
            "source_rgb_policy": (
                "source RGB images are not used because the immutable frozen "
                "artifacts bind only foreground-probability and truth arrays"
            ),
            "selection_completed_without_payload_reads": True,
        },
        "selection_rules": SELECTION_RULES,
        "provenance": {
            **campaign_provenance,
            "selection_source_sha256": _source_sha256(),
            "validated_frozen_artifact_manifests": [
                {
                    "dataset": key[0],
                    "condition": key[1],
                    "artifact_id": artifact_by_key[key]["artifact_id"],
                    "manifest_path": artifact_by_key[key]["manifest_path"],
                    "manifest_sha256": artifact_by_key[key]["manifest_sha256"],
                }
                for key in EXPECTED_CONDITIONS
            ],
        },
        "condition_counts": {
            "validated_conditions": 16,
            "eligible_target_conditions": 10,
            "datasets": 5,
        },
        "datasets": datasets,
    }
    selection_id = hashlib.sha256(_canonical_json(base)).hexdigest()[:16]
    report = {**base, "selection_id": selection_id}
    _canonical_json(report)
    return report


def validate_selection_id(report: Mapping[str, Any]) -> str:
    selection_id = report.get("selection_id")
    if not isinstance(selection_id, str) or len(selection_id) != 16:
        raise ValueError("selection_id must be a 16-character content identifier")
    base = dict(report)
    del base["selection_id"]
    expected = hashlib.sha256(_canonical_json(base)).hexdigest()[:16]
    if selection_id != expected:
        raise ValueError("selection_id does not match the selection content")
    return selection_id


def write_selection(
    report: Mapping[str, Any], output_root: str | os.PathLike[str]
) -> Path:
    selection_id = validate_selection_id(report)
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / selection_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite selection package: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{selection_id}.tmp-", dir=root))
    try:
        payload = (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        output = temporary / "selection.json"
        with output.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination / "selection.json"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_selection(args.inputs, campaign_lock=args.campaign_lock)
    output = write_selection(report, args.output_root)
    print(_portable_path(output))


if __name__ == "__main__":
    main()
