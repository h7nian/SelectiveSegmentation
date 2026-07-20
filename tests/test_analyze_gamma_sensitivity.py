"""Tests for strict gamma-sensitivity analysis and deterministic TeX rendering."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_binary import CONTRASTS, EXPECTED_CONDITIONS, ConditionData
from scripts.analyze_gamma_sensitivity import (
    ARTIFACT_TYPE,
    AUXILIARY_GAMMAS,
    GAMMA_INVARIANT_FIELDS,
    SCHEMA_VERSION,
    GammaConditionData,
    aggregate_targets,
    analyze_condition,
    fractional_acceptance_weights,
    load_gamma_condition,
    load_gamma_inputs,
    validate_primary_analysis,
    write_report,
)
from scripts.render_gamma_sensitivity import (
    DEFAULT_OUTPUT_ROOT,
    OUTPUT_NAME,
    _aurc_range,
    load_analysis,
    parse_args,
    render_analysis,
    validate_analysis,
    write_output,
)
from selectseg.score_binary_common import (
    AUXILIARY_FIELDS,
    COMMON_SCORE_FIELDS,
    RISK_FIELDS,
)
from selectseg.score_binary_gamma_sensitivity import (
    AUXILIARY_ARTIFACT_TYPE,
    M32_SCORE_FIELDS,
    OUTPUT_ROW_FIELDS,
)


def _sample_hash(sample_ids):
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def _write_auxiliary(tmp_path: Path, *, corrupt_score: bool = False) -> Path:
    directory = tmp_path / "gamma-0p3" / "toy-run"
    directory.mkdir(parents=True)
    sample_ids = ["sample-a", "sample-b"]
    rows = []
    for index, sample_id in enumerate(sample_ids):
        row = {
            "schema_version": 2,
            "run_id": "toy-run",
            "sample_id": sample_id,
            "image_id": sample_id,
            "image_index": index,
            "class_index": 1,
            "class_name": "foreground",
            "height": 12,
            "width": 14,
            "image_diagonal": float(np.hypot(12, 14)),
            "truth_foreground_fraction": 0.2 + 0.1 * index,
            "prediction_foreground_fraction": 0.3 + 0.1 * index,
            "risk_dice": 0.2 + 0.1 * index,
            "risk_nhd": 0.3 + 0.1 * index,
            "risk_nhd95": 0.25 + 0.1 * index,
            "risk_hd_pixels": 4.0 + index,
            "risk_hd95_pixels": 3.0 + index,
            "confidence_sdc": 0.8 - 0.1 * index,
            "confidence_mean_max_probability": 0.9 - 0.1 * index,
            "confidence_negative_entropy": -0.1 - 0.1 * index,
            "confidence_dice_exact": -0.2 - 0.1 * index,
            "confidence_qfr_entropy": -0.15 - 0.1 * index,
            "confidence_plm10_entropy": -0.16 - 0.1 * index,
            "confidence_mmmc_entropy": -0.17 - 0.1 * index,
            "confidence_foreground_entropy": -0.18 - 0.1 * index,
            "confidence_dice_m32": -0.21 - 0.1 * index,
            "confidence_nhd_m32": -0.31 - 0.1 * index,
            "confidence_nhd95_m32": -0.26 - 0.1 * index,
        }
        if corrupt_score and index == 1:
            row["confidence_nhd_m32"] = 0.1
        assert set(row) == set(OUTPUT_ROW_FIELDS)
        rows.append(row)
    records = directory / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 1,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "run_id": "toy-run",
        "dataset": "toy",
        "condition": "clipseg-target",
        "model": "clipseg",
        "split": "test",
        "num_images": 2,
        "num_rows": 2,
        "checkpoint": None,
        "sample_id_sha256": _sample_hash(sample_ids),
        "source_sha256": "a" * 64,
        "score_fields": [*COMMON_SCORE_FIELDS, *M32_SCORE_FIELDS],
        "risk_fields": list(RISK_FIELDS),
        "auxiliary_fields": list(AUXILIARY_FIELDS),
        "quadrature": {
            "32": {
                "rule": "midpoint",
                "nodes": [(index + 0.5) / 32 for index in range(32)],
                "weights": [1 / 32] * 32,
            }
        },
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": 0.3,
        },
        "provenance": {},
        "canonical_schema_v2_compatible": False,
        "jsonl_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return records


def _rows(dataset: str, condition: str, *, gamma: float, condition_index: int):
    rows = []
    for image_index in range(8):
        sample_id = f"{dataset}-{condition}-{image_index}"
        base = (image_index + 1) / 10 + condition_index / 10_000
        # Keep invariant values bit-identical for every gamma.
        row = {
            "sample_id": sample_id,
            "image_id": sample_id,
            "image_index": image_index,
            "class_index": 1,
            "class_name": "foreground",
            "height": 32,
            "width": 40,
            "image_diagonal": float(np.hypot(32, 40)),
            "truth_foreground_fraction": 0.1 + image_index / 100,
            "confidence_mean_max_probability": 0.9 - base / 10,
            "confidence_negative_entropy": -base / 10,
            "confidence_qfr_entropy": -base / 11,
            "confidence_plm10_entropy": -base / 12,
            "confidence_mmmc_entropy": -base / 13,
            "confidence_foreground_entropy": -base / 14,
        }
        shift = gamma - 0.5
        # Reverse one score only at gamma=.7 in some conditions, exercising the
        # strict direction/reversal bookkeeping without relying on random data.
        orientation = -1 if gamma == 0.7 and condition_index % 3 == 0 else 1
        row.update(
            {
                "schema_version": 2,
                "run_id": f"run-{gamma}",
                "prediction_foreground_fraction": max(
                    0.0, 0.35 - 0.35 * shift + image_index / 1000
                ),
                "risk_dice": min(1.0, 0.12 + base / 2 + abs(shift) / 10),
                "risk_nhd": min(1.0, 0.18 + base / 2.5 + abs(shift) / 12),
                "risk_nhd95": min(1.0, 0.15 + base / 3 + abs(shift) / 14),
                "risk_hd_pixels": 3.0 + image_index,
                "risk_hd95_pixels": 2.0 + image_index,
                "confidence_sdc": 0.85 - base / 5 + shift / 20,
                "confidence_dice_exact": -0.10 - base / 2 - shift / 20,
                "confidence_dice_m32": -0.10 - orientation * base / 2 - shift / 20,
                "confidence_nhd_m32": -0.16 - base / 2.6 + shift / 25,
                "confidence_nhd95_m32": -0.14 - base / 3.0 - shift / 30,
            }
        )
        rows.append(row)
    return tuple(rows)


def _condition_triplet(dataset: str, condition: str, condition_index: int):
    model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
    canonical_rows = _rows(
        dataset, condition, gamma=0.5, condition_index=condition_index
    )
    canonical = ConditionData(
        Path(f"{dataset}-{condition}.jsonl"),
        Path(f"{dataset}-{condition}.manifest.json"),
        {
            "dataset": dataset,
            "condition": condition,
            "model": model,
            "jsonl_sha256": "a" * 64,
        },
        canonical_rows,
    )
    auxiliary = {}
    for gamma in AUXILIARY_GAMMAS:
        auxiliary[gamma] = GammaConditionData(
            Path(f"{dataset}-{condition}-{gamma}.jsonl"),
            Path(f"{dataset}-{condition}-{gamma}.manifest.json"),
            {
                "dataset": dataset,
                "condition": condition,
                "model": model,
                "decision_rule": {"gamma": gamma},
            },
            _rows(dataset, condition, gamma=gamma, condition_index=condition_index),
        )
    return canonical, auxiliary


def _report():
    conditions = []
    for index, (dataset, condition) in enumerate(EXPECTED_CONDITIONS):
        canonical, auxiliary = _condition_triplet(dataset, condition, index)
        conditions.append(analyze_condition(canonical, auxiliary))
    targets = [
        f"{row['dataset']}/{row['condition']}"
        for row in conditions
        if row["is_target_condition"]
    ]
    auxiliary_inputs = [
        {"manifest_sha256": hashlib.sha256(f"manifest-{index}".encode()).hexdigest()}
        for index in range(32)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": "0123456789abcdef",
        "scope": {
            "status": "descriptive; neither threshold tuning nor a robustness guarantee"
        },
        "specification": {},
        "condition_sets": {
            "all_conditions": [
                f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
            ],
            "target_conditions": targets,
            "num_conditions": 16,
            "num_target_conditions": 10,
            "num_auxiliary_experiments": 32,
        },
        "provenance": {
            "analysis_source_sha256": "a" * 64,
            "auxiliary_lock": {"sha256": "b" * 64},
            "canonical_primary_analysis": {"sha256": "c" * 64},
            "auxiliary_inputs": auxiliary_inputs,
        },
        "target_headline": aggregate_targets(conditions),
        "conditions": conditions,
    }


def test_auxiliary_loader_enforces_schema_hash_midpoints_and_score_range(tmp_path):
    records = _write_auxiliary(tmp_path)
    loaded = load_gamma_condition(records)
    assert loaded.gamma == 0.3
    assert len(loaded.rows) == 2

    corrupt = _write_auxiliary(tmp_path / "corrupt", corrupt_score=True)
    with pytest.raises(ValueError, match=r"must be <= 0\.0"):
        load_gamma_condition(corrupt)

    manifest_path = records.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["quadrature"]["32"]["nodes"][0] = 0.0
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="locked M32 midpoint rule"):
        load_gamma_condition(records)


def test_incomplete_grid_has_clear_error_before_file_discovery():
    with pytest.raises(ValueError, match="incomplete.*exactly 32.*observed 0"):
        load_gamma_inputs([], binding={})


def test_condition_join_metrics_reversals_and_tie_aware_mass():
    canonical, auxiliary = _condition_triplet("pet", "clipseg-target", 0)
    result = analyze_condition(canonical, auxiliary)
    assert result["is_target_condition"] is True
    assert set(result["contrasts"]) == {spec.name for spec in CONTRASTS}
    assert set(result["action_quality_by_gamma"]) == {"0.3", "0.5", "0.7"}
    assert set(result["indexed_score_stability"]) == {
        "confidence_dice_m32",
        "confidence_nhd_m32",
        "confidence_nhd95_m32",
    }
    weights = fractional_acceptance_weights([1, 1, 0, 0], 0.25)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:2].tolist() == pytest.approx([0.5, 0.5])

    # Jobs on different CPU partitions can differ by one ULP in an otherwise
    # invariant floating-point reduction; this is not artifact contamination.
    roundoff_rows = copy.deepcopy(auxiliary[0.7].rows)
    value = roundoff_rows[0][GAMMA_INVARIANT_FIELDS[-1]]
    roundoff_rows[0][GAMMA_INVARIANT_FIELDS[-1]] = float(np.nextafter(value, np.inf))
    roundoff = dict(auxiliary)
    roundoff[0.7] = GammaConditionData(
        auxiliary[0.7].records_path,
        auxiliary[0.7].manifest_path,
        auxiliary[0.7].manifest,
        tuple(roundoff_rows),
    )
    analyze_condition(canonical, roundoff)

    broken_rows = copy.deepcopy(auxiliary[0.3].rows)
    broken_rows[0][GAMMA_INVARIANT_FIELDS[-1]] += 0.01
    broken = dict(auxiliary)
    broken[0.3] = GammaConditionData(
        auxiliary[0.3].records_path,
        auxiliary[0.3].manifest_path,
        auxiliary[0.3].manifest,
        tuple(broken_rows),
    )
    with pytest.raises(ValueError, match="gamma-invariant join mismatch"):
        analyze_condition(canonical, broken)


def test_primary_analysis_values_are_recomputed_and_tampering_is_rejected(tmp_path):
    canonical = []
    primary_conditions = []
    for index, (dataset, condition) in enumerate(EXPECTED_CONDITIONS):
        data, _ = _condition_triplet(dataset, condition, index)
        manifest_path = tmp_path / f"{index}.manifest.json"
        manifest_path.write_text(json.dumps({"index": index}) + "\n")
        data = ConditionData(
            data.jsonl_path,
            manifest_path,
            data.manifest,
            data.rows,
        )
        canonical.append(data)
        comparisons = {}
        risks = {risk: {"methods": {}} for risk in RISK_FIELDS}
        for spec in CONTRASTS:
            observed = np.asarray([row[spec.risk] for row in data.rows])
            left = np.asarray([row[spec.left] for row in data.rows])
            right = np.asarray([row[spec.right] for row in data.rows])
            from selectseg.binary_framework import tie_aware_expected_aurc

            left_aurc = tie_aware_expected_aurc(left, observed)
            right_aurc = tie_aware_expected_aurc(right, observed)
            comparisons[spec.name] = {
                "difference_left_minus_right": left_aurc - right_aurc
            }
            risks[spec.risk]["methods"][spec.left] = {"aurc": left_aurc}
            risks[spec.risk]["methods"][spec.right] = {"aurc": right_aurc}
        primary_conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "jsonl_sha256": data.manifest["jsonl_sha256"],
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "num_rows": len(data.rows),
                "comparisons": comparisons,
                "risks": risks,
            }
        )
    primary = {
        "schema_version": 2,
        "provenance": {
            "binding": "campaign-lock",
            "campaign_lock": {"sha256": "d" * 64},
        },
        "analysis": {
            "contrast_definitions": [
                {
                    "name": spec.name,
                    "left": spec.left,
                    "right": spec.right,
                    "risk": spec.risk,
                }
                for spec in CONTRASTS
            ]
        },
        "conditions": primary_conditions,
    }
    validate_primary_analysis(primary, canonical, campaign_lock_sha256="d" * 64)
    tampered = copy.deepcopy(primary)
    first = next(iter(tampered["conditions"][0]["comparisons"].values()))
    first["difference_left_minus_right"] += 0.01
    with pytest.raises(ValueError, match="primary contrast value differs"):
        validate_primary_analysis(tampered, canonical, campaign_lock_sha256="d" * 64)


def test_renderer_is_complete_deterministic_labeled_and_content_addressed(tmp_path):
    report = _report()
    by_key = validate_analysis(report)
    assert len(by_key) == 16
    tex = render_analysis(report, source_hash="e" * 64)
    assert tex.count(r"\begin{table*}[t]") == 2
    assert r"\label{tab:gamma-contrast-sensitivity}" in tex
    assert r"\label{tab:gamma-action-and-ranking-sensitivity}" in tex
    assert "not threshold tuning or a robustness guarantee" in tex
    assert "Predicted-empty rate" in tex
    assert "Spearman" in tex and "J_{0.25}" in tex
    assert r"Range of $100(\Delta_\gamma-\Delta_{.5})$" in tex
    assert tex.count(r"\resizebox{\linewidth}{!}{%") == 2
    assert tex.count(r"\textbf{Macro mean}") == 1
    assert "macro-mean row" in tex
    assert tex.index("Oxford Pet") < tex.index("Kvasir-SEG") < tex.index("FIVES")
    reversed_report = copy.deepcopy(report)
    reversed_report["conditions"].reverse()
    assert render_analysis(reversed_report, source_hash="e" * 64) == tex

    report_path = write_report(report, tmp_path / "json")
    assert report_path.parent.name == report["analysis_id"]
    loaded, source_hash = load_analysis(report_path)
    assert loaded == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(report, tmp_path / "json")

    destination = write_output(tex, tmp_path / "tex", source_hash=source_hash)
    assert destination.name == OUTPUT_NAME
    assert destination.parent.name == source_hash[:16]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_output(tex, tmp_path / "tex", source_hash=source_hash)


def test_renderer_default_and_aurc_change_use_versioned_times_100_display():
    args = parse_args(["--analysis", "analysis.json"])
    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert args.output_root.endswith("/rendered_v3")
    assert _aurc_range({"min": -0.0123, "max": 0.0456}) == (r"$[-1.230,\,4.560]$")


def test_renderer_rejects_tampered_headline_and_duplicate_json(tmp_path):
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["target_headline"]["num_target_conditions"] = 9
    with pytest.raises(ValueError, match="target headline differs"):
        validate_analysis(tampered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_analysis(duplicate)
