"""Tests for locked M128 analysis and manuscript rendering."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze.main import EXPECTED_CONDITIONS, ConditionData
from scripts.analyze.m128 import (
    ARTIFACT_TYPE,
    COMPARISONS,
    SCHEMA_VERSION,
    M128ConditionData,
    _aggregate_target_ranges,
    analyze_condition,
    load_m128_condition,
    write_report,
)
from scripts.render.m128 import (
    DEFAULT_OUTPUT_DIR,
    OUTPUT_NAME,
    _aurc_format,
    load_analysis,
    parse_args,
    render_analysis,
    render_threshold_figure,
    threshold_aurc_series,
    validate_analysis,
    write_output,
)
from selectseg.studies.m128 import (
    AUXILIARY_ARTIFACT_TYPE,
    M128_SCORE_FIELDS,
)


def _sample_hash(sample_ids):
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def _write_auxiliary(tmp_path: Path, *, corrupt_score=False):
    directory = tmp_path / "m128" / "toy-run"
    directory.mkdir(parents=True)
    sample_ids = ["sample-a", "sample-b"]
    rows = []
    for index, sample_id in enumerate(sample_ids):
        score = 0.25 if corrupt_score and index == 1 else -0.2 - 0.1 * index
        rows.append(
            {
                "schema_version": 1,
                "run_id": "toy-run",
                "sample_id": sample_id,
                "image_id": sample_id,
                "image_index": index,
                "class_index": 1,
                "class_name": "foreground",
                "height": 10,
                "width": 12,
                "confidence_dice_m128_aux": score,
                "confidence_nhd_m128_aux": -0.3 - 0.1 * index,
                "confidence_nhd95_m128_aux": -0.25 - 0.1 * index,
            }
        )
    records = directory / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 1,
        "artifact_type": AUXILIARY_ARTIFACT_TYPE,
        "run_id": "toy-run",
        "auxiliary_id": "toy-run",
        "dataset": "toy",
        "condition": "clipseg-target",
        "model": "clipseg",
        "split": "test",
        "num_images": 2,
        "num_rows": 2,
        "sample_id_sha256": _sample_hash(sample_ids),
        "source_sha256": "a" * 64,
        "score_fields": list(M128_SCORE_FIELDS),
        "diagnostic_fields": [],
        "quadrature": {
            "128": {
                "rule": "midpoint",
                "nodes": [(index + 0.5) / 128 for index in range(128)],
                "weights": [1 / 128] * 128,
            }
        },
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": 0.5,
        },
        "provenance": {},
        "canonical_schema_v2_compatible": False,
        "jsonl_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return records


def _condition_pair(dataset: str, condition: str, index: int):
    canonical_rows = []
    auxiliary_rows = []
    for image_index in range(6):
        sample_id = f"{dataset}-{condition}-{image_index}"
        common = {
            "sample_id": sample_id,
            "image_id": sample_id,
            "image_index": image_index,
            "class_index": 1,
            "class_name": "foreground",
            "height": 32,
            "width": 40,
        }
        exact = -0.12 - 0.10 * image_index - 0.0001 * index
        nhd128 = -0.18 - 0.08 * image_index - 0.0001 * index
        nhd95128 = -0.15 - 0.07 * image_index - 0.0001 * index
        canonical_rows.append(
            {
                **common,
                "confidence_dice_exact": exact,
                "confidence_nhd_m32": nhd128 + (image_index - 2.5) * 0.0002,
                "confidence_nhd95_m32": nhd95128 + (image_index - 2.5) * 0.0003,
                "risk_dice": 0.15 + 0.10 * image_index,
                "risk_nhd": 0.20 + 0.08 * image_index,
                "risk_nhd95": 0.16 + 0.07 * image_index,
            }
        )
        auxiliary_rows.append(
            {
                "schema_version": 1,
                "run_id": "aux-run",
                **common,
                "confidence_dice_m128_aux": exact + (image_index - 2.5) * 0.0001,
                "confidence_nhd_m128_aux": nhd128,
                "confidence_nhd95_m128_aux": nhd95128,
            }
        )
    model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
    canonical = ConditionData(
        Path("canonical.jsonl"),
        Path("canonical.manifest.json"),
        {"dataset": dataset, "condition": condition, "model": model},
        tuple(canonical_rows),
    )
    auxiliary = M128ConditionData(
        Path("m128.jsonl"),
        Path("m128.manifest.json"),
        {"dataset": dataset, "condition": condition, "model": model},
        tuple(auxiliary_rows),
    )
    return canonical, auxiliary


def _report():
    conditions = []
    for index, (dataset, condition) in enumerate(EXPECTED_CONDITIONS):
        canonical, auxiliary = _condition_pair(dataset, condition, index)
        conditions.append(analyze_condition(canonical, auxiliary))
    targets = [
        f"{row['dataset']}/{row['condition']}"
        for row in conditions
        if row["is_target_condition"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_id": "0123456789abcdef",
        "scope": {
            "m128_status": "M128 is a numerical reference, not an exact integral"
        },
        "specification": {},
        "condition_sets": {
            "all_conditions": [
                f"{dataset}/{condition}" for dataset, condition in EXPECTED_CONDITIONS
            ],
            "target_condition_definition": "target models",
            "target_conditions": targets,
            "num_conditions": 16,
            "num_target_conditions": 10,
        },
        "provenance": {},
        "target_aggregate_ranges": _aggregate_target_ranges(conditions),
        "conditions": conditions,
    }


def test_m128_loader_validates_hash_schema_midpoints_and_score_range(tmp_path):
    records = _write_auxiliary(tmp_path)
    loaded = load_m128_condition(records)
    assert loaded.dataset == "toy"
    assert len(loaded.rows) == 2

    corrupt = _write_auxiliary(tmp_path / "bad", corrupt_score=True)
    with pytest.raises(ValueError, match=r"must be finite and lie in \[-1, 0\]"):
        load_m128_condition(corrupt)

    manifest_path = records.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["quadrature"]["128"]["nodes"][0] = 0.0
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="locked midpoint rule"):
        load_m128_condition(records)


def test_condition_join_and_metrics_use_matched_risks():
    canonical, auxiliary = _condition_pair("pet", "clipseg-target", 0)
    result = analyze_condition(canonical, auxiliary)
    assert result["is_target_condition"] is True
    assert set(result["comparisons"]) == {spec.name for spec in COMPARISONS}
    nhd = result["comparisons"]["nhd_m32_vs_m128"]
    assert nhd["per_image_absolute_score_error"]["max"] == pytest.approx(0.0005)
    assert nhd["rank_agreement"]["spearman_rho"]["value"] == pytest.approx(1.0)
    assert nhd["matched_risk_aurc"]["absolute_gap"] == pytest.approx(0.0)
    dice = result["comparisons"]["dice_m128_vs_exact"]
    assert "exact level-set Dice" in dice["reference_interpretation"]

    broken = copy.deepcopy(auxiliary.rows)
    broken[0]["height"] = 31
    bad_auxiliary = M128ConditionData(
        auxiliary.jsonl_path,
        auxiliary.manifest_path,
        auxiliary.manifest,
        tuple(broken),
    )
    with pytest.raises(ValueError, match="join identity mismatch"):
        analyze_condition(canonical, bad_auxiliary)


def test_renderer_is_deterministic_complete_and_explicitly_nonexact(tmp_path):
    report = _report()
    by_key = validate_analysis(report)
    assert len(by_key) == 16
    tex = render_analysis(report, source_hash="b" * 64)
    assert tex.count(r"\begin{table*}[t]") == 1
    assert r"\label{tab:m128-numerical-reference}" in tex
    assert "M128 is a numerical reference" in tex
    assert "not an exact integral" in tex
    assert "Dice: M128 vs Exact" in tex
    assert "HD95: M32 vs M128" in tex
    assert tex.index("Oxford Pet") < tex.index("Kvasir-SEG") < tex.index("FIVES")
    reversed_report = copy.deepcopy(report)
    reversed_report["conditions"].reverse()
    assert render_analysis(reversed_report, source_hash="b" * 64) == tex

    output = write_output(tex, tmp_path / "rendered")
    assert output.name == OUTPUT_NAME
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_output(tex, tmp_path / "rendered")

    primary = []
    for index, row in enumerate(report["conditions"]):
        comparisons = row["comparisons"]
        primary.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "risks": {
                    "risk_dice": {
                        "methods": {
                            f"confidence_dice_m{m}": {
                                "aurc": 0.2 + index / 1000 + 1 / m
                            }
                            for m in (2, 8, 32)
                        }
                    },
                    "risk_nhd": {
                        "methods": {
                            "confidence_nhd_m2": {"aurc": 0.3 + index / 1000},
                            "confidence_nhd_m8": {"aurc": 0.25 + index / 1000},
                            "confidence_nhd_m32": {
                                "aurc": comparisons["nhd_m32_vs_m128"][
                                    "matched_risk_aurc"
                                ]["candidate"]
                            },
                        }
                    },
                    "risk_nhd95": {
                        "methods": {
                            "confidence_nhd95_m2": {"aurc": 0.28 + index / 1000},
                            "confidence_nhd95_m8": {"aurc": 0.24 + index / 1000},
                            "confidence_nhd95_m32": {
                                "aurc": comparisons["nhd95_m32_vs_m128"][
                                    "matched_risk_aurc"
                                ]["candidate"]
                            },
                        }
                    },
                },
            }
        )
    series = threshold_aurc_series(primary, by_key)
    assert set(series) == {"Dice", "HD", "HD95"}
    assert all(len(values["midpoint"]) == 4 for values in series.values())
    figure = render_threshold_figure(primary, by_key, tmp_path / "thresholds.pdf")
    assert figure.stat().st_size > 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_threshold_figure(primary, by_key, figure)


def test_renderer_default_uses_versioned_times_100_destination():
    args = parse_args(["--analysis", "analysis.json"])
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_dir.endswith("/rendered_v2")
    assert _aurc_format(0.0123) == "1.230"


def test_strict_json_io_hash_and_no_overwrite(tmp_path):
    report = _report()
    report_path = write_report(report, tmp_path / "analysis.json")
    loaded, source_hash = load_analysis(report_path)
    assert loaded == report
    assert source_hash == hashlib.sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(report, report_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_analysis(duplicate)

    inconsistent = copy.deepcopy(report)
    inconsistent["target_aggregate_ranges"]["nhd_m32_vs_m128"][
        "matched_risk_aurc_absolute_gap"
    ]["max"] = 0.9
    with pytest.raises(ValueError, match="aggregate ranges"):
        validate_analysis(inconsistent)
