"""Tests for the isolated matched-risk reliability workflow."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_binary import EXPECTED_CONDITIONS, ConditionData
from scripts.analyze_matched_risk_reliability import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GROUP_BINS,
    MATCHED_PAIRS,
    TARGET_CONDITIONS,
    build_report,
    equal_count_bootstrap_reliability,
    load_inputs,
    write_report,
)
from scripts.render_matched_risk_reliability import (
    DATASET_ORDER,
    load_analysis,
    render_figures,
    validate_analysis,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _conditions(num_images: int = 23) -> list[ConditionData]:
    result = []
    for condition_index, (dataset, condition) in enumerate(EXPECTED_CONDITIONS):
        rows = []
        for image_index in range(num_images):
            base = (image_index + 1) / (num_images + 2)
            shift = condition_index / 1_000
            dice_predicted = min(0.98, 0.04 + 0.75 * base + shift)
            nhd_predicted = min(0.98, 0.02 + 0.45 * base + shift)
            nhd95_predicted = min(0.98, 0.01 + 0.35 * base + shift)
            rows.append(
                {
                    "sample_id": f"{dataset}-{condition}-{image_index:03d}",
                    "confidence_dice_exact": -dice_predicted,
                    "confidence_nhd_m32": -nhd_predicted,
                    "confidence_nhd95_m32": -nhd95_predicted,
                    "risk_dice": min(1.0, 0.02 + 0.90 * dice_predicted),
                    "risk_nhd": min(1.0, 0.01 + 0.92 * nhd_predicted),
                    "risk_nhd95": min(1.0, 0.01 + 0.88 * nhd95_predicted),
                }
            )
        model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
        result.append(
            ConditionData(
                jsonl_path=Path(f"/{dataset}/{condition}/records.jsonl"),
                manifest_path=Path(f"/{dataset}/{condition}/manifest.json"),
                manifest={"dataset": dataset, "condition": condition, "model": model},
                rows=tuple(rows),
            )
        )
    return result


def _provenance(num_images: int = 23):
    return {
        "binding": "campaign-lock",
        "campaign_id": "unit-reliability-campaign",
        "campaign_lock": {"logical_name": "campaign.lock.json", "sha256": "a" * 64},
        "config_sha256": "b" * 64,
        "analysis_source_sha256": "c" * 64,
        "inputs": [
            {
                "logical_id": f"{dataset}/{condition}/unit",
                "dataset": dataset,
                "condition": condition,
                "assembly_run_id": "unit",
                "assembly_source_sha256": _digest(f"source-{dataset}-{condition}"),
                "artifact_id": _digest(f"artifact-{dataset}-{condition}")[:16],
                "manifest_sha256": _digest(f"manifest-{dataset}-{condition}"),
                "records_sha256": _digest(f"records-{dataset}-{condition}"),
                "sample_id_sha256": _digest(f"samples-{dataset}-{condition}"),
                "num_samples": num_images,
            }
            for dataset, condition in EXPECTED_CONDITIONS
        ],
    }


def test_equal_count_bootstrap_is_fixed_ordered_and_pointwise():
    ids = [f"id-{index:03d}" for index in reversed(range(20))]
    observed = np.asarray([int(item[-3:]) / 19 for item in ids])
    predicted = np.full(20, 0.4)
    first = equal_count_bootstrap_reliability(
        predicted,
        observed,
        ids,
        rng=np.random.default_rng(BOOTSTRAP_SEED),
    )
    second = equal_count_bootstrap_reliability(
        predicted[::-1],
        observed[::-1],
        ids[::-1],
        rng=np.random.default_rng(BOOTSTRAP_SEED),
    )
    assert first == second
    assert len(first) == GROUP_BINS
    assert all(item["num_images"] == 2 for item in first)
    assert first[0]["mean_observed_loss"] == pytest.approx(0.5 / 19)
    assert all(
        item["pointwise_ci_lower"]
        <= item["mean_observed_loss"]
        <= item["pointwise_ci_upper"]
        for item in first
    )


def test_complete_report_is_order_invariant_fixed_and_write_once(tmp_path):
    conditions = _conditions()
    provenance = _provenance()
    report = build_report(conditions, canonical_provenance=provenance)
    assert report == build_report(
        list(reversed(conditions)), canonical_provenance=provenance
    )
    assert report["protocol"]["num_bins"] == 10
    assert report["protocol"]["bootstrap_seed"] == 20_260_720
    assert report["protocol"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert len(report["conditions"]) == len(TARGET_CONDITIONS) == 10
    for condition in report["conditions"]:
        assert len(condition["panels"]) == len(MATCHED_PAIRS) == 3
        for panel in condition["panels"]:
            assert len(panel["bins"]) == 10
            assert sum(item["num_images"] for item in panel["bins"]) == 23
            assert max(item["num_images"] for item in panel["bins"]) == 3
            assert min(item["num_images"] for item in panel["bins"]) == 2

    output = write_report(report, tmp_path / "analysis.json")
    loaded, source_hash = load_analysis(output)
    assert loaded == report
    assert len(source_hash) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(report, output)


def test_renderer_is_strict_emits_five_pdfs_and_refuses_overwrite(tmp_path):
    report = build_report(_conditions(), canonical_provenance=_provenance())
    by_key = validate_analysis(report)
    assert set(by_key) == set(TARGET_CONDITIONS)
    source_hash = "d" * 64
    paths = render_figures(
        report,
        source_hash=source_hash,
        output_dir=tmp_path / "figures",
    )
    assert [path.stem.removeprefix("matched_risk_reliability_") for path in paths] == list(
        DATASET_ORDER
    )
    assert all(path.read_bytes().startswith(b"%PDF") for path in paths)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_figures(
            report,
            source_hash=source_hash,
            output_dir=tmp_path / "figures",
        )

    tampered = copy.deepcopy(report)
    tampered["conditions"][0]["panels"][0]["bins"].pop()
    with pytest.raises(ValueError, match="ten bins"):
        validate_analysis(tampered)


def test_cli_input_contract_rejects_noncanonical_count_before_file_access():
    with pytest.raises(ValueError, match="exactly 16 explicit"):
        load_inputs([])
