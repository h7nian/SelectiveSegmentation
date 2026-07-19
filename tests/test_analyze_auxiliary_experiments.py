"""Focused synthetic tests for the strict auxiliary-experiment analyzer."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_auxiliary_experiments import (
    EXPECTED_SOURCE_SHA256,
    JSON_NAME,
    LATEX_NAME,
    PRIMARY_M_VALUES,
    analyze,
    main,
)


CONDITIONS = (
    ("pet", "clipseg-general"),
    ("kvasir", "deeplabv3-target"),
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quadrature(m_values):
    return {
        str(count): {
            "rule": "midpoint",
            "nodes": [(index + 0.5) / count for index in range(count)],
            "weights": [1 / count] * count,
        }
        for count in m_values
    }


def _row(index, *, gamma, run_id, m_values):
    dice_risk = 0.08 + 0.11 * index + 0.05 * abs(gamma - 0.5)
    hd95_risk = 0.04 + 0.07 * ((2 * index) % 5) + 0.03 * abs(gamma - 0.5)
    row = {
        "schema_version": 1,
        "run_id": run_id,
        "sample_id": f"image-{index}",
        "image_id": f"image-{index}",
        "image_index": index,
        "class_index": 1,
        "class_name": "foreground",
        "height": 16 + index,
        "width": 20 + index,
        "image_diagonal": ((16 + index) ** 2 + (20 + index) ** 2) ** 0.5,
        "truth_foreground_fraction": 0.1 + 0.02 * index,
        "prediction_foreground_fraction": 0.2 + 0.01 * index + gamma / 100,
        "risk_dice": dice_risk,
        "risk_nhd95": hd95_risk,
        "risk_hd95_pixels": 20 * hd95_risk,
        "confidence_sdc": 0.9 - dice_risk,
        "confidence_mean_max_probability": 0.95 - 0.03 * index,
        "confidence_negative_entropy": -0.1 - 0.04 * index,
    }
    direction = -1 if index % 2 else 1
    for count in m_values:
        error = direction * (index + 1) / (10 * count)
        row[f"confidence_dice_m{count}"] = -dice_risk + error
        row[f"confidence_nhd95_m{count}"] = -hd95_risk - error / 2
    return row


def _write_run(
    root,
    *,
    dataset,
    condition,
    gamma,
    m_values,
    source_sha256=EXPECTED_SOURCE_SHA256,
    tag="valid",
    mutate=None,
):
    run_id = hashlib.sha256(
        f"{root}|{dataset}|{condition}|{gamma}|{m_values}|{tag}".encode()
    ).hexdigest()[:12]
    rows = [
        _row(index, gamma=gamma, run_id=run_id, m_values=m_values)
        for index in range(6)
    ]
    if mutate is not None:
        rows = mutate(dataset, condition, rows)
    run_dir = Path(root) / dataset / condition / run_id
    run_dir.mkdir(parents=True)
    records = run_dir / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scores = [
        "confidence_sdc",
        "confidence_mean_max_probability",
        "confidence_negative_entropy",
    ]
    for count in m_values:
        scores.extend(
            [f"confidence_dice_m{count}", f"confidence_nhd95_m{count}"]
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "condition": condition,
        "model": condition.split("-")[0],
        "dataset": dataset,
        "split": "test",
        "num_images": len(rows),
        "num_rows": len(rows),
        "checkpoint": {"sha256": "c" * 64},
        "base_model": {"name": condition.split("-")[0]},
        "source_sha256": source_sha256,
        "cohort": "synthetic native binary cohort",
        "preprocessing": {"resize": "synthetic"},
        "losses": {"dice": "synthetic", "hd95": "synthetic"},
        "void_policy": "total binary domain",
        "sdc_empty_convention": "zero",
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": gamma,
        },
        "risk_fields": ["risk_dice", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd95_pixels"],
        "score_fields": scores,
        "quadrature": _quadrature(m_values),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode()
        ).hexdigest(),
        "jsonl_sha256": _sha256(records),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")


def _write_matrix(
    root,
    *,
    gamma,
    m_values,
    source_sha256=EXPECTED_SOURCE_SHA256,
    tag="valid",
    mutate=None,
):
    for dataset, condition in CONDITIONS:
        _write_run(
            root,
            dataset=dataset,
            condition=condition,
            gamma=gamma,
            m_values=m_values,
            source_sha256=source_sha256,
            tag=tag,
            mutate=mutate,
        )
    return Path(root)


def _roots(tmp_path, *, gamma03_mutate=None, m128_mutate=None):
    primary = _write_matrix(
        tmp_path / "primary", gamma=0.5, m_values=PRIMARY_M_VALUES
    )
    gamma03 = _write_matrix(
        tmp_path / "gamma03",
        gamma=0.3,
        m_values=PRIMARY_M_VALUES,
        mutate=gamma03_mutate,
    )
    gamma07 = _write_matrix(
        tmp_path / "gamma07", gamma=0.7, m_values=PRIMARY_M_VALUES
    )
    m128 = _write_matrix(
        tmp_path / "m128", gamma=0.5, m_values=(128,), mutate=m128_mutate
    )
    return primary, gamma03, gamma07, m128


def test_cli_emits_stable_json_and_method_by_dataset_tables(tmp_path):
    primary, gamma03, gamma07, m128 = _roots(tmp_path)
    output = tmp_path / "analysis"
    arguments = [
        "--primary-root",
        str(primary),
        "--gamma03-root",
        str(gamma03),
        "--gamma07-root",
        str(gamma07),
        "--m128-root",
        str(m128),
        "--output-dir",
        str(output),
    ]
    main(arguments)
    first_json = (output / JSON_NAME).read_bytes()
    first_tex = (output / LATEX_NAME).read_bytes()
    main(arguments)
    assert (output / JSON_NAME).read_bytes() == first_json
    assert (output / LATEX_NAME).read_bytes() == first_tex

    summary = json.loads(first_json)
    assert [(item["dataset"], item["condition"]) for item in summary["conditions"]] == list(
        CONDITIONS
    )
    condition = summary["conditions"][0]
    primary_rows = {
        row["sample_id"]: row
        for row in map(
            json.loads,
            Path(condition["sources"]["gamma"]["0.5"])
            .with_name("records.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    reference_rows = {
        row["sample_id"]: row
        for row in map(
            json.loads,
            Path(condition["sources"]["m128"])
            .with_name("records.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    errors = [
        abs(
            primary_rows[sample_id]["confidence_dice_m2"]
            - reference_rows[sample_id]["confidence_dice_m128"]
        )
        for sample_id in sorted(primary_rows)
    ]
    observed = condition["convergence_to_m128"]["dice"]["approximations"]["2"]
    assert observed["mae"] == pytest.approx(np.mean(errors))
    assert observed["q95_absolute_error"] == pytest.approx(
        np.quantile(errors, 0.95)
    )
    assert set(summary["datasets"]) == {"pet", "kvasir"}

    latex = first_tex.decode()
    assert r"\label{tab:aux-threshold-robustness}" in latex
    assert r"\label{tab:aux-m128-convergence}" in latex
    assert "Metric & Oxford Pet (1) & Kvasir-SEG (1)" in latex
    assert "Dice-M32 AURC" in latex
    assert "nHD95-M8: Q95" in latex


def test_duplicate_sample_id_is_rejected(tmp_path):
    def duplicate(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[0]:
            rows[1]["sample_id"] = rows[0]["sample_id"]
        return rows

    primary = _write_matrix(
        tmp_path / "primary",
        gamma=0.5,
        m_values=PRIMARY_M_VALUES,
        mutate=duplicate,
    )
    gamma03 = _write_matrix(
        tmp_path / "gamma03", gamma=0.3, m_values=PRIMARY_M_VALUES
    )
    gamma07 = _write_matrix(
        tmp_path / "gamma07", gamma=0.7, m_values=PRIMARY_M_VALUES
    )
    m128 = _write_matrix(tmp_path / "m128", gamma=0.5, m_values=(128,))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        analyze(primary, gamma03, gamma07, m128)


def test_gamma_and_m128_exact_join_mismatches_are_rejected(tmp_path):
    def change_probability_baseline(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[0]:
            rows[2]["confidence_mean_max_probability"] += 1e-12
        return rows

    primary, gamma03, gamma07, m128 = _roots(
        tmp_path / "gamma-mismatch", gamma03_mutate=change_probability_baseline
    )
    with pytest.raises(
        ValueError, match="confidence_mean_max_probability.*differs"
    ):
        analyze(primary, gamma03, gamma07, m128)

    def change_risk(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[1]:
            rows[4]["risk_nhd95"] += 1e-12
        return rows

    primary, gamma03, gamma07, m128 = _roots(
        tmp_path / "m128-mismatch", m128_mutate=change_risk
    )
    with pytest.raises(ValueError, match="risk_nhd95.*differs"):
        analyze(primary, gamma03, gamma07, m128)


def test_stale_fingerprint_is_rejected_and_valid_rerun_is_selected(tmp_path):
    primary, gamma03, gamma07, m128 = _roots(tmp_path)
    stale_source = "b" * 64
    _write_matrix(
        m128,
        gamma=0.5,
        m_values=(128,),
        source_sha256=stale_source,
        tag="stale",
    )
    summary = analyze(primary, gamma03, gamma07, m128)
    rejected = summary["rejected_fingerprint_candidates"]["m128"]
    assert len(rejected) == len(CONDITIONS)
    assert {item["observed_source_sha256"] for item in rejected} == {stale_source}
    assert all(
        EXPECTED_SOURCE_SHA256
        in Path(item["sources"]["m128"]).read_text()
        for item in summary["conditions"]
    )

    invalid_only = _write_matrix(
        tmp_path / "invalid-only",
        gamma=0.5,
        m_values=(128,),
        source_sha256=stale_source,
    )
    with pytest.raises(ValueError, match="no runs.*match required source_sha256"):
        analyze(primary, gamma03, gamma07, invalid_only)
