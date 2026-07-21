"""Synthetic tests for the strict auxiliary binary analysis."""

import hashlib
import json
from pathlib import Path

import pytest

from selectseg.binary_framework import tie_aware_expected_aurc

from scripts.analyze_binary_auxiliary import (
    JSON_NAME,
    LATEX_NAME,
    M128_SCORE_FIELDS,
    PRIMARY_SCORE_FIELDS,
    _latex_aurc,
    analyze_auxiliary,
    load_root,
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
    base_dice = 0.08 + 0.12 * index
    base_hd95 = 0.04 + 0.08 * ((2 * index) % 5)
    gamma_shift = abs(gamma - 0.5)
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
        "prediction_foreground_fraction": 0.2 + 0.01 * index + gamma_shift,
        "risk_dice": min(1.0, base_dice + gamma_shift * (0.1 + 0.01 * index)),
        "risk_nhd95": min(1.0, base_hd95 + gamma_shift * (0.05 + 0.01 * index)),
        "risk_hd95_pixels": 20
        * min(1.0, base_hd95 + gamma_shift * (0.05 + 0.01 * index)),
        "confidence_sdc": 0.9 - base_dice - gamma_shift * (0.1 + 0.01 * index),
        "confidence_mean_max_probability": 0.95 - 0.03 * index,
        "confidence_negative_entropy": -0.1 - 0.04 * index,
    }
    for count in m_values:
        row[f"confidence_dice_m{count}"] = -(
            base_dice + gamma_shift + 1 / (100 * count)
        )
        row[f"confidence_nhd95_m{count}"] = -(
            base_hd95 + gamma_shift + 1 / (200 * count)
        )
    return row


def _write_matrix(
    root,
    *,
    gamma,
    m_values,
    conditions=CONDITIONS,
    mutate_rows=None,
):
    root = Path(root)
    for dataset, condition in conditions:
        run_id = hashlib.sha256(
            f"{root.name}|{dataset}|{condition}|{gamma}|{m_values}".encode()
        ).hexdigest()[:12]
        rows = [
            _row(index, gamma=gamma, run_id=run_id, m_values=m_values)
            for index in range(6)
        ]
        if mutate_rows is not None:
            rows = mutate_rows(dataset, condition, rows)
        run_dir = root / dataset / condition / run_id
        run_dir.mkdir(parents=True)
        records = run_dir / "records.jsonl"
        records.write_text("".join(json.dumps(row) + "\n" for row in rows))
        score_fields = [
            "confidence_sdc",
            "confidence_mean_max_probability",
            "confidence_negative_entropy",
        ]
        for count in m_values:
            score_fields.extend(
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
            "cohort": "synthetic native binary cohort",
            "checkpoint": None,
            "base_model": {"name": condition.split("-")[0], "source": "synthetic"},
            "source_sha256": "a" * 64,
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
            "score_fields": score_fields,
            "quadrature": _quadrature(m_values),
            "sample_id_sha256": hashlib.sha256(
                "\n".join(row["sample_id"] for row in rows).encode()
            ).hexdigest(),
            "jsonl_sha256": _sha256(records),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def _experiment_roots(tmp_path, *, gamma_mutation=None, m128_mutation=None):
    primary = _write_matrix(tmp_path / "primary", gamma=0.5, m_values=(2, 8, 32))
    gamma03 = _write_matrix(
        tmp_path / "gamma03",
        gamma=0.3,
        m_values=(2, 8, 32),
        mutate_rows=gamma_mutation,
    )
    gamma07 = _write_matrix(tmp_path / "gamma07", gamma=0.7, m_values=(2, 8, 32))
    m128 = _write_matrix(
        tmp_path / "m128",
        gamma=0.5,
        m_values=(128,),
        mutate_rows=m128_mutation,
    )
    return primary, gamma03, gamma07, m128


def test_cli_writes_json_and_method_by_dataset_tex(tmp_path):
    primary, gamma03, gamma07, m128 = _experiment_roots(tmp_path)
    output = tmp_path / "analysis"
    main(
        [
            "--primary-root",
            str(primary),
            "--gamma-root",
            f"0.3={gamma03}",
            "--gamma-root",
            f"0.7={gamma07}",
            "--m128-root",
            str(m128),
            "--output-dir",
            str(output),
        ]
    )

    summary = json.loads((output / JSON_NAME).read_text())
    assert summary["analysis"]["gamma_values"] == [0.3, 0.5, 0.7]
    assert [
        (condition["dataset"], condition["condition"])
        for condition in summary["conditions"]
    ] == list(CONDITIONS)
    assert summary["analysis"]["m128_required_common_fields"] == sorted(
        {
            "risk_dice",
            "risk_nhd95",
            "risk_hd95_pixels",
            "confidence_sdc",
            "confidence_mean_max_probability",
            "confidence_negative_entropy",
        }
    )

    first = summary["conditions"][0]
    source_rows = []
    gamma_records = Path(first["sources"]["gamma"]["0.3"]["records"])
    with gamma_records.open() as handle:
        source_rows.extend(json.loads(line) for line in handle)
    expected = tie_aware_expected_aurc(
        [row["confidence_sdc"] for row in source_rows],
        [row["risk_dice"] for row in source_rows],
    )
    observed = first["threshold_robustness"]["0.3"]["risk_dice"]["methods"]
    assert observed["confidence_sdc"]["aurc"] == pytest.approx(expected)
    assert "aurc_difference_m128_minus_m32" in first["m128_ablation"]["risk_dice"]
    assert _latex_aurc(0.0123) == "1.2300"

    latex = (output / LATEX_NAME).read_text()
    assert r"\label{tab:threshold-robustness}" in latex
    assert r"\label{tab:m128-ablation}" in latex
    assert "Oxford Pet" in latex and "Kvasir-SEG" in latex
    assert "SDC &" in latex
    assert "Dice-M128 &" in latex
    assert r"$100\times\mathrm{AURC}$" in latex
    assert (
        r"\multicolumn{3}{l}{\textit{Dice risk; displayed as "
        r"$100\times\mathrm{AURC}$}} \\" in latex
    )
    assert "display-only transformation; the JSON retains raw AURC" in latex
    assert _latex_aurc(expected) in latex
    assert latex.index("SDC &") > latex.index("Confidence method &")


def test_gamma_join_rejects_changed_probability_only_baseline(tmp_path):
    def mutate(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[0]:
            rows[2]["confidence_mean_max_probability"] += 1e-12
        return rows

    primary, gamma03, gamma07, m128 = _experiment_roots(tmp_path, gamma_mutation=mutate)
    with pytest.raises(
        ValueError,
        match=r"gamma=0\.3 invariant join.*confidence_mean_max_probability",
    ):
        analyze_auxiliary(primary, {0.3: gamma03, 0.7: gamma07}, m128)


def test_m128_join_rejects_changed_common_risk(tmp_path):
    def mutate(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[1]:
            rows[4]["risk_nhd95"] += 1e-12
        return rows

    primary, gamma03, gamma07, m128 = _experiment_roots(tmp_path, m128_mutation=mutate)
    with pytest.raises(
        ValueError,
        match=r"M=128 common-field join.*risk_nhd95",
    ):
        analyze_auxiliary(primary, {0.3: gamma03, 0.7: gamma07}, m128)


def test_join_rejects_missing_sample_and_condition(tmp_path):
    def drop_sample(dataset, condition, rows):
        if (dataset, condition) == CONDITIONS[0]:
            return rows[:-1]
        return rows

    primary, gamma03, gamma07, m128 = _experiment_roots(
        tmp_path, gamma_mutation=drop_sample
    )
    with pytest.raises(ValueError, match="sample_id join mismatch"):
        analyze_auxiliary(primary, {0.3: gamma03, 0.7: gamma07}, m128)

    incomplete = _write_matrix(
        tmp_path / "m128-incomplete",
        gamma=0.5,
        m_values=(128,),
        conditions=CONDITIONS[:1],
    )
    with pytest.raises(ValueError, match="condition-set mismatch"):
        analyze_auxiliary(primary, {0.3: gamma03, 0.7: gamma07}, incomplete)


def test_loader_rejects_wrong_schema_hash_and_quadrature(tmp_path):
    root = _write_matrix(tmp_path / "root", gamma=0.5, m_values=(2, 8, 32))
    assert len(
        load_root(
            root,
            expected_score_fields=PRIMARY_SCORE_FIELDS,
            expected_m_values=(2, 8, 32),
            expected_gamma=0.5,
        )
    ) == len(CONDITIONS)

    first_manifest = next(root.rglob("manifest.json"))
    manifest = json.loads(first_manifest.read_text())
    manifest["quadrature"]["32"]["weights"][0] = 0.5
    first_manifest.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="unexpected nodes or weights"):
        load_root(
            root,
            expected_score_fields=PRIMARY_SCORE_FIELDS,
            expected_m_values=(2, 8, 32),
            expected_gamma=0.5,
        )

    m128_root = _write_matrix(tmp_path / "m128-only", gamma=0.5, m_values=(128,))
    first_records = next(m128_root.rglob("records.jsonl"))
    first_records.write_text(first_records.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_root(
            m128_root,
            expected_score_fields=M128_SCORE_FIELDS,
            expected_m_values=(128,),
            expected_gamma=0.5,
        )
