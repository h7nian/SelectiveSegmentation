"""Binary schema-v2 risk--coverage plotting and tie semantics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_binary import METHODS
from scripts.plot_risk_coverage import (
    ASSEMBLY_ARTIFACT_TYPE,
    RISK_SPECS,
    condition_all_indexed_curves,
    condition_curves,
    load_assembled_conditions,
    main,
    render_conditions,
    tie_aware_risk_coverage_curve,
)
from selectseg.binary_framework import tie_aware_expected_aurc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(run_id="plot-run"):
    dice_risks = [0.10, 0.70, 0.25, 0.90, 0.45]
    nhd_risks = [0.15, 0.80, 0.35, 0.65, 0.40]
    nhd95_risks = [0.12, 0.60, 0.30, 0.55, 0.38]
    sdc_scores = [0.9, 0.6, 0.6, 0.1, 0.1]
    rows = []
    for index, (risk_dice, risk_nhd, risk_nhd95) in enumerate(
        zip(dice_risks, nhd_risks, nhd95_risks)
    ):
        row = {
            "schema_version": 2,
            "run_id": run_id,
            "sample_id": f"sample-{index}",
            "image_id": f"image-{index}",
            "risk_dice": risk_dice,
            "risk_nhd": risk_nhd,
            "risk_nhd95": risk_nhd95,
            "risk_hd_pixels": 100 * risk_nhd,
            "risk_hd95_pixels": 100 * risk_nhd95,
        }
        for method_index, (field, _) in enumerate(METHODS):
            row[field] = float(1 - index / 10 - method_index / 1000)
        row["confidence_sdc"] = sdc_scores[index]
        # Exact ties exercise the analytic curve convention in every panel.
        row["confidence_dice_m32"] = [0.9, 0.7, 0.7, 0.2, 0.2][index]
        row["confidence_nhd_m32"] = [0.8, 0.5, 0.5, 0.1, 0.1][index]
        row["confidence_nhd95_m32"] = [0.95, 0.6, 0.6, 0.3, 0.3][index]
        rows.append(row)
    return rows


def _write_assembly(
    directory,
    *,
    dataset="pet",
    condition="clipseg-general",
    artifact_type=ASSEMBLY_ARTIFACT_TYPE,
):
    directory = Path(directory)
    directory.mkdir(parents=True)
    run_id = f"{dataset}-{condition}-plot-run"
    rows = _rows(run_id)
    records = directory / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 2,
        "artifact_type": artifact_type,
        "run_id": run_id,
        "condition": condition,
        "dataset": dataset,
        "split": "test",
        "num_images": len(rows),
        "num_rows": len(rows),
        "jsonl_sha256": _sha256(records),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode()
        ).hexdigest(),
        "risk_fields": ["risk_dice", "risk_nhd", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd_pixels", "risk_hd95_pixels"],
        "score_fields": [field for field, _ in METHODS],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return records, manifest_path


def test_curve_is_tie_aware_order_invariant_and_exactly_matches_binary_aurc():
    confidence = np.array([3.0, 2.0, 2.0])
    risk = np.array([0.0, 0.2, 1.0])
    coverage, curve = tie_aware_risk_coverage_curve(confidence, risk)

    assert coverage == pytest.approx([1 / 3, 2 / 3, 1])
    assert curve == pytest.approx([0.0, 0.3, 0.4])
    assert curve.mean() == pytest.approx(tie_aware_expected_aurc(confidence, risk))

    _, permuted = tie_aware_risk_coverage_curve(confidence[[0, 2, 1]], risk[[0, 2, 1]])
    assert permuted == pytest.approx(curve)

    _, all_tied = tie_aware_risk_coverage_curve([1, 1, 1], risk)
    assert all_tied == pytest.approx([risk.mean()] * 3)


@pytest.mark.parametrize(
    ("confidence", "risk", "message"),
    [
        ([], [], "non-empty"),
        ([1, 2], [0.1], "same length"),
        ([[1, 2]], [[0.1, 0.2]], "one-dimensional"),
        ([1, float("nan")], [0.1, 0.2], "finite"),
    ],
)
def test_curve_rejects_invalid_arrays(confidence, risk, message):
    with pytest.raises(ValueError, match=message):
        tie_aware_risk_coverage_curve(confidence, risk)


def test_inputs_are_explicit_hashed_assemblies_and_may_name_either_file(tmp_path):
    records, manifest = _write_assembly(tmp_path / "assembly")

    from_records = load_assembled_conditions([records])
    from_manifest = load_assembled_conditions([manifest])
    assert from_records[0].rows == from_manifest[0].rows
    assert from_records[0].manifest_path == manifest

    with pytest.raises(ValueError, match="exactly once"):
        load_assembled_conditions([records, manifest])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_assembled_conditions([tmp_path / "missing.jsonl"])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_assembled_conditions([tmp_path / "assembly"])


def test_loader_rejects_nonassembly_and_analyzer_schema_errors(tmp_path):
    legacy_records, _ = _write_assembly(
        tmp_path / "legacy", artifact_type="selectseg.legacy"
    )
    with pytest.raises(ValueError, match="not a final binary assembly"):
        load_assembled_conditions([legacy_records])

    records, manifest = _write_assembly(tmp_path / "bad-schema")
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 1
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_assembled_conditions([records])


def test_three_panels_use_the_predeclared_matched_score_and_sdc(tmp_path):
    records, _ = _write_assembly(tmp_path / "assembly")
    condition = load_assembled_conditions([records])[0]

    assert [(spec.risk_field, spec.score_field) for spec in RISK_SPECS] == [
        ("risk_dice", "confidence_dice_m32"),
        ("risk_nhd", "confidence_nhd_m32"),
        ("risk_nhd95", "confidence_nhd95_m32"),
    ]
    for spec in RISK_SPECS:
        curves = condition_curves(condition, spec)
        assert set(curves) == {
            "coverage",
            "matched",
            "sdc",
            "oracle",
            "random",
            "matched_aurc",
            "sdc_aurc",
            "oracle_aurc",
            "random_aurc",
        }
        assert np.mean(curves["matched"]) == pytest.approx(curves["matched_aurc"])
        assert np.mean(curves["sdc"]) == pytest.approx(curves["sdc_aurc"])
        assert np.mean(curves["oracle"]) == pytest.approx(curves["oracle_aurc"])
        assert curves["random"] == pytest.approx(
            [curves["random_aurc"]] * len(condition.rows)
        )


def test_all_indexed_mode_contains_the_complete_cross_loss_overlay(tmp_path):
    records, _ = _write_assembly(tmp_path / "assembly")
    condition = load_assembled_conditions([records])[0]

    for risk_spec in RISK_SPECS:
        curves = condition_all_indexed_curves(condition, risk_spec)
        assert set(curves["indexed"]) == {spec.score_field for spec in RISK_SPECS}
        assert set(curves["indexed_aurcs"]) == set(curves["indexed"])
        risks = [row[risk_spec.risk_field] for row in condition.rows]
        for indexed_spec in RISK_SPECS:
            confidence = [row[indexed_spec.score_field] for row in condition.rows]
            assert np.mean(curves["indexed"][indexed_spec.score_field]) == (
                pytest.approx(tie_aware_expected_aurc(confidence, risks))
            )

    outputs = render_conditions(
        [condition],
        tmp_path / "all-indexed",
        all_indexed=True,
    )
    repeated = render_conditions(
        [condition],
        tmp_path / "all-indexed-repeat",
        all_indexed=True,
    )
    assert [path.name for path in outputs] == ["risk_coverage_all_indexed_pet.pdf"]
    assert outputs[0].read_bytes().startswith(b"%PDF")
    assert outputs[0].read_bytes() == repeated[0].read_bytes()


def test_publication_pdf_and_optional_png_are_byte_deterministic(tmp_path):
    records, _ = _write_assembly(tmp_path / "assembly")
    condition = load_assembled_conditions([records])[0]

    first = render_conditions([condition], tmp_path / "first", png=True, dpi=90)
    second = render_conditions([condition], tmp_path / "second", png=True, dpi=90)
    assert [path.name for path in first] == [
        "risk_coverage_pet.pdf",
        "risk_coverage_pet.png",
    ]
    assert [path.name for path in second] == [path.name for path in first]
    assert first[0].read_bytes().startswith(b"%PDF")
    assert first[1].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_cli_requires_explicit_inputs_and_writes_one_pdf_per_dataset(tmp_path):
    first, _ = _write_assembly(
        tmp_path / "pet", dataset="pet", condition="clipseg-general"
    )
    second, second_manifest = _write_assembly(
        tmp_path / "isic", dataset="isic", condition="deeplabv3-target"
    )
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path / "unused")])

    outputs = main(
        [
            "--inputs",
            str(first),
            str(second_manifest),
            "--allow-incomplete",
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )
    assert [path.name for path in outputs] == [
        "risk_coverage_pet.pdf",
        "risk_coverage_isic.pdf",
    ]
    assert all(path.is_file() for path in outputs)
    assert second.is_file()
