"""Tests for the descriptive mechanism-evidence renderer."""

from scripts.render.mechanism import (
    CONDITIONS,
    DATASETS,
    condition_diagnostics,
    render_table,
)


def test_condition_diagnostics_has_expected_signs_on_ordered_rows():
    rows = []
    for index in range(6):
        rows.append(
            {
                "risk_dice": index / 10,
                "risk_nhd": index / 10,
                "confidence_dice_m32": -index,
                "confidence_dice_exact": -index,
                "confidence_nhd_m32": -index,
                "confidence_sdc": index,
                "confidence_foreground_entropy": -index,
            }
        )
    result = condition_diagnostics(rows)
    assert result["dice_fg_redundancy"] == 1.0
    assert result["dice_minus_sdc"] == 2.0
    assert result["exact_m32_difference"] == 0.0


def test_table_is_symmetric_across_datasets_and_models():
    values = {
        (dataset, condition): {
            "dice_fg_redundancy": 0.9,
            "dice_minus_sdc": 0.0,
            "nhd_spatial_increment": 0.2,
            "exact_m32_difference": 0.001,
        }
        for dataset in DATASETS
        for condition in CONDITIONS
    }
    table = render_table(values)
    assert "Each cell is CLIP-T/DL-T" in table
    assert table.count("0.900/0.900") == len(DATASETS)
    assert "\\begin{table}[h!]" in table
