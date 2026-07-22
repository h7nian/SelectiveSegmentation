import copy
import hashlib
import json

import pytest

from scripts.render import seed_gate as renderer


def _summary(values, *, reversal=False, seed0_majority=True):
    return {
        "values": {str(index): value for index, value in enumerate(values)},
        "direction_reversal": reversal,
        "seed0_is_majority_direction": seed0_majority,
    }


def _fixture():
    reversals = {
        ("clipseg-target", "isic"): _summary(
            (0.00002, 0.00020, -0.00011), reversal=True
        ),
        ("clipseg-target", "tn3k"): _summary(
            (0.00014, -0.00037, -0.00036),
            reversal=True,
            seed0_majority=False,
        ),
        ("deeplabv3-target", "kvasir"): _summary(
            (-0.00023, -0.00047, 0.00016), reversal=True
        ),
        ("deeplabv3-target", "isic"): _summary(
            (0.00024, -0.00001, -0.00004),
            reversal=True,
            seed0_majority=False,
        ),
        ("deeplabv3-target", "pet"): _summary(
            (0.00024, -0.00067, -0.00086),
            reversal=True,
            seed0_majority=False,
        ),
    }
    by_key = {}
    for condition in renderer.CONDITIONS:
        for dataset in renderer.DATASETS:
            summary = reversals.get(
                (condition, dataset), _summary((0.001, 0.002, 0.003))
            )
            by_key[(dataset, condition)] = {
                "summary": {"contrasts": {renderer.CONTRAST: summary}}
            }
    nonmajority = [
        {
            "condition": condition,
            "dataset": dataset,
            "contrast": renderer.CONTRAST,
        }
        for condition, dataset in sorted(renderer.EXPECTED_DAGGER_CELLS)
    ]
    analysis = {
        "gate_c": {
            "fired": True,
            "direction_reversal_counts": {renderer.CONTRAST: 5},
            "contrasts_with_at_least_three_reversals": [renderer.CONTRAST],
            "seed0_not_majority_cells": nonmajority,
        }
    }
    return analysis, by_key


def test_render_is_exactly_the_five_target_reversals():
    analysis, by_key = _fixture()
    digest = "a" * 64
    payload = renderer.render_table(analysis, by_key, analysis_sha256=digest)

    assert f"% Source seed analysis SHA-256: {digest}" in payload
    assert "Oxford Pet & Kvasir-SEG & FIVES & ISIC & TN3K" in payload
    assert r"\begin{tabular}{lccccc}" in payload
    assert payload.count(r"\bigl(") == 5
    assert payload.count(r"^{\dagger}") == 3
    assert payload.count("--") == 5
    assert (
        "CLIP-T & -- & -- & -- & "
        "$\\bigl(+0.002/+0.020/-0.011\\bigr)$" in payload
    )
    assert (
        "DL-T & $\\bigl(+0.024/-0.067/-0.086\\bigr)^{\\dagger}$ & "
        "$\\bigl(-0.023/-0.047/+0.016\\bigr)$ & --" in payload
    )
    assert "negative values favor HD" in payload
    assert "three independently trained checkpoints" in payload


def test_render_rejects_a_changed_gate_result():
    analysis, by_key = _fixture()
    changed = copy.deepcopy(by_key)
    changed[("isic", "clipseg-target")]["summary"]["contrasts"][
        renderer.CONTRAST
    ]["direction_reversal"] = False

    with pytest.raises(ValueError, match="fixed Gate-C result"):
        renderer.render_table(analysis, changed, analysis_sha256="a" * 64)


def test_load_analysis_requires_exact_hash_and_regular_file(tmp_path, monkeypatch):
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps({"gate_c": {}}), encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    sentinel = {("dataset", "condition"): object()}
    monkeypatch.setattr(renderer, "validate_analysis_document", lambda value: sentinel)

    analysis, by_key, observed = renderer.load_analysis(
        source, expected_sha256=digest.upper()
    )
    assert analysis == {"gate_c": {}}
    assert by_key is sentinel
    assert observed == digest

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        renderer.load_analysis(source, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        renderer.load_analysis(source, expected_sha256="not-a-digest")

    link = tmp_path / "linked-analysis.json"
    link.symlink_to(source)
    with pytest.raises(FileNotFoundError, match="non-symlink"):
        renderer.load_analysis(link, expected_sha256=digest)


def test_no_overwrite_writer_preserves_existing_destination(tmp_path):
    destination = tmp_path / "table.tex"
    renderer._write_text_new(destination, "first\n")
    assert destination.read_text(encoding="utf-8") == "first\n"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        renderer._write_text_new(destination, "second\n")
    assert destination.read_text(encoding="utf-8") == "first\n"
    assert list(tmp_path.glob(".table.tex.*.tmp")) == []
