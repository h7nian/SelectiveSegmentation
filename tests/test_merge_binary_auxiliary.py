"""Synthetic tests for the strict canonical/auxiliary run merger."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.maintenance.merge_auxiliary import (
    ADDED_SCORE_FIELDS,
    AUXILIARY_SCORE_FIELDS,
    CANONICAL_SCORE_FIELDS,
    merge_roots,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quadrature(counts):
    return {
        str(count): {
            "rule": "midpoint",
            "nodes": [(index + 0.5) / count for index in range(count)],
            "weights": [1 / count] * count,
        }
        for count in counts
    }


def _row(index, *, run_id, auxiliary):
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
        "truth_foreground_fraction": 0.1 + index / 100,
        "prediction_foreground_fraction": 0.2 + index / 100,
        "risk_dice": 0.1 + index / 10,
        "risk_nhd95": 0.05 + index / 20,
        "risk_hd95_pixels": 1.0 + index,
        "confidence_sdc": 0.9 - index / 10,
        "confidence_mean_max_probability": 0.8 - index / 20,
        "confidence_negative_entropy": -0.1 - index / 20,
        "confidence_dice_m2": -0.2 - index / 10,
        "confidence_nhd95_m2": -0.1 - index / 20,
    }
    if auxiliary:
        for offset, field in enumerate(ADDED_SCORE_FIELDS):
            row[field] = -0.3 - index / 10 - offset / 100
    else:
        row.update(
            {
                "confidence_dice_m8": -0.21 - index / 10,
                "confidence_nhd95_m8": -0.11 - index / 20,
                "confidence_dice_m32": -0.22 - index / 10,
                "confidence_nhd95_m32": -0.12 - index / 20,
            }
        )
    return row


def _write_run(root, *, auxiliary=False, condition="clipseg-target"):
    role = "auxiliary" if auxiliary else "canonical"
    run_id = f"{role[:3]}-{condition}"
    rows = [_row(index, run_id=run_id, auxiliary=auxiliary) for index in range(3)]
    run_dir = Path(root) / "pet" / condition / run_id
    run_dir.mkdir(parents=True)
    records = run_dir / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scores = AUXILIARY_SCORE_FIELDS if auxiliary else CANONICAL_SCORE_FIELDS
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "condition": condition,
        "model": "clipseg",
        "dataset": "pet",
        "split": "test",
        "num_images": len(rows),
        "num_rows": len(rows),
        "checkpoint": {"path": "checkpoint.pt", "sha256": "c" * 64},
        "base_model": {"name": "clipseg", "source": "synthetic"},
        "source_sha256": ("b" if auxiliary else "a") * 64,
        "cohort": "all images from a native binary segmentation split",
        "decision_rule": {
            "form": "foreground_probability >= gamma",
            "gamma": 0.5,
        },
        "preprocessing": {"resize": "synthetic"},
        "losses": {"dice": "synthetic", "hd95": "synthetic"},
        "risk_fields": ["risk_dice", "risk_nhd95"],
        "auxiliary_fields": ["risk_hd95_pixels"],
        "score_fields": list(scores),
        "quadrature": _quadrature((2,) if auxiliary else (2, 8, 32)),
        "void_policy": "total binary domain",
        "sdc_empty_convention": "zero",
        "sample_id_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode()
        ).hexdigest(),
        "jsonl_sha256": _sha256(records),
        "command": [
            "python",
            "-m",
            "selectseg.evaluate",
            *(["--batch-size", "8"] if auxiliary else []),
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def _sources(tmp_path):
    canonical = tmp_path / "canonical"
    auxiliary = tmp_path / "auxiliary"
    canonical_run = _write_run(canonical)
    auxiliary_run = _write_run(auxiliary, auxiliary=True)
    return canonical, auxiliary, canonical_run, auxiliary_run


def _rewrite(run_dir, mutate):
    records = run_dir / "records.jsonl"
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    mutate(rows)
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["num_images"] = len(rows)
    manifest["num_rows"] = len(rows)
    manifest["sample_id_sha256"] = hashlib.sha256(
        "\n".join(row["sample_id"] for row in rows).encode()
    ).hexdigest()
    manifest["jsonl_sha256"] = _sha256(records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_happy_path_preserves_canonical_and_appends_only_named_scores(tmp_path):
    canonical, auxiliary, canonical_run, auxiliary_run = _sources(tmp_path)
    paths = merge_roots(canonical, auxiliary, tmp_path / "merged")
    assert len(paths) == 1

    canonical_rows = [
        json.loads(line)
        for line in (canonical_run / "records.jsonl").read_text().splitlines()
    ]
    auxiliary_rows = [
        json.loads(line)
        for line in (auxiliary_run / "records.jsonl").read_text().splitlines()
    ]
    merged_rows = [
        json.loads(line) for line in (paths[0] / "records.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((paths[0] / "manifest.json").read_text())

    assert manifest["score_fields"] == [*CANONICAL_SCORE_FIELDS, *ADDED_SCORE_FIELDS]
    assert manifest["jsonl_sha256"] == _sha256(paths[0] / "records.jsonl")
    assert manifest["canonical_source_sha256"] == "a" * 64
    assert manifest["auxiliary_source_sha256"] == "b" * 64
    assert manifest["source_manifests"]["canonical"]["sha256"] == _sha256(
        canonical_run / "manifest.json"
    )
    assert manifest["source_manifests"]["auxiliary"]["sha256"] == _sha256(
        auxiliary_run / "manifest.json"
    )
    for canonical_row, auxiliary_row, merged_row in zip(
        canonical_rows, auxiliary_rows, merged_rows
    ):
        expected = dict(canonical_row)
        expected["run_id"] = manifest["run_id"]
        expected.update({field: auxiliary_row[field] for field in ADDED_SCORE_FIELDS})
        assert merged_row == expected

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge_roots(canonical, auxiliary, tmp_path / "merged")


def test_shared_value_mismatch_is_rejected_before_writing(tmp_path):
    canonical, auxiliary, _, auxiliary_run = _sources(tmp_path)

    def mutate(rows):
        rows[1]["risk_dice"] += 1e-12

    _rewrite(auxiliary_run, mutate)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="exact shared field 'risk_dice' differs"):
        merge_roots(canonical, auxiliary, output)
    assert not output.exists()


def test_duplicate_sample_id_is_rejected(tmp_path):
    canonical, auxiliary, _, auxiliary_run = _sources(tmp_path)

    def mutate(rows):
        rows[1]["sample_id"] = rows[0]["sample_id"]

    _rewrite(auxiliary_run, mutate)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        merge_roots(canonical, auxiliary, tmp_path / "merged")


def test_unauthorized_auxiliary_row_field_is_rejected(tmp_path):
    canonical, auxiliary, _, auxiliary_run = _sources(tmp_path)

    def mutate(rows):
        rows[0]["confidence_unregistered"] = 0.25

    _rewrite(auxiliary_run, mutate)
    with pytest.raises(ValueError, match="unauthorized row schema"):
        merge_roots(canonical, auxiliary, tmp_path / "merged")


def test_output_is_byte_deterministic_across_destinations(tmp_path):
    canonical, auxiliary, _, _ = _sources(tmp_path)
    first = merge_roots(canonical, auxiliary, tmp_path / "first")[0]
    second = merge_roots(canonical, auxiliary, tmp_path / "second")[0]
    assert first.name == second.name
    assert (first / "records.jsonl").read_bytes() == (
        second / "records.jsonl"
    ).read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
