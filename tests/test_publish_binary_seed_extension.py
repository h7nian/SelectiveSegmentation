import copy
import hashlib
import json

import pytest

from scripts.analyze_binary import CONTRASTS, METHODS, RISKS
from scripts.analyze_binary_seed_extension import (
    COHORT_JOIN_FIELDS,
    _analysis_source_sha256,
    _contrast_seed_summary,
    _gate_c,
    _three_seed_summary,
)
from scripts.publish_binary_seed_extension import publish_seed_table
from scripts.render_binary_seed_extension import load_analysis, render_table
from selectseg.binary_seed_extension import _sha256


DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
CONDITIONS = ("clipseg-target", "deeplabv3-target")


def _analysis_payload():
    contrast_values = {0: 0.0123, 1: -0.0045, 2: 0.0}
    cells = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            per_seed_raw = {
                seed: {
                    risk: {method: 0.5 for method, _ in METHODS} for risk, _ in RISKS
                }
                for seed in (0, 1, 2)
            }
            for contrast in CONTRASTS:
                for seed, value in contrast_values.items():
                    # nHD is the shared anchor where two contrasts use risk_nhd.
                    if contrast.name == "nhd_vs_nhd95_under_nhd":
                        per_seed_raw[seed][contrast.risk][contrast.right] = 0.5 - value
                    else:
                        per_seed_raw[seed][contrast.risk][contrast.left] = 0.5 + value
                        per_seed_raw[seed][contrast.risk][contrast.right] = 0.5
            raw_summary = {
                risk: {
                    method: _three_seed_summary(
                        {seed: per_seed_raw[seed][risk][method] for seed in (0, 1, 2)}
                    )
                    for method, _ in METHODS
                }
                for risk, _ in RISKS
            }
            contrast_summary = {
                contrast.name: _contrast_seed_summary(
                    {
                        seed: (
                            per_seed_raw[seed][contrast.risk][contrast.left]
                            - per_seed_raw[seed][contrast.risk][contrast.right]
                        )
                        for seed in (0, 1, 2)
                    }
                )
                for contrast in CONTRASTS
            }
            sources = {}
            for seed in (0, 1, 2):
                stem = f"{dataset}-{condition}-seed-{seed}"
                sources[str(seed)] = {
                    "records": f"outputs/test/{stem}/records.jsonl",
                    "records_sha256": hashlib.sha256(
                        f"{stem}-records".encode()
                    ).hexdigest(),
                    "manifest": f"outputs/test/{stem}/manifest.json",
                    "manifest_sha256": hashlib.sha256(
                        f"{stem}-manifest".encode()
                    ).hexdigest(),
                }
            cells.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "model": (
                        "clipseg" if condition == "clipseg-target" else "deeplabv3"
                    ),
                    "num_images_per_seed": 2,
                    "sources": sources,
                    "summary": {
                        "raw_aurc": raw_summary,
                        "contrasts": contrast_summary,
                    },
                }
            )
    return {
        "schema_version": 1,
        "analysis": {
            "estimand": (
                "descriptive target-model training-seed variation over seeds 0,1,2"
            ),
            "replication_unit": "one independently trained checkpoint",
            "inference": "none; no image pooling and no seed-level hypothesis test",
            "statistics": ("three values, mean, range, and sample standard deviation"),
            "aurc_scale": "raw [0,1]; renderers may display 100 x AURC",
            "contrast_definition": "AURC(left score) - AURC(right score)",
            "contrast_definitions": [contrast.__dict__ for contrast in CONTRASTS],
            "cohort_join_fields": list(COHORT_JOIN_FIELDS),
        },
        "provenance": {
            "downstream_lock": {
                "path": "outputs/test/downstream.lock.json",
                "sha256": "a" * 64,
            },
            "canonical_seed0": {
                "path": "outputs/test/canonical-analysis.json",
                "sha256": "b" * 64,
                "campaign_lock_path": "outputs/test/campaign.lock.json",
                "campaign_lock_sha256": "c" * 64,
            },
            "analysis_source_sha256": _analysis_source_sha256(),
        },
        "cells": cells,
        "gate_c": _gate_c(cells),
    }


def _valid_inputs(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(_analysis_payload(), sort_keys=True) + "\n", encoding="utf-8"
    )
    analysis_sha = _sha256(analysis_path)
    analysis, by_key, observed = load_analysis(
        analysis_path, expected_sha256=analysis_sha
    )
    table_path = tmp_path / "seed_robustness.tex"
    table_path.write_text(
        render_table(analysis, by_key, analysis_sha256=observed), encoding="utf-8"
    )
    return analysis_path, analysis_sha, table_path, _sha256(table_path)


def _publish(inputs, destination):
    analysis_path, analysis_sha, table_path, table_sha = inputs
    return publish_seed_table(
        analysis_path,
        expected_analysis_sha256=analysis_sha,
        table_path=table_path,
        expected_table_sha256=table_sha,
        destination=destination,
    )


def test_seed_publisher_recomputes_exact_tex_and_is_idempotent(tmp_path):
    inputs = _valid_inputs(tmp_path)
    destination = tmp_path / "docs" / "Tables" / "seed_robustness.tex"
    destination.parent.mkdir(parents=True)

    first = _publish(inputs, destination)
    assert first["status"] == "published"
    assert destination.read_bytes() == inputs[2].read_bytes()
    assert _sha256(destination) == inputs[3]

    second = _publish(inputs, destination)
    assert second["status"] == "unchanged"
    assert destination.read_bytes() == inputs[2].read_bytes()


def test_seed_publisher_rejects_wrong_analysis_and_table_hashes(tmp_path):
    inputs = _valid_inputs(tmp_path)
    destination = tmp_path / "docs" / "Tables" / "seed_robustness.tex"
    destination.parent.mkdir(parents=True)
    analysis_path, analysis_sha, table_path, table_sha = inputs

    with pytest.raises(ValueError, match="analysis SHA-256 mismatch"):
        publish_seed_table(
            analysis_path,
            expected_analysis_sha256="f" * 64,
            table_path=table_path,
            expected_table_sha256=table_sha,
            destination=destination,
        )
    with pytest.raises(ValueError, match="rendered seed table SHA-256 mismatch"):
        publish_seed_table(
            analysis_path,
            expected_analysis_sha256=analysis_sha,
            table_path=table_path,
            expected_table_sha256="f" * 64,
            destination=destination,
        )
    assert not destination.exists()


def test_seed_publisher_rejects_self_consistent_hash_for_wrong_tex(tmp_path):
    analysis_path, analysis_sha, table_path, _ = _valid_inputs(tmp_path)
    table_path.write_text(table_path.read_text() + "% changed\n", encoding="utf-8")
    destination = tmp_path / "docs" / "Tables" / "seed_robustness.tex"
    destination.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="differs from the current renderer"):
        publish_seed_table(
            analysis_path,
            expected_analysis_sha256=analysis_sha,
            table_path=table_path,
            expected_table_sha256=_sha256(table_path),
            destination=destination,
        )
    assert not destination.exists()


def test_seed_publisher_rejects_symlink_inputs_and_destination(tmp_path):
    inputs = _valid_inputs(tmp_path)
    analysis_path, analysis_sha, table_path, table_sha = inputs
    analysis_link = tmp_path / "analysis-link.json"
    analysis_link.symlink_to(analysis_path)
    table_link = tmp_path / "table-link.tex"
    table_link.symlink_to(table_path)
    destination = tmp_path / "docs" / "Tables" / "seed_robustness.tex"
    destination.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="non-symlink"):
        publish_seed_table(
            analysis_link,
            expected_analysis_sha256=analysis_sha,
            table_path=table_path,
            expected_table_sha256=table_sha,
            destination=destination,
        )
    with pytest.raises(FileNotFoundError, match="non-symlink"):
        publish_seed_table(
            analysis_path,
            expected_analysis_sha256=analysis_sha,
            table_path=table_link,
            expected_table_sha256=table_sha,
            destination=destination,
        )

    decoy = tmp_path / "decoy.tex"
    decoy.write_text("decoy\n", encoding="utf-8")
    destination.symlink_to(decoy)
    with pytest.raises(FileExistsError, match="symlink manuscript destination"):
        _publish(inputs, destination)
    assert decoy.read_text() == "decoy\n"


def test_seed_publisher_never_overwrites_different_table(tmp_path):
    inputs = _valid_inputs(tmp_path)
    destination = tmp_path / "docs" / "Tables" / "seed_robustness.tex"
    destination.parent.mkdir(parents=True)
    destination.write_text("% independently edited\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite different"):
        _publish(inputs, destination)
    assert destination.read_text() == "% independently edited\n"


def test_seed_publisher_rejects_symlink_parent(tmp_path):
    inputs = _valid_inputs(tmp_path)
    real_parent = tmp_path / "real-tables"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-tables"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="real directory"):
        _publish(inputs, linked_parent / "seed_robustness.tex")
    assert not (real_parent / "seed_robustness.tex").exists()


def test_seed_analysis_loader_recomputes_statistics_contrasts_and_gate(tmp_path):
    payload = _analysis_payload()

    inconsistent_mean = copy.deepcopy(payload)
    inconsistent_mean["cells"][0]["summary"]["contrasts"][CONTRASTS[0].name][
        "mean"
    ] += 0.1
    path = tmp_path / "bad-mean.json"
    path.write_text(json.dumps(inconsistent_mean) + "\n")
    with pytest.raises(ValueError, match="inconsistent with the three seed"):
        load_analysis(path, expected_sha256=_sha256(path))

    inconsistent_contrast = copy.deepcopy(payload)
    raw = inconsistent_contrast["cells"][0]["summary"]["raw_aurc"]
    contrast = CONTRASTS[0]
    raw[contrast.risk][contrast.left] = _three_seed_summary({0: 0.61, 1: 0.61, 2: 0.61})
    path = tmp_path / "bad-contrast.json"
    path.write_text(json.dumps(inconsistent_contrast) + "\n")
    with pytest.raises(ValueError, match="differs from the raw AURCs"):
        load_analysis(path, expected_sha256=_sha256(path))

    inconsistent_gate = copy.deepcopy(payload)
    inconsistent_gate["gate_c"]["fired"] = not inconsistent_gate["gate_c"]["fired"]
    path = tmp_path / "bad-gate.json"
    path.write_text(json.dumps(inconsistent_gate) + "\n")
    with pytest.raises(ValueError, match="Gate C decision"):
        load_analysis(path, expected_sha256=_sha256(path))
