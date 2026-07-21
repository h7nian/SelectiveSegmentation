"""Tests for the strict, redacted public provenance exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.maintenance.export_provenance import (
    BaseModelSpec,
    PHASES,
    build_public_provenance,
    write_public_provenance,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _artifact(dataset, condition, model, digit):
    return {
        "manifest_path": f"/private/alice/artifacts/{condition}/manifest.json",
        "manifest_sha256": digit * 64,
        "artifact_id": digit * 16,
        "dataset": dataset,
        "condition": condition,
        "model": model,
        "split": "test",
        "checkpoint_sha256": chr(ord(digit) + 1) * 64,
        "source_sha256": "c" * 64,
        "sample_id_sha256": chr(ord(digit) + 2) * 64,
        "num_samples": 3,
    }


def _write_receipt(path, phase, count, *, unresolved=False):
    rows = []
    for index in range(count):
        common = {
            "receipt_schema_version": 1,
            "created_utc": "2026-07-19T00:00:00+00:00",
            "phase": phase,
            "key": [f"experiment-{index}", "private-partition"],
            "command": [
                "sbatch",
                "--account",
                "private-account",
                "/home/alice/run.sh",
                "--password",
                "do-not-publish",
            ],
        }
        rows.append({**common, "status": "submitting", "job_id": None})
        if not (unresolved and index == count - 1):
            rows.append(
                {**common, "status": "submitted", "job_id": f"secret-job-{index}"}
            )
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return Path(path)


def _fixture(tmp_path):
    artifacts = [
        _artifact("toy", "clipseg-target", "clipseg", "1"),
        _artifact("toy", "deeplabv3-target", "deeplabv3", "4"),
    ]
    lock = {
        "lock_schema_version": 1,
        "campaign_id": "unit-public-v1",
        "config": {
            "path": "/home/alice/private/config.json",
            "sha256": "a" * 64,
        },
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "estimator": {
            "spec_path": "/home/alice/private/estimator.json",
            "spec_sha256": "b" * 64,
            "estimator_id": "midpoint-v1",
            "target_measure": "uniform-threshold",
        },
        "paths": {
            "artifact_output_root": "/scratch/private/alice/artifacts",
            "common_output_root": "/scratch/private/alice/common",
            "simulation_output_root": "/scratch/private/alice/scores",
            "assembly_output_root": "/scratch/private/alice/assembled",
        },
        "artifacts": artifacts,
    }
    lock_path = _write_json(tmp_path / "private" / "campaign.lock.json", lock)
    lock_sha = _sha256(lock_path)
    keys = [(item["dataset"], item["condition"]) for item in artifacts]

    analysis = {
        "schema_version": 2,
        "provenance": {
            "binding": "campaign-lock",
            "campaign_id": "unit-public-v1",
            "campaign_lock": {
                "logical_name": "/home/alice/private/campaign.lock.json",
                "sha256": lock_sha,
            },
            "config_sha256": "a" * 64,
            "analysis_source_sha256": "d" * 64,
            "inputs": [
                {
                    "logical_id": f"{dataset}/{condition}/run-{index}",
                    "dataset": dataset,
                    "condition": condition,
                    "assembly_run_id": f"run-{index}",
                    "artifact_id": artifacts[index]["artifact_id"],
                    "manifest_sha256": ("e", "f")[index] * 64,
                    "records_sha256": ("7", "8")[index] * 64,
                    "sample_id_sha256": artifacts[index]["sample_id_sha256"],
                    "num_samples": 3,
                }
                for index, (dataset, condition) in enumerate(keys)
            ],
        },
        "analysis": {"bootstrap_samples": 100},
        "conditions": [
            {
                "dataset": dataset,
                "condition": condition,
                "num_rows": 3,
                "num_image_clusters": 3,
                "jsonl": "/home/alice/private/records.jsonl",
            }
            for dataset, condition in keys
        ],
        "multiple_testing": {},
    }
    analysis_path = _write_json(tmp_path / "private" / "analysis.json", analysis)
    diagnostics = {
        "schema_version": 1,
        "artifact_type": "selectseg.diagnostics_analysis",
        "campaign": {
            "campaign_id": "unit-public-v1",
            "lock_path": "/home/alice/private/campaign.lock.json",
            "lock_sha256": lock_sha,
            "num_locked_conditions": 2,
            "num_analyzed_conditions": 2,
            "complete_predeclared_campaign": True,
            "diagnostics_source_sha256": "i" * 64,
        },
        "scope": {"private_note": "do-not-publish"},
        "aggregation": {"private_path": "/home/alice"},
        "conditions": [
            {
                "dataset": dataset,
                "condition": condition,
                "num_images": 3,
                "num_pixels": 30,
                "diagnostics_path": "/home/alice/private/diagnostics.json",
            }
            for dataset, condition in keys
        ],
    }
    diagnostics_path = _write_json(
        tmp_path / "private" / "diagnostics_analysis.json", diagnostics
    )

    expected_counts = {
        "freeze": 2,
        "common": 2,
        "score": 4,
        "assemble": 2,
        "diagnose": 2,
    }
    receipts = {
        phase: _write_receipt(
            tmp_path / "private" / f"{phase}.receipt.jsonl", phase, count
        )
        for phase, count in expected_counts.items()
    }
    configs = {}
    histories = {}
    for dataset, model in (("toy", "clipseg"), ("toy", "deeplabv3")):
        logical_id = f"{dataset}/{model}/seed-0"
        configs[logical_id] = _write_json(
            tmp_path / "private" / model / "train_config.json",
            {
                "dataset": dataset,
                "model": model,
                "seed": "0",
                "output_dir": f"/home/alice/private/{model}",
                "password": "do-not-publish",
            },
        )
    histories["toy/clipseg/seed-0"] = _write_json(
        tmp_path / "private" / "clipseg" / "history.json",
        [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.25}],
    )
    base_models = [
        BaseModelSpec("clipseg", "publisher/clipseg-rd64-refined", "abc1234", "a" * 64),
        BaseModelSpec(
            "deeplabv3", "torchvision/DeepLabV3-ResNet50", "v0.27.1", "b" * 64
        ),
    ]
    return {
        "lock": lock_path,
        "analysis": analysis_path,
        "diagnostics": diagnostics_path,
        "receipts": receipts,
        "configs": configs,
        "histories": histories,
        "base_models": base_models,
        "expected_counts": expected_counts,
    }


def _build(files):
    return build_public_provenance(
        files["lock"],
        files["analysis"],
        files["diagnostics"],
        phase_receipts=files["receipts"],
        training_configs=files["configs"],
        training_histories=files["histories"],
        base_models=files["base_models"],
    )


def test_export_is_deterministic_portable_and_strictly_redacted(tmp_path):
    files = _fixture(tmp_path)
    result = _build(files)
    output = write_public_provenance(result, tmp_path / "public.json")
    first = output.read_bytes()

    # Reverse every caller-controlled collection: the public bytes remain stable.
    files["receipts"] = dict(reversed(list(files["receipts"].items())))
    files["configs"] = dict(reversed(list(files["configs"].items())))
    files["histories"] = dict(reversed(list(files["histories"].items())))
    files["base_models"] = list(reversed(files["base_models"]))
    write_public_provenance(_build(files), output)
    assert output.read_bytes() == first

    text = first.decode()
    forbidden_values = (
        "/home/",
        "/scratch/",
        "alice",
        "do-not-publish",
        "secret-job",
        "private-account",
        "private-partition",
        "sbatch",
    )
    assert not any(value in text for value in forbidden_values)
    forbidden_keys = {
        "path",
        "job_id",
        "command",
        "account",
        "partition",
        "created_utc",
        "username",
        "password",
        "secret",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(json.loads(text))
    assert [item["phase_id"] for item in result["phases"]] == list(PHASES)
    assert {
        item["phase_id"]: (item["submitted_count"], item["completed_count"])
        for item in result["phases"]
    } == {phase: (count, count) for phase, count in files["expected_counts"].items()}
    training_by_id = {item["logical_id"]: item for item in result["training"]}
    assert training_by_id["toy/clipseg/seed-0"]["history_record_count"] == 2


def test_write_revalidates_nested_whitelists_before_publication(tmp_path):
    files = _fixture(tmp_path)
    result = _build(files)
    result["campaign"]["protocol"]["private_path"] = "/home/alice/private"
    with pytest.raises(ValueError, match="must contain exactly"):
        write_public_provenance(result, tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_analysis_must_bind_to_exact_lock_bytes(tmp_path):
    files = _fixture(tmp_path)
    analysis = json.loads(files["analysis"].read_text())
    analysis["provenance"]["campaign_lock"]["sha256"] = "0" * 64
    _write_json(files["analysis"], analysis)
    with pytest.raises(ValueError, match="does not match campaign-lock bytes"):
        _build(files)


def test_diagnostics_must_bind_to_exact_lock_bytes(tmp_path):
    files = _fixture(tmp_path)
    diagnostics = json.loads(files["diagnostics"].read_text())
    diagnostics["campaign"]["lock_sha256"] = "0" * 64
    _write_json(files["diagnostics"], diagnostics)
    with pytest.raises(ValueError, match="does not match campaign-lock bytes"):
        _build(files)


def test_unresolved_or_incomplete_receipt_is_rejected(tmp_path):
    files = _fixture(tmp_path)
    files["receipts"]["score"] = _write_receipt(
        tmp_path / "private" / "score-unresolved.jsonl",
        "score",
        4,
        unresolved=True,
    )
    with pytest.raises(ValueError, match="unresolved submissions"):
        _build(files)

    files = _fixture(tmp_path / "second")
    files["receipts"]["score"] = _write_receipt(
        tmp_path / "second" / "private" / "score-short.jsonl", "score", 3
    )
    with pytest.raises(ValueError, match="expected 4"):
        _build(files)


def test_missing_phase_receipt_is_rejected(tmp_path):
    files = _fixture(tmp_path)
    del files["receipts"]["diagnose"]
    with pytest.raises(ValueError, match="must contain exactly"):
        _build(files)


def test_training_files_are_complete_and_identity_bound(tmp_path):
    files = _fixture(tmp_path)
    del files["configs"]["toy/deeplabv3/seed-0"]
    with pytest.raises(ValueError, match="missing training configs"):
        _build(files)

    files = _fixture(tmp_path / "second")
    config_path = files["configs"].pop("toy/clipseg/seed-0")
    files["configs"]["toy/clipseg/seed-1"] = config_path
    del files["histories"]["toy/clipseg/seed-0"]
    with pytest.raises(ValueError, match="does not match config identity"):
        _build(files)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("identifier", "/home/alice/model", "portable public identifier"),
        ("identifier", "publisher/secret-token", "secret-like marker"),
        ("revision", "main", "not immutable"),
        ("weights_sha256", "not-a-digest", "SHA-256"),
    ],
)
def test_base_model_spec_rejects_paths_secrets_and_mutable_revisions(
    tmp_path, field, value, match
):
    files = _fixture(tmp_path)
    original = files["base_models"][0]
    values = {
        "model_id": original.model_id,
        "identifier": original.identifier,
        "revision": original.revision,
        "weights_sha256": original.weights_sha256,
    }
    values[field] = value
    files["base_models"][0] = BaseModelSpec(**values)
    with pytest.raises(ValueError, match=match):
        _build(files)


def test_duplicate_json_keys_and_nonfinite_inputs_fail_closed(tmp_path):
    files = _fixture(tmp_path)
    files["analysis"].write_text('{"schema_version":2,"schema_version":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _build(files)

    files = _fixture(tmp_path / "second")
    files["configs"]["toy/clipseg/seed-0"].write_text(
        '{"dataset":"toy","model":"clipseg","seed":"0","loss":NaN}\n'
    )
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        _build(files)
