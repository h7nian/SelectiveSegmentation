"""Focused tests for the deterministic anonymous analysis artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from scripts.maintenance import build_release as artifact


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = (
    WORKSPACE_ROOT / "github"
    if (WORKSPACE_ROOT / "github" / "results" / "analysis.json").is_file()
    else WORKSPACE_ROOT
)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _public_gate_analysis(source_analysis_sha256):
    summaries = {
        ("clipseg-target", "isic"): ([0.00002, 0.00020, -0.00011], True),
        ("clipseg-target", "tn3k"): ([0.00014, -0.00037, -0.00036], False),
        ("deeplabv3-target", "kvasir"): ([-0.00023, -0.00047, 0.00016], True),
        ("deeplabv3-target", "isic"): ([0.00024, -0.00001, -0.00004], False),
        ("deeplabv3-target", "pet"): ([0.00024, -0.00067, -0.00086], False),
    }
    cells = []
    for condition in artifact.SEED_GATE_CONDITIONS:
        for dataset in artifact.SEED_GATE_DATASETS:
            values, seed0_majority = summaries.get(
                (condition, dataset), ([0.001, 0.002, 0.003], True)
            )
            cells.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "summary": {
                        "contrasts": {
                            artifact.SEED_GATE_CONTRAST: {
                                "values": {
                                    str(seed): value
                                    for seed, value in enumerate(values)
                                },
                                "direction_reversal": (condition, dataset)
                                in artifact.EXPECTED_SEED_GATE_REVERSAL_CELLS,
                                "seed0_is_majority_direction": seed0_majority,
                            }
                        }
                    },
                }
            )
    return {
        "provenance": {"source_analysis_sha256": source_analysis_sha256},
        "cells": cells,
        "gate_c": {
            "fired": True,
            "direction_reversal_counts": {artifact.SEED_GATE_CONTRAST: 5},
            "contrasts_with_at_least_three_reversals": [artifact.SEED_GATE_CONTRAST],
            "seed0_not_majority_cells": [
                {
                    "condition": condition,
                    "dataset": dataset,
                    "contrast": artifact.SEED_GATE_CONTRAST,
                }
                for condition, dataset in sorted(
                    artifact.EXPECTED_SEED_GATE_DAGGER_CELLS
                )
            ],
        },
    }


@pytest.fixture
def portable_seed_inputs(monkeypatch):
    source_analysis_sha256 = "a" * 64
    table = (
        "% AUTO-GENERATED; DO NOT EDIT.\n"
        f"% Source analysis.json SHA-256: {source_analysis_sha256}\n"
        "seed robustness table\n"
    ).encode("utf-8")
    public_analysis = _public_gate_analysis(source_analysis_sha256)
    gate_table = artifact._render_seed_gate_table(public_analysis)
    payloads = {
        artifact.PUBLIC_SEED_ANALYSIS_SOURCE: _json_bytes(public_analysis),
        artifact.PUBLIC_SEED_SCHEDULER_SOURCE: _json_bytes({"status": "complete"}),
        artifact.PUBLIC_SEED_PROVENANCE_SOURCE: _json_bytes(
            {"analysis": {"table_sha256": hashlib.sha256(table).hexdigest()}}
        ),
        artifact.PUBLIC_SEED_TABLE_SOURCE: table,
        artifact.PUBLIC_SEED_GATE_TABLE_SOURCE: gate_table,
    }
    original_reader = artifact._read_regular_source

    def read_source(repo_root, relative):
        if relative in payloads:
            return payloads[relative]
        return original_reader(repo_root, relative)

    calls = []

    def load_release(public_analysis, public_scheduler, public_provenance):
        paths = (Path(public_analysis), Path(public_scheduler), Path(public_provenance))
        assert tuple(path.name for path in paths) == (
            "seed_robustness_analysis.json",
            "seed_scheduler_summary.json",
            "seed_provenance.json",
        )
        calls.append(paths)
        values = [json.loads(path.read_bytes()) for path in paths]
        return {
            "analysis": values[0],
            "scheduler": values[1],
            "provenance": values[2],
            "sha256": {
                "analysis": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "scheduler": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
                "provenance": hashlib.sha256(paths[2].read_bytes()).hexdigest(),
            },
        }

    monkeypatch.setattr(artifact, "_read_regular_source", read_source)
    monkeypatch.setattr(artifact, "load_public_seed_release", load_release)

    def verify_replay(lock_payload, read_payload):
        assert lock_payload
        assert read_payload(
            "results/seed_records/seed-0/fives/clipseg-target/627922c22aa3aa21/records.jsonl"
        )
        return {}, {
            "verified": True,
            "condition_count": 30,
            "seed_count": 3,
            "output_sha256": {
                "analysis": hashlib.sha256(
                    payloads[artifact.PUBLIC_SEED_ANALYSIS_SOURCE]
                ).hexdigest(),
                "robustness_table": hashlib.sha256(
                    payloads[artifact.PUBLIC_SEED_TABLE_SOURCE]
                ).hexdigest(),
                "gate_table": hashlib.sha256(
                    payloads[artifact.PUBLIC_SEED_GATE_TABLE_SOURCE]
                ).hexdigest(),
            },
        }

    monkeypatch.setattr(artifact, "verify_replay_payloads", verify_replay)
    return payloads, calls


def test_repeated_builds_are_identical_verified_and_importable(
    tmp_path, monkeypatch, portable_seed_inputs
):
    payloads, validator_calls = portable_seed_inputs
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    artifact.build_anonymous_analysis_artifact(REPO_ROOT, first)
    artifact.build_anonymous_analysis_artifact(REPO_ROOT, second)

    assert first.read_bytes() == second.read_bytes()
    expected_names = artifact.expected_archive_member_names()
    assert len(artifact.RELEASE_FILES) == 114
    assert len(expected_names) == 117

    report = artifact.verify_archive(first)
    assert report == {
        "archive_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "member_count": 117,
        "root": artifact.ARCHIVE_ROOT,
        "verified": True,
    }
    assert artifact.main(["verify", str(first)]) == 0

    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert tuple(member.name for member in members) == expected_names
        assert all(member.isfile() and not member.issym() for member in members)
        assert all(member.mtime == 0 for member in members)
        assert all(member.uid == member.gid == 0 for member in members)
        assert all(member.mode == 0o644 for member in members)
        assert all(not PurePosixPath(member.name).is_absolute() for member in members)
        assert all(".git" not in PurePosixPath(member.name).parts for member in members)

        extracted = tmp_path / "extracted"
        archive.extractall(extracted, filter="data")

    unpacked_root = extracted / artifact.ARCHIVE_ROOT
    assert (
        unpacked_root / "tables" / "seed_sensitivity_main.tex"
    ).read_bytes() == payloads[artifact.PUBLIC_SEED_GATE_TABLE_SOURCE]
    readme = (unpacked_root / "README.md").read_text(encoding="utf-8")
    assert "all 30 manifest/record pairs" in readme
    assert "match the released\nreference byte for byte" in readme
    assert "five reversal cells" in readme
    for module in (
        "scripts.analyze.main",
        "scripts.render.paper",
        "scripts.maintenance.replay_seed",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=unpacked_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifact.build_anonymous_analysis_artifact(REPO_ROOT, first)

    assert len(validator_calls) >= 5
    monkeypatch.setattr(
        artifact,
        "load_public_seed_release",
        lambda *args: (_ for _ in ()).throw(ValueError("semantic seed drift")),
    )
    with pytest.raises(artifact.ArtifactValidationError, match="semantic seed drift"):
        artifact.verify_archive(first)


@pytest.mark.parametrize(
    ("payload", "label"),
    (
        (b"identity=zhan9381\n", "identity marker"),
        (b"source=https://example.invalid/data\n", "URL"),
        (b"partition=saffo-a100\n", "private queue marker"),
        (b"submit with sbatch now\n", "Slurm marker"),
        (b'path="/scratch.global/private/run"\n', "absolute filesystem path"),
        (b"token=" + b"olp" + b"_not-a-real-token\n", "credential marker"),
        (b"token=" + b"sk" + b"-proj-not-a-real-token\n", "credential marker"),
        (b'{"job_id":"12345"}\n', "raw job identifier"),
        (b'{"receipt_schema_version":1}\n', "submission receipt content"),
        (b"PK\x03\x04fake-npz", "NPZ/checkpoint payload"),
    ),
)
def test_anonymity_scanner_fails_closed_on_forbidden_markers(payload, label):
    with pytest.raises(artifact.ArtifactValidationError, match=label):
        artifact.scan_anonymous_bytes(payload, source="injected.txt")


def test_anonymity_scanner_does_not_confuse_dataset_identifiers_with_identity():
    artifact.scan_anonymous_bytes(
        b'{"sample_id":"Abyssinian_201","risk_dice":0.1}\n',
        source="records.jsonl",
    )


def test_source_reader_rejects_missing_file_and_symlink(tmp_path):
    with pytest.raises(FileNotFoundError, match="required release input is missing"):
        artifact._read_regular_source(tmp_path, "missing.txt")

    target = tmp_path / "target.txt"
    target.write_text("safe\n")
    link = tmp_path / "linked.txt"
    link.symlink_to(target.name)
    with pytest.raises(artifact.ArtifactValidationError, match="symlink"):
        artifact._read_regular_source(tmp_path, "linked.txt")


def test_member_allowlist_has_exact_canonical_cardinalities():
    roles = [item.role for item in artifact.RELEASE_FILES]
    assert roles.count("code") == 4
    assert roles.count("campaign-lock") == 1
    assert roles.count("analysis") == 1
    assert roles.count("csv") == 1
    assert roles.count("manifest") == 16
    assert roles.count("records") == 16
    assert roles.count("table") == 7
    assert roles.count("seed-analysis") == 1
    assert roles.count("seed-scheduler-summary") == 1
    assert roles.count("seed-provenance") == 1
    assert roles.count("seed-table") == 1
    assert roles.count("seed-gate-table") == 1
    assert roles.count("seed-replay-code") == 1
    assert roles.count("seed-replay-lock") == 1
    assert roles.count("seed-replay-guard") == 1
    assert roles.count("seed-record-manifest") == 30
    assert roles.count("seed-records") == 30

    sources = [item.source for item in artifact.RELEASE_FILES]
    destinations = [item.destination for item in artifact.RELEASE_FILES]
    assert len(sources) == len(set(sources)) == 114
    assert len(destinations) == len(set(destinations)) == 114
    seed_sources = {
        item.source for item in artifact.RELEASE_FILES if item.role.startswith("seed-")
    }
    assert {
        "results/seed_robustness_analysis.json",
        "results/seed_scheduler_summary.json",
        "results/seed_provenance.json",
        "docs/Tables/seed_robustness.tex",
        "docs/Tables/seed_sensitivity_main.tex",
        "results/seed_replay.lock.json",
        "results/seed_replay.complete.json",
        "scripts/maintenance/replay_seed.py",
    } <= seed_sources
    seed_destinations = {
        item.destination
        for item in artifact.RELEASE_FILES
        if item.role.startswith("seed-")
    }
    assert {
        "results/seed_robustness_analysis.json",
        "results/seed_scheduler_summary.json",
        "results/seed_provenance.json",
        "tables/seed_robustness.tex",
        "tables/seed_sensitivity_main.tex",
        "results/seed_replay.lock.json",
        "results/seed_replay.complete.json",
        "scripts/maintenance/replay_seed.py",
    } <= seed_destinations
    assert not any(
        any(marker in path.lower() for marker in ("receipt", "checkpoint"))
        or Path(path).suffix.lower() in artifact._FORBIDDEN_RELEASE_SUFFIXES
        for path in sources + destinations
    )
    assert set(artifact.GENERATED_MEMBER_NAMES) == {
        "README.md",
        "requirements-analysis.txt",
        "MANIFEST.sha256",
    }


@pytest.mark.parametrize(
    "missing_source",
    (
        artifact.PUBLIC_SEED_ANALYSIS_SOURCE,
        artifact.PUBLIC_SEED_SCHEDULER_SOURCE,
        artifact.PUBLIC_SEED_PROVENANCE_SOURCE,
        artifact.PUBLIC_SEED_TABLE_SOURCE,
        artifact.PUBLIC_SEED_GATE_TABLE_SOURCE,
        artifact.PUBLIC_SEED_REPLAY_LOCK_SOURCE,
        artifact.PUBLIC_SEED_REPLAY_GUARD_SOURCE,
    ),
)
def test_every_public_seed_member_is_mandatory(
    missing_source, monkeypatch, portable_seed_inputs
):
    original_reader = artifact._read_regular_source

    def missing_one(repo_root, relative):
        if relative == missing_source:
            raise FileNotFoundError(f"required release input is missing: {relative}")
        return original_reader(repo_root, relative)

    monkeypatch.setattr(artifact, "_read_regular_source", missing_one)
    with pytest.raises(FileNotFoundError, match="required release input is missing"):
        artifact._load_release_files(REPO_ROOT)


def test_seed_table_is_bound_to_public_provenance_and_source_comment(
    portable_seed_inputs,
):
    payloads, calls = portable_seed_inputs
    artifact._validate_public_seed_closure(dict(payloads))
    assert len(calls) == 1

    changed_table = dict(payloads)
    changed_table[artifact.PUBLIC_SEED_TABLE_SOURCE] += b"tampered\n"
    with pytest.raises(artifact.ArtifactValidationError, match="provenance binding"):
        artifact._validate_public_seed_closure(changed_table)

    wrong_comment = dict(payloads)
    table = wrong_comment[artifact.PUBLIC_SEED_TABLE_SOURCE].replace(
        b"a" * 64, b"b" * 64
    )
    wrong_comment[artifact.PUBLIC_SEED_TABLE_SOURCE] = table
    provenance = json.loads(wrong_comment[artifact.PUBLIC_SEED_PROVENANCE_SOURCE])
    provenance["analysis"]["table_sha256"] = hashlib.sha256(table).hexdigest()
    wrong_comment[artifact.PUBLIC_SEED_PROVENANCE_SOURCE] = _json_bytes(provenance)
    with pytest.raises(
        artifact.ArtifactValidationError, match="source-analysis SHA-256 comment"
    ):
        artifact._validate_public_seed_closure(wrong_comment)


def test_compact_gate_table_is_rebuilt_byte_exactly_from_public_analysis(
    portable_seed_inputs,
):
    payloads, _ = portable_seed_inputs
    artifact._validate_public_seed_closure(dict(payloads))

    changed = dict(payloads)
    changed[artifact.PUBLIC_SEED_GATE_TABLE_SOURCE] += b"tampered\n"
    with pytest.raises(
        artifact.ArtifactValidationError, match="byte-exact public-analysis rebuild"
    ):
        artifact._validate_public_seed_closure(changed)

    analysis = json.loads(payloads[artifact.PUBLIC_SEED_ANALYSIS_SOURCE])
    rebuilt = artifact._render_seed_gate_table(analysis)
    assert rebuilt == payloads[artifact.PUBLIC_SEED_GATE_TABLE_SOURCE]
    assert rebuilt.count(rb"\bigl(") == 5
    assert rebuilt.count(rb"^{\dagger}") == 3
    assert rebuilt.count(b"--") == 5
    assert b"Oxford Pet & Kvasir-SEG & FIVES & ISIC & TN3K" in rebuilt


def test_compact_gate_table_rejects_a_different_observed_gate():
    analysis = _public_gate_analysis("a" * 64)
    analysis["cells"][0]["summary"]["contrasts"][artifact.SEED_GATE_CONTRAST][
        "direction_reversal"
    ] = True
    with pytest.raises(
        artifact.ArtifactValidationError, match="fixed compact Gate-C result"
    ):
        artifact._render_seed_gate_table(analysis)


def test_seed_validation_resolves_a_symlinked_temporary_prefix(
    tmp_path, monkeypatch, portable_seed_inputs
):
    payloads, calls = portable_seed_inputs
    real_root = tmp_path / "real-temporary-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-temporary-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    class ExistingTemporaryDirectory:
        def __init__(self, **kwargs):
            assert kwargs["prefix"] == "selectseg-public-seed-"

        def __enter__(self):
            return linked_root

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        artifact.tempfile, "TemporaryDirectory", ExistingTemporaryDirectory
    )
    artifact._validate_public_seed_closure(dict(payloads))

    assert len(calls) == 1
    assert all(path.parent == real_root.resolve(strict=True) for path in calls[0])
