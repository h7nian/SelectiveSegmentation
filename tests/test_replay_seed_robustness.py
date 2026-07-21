"""Focused tests for the path-free 30-condition seed replay."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.maintenance.build_release import scan_anonymous_bytes
from scripts.maintenance.replay_seed import (
    ReplayValidationError,
    load_replay_lock,
    replay_release,
    verify_replay_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "outputs" / "public_seed"
LOCK_PATH = PUBLIC_ROOT / "seed_replay.lock.json"
GUARD_PATH = PUBLIC_ROOT / "seed_replay.complete.json"


def _source_for_public(relative: str) -> Path:
    aliases = {
        "results/seed_robustness_analysis.json": (
            PUBLIC_ROOT / "seed_robustness_analysis.json"
        ),
        "tables/seed_robustness.tex": ROOT / "docs" / "Tables" / "seed_robustness.tex",
        "tables/seed_sensitivity_main.tex": (
            ROOT / "docs" / "Tables" / "seed_sensitivity_main.tex"
        ),
    }
    if relative in aliases:
        return aliases[relative]
    prefix = "results/seed_records/"
    assert relative.startswith(prefix)
    return PUBLIC_ROOT / "seed_records" / relative.removeprefix(prefix)


def test_actual_portable_bundle_rebuilds_all_three_outputs_byte_exactly():
    rebuilt, report = verify_replay_payloads(
        LOCK_PATH.read_bytes(), lambda relative: _source_for_public(relative).read_bytes()
    )
    assert report["verified"] is True
    assert report["condition_count"] == 30
    assert report["seed_count"] == 3
    assert rebuilt["analysis"] == (PUBLIC_ROOT / "seed_robustness_analysis.json").read_bytes()
    assert rebuilt["robustness_table"] == (
        ROOT / "docs" / "Tables" / "seed_robustness.tex"
    ).read_bytes()
    assert rebuilt["gate_table"] == (
        ROOT / "docs" / "Tables" / "seed_sensitivity_main.tex"
    ).read_bytes()


def test_clean_layout_one_call_writes_byte_exact_rebuild(tmp_path):
    lock = load_replay_lock(LOCK_PATH.read_bytes())
    required = {"results/seed_replay.lock.json": LOCK_PATH}
    for row in lock["conditions"]:
        required[row["manifest_path"]] = _source_for_public(row["manifest_path"])
        required[row["records_path"]] = _source_for_public(row["records_path"])
    for binding in lock["expected_outputs"].values():
        required[binding["path"]] = _source_for_public(binding["path"])
    for relative, source in required.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    report = replay_release(tmp_path)
    rebuilt = tmp_path / "rebuild" / "seed_replay"
    assert report["condition_count"] == 30
    assert (rebuilt / "seed_robustness_analysis.json").read_bytes() == required[
        "results/seed_robustness_analysis.json"
    ].read_bytes()
    assert (rebuilt / "seed_robustness.tex").read_bytes() == required[
        "tables/seed_robustness.tex"
    ].read_bytes()
    assert (rebuilt / "seed_sensitivity_main.tex").read_bytes() == required[
        "tables/seed_sensitivity_main.tex"
    ].read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay_release(tmp_path)


def test_portable_bundle_is_complete_and_passes_the_anonymous_scanner():
    lock_payload = LOCK_PATH.read_bytes()
    scan_anonymous_bytes(lock_payload, source="seed_replay.lock.json")
    scan_anonymous_bytes(
        GUARD_PATH.read_bytes(), source="seed_replay.complete.json"
    )
    lock = load_replay_lock(lock_payload)
    assert len(lock["conditions"]) == 30
    assert len({row["logical_id"] for row in lock["conditions"]}) == 30
    forbidden_manifest_keys = {
        "path",
        "command",
        "account",
        "partition",
        "node",
        "device",
        "checkpoint",
    }
    for row in lock["conditions"]:
        manifest_path = _source_for_public(row["manifest_path"])
        records_path = _source_for_public(row["records_path"])
        manifest_payload = manifest_path.read_bytes()
        records_payload = records_path.read_bytes()
        scan_anonymous_bytes(manifest_payload, source=row["manifest_path"])
        scan_anonymous_bytes(records_payload, source=row["records_path"])
        manifest = json.loads(manifest_payload)
        assert not forbidden_manifest_keys.intersection(manifest)
        assert hashlib.sha256(manifest_payload).hexdigest() == row["manifest_sha256"]
        assert hashlib.sha256(records_payload).hexdigest() == row["records_sha256"]


def test_one_changed_record_byte_is_rejected_before_analysis():
    lock = load_replay_lock(LOCK_PATH.read_bytes())
    target = lock["conditions"][0]["records_path"]

    def changed_reader(relative: str) -> bytes:
        payload = _source_for_public(relative).read_bytes()
        if relative == target:
            return payload + b"\n"
        return payload

    with pytest.raises(ReplayValidationError, match="records SHA-256 mismatch"):
        verify_replay_payloads(LOCK_PATH.read_bytes(), changed_reader)
