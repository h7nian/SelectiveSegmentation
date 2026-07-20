"""Focused tests for the deterministic anonymous analysis artifact."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from scripts import build_anonymous_analysis_artifact as artifact


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = (
    WORKSPACE_ROOT / "github"
    if (WORKSPACE_ROOT / "github" / "results" / "analysis.json").is_file()
    else WORKSPACE_ROOT
)


def test_repeated_builds_are_identical_verified_and_importable(tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    artifact.build_anonymous_analysis_artifact(REPO_ROOT, first)
    artifact.build_anonymous_analysis_artifact(REPO_ROOT, second)

    assert first.read_bytes() == second.read_bytes()
    expected_names = artifact.expected_archive_member_names()
    assert len(artifact.RELEASE_FILES) == 46
    assert len(expected_names) == 49

    report = artifact.verify_archive(first)
    assert report == {
        "archive_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "member_count": 49,
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
    for module in ("scripts.analyze_binary", "scripts.render_paper_tables"):
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


@pytest.mark.parametrize(
    ("payload", "label"),
    (
        (b"identity=zhan9381\n", "identity marker"),
        (b"source=https://example.invalid/data\n", "URL"),
        (b"partition=saffo-a100\n", "private queue marker"),
        (b"submit with sbatch now\n", "Slurm marker"),
        (b'path="/scratch.global/private/run"\n', "absolute filesystem path"),
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

    sources = [item.source for item in artifact.RELEASE_FILES]
    destinations = [item.destination for item in artifact.RELEASE_FILES]
    assert len(sources) == len(set(sources)) == 46
    assert len(destinations) == len(set(destinations)) == 46
    assert set(artifact.GENERATED_MEMBER_NAMES) == {
        "README.md",
        "requirements-analysis.txt",
        "MANIFEST.sha256",
    }
