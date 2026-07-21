"""Strictly publish the verified training-seed table into the manuscript.

This is an intentionally local, write-once publication step.  It verifies the
analysis and rendered-table hashes, regenerates the expected TeX with the
current renderer, and requires byte-for-byte equality before it creates the
fixed manuscript destination.  An identical existing destination is accepted
idempotently; a symlink, non-regular file, or different existing payload is
never replaced.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from scripts.render.seed import load_analysis, render_table
from selectseg.seed.extension import _sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_DESTINATION = PROJECT_ROOT / "docs" / "Tables" / "seed_robustness.tex"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--expected-analysis-sha256", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--expected-table-sha256", required=True)
    return parser.parse_args(argv)


def _strict_digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _regular_file(path, expected_sha256, *, name):
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(
            f"{name} must be a regular non-symlink file: {source}"
        )
    expected = _strict_digest(
        expected_sha256, location=f"expected {name} SHA-256"
    )
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return source, observed


def _validate_existing_destination(destination, expected_bytes):
    if destination.is_symlink():
        raise FileExistsError(
            f"refusing symlink manuscript destination: {destination}"
        )
    if not destination.is_file():
        raise FileExistsError(
            f"refusing non-regular manuscript destination: {destination}"
        )
    if destination.read_bytes() != expected_bytes:
        raise FileExistsError(
            f"refusing to overwrite different manuscript table: {destination}"
        )
    return "unchanged"


def _publish_bytes(destination, payload):
    destination = Path(destination)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FileNotFoundError(
            f"manuscript table parent must be a real directory: {parent}"
        )
    if destination.exists() or destination.is_symlink():
        return _validate_existing_destination(destination, payload)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return _validate_existing_destination(destination, payload)
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return "published"


def publish_seed_table(
    analysis_path,
    *,
    expected_analysis_sha256,
    table_path,
    expected_table_sha256,
    destination=PUBLISH_DESTINATION,
):
    """Validate both inputs, reconstruct TeX, and publish exact bytes once."""

    analysis_digest = _strict_digest(
        expected_analysis_sha256, location="expected analysis SHA-256"
    )
    analysis, by_key, observed_analysis_digest = load_analysis(
        analysis_path, expected_sha256=analysis_digest
    )
    table, observed_table_digest = _regular_file(
        table_path, expected_table_sha256, name="rendered seed table"
    )
    expected_text = render_table(
        analysis, by_key, analysis_sha256=observed_analysis_digest
    )
    expected_bytes = expected_text.encode("utf-8")
    if table.read_bytes() != expected_bytes:
        raise ValueError(
            "rendered seed table content differs from the current renderer"
        )
    status = _publish_bytes(Path(destination), expected_bytes)
    return {
        "analysis_sha256": observed_analysis_digest,
        "table_sha256": observed_table_digest,
        "destination": Path(destination),
        "status": status,
    }


def main(argv=None):
    args = parse_args(argv)
    result = publish_seed_table(
        args.analysis,
        expected_analysis_sha256=args.expected_analysis_sha256,
        table_path=args.table,
        expected_table_sha256=args.expected_table_sha256,
    )
    print(f"analysis_sha256={result['analysis_sha256']}")
    print(f"table_sha256={result['table_sha256']}")
    print(f"publication_status={result['status']}")
    print(f"published_table={result['destination']}")
    return result["destination"]


if __name__ == "__main__":
    main()
