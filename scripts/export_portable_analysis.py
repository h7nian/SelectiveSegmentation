"""Export one strict analysis JSON with repository-relative provenance paths.

The statistical payload is not recomputed.  Every absolute path must resolve
inside the declared repository root and is rewritten to a POSIX path relative
to that root.  The exporter rejects non-standard JSON, non-finite values,
credential markers, symlinks, hash mismatches, and overwrite attempts.  A
deterministic sidecar binds the private source bytes to the portable bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import string
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCHEMA_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="strict source analysis JSON")
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", required=True, help="write-once portable JSON")
    parser.add_argument("--manifest", required=True, help="write-once export sidecar")
    parser.add_argument(
        "--repository-root",
        default=str(REPO_ROOT),
        help="root against which absolute provenance paths are made portable",
    )
    return parser.parse_args(argv)


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_digest(value: str, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{location} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_tree(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, location=f"{location}[{index}]")


def _secret_marker(payload: bytes) -> bool:
    markers = (
        b"olp" + b"_",
        b"ghp" + b"_",
        b"github_pat" + b"_",
        b"sk" + b"-proj-",
        b"-----BEGIN " + b"PRIVATE KEY-----",
    )
    return any(marker in payload for marker in markers)


def _strict_json(raw: bytes, *, source: Path) -> Any:
    if _secret_marker(raw):
        raise ValueError(f"possible credential marker in {source}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    _finite_tree(value, location=str(source))
    return value


def _regular_file(path: str | os.PathLike[str], *, name: str) -> Path:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular non-symlink file: {source}")
    return source.resolve()


def _inside(path: Path, root: Path, *, name: str) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{name} escapes repository root {root}: {path}") from error


def _portable_tree(value: Any, *, root: Path) -> tuple[Any, int]:
    if isinstance(value, dict):
        result = {}
        count = 0
        for key, child in value.items():
            converted, child_count = _portable_tree(child, root=root)
            result[key] = converted
            count += child_count
        return result, count
    if isinstance(value, list):
        result = []
        count = 0
        for child in value:
            converted, child_count = _portable_tree(child, root=root)
            result.append(converted)
            count += child_count
        return result, count
    if isinstance(value, str):
        if PureWindowsPath(value).is_absolute():
            raise ValueError(f"Windows absolute path is not portable: {value}")
        candidate = Path(value)
        if candidate.is_absolute():
            return _inside(candidate, root, name="analysis path"), 1
    return value, 0


def _json_bytes(value: Any) -> bytes:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if _secret_marker(payload):
        raise ValueError("portable payload contains a possible credential marker")
    return payload


def _validate_destination(path: Path, *, root: Path, name: str) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    resolved = path.resolve(strict=False)
    _inside(resolved, root, name=name)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {path}")
    current = lexical.parent
    while current != root:
        if current == current.parent:
            raise ValueError(f"{name} lexical parent escapes repository root")
        if current.is_symlink():
            raise ValueError(f"{name} parent is a symlink: {current}")
        if not current.is_relative_to(root):
            raise ValueError(f"{name} parent escapes repository root: {current}")
        current = current.parent
    return resolved


def _publish_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {path}") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def export_analysis(
    source,
    *,
    expected_source_sha256,
    output,
    manifest,
    repository_root=REPO_ROOT,
):
    root = _regular_file(Path(repository_root) / "README.md", name="root marker").parent
    source_path = _regular_file(source, name="source analysis")
    source_relative = _inside(source_path, root, name="source analysis")
    raw = source_path.read_bytes()
    source_sha256 = _digest_bytes(raw)
    if source_sha256 != _strict_digest(
        expected_source_sha256, location="expected input SHA-256"
    ):
        raise ValueError("source analysis SHA-256 mismatch")
    analysis = _strict_json(raw, source=source_path)
    portable, rewrite_count = _portable_tree(analysis, root=root)
    portable_payload = _json_bytes(portable)
    portable_sha256 = _digest_bytes(portable_payload)

    output_path = _validate_destination(
        Path(output), root=root, name="portable analysis"
    )
    manifest_path = _validate_destination(
        Path(manifest), root=root, name="portable export manifest"
    )
    if output_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ValueError("portable analysis and export manifest must differ")
    export_manifest = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "transformation": "repository_absolute_paths_to_relative_posix_v1",
        "rewritten_absolute_path_count": rewrite_count,
        "source": {"path": source_relative, "sha256": source_sha256},
        "portable": {
            "path": _inside(output_path, root, name="portable analysis"),
            "sha256": portable_sha256,
        },
    }
    manifest_payload = _json_bytes(export_manifest)

    # Validate both payloads fully before publishing either one.
    _strict_json(portable_payload, source=output_path)
    _strict_json(manifest_payload, source=manifest_path)
    _publish_new(output_path, portable_payload)
    try:
        _publish_new(manifest_path, manifest_payload)
    except BaseException:
        # The analysis is immutable and valid even if the sidecar publication
        # is interrupted; never delete or overwrite it behind the user's back.
        raise
    return output_path, manifest_path, export_manifest


def main(argv=None):
    args = parse_args(argv)
    output, manifest, record = export_analysis(
        args.input,
        expected_source_sha256=args.expected_input_sha256,
        output=args.output,
        manifest=args.manifest,
        repository_root=args.repository_root,
    )
    print(f"saved {output}")
    print(f"portable_sha256={record['portable']['sha256']}")
    print(f"saved {manifest}")
    return output, manifest


if __name__ == "__main__":
    main()
