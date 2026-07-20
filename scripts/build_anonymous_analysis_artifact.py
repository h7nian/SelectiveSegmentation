"""Build and verify the deterministic anonymous core-analysis artifact.

The release is deliberately smaller than the full project.  It contains only
the code and immutable data needed to reproduce the locked selective-risk
analysis and its seven canonical table artifacts.  Every input and archive
destination is explicitly allowlisted below; no filesystem discovery is used.

The builder is fail closed: it validates the analysis/campaign/assembly hash
closure, rejects symlinked inputs and privacy markers, verifies the completed
archive before publishing it, and creates the destination atomically without
overwriting an existing file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


ARCHIVE_ROOT = "selective-segmentation-analysis-artifact"
ARCHIVE_MTIME = 0
ARCHIVE_FILE_MODE = 0o644
EXPECTED_CONDITION_COUNT = 16

# The builder consumes the byte-exact public-mirror layout.  This makes the
# same command usable in a clean public clone and prevents the anonymous
# release from depending on private workspace-only output paths.
ANALYSIS_SOURCE = "results/analysis.json"
CSV_SOURCE = "results/main_table.csv"
CAMPAIGN_LOCK_SOURCE = "results/campaign.lock.json"

CANONICAL_TABLE_NAMES = (
    "main_results.tex",
    "full_target_results.tex",
    "complete_results.tex",
    "cross_loss_results.tex",
    "quadrature_ablation.tex",
    "statistical_tests.tex",
    "results_complete.tex",
)

# This is the audited 16-condition closure.  The tuple is data, not a search
# pattern: changing a run requires an intentional source change and review.
ASSEMBLED_RUNS = (
    ("fives", "clipseg-general", "74db5fc8c0672236"),
    ("fives", "clipseg-target", "627922c22aa3aa21"),
    ("fives", "deeplabv3-target", "a62d754378787146"),
    ("isic", "clipseg-general", "500bf2dd2f3564b2"),
    ("isic", "clipseg-target", "69fc681ff737f148"),
    ("isic", "deeplabv3-target", "a42288ea78a340b1"),
    ("kvasir", "clipseg-general", "8f5786edfc8d5993"),
    ("kvasir", "clipseg-target", "1a1c02b968ab86a4"),
    ("kvasir", "deeplabv3-target", "5cca940975b7f02d"),
    ("pet", "clipseg-general", "88d2033138fc971a"),
    ("pet", "clipseg-target", "cd4341aaed2bda20"),
    ("pet", "deeplabv3-external", "014bfe3ad787f51e"),
    ("pet", "deeplabv3-target", "fd2f61c609c18fba"),
    ("tn3k", "clipseg-general", "4faed2465c790f8c"),
    ("tn3k", "clipseg-target", "eccb8f4f045a5473"),
    ("tn3k", "deeplabv3-target", "01b4c3a58986cf27"),
)


@dataclass(frozen=True)
class ReleaseFile:
    """One immutable source-to-archive allowlist entry."""

    source: str
    destination: str
    role: str


_CODE_FILES = (
    ReleaseFile("scripts/analyze_binary.py", "scripts/analyze_binary.py", "code"),
    ReleaseFile(
        "scripts/render_paper_tables.py",
        "scripts/render_paper_tables.py",
        "code",
    ),
    ReleaseFile(
        "selectseg/binary_framework.py",
        "selectseg/binary_framework.py",
        "code",
    ),
    ReleaseFile("selectseg/__init__.py", "selectseg/__init__.py", "code"),
)

_LOCKED_RESULT_FILES = (
    ReleaseFile(CAMPAIGN_LOCK_SOURCE, "results/campaign.lock.json", "campaign-lock"),
    ReleaseFile(ANALYSIS_SOURCE, "results/analysis.json", "analysis"),
    ReleaseFile(CSV_SOURCE, "results/main_table.csv", "csv"),
)

_ASSEMBLED_FILES = tuple(
    ReleaseFile(
        f"results/assembled/{dataset}/{condition}/{run_id}/{name}",
        f"results/assembled/{dataset}/{condition}/{run_id}/{name}",
        role,
    )
    for dataset, condition, run_id in ASSEMBLED_RUNS
    for name, role in (("manifest.json", "manifest"), ("records.jsonl", "records"))
)

_TABLE_FILES = tuple(
    ReleaseFile(f"docs/Tables/{name}", f"tables/{name}", "table")
    for name in CANONICAL_TABLE_NAMES
)

# The only repository files eligible for inclusion.  Generated metadata below
# is separately fixed by GENERATED_MEMBER_NAMES.
RELEASE_FILES = _CODE_FILES + _LOCKED_RESULT_FILES + _ASSEMBLED_FILES + _TABLE_FILES
GENERATED_MEMBER_NAMES = (
    "MANIFEST.sha256",
    "README.md",
    "requirements-analysis.txt",
)

REQUIREMENTS = "numpy==2.5.1\nscipy==1.18.0\n"

ANONYMOUS_README = """# Core analysis reproduction artifact

This anonymous artifact contains the fixed 16-condition records, their
manifests, the immutable campaign lock, the analysis and table source code,
and the seven canonical table artifacts.  It does not contain model weights,
raw images, training data, scheduler metadata, or author identities.

Use Python 3.12.4 and install the pinned analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
```

From this directory, rerun the locked analysis with all records listed
explicitly (the analyzer rejects an incomplete final campaign):

```bash
mapfile -t inputs < <(find results/assembled -type f -name records.jsonl -print | LC_ALL=C sort)
test "${#inputs[@]}" -eq 16
python -m scripts.analyze_binary \
  --inputs "${inputs[@]}" \
  --campaign-lock results/campaign.lock.json \
  --bootstrap-samples 10000 \
  --bootstrap-workers 4 \
  --output-dir rebuild/analysis
cmp results/analysis.json rebuild/analysis/analysis.json
cmp results/main_table.csv rebuild/analysis/main_table.csv
python -m scripts.render_paper_tables \
  --analysis rebuild/analysis/analysis.json \
  --output-dir rebuild/tables
for name in main_results full_target_results complete_results cross_loss_results \
  quadrature_ablation statistical_tests results_complete; do
  cmp "tables/${name}.tex" "rebuild/tables/${name}.tex"
done
```

`MANIFEST.sha256` authenticates every other archive member.  Verification can
also be performed before extraction with the artifact builder's `verify`
subcommand.
"""


class ArtifactValidationError(RuntimeError):
    """Raised when a release input or archive violates the fixed contract."""


_FORBIDDEN_PATTERNS = (
    (
        "URL",
        re.compile(rb"(?i)(?:\b(?:https?|ftp|ssh|file)://|\bgit@|\bwww\.)"),
    ),
    (
        "email address",
        re.compile(rb"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    ),
    (
        "Slurm marker",
        re.compile(rb"(?i)\b(?:slurm|sbatch|squeue|sacct|srun)\b"),
    ),
    (
        "private queue marker",
        re.compile(rb"(?i)\b(?:saffo(?:-a100)?|apollo_agate|apollo|agate)\b"),
    ),
    (
        "absolute filesystem path",
        re.compile(
            rb"(?i)(?:^|[\s\"'=:(])/(?![/\s])"
            rb"(?:[A-Z0-9._-]+/)*[A-Z0-9._-]+"
        ),
    ),
    ("Windows absolute path", re.compile(rb"(?i)\b[A-Z]:[\\/]")),
    (
        "identity marker",
        re.compile(rb"(?i)\b(?:zhan9381|sinianzhang|sinian[ _.-]+zhang)\b"),
    ),
    (
        "credential marker",
        re.compile(
            rb"(?i)(?:github"
            rb"_pat_[A-Za-z0-9_]+|ghp"
            rb"_[A-Za-z0-9]+|"
            rb"\b(?:api[_-]?key|access[_-]?token|password|passwd|bearer)\b\s*[:=])"
        ),
    ),
)

_TABLE_ANALYSIS_HASH = re.compile(rb"Source analysis\.json SHA-256: ([0-9a-f]{64})")
_MANIFEST_LINE = re.compile(rb"([0-9a-f]{64})  ([^\r\n]+)\n")
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactValidationError(f"{label} is not a safe POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactValidationError(
            f"{label} is not relative and normalized: {value!r}"
        )
    if ".git" in path.parts:
        raise ArtifactValidationError(f"{label} contains forbidden .git component")
    return path


def scan_anonymous_bytes(data: bytes, *, source: str) -> None:
    """Reject identity, infrastructure, URL, path, and credential markers."""

    if b"\x00" in data:
        raise ArtifactValidationError(f"{source} contains a NUL byte")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactValidationError(f"{source} is not UTF-8 text") from error
    for label, pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(data):
            raise ArtifactValidationError(f"{source} contains forbidden {label}")


def _validate_allowlist_contract() -> None:
    if len(ASSEMBLED_RUNS) != EXPECTED_CONDITION_COUNT:
        raise ArtifactValidationError("allowlist must declare exactly 16 runs")
    condition_keys = [(dataset, condition) for dataset, condition, _ in ASSEMBLED_RUNS]
    if len(condition_keys) != len(set(condition_keys)):
        raise ArtifactValidationError("allowlist contains duplicate conditions")
    if len({run_id for _, _, run_id in ASSEMBLED_RUNS}) != EXPECTED_CONDITION_COUNT:
        raise ArtifactValidationError("allowlist contains duplicate assembly run IDs")

    sources = [item.source for item in RELEASE_FILES]
    destinations = [item.destination for item in RELEASE_FILES]
    if len(sources) != len(set(sources)):
        raise ArtifactValidationError("release source allowlist contains duplicates")
    if len(destinations) != len(set(destinations)):
        raise ArtifactValidationError(
            "release destination allowlist contains duplicates"
        )
    for item in RELEASE_FILES:
        _safe_relative_path(item.source, label="release source")
        _safe_relative_path(item.destination, label="release destination")

    for role, count in (("records", 16), ("manifest", 16), ("table", 7)):
        observed = sum(item.role == role for item in RELEASE_FILES)
        if observed != count:
            raise ArtifactValidationError(
                f"allowlist must contain exactly {count} {role} files; got {observed}"
            )
    table_destinations = tuple(
        PurePosixPath(item.destination).name
        for item in RELEASE_FILES
        if item.role == "table"
    )
    if table_destinations != CANONICAL_TABLE_NAMES:
        raise ArtifactValidationError("canonical table allowlist is incomplete")


def _read_regular_source(repo_root: Path, relative: str) -> bytes:
    """Read one allowlisted source while rejecting symlinks in its path."""

    relative_path = _safe_relative_path(relative, label="release source")
    root = Path(repo_root)
    try:
        root_status = root.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"repository root does not exist: {root}") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise ArtifactValidationError(
            f"repository root is not a real directory: {root}"
        )

    current = root
    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"required release input is missing: {relative}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise ArtifactValidationError(
                f"release input traverses a symlink: {relative}"
            )
        is_last = index == len(relative_path.parts) - 1
        if is_last:
            if not stat.S_ISREG(status.st_mode):
                raise ArtifactValidationError(
                    f"release input is not a regular file: {relative}"
                )
        elif not stat.S_ISDIR(status.st_mode):
            raise ArtifactValidationError(
                f"release input parent is not a directory: {relative}"
            )
    return current.read_bytes()


def _strict_json(data: bytes, *, source: str):
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid strict JSON in {source}: {error}"
        ) from error


def _analysis_source_sha256(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path in ("scripts/analyze_binary.py", "selectseg/binary_framework.py"):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[path])
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_canonical_closure(files: dict[str, bytes]) -> None:
    """Cross-check every selected run against the locked canonical analysis."""

    lock = _strict_json(files[CAMPAIGN_LOCK_SOURCE], source=CAMPAIGN_LOCK_SOURCE)
    analysis = _strict_json(files[ANALYSIS_SOURCE], source=ANALYSIS_SOURCE)
    if not isinstance(lock, dict) or not isinstance(analysis, dict):
        raise ArtifactValidationError("campaign lock and analysis must be JSON objects")

    expected_keys = {(dataset, condition) for dataset, condition, _ in ASSEMBLED_RUNS}
    lock_artifacts = lock.get("artifacts")
    if not isinstance(lock_artifacts, list) or len(lock_artifacts) != 16:
        raise ArtifactValidationError("campaign lock must contain exactly 16 artifacts")
    lock_keys = {
        (item.get("dataset"), item.get("condition"))
        for item in lock_artifacts
        if isinstance(item, dict)
    }
    if lock_keys != expected_keys:
        raise ArtifactValidationError(
            "campaign lock conditions differ from the allowlist"
        )

    provenance = analysis.get("provenance")
    if not isinstance(provenance, dict):
        raise ArtifactValidationError("analysis is missing provenance")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 16:
        raise ArtifactValidationError("analysis provenance must bind exactly 16 inputs")
    input_by_key = {}
    for item in inputs:
        if not isinstance(item, dict):
            raise ArtifactValidationError("analysis provenance input is not an object")
        key = (item.get("dataset"), item.get("condition"))
        if key in input_by_key:
            raise ArtifactValidationError(
                "analysis provenance contains duplicate inputs"
            )
        input_by_key[key] = item
    if set(input_by_key) != expected_keys:
        raise ArtifactValidationError("analysis conditions differ from the allowlist")

    lock_sha = _sha256_bytes(files[CAMPAIGN_LOCK_SOURCE])
    lock_binding = provenance.get("campaign_lock")
    if not isinstance(lock_binding, dict) or lock_binding.get("sha256") != lock_sha:
        raise ArtifactValidationError(
            "analysis is not bound to the selected campaign lock"
        )
    if provenance.get("campaign_id") != lock.get("campaign_id"):
        raise ArtifactValidationError("analysis and campaign IDs differ")

    expected_source_sha = _analysis_source_sha256(files)
    if provenance.get("analysis_source_sha256") != expected_source_sha:
        raise ArtifactValidationError(
            "analysis is not bound to the selected analysis code"
        )

    for dataset, condition, run_id in ASSEMBLED_RUNS:
        base = f"results/assembled/{dataset}/{condition}/{run_id}"
        manifest_path = f"{base}/manifest.json"
        records_path = f"{base}/records.jsonl"
        manifest_bytes = files[manifest_path]
        records_bytes = files[records_path]
        manifest = _strict_json(manifest_bytes, source=manifest_path)
        if not isinstance(manifest, dict):
            raise ArtifactValidationError(f"{manifest_path} must contain one object")
        identity = (
            manifest.get("dataset"),
            manifest.get("condition"),
            manifest.get("run_id"),
        )
        if identity != (dataset, condition, run_id):
            raise ArtifactValidationError(
                f"assembly identity mismatch in {manifest_path}"
            )
        records_sha = _sha256_bytes(records_bytes)
        manifest_sha = _sha256_bytes(manifest_bytes)
        if manifest.get("jsonl_sha256") != records_sha:
            raise ArtifactValidationError(f"records hash mismatch in {manifest_path}")
        assembly = manifest.get("assembly")
        if not isinstance(assembly, dict):
            raise ArtifactValidationError(
                f"assembly binding missing in {manifest_path}"
            )
        if assembly.get("campaign_lock_sha256") != lock_sha:
            raise ArtifactValidationError(f"campaign hash mismatch in {manifest_path}")
        if assembly.get("campaign_id") != lock.get("campaign_id"):
            raise ArtifactValidationError(f"campaign ID mismatch in {manifest_path}")

        bound = input_by_key[(dataset, condition)]
        expected_logical_id = f"{dataset}/{condition}/{run_id}"
        expected_values = {
            "logical_id": expected_logical_id,
            "assembly_run_id": run_id,
            "manifest_sha256": manifest_sha,
            "records_sha256": records_sha,
            "num_samples": manifest.get("num_images"),
        }
        for field, expected in expected_values.items():
            if bound.get(field) != expected:
                raise ArtifactValidationError(
                    f"analysis {field} binding differs for {dataset}/{condition}"
                )

    analysis_conditions = analysis.get("conditions")
    if not isinstance(analysis_conditions, list) or len(analysis_conditions) != 16:
        raise ArtifactValidationError("analysis must report exactly 16 conditions")
    reported_keys = {
        (item.get("dataset"), item.get("condition"))
        for item in analysis_conditions
        if isinstance(item, dict)
    }
    if reported_keys != expected_keys:
        raise ArtifactValidationError("reported analysis conditions are incomplete")

    analysis_sha = _sha256_bytes(files[ANALYSIS_SOURCE])
    for table_name in CANONICAL_TABLE_NAMES:
        source = f"docs/Tables/{table_name}"
        matches = _TABLE_ANALYSIS_HASH.findall(files[source])
        if not matches or any(
            match != analysis_sha.encode("ascii") for match in matches
        ):
            raise ArtifactValidationError(
                f"canonical table {table_name} is not bound only to analysis.json"
            )


def _load_release_files(repo_root: Path) -> dict[str, bytes]:
    _validate_allowlist_contract()
    files = {}
    for item in RELEASE_FILES:
        data = _read_regular_source(repo_root, item.source)
        scan_anonymous_bytes(data, source=item.source)
        files[item.source] = data
    _validate_canonical_closure(files)
    return files


def _manifest_bytes(members: dict[str, bytes]) -> bytes:
    lines = [
        f"{_sha256_bytes(members[name])}  {name}\n"
        for name in sorted(members)
        if name != "MANIFEST.sha256"
    ]
    return "".join(lines).encode("utf-8")


def _archive_payload(repo_root: Path) -> dict[str, bytes]:
    source_bytes = _load_release_files(repo_root)
    members = {item.destination: source_bytes[item.source] for item in RELEASE_FILES}
    members["README.md"] = ANONYMOUS_README.encode("utf-8")
    members["requirements-analysis.txt"] = REQUIREMENTS.encode("utf-8")
    for name in ("README.md", "requirements-analysis.txt"):
        scan_anonymous_bytes(members[name], source=name)
    members["MANIFEST.sha256"] = _manifest_bytes(members)
    return members


def expected_archive_member_names() -> tuple[str, ...]:
    """Return the complete, ordered archive member allowlist."""

    relative = [item.destination for item in RELEASE_FILES]
    relative.extend(GENERATED_MEMBER_NAMES)
    return tuple(sorted(f"{ARCHIVE_ROOT}/{name}" for name in relative))


def _write_deterministic_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=9,
            mtime=ARCHIVE_MTIME,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for relative_name in sorted(members):
                    member_name = f"{ARCHIVE_ROOT}/{relative_name}"
                    info = tarfile.TarInfo(member_name)
                    info.size = len(members[relative_name])
                    info.mtime = ARCHIVE_MTIME
                    info.mode = ARCHIVE_FILE_MODE
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = tarfile.REGTYPE
                    archive.addfile(info, io.BytesIO(members[relative_name]))
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    path.chmod(0o644)


def _deterministic_tar_bytes(members: dict[str, bytes]) -> bytes:
    """Materialize the canonical uncompressed tar stream for verification."""

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative_name in sorted(members):
            member_name = f"{ARCHIVE_ROOT}/{relative_name}"
            info = tarfile.TarInfo(member_name)
            info.size = len(members[relative_name])
            info.mtime = ARCHIVE_MTIME
            info.mode = ARCHIVE_FILE_MODE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(members[relative_name]))
    return output.getvalue()


def _parse_checksum_manifest(data: bytes) -> dict[str, str]:
    position = 0
    result = {}
    while position < len(data):
        match = _MANIFEST_LINE.match(data, position)
        if match is None:
            raise ArtifactValidationError("MANIFEST.sha256 has invalid syntax")
        digest = match.group(1).decode("ascii")
        name = match.group(2).decode("utf-8")
        _safe_relative_path(name, label="checksum member")
        if name in result:
            raise ArtifactValidationError("MANIFEST.sha256 contains duplicate members")
        result[name] = digest
        position = match.end()
    return result


def verify_archive(path: str | os.PathLike[str]) -> dict[str, object]:
    """Verify the exact allowlist, metadata, privacy scan, and member hashes."""

    archive_path = Path(path)
    try:
        status = archive_path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"archive does not exist: {archive_path}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ArtifactValidationError("archive must be a non-symlink regular file")
    raw = archive_path.read_bytes()
    if len(raw) < len(_GZIP_HEADER) or raw[: len(_GZIP_HEADER)] != _GZIP_HEADER:
        raise ArtifactValidationError("archive has a non-deterministic gzip header")
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        uncompressed = decompressor.decompress(raw) + decompressor.flush()
    except zlib.error as error:
        raise ArtifactValidationError(f"invalid gzip stream: {error}") from error
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise ArtifactValidationError("archive must contain exactly one gzip member")

    expected_names = expected_archive_member_names()
    extracted = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = tuple(member.name for member in members)
            if names != expected_names:
                raise ArtifactValidationError(
                    "archive member list/order differs from the explicit allowlist"
                )
            for member in members:
                safe_name = _safe_relative_path(member.name, label="archive member")
                if safe_name.parts[0] != ARCHIVE_ROOT:
                    raise ArtifactValidationError(
                        "archive member is outside the fixed root"
                    )
                if not member.isfile() or member.islnk() or member.issym():
                    raise ArtifactValidationError(
                        f"archive contains a non-regular member: {member.name}"
                    )
                if (
                    member.mtime != ARCHIVE_MTIME
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mode != ARCHIVE_FILE_MODE
                    or member.pax_headers
                ):
                    raise ArtifactValidationError(
                        f"archive metadata is non-deterministic: {member.name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ArtifactValidationError(
                        f"cannot read archive member {member.name}"
                    )
                data = handle.read()
                if len(data) != member.size:
                    raise ArtifactValidationError(
                        f"truncated archive member {member.name}"
                    )
                relative_name = (
                    PurePosixPath(member.name).relative_to(ARCHIVE_ROOT).as_posix()
                )
                scan_anonymous_bytes(data, source=relative_name)
                extracted[relative_name] = data
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ArtifactValidationError(f"invalid tar.gz archive: {error}") from error

    manifest = _parse_checksum_manifest(extracted["MANIFEST.sha256"])
    expected_manifest_names = set(extracted) - {"MANIFEST.sha256"}
    if set(manifest) != expected_manifest_names:
        raise ArtifactValidationError(
            "MANIFEST.sha256 does not cover every other member"
        )
    for name in sorted(expected_manifest_names):
        if manifest[name] != _sha256_bytes(extracted[name]):
            raise ArtifactValidationError(
                f"checksum mismatch for archive member {name}"
            )
    if uncompressed != _deterministic_tar_bytes(extracted):
        raise ArtifactValidationError("archive tar bytes are not canonical")

    return {
        "archive_sha256": _sha256_bytes(raw),
        "member_count": len(extracted),
        "root": ARCHIVE_ROOT,
        "verified": True,
    }


def build_anonymous_analysis_artifact(
    repo_root: str | os.PathLike[str], output: str | os.PathLike[str]
) -> Path:
    """Build, self-verify, and atomically publish the fixed artifact."""

    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _archive_payload(Path(repo_root))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_tar_gz(temporary, payload)
        verify_archive(temporary)
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output_path}"
            ) from error
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


# Short aliases are convenient for callers while the descriptive functions
# remain the documented public API.
build_artifact = build_anonymous_analysis_artifact
verify_anonymous_analysis_artifact = verify_archive


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build the fixed artifact")
    build_parser.add_argument(
        "--repo-root",
        required=True,
        help="repository root containing the explicit canonical inputs",
    )
    build_parser.add_argument("--output", required=True, help="new .tar.gz path")

    verify_parser = subparsers.add_parser("verify", help="verify an existing artifact")
    verify_parser.add_argument("archive", help="artifact .tar.gz path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        path = build_anonymous_analysis_artifact(args.repo_root, args.output)
        report = verify_archive(path)
        print(json.dumps({"path": str(path), **report}, sort_keys=True))
    else:
        report = verify_archive(args.archive)
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
