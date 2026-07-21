"""Build and verify the deterministic anonymous analysis artifact v4.

The release is deliberately smaller than the full project.  It contains only
the code and immutable data needed to reproduce the locked selective-risk
analysis and its seven canonical table artifacts, plus the portable seed
robustness verification bundle and its compact Gate-C table.  Every input and
archive destination is explicitly allowlisted below; no filesystem discovery
is used.

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

from scripts.export_binary_seed_provenance import load_public_seed_release
from scripts.replay_seed_robustness import (
    ReplayValidationError,
    verify_replay_payloads,
)


ARCHIVE_ROOT = "selective-segmentation-analysis-artifact-v4"
ARCHIVE_MTIME = 0
ARCHIVE_FILE_MODE = 0o644
EXPECTED_CONDITION_COUNT = 16
EXPECTED_RELEASE_FILE_COUNT = 114
EXPECTED_ARCHIVE_MEMBER_COUNT = 117

# The builder consumes the byte-exact public-mirror layout.  This makes the
# same command usable in a clean public clone and prevents the anonymous
# release from depending on private workspace-only output paths.
ANALYSIS_SOURCE = "results/analysis.json"
CSV_SOURCE = "results/main_table.csv"
CAMPAIGN_LOCK_SOURCE = "results/campaign.lock.json"
PUBLIC_SEED_ANALYSIS_SOURCE = "results/seed_robustness_analysis.json"
PUBLIC_SEED_SCHEDULER_SOURCE = "results/seed_scheduler_summary.json"
PUBLIC_SEED_PROVENANCE_SOURCE = "results/seed_provenance.json"
PUBLIC_SEED_TABLE_SOURCE = "docs/Tables/seed_robustness.tex"
PUBLIC_SEED_GATE_TABLE_SOURCE = "docs/Tables/seed_sensitivity_main.tex"
PUBLIC_SEED_REPLAY_LOCK_SOURCE = "results/seed_replay.lock.json"
PUBLIC_SEED_REPLAY_GUARD_SOURCE = "results/seed_replay.complete.json"

SEED_GATE_CONTRAST = "nhd_vs_nhd95_under_nhd95"
SEED_GATE_DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
SEED_GATE_DATASET_LABELS = {
    "pet": "Oxford Pet",
    "kvasir": "Kvasir-SEG",
    "fives": "FIVES",
    "isic": "ISIC",
    "tn3k": "TN3K",
}
SEED_GATE_CONDITIONS = ("clipseg-target", "deeplabv3-target")
SEED_GATE_CONDITION_LABELS = {
    "clipseg-target": "CLIP-T",
    "deeplabv3-target": "DL-T",
}
EXPECTED_SEED_GATE_REVERSAL_CELLS = frozenset(
    {
        ("clipseg-target", "isic"),
        ("clipseg-target", "tn3k"),
        ("deeplabv3-target", "kvasir"),
        ("deeplabv3-target", "isic"),
        ("deeplabv3-target", "pet"),
    }
)
EXPECTED_SEED_GATE_DAGGER_CELLS = frozenset(
    {
        ("clipseg-target", "tn3k"),
        ("deeplabv3-target", "isic"),
        ("deeplabv3-target", "pet"),
    }
)

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

# Thirty path-free replay inputs: the same ten target conditions for each of
# training seeds 0, 1, and 2.  These are explicit release data, not discovery
# patterns.  Every record file has one strictly allowlisted portable manifest.
PORTABLE_SEED_RUNS = (
    (0, "fives", "clipseg-target", "627922c22aa3aa21"),
    (0, "fives", "deeplabv3-target", "a62d754378787146"),
    (0, "isic", "clipseg-target", "69fc681ff737f148"),
    (0, "isic", "deeplabv3-target", "a42288ea78a340b1"),
    (0, "kvasir", "clipseg-target", "1a1c02b968ab86a4"),
    (0, "kvasir", "deeplabv3-target", "5cca940975b7f02d"),
    (0, "pet", "clipseg-target", "cd4341aaed2bda20"),
    (0, "pet", "deeplabv3-target", "fd2f61c609c18fba"),
    (0, "tn3k", "clipseg-target", "eccb8f4f045a5473"),
    (0, "tn3k", "deeplabv3-target", "01b4c3a58986cf27"),
    (1, "fives", "clipseg-target", "2435c524ef0766f3"),
    (1, "fives", "deeplabv3-target", "9f314f06a2eab595"),
    (1, "isic", "clipseg-target", "c848014159175786"),
    (1, "isic", "deeplabv3-target", "46dc7182fd5fc6e3"),
    (1, "kvasir", "clipseg-target", "2860e2e7601882f3"),
    (1, "kvasir", "deeplabv3-target", "acc2da1ce32ad656"),
    (1, "pet", "clipseg-target", "0fec2bd382f23d4f"),
    (1, "pet", "deeplabv3-target", "127e78caea1a1586"),
    (1, "tn3k", "clipseg-target", "63176cdfefc0aa5f"),
    (1, "tn3k", "deeplabv3-target", "f7530ce073e35ec3"),
    (2, "fives", "clipseg-target", "3c23e51d0dbd3f7c"),
    (2, "fives", "deeplabv3-target", "164b183d9f7e8b7b"),
    (2, "isic", "clipseg-target", "1bef05e564bac5ac"),
    (2, "isic", "deeplabv3-target", "db6ed7fad1d713ac"),
    (2, "kvasir", "clipseg-target", "6b50e05a2e7b7bb2"),
    (2, "kvasir", "deeplabv3-target", "7c7a5f46a399fb83"),
    (2, "pet", "clipseg-target", "4f23a8a500f3ad40"),
    (2, "pet", "deeplabv3-target", "42833caf0bdfc194"),
    (2, "tn3k", "clipseg-target", "73f6639aefd90529"),
    (2, "tn3k", "deeplabv3-target", "2b5e117d5edba5b6"),
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
    ReleaseFile(
        "scripts/replay_seed_robustness.py",
        "scripts/replay_seed_robustness.py",
        "seed-replay-code",
    ),
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

_PUBLIC_SEED_FILES = (
    ReleaseFile(
        PUBLIC_SEED_ANALYSIS_SOURCE,
        "results/seed_robustness_analysis.json",
        "seed-analysis",
    ),
    ReleaseFile(
        PUBLIC_SEED_SCHEDULER_SOURCE,
        "results/seed_scheduler_summary.json",
        "seed-scheduler-summary",
    ),
    ReleaseFile(
        PUBLIC_SEED_PROVENANCE_SOURCE,
        "results/seed_provenance.json",
        "seed-provenance",
    ),
    ReleaseFile(
        PUBLIC_SEED_TABLE_SOURCE,
        "tables/seed_robustness.tex",
        "seed-table",
    ),
    ReleaseFile(
        PUBLIC_SEED_GATE_TABLE_SOURCE,
        "tables/seed_sensitivity_main.tex",
        "seed-gate-table",
    ),
)

_PUBLIC_SEED_REPLAY_FILES = (
    ReleaseFile(
        PUBLIC_SEED_REPLAY_LOCK_SOURCE,
        "results/seed_replay.lock.json",
        "seed-replay-lock",
    ),
    *tuple(
        ReleaseFile(
            (
                f"results/seed_records/seed-{seed}/{dataset}/{condition}/"
                f"{run_id}/{name}"
            ),
            (
                f"results/seed_records/seed-{seed}/{dataset}/{condition}/"
                f"{run_id}/{name}"
            ),
            role,
        )
        for seed, dataset, condition, run_id in PORTABLE_SEED_RUNS
        for name, role in (
            ("manifest.json", "seed-record-manifest"),
            ("records.jsonl", "seed-records"),
        )
    ),
    ReleaseFile(
        PUBLIC_SEED_REPLAY_GUARD_SOURCE,
        "results/seed_replay.complete.json",
        "seed-replay-guard",
    ),
)

# The only repository files eligible for inclusion.  Generated metadata below
# is separately fixed by GENERATED_MEMBER_NAMES.
RELEASE_FILES = (
    _CODE_FILES
    + _LOCKED_RESULT_FILES
    + _ASSEMBLED_FILES
    + _TABLE_FILES
    + _PUBLIC_SEED_FILES
    + _PUBLIC_SEED_REPLAY_FILES
)
GENERATED_MEMBER_NAMES = (
    "MANIFEST.sha256",
    "README.md",
    "requirements-analysis.txt",
)

REQUIREMENTS = "numpy==2.5.1\nscipy==1.18.0\n"

ANONYMOUS_README = """# Core analysis reproduction artifact

This anonymous artifact contains the fixed 16-condition records, their
manifests, the immutable campaign lock, the analysis and table source code,
the seven canonical tables, the public seed-robustness bundle, all 30 portable
seed 0/1/2 condition records, and the compact Gate-C table used in the main
text.  It does not contain model
weights, raw images, training data, per-job scheduler records, submission
receipts, checkpoint bytes, or author identities.

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

The seed release can be replayed end to end with one command:

```bash
python -m scripts.replay_seed_robustness
```

This validates the path-free lock and all 30 manifest/record pairs, rejoins the
same held-out cohort across seeds, recomputes every AURC and adjacent-loss
contrast, rebuilds `seed_robustness_analysis.json`, `seed_robustness.tex`, and
`seed_sensitivity_main.tex`, and requires each result to match the released
reference byte for byte.  The replay also checks the five reversal cells,
three seed-0 minority-direction markers, and the displayed `100 x AURC` scale.
The outputs are written below `rebuild/seed_replay`.  Probability maps and
model weights are not required for this post-inference statistical replay.

`MANIFEST.sha256` authenticates every other archive member.  Verification can
also be performed before extraction with the artifact builder's `verify`
subcommand, which reruns both the core and seed cross-file validators on the
archive members rather than trusting the checksum manifest alone.
"""


class ArtifactValidationError(RuntimeError):
    """Raised when a release input or archive violates the fixed contract."""


_OVERLEAF_TOKEN_PATTERN = b"olp" + rb"_[A-Za-z0-9]+"
_OPENAI_PROJECT_TOKEN_PATTERN = b"sk" + rb"-proj-[A-Za-z0-9_-]+"
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
            + _OVERLEAF_TOKEN_PATTERN
            + rb"|"
            + _OPENAI_PROJECT_TOKEN_PATTERN
            + rb"|"
            rb"\b(?:api[_-]?key|access[_-]?token|password|passwd|bearer)\b\s*[:=])"
        ),
    ),
    (
        "raw job identifier",
        re.compile(rb"(?i)[\"'](?:job_id|receipt_job_id|record_slurm_job_id)[\"']\s*:"),
    ),
    (
        "submission receipt content",
        re.compile(rb"(?i)[\"']receipt_schema_version[\"']\s*:"),
    ),
)

_TABLE_ANALYSIS_HASH = re.compile(rb"Source analysis\.json SHA-256: ([0-9a-f]{64})")
_MANIFEST_LINE = re.compile(rb"([0-9a-f]{64})  ([^\r\n]+)\n")
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
_FORBIDDEN_RELEASE_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".npy",
        ".npz",
        ".onnx",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


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

    if data.startswith(b"PK\x03\x04"):
        raise ArtifactValidationError(
            f"{source} contains a forbidden NPZ/checkpoint payload"
        )
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
    if len(RELEASE_FILES) != EXPECTED_RELEASE_FILE_COUNT:
        raise ArtifactValidationError(
            "release allowlist must contain exactly "
            f"{EXPECTED_RELEASE_FILE_COUNT} sources"
        )
    if len(sources) != len(set(sources)):
        raise ArtifactValidationError("release source allowlist contains duplicates")
    if len(destinations) != len(set(destinations)):
        raise ArtifactValidationError(
            "release destination allowlist contains duplicates"
        )
    for item in RELEASE_FILES:
        source = _safe_relative_path(item.source, label="release source")
        destination = _safe_relative_path(item.destination, label="release destination")
        for path in (source, destination):
            lowered_parts = tuple(part.lower() for part in path.parts)
            if path.suffix.lower() in _FORBIDDEN_RELEASE_SUFFIXES or any(
                "receipt" in part or "checkpoint" in part for part in lowered_parts
            ):
                raise ArtifactValidationError(
                    f"release allowlist admits a private binary or receipt: {path}"
                )

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
    seed_roles = tuple(item.role for item in _PUBLIC_SEED_FILES)
    if seed_roles != (
        "seed-analysis",
        "seed-scheduler-summary",
        "seed-provenance",
        "seed-table",
        "seed-gate-table",
    ):
        raise ArtifactValidationError("public seed release allowlist is incomplete")
    replay_roles = [item.role for item in _PUBLIC_SEED_REPLAY_FILES]
    if (
        len(PORTABLE_SEED_RUNS) != 30
        or replay_roles.count("seed-replay-lock") != 1
        or replay_roles.count("seed-replay-guard") != 1
        or replay_roles.count("seed-record-manifest") != 30
        or replay_roles.count("seed-records") != 30
    ):
        raise ArtifactValidationError("portable seed replay allowlist is incomplete")
    replay_identities = [row[:3] for row in PORTABLE_SEED_RUNS]
    if len(replay_identities) != len(set(replay_identities)):
        raise ArtifactValidationError("portable seed replay identities are duplicated")


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


def _render_seed_gate_table(analysis: dict) -> bytes:
    """Independently rebuild the fixed compact Gate-C table from public data."""

    try:
        source_sha256 = analysis["provenance"]["source_analysis_sha256"]
        cells = analysis["cells"]
        gate = analysis["gate_c"]
    except (KeyError, TypeError) as error:
        raise ArtifactValidationError(
            "public seed analysis cannot drive the compact Gate-C table"
        ) from error
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ArtifactValidationError(
            "public seed analysis has an invalid source-analysis digest"
        )

    by_key = {}
    try:
        for cell in cells:
            key = (cell["dataset"], cell["condition"])
            if key in by_key:
                raise ArtifactValidationError(
                    "public seed analysis repeats a compact-table cell"
                )
            by_key[key] = cell
    except (KeyError, TypeError) as error:
        raise ArtifactValidationError(
            "public seed analysis has an invalid compact-table grid"
        ) from error
    expected_grid = {
        (dataset, condition)
        for condition in SEED_GATE_CONDITIONS
        for dataset in SEED_GATE_DATASETS
    }
    if set(by_key) != expected_grid:
        raise ArtifactValidationError(
            "public seed analysis has an incomplete compact-table grid"
        )

    reversal_cells = set()
    dagger_cells = set()
    try:
        for condition in SEED_GATE_CONDITIONS:
            for dataset in SEED_GATE_DATASETS:
                summary = by_key[(dataset, condition)]["summary"]["contrasts"][
                    SEED_GATE_CONTRAST
                ]
                if summary["direction_reversal"]:
                    reversal_cells.add((condition, dataset))
                    if not summary["seed0_is_majority_direction"]:
                        dagger_cells.add((condition, dataset))
        listed_nonmajority = {
            (entry["condition"], entry["dataset"], entry["contrast"])
            for entry in gate["seed0_not_majority_cells"]
        }
    except (KeyError, TypeError) as error:
        raise ArtifactValidationError(
            "public seed analysis lacks a compact Gate-C statistic"
        ) from error
    expected_nonmajority = {
        (condition, dataset, SEED_GATE_CONTRAST)
        for condition, dataset in EXPECTED_SEED_GATE_DAGGER_CELLS
    }
    if (
        gate.get("fired") is not True
        or gate.get("direction_reversal_counts", {}).get(SEED_GATE_CONTRAST) != 5
        or gate.get("contrasts_with_at_least_three_reversals")
        != [SEED_GATE_CONTRAST]
        or reversal_cells != EXPECTED_SEED_GATE_REVERSAL_CELLS
        or dagger_cells != EXPECTED_SEED_GATE_DAGGER_CELLS
        or listed_nonmajority != expected_nonmajority
    ):
        raise ArtifactValidationError(
            "public seed analysis differs from the fixed compact Gate-C result"
        )

    def contrast_cell(summary, *, dagger):
        try:
            values = summary["values"]
            displayed = "/".join(
                f"{100 * float(values[str(seed)]):+.3f}" for seed in (0, 1, 2)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactValidationError(
                "public seed analysis has invalid compact-table values"
            ) from error
        marker = r"^{\dagger}" if dagger else ""
        return rf"$\bigl({displayed}\bigr){marker}$"

    lines = [
        "% AUTO-GENERATED by scripts/render_seed_gate_table.py; DO NOT EDIT.",
        f"% Source seed analysis SHA-256: {source_sha256}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Checkpoint-level direction reversals for nHD versus nHD95 under "
        r"nHD95 risk. Each populated cell reports "
        r"$100\!\times\![\operatorname{AURC}(C_{\mathrm{nHD}})-"
        r"\operatorname{AURC}(C_{\mathrm{nHD95}})]$ for seeds 0/1/2; negative "
        r"values favor nHD. Only the five cells that reverse direction are shown; "
        r"all other cells are dashes. $\dagger$ marks the three cells for which "
        r"seed 0 disagrees with the majority direction. These are descriptive "
        r"results from three independently trained checkpoints, not seed-level "
        r"inference.}",
        r"\label{tab:seed-sensitivity-main}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Model & "
        + " & ".join(SEED_GATE_DATASET_LABELS[item] for item in SEED_GATE_DATASETS)
        + r" \\",
        r"\midrule",
    ]
    for condition in SEED_GATE_CONDITIONS:
        rendered_cells = []
        for dataset in SEED_GATE_DATASETS:
            key = (condition, dataset)
            if key not in reversal_cells:
                rendered_cells.append(r"--")
                continue
            summary = by_key[(dataset, condition)]["summary"]["contrasts"][
                SEED_GATE_CONTRAST
            ]
            rendered_cells.append(
                contrast_cell(summary, dagger=key in dagger_cells)
            )
        lines.append(
            " & ".join(
                [SEED_GATE_CONDITION_LABELS[condition], *rendered_cells]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _validate_public_seed_closure(files: dict[str, bytes]) -> None:
    """Validate the mandatory portable seed bundle and its rendered table."""

    seed_sources = (
        PUBLIC_SEED_ANALYSIS_SOURCE,
        PUBLIC_SEED_SCHEDULER_SOURCE,
        PUBLIC_SEED_PROVENANCE_SOURCE,
    )
    with tempfile.TemporaryDirectory(prefix="selectseg-public-seed-") as directory:
        # macOS commonly reports temporary paths below /var, which is itself a
        # symlink to /private/var.  The public-release loader deliberately
        # rejects every symlink ancestor, so pass it the canonical real path.
        root = Path(directory).resolve(strict=True)
        paths = []
        for source in seed_sources:
            destination = root / PurePosixPath(source).name
            destination.write_bytes(files[source])
            paths.append(destination)
        try:
            release = load_public_seed_release(*paths)
        except (OSError, ValueError) as error:
            raise ArtifactValidationError(
                f"invalid public seed release: {error}"
            ) from error

    observed_hashes = {
        "analysis": _sha256_bytes(files[PUBLIC_SEED_ANALYSIS_SOURCE]),
        "scheduler": _sha256_bytes(files[PUBLIC_SEED_SCHEDULER_SOURCE]),
        "provenance": _sha256_bytes(files[PUBLIC_SEED_PROVENANCE_SOURCE]),
    }
    if release.get("sha256") != observed_hashes:
        raise ArtifactValidationError(
            "public seed loader did not bind the selected release bytes"
        )

    table = files[PUBLIC_SEED_TABLE_SOURCE]
    table_sha256 = _sha256_bytes(table)
    provenance = release["provenance"]
    if provenance["analysis"]["table_sha256"] != table_sha256:
        raise ArtifactValidationError(
            "seed robustness table differs from the public provenance binding"
        )

    source_analysis_sha256 = release["analysis"]["provenance"]["source_analysis_sha256"]
    matches = _TABLE_ANALYSIS_HASH.findall(table)
    if matches != [source_analysis_sha256.encode("ascii")]:
        raise ArtifactValidationError(
            "seed robustness table must contain exactly one matching source-analysis "
            "SHA-256 comment"
        )

    expected_gate_table = _render_seed_gate_table(release["analysis"])
    if files[PUBLIC_SEED_GATE_TABLE_SOURCE] != expected_gate_table:
        raise ArtifactValidationError(
            "compact Gate-C table is not the byte-exact public-analysis rebuild"
        )


def _validate_seed_replay_closure(files: dict[str, bytes]) -> None:
    """Recompute all seed outputs from the 30 portable condition records."""

    guard = _strict_json(
        files[PUBLIC_SEED_REPLAY_GUARD_SOURCE],
        source=PUBLIC_SEED_REPLAY_GUARD_SOURCE,
    )
    expected_guard_fields = {
        "schema_version",
        "artifact_type",
        "lock_sha256",
        "condition_count",
        "portable_file_count",
        "portable_bundle_sha256",
    }
    if not isinstance(guard, dict) or set(guard) != expected_guard_fields:
        raise ArtifactValidationError("portable seed replay guard schema is invalid")
    if (
        guard["schema_version"] != 1
        or guard["artifact_type"] != "selectseg.portable_seed_replay_complete"
        or guard["condition_count"] != 30
        or guard["portable_file_count"] != 60
        or guard["lock_sha256"]
        != _sha256_bytes(files[PUBLIC_SEED_REPLAY_LOCK_SOURCE])
    ):
        raise ArtifactValidationError("portable seed replay guard binding is invalid")
    bundle_digest = hashlib.sha256()
    prefix = "results/seed_records/"
    replay_payloads = sorted(
        (item.source, files[item.source])
        for item in _PUBLIC_SEED_REPLAY_FILES
        if item.role in {"seed-record-manifest", "seed-records"}
    )
    for source, payload in replay_payloads:
        if not source.startswith(prefix):
            raise ArtifactValidationError("portable seed replay source is noncanonical")
        bundle_digest.update(source.removeprefix(prefix).encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(hashlib.sha256(payload).digest())
        bundle_digest.update(b"\0")
    if guard["portable_bundle_sha256"] != bundle_digest.hexdigest():
        raise ArtifactValidationError("portable seed replay bundle digest mismatch")

    aliases = {
        "tables/seed_robustness.tex": PUBLIC_SEED_TABLE_SOURCE,
        "tables/seed_sensitivity_main.tex": PUBLIC_SEED_GATE_TABLE_SOURCE,
    }

    def read_payload(relative: str) -> bytes:
        source = aliases.get(relative, relative)
        try:
            return files[source]
        except KeyError as error:
            raise ArtifactValidationError(
                f"seed replay requests a non-release member: {relative}"
            ) from error

    try:
        _, report = verify_replay_payloads(
            files[PUBLIC_SEED_REPLAY_LOCK_SOURCE], read_payload
        )
    except (KeyError, OSError, ValueError, ReplayValidationError) as error:
        raise ArtifactValidationError(f"invalid portable seed replay: {error}") from error
    if report != {
        "verified": True,
        "condition_count": 30,
        "seed_count": 3,
        "output_sha256": {
            "analysis": _sha256_bytes(files[PUBLIC_SEED_ANALYSIS_SOURCE]),
            "robustness_table": _sha256_bytes(files[PUBLIC_SEED_TABLE_SOURCE]),
            "gate_table": _sha256_bytes(files[PUBLIC_SEED_GATE_TABLE_SOURCE]),
        },
    }:
        raise ArtifactValidationError("portable seed replay returned an invalid report")


def _validate_release_closure(files: dict[str, bytes]) -> None:
    _validate_canonical_closure(files)
    _validate_public_seed_closure(files)
    _validate_seed_replay_closure(files)


def _load_release_files(repo_root: Path) -> dict[str, bytes]:
    _validate_allowlist_contract()
    files = {}
    for item in RELEASE_FILES:
        data = _read_regular_source(repo_root, item.source)
        scan_anonymous_bytes(data, source=item.source)
        files[item.source] = data
    _validate_release_closure(files)
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

    _validate_allowlist_contract()
    relative = [item.destination for item in RELEASE_FILES]
    relative.extend(GENERATED_MEMBER_NAMES)
    names = tuple(sorted(f"{ARCHIVE_ROOT}/{name}" for name in relative))
    if len(names) != EXPECTED_ARCHIVE_MEMBER_COUNT:
        raise ArtifactValidationError(
            "archive allowlist must contain exactly "
            f"{EXPECTED_ARCHIVE_MEMBER_COUNT} members"
        )
    return names


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
    archived_sources = {
        item.source: extracted[item.destination] for item in RELEASE_FILES
    }
    _validate_release_closure(archived_sources)
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
