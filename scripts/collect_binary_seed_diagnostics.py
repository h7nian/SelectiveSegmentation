#!/usr/bin/env python3
"""Resolve the exact diagnostic closure for the binary seed extension.

The collector is deliberately read-only.  It validates the immutable
downstream lock and the fixed diagnose submission receipt, then inspects only
the 20 lock-derived artifact directories.  Every returned summary is strictly
loaded together with its descriptor payload and rebound to the corresponding
frozen artifact.  No directory-wide discovery, result mutation, or public
artifact publication is performed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from scripts.export_binary_seed_provenance import _validate_receipt
from scripts.submit_binary_seed_extension import (
    _expected_receipt_path,
    plan_downstream_jobs,
)
from selectseg.binary_diagnostics import SUMMARY_NAME, load_binary_diagnostics
from selectseg.binary_seed_downstream import load_downstream_lock
from selectseg.binary_seed_extension import (
    _digest,
    _reject_symlink_ancestors,
    iter_experiments,
)


EXPECTED_DIAGNOSTICS = 20


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-lock", required=True)
    parser.add_argument("--expected-downstream-lock-sha256", required=True)
    parser.add_argument(
        "--diagnose-receipt",
        required=True,
        help="the fixed append-only diagnose receipt recorded by the seed planner",
    )
    parser.add_argument(
        "--format",
        choices=("arguments", "json", "argv0"),
        default="arguments",
        help=(
            "human-readable repeated arguments, a JSON path array, or NUL-delimited "
            "argv tokens for direct Bash consumption"
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _flag_value(command: Sequence[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"diagnose command must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"diagnose command has no value for {flag}")
    value = command[index + 1]
    if not isinstance(value, str) or not value:
        raise ValueError(f"diagnose command has an invalid value for {flag}")
    return value


def _locked_cells(downstream_binding):
    cells = {}
    for campaign in downstream_binding["lock"]["campaigns"]:
        seed = campaign["training_seed"]
        for row in campaign["freeze_records"]:
            key = (seed, row["dataset"], row["condition"])
            if key in cells:
                raise ValueError("downstream lock repeats a diagnostic cell")
            cells[key] = row
    if len(cells) != EXPECTED_DIAGNOSTICS:
        raise ValueError("downstream lock must bind exactly 20 diagnostic cells")
    return cells


def _one_summary(parent: Path) -> Path:
    """Return the single direct diagnostic summary below one locked artifact."""

    _reject_symlink_ancestors(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise FileNotFoundError(
            f"locked diagnostic artifact directory is missing: {parent}"
        )
    candidates = []
    for child in sorted(parent.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise ValueError(f"symlink diagnostic directory is forbidden: {child}")
        if not child.is_dir():
            continue
        summary = child / SUMMARY_NAME
        if summary.exists() or summary.is_symlink():
            _reject_symlink_ancestors(summary)
            if not summary.is_file() or summary.is_symlink():
                raise FileNotFoundError(
                    f"expected a regular non-symlink diagnostic summary: {summary}"
                )
            candidates.append(summary)
    if len(candidates) != 1:
        raise ValueError(
            f"locked artifact must have exactly one completed diagnostic summary; "
            f"found {len(candidates)} below {parent}"
        )
    return candidates[0]


def _validate_summary_binding(loaded, row, *, seed: int):
    summary = loaded.summary
    artifact = summary["artifact"]
    expected_manifest = _resolve(row["artifact_manifest_path"])
    observed_manifest = _resolve(artifact["manifest_path"])
    expected = {
        "artifact_id": row["artifact_id"],
        "manifest_sha256": row["artifact_manifest_sha256"],
        "sample_id_sha256": row["sample_id_sha256"],
        "num_samples": row["num_samples"],
        "dataset": row["dataset"],
        "condition": row["condition"],
        "model": row["model"],
    }
    for field, value in expected.items():
        if artifact[field] != value:
            raise ValueError(
                f"diagnostic summary differs from locked seed-{seed} cell on {field}"
            )
    if observed_manifest != expected_manifest:
        raise ValueError(
            f"diagnostic summary names a different artifact manifest for seed-{seed}"
        )
    descriptors = summary["descriptors"]
    if descriptors["included"] is not True:
        raise ValueError("seed diagnostic must include its descriptor payload")
    if descriptors["num_rows"] != row["num_samples"]:
        raise ValueError(
            "seed diagnostic descriptor count differs from the locked cohort"
        )


def collect_diagnostic_summaries(
    *,
    downstream_lock: str | Path,
    expected_downstream_lock_sha256: str,
    diagnose_receipt: str | Path,
) -> tuple[Path, ...]:
    """Return 20 strictly validated summaries in the frozen experiment order."""

    expected_sha = _digest(
        expected_downstream_lock_sha256,
        location="expected downstream-lock sha256",
    )
    downstream_binding = load_downstream_lock(
        downstream_lock,
        expected_sha256=expected_sha,
    )
    binding = downstream_binding["binding"]
    expected_receipt = _expected_receipt_path(binding, "diagnose")
    receipt = Path(diagnose_receipt)
    _reject_symlink_ancestors(receipt)
    if receipt.resolve() != expected_receipt.resolve():
        raise ValueError(
            f"diagnose receipt must use the fixed planner path {expected_receipt}"
        )
    if not receipt.is_file() or receipt.is_symlink():
        raise FileNotFoundError(f"diagnose receipt must be a regular file: {receipt}")

    jobs = plan_downstream_jobs(downstream_binding, "diagnose")
    if (
        len(jobs) != EXPECTED_DIAGNOSTICS
        or len({(job.phase, job.key) for job in jobs}) != EXPECTED_DIAGNOSTICS
    ):
        raise ValueError("diagnose plan must contain exactly 20 unique jobs")
    receipt_evidence = _validate_receipt(receipt, jobs, phase="diagnose")
    if (
        receipt_evidence["count"] != EXPECTED_DIAGNOSTICS
        or len(set(receipt_evidence["job_ids"])) != EXPECTED_DIAGNOSTICS
    ):
        raise ValueError("diagnose receipt must bind 20 unique submitted job IDs")

    cells = _locked_cells(downstream_binding)
    loaded_by_cell = {}
    expected_root = binding["spec"]["paths"]["diagnostic_root"]
    for job in jobs:
        if job.phase != "seed_diagnose" or len(job.key) != 4:
            raise ValueError("diagnose plan contains an invalid seed-aware identity")
        seed, dataset, condition, _partition = job.key
        key = (seed, dataset, condition)
        if key not in cells or key in loaded_by_cell:
            raise ValueError("diagnose plan differs from the locked 20-cell grid")
        row = cells[key]
        manifest_path = _flag_value(job.command, "--artifact-manifest")
        manifest_sha = _digest(
            _flag_value(job.command, "--expected-artifact-manifest-sha256"),
            location="diagnose command artifact-manifest sha256",
        )
        output_root = _flag_value(job.command, "--output-root")
        if (
            _resolve(manifest_path) != _resolve(row["artifact_manifest_path"])
            or manifest_sha != row["artifact_manifest_sha256"]
        ):
            raise ValueError(
                "diagnose command differs from its locked artifact binding"
            )
        if output_root != expected_root:
            raise ValueError("diagnose command differs from the locked diagnostic root")
        parent = (
            _resolve(output_root)
            / row["dataset"]
            / row["condition"]
            / row["artifact_id"]
        )
        summary_path = _one_summary(parent)
        loaded = load_binary_diagnostics(summary_path, validate_descriptors=True)
        if Path(loaded.summary_path).resolve() != summary_path.resolve():
            raise ValueError("diagnostic loader returned a different summary path")
        _validate_summary_binding(loaded, row, seed=seed)
        loaded_by_cell[key] = loaded.summary_path

    if set(loaded_by_cell) != set(cells):
        raise ValueError("diagnose plan does not cover the complete locked cell grid")
    order = [
        (
            experiment["training_seed"],
            experiment["dataset"]["name"],
            experiment["model"]["condition"],
        )
        for experiment in iter_experiments(binding["spec"])
    ]
    if len(order) != EXPECTED_DIAGNOSTICS or set(order) != set(cells):
        raise ValueError("seed specification order differs from the downstream lock")
    return tuple(Path(loaded_by_cell[key]).resolve() for key in order)


def formatted_output(paths: Sequence[Path], output_format: str) -> bytes:
    portable = [_portable(Path(path)) for path in paths]
    if len(portable) != EXPECTED_DIAGNOSTICS or len(set(portable)) != len(portable):
        raise ValueError("collector output must contain 20 unique diagnostic paths")
    if output_format == "arguments":
        return (
            "".join(f"--diagnostic-summary {shlex.quote(path)}\n" for path in portable)
        ).encode("utf-8")
    if output_format == "json":
        return (json.dumps(portable, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    if output_format == "argv0":
        tokens = [
            token for path in portable for token in ("--diagnostic-summary", path)
        ]
        return b"".join(token.encode("utf-8") + b"\0" for token in tokens)
    raise ValueError(f"unsupported collector output format {output_format!r}")


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    paths = collect_diagnostic_summaries(
        downstream_lock=args.downstream_lock,
        expected_downstream_lock_sha256=args.expected_downstream_lock_sha256,
        diagnose_receipt=args.diagnose_receipt,
    )
    sys.stdout.buffer.write(formatted_output(paths, args.format))
    sys.stdout.buffer.flush()
    return paths


if __name__ == "__main__":
    main()
