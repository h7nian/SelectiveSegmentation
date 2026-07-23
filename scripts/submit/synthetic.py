"""Plan one CPU Slurm job per known-posterior synthetic grid cell.

Dry-run is the default.  ``pilot`` emits exactly 12 jobs; ``full`` emits the
remaining 348 jobs only after the pilot gate.  Slurm arrays and GPUs are never
used, and submission reuses the repository's append-only receipt protocol.
"""

from __future__ import annotations

import argparse
import copy
import math
from datetime import datetime
from pathlib import Path

from scripts.submit.main import PlannedJob, execute_plan, slurm_command
from scripts.analyze.synthetic import (
    ANALYSIS_SCHEMA_VERSION,
    analyze as analyze_synthetic_posterior,
)
from selectseg.studies.synthetic import (
    CPU_PARTITIONS,
    _canonical_json,
    _load_json,
    cell_seeds,
    load_synthetic_lock,
    selected_cells,
)
from selectseg.studies.synthetic_matrix import (
    _load_config as load_matrix_config,
    all_cells as all_matrix_cells,
)
from selectseg.quadrature import sha256_file


DEFAULT_LOCK = "configs/auxiliary/synthetic_posterior-v1.lock.json"
DEFAULT_MATRIX_CONFIG = "configs/auxiliary/synthetic_posterior_matrix_v1.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_RECEIPT = Path("outputs/synthetic_posterior_campaign/full-submissions.jsonl")
_PILOT_ANALYSIS_FIELDS = frozenset(
    {
        "analysis_schema_version",
        "analysis_id",
        "created_utc",
        "campaign_id",
        "mode",
        "lock",
        "num_cells",
        "source_manifests",
        "pilot_gate",
        "groups",
        "headline_groups",
        "coupling_summaries",
        "interpretation",
    }
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("posterior", "matrix"),
        default="posterior",
        help="immutable v1 stress test or estimator-coupling matrix",
    )
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--config", default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--submit", action="store_true", help="call sbatch; otherwise print the plan"
    )
    parser.add_argument(
        "--receipt", help="append-only receipt; required only when --submit is used"
    )
    parser.add_argument(
        "--pilot-analysis",
        help="strict 12-cell pilot analysis required explicitly by --phase full",
    )
    parser.add_argument(
        "--expected-pilot-analysis-sha256",
        help="expected SHA-256 of --pilot-analysis; required by --phase full",
    )
    return parser.parse_args(argv)


def plan_matrix_jobs(config=DEFAULT_MATRIX_CONFIG, *, phase="pilot"):
    """Plan one independent job per matrix simulation/repeat."""

    config_path, value = load_matrix_config(Path(config))
    cells = all_matrix_cells(value)
    pilot = {
        cell
        for cell in cells
        if cell.morphology == "disk" and cell.replicate == 0
    }
    selected = tuple(
        cell for cell in cells if (cell in pilot) == (phase == "pilot")
    )
    if phase not in {"pilot", "full"}:
        raise ValueError("matrix phase must be pilot or full")
    config_sha = sha256_file(config_path)
    scheduler = value["scheduler"]
    partitions = scheduler["cpu_partition_candidates"]
    jobs = []
    for index, cell in enumerate(selected):
        rotated = partitions[index % len(partitions) :] + partitions[: index % len(partitions)]
        seed = cell_seeds(value, cell)["cell_seed"]
        command = slurm_command(
            job_name=(
                f"selseg-matrix-{cell.coupling[:6]}-{cell.sharpness[:3]}-"
                f"{cell.morphology[:5]}-r{cell.replicate:02d}"
            ),
            partition=",".join(rotated),
            cpus=scheduler["cpus_per_job"],
            memory=scheduler["memory"],
            time_limit=scheduler["time_limit"],
            argv=(
                "python",
                "-m",
                "selectseg.studies.synthetic_matrix",
                "--config",
                str(config_path),
                "--expected-config-sha256",
                config_sha,
                "--true-coupling",
                cell.coupling,
                "--sharpness",
                cell.sharpness,
                "--morphology",
                cell.morphology,
                "--replicate",
                str(cell.replicate),
                "--expected-cell-seed",
                str(seed),
            ),
        )
        jobs.append(
            PlannedJob(
                phase=f"synthetic_matrix_{phase}",
                key=(*cell.key, seed, ",".join(rotated)),
                command=command,
            )
        )
    expected = 12 if phase == "pilot" else 348
    if len(jobs) != expected or len({job.key[:5] for job in jobs}) != expected:
        raise RuntimeError(f"matrix {phase} plan must contain {expected} unique jobs")
    return tuple(jobs)


def _strict_sha256(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _without_created_utc(analysis):
    comparable = copy.deepcopy(analysis)
    comparable.pop("created_utc", None)
    return comparable


def validate_pilot_gate(binding, pilot_analysis, expected_sha256):
    """Revalidate the immutable 12-cell pilot before expanding the full grid."""

    if pilot_analysis is None or expected_sha256 is None:
        raise ValueError(
            "full phase requires explicit --pilot-analysis and "
            "--expected-pilot-analysis-sha256"
        )
    analysis_path = Path(pilot_analysis)
    if analysis_path.is_symlink() or not analysis_path.is_file():
        raise FileNotFoundError(
            f"pilot analysis must be a regular, non-symlink file: {analysis_path}"
        )
    expected = _strict_sha256(
        expected_sha256, location="expected pilot-analysis sha256"
    )
    if sha256_file(analysis_path) != expected:
        raise ValueError("pilot-analysis SHA-256 mismatch")
    _, analysis = _load_json(analysis_path, name="synthetic pilot analysis")
    if set(analysis) != _PILOT_ANALYSIS_FIELDS:
        raise ValueError("pilot analysis has an unexpected schema")
    if analysis.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("pilot analysis has an unsupported schema version")
    if analysis.get("mode") != "pilot" or analysis.get("num_cells") != 12:
        raise ValueError("full phase requires an exact 12-cell pilot analysis")
    if analysis.get("campaign_id") != binding["spec"]["campaign_id"]:
        raise ValueError("pilot analysis belongs to a different synthetic campaign")
    lock_provenance = analysis.get("lock")
    if not isinstance(lock_provenance, dict) or set(lock_provenance) != {
        "path",
        "sha256",
    }:
        raise ValueError("pilot analysis lock provenance is malformed")
    if lock_provenance["sha256"] != binding["sha256"]:
        raise ValueError("pilot analysis is bound to a different synthetic lock")
    if Path(lock_provenance["path"]).resolve() != binding["path"].resolve():
        raise ValueError("pilot analysis lock path differs from the active lock")
    sources = analysis.get("source_manifests")
    if (
        not isinstance(sources, list)
        or len(sources) != 12
        or not all(
            isinstance(source, dict) and set(source) == {"path", "sha256"}
            for source in sources
        )
        or len({source["path"] for source in sources}) != 12
        or len({source["sha256"] for source in sources}) != 12
    ):
        raise ValueError("pilot analysis must bind 12 distinct source manifests")
    for index, source in enumerate(sources):
        _strict_sha256(
            source["sha256"], location=f"pilot source_manifests[{index}].sha256"
        )
    created_utc = analysis.get("created_utc")
    if not isinstance(created_utc, str):
        raise ValueError("pilot analysis created_utc must be an ISO-8601 string")
    try:
        timestamp = datetime.fromisoformat(created_utc)
    except ValueError as error:
        raise ValueError("pilot analysis created_utc is not valid ISO-8601") from error
    if timestamp.tzinfo is None:
        raise ValueError("pilot analysis created_utc must include a timezone")

    # This reloads the exact 12 expected pilot cells from the immutable lock,
    # rehashes every manifest and summary, checks cell IDs/seeds/code sources,
    # and recomputes both the recovery criteria and runtime threshold.  Only
    # the wall-clock creation stamp is intentionally excluded from equality.
    recomputed = analyze_synthetic_posterior(binding["path"], mode="pilot")
    if _canonical_json(_without_created_utc(analysis)) != _canonical_json(
        _without_created_utc(recomputed)
    ):
        raise ValueError(
            "pilot analysis differs from strict recomputation of its 12 cells"
        )
    gate = analysis.get("pilot_gate")
    if not isinstance(gate, dict):
        raise ValueError("pilot analysis is missing the predeclared gate")
    runtime = gate.get("observed_maximum_runtime_seconds")
    if (
        gate.get("passed") is not True
        or gate.get("reasons") != []
        or isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(runtime)
        or runtime > 10_800
    ):
        raise ValueError("synthetic pilot gate did not pass")
    return analysis_path.resolve(), expected


def _validated_receipt(*, phase, submit, receipt):
    if phase != "full":
        return receipt
    if not submit:
        if receipt is not None:
            raise ValueError("--receipt is valid only together with --submit")
        return None
    if receipt is None:
        raise ValueError(f"full submission requires the fixed receipt {FULL_RECEIPT}")
    supplied = Path(receipt)
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    expected = REPO_ROOT / FULL_RECEIPT
    if supplied.resolve() != expected.resolve():
        raise ValueError(f"full submission must reuse the fixed receipt {FULL_RECEIPT}")
    return expected.resolve()


def plan_synthetic_jobs(
    lock=DEFAULT_LOCK,
    *,
    phase="pilot",
    pilot_analysis=None,
    expected_pilot_analysis_sha256=None,
):
    binding = load_synthetic_lock(lock)
    if phase == "full":
        validate_pilot_gate(binding, pilot_analysis, expected_pilot_analysis_sha256)
    elif pilot_analysis is not None or expected_pilot_analysis_sha256 is not None:
        raise ValueError("pilot phase does not accept full-expansion gate inputs")
    jobs = []
    cells = selected_cells(binding["spec"], phase)
    for index, cell in enumerate(cells):
        partition = CPU_PARTITIONS[index % len(CPU_PARTITIONS)]
        seed = cell_seeds(binding["spec"], cell)["cell_seed"]
        command = slurm_command(
            job_name=(f"selseg-syn-{cell.coupling[:8]}-{cell.sharpness[:3]}-"
                      f"{cell.morphology[:5]}-r{cell.replicate:02d}"),
            partition=partition,
            cpus=4,
            memory="16g",
            time_limit="04:00:00",
            argv=("python", "-m", "selectseg.studies.synthetic",
            "--lock",
            str(binding["path"]),
            "--expected-lock-sha256",
            binding["sha256"],
            "--phase",
            phase,
            "--coupling",
            cell.coupling,
            "--sharpness",
            cell.sharpness,
            "--morphology",
            cell.morphology,
            "--replicate",
            str(cell.replicate),
            "--expected-cell-seed",
            str(seed),
            ),
        )
        jobs.append(
            PlannedJob(
                phase=f"synthetic_{phase}",
                key=(*cell.key, seed, partition),
                command=command,
            )
        )
    expected = 12 if phase == "pilot" else 348
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError(
            f"synthetic {phase} plan is not one-to-one with {expected} cells"
        )
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    if args.study == "matrix":
        if args.pilot_analysis or args.expected_pilot_analysis_sha256:
            raise ValueError("matrix submission does not use the v1 pilot gate")
        jobs = plan_matrix_jobs(args.config, phase=args.phase)
        receipt = args.receipt
        if args.submit and receipt is None:
            receipt = (
                "outputs/synthetic_posterior_matrix_v1/"
                f"{args.phase}-submissions.jsonl"
            )
        execute_plan(jobs, submit=args.submit, receipt_path=receipt)
        return
    jobs = plan_synthetic_jobs(
        Path(args.lock),
        phase=args.phase,
        pilot_analysis=args.pilot_analysis,
        expected_pilot_analysis_sha256=args.expected_pilot_analysis_sha256,
    )
    receipt = _validated_receipt(
        phase=args.phase,
        submit=args.submit,
        receipt=args.receipt,
    )
    return execute_plan(jobs, submit=args.submit, receipt_path=receipt)


if __name__ == "__main__":
    main()
