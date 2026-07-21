"""Plan one CPU-only M=128 auxiliary job per locked model condition.

The planner validates the immutable main campaign lock and every lock-listed
frozen manifest before emitting commands.  It never uses directory discovery,
Slurm arrays, GPUs, or one job for multiple conditions.  Dry-run is the
default; ``--submit`` reuses the append-only receipt protocol of the canonical
planner and therefore requires an explicit ``--receipt`` path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.submit.main import (
    PlannedJob,
    _project_path,
    execute_plan,
    load_campaign_lock,
    slurm_command,
)


CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
DEFAULT_CAMPAIGN_LOCK = "outputs/binary_campaign/campaign.lock.json"
DEFAULT_OUTPUT_ROOT = "outputs/binary_m128_auxiliary"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", default=DEFAULT_CAMPAIGN_LOCK)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--include-m32-diagnostics",
        action="store_true",
        help="recompute M=32 and retain per-image M128-minus-M32 diagnostics",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="call sbatch; otherwise print the exact dry-run plan",
    )
    parser.add_argument(
        "--receipt",
        help="append-only submission receipt; required only with --submit",
    )
    return parser.parse_args(argv)


def plan_m128_jobs(
    campaign_lock,
    *,
    output_root=DEFAULT_OUTPUT_ROOT,
    include_m32_diagnostics=False,
):
    """Return exactly one CPU-only job for each artifact in ``campaign_lock``."""

    lock_path, lock_sha256, lock = load_campaign_lock(campaign_lock)
    gamma_values = lock["protocol"]["gamma_values"]
    if len(gamma_values) != 1:
        raise ValueError("M=128 auxiliary planning requires one locked gamma")
    gamma = gamma_values[0]
    estimator = lock["estimator"]
    estimator_path = _project_path(lock_path, estimator["spec_path"])
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output_root must be a non-empty path string")

    jobs = []
    for artifact_index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = CPU_PARTITIONS[artifact_index % len(CPU_PARTITIONS)]
        diagnostic_tag = "-with-m32" if include_m32_diagnostics else ""
        job_name = (
            f"selseg-m128-{artifact['dataset']}-{artifact['condition']}{diagnostic_tag}"
        )
        argv = [
            "python", "-m", "selectseg.studies.m128",
            "--score-workers", "8", "--max-pending-scores", "16",
            "--campaign-id",
            lock["campaign_id"],
            "--campaign-lock",
            str(lock_path),
            "--expected-campaign-lock-sha256",
            lock_sha256,
            "--artifact-manifest",
            str(artifact_path),
            "--expected-artifact-manifest-sha256",
            artifact["manifest_sha256"],
            "--estimator-spec",
            str(estimator_path),
            "--expected-estimator-spec-sha256",
            estimator["spec_sha256"],
            "--gamma",
            str(gamma),
            "--output-root",
            output_root,
        ]
        if include_m32_diagnostics:
            argv.append("--include-m32-diagnostics")
        command = slurm_command(
            job_name=job_name,
            partition=partition,
            cpus=8,
            memory="24g",
            time_limit="12:00:00",
            argv=argv,
        )
        key = (
            artifact["dataset"],
            artifact["condition"],
            partition,
            gamma,
            128,
            bool(include_m32_diagnostics),
        )
        jobs.append(
            PlannedJob(
                phase="m128_auxiliary",
                key=key,
                command=command,
            )
        )

    expected_count = len(lock["artifacts"])
    if len(jobs) != expected_count or len({job.key for job in jobs}) != expected_count:
        raise RuntimeError("M=128 plan is not one-to-one with locked artifacts")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_m128_jobs(
        Path(args.campaign_lock),
        output_root=args.output_root,
        include_m32_diagnostics=args.include_m32_diagnostics,
    )
    return execute_plan(
        jobs,
        submit=args.submit,
        receipt_path=args.receipt,
    )


if __name__ == "__main__":
    main()
