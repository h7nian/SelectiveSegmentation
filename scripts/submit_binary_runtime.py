"""Plan one CPU confidence-runtime job for each locked frozen artifact.

The planner performs no directory discovery and emits exactly sixteen jobs,
one for every artifact named by the immutable main campaign lock.  Dry-run is
the default; submission uses the shared append-only receipt protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.submit_binary_simulations import (
    GPU_ACCOUNT,
    PlannedJob,
    _project_path,
    execute_plan,
    load_campaign_lock,
)
from selectseg.benchmark_binary_runtime import (
    EXPECTED_ARTIFACT_COUNT,
    _validate_benchmark_spec,
)
from selectseg.threshold_estimators import sha256_file


CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
DEFAULT_CAMPAIGN_LOCK = "outputs/binary_campaign/campaign.lock.json"
DEFAULT_BENCHMARK_SPEC = "configs/auxiliary/binary_runtime-v1.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-lock", default=DEFAULT_CAMPAIGN_LOCK)
    parser.add_argument("--benchmark-spec", default=DEFAULT_BENCHMARK_SPEC)
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


def plan_runtime_jobs(
    campaign_lock=DEFAULT_CAMPAIGN_LOCK,
    benchmark_spec=DEFAULT_BENCHMARK_SPEC,
):
    """Return exactly one four-hour CPU benchmark per locked condition."""

    lock_path, lock_sha256, lock = load_campaign_lock(campaign_lock)
    if len(lock["artifacts"]) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError("runtime plan requires exactly 16 locked artifacts")
    spec_path = Path(benchmark_spec).resolve()
    spec_sha256 = sha256_file(spec_path)
    spec, observed_spec_sha256 = _validate_benchmark_spec(spec_path, spec_sha256)
    if observed_spec_sha256 != spec_sha256:
        raise AssertionError("benchmark spec hash changed during planning")
    estimator = lock["estimator"]
    estimator_path = _project_path(lock_path, estimator["spec_path"])

    jobs = []
    for index, artifact in enumerate(lock["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = CPU_PARTITIONS[index % len(CPU_PARTITIONS)]
        job_name = f"selseg-runtime-{artifact['dataset']}-{artifact['condition']}"
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            job_name[:128],
            "--partition",
            partition,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/benchmark_binary_runtime.sbatch",
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
            "--benchmark-spec",
            str(spec_path),
            "--expected-benchmark-spec-sha256",
            spec_sha256,
        )
        jobs.append(
            PlannedJob(
                phase="binary_runtime",
                key=(
                    artifact["dataset"],
                    artifact["condition"],
                    partition,
                    spec["benchmark_id"],
                ),
                command=command,
            )
        )
    if len(jobs) != EXPECTED_ARTIFACT_COUNT or len({job.key for job in jobs}) != len(jobs):
        raise RuntimeError("runtime plan is not one-to-one with the locked artifacts")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_runtime_jobs(args.campaign_lock, args.benchmark_spec)
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
