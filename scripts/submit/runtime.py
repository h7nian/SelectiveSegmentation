"""Plan one CPU runtime job per condition for either benchmark protocol.

The planner performs no directory discovery and emits exactly sixteen jobs,
one for every artifact named by the immutable main campaign lock.  Dry-run is
the default; submission uses the shared append-only receipt protocol.
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
from selectseg.studies.runtime import (
    EXPECTED_ARTIFACT_COUNT,
    EXPECTED_LADDER_SPEC_ID,
    _resolve_project_path,
    _validate_benchmark_spec,
    load_runtime_ladder_lock,
)
from selectseg.quadrature import sha256_file


CPU_PARTITIONS = ("amdsmall", "agsmall", "msismall", "saffo-2tb")
DEFAULT_CAMPAIGN_LOCK = "outputs/binary_campaign/campaign.lock.json"
DEFAULT_BENCHMARK_SPEC = "configs/auxiliary/binary_runtime-v1.json"
DEFAULT_RUNTIME_LOCK = "configs/auxiliary/binary_runtime_ladder-v2.lock.json"
DEFAULT_RUNTIME_LOCK_SHA256 = (
    "944e14889e3ae501e756410ece66e00e6e4737c926adaced88b51d634ed2f365"
)


def _partition_request(index: int) -> str:
    offset = index % len(CPU_PARTITIONS)
    rotated = CPU_PARTITIONS[offset:] + CPU_PARTITIONS[:offset]
    return ",".join(rotated)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("basic", "ladder"), default="basic")
    parser.add_argument("--campaign-lock", default=DEFAULT_CAMPAIGN_LOCK)
    parser.add_argument("--benchmark-spec", default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--runtime-lock", default=DEFAULT_RUNTIME_LOCK)
    parser.add_argument("--expected-runtime-lock-sha256")
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
        partition = _partition_request(index)
        job_name = f"selseg-runtime-{artifact['dataset']}-{artifact['condition']}"
        command = slurm_command(
            job_name=job_name,
            partition=partition,
            cpus=8,
            memory="24g",
            time_limit="04:00:00",
            argv=(
                "env",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "python",
                "-m",
                "selectseg.studies.runtime",
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
            ),
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
    if len(jobs) != EXPECTED_ARTIFACT_COUNT or len({job.key for job in jobs}) != len(
        jobs
    ):
        raise RuntimeError("runtime plan is not one-to-one with the locked artifacts")
    return tuple(jobs)


def _expected_lock_sha(path, explicit):
    if explicit:
        return explicit
    if Path(path).as_posix() == DEFAULT_RUNTIME_LOCK:
        return DEFAULT_RUNTIME_LOCK_SHA256
    raise ValueError("a non-default runtime lock requires its expected SHA-256")


def plan_runtime_ladder_jobs(
    runtime_lock=DEFAULT_RUNTIME_LOCK,
    *,
    expected_runtime_lock_sha256=None,
):
    """Return one four-method benchmark job per lock-listed condition."""

    expected_sha = _expected_lock_sha(runtime_lock, expected_runtime_lock_sha256)
    binding = load_runtime_ladder_lock(runtime_lock, expected_sha256=expected_sha)
    locked = binding["lock"]
    spec_path = _resolve_project_path(locked["spec"]["path"])
    spec_sha = locked["spec"]["sha256"]
    spec, observed_spec_sha = _validate_benchmark_spec(spec_path, spec_sha)
    if observed_spec_sha != spec_sha or spec["benchmark_id"] != EXPECTED_LADDER_SPEC_ID:
        raise ValueError("runtime ladder lock does not bind the v2 specification")

    campaign_binding = locked["canonical_campaign_lock"]
    lock_path = _resolve_project_path(campaign_binding["path"])
    lock_path, campaign_sha, campaign = load_campaign_lock(lock_path)
    if (
        campaign_sha != campaign_binding["sha256"]
        or campaign["campaign_id"] != campaign_binding["campaign_id"]
    ):
        raise ValueError("runtime ladder lock and canonical campaign disagree")
    if len(campaign["artifacts"]) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError("runtime ladder requires exactly 16 locked artifacts")
    estimator = campaign["estimator"]
    estimator_path = _project_path(lock_path, estimator["spec_path"])
    estimator_binding = locked["estimator_spec"]
    if (
        estimator_path != _resolve_project_path(estimator_binding["path"])
        or estimator["spec_sha256"] != estimator_binding["sha256"]
        or sha256_file(estimator_path) != estimator_binding["sha256"]
    ):
        raise ValueError("runtime ladder lock and estimator binding disagree")

    jobs = []
    for index, artifact in enumerate(campaign["artifacts"]):
        artifact_path = _project_path(lock_path, artifact["manifest_path"])
        partition = _partition_request(index)
        command = slurm_command(
            job_name=f"selseg-runtime-v2-{artifact['dataset']}-{artifact['condition']}",
            partition=partition,
            cpus=8,
            memory="24g",
            time_limit="04:00:00",
            argv=(
                "env",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "python",
                "-m",
                "selectseg.studies.runtime",
                "--campaign-id",
                campaign["campaign_id"],
                "--campaign-lock",
                str(lock_path),
                "--expected-campaign-lock-sha256",
                campaign_sha,
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
                spec_sha,
                "--benchmark-lock",
                str(binding["path"]),
                "--expected-benchmark-lock-sha256",
                binding["sha256"],
            ),
        )
        jobs.append(
            PlannedJob(
                phase="binary_runtime_ladder_v2",
                key=(
                    artifact["dataset"],
                    artifact["condition"],
                    partition,
                    spec["benchmark_id"],
                ),
                command=command,
            )
        )
    if len(jobs) != EXPECTED_ARTIFACT_COUNT or len({job.key for job in jobs}) != len(
        jobs
    ):
        raise RuntimeError("runtime ladder plan is not one-to-one with artifacts")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "ladder":
        jobs = plan_runtime_ladder_jobs(
            args.runtime_lock,
            expected_runtime_lock_sha256=args.expected_runtime_lock_sha256,
        )
    else:
        jobs = plan_runtime_jobs(args.campaign_lock, args.benchmark_spec)
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
