"""Plan the immutable four-method runtime ladder, one condition per CPU job.

The v2 lock binds the untouched main campaign, midpoint estimator, benchmark
specification, and executable source bytes.  The planner performs no output
discovery and emits exactly sixteen jobs.  Submission remains opt-in and uses
the shared append-only receipt state machine.
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
    EXPECTED_LADDER_SPEC_ID,
    _resolve_project_path,
    _validate_benchmark_spec,
    load_runtime_ladder_lock,
)
from selectseg.threshold_estimators import sha256_file


CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
DEFAULT_RUNTIME_LOCK = "configs/auxiliary/binary_runtime_ladder-v2.lock.json"
# Filled only after every lock-bound source file reaches its final bytes.
DEFAULT_RUNTIME_LOCK_SHA256 = (
    "3737c3751fd368f7abf55561493ea2eacbcd3ac788db72925a54cd3d7cdf9b33"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-lock", default=DEFAULT_RUNTIME_LOCK)
    parser.add_argument("--expected-runtime-lock-sha256")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="call sbatch; otherwise print the exact sixteen-job plan",
    )
    parser.add_argument(
        "--receipt",
        help="append-only submission receipt; required only with --submit",
    )
    return parser.parse_args(argv)


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
    """Return exactly one four-method benchmark job per locked condition."""

    expected_sha = _expected_lock_sha(
        runtime_lock, expected_runtime_lock_sha256
    )
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
        partition = CPU_PARTITIONS[index % len(CPU_PARTITIONS)]
        job_name = (
            f"selseg-runtime-v2-{artifact['dataset']}-{artifact['condition']}"
        )
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
    if (
        len(jobs) != EXPECTED_ARTIFACT_COUNT
        or len({job.key for job in jobs}) != EXPECTED_ARTIFACT_COUNT
    ):
        raise RuntimeError("runtime ladder plan is not one-to-one with artifacts")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_runtime_ladder_jobs(
        args.runtime_lock,
        expected_runtime_lock_sha256=args.expected_runtime_lock_sha256,
    )
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
