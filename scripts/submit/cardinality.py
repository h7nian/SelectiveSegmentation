"""Plan one exact-cardinality diagnostic job per locked frozen condition.

The immutable auxiliary lock projects all sixteen canonical frozen artifacts.
The planner performs no discovery, uses no arrays, and emits one CPU job per
condition.  Dry-run is the default; ``--submit`` uses the shared append-only
submission receipt protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.submit.main import PlannedJob, execute_plan, slurm_command
from selectseg.studies.cardinality import load_auxiliary_lock


DEFAULT_AUXILIARY_LOCK = (
    "configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary-lock", default=DEFAULT_AUXILIARY_LOCK)
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


def plan_cardinality_jobs(auxiliary_lock=DEFAULT_AUXILIARY_LOCK):
    """Return exactly one CPU job per artifact in the immutable lock."""

    binding = load_auxiliary_lock(auxiliary_lock)
    jobs = []
    for index, ((artifact_path, artifact_sha256), canonical) in enumerate(
        zip(binding.artifacts, binding.campaign["artifacts"], strict=True)
    ):
        partition = binding.data["cpu_partitions"][
            index % len(binding.data["cpu_partitions"])
        ]
        job_name = (
            f"selseg-cardinality-{canonical['dataset']}-{canonical['condition']}"
        )
        command = slurm_command(
            job_name=job_name,
            partition=partition,
            cpus=1,
            memory="16g",
            time_limit="04:00:00",
            argv=("python", "-m", "selectseg.studies.cardinality",
            "--auxiliary-lock",
            str(binding.path),
            "--expected-auxiliary-lock-sha256",
            binding.sha256,
            "--artifact-manifest",
            str(artifact_path),
            "--expected-artifact-manifest-sha256",
            artifact_sha256),
        )
        jobs.append(
            PlannedJob(
                phase="cardinality_diagnostics",
                key=(canonical["dataset"], canonical["condition"], partition),
                command=command,
            )
        )
    expected = len(binding.artifacts)
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError("cardinality plan is not one-to-one with locked artifacts")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_cardinality_jobs(Path(args.auxiliary_lock))
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
