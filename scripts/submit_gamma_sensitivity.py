"""Plan one CPU job per locked ``(condition, gamma)`` sensitivity experiment.

The immutable auxiliary lock binds the canonical campaign lock and all 16
frozen artifact manifests.  The planner performs no discovery, uses no arrays
or GPUs, and is a dry run unless ``--submit`` is explicitly provided.  For the
frozen gamma set ``{0.3, 0.7}``, the canonical plan contains exactly 32 jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.submit_binary_simulations import (
    GPU_ACCOUNT,
    PlannedJob,
    execute_plan,
)
from selectseg.score_binary_gamma_sensitivity import load_auxiliary_lock


DEFAULT_AUXILIARY_LOCK = (
    "configs/auxiliary/binary_gamma_sensitivity-v1.lock.json"
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


def plan_gamma_sensitivity_jobs(auxiliary_lock=DEFAULT_AUXILIARY_LOCK):
    """Return one independent CPU job per predeclared condition and gamma."""

    binding = load_auxiliary_lock(auxiliary_lock)
    lock = binding["data"]
    jobs = []
    row_index = 0
    canonical_by_path = {
        path: entry
        for path, entry in zip(
            (path for path, _ in binding["artifacts"]),
            binding["campaign"]["artifacts"],
            strict=True,
        )
    }
    for artifact_path, artifact_sha256 in binding["artifacts"]:
        canonical = canonical_by_path[artifact_path]
        for gamma in lock["protocol"]["gamma_values"]:
            partition = lock["cpu_partitions"][
                row_index % len(lock["cpu_partitions"])
            ]
            row_index += 1
            gamma_tag = str(gamma).replace(".", "p")
            job_name = (
                f"selseg-gamma-{gamma_tag}-{canonical['dataset']}-"
                f"{canonical['condition']}"
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
                "scripts/slurm/score_binary_gamma_sensitivity.sbatch",
                "--auxiliary-lock",
                str(binding["path"]),
                "--expected-auxiliary-lock-sha256",
                binding["sha256"],
                "--artifact-manifest",
                str(artifact_path),
                "--expected-artifact-manifest-sha256",
                artifact_sha256,
                "--gamma",
                str(gamma),
            )
            jobs.append(
                PlannedJob(
                    phase="gamma_sensitivity",
                    key=(
                        canonical["dataset"],
                        canonical["condition"],
                        float(gamma),
                        partition,
                    ),
                    command=command,
                )
            )
    expected = len(binding["artifacts"]) * len(lock["protocol"]["gamma_values"])
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError("gamma-sensitivity plan is not one-to-one with its grid")
    return tuple(jobs)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_gamma_sensitivity_jobs(Path(args.auxiliary_lock))
    return execute_plan(
        jobs,
        submit=args.submit,
        receipt_path=args.receipt,
    )


if __name__ == "__main__":
    main()
