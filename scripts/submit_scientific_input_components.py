"""Plan or submit one Slurm job per eval-dataset scientific manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.submit_binary_simulations import (
    CPU_PARTITION_CANDIDATES,
    GPU_ACCOUNT,
    PlannedJob,
    execute_plan,
    preflight_plan,
)
from selectseg.scientific_inputs import EVAL_DATASETS


DEFAULT_OUTPUT_DIR = Path(
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets"
)
DEFAULT_RECEIPT = Path(
    "outputs/binary_midpoint_main_v2/scientific_inputs/"
    "dataset-build-receipt.jsonl"
)


def plan_jobs(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    partition_request = ",".join(CPU_PARTITION_CANDIDATES)
    jobs = []
    for dataset in EVAL_DATASETS:
        output = output_dir / f"{dataset}.json"
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite dataset component: {output}")
        command = (
            "sbatch",
            "--parsable",
            "--job-name",
            f"selseg-science-data-{dataset}",
            "--partition",
            partition_request,
            "--account",
            GPU_ACCOUNT,
            "scripts/slurm/build_scientific_dataset.sbatch",
            dataset,
            output.as_posix(),
        )
        jobs.append(
            PlannedJob(
                phase="scientific-dataset",
                key=(dataset, partition_request),
                command=command,
            )
        )
    if len(jobs) != len(EVAL_DATASETS) or len({job.key for job in jobs}) != len(
        jobs
    ):
        raise RuntimeError("dataset component plan is not one-to-one")
    return tuple(jobs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--submit", action="store_true")
    execution.add_argument("--scheduler-preflight-only", action="store_true")
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    jobs = plan_jobs(args.output_dir)
    if args.scheduler_preflight_only:
        preflight_plan(jobs)
        print(
            f"planned_jobs={len(jobs)} scheduler_preflight_jobs={len(jobs)} "
            "submitted_jobs=0"
        )
        return ()
    return execute_plan(
        jobs,
        submit=args.submit,
        receipt_path=(args.receipt if args.submit else None),
    )


if __name__ == "__main__":
    main()
