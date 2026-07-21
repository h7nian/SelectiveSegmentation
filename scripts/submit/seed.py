"""Plan the isolated target-model training-seed extension.

The immutable grid is five datasets by two target architectures by seeds 1
and 2.  ``train`` therefore plans exactly 20 independent GPU jobs.  After all
20 no-overwrite training records exist, ``checkpoint-lock`` validates their
final-epoch checkpoints and writes one immutable lock.  Only then can
``freeze`` plan exactly 20 independent GPU inference jobs.  Once all freeze
records validate, ``downstream-lock`` creates one canonical-compatible
campaign for each training seed and one parent lock binding both campaigns.
The remaining compute phases delegate to the unchanged canonical common,
score, assembly, and diagnostic planners; one strict descriptive analysis and
one display-only renderer close the auxiliary workflow.

Submission is opt-in.  Before the first real submission, both locked GPU
profiles are checked with ``sbatch --test-only``; failure aborts the entire
wave before any job is sent.  Slurm arrays are never used.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from scripts.submit.main import (
    PlannedJob,
    execute_plan,
    plan_assemble_jobs,
    plan_common_jobs,
    plan_diagnose_jobs,
    plan_score_jobs,
    slurm_command,
)
from selectseg.seed.downstream import (
    load_downstream_lock,
    write_downstream_lock,
)
from selectseg.seed.extension import (
    DEFAULT_SPEC_LOCK,
    DEFAULT_SPEC_LOCK_SHA256,
    iter_experiments,
    load_checkpoint_lock,
    load_spec_lock,
    write_checkpoint_lock,
)


TRAIN_RECEIPT = Path("outputs/binary_seed_extension_campaign/train-submissions.jsonl")
DOWNSTREAM_RECEIPT_NAMES = {
    "freeze": "freeze-submissions.jsonl",
    "common": "common-submissions.jsonl",
    "score": "score-submissions.jsonl",
    "assemble": "assemble-submissions.jsonl",
    "diagnose": "diagnose-submissions.jsonl",
    "analyze": "analyze-submissions.jsonl",
    "render": "render-submissions.jsonl",
}

# Slurm accepts a comma-delimited partition request and may canonicalize its
# display before dispatching the one job to the eligible partition expected to
# start earliest.  The order below is fixed serialization for receipt bytes,
# not a queue-priority claim.
# This policy is deliberately limited to phases whose receipts do not yet
# exist.  The completed v1 train/freeze and common/score/diagnose receipts
# retain their original single-partition commands byte-for-byte.
CPU_PARTITION_CANDIDATES = ("saffo-2tb", "agsmall", "amdsmall", "msismall")
CPU_PARTITION_REQUEST = ",".join(CPU_PARTITION_CANDIDATES)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "train",
            "checkpoint-lock",
            "freeze",
            "downstream-lock",
            "common",
            "score",
            "assemble",
            "diagnose",
            "analyze",
            "render",
        ),
        default="train",
    )
    parser.add_argument("--spec-lock", default=DEFAULT_SPEC_LOCK)
    parser.add_argument("--expected-spec-lock-sha256")
    parser.add_argument(
        "--expected-scheduler-summary-sha256",
        help=(
            "required by checkpoint-lock; binds the fixed complete training "
            "scheduler closure reviewed after finalization"
        ),
    )
    parser.add_argument("--checkpoint-lock")
    parser.add_argument("--expected-checkpoint-lock-sha256")
    parser.add_argument("--downstream-lock")
    parser.add_argument("--expected-downstream-lock-sha256")
    parser.add_argument(
        "--write-checkpoint-lock",
        action="store_true",
        help="required to publish the validated post-training checkpoint lock",
    )
    parser.add_argument(
        "--write-downstream-lock",
        action="store_true",
        help="required to publish both validated seed-specific campaign locks",
    )
    parser.add_argument(
        "--canonical-analysis",
        default="outputs/binary_final_v3_analysis/analysis.json",
        help="locked seed-0 analysis consumed only by the analyze phase",
    )
    parser.add_argument("--expected-canonical-analysis-sha256")
    parser.add_argument(
        "--seed-analysis",
        help="completed seed analysis consumed only by the render phase",
    )
    parser.add_argument("--expected-seed-analysis-sha256")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="run scheduler preflight and call sbatch; otherwise print a dry run",
    )
    parser.add_argument(
        "--receipt",
        help="append-only submission receipt; required with --submit",
    )
    return parser.parse_args(argv)


def _expected_spec_sha(path, explicit):
    if explicit:
        return explicit
    if Path(path).as_posix() == DEFAULT_SPEC_LOCK:
        return DEFAULT_SPEC_LOCK_SHA256
    raise ValueError("a non-default spec lock requires its expected SHA-256")


def plan_training_jobs(binding):
    """Return the exact 20-cell training plan with a balanced 10/10 split."""

    jobs = []
    for experiment in iter_experiments(binding["spec"]):
        dataset = experiment["dataset"]["name"]
        model = experiment["model"]["name"]
        seed = experiment["training_seed"]
        profile = experiment["gpu_profile"]
        job_name = f"selseg-seed-train-{dataset}-{model}-s{seed}"
        command = slurm_command(
            job_name=job_name,
            partition=profile["partition"],
            account=profile["account"],
            gres=profile["gres"],
            cpus=16,
            memory="64g",
            time_limit="24:00:00",
            requeue=False,
            argv=("python", "-m", "selectseg.seed.extension",
            "train",
            "--spec-lock",
            binding["path"].as_posix(),
            "--expected-spec-lock-sha256",
            binding["sha256"],
            "--dataset",
            dataset,
            "--model",
            model,
            "--training-seed",
            str(seed),
            "--expected-partition",
            profile["partition"],),
        )
        jobs.append(
            PlannedJob(
                phase="seed_train",
                key=(dataset, model, seed, profile["partition"]),
                command=command,
            )
        )
    _validate_gpu_plan(jobs, expected_phase="seed_train")
    return tuple(jobs)


def plan_freeze_jobs(binding, checkpoint_binding):
    """Return 20 freeze jobs only after the complete checkpoint lock validates."""

    jobs = []
    for experiment in iter_experiments(binding["spec"]):
        dataset = experiment["dataset"]["name"]
        model = experiment["model"]["name"]
        seed = experiment["training_seed"]
        profile = experiment["gpu_profile"]
        job_name = f"selseg-seed-freeze-{dataset}-{model}-s{seed}"
        command = slurm_command(
            job_name=job_name,
            partition=profile["partition"],
            account=profile["account"],
            gres=profile["gres"],
            cpus=8,
            memory="48g",
            time_limit="12:00:00",
            requeue=False,
            argv=("python", "-m", "selectseg.seed.extension",
            "freeze",
            "--spec-lock",
            binding["path"].as_posix(),
            "--expected-spec-lock-sha256",
            binding["sha256"],
            "--checkpoint-lock",
            checkpoint_binding["path"].as_posix(),
            "--expected-checkpoint-lock-sha256",
            checkpoint_binding["sha256"],
            "--dataset",
            dataset,
            "--model",
            model,
            "--training-seed",
            str(seed),
            "--expected-partition",
            profile["partition"],),
        )
        jobs.append(
            PlannedJob(
                phase="seed_freeze",
                key=(dataset, model, seed, profile["partition"]),
                command=command,
            )
        )
    _validate_gpu_plan(jobs, expected_phase="seed_freeze")
    return tuple(jobs)


def _validate_gpu_plan(jobs, *, expected_phase):
    if len(jobs) != 20 or len({job.key for job in jobs}) != 20:
        raise RuntimeError(f"{expected_phase} must contain exactly 20 unique jobs")
    if {job.phase for job in jobs} != {expected_phase}:
        raise RuntimeError("GPU plan contains the wrong phase")
    partitions = Counter(job.key[-1] for job in jobs)
    if partitions != {"saffo-a100": 10, "apollo_agate": 10}:
        raise RuntimeError(
            f"GPU plan is not balanced across private queues: {partitions}"
        )
    for job in jobs:
        command = job.command
        if "--array" in command or any(
            token.startswith("--array=") for token in command
        ):
            raise RuntimeError("Slurm arrays are forbidden")
        if command.count("--partition") != 1 or command.count("--gres") != 1:
            raise RuntimeError("each GPU experiment needs one explicit profile")


def downstream_job_design(binding):
    """Describe the post-freeze CPU graph without changing primary validators."""

    design = binding["spec"]["downstream_job_design"]
    cells = [
        (
            experiment["dataset"]["name"],
            experiment["model"]["condition"],
            experiment["training_seed"],
        )
        for experiment in iter_experiments(binding["spec"])
    ]
    return {
        "cells": cells,
        "common": {
            "jobs": design["common_jobs"],
            "unit": "one frozen seed artifact",
            "output_root": binding["spec"]["paths"]["common_root"],
        },
        "m_score": {
            "jobs": design["m_score_jobs"],
            "unit": "one frozen seed artifact x one M in {2,8,32}",
            "output_root": binding["spec"]["paths"]["simulation_root"],
        },
        "assemble": {
            "jobs": design["assembly_jobs"],
            "unit": "one seed condition after common + three M shards",
            "output_root": binding["spec"]["paths"]["assembly_root"],
        },
        "diagnose": {
            "jobs": design["diagnostic_jobs"],
            "unit": "one frozen seed artifact",
            "output_root": binding["spec"]["paths"]["diagnostic_root"],
        },
        "guard": (
            "Create a new seed-specific campaign lock and auxiliary validators; "
            "do not broaden the seed-0 campaign validator or merge rows into its analysis."
        ),
    }


def _retag_job(job, *, training_seed, phase):
    """Add training seed to receipt/job identity without changing computation."""

    command = list(job.command)
    job_name_index = command.index("--job-name") + 1
    command[job_name_index] = f"selseg-seed-s{training_seed}-{command[job_name_index]}"[
        :128
    ]
    return PlannedJob(
        phase=phase,
        key=(training_seed, *job.key),
        command=tuple(command),
    )


def _with_cpu_partition_candidates(job):
    """Retarget one unsubmitted CPU job without changing its experiment.

    The partition is operational metadata, but it is part of the private
    receipt identity.  Therefore this helper is used only while constructing
    future assembly/analyze/render plans, never to reconstruct the already
    submitted common, score, or diagnose plans.
    """

    command = list(job.command)
    if command.count("--partition") != 1:
        raise RuntimeError("candidate-partition job needs one --partition flag")
    partition_index = command.index("--partition") + 1
    original_partition = command[partition_index]
    if not isinstance(original_partition, str) or not original_partition:
        raise RuntimeError("candidate-partition job has an invalid partition")
    if not job.key or job.key[-1] != original_partition:
        raise RuntimeError("job key and command partition are inconsistent")
    command[partition_index] = CPU_PARTITION_REQUEST
    return PlannedJob(
        phase=job.phase,
        key=(*job.key[:-1], CPU_PARTITION_REQUEST),
        command=tuple(command),
    )


def plan_downstream_jobs(downstream_binding, phase):
    """Reuse canonical planners and return the exact seed-aware CPU wave."""

    planners = {
        "common": plan_common_jobs,
        "score": plan_score_jobs,
        "assemble": plan_assemble_jobs,
    }
    if phase not in {*planners, "diagnose"}:
        raise ValueError(f"unsupported seed downstream phase {phase!r}")
    jobs = []
    for campaign in downstream_binding["campaigns"]:
        seed = campaign["training_seed"]
        config = campaign["config"]
        lock_path = campaign["campaign_lock_path"]
        if phase == "diagnose":
            canonical_jobs = plan_diagnose_jobs(
                config,
                lock_path,
                output_root=downstream_binding["binding"]["spec"]["paths"][
                    "diagnostic_root"
                ],
            )
        else:
            canonical_jobs = planners[phase](config, lock_path)
        seed_jobs = tuple(
            _retag_job(
                job,
                training_seed=seed,
                phase=f"seed_{phase}",
            )
            for job in canonical_jobs
        )
        if phase == "assemble":
            seed_jobs = tuple(
                _with_cpu_partition_candidates(job) for job in seed_jobs
            )
        jobs.extend(seed_jobs)
    expected = {"common": 20, "score": 60, "assemble": 20, "diagnose": 20}[phase]
    if len(jobs) != expected or len({(job.phase, job.key) for job in jobs}) != expected:
        raise RuntimeError(
            f"seed {phase} plan must contain exactly {expected} unique jobs"
        )
    for job in jobs:
        if "--array" in job.command or any(
            token.startswith("--array=") for token in job.command
        ):
            raise RuntimeError("Slurm arrays are forbidden in the seed extension")
    _validate_downstream_job_isolation(jobs, phase=phase)
    return tuple(jobs)


def _validate_downstream_job_isolation(jobs, *, phase):
    """Prove that every CPU job carries exactly one experiment identity.

    Counting jobs is insufficient: one malformed command could still carry two
    artifact flags while another carried none.  Validate the singular flags at
    the final seed-aware plan boundary immediately before dry-run/submission.
    Assembly is the intentional exception: one condition consumes exactly one
    common shard and the three locked M-specific shards.
    """

    modules = {
        "common": "selectseg.pipeline.common",
        "score": "selectseg.pipeline.score",
        "assemble": "scripts.assemble",
        "diagnose": "scripts.diagnose",
    }
    expected_flags = {
        "common": {
            "--artifact-manifest": 1,
            "--expected-artifact-manifest-sha256": 1,
            "--campaign-lock": 1,
            "--gamma": 1,
            "--m": 0,
            "--seed": 0,
            "--common": 0,
            "--input": 0,
        },
        "score": {
            "--artifact-manifest": 1,
            "--expected-artifact-manifest-sha256": 1,
            "--campaign-lock": 1,
            "--gamma": 1,
            "--m": 1,
            "--seed": 1,
            "--common": 0,
            "--input": 0,
        },
        "assemble": {
            "--artifact-manifest": 0,
            "--expected-artifact-manifest-sha256": 0,
            "--campaign-lock": 1,
            "--gamma": 0,
            "--m": 0,
            "--seed": 0,
            "--common": 1,
            "--input": 3,
        },
        "diagnose": {
            "--artifact-manifest": 1,
            "--expected-artifact-manifest-sha256": 1,
            "--campaign-lock": 0,
            "--gamma": 0,
            "--m": 0,
            "--seed": 0,
            "--common": 0,
            "--input": 0,
        },
    }
    if phase not in modules:
        raise ValueError(f"unsupported seed downstream phase {phase!r}")
    job_names = set()
    for job in jobs:
        command = job.command
        if command[:2] != ("sbatch", "--parsable"):
            raise RuntimeError(f"seed {phase} job is not an explicit sbatch command")
        if command.count("scripts/slurm/run.sbatch") != 1:
            raise RuntimeError(f"seed {phase} job has the wrong Slurm runner")
        if command.count(modules[phase]) != 1:
            raise RuntimeError(f"seed {phase} job has the wrong Python module")
        for flag in ("--job-name", "--partition", "--account", "--output-root"):
            if command.count(flag) != 1:
                raise RuntimeError(f"seed {phase} job must contain one {flag}")
        job_name = command[command.index("--job-name") + 1]
        if job_name in job_names:
            raise RuntimeError(f"seed {phase} jobs have duplicate Slurm names")
        job_names.add(job_name)
        for flag, count in expected_flags[phase].items():
            if command.count(flag) != count:
                raise RuntimeError(
                    f"seed {phase} job must contain {count} occurrence(s) of {flag}"
                )
        partition = command[command.index("--partition") + 1]
        if phase == "assemble":
            if partition != CPU_PARTITION_REQUEST:
                raise RuntimeError(
                    "seed assembly jobs must request the exact CPU candidate pool"
                )
        elif partition == CPU_PARTITION_REQUEST:
            raise RuntimeError(
                f"submitted seed {phase} receipts must retain their legacy partition"
            )


def _expected_receipt_path(binding, phase):
    if phase == "train":
        return TRAIN_RECEIPT
    if phase not in DOWNSTREAM_RECEIPT_NAMES:
        raise ValueError(f"phase {phase!r} does not submit Slurm jobs")
    campaign_root = Path(binding["spec"]["paths"]["checkpoint_lock"]).parent
    return campaign_root / DOWNSTREAM_RECEIPT_NAMES[phase]


def _validate_receipt_argument(binding, *, phase, submit, receipt):
    if not submit:
        if receipt:
            raise ValueError("--receipt is valid only together with --submit")
        return
    if not receipt:
        raise ValueError("--submit requires the fixed append-only --receipt path")
    expected = _expected_receipt_path(binding, phase)
    if Path(receipt).resolve() != expected.resolve():
        raise ValueError(
            f"{phase} must reuse the fixed duplicate guard receipt {expected}"
        )


def plan_analysis_job(
    downstream_binding,
    *,
    canonical_analysis,
    expected_canonical_analysis_sha256,
):
    """Plan the one descriptive cross-seed analysis after all assemblies exist."""

    if not expected_canonical_analysis_sha256:
        raise ValueError("analyze requires --expected-canonical-analysis-sha256")
    # Import lazily: this validation reads all completed assemblies and must not
    # run while earlier score waves are merely being planned.
    from scripts.analyze.seed import validate_analysis_inputs

    validate_analysis_inputs(
        downstream_binding,
        canonical_analysis=canonical_analysis,
        expected_canonical_analysis_sha256=expected_canonical_analysis_sha256,
    )
    output = (
        Path(downstream_binding["binding"]["spec"]["paths"]["analysis_root"])
        / "analysis.json"
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"seed analysis output already exists: {output}")
    partition = CPU_PARTITION_REQUEST
    command = slurm_command(
        job_name="selseg-seed-analysis",
        partition=partition,
        cpus=4,
        memory="24g",
        time_limit="02:00:00",
        requeue=False,
        argv=("python", "-m", "scripts.analyze.seed",
        "--downstream-lock",
        downstream_binding["path"].as_posix(),
        "--expected-downstream-lock-sha256",
        downstream_binding["sha256"],
        "--canonical-analysis",
        str(canonical_analysis),
        "--expected-canonical-analysis-sha256",
        expected_canonical_analysis_sha256,
        "--output",
        output.as_posix(),),
    )
    return (
        PlannedJob(
            phase="seed_analyze",
            key=("all-seeds", partition),
            command=command,
        ),
    )


def plan_render_job(
    downstream_binding,
    *,
    seed_analysis=None,
    expected_seed_analysis_sha256,
):
    """Plan one renderer after strict validation of the completed analysis."""

    from scripts.render.seed import load_analysis

    expected_analysis_path = (
        Path(downstream_binding["binding"]["spec"]["paths"]["analysis_root"])
        / "analysis.json"
    )
    analysis_path = Path(seed_analysis or expected_analysis_path)
    if analysis_path.resolve() != expected_analysis_path.resolve():
        raise ValueError(
            f"render must consume the fixed analysis {expected_analysis_path}"
        )
    if not expected_seed_analysis_sha256:
        raise ValueError("render requires --expected-seed-analysis-sha256")
    analysis, _, _ = load_analysis(
        analysis_path, expected_sha256=expected_seed_analysis_sha256
    )
    provenance = analysis.get("provenance", {}).get("downstream_lock", {})
    if (
        provenance.get("sha256") != downstream_binding["sha256"]
        or Path(provenance.get("path", "")).resolve()
        != downstream_binding["path"].resolve()
    ):
        raise ValueError("seed analysis is bound to a different downstream lock")
    output = analysis_path.with_name("seed_robustness.tex")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"seed robustness table already exists: {output}")
    partition = CPU_PARTITION_REQUEST
    command = slurm_command(
        job_name="selseg-seed-render",
        partition=partition,
        cpus=1,
        memory="4g",
        time_limit="00:10:00",
        requeue=False,
        argv=("python", "-m", "scripts.render.seed",
        "--analysis",
        analysis_path.as_posix(),
        "--expected-analysis-sha256",
        expected_seed_analysis_sha256,
        "--output",
        output.as_posix(),),
    )
    return (
        PlannedJob(
            phase="seed_render",
            key=("seed-robustness-table", partition),
            command=command,
        ),
    )


def _scheduler_preflight(jobs):
    """Fail the whole wave before submission if either profile is ineligible."""

    representatives = {}
    for job in jobs:
        representatives.setdefault(job.key[-1], job)
    if set(representatives) != {"saffo-a100", "apollo_agate"}:
        raise RuntimeError("scheduler preflight requires both private partitions")
    for partition in ("saffo-a100", "apollo_agate"):
        command = list(representatives[partition].command)
        command.remove("--parsable")
        command.insert(1, "--test-only")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Slurm preflight failed for {partition}; no jobs submitted: {message}"
            )


def _cpu_candidate_preflight(jobs):
    """Check the exact multi-partition request before a future CPU wave."""

    jobs = tuple(jobs)
    if not jobs:
        raise RuntimeError("CPU candidate preflight received an empty plan")
    for job in jobs:
        command = job.command
        if command.count("--partition") != 1:
            raise RuntimeError("CPU candidate job needs one --partition flag")
        if command[command.index("--partition") + 1] != CPU_PARTITION_REQUEST:
            raise RuntimeError("CPU candidate job has the wrong partition pool")
    command = list(jobs[0].command)
    command.remove("--parsable")
    command.insert(1, "--test-only")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Slurm preflight failed for the CPU candidate pool; "
            f"no jobs submitted: {message}"
        )


def main(argv=None):
    args = parse_args(argv)
    expected_spec_sha = _expected_spec_sha(
        args.spec_lock, args.expected_spec_lock_sha256
    )
    binding = load_spec_lock(args.spec_lock, expected_sha256=expected_spec_sha)

    if args.phase == "checkpoint-lock":
        if args.submit:
            raise ValueError(
                "checkpoint-lock is a local validation phase, not a Slurm job"
            )
        if not args.write_checkpoint_lock:
            raise ValueError("checkpoint-lock requires --write-checkpoint-lock")
        if not args.expected_scheduler_summary_sha256:
            raise ValueError(
                "checkpoint-lock requires --expected-scheduler-summary-sha256"
            )
        if args.expected_checkpoint_lock_sha256:
            raise ValueError(
                "checkpoint-lock creates the checkpoint lock; "
                "omit --expected-checkpoint-lock-sha256"
            )
        if args.receipt:
            raise ValueError("checkpoint-lock does not submit; omit --receipt")
        if (
            args.downstream_lock
            or args.expected_downstream_lock_sha256
            or args.write_downstream_lock
            or args.expected_canonical_analysis_sha256
            or args.seed_analysis
            or args.expected_seed_analysis_sha256
        ):
            raise ValueError("checkpoint-lock received a downstream-only argument")
        destination = (
            args.checkpoint_lock or binding["spec"]["paths"]["checkpoint_lock"]
        )
        # Import only after this submitter is fully initialized.  The finalizer
        # accepts the already constructed plan, so this checkpoint-only gate
        # does not need to reconstruct or submit any job.
        from scripts.maintenance.finalize_seed import (
            validate_complete_training_closure,
        )

        closure = validate_complete_training_closure(
            binding,
            plan_training_jobs(binding),
            expected_public_summary_sha256=(args.expected_scheduler_summary_sha256),
        )
        return write_checkpoint_lock(
            binding,
            destination,
            expected_training_record_set_sha256=closure["training_record_set_sha256"],
        )

    if args.phase == "downstream-lock":
        if args.submit:
            raise ValueError("downstream-lock is local validation, not a Slurm job")
        if not args.write_downstream_lock:
            raise ValueError("downstream-lock requires --write-downstream-lock")
        if args.receipt:
            raise ValueError("downstream-lock does not submit; omit --receipt")
        if args.write_checkpoint_lock:
            raise ValueError("--write-checkpoint-lock applies only to checkpoint-lock")
        if args.expected_scheduler_summary_sha256:
            raise ValueError(
                "--expected-scheduler-summary-sha256 applies only to checkpoint-lock"
            )
        if not args.expected_checkpoint_lock_sha256:
            raise ValueError(
                "downstream-lock requires --expected-checkpoint-lock-sha256"
            )
        if args.expected_downstream_lock_sha256:
            raise ValueError("downstream-lock creates its lock; omit its expected hash")
        if args.expected_canonical_analysis_sha256:
            raise ValueError("downstream-lock does not consume canonical analysis")
        if args.seed_analysis or args.expected_seed_analysis_sha256:
            raise ValueError("downstream-lock does not consume seed analysis")
        checkpoint_path = (
            args.checkpoint_lock or binding["spec"]["paths"]["checkpoint_lock"]
        )
        checkpoint_binding = load_checkpoint_lock(
            binding,
            checkpoint_path,
            expected_sha256=args.expected_checkpoint_lock_sha256,
            verify_files=True,
        )
        destination = args.downstream_lock
        return write_downstream_lock(binding, checkpoint_binding, destination)

    if args.write_checkpoint_lock:
        raise ValueError("--write-checkpoint-lock applies only to checkpoint-lock")
    if args.expected_scheduler_summary_sha256:
        raise ValueError(
            "--expected-scheduler-summary-sha256 applies only to checkpoint-lock"
        )
    if args.write_downstream_lock:
        raise ValueError("--write-downstream-lock applies only to downstream-lock")
    if args.phase == "train":
        if (
            args.checkpoint_lock
            or args.expected_checkpoint_lock_sha256
            or args.downstream_lock
            or args.expected_downstream_lock_sha256
        ):
            raise ValueError("training does not consume a checkpoint lock")
        jobs = plan_training_jobs(binding)
    elif args.phase == "freeze":
        if args.downstream_lock or args.expected_downstream_lock_sha256:
            raise ValueError("freeze does not consume a downstream lock")
        if not args.expected_checkpoint_lock_sha256:
            raise ValueError("freeze requires --expected-checkpoint-lock-sha256")
        checkpoint_path = (
            args.checkpoint_lock or binding["spec"]["paths"]["checkpoint_lock"]
        )
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                "freeze is gated on the complete post-training checkpoint lock"
            )
        checkpoint_binding = load_checkpoint_lock(
            binding,
            checkpoint_path,
            expected_sha256=args.expected_checkpoint_lock_sha256,
            verify_files=True,
        )
        jobs = plan_freeze_jobs(binding, checkpoint_binding)

    else:
        if args.checkpoint_lock or args.expected_checkpoint_lock_sha256:
            raise ValueError(
                f"{args.phase} reads checkpoint provenance only through downstream lock"
            )
        if not args.expected_downstream_lock_sha256:
            raise ValueError(f"{args.phase} requires --expected-downstream-lock-sha256")
        downstream_path = args.downstream_lock or (
            Path(binding["spec"]["paths"]["checkpoint_lock"]).with_name(
                "downstream.lock.json"
            )
        )
        downstream_binding = load_downstream_lock(
            downstream_path,
            expected_sha256=args.expected_downstream_lock_sha256,
        )
        if downstream_binding["binding"]["sha256"] != binding["sha256"]:
            raise ValueError("downstream lock is bound to a different seed spec")
        if args.phase == "analyze":
            if args.seed_analysis or args.expected_seed_analysis_sha256:
                raise ValueError("analyze creates, rather than consumes, seed analysis")
            jobs = plan_analysis_job(
                downstream_binding,
                canonical_analysis=args.canonical_analysis,
                expected_canonical_analysis_sha256=(
                    args.expected_canonical_analysis_sha256
                ),
            )
        elif args.phase == "render":
            if args.expected_canonical_analysis_sha256:
                raise ValueError(
                    "--expected-canonical-analysis-sha256 applies only to analyze"
                )
            jobs = plan_render_job(
                downstream_binding,
                seed_analysis=args.seed_analysis,
                expected_seed_analysis_sha256=args.expected_seed_analysis_sha256,
            )
        else:
            if args.expected_canonical_analysis_sha256:
                raise ValueError(
                    "--expected-canonical-analysis-sha256 applies only to analyze"
                )
            if args.seed_analysis or args.expected_seed_analysis_sha256:
                raise ValueError("seed-analysis arguments apply only to render")
            jobs = plan_downstream_jobs(downstream_binding, args.phase)

    if args.phase in {"train", "freeze"}:
        partition_counts = Counter(job.key[-1] for job in jobs)
        print(
            "partition_distribution="
            + ",".join(
                f"{partition}:{partition_counts[partition]}"
                for partition in ("saffo-a100", "apollo_agate")
            )
        )
    elif args.phase in {"assemble", "analyze", "render"}:
        print(f"partition_candidates={CPU_PARTITION_REQUEST}")
    _validate_receipt_argument(
        binding,
        phase=args.phase,
        submit=args.submit,
        receipt=args.receipt,
    )
    if args.submit and args.phase in {"train", "freeze"}:
        _scheduler_preflight(jobs)
    if args.submit and args.phase in {"assemble", "analyze", "render"}:
        _cpu_candidate_preflight(jobs)
    return execute_plan(jobs, submit=args.submit, receipt_path=args.receipt)


if __name__ == "__main__":
    main()
