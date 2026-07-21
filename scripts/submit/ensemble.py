"""Plan probability-ensemble build or baseline jobs.

Dry-run is the default.  Real submission uses a comma-separated candidate
partition list, never a Slurm array, and records one append-only JSONL receipt
per successful ``sbatch`` call.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from selectseg.artifacts import load_binary_artifact, sha256_file
from selectseg.paths import repository_path
from scripts.submit.main import slurm_command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/auxiliary/binary_probability_ensemble_v1.json",
    )
    parser.add_argument("--phase", choices=("build", "baselines"), default="build")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mean_manifest(config: dict, cell: dict) -> Path:
    root = repository_path(config["paths"]["artifact_output_root"])
    candidates = sorted(
        (root / cell["dataset"] / cell["condition"]).glob("*/manifest.json")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one mean artifact for {cell['dataset']}/{cell['condition']}"
        )
    artifact = load_binary_artifact(candidates[0], validate_payloads=False)
    if artifact.manifest["num_samples"] != cell["expected_num_samples"]:
        raise ValueError("mean ensemble artifact has an unexpected cohort size")
    return candidates[0]


def plan_commands(
    config: dict, source_lock_hash: str, *, phase="build"
) -> list[tuple[str, list[str]]]:
    if phase not in {"build", "baselines"}:
        raise ValueError(f"unsupported ensemble phase: {phase}")
    partitions = config["scheduler"]["cpu_partition_candidates"]
    if not partitions or len(partitions) != len(set(partitions)):
        raise ValueError("CPU partition candidates must be nonempty and unique")
    if config["scheduler"] != {
        "cpu_partition_candidates": partitions,
        "one_experiment_per_job": True,
        "slurm_arrays": False,
    }:
        raise ValueError("scheduler policy must require one non-array job per cell")
    source_lock = str(repository_path(config["paths"]["source_lock"]))
    commands = []
    for index, cell in enumerate(config["conditions"]):
        rotated = (
            partitions[index % len(partitions) :]
            + partitions[: index % len(partitions)]
        )
        key = f"{cell['dataset']}/{cell['condition']}"
        if phase == "build":
            job_name = f"selseg-ens-{cell['dataset']}-{cell['model']}"
            cpus, memory, time_limit = 4, "32g", "12:00:00"
            argv = (
                "python",
                "-m",
                "selectseg.ensemble",
                "--source-lock",
                source_lock,
                "--expected-source-lock-sha256",
                source_lock_hash,
                "--dataset",
                cell["dataset"],
                "--condition",
                cell["condition"],
                "--output-root",
                str(repository_path(config["paths"]["artifact_output_root"])),
            )
        else:
            mean_manifest = _mean_manifest(config, cell)
            job_name = f"selseg-ensbase-{cell['dataset']}-{cell['model']}"
            cpus, memory, time_limit = 8, "48g", "18:00:00"
            argv = (
                "python",
                "-m",
                "selectseg.studies.ensemble",
                "--source-lock",
                source_lock,
                "--expected-source-lock-sha256",
                source_lock_hash,
                "--mean-artifact-manifest",
                str(mean_manifest),
                "--expected-mean-artifact-manifest-sha256",
                sha256_file(mean_manifest),
                "--dataset",
                cell["dataset"],
                "--condition",
                cell["condition"],
                "--gamma",
                str(config["protocol"]["gamma"]),
                "--logit-offset",
                "1.0",
                "--output-root",
                str(repository_path(config["paths"]["baseline_output_root"])),
            )
        command = list(
            slurm_command(
                job_name=job_name,
                partition=",".join(rotated),
                cpus=cpus,
                memory=memory,
                time_limit=time_limit,
                argv=argv,
            )
        )
        commands.append((key, command))
    if len(commands) != 10 or any(
        "--array" in token for _, command in commands for token in command
    ):
        raise ValueError("the ensemble wave must contain 10 independent non-array jobs")
    return commands


def _append_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_json(config_path)
    source_lock_path = repository_path(config["paths"]["source_lock"])
    if not source_lock_path.is_file() or source_lock_path.is_symlink():
        raise FileNotFoundError(f"missing regular source lock: {source_lock_path}")
    source_lock_hash = sha256_file(source_lock_path)
    commands = plan_commands(config, source_lock_hash, phase=args.phase)
    receipt = args.receipt or (
        "outputs/binary_probability_ensemble_v1/submission_receipts.jsonl"
        if args.phase == "build"
        else "outputs/binary_probability_ensemble_v1/baseline_submission_receipt.jsonl"
    )
    for key, command in commands:
        print(f"[{key}] {shlex.join(command)}")
        if not args.submit:
            continue
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"could not parse sbatch job id: {completed.stdout!r}")
        _append_receipt(
            Path(receipt),
            {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase": args.phase,
                "key": key,
                "source_lock_sha256": source_lock_hash,
                "command": command,
                "job_id": job_id,
            },
        )


if __name__ == "__main__":
    main()
