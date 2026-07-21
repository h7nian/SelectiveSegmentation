"""Plan Dice count-posterior jobs for any declared coupling configuration."""

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
        default="configs/auxiliary/dice_coupling_analysis_v1.json",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _variant_id(variant: dict) -> str:
    coupling = variant["coupling"]
    threshold = variant["proposal_threshold"]
    return coupling if threshold is None else f"{coupling}-t{int(round(100 * threshold)):02d}"


def plan_commands(config: dict, config_path: Path) -> list[tuple[str, list[str]]]:
    status = config.get("status")
    if status not in {
        "predeclared-before-computing-coupling-ladder-scores",
        "predeclared-before-computing-component-or-grid-scores",
    }:
        raise ValueError("Dice coupling analysis contract has an unexpected status")
    is_partition_wave = status.endswith("component-or-grid-scores")
    if is_partition_wave:
        parent = config["parent_contract"]
        if sha256_file(repository_path(parent["path"])) != parent["sha256"]:
            raise ValueError("parent coupling contract changed")
    target = {
        (dataset, model)
        for dataset in config["conditions"]["datasets"]
        for model in config["conditions"]["models"]
    }
    if len(target) != config["conditions"]["count"] or len(target) != 10:
        raise ValueError("Dice coupling contract must identify ten target conditions")
    paths = config["paths"]
    workspace_root = repository_path(paths["workspace_root"])
    lock_path = repository_path(paths["campaign_lock"])
    lock = _load_json(lock_path)
    lock_artifacts = {
        (entry["dataset"], entry["condition"]): entry for entry in lock["artifacts"]
    }
    if set(lock_artifacts).issuperset(target) is False:
        raise ValueError("campaign lock does not contain every target condition")

    contract_sha256 = sha256_file(config_path)
    partitions = config["scheduler"]["cpu_partition_candidates"]
    variants = config.get(
        "variants", [{"coupling": "action-two-block", "proposal_threshold": None}]
    )
    if is_partition_wave and (
        len(variants) != 8 or len({_variant_id(value) for value in variants}) != 8
    ):
        raise ValueError("partition contract must identify eight unique variants")
    commands = []
    jobs = [(key, variant) for key in sorted(target) for variant in variants]
    for index, (key, variant) in enumerate(jobs):
        artifact_entry = lock_artifacts[key]
        artifact_manifest = workspace_root / artifact_entry["manifest_path"]
        artifact = load_binary_artifact(artifact_manifest, validate_payloads=False)
        if artifact.manifest_sha256 != artifact_entry["manifest_sha256"]:
            raise ValueError(f"campaign artifact changed for {key}")
        rotations = partitions[index % len(partitions) :] + partitions[: index % len(partitions)]
        variant_id = _variant_id(variant)
        numerics = config.get("numerics", {})
        argv = [
            "python", "-m", "selectseg.studies.counts",
            "--artifact-manifest",
            str(artifact_manifest),
            "--expected-artifact-manifest-sha256",
            artifact.manifest_sha256,
            "--analysis-contract",
            str(config_path),
            "--expected-analysis-contract-sha256",
            contract_sha256,
            "--output-root",
            str(repository_path(paths["output_root"])),
            "--gamma",
            "0.5",
            "--m",
            "32",
            "--coupling",
            variant["coupling"].replace("_", "-"),
        ]
        if variant["proposal_threshold"] is not None:
            argv.extend(
                ["--proposal-threshold", str(variant["proposal_threshold"])]
            )
        if is_partition_wave:
            argv.extend(
                [
                    "--draws",
                    str(numerics["monte_carlo_draws"]),
                    "--repeats",
                    str(numerics["monte_carlo_repeats"]),
                    "--master-seed",
                    str(numerics["master_seed"]),
                ]
            )
        command = list(slurm_command(
            job_name=f"selseg-count-{key[0]}-{key[1]}-{variant_id}",
            partition=",".join(rotations),
            cpus=2,
            memory="24g",
            time_limit="08:00:00",
            argv=argv,
        ))
        commands.append((f"{key[0]}/{key[1]}/{variant_id}", command))
    expected_jobs = 80 if is_partition_wave else 10
    if len(commands) != expected_jobs or any(
        token.startswith("--array") for _, command in commands for token in command
    ):
        raise ValueError(
            f"Dice count wave must contain {expected_jobs} independent non-array jobs"
        )
    return commands


def _append_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        output.flush()


def _load_receipt(path: Path, contract_sha256: str) -> dict[str, dict]:
    if path.is_symlink():
        raise ValueError(f"submission receipt cannot be a symlink: {path}")
    if not path.exists():
        return {}
    events = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank line in submission receipt at {line_number}")
            event = json.loads(line)
            if set(event) != {
                "created_utc",
                "key",
                "analysis_contract_sha256",
                "command",
                "job_id",
            }:
                raise ValueError(f"invalid submission receipt row at {line_number}")
            if event["analysis_contract_sha256"] != contract_sha256:
                raise ValueError("submission receipt belongs to another contract")
            key = event["key"]
            if not isinstance(key, str) or not key or key in events:
                raise ValueError(f"invalid or duplicate receipt key at {line_number}")
            if not isinstance(event["command"], list) or not event["command"]:
                raise ValueError(f"invalid receipt command at {line_number}")
            if not str(event["job_id"]).isdigit():
                raise ValueError(f"invalid receipt job id at {line_number}")
            events[key] = event
    return events


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_json(config_path)
    contract_sha256 = sha256_file(config_path)
    jobs = plan_commands(config, config_path)
    receipt_path = Path(
        args.receipt
        or repository_path(config["paths"]["output_root"])
        / "submission_receipt.jsonl"
    )
    submitted = _load_receipt(receipt_path, contract_sha256) if args.submit else {}
    for key, command in jobs:
        print(f"[{key}] {shlex.join(command)}")
        if not args.submit:
            continue
        if key in submitted:
            if submitted[key]["command"] != command:
                raise ValueError(f"submitted command changed for {key}")
            print(f"[{key}] already submitted as job {submitted[key]['job_id']}")
            continue
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"could not parse sbatch job id: {completed.stdout!r}")
        _append_receipt(
            receipt_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "key": key,
                "analysis_contract_sha256": contract_sha256,
                "command": command,
                "job_id": job_id,
            },
        )


if __name__ == "__main__":
    main()
