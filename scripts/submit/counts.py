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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-job commands during large submitted waves",
    )
    parser.add_argument("--receipt")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="optional exact DATASET/CONDITION filter; repeat as needed",
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        default=[],
        help="optional exact configured variant id; repeat as needed",
    )
    parser.add_argument(
        "--repeat-index",
        action="append",
        type=int,
        default=[],
        help="optional exact configured repeat index; repeat as needed",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _variant_id(variant: dict) -> str:
    coupling = variant["coupling"]
    if coupling == "spatial-copula":
        value = variant.get("id")
        if (
            not isinstance(value, str)
            or not value
            or not value.replace("-", "").isalnum()
        ):
            raise ValueError("spatial-copula variant id must be a simple token")
        return value
    threshold = variant["proposal_threshold"]
    return coupling if threshold is None else f"{coupling}-t{int(round(100 * threshold)):02d}"


def plan_commands(
    config: dict,
    config_path: Path,
    *,
    condition_filters=(),
    variant_filters=(),
    repeat_filters=(),
) -> list[tuple[str, list[str]]]:
    status = config.get("status")
    if status not in {
        "predeclared-before-computing-coupling-ladder-scores",
        "predeclared-before-computing-component-or-grid-scores",
        "predeclared-before-computing-spatial-copula-scores",
    }:
        raise ValueError("Dice coupling analysis contract has an unexpected status")
    is_partition_wave = status.endswith("component-or-grid-scores")
    is_spatial_copula_wave = status.endswith("spatial-copula-scores")
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

    requested_conditions = set(condition_filters)
    if requested_conditions:
        malformed = {
            value for value in requested_conditions if value.count("/") != 1
        }
        parsed_conditions = {
            tuple(value.split("/", maxsplit=1))
            for value in requested_conditions
            if value not in malformed
        }
        unknown = parsed_conditions - target
        if malformed or unknown:
            raise ValueError(
                "invalid condition filters: "
                f"{sorted(malformed)}; unknown: {sorted(unknown)}"
            )
        target = parsed_conditions

    contract_sha256 = sha256_file(config_path)
    partition_field = (
        "gpu_partition_candidates"
        if is_spatial_copula_wave
        else "cpu_partition_candidates"
    )
    partitions = config["scheduler"][partition_field]
    if (
        not isinstance(partitions, list)
        or not partitions
        or len(set(partitions)) != len(partitions)
    ):
        raise ValueError(f"scheduler.{partition_field} must contain unique partitions")
    variants = config.get(
        "variants", [{"coupling": "action-two-block", "proposal_threshold": None}]
    )
    if is_partition_wave and (
        len(variants) != 8 or len({_variant_id(value) for value in variants}) != 8
    ):
        raise ValueError("partition contract must identify eight unique variants")
    variant_by_id = {_variant_id(value): value for value in variants}
    if len(variant_by_id) != len(variants):
        raise ValueError("coupling variants must have unique ids")
    if variant_filters:
        if len(variant_filters) != len(set(variant_filters)):
            raise ValueError("variant filters must be unique")
        unknown = set(variant_filters) - set(variant_by_id)
        if unknown:
            raise ValueError(f"unknown variant filters: {sorted(unknown)}")
        variants = [variant_by_id[value] for value in variant_filters]

    repeat_indices = [None]
    if is_spatial_copula_wave:
        repeat_indices = list(config["numerics"]["repeat_indices"])
        if (
            not repeat_indices
            or len(set(repeat_indices)) != len(repeat_indices)
            or any(not isinstance(value, int) or value < 0 for value in repeat_indices)
        ):
            raise ValueError(
                "spatial-copula repeat indices must be unique and non-negative"
            )
        if repeat_filters:
            if len(repeat_filters) != len(set(repeat_filters)):
                raise ValueError("repeat filters must be unique")
            unknown = set(repeat_filters) - set(repeat_indices)
            if unknown:
                raise ValueError(f"unknown repeat filters: {sorted(unknown)}")
            repeat_indices = list(dict.fromkeys(repeat_filters))
    elif repeat_filters:
        raise ValueError("repeat filters are available only for spatial-copula jobs")

    commands = []
    artifact_bindings = {}
    for key in sorted(target):
        artifact_entry = lock_artifacts[key]
        artifact_manifest = workspace_root / artifact_entry["manifest_path"]
        artifact = load_binary_artifact(artifact_manifest, validate_payloads=False)
        if artifact.manifest_sha256 != artifact_entry["manifest_sha256"]:
            raise ValueError(f"campaign artifact changed for {key}")
        artifact_bindings[key] = (artifact_manifest, artifact.manifest_sha256)

    jobs = [
        (key, variant, repeat_index)
        for key in sorted(target)
        for variant in variants
        for repeat_index in repeat_indices
    ]
    for index, (key, variant, repeat_index) in enumerate(jobs):
        artifact_manifest, artifact_manifest_sha256 = artifact_bindings[key]
        rotations = (
            partitions[index % len(partitions) :]
            + partitions[: index % len(partitions)]
        )
        variant_id = _variant_id(variant)
        numerics = config.get("numerics", {})
        argv = [
            "python", "-m", "selectseg.studies.counts",
            "--artifact-manifest",
            str(artifact_manifest),
            "--expected-artifact-manifest-sha256",
            artifact_manifest_sha256,
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
        if variant.get("proposal_threshold") is not None:
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
        if is_spatial_copula_wave:
            argv.extend(
                [
                    "--posterior-draws",
                    str(numerics["posterior_draws"]),
                    "--repeat-index",
                    str(repeat_index),
                    "--global-variance-weight",
                    str(variant["global_variance_weight"]),
                    "--spatial-variance-weight",
                    str(variant["spatial_variance_weight"]),
                    "--spatial-knot-spacing-diagonal",
                    str(variant["spatial_knot_spacing_diagonal"]),
                    "--posterior-batch-size",
                    str(numerics["posterior_batch_size"]),
                    "--master-seed",
                    str(numerics["master_seed"]),
                    "--device",
                    "cuda",
                ]
            )
        key_suffix = (
            f"{variant_id}/repeat-{repeat_index}"
            if is_spatial_copula_wave
            else variant_id
        )
        command = list(
            slurm_command(
                job_name=(
                    f"selseg-count-{key[0]}-{key[1]}-{variant_id}"
                    + (f"-r{repeat_index}" if is_spatial_copula_wave else "")
                ),
                partition=",".join(rotations),
                cpus=4 if is_spatial_copula_wave else 2,
                memory="48g" if is_spatial_copula_wave else "24g",
                time_limit="08:00:00",
                argv=argv,
                gres="gpu:1" if is_spatial_copula_wave else None,
            )
        )
        commands.append((f"{key[0]}/{key[1]}/{key_suffix}", command))
    expected_jobs = len(target) * len(variants) * len(repeat_indices)
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


def _commands_equivalent(left: list[str], right: list[str]) -> bool:
    """Compare commands while ignoring only candidate-partition order."""

    if left == right:
        return True
    if left.count("--partition") != 1 or right.count("--partition") != 1:
        return False
    left_index = left.index("--partition") + 1
    right_index = right.index("--partition") + 1
    left_partitions = left[left_index].split(",")
    right_partitions = right[right_index].split(",")
    if (
        len(left_partitions) != len(set(left_partitions))
        or len(right_partitions) != len(set(right_partitions))
        or set(left_partitions) != set(right_partitions)
    ):
        return False
    normalized_left = list(left)
    normalized_right = list(right)
    normalized_left[left_index] = ",".join(sorted(left_partitions))
    normalized_right[right_index] = ",".join(sorted(right_partitions))
    return normalized_left == normalized_right


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_json(config_path)
    contract_sha256 = sha256_file(config_path)
    jobs = plan_commands(
        config,
        config_path,
        condition_filters=args.condition,
        variant_filters=args.variant_id,
        repeat_filters=args.repeat_index,
    )
    receipt_path = Path(
        args.receipt
        or repository_path(config["paths"]["output_root"])
        / "submission_receipt.jsonl"
    )
    submitted = _load_receipt(receipt_path, contract_sha256) if args.submit else {}
    for key, command in jobs:
        if not args.submit or not args.quiet:
            print(f"[{key}] {shlex.join(command)}")
        if not args.submit:
            continue
        if key in submitted:
            if not _commands_equivalent(submitted[key]["command"], command):
                raise ValueError(f"submitted command changed for {key}")
            if not args.quiet:
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
