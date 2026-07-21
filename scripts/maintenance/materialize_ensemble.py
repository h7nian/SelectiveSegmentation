"""Materialize the scorer config and lock for completed ensemble artifacts.

The tracked auxiliary specification remains the only handwritten experiment
grid.  This command derives the compatible scorer campaign after all ten build
jobs finish, validates exactly one artifact per condition, and publishes both
files with no-overwrite semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.submit.main import (
    build_campaign_lock,
    load_config,
    write_campaign_lock,
)
from selectseg.artifacts import load_binary_artifact, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/auxiliary/binary_probability_ensemble_v1.json",
    )
    parser.add_argument("--campaign-config")
    parser.add_argument("--campaign-lock")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _completed_manifests(config: dict, source_lock_hash: str) -> list[Path]:
    artifact_root = Path(config["paths"]["artifact_output_root"])
    manifests = []
    for cell in config["conditions"]:
        condition_root = artifact_root / cell["dataset"] / cell["condition"]
        candidates = sorted(condition_root.glob("*/manifest.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one completed artifact for {cell['dataset']}/"
                f"{cell['condition']}, found {len(candidates)}"
            )
        artifact = load_binary_artifact(candidates[0], validate_payloads=False)
        manifest = artifact.manifest
        for field, expected in (
            ("dataset", cell["dataset"]),
            ("condition", cell["condition"]),
            ("model", cell["model"]),
            ("num_samples", cell["expected_num_samples"]),
        ):
            if manifest[field] != expected:
                raise ValueError(f"completed ensemble artifact has wrong {field}")
        if manifest["checkpoint"]["sha256"] != source_lock_hash:
            raise ValueError("completed ensemble artifact binds a different source lock")
        manifests.append(candidates[0])
    return manifests


def _campaign_config(config: dict, source_lock_path: Path) -> dict:
    candidates = config["scheduler"]["cpu_partition_candidates"]
    rotations = [
        ",".join(candidates[index:] + candidates[:index])
        for index in range(len(candidates))
    ]
    return {
        "config_schema_version": 1,
        "campaign_id": config["auxiliary_id"],
        "protocol": {
            "gamma_values": [config["protocol"]["gamma"]],
            "m_values": config["protocol"]["m_values"],
            "quadrature_rule": config["protocol"]["quadrature_rule"],
            "seeds": [0],
        },
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "cpu_partitions": rotations,
        "estimator_spec": "configs/estimators/midpoint-v1.json",
        "paths": {
            field: config["paths"][field]
            for field in (
                "artifact_output_root",
                "common_output_root",
                "simulation_output_root",
                "assembly_output_root",
            )
        },
        "conditions": [
            {
                "dataset": cell["dataset"],
                "condition": cell["condition"],
                "model": cell["model"],
                "checkpoint": source_lock_path.as_posix(),
                "batch_size": 1,
                "expected_num_samples": cell["expected_num_samples"],
            }
            for cell in config["conditions"]
        ],
    }


def _write_json_new_or_identical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to replace non-identical campaign config: {path}"
            )


def main(argv=None):
    args = parse_args(argv)
    auxiliary_config_path = Path(args.config)
    auxiliary_config = _load_json(auxiliary_config_path)
    source_lock_path = Path(auxiliary_config["paths"]["source_lock"])
    source_lock_hash = sha256_file(source_lock_path)
    campaign_root = source_lock_path.parent
    campaign_config_path = Path(
        args.campaign_config or campaign_root / "campaign.config.json"
    )
    campaign_lock_path = Path(args.campaign_lock or campaign_root / "campaign.lock.json")
    manifests = _completed_manifests(auxiliary_config, source_lock_hash)
    _write_json_new_or_identical(
        campaign_config_path,
        _campaign_config(auxiliary_config, source_lock_path),
    )
    config = load_config(campaign_config_path)
    campaign_lock = build_campaign_lock(config, [str(path) for path in manifests])
    lock_path, lock_hash = write_campaign_lock(campaign_lock, campaign_lock_path)
    print(f"saved {campaign_config_path}")
    print(f"campaign_config_sha256={sha256_file(campaign_config_path)}")
    print(f"saved {lock_path}")
    print(f"campaign_lock_sha256={lock_hash}")


if __name__ == "__main__":
    main()
