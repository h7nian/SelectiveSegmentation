"""Create the immutable three-seed probability-ensemble source lock.

The lock resolves all 30 source artifacts from the canonical seed-0 campaign
lock and the published seed-extension provenance.  It validates every manifest
hash and metadata binding before writing a new file with exclusive creation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selectseg.binary_artifacts import load_binary_artifact, sha256_file
from selectseg.binary_probability_ensemble import (
    SOURCE_LOCK_ARTIFACT_TYPE,
    SOURCE_LOCK_SCHEMA_VERSION,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/auxiliary/binary_probability_ensemble_v1.json",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _source_record(manifest_path: Path, expected_hash: str, training_seed: int) -> dict:
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError(f"source manifest SHA-256 mismatch: {manifest_path}")
    artifact = load_binary_artifact(manifest_path, validate_payloads=False)
    manifest = artifact.manifest
    return {
        "training_seed": training_seed,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": artifact.manifest_sha256,
        "artifact_id": manifest["artifact_id"],
        "dataset": manifest["dataset"],
        "condition": manifest["condition"],
        "model": manifest["model"],
        "split": manifest["split"],
        "num_samples": manifest["num_samples"],
        "sample_id_sha256": manifest["sample_id_sha256"],
    }


def _seed0_sources(campaign_lock: dict) -> dict[tuple[str, str], dict]:
    result = {}
    for record in campaign_lock["artifacts"]:
        key = (record["dataset"], record["condition"])
        if record["condition"] not in {"clipseg-target", "deeplabv3-target"}:
            continue
        result[key] = _source_record(
            Path("../") / record["manifest_path"],
            record["manifest_sha256"],
            training_seed=0,
        )
    return result


def _extension_sources(
    config: dict, provenance: dict
) -> dict[tuple[str, str, int], dict]:
    artifact_root = Path(config["source"]["seed_extension_artifact_root"])
    result = {}
    for cell in provenance["cells"]:
        seed = cell["training_seed"]
        if seed not in {1, 2}:
            continue
        key = (cell["dataset_id"], cell["condition_id"], seed)
        manifest_path = (
            artifact_root
            / cell["dataset_id"]
            / cell["condition_id"]
            / cell["frozen"]["artifact_id"]
            / "manifest.json"
        )
        result[key] = _source_record(
            manifest_path,
            cell["frozen"]["manifest_sha256"],
            training_seed=seed,
        )
    return result


def build_source_lock(config_path: Path) -> dict:
    config = _load_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported probability-ensemble config schema")
    seed0_lock_path = Path(config["source"]["seed0_campaign_lock"])
    provenance_path = Path(config["source"]["seed_extension_provenance"])
    seed0_lock = _load_json(seed0_lock_path)
    provenance = _load_json(provenance_path)
    seed0 = _seed0_sources(seed0_lock)
    extension = _extension_sources(config, provenance)

    cells = []
    for condition in config["conditions"]:
        dataset = condition["dataset"]
        condition_name = condition["condition"]
        key = (dataset, condition_name)
        sources = [
            seed0[key],
            extension[(dataset, condition_name, 1)],
            extension[(dataset, condition_name, 2)],
        ]
        for source in sources:
            for field in ("dataset", "condition", "model", "num_samples"):
                expected_field = (
                    "expected_num_samples" if field == "num_samples" else field
                )
                if source[field] != condition[expected_field]:
                    raise ValueError(
                        f"{dataset}/{condition_name}/seed-{source['training_seed']} "
                        f"has inconsistent {field}"
                    )
        if len({source["sample_id_sha256"] for source in sources}) != 1:
            raise ValueError(f"{dataset}/{condition_name} source sample orders differ")
        cells.append(
            {
                "dataset": dataset,
                "condition": condition_name,
                "model": condition["model"],
                "expected_num_samples": condition["expected_num_samples"],
                "sources": sources,
            }
        )

    if len(cells) != 10 or sum(len(cell["sources"]) for cell in cells) != 30:
        raise ValueError("ensemble source lock must contain 10 cells and 30 sources")
    return {
        "schema_version": SOURCE_LOCK_SCHEMA_VERSION,
        "artifact_type": SOURCE_LOCK_ARTIFACT_TYPE,
        "auxiliary_id": config["auxiliary_id"],
        "config": {
            "path": config_path.as_posix(),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            "seed0_campaign_lock": {
                "path": seed0_lock_path.as_posix(),
                "sha256": sha256_file(seed0_lock_path),
            },
            "seed_extension_provenance": {
                "path": provenance_path.as_posix(),
                "sha256": sha256_file(provenance_path),
            },
        },
        "aggregation": config["aggregation"],
        "cells": cells,
    }


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_json(config_path)
    output_path = Path(args.output or config["paths"]["source_lock"])
    source_lock = build_source_lock(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(source_lock, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"saved {output_path}")
    print(f"sha256={sha256_file(output_path)}")


if __name__ == "__main__":
    main()
