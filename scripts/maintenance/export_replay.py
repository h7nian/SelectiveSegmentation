"""Export the fixed three-seed assemblies as a path-free replay bundle.

The original assembly records are already scientific, row-level JSON.  Their
manifests also contain execution metadata, so this exporter never copies those
manifests.  It validates the 30 source pairs against the released seed
analysis, preserves the record bytes, creates manifests from an exact public
field allowlist, verifies a full numerical replay in memory, publishes the
lock, and publishes a small completion guard last. Existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from scripts.analyze.main import load_condition
from scripts.maintenance.build_release import scan_anonymous_bytes
from scripts.maintenance.replay_seed import (
    LOCK_ARTIFACT_TYPE,
    LOCK_SCHEMA_VERSION,
    MANIFEST_ARTIFACT_TYPE,
    MANIFEST_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    TARGET_CONDITIONS,
    TARGET_DATASETS,
    TRAINING_SEEDS,
    _EXPECTED_OUTPUT_PATHS,
    _json_bytes,
    _sha256_bytes,
    validate_portable_condition,
    verify_replay_payloads,
)


DEFAULT_PUBLIC_ANALYSIS = "outputs/public_seed/seed_robustness_analysis.json"
DEFAULT_SEED0_ROOT = "outputs/binary_assembled"
DEFAULT_EXTENSION_ROOT = "outputs/binary_seed_assembled"
DEFAULT_ROBUSTNESS_TABLE = "docs/Tables/seed_robustness.tex"
DEFAULT_GATE_TABLE = "docs/Tables/seed_sensitivity_main.tex"
DEFAULT_OUTPUT_ROOT = "outputs/public_seed/seed_records"
DEFAULT_LOCK_OUTPUT = "outputs/public_seed/seed_replay.lock.json"
DEFAULT_GUARD_OUTPUT = "outputs/public_seed/seed_replay.complete.json"


def _strict_public_analysis(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"public seed analysis must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("artifact_type") != "selectseg.binary_seed_public_analysis"
        or not isinstance(value.get("cells"), list)
        or len(value["cells"]) != 10
    ):
        raise ValueError("public seed analysis has an unsupported schema")
    keys = [(row.get("dataset"), row.get("condition")) for row in value["cells"]]
    expected = sorted(
        (dataset, condition)
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    )
    if keys != expected:
        raise ValueError("public seed analysis does not contain the fixed ordered grid")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay export: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_pair(root: Path, *, dataset: str, condition: str, run_id: str):
    parent = root / dataset / condition / run_id
    return parent / "records.jsonl", parent / "manifest.json"


def _portable_manifest(*, source, seed: int, logical_id: str) -> dict:
    manifest = source.manifest
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "training_seed": seed,
        "logical_id": logical_id,
        "dataset": manifest["dataset"],
        "condition": manifest["condition"],
        "model": manifest["model"],
        "split": manifest["split"],
        "run_id": manifest["run_id"],
        "num_images": manifest["num_images"],
        "num_rows": manifest["num_rows"],
        "records_sha256": manifest["jsonl_sha256"],
        "sample_id_sha256": manifest["sample_id_sha256"],
        "score_fields": manifest["score_fields"],
        "risk_fields": manifest["risk_fields"],
        "auxiliary_fields": manifest["auxiliary_fields"],
        "source_assembly_manifest_sha256": hashlib.sha256(
            source.manifest_path.read_bytes()
        ).hexdigest(),
    }


def export_seed_replay_bundle(
    *,
    public_analysis: str | os.PathLike[str],
    seed0_root: str | os.PathLike[str],
    extension_root: str | os.PathLike[str],
    robustness_table: str | os.PathLike[str],
    gate_table: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    lock_output: str | os.PathLike[str],
    guard_output: str | os.PathLike[str],
) -> dict:
    analysis_path = Path(public_analysis)
    analysis_payload = analysis_path.read_bytes()
    analysis = _strict_public_analysis(analysis_path)
    table_payload = Path(robustness_table).read_bytes()
    gate_payload = Path(gate_table).read_bytes()
    source_roots = {0: Path(seed0_root), 1: Path(extension_root), 2: Path(extension_root)}
    cells = {(row["dataset"], row["condition"]): row for row in analysis["cells"]}

    payloads = {
        _EXPECTED_OUTPUT_PATHS["analysis"]: analysis_payload,
        _EXPECTED_OUTPUT_PATHS["robustness_table"]: table_payload,
        _EXPECTED_OUTPUT_PATHS["gate_table"]: gate_payload,
    }
    condition_bindings = []
    publication = []
    for seed, dataset, condition in sorted(
        (seed, dataset, condition)
        for seed in TRAINING_SEEDS
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    ):
        cell = cells[(dataset, condition)]
        public_source = cell["sources"][str(seed)]
        logical_id = public_source["logical_id"]
        parts = PurePosixPath(logical_id).parts
        if len(parts) != 4 or parts[:3] != (f"seed-{seed}", dataset, condition):
            raise ValueError("public source logical identifier is inconsistent")
        run_id = parts[3]
        records_path, manifest_path = _source_pair(
            source_roots[seed], dataset=dataset, condition=condition, run_id=run_id
        )
        if hashlib.sha256(records_path.read_bytes()).hexdigest() != public_source[
            "records_sha256"
        ]:
            raise ValueError(f"source records changed for {logical_id}")
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != public_source[
            "manifest_sha256"
        ]:
            raise ValueError(f"source manifest changed for {logical_id}")
        loaded = load_condition(records_path)
        if (
            loaded.dataset != dataset
            or loaded.condition != condition
            or loaded.manifest["run_id"] != run_id
            or loaded.manifest["model"] != cell["model"]
            or len(loaded.rows) != cell["num_images_per_seed"]
        ):
            raise ValueError(f"source assembly metadata differs for {logical_id}")

        records_payload = records_path.read_bytes()
        manifest_payload = _json_bytes(
            _portable_manifest(source=loaded, seed=seed, logical_id=logical_id)
        )
        validate_portable_condition(manifest_payload, records_payload)
        scan_anonymous_bytes(records_payload, source=f"{logical_id}/records.jsonl")
        scan_anonymous_bytes(manifest_payload, source=f"{logical_id}/manifest.json")
        public_parent = f"results/seed_records/{logical_id}"
        public_manifest = f"{public_parent}/manifest.json"
        public_records = f"{public_parent}/records.jsonl"
        payloads[public_manifest] = manifest_payload
        payloads[public_records] = records_payload
        condition_bindings.append(
            {
                "training_seed": seed,
                "dataset": dataset,
                "condition": condition,
                "model": cell["model"],
                "logical_id": logical_id,
                "manifest_path": public_manifest,
                "manifest_sha256": _sha256_bytes(manifest_payload),
                "records_path": public_records,
                "records_sha256": _sha256_bytes(records_payload),
                "source_assembly_manifest_sha256": public_source[
                    "manifest_sha256"
                ],
                "num_images": len(loaded.rows),
                "sample_id_sha256": loaded.manifest["sample_id_sha256"],
            }
        )
        relative_parent = PurePosixPath(logical_id)
        publication.extend(
            (
                (relative_parent / "manifest.json", manifest_payload),
                (relative_parent / "records.jsonl", records_payload),
            )
        )

    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "artifact_type": LOCK_ARTIFACT_TYPE,
        "analysis_provenance": analysis["provenance"],
        "conditions": condition_bindings,
        "expected_outputs": {
            "analysis": {
                "path": _EXPECTED_OUTPUT_PATHS["analysis"],
                "sha256": _sha256_bytes(analysis_payload),
            },
            "robustness_table": {
                "path": _EXPECTED_OUTPUT_PATHS["robustness_table"],
                "sha256": _sha256_bytes(table_payload),
            },
            "gate_table": {
                "path": _EXPECTED_OUTPUT_PATHS["gate_table"],
                "sha256": _sha256_bytes(gate_payload),
            },
        },
    }
    lock_payload = _json_bytes(lock)
    scan_anonymous_bytes(lock_payload, source="seed_replay.lock.json")
    verify_replay_payloads(lock_payload, payloads.__getitem__)

    bundle_digest = hashlib.sha256()
    for relative, payload in sorted(
        publication, key=lambda item: item[0].as_posix()
    ):
        bundle_digest.update(relative.as_posix().encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(hashlib.sha256(payload).digest())
        bundle_digest.update(b"\0")
    guard_payload = _json_bytes(
        {
            "schema_version": 1,
            "artifact_type": "selectseg.portable_seed_replay_complete",
            "lock_sha256": _sha256_bytes(lock_payload),
            "condition_count": len(condition_bindings),
            "portable_file_count": len(publication),
            "portable_bundle_sha256": bundle_digest.hexdigest(),
        }
    )
    scan_anonymous_bytes(guard_payload, source="seed_replay.complete.json")

    destination_root = Path(output_root)
    lock_destination = Path(lock_output)
    guard_destination = Path(guard_output)
    if destination_root.exists() or destination_root.is_symlink():
        raise FileExistsError(f"refusing existing replay output root: {destination_root}")
    if lock_destination.exists() or lock_destination.is_symlink():
        raise FileExistsError(f"refusing existing replay lock: {lock_destination}")
    if guard_destination.exists() or guard_destination.is_symlink():
        raise FileExistsError(f"refusing existing replay guard: {guard_destination}")
    for relative, payload in publication:
        _write_new(destination_root / relative, payload)
    _write_new(lock_destination, lock_payload)
    _write_new(guard_destination, guard_payload)
    return {
        "condition_count": len(condition_bindings),
        "file_count": len(publication),
        "lock": lock_destination,
        "lock_sha256": _sha256_bytes(lock_payload),
        "guard": guard_destination,
        "guard_sha256": _sha256_bytes(guard_payload),
        "records_bytes": sum(len(payload) for relative, payload in publication if relative.name == "records.jsonl"),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-analysis", default=DEFAULT_PUBLIC_ANALYSIS)
    parser.add_argument("--seed0-root", default=DEFAULT_SEED0_ROOT)
    parser.add_argument("--extension-root", default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--robustness-table", default=DEFAULT_ROBUSTNESS_TABLE)
    parser.add_argument("--gate-table", default=DEFAULT_GATE_TABLE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock-output", default=DEFAULT_LOCK_OUTPUT)
    parser.add_argument("--guard-output", default=DEFAULT_GUARD_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_seed_replay_bundle(
        public_analysis=args.public_analysis,
        seed0_root=args.seed0_root,
        extension_root=args.extension_root,
        robustness_table=args.robustness_table,
        gate_table=args.gate_table,
        output_root=args.output_root,
        lock_output=args.lock_output,
        guard_output=args.guard_output,
    )
    print(json.dumps({key: str(value) for key, value in report.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
