"""Strictly merge canonical binary runs with auxiliary confidence scores.

The canonical source must contain the M=2/8/32 experiment.  The auxiliary
source must be a matched M=2 rerun that adds exactly the five predeclared
scores below.  The merger never replaces a canonical measurement: after an
exact cohort/provenance join it changes only ``run_id`` and appends those five
scores.

Example::

    python scripts/merge_binary_auxiliary.py \
        --canonical-root outputs/binary_final \
        --auxiliary-root outputs/binary_baselines \
        --output-root outputs/binary_merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
MERGE_SCHEMA_VERSION = 1
EVALUATOR_DEFAULT_BATCH_SIZE = 8
RISK_FIELDS = ("risk_dice", "risk_nhd95")
AUXILIARY_FIELDS = ("risk_hd95_pixels",)
BASE_SCORE_FIELDS = (
    "confidence_sdc",
    "confidence_mean_max_probability",
    "confidence_negative_entropy",
)
M2_SCORE_FIELDS = ("confidence_dice_m2", "confidence_nhd95_m2")
CANONICAL_SCORE_FIELDS = (
    *BASE_SCORE_FIELDS,
    *M2_SCORE_FIELDS,
    "confidence_dice_m8",
    "confidence_nhd95_m8",
    "confidence_dice_m32",
    "confidence_nhd95_m32",
)
ADDED_SCORE_FIELDS = (
    "confidence_dice_exact",
    "confidence_qfr_entropy",
    "confidence_plm10_entropy",
    "confidence_mmmc_entropy",
    "confidence_foreground_entropy",
)
AUXILIARY_SCORE_FIELDS = (*BASE_SCORE_FIELDS, *M2_SCORE_FIELDS, *ADDED_SCORE_FIELDS)
BASE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "sample_id",
        "image_id",
        "image_index",
        "class_index",
        "class_name",
        "height",
        "width",
        "image_diagonal",
        "truth_foreground_fraction",
        "prediction_foreground_fraction",
    }
)
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "condition",
        "model",
        "dataset",
        "split",
        "num_images",
        "num_rows",
        "checkpoint",
        "base_model",
        "source_sha256",
        "cohort",
        "decision_rule",
        "preprocessing",
        "losses",
        "risk_fields",
        "auxiliary_fields",
        "score_fields",
        "quadrature",
        "void_policy",
        "sdc_empty_convention",
        "sample_id_sha256",
        "jsonl_sha256",
        "command",
    }
)
MATCHED_MANIFEST_FIELDS = (
    "condition",
    "model",
    "dataset",
    "split",
    "num_images",
    "num_rows",
    "checkpoint",
    "base_model",
    "cohort",
    "decision_rule",
    "preprocessing",
    "losses",
    "void_policy",
    "sdc_empty_convention",
)


@dataclass(frozen=True)
class Run:
    manifest_path: Path
    records_path: Path
    manifest_sha256: str
    manifest: dict
    rows: tuple[dict, ...]
    batch_size: int

    @property
    def key(self) -> tuple[str, str]:
        return self.manifest["dataset"], self.manifest["condition"]


@dataclass(frozen=True)
class MergedArtifact:
    key: tuple[str, str]
    run_id: str
    records: bytes
    manifest: bytes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--auxiliary-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(text, *, source):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _nonempty_string(value, *, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _positive_integer(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _assert_finite(value, *, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, location=f"{location}[{index}]")


def _field_list(manifest, field, *, location):
    values = manifest[field]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{location}.{field} must be a nonempty unique field list")
    return tuple(values)


def _validate_quadrature(value, expected_counts, *, location):
    expected_keys = {str(count) for count in expected_counts}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{location} must contain exactly {sorted(expected_keys)}")
    for count in expected_counts:
        rule = value[str(count)]
        expected_nodes = [(index + 0.5) / count for index in range(count)]
        expected_weights = [1 / count] * count
        if not isinstance(rule, dict) or set(rule) != {"rule", "nodes", "weights"}:
            raise ValueError(f"{location}.{count} has an invalid schema")
        if rule["rule"] != "midpoint":
            raise ValueError(f"{location}.{count} must use midpoint quadrature")
        if rule["nodes"] != expected_nodes or rule["weights"] != expected_weights:
            raise ValueError(f"{location}.{count} has unexpected nodes or weights")


def _command_batch_size(command, *, location):
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError(f"{location} must be a list of strings")
    values = []
    for index, argument in enumerate(command):
        if argument == "--batch-size":
            if index + 1 >= len(command):
                raise ValueError(f"{location} has --batch-size without a value")
            values.append(command[index + 1])
        elif argument.startswith("--batch-size="):
            values.append(argument.split("=", maxsplit=1)[1])
    if not values:
        return EVALUATOR_DEFAULT_BATCH_SIZE
    if len(values) != 1:
        raise ValueError(f"{location} must specify --batch-size at most once")
    try:
        batch_size = int(values[0])
    except ValueError as error:
        raise ValueError(f"{location} has a non-integer --batch-size") from error
    return _positive_integer(batch_size, location=f"{location} --batch-size")


def _expected_row_fields(score_fields):
    return BASE_ROW_FIELDS | set(RISK_FIELDS) | set(AUXILIARY_FIELDS) | set(score_fields)


def _validate_manifest(manifest, *, role, location):
    if not isinstance(manifest, dict):
        raise ValueError(f"{location} must contain one JSON object")
    _assert_finite(manifest, location=location)
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"{location} is missing required fields {missing}")
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"{location}.schema_version must equal {SCHEMA_VERSION}")
    for field in ("run_id", "condition", "model", "dataset", "split", "cohort"):
        _nonempty_string(manifest[field], location=f"{location}.{field}")
    for field in ("void_policy", "sdc_empty_convention"):
        _nonempty_string(manifest[field], location=f"{location}.{field}")
    for field in ("num_images", "num_rows"):
        _positive_integer(manifest[field], location=f"{location}.{field}")
    if manifest["num_images"] != manifest["num_rows"]:
        raise ValueError(f"{location} must contain exactly one row per image")
    for field in ("base_model", "decision_rule", "preprocessing", "losses"):
        if not isinstance(manifest[field], dict) or not manifest[field]:
            raise ValueError(f"{location}.{field} must be a nonempty object")
    if manifest["checkpoint"] is not None and not isinstance(
        manifest["checkpoint"], dict
    ):
        raise ValueError(f"{location}.checkpoint must be null or an object")
    decision_rule = manifest["decision_rule"]
    gamma = decision_rule.get("gamma")
    if (
        decision_rule.get("form") != "foreground_probability >= gamma"
        or isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not 0 < gamma < 1
    ):
        raise ValueError(f"{location}.decision_rule is invalid")
    _digest(manifest["source_sha256"], location=f"{location}.source_sha256")
    _digest(manifest["jsonl_sha256"], location=f"{location}.jsonl_sha256")
    _digest(manifest["sample_id_sha256"], location=f"{location}.sample_id_sha256")
    risks = _field_list(manifest, "risk_fields", location=location)
    auxiliaries = _field_list(manifest, "auxiliary_fields", location=location)
    scores = _field_list(manifest, "score_fields", location=location)
    expected_scores = (
        CANONICAL_SCORE_FIELDS if role == "canonical" else AUXILIARY_SCORE_FIELDS
    )
    if set(risks) != set(RISK_FIELDS):
        raise ValueError(f"{location}.risk_fields must be exactly {sorted(RISK_FIELDS)}")
    if set(auxiliaries) != set(AUXILIARY_FIELDS):
        raise ValueError(
            f"{location}.auxiliary_fields must be exactly {sorted(AUXILIARY_FIELDS)}"
        )
    if set(scores) != set(expected_scores):
        raise ValueError(
            f"{location}.score_fields must be exactly {sorted(expected_scores)}"
        )
    expected_counts = (2, 8, 32) if role == "canonical" else (2,)
    _validate_quadrature(
        manifest["quadrature"], expected_counts, location=f"{location}.quadrature"
    )
    return scores, _command_batch_size(manifest["command"], location=f"{location}.command")


def _load_run(manifest_path, *, role):
    manifest_path = Path(manifest_path)
    records_path = manifest_path.with_name("records.jsonl")
    if not records_path.is_file():
        raise FileNotFoundError(f"records file is missing beside {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _loads_strict(manifest_bytes.decode(), source=str(manifest_path))
    score_fields, batch_size = _validate_manifest(
        manifest, role=role, location=str(manifest_path)
    )
    actual_records_hash = _sha256(records_path)
    expected_records_hash = manifest["jsonl_sha256"].lower()
    if actual_records_hash != expected_records_hash:
        raise ValueError(
            f"JSONL SHA-256 mismatch for {records_path}: "
            f"manifest={expected_records_hash}, actual={actual_records_hash}"
        )

    rows = []
    expected_fields = _expected_row_fields(score_fields)
    sample_ids = []
    seen_sample_ids = set()
    seen_image_ids = set()
    with records_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{records_path}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank JSONL row at {location}")
            row = _loads_strict(line, source=location)
            if not isinstance(row, dict):
                raise ValueError(f"{location} must contain one JSON object")
            _assert_finite(row, location=location)
            if set(row) != expected_fields:
                missing = sorted(expected_fields - set(row))
                extra = sorted(set(row) - expected_fields)
                raise ValueError(
                    f"unauthorized row schema at {location}: "
                    f"missing={missing}, extra={extra}"
                )
            if (
                isinstance(row["schema_version"], bool)
                or not isinstance(row["schema_version"], int)
                or row["schema_version"] != SCHEMA_VERSION
            ):
                raise ValueError(f"schema_version mismatch at {location}")
            if row["run_id"] != manifest["run_id"]:
                raise ValueError(f"run_id mismatch at {location}")
            sample_id = _nonempty_string(
                row["sample_id"], location=f"{location}.sample_id"
            )
            image_id = _nonempty_string(
                row["image_id"], location=f"{location}.image_id"
            )
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {records_path}")
            if image_id in seen_image_ids:
                raise ValueError(f"duplicate image_id {image_id!r} in {records_path}")
            seen_sample_ids.add(sample_id)
            seen_image_ids.add(image_id)
            sample_ids.append(sample_id)
            _nonempty_string(row["class_name"], location=f"{location}.class_name")
            for field in ("image_index", "class_index", "height", "width"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{location}.{field} must be an integer")
            if row["image_index"] < 0 or row["class_index"] < 0:
                raise ValueError(f"{location} contains a negative index")
            if row["height"] <= 0 or row["width"] <= 0:
                raise ValueError(f"{location} contains a non-positive image size")
            for field in (
                "image_diagonal",
                "truth_foreground_fraction",
                "prediction_foreground_fraction",
            ):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{location}.{field} must be numeric")
            if row["image_diagonal"] <= 0:
                raise ValueError(f"{location}.image_diagonal must be positive")
            for field in (
                "truth_foreground_fraction",
                "prediction_foreground_fraction",
            ):
                if not 0 <= row[field] <= 1:
                    raise ValueError(f"{location}.{field} must lie in [0, 1]")
            for field in (*RISK_FIELDS, *AUXILIARY_FIELDS, *score_fields):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{location}.{field} must be numeric")
            if any(not 0 <= row[field] <= 1 for field in RISK_FIELDS):
                raise ValueError(f"{location} contains a risk outside [0, 1]")
            if row["risk_hd95_pixels"] < 0:
                raise ValueError(f"{location}.risk_hd95_pixels must be non-negative")
            rows.append(row)
    if len(rows) != manifest["num_rows"]:
        raise ValueError(
            f"row-count mismatch for {records_path}: "
            f"manifest={manifest['num_rows']}, actual={len(rows)}"
        )
    sample_hash = _sha256_bytes("\n".join(sample_ids).encode())
    if sample_hash != manifest["sample_id_sha256"].lower():
        raise ValueError(
            f"sample_id_sha256 mismatch for {records_path}: "
            f"manifest={manifest['sample_id_sha256']}, actual={sample_hash}"
        )
    return Run(
        manifest_path=manifest_path.resolve(),
        records_path=records_path.resolve(),
        manifest_sha256=_sha256_bytes(manifest_bytes),
        manifest=manifest,
        rows=tuple(rows),
        batch_size=batch_size,
    )


def load_root(root, *, role):
    root = Path(root)
    if role not in {"canonical", "auxiliary"}:
        raise ValueError("role must be 'canonical' or 'auxiliary'")
    if not root.is_dir():
        raise FileNotFoundError(f"{role} root does not exist: {root}")
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"{role} root contains no run manifests: {root}")
    runs = {}
    for manifest_path in manifests:
        run = _load_run(manifest_path, role=role)
        if run.key in runs:
            raise ValueError(f"duplicate {role} condition {run.key} under {root}")
        runs[run.key] = run
    records = {path.resolve() for path in root.rglob("records.jsonl")}
    claimed = {run.records_path for run in runs.values()}
    if records != claimed:
        raise ValueError(
            f"{role} root contains records without matched manifests: "
            f"{sorted(map(str, records - claimed))}"
        )
    return runs


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _portable_path(path):
    """Represent provenance paths relative to the invocation directory."""

    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _require_exact_join(canonical, auxiliary):
    key = canonical.key
    for field in MATCHED_MANIFEST_FIELDS:
        if _canonical_json(canonical.manifest[field]) != _canonical_json(
            auxiliary.manifest[field]
        ):
            raise ValueError(f"condition {key}: manifest field {field!r} differs")
    if canonical.batch_size != auxiliary.batch_size:
        raise ValueError(
            f"condition {key}: evaluation batch size differs "
            f"({canonical.batch_size} != {auxiliary.batch_size})"
        )
    canonical_ids = tuple(row["sample_id"] for row in canonical.rows)
    auxiliary_ids = tuple(row["sample_id"] for row in auxiliary.rows)
    if canonical_ids != auxiliary_ids:
        raise ValueError(f"condition {key}: ordered sample_id cohort differs")
    shared_fields = (set(canonical.rows[0]) & set(auxiliary.rows[0])) - {"run_id"}
    expected_shared = _expected_row_fields((*BASE_SCORE_FIELDS, *M2_SCORE_FIELDS)) - {
        "run_id"
    }
    if shared_fields != expected_shared:
        raise ValueError(
            f"condition {key}: unexpected shared row fields; "
            f"expected={sorted(expected_shared)}, observed={sorted(shared_fields)}"
        )
    for canonical_row, auxiliary_row in zip(canonical.rows, auxiliary.rows):
        sample_id = canonical_row["sample_id"]
        for field in sorted(shared_fields):
            if _canonical_json(canonical_row[field]) != _canonical_json(
                auxiliary_row[field]
            ):
                raise ValueError(
                    f"condition {key}: exact shared field {field!r} differs "
                    f"for sample_id={sample_id!r}"
                )


def _merge_source_sha256():
    return _sha256(Path(__file__).resolve())


def _prepare_artifact(canonical, auxiliary):
    _require_exact_join(canonical, auxiliary)
    merge_source_sha256 = _merge_source_sha256()
    identity = {
        "merge_schema_version": MERGE_SCHEMA_VERSION,
        "merge_source_sha256": merge_source_sha256,
        "canonical_manifest_sha256": canonical.manifest_sha256,
        "auxiliary_manifest_sha256": auxiliary.manifest_sha256,
        "canonical_jsonl_sha256": canonical.manifest["jsonl_sha256"].lower(),
        "auxiliary_jsonl_sha256": auxiliary.manifest["jsonl_sha256"].lower(),
        "added_score_fields": list(ADDED_SCORE_FIELDS),
    }
    run_id = _sha256_bytes(_canonical_json(identity).encode())[:16]
    auxiliary_by_id = {row["sample_id"]: row for row in auxiliary.rows}
    merged_rows = []
    for source_row in canonical.rows:
        row = dict(source_row)
        row["run_id"] = run_id
        auxiliary_row = auxiliary_by_id[row["sample_id"]]
        for field in ADDED_SCORE_FIELDS:
            row[field] = auxiliary_row[field]
        merged_rows.append(row)
    records = "".join(
        _canonical_json(row) + "\n" for row in merged_rows
    ).encode()
    ordered_sample_ids = [row["sample_id"] for row in merged_rows]
    sample_id_sha256 = _sha256_bytes("\n".join(ordered_sample_ids).encode())
    score_fields = [*canonical.manifest["score_fields"], *ADDED_SCORE_FIELDS]
    source_manifests = {
        "canonical": {
            "path": _portable_path(canonical.manifest_path),
            "sha256": canonical.manifest_sha256,
            "records_path": _portable_path(canonical.records_path),
            "records_sha256": canonical.manifest["jsonl_sha256"].lower(),
            "contents": canonical.manifest,
        },
        "auxiliary": {
            "path": _portable_path(auxiliary.manifest_path),
            "sha256": auxiliary.manifest_sha256,
            "records_path": _portable_path(auxiliary.records_path),
            "records_sha256": auxiliary.manifest["jsonl_sha256"].lower(),
            "contents": auxiliary.manifest,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "merge_schema_version": MERGE_SCHEMA_VERSION,
        "run_id": run_id,
        "condition": canonical.manifest["condition"],
        "model": canonical.manifest["model"],
        "dataset": canonical.manifest["dataset"],
        "split": canonical.manifest["split"],
        "num_images": len(merged_rows),
        "num_rows": len(merged_rows),
        "checkpoint": canonical.manifest["checkpoint"],
        "base_model": canonical.manifest["base_model"],
        "source_sha256": merge_source_sha256,
        "canonical_source_sha256": canonical.manifest["source_sha256"].lower(),
        "auxiliary_source_sha256": auxiliary.manifest["source_sha256"].lower(),
        "source_manifests": source_manifests,
        "cohort": canonical.manifest["cohort"],
        "decision_rule": canonical.manifest["decision_rule"],
        "evaluation_batch_size": canonical.batch_size,
        "preprocessing": canonical.manifest["preprocessing"],
        "losses": canonical.manifest["losses"],
        "risk_fields": list(canonical.manifest["risk_fields"]),
        "auxiliary_fields": list(canonical.manifest["auxiliary_fields"]),
        "score_fields": score_fields,
        "quadrature": canonical.manifest["quadrature"],
        "void_policy": canonical.manifest["void_policy"],
        "sdc_empty_convention": canonical.manifest["sdc_empty_convention"],
        "merge": {
            "policy": "preserve canonical rows except run_id; append named scores",
            "added_score_fields": list(ADDED_SCORE_FIELDS),
            "exact_shared_row_fields": sorted(
                _expected_row_fields((*BASE_SCORE_FIELDS, *M2_SCORE_FIELDS))
                - {"run_id"}
            ),
        },
        "sample_id_sha256": sample_id_sha256,
        "jsonl_sha256": _sha256_bytes(records),
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return MergedArtifact(canonical.key, run_id, records, manifest_bytes)


def prepare_merges(canonical_root, auxiliary_root):
    canonical = load_root(canonical_root, role="canonical")
    auxiliary = load_root(auxiliary_root, role="auxiliary")
    if set(canonical) != set(auxiliary):
        missing = sorted(set(canonical) - set(auxiliary))
        extra = sorted(set(auxiliary) - set(canonical))
        raise ValueError(
            f"condition-set mismatch: missing auxiliary={missing}, extra auxiliary={extra}"
        )
    return tuple(
        _prepare_artifact(canonical[key], auxiliary[key]) for key in sorted(canonical)
    )


def _write_file(path, content):
    with Path(path).open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_merges(artifacts, output_root):
    output_root = Path(output_root)
    targets = {
        artifact: output_root / artifact.key[0] / artifact.key[1] / artifact.run_id
        for artifact in artifacts
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite merged runs: {sorted(existing)}")
    written = []
    for artifact, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{artifact.run_id}.tmp-", dir=target.parent)
        )
        try:
            _write_file(temporary / "records.jsonl", artifact.records)
            _write_file(temporary / "manifest.json", artifact.manifest)
            temporary.rename(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        written.append(target)
    return tuple(written)


def merge_roots(canonical_root, auxiliary_root, output_root):
    artifacts = prepare_merges(canonical_root, auxiliary_root)
    return write_merges(artifacts, output_root)


def main(argv=None):
    args = parse_args(argv)
    paths = merge_roots(args.canonical_root, args.auxiliary_root, args.output_root)
    for path in paths:
        print(f"saved {path / 'records.jsonl'}")
        print(f"saved {path / 'manifest.json'}")


if __name__ == "__main__":
    main()
