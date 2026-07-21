"""Content-addressed scientific inputs for the binary benchmark.

The root lock is deliberately small.  Large, independently buildable component
manifests hold dataset samples and ordinary file bindings; the root contains
only their paths and SHA-256 digests.  Dataset manifests follow the eval loader
order in :mod:`selectseg.data` and bind the source image and mask bytes.

``fast`` verification is a compute-node guard: it checks manifest hashes,
loader order, paths, sizes, and symlink targets without re-reading the
large payloads.  ``full`` verification hashes every selected byte and is the
authoritative audit.  A fast check is therefore not a substitute for the full
audit that creates/seals a component.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


COMPONENT_SCHEMA_VERSION = 1
ROOT_LOCK_SCHEMA_VERSION = 1
SCIENCE_PROJECTION_SCHEMA_VERSION = 1
EVAL_DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
DATASET_PROTOCOL = {
    "pet": ("test", "PetSegmentation"),
    "kvasir": ("test", "KvasirSegmentation"),
    "fives": ("test", "FivesSegmentation"),
    "isic": ("test", "ISICSegmentation"),
    "tn3k": ("test", "TN3KSegmentation"),
}
DEFAULT_ENVIRONMENT_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "transformers",
    "Pillow",
    "scipy",
)

# These fields affect scheduling or output placement, not the scientific
# estimand.  In particular the root-lock binding itself is excluded, avoiding
# a config -> root -> config hash cycle.  Unknown future fields remain bound.
NON_SCIENTIFIC_CONFIG_FIELDS = frozenset(
    {
        "execution_policy",
        "gpu_partitions",
        "gpu_partition_candidates",
        "cpu_partition_candidates",
        "paths",
        "scientific_input_lock",
    }
)

_FILE_REGULAR_FIELDS = frozenset({"kind", "path", "size_bytes", "sha256"})
_FILE_SYMLINK_FIELDS = _FILE_REGULAR_FIELDS | {
    "symlink_target",
    "resolved_path",
}
_BINDING_FIELDS = frozenset({"path", "sha256"})
_DATASET_FIELDS = frozenset(
    {
        "component_schema_version",
        "component_kind",
        "dataset",
        "data_root",
        "split",
        "loader_class",
        "sample_count",
        "sample_id_sha256",
        "loader_order_sha256",
        "selection_files",
        "samples",
    }
)
_SOURCE_FIELDS = frozenset(
    {"component_schema_version", "component_kind", "files"}
)
_BASE_MODEL_FIELDS = frozenset(
    {"component_schema_version", "component_kind", "entries"}
)
_CHECKPOINT_FIELDS = frozenset(
    {"component_schema_version", "component_kind", "entries"}
)
_ENVIRONMENT_FIELDS = frozenset(
    {"component_schema_version", "component_kind", "packages", "environment"}
)
_ROOT_FIELDS = frozenset(
    {
        "scientific_input_lock_schema_version",
        "campaign_id",
        "science_config",
        "components",
    }
)


@dataclass(frozen=True)
class LockedEvalSample:
    """Immutable expected bytes for one item in the real eval-loader order."""

    index: int
    sample_id: str
    prompt_index: int
    image_path: str
    image_sha256: str
    image_size_bytes: int
    mask_path: str
    mask_sha256: str
    mask_size_bytes: int


@dataclass(frozen=True)
class VerifiedEvalDataset:
    """Verified consumed prefix plus its full-cohort component identity."""

    repo_root: Path
    dataset: str
    component_sha256: str
    dataset_sample_count: int
    sample_count: int
    sample_id_sha256: str
    loader_order_sha256: str
    samples_by_id: Mapping[str, LockedEvalSample]


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_value(value, *, location="value"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} has a non-string object key")
            _json_value(item, location=f"{location}.{key}")
        return
    raise ValueError(f"{location} contains a non-JSON value {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    """Return the sole canonical byte representation used for logical hashes."""

    _json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def logical_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_fields(value, expected, *, location):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} must contain exactly {sorted(expected)}")


def _digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value, *, location):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _nonempty_string(value, *, location):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def portable_path(value, *, location="path") -> PurePosixPath:
    """Validate a normalized repository-relative POSIX path."""

    _nonempty_string(value, location=location)
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"{location} must be a relative POSIX path")
    path = PurePosixPath(value)
    if value in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{location} must not traverse or contain dot components")
    if path.as_posix() != value:
        raise ValueError(f"{location} must be normalized POSIX syntax")
    return path


def _absolute_root(repo_root) -> Path:
    root = Path(repo_root).absolute()
    _reject_symlink_ancestors(root)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"repository root must be a regular directory: {root}")
    return root


def _reject_symlink_ancestors(path, *, allow_leaf=False):
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) and not (allow_leaf and index == len(parts) - 1):
            raise ValueError(f"symlink path components are forbidden: {path}")


def _within_root(path: Path, root: Path, *, location) -> Path:
    candidate = Path(path)
    absolute = (candidate if candidate.is_absolute() else root / candidate).absolute()
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{location} must stay within repository root {root}") from error
    return absolute


def _repo_path(root: Path, value, *, location) -> Path:
    portable = portable_path(value, location=location)
    return _within_root(root.joinpath(*portable.parts), root, location=location)


def _portable_from_path(path, root: Path, *, location) -> str:
    absolute = _within_root(Path(path), root, location=location)
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:  # pragma: no cover - guarded by _within_root
        raise ValueError(f"{location} is outside repository root") from error
    return portable_path(relative.as_posix(), location=location).as_posix()


def load_strict_json(path, *, repo_root="."):
    """Load strict JSON from a regular, non-symlink repository file."""

    root = _absolute_root(repo_root)
    source = _within_root(Path(path), root, location="JSON path")
    _reject_symlink_ancestors(source)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"expected a regular non-symlink JSON file: {source}")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error
    _json_value(value, location=source.as_posix())
    return value


def _hash_open_regular(path: Path):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open regular file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"expected a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"file changed while it was being hashed: {path}")
        return digest.hexdigest(), before.st_size, before.st_mtime_ns
    finally:
        os.close(descriptor)


def sha256_file(path, *, repo_root=".") -> str:
    root = _absolute_root(repo_root)
    source = _within_root(Path(path), root, location="file")
    _reject_symlink_ancestors(source)
    return _hash_open_regular(source)[0]


def _relative_symlink_target(value, *, location):
    _nonempty_string(value, location=location)
    if "\\" in value or Path(value).is_absolute():
        raise ValueError(f"{location} must be a relative symlink target")
    return value


def _resolved_symlink(path: Path, root: Path):
    _reject_symlink_ancestors(path, allow_leaf=True)
    if not path.is_symlink():
        raise ValueError(f"expected a symbolic link: {path}")
    target = _relative_symlink_target(os.readlink(path), location="symlink target")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(f"broken or cyclic symlink is forbidden: {path}") from error
    _within_root(resolved, root, location="resolved symlink")
    _reject_symlink_ancestors(resolved)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"symlink must resolve to one regular file: {path}")
    return target, resolved


def _build_file_record(path, root: Path, *, allow_leaf_symlink=False):
    source = _within_root(Path(path), root, location="input file")
    portable = _portable_from_path(source, root, location="input file")
    _reject_symlink_ancestors(source, allow_leaf=allow_leaf_symlink)
    if source.is_symlink():
        if not allow_leaf_symlink:
            raise ValueError(f"input symlinks are forbidden here: {source}")
        target, resolved = _resolved_symlink(source, root)
        digest, size, _ = _hash_open_regular(resolved)
        if os.readlink(source) != target or source.resolve(strict=True) != resolved:
            raise RuntimeError(f"symlink changed while its target was hashed: {source}")
        return {
            "kind": "relative_symlink",
            "path": portable,
            "size_bytes": size,
            "sha256": digest,
            "symlink_target": target,
            "resolved_path": _portable_from_path(
                resolved, root, location="resolved symlink"
            ),
        }
    if not source.is_file():
        raise FileNotFoundError(f"input is not a regular file: {source}")
    digest, size, _ = _hash_open_regular(source)
    return {
        "kind": "regular",
        "path": portable,
        "size_bytes": size,
        "sha256": digest,
    }


def _validate_file_record(record, *, location, allow_symlink=False):
    if not isinstance(record, dict):
        raise ValueError(f"{location} must be an object")
    kind = record.get("kind")
    expected = _FILE_SYMLINK_FIELDS if kind == "relative_symlink" else _FILE_REGULAR_FIELDS
    _exact_fields(record, expected, location=location)
    if kind not in {"regular", "relative_symlink"}:
        raise ValueError(f"{location}.kind is unsupported")
    if kind == "relative_symlink" and not allow_symlink:
        raise ValueError(f"{location} cannot be a symlink binding")
    portable_path(record["path"], location=f"{location}.path")
    _nonnegative_int(record["size_bytes"], location=f"{location}.size_bytes")
    _digest(record["sha256"], location=f"{location}.sha256")
    if kind == "relative_symlink":
        _relative_symlink_target(
            record["symlink_target"], location=f"{location}.symlink_target"
        )
        portable_path(record["resolved_path"], location=f"{location}.resolved_path")


def _verify_file_record(record, root: Path, *, mode, location, allow_symlink=False):
    _validate_file_record(record, location=location, allow_symlink=allow_symlink)
    if mode not in {"fast", "full"}:
        raise ValueError("verification mode must be 'fast' or 'full'")
    source = _repo_path(root, record["path"], location=f"{location}.path")
    _reject_symlink_ancestors(source, allow_leaf=allow_symlink)
    if record["kind"] == "relative_symlink":
        target, payload = _resolved_symlink(source, root)
        if target != record["symlink_target"]:
            raise ValueError(f"{location} symlink target changed")
        resolved_portable = _portable_from_path(
            payload, root, location=f"{location}.resolved_path"
        )
        if resolved_portable != record["resolved_path"]:
            raise ValueError(f"{location} resolves to a different repository file")
    else:
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"locked regular file is missing: {source}")
        payload = source
    if mode == "full":
        digest, size, _ = _hash_open_regular(payload)
        if size != record["size_bytes"] or digest != record["sha256"]:
            raise ValueError(f"{location} content differs from its lock")
    else:
        current = payload.stat()
        if current.st_size != record["size_bytes"]:
            raise ValueError(f"{location} fast metadata differs from its lock")
    return payload


def _atomic_write_new(path, payload, *, repo_root="."):
    root = _absolute_root(repo_root)
    destination = _within_root(Path(path), root, location="output path")
    _reject_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlink_ancestors(destination)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {destination}") from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination,
        "sha256": _hash_open_regular(destination)[0],
        "manifest": payload,
    }


def science_projection(config: Mapping) -> dict:
    """Project a campaign config onto science-bearing fields without a cycle."""

    if not isinstance(config, Mapping):
        raise ValueError("campaign config must be an object")
    _json_value(dict(config), location="campaign config")
    projected = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in NON_SCIENTIFIC_CONFIG_FIELDS
    }
    # Preview configs historically relied on the worker default. Materialize
    # that scientific default so sealing the lock and then spelling out
    # ``data_root: data`` does not create a bootstrap cycle.
    projected.setdefault("data_root", "data")
    if not isinstance(projected.get("campaign_id"), str):
        raise ValueError("campaign config must contain a string campaign_id")
    if not isinstance(projected.get("conditions"), list) or not projected["conditions"]:
        raise ValueError("campaign config must contain non-empty conditions")
    return {
        "science_projection_schema_version": SCIENCE_PROJECTION_SCHEMA_VERSION,
        "config": projected,
    }


def science_projection_sha256(config: Mapping) -> str:
    return logical_sha256(science_projection(config))


def _sample_id_sha256(sample_ids):
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _loader_order_rows(samples):
    return [
        {
            "index": row["index"],
            "sample_id": row["sample_id"],
            "prompt_index": row["prompt_index"],
            "image_path": row["image"]["path"],
            "mask_path": row["mask"]["path"],
        }
        for row in samples
    ]


def _dataset_layout(dataset, data_root: Path):
    from selectseg import data as data_module

    if dataset not in EVAL_DATASETS:
        raise ValueError(f"dataset must be one of {list(EVAL_DATASETS)}")
    spec = data_module.SPECS[dataset]
    split = spec.eval_split
    base = data_module._base_dataset(spec, data_root, split)
    expected_split, expected_class = DATASET_PROTOCOL[dataset]
    if split != expected_split or type(base).__name__ != expected_class:
        raise RuntimeError(f"dataset protocol changed for {dataset}")

    rows = []
    if dataset == "pet":
        selection = data_root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
        selection_paths = [selection]
        for index, (image_id, species) in enumerate(base.samples):
            sample_id = base.sample_id(index)
            if sample_id != image_id:
                raise RuntimeError("Pet loader sample_id differs from its selection row")
            rows.append(
                (
                    index,
                    sample_id,
                    base.SPECIES_TO_PROMPT[species],
                    base.images_dir / f"{image_id}.jpg",
                    base.trimaps_dir / f"{image_id}.png",
                )
            )
    else:
        selection_paths = []
        for index, (stem, image_path, mask_path) in enumerate(base.samples):
            sample_id = base.sample_id(index)
            if sample_id != stem:
                raise RuntimeError(f"{dataset} loader sample_id differs from sample tuple")
            rows.append((index, sample_id, 0, Path(image_path), Path(mask_path)))
    return split, type(base).__name__, selection_paths, rows


def build_dataset_component(
    dataset,
    *,
    data_root="data",
    output_path,
    repo_root=".",
):
    """Hash one eval cohort in exactly the order used by its real loader."""

    root = _absolute_root(repo_root)
    data_root_portable = portable_path(str(data_root), location="data_root").as_posix()
    data_root_path = _repo_path(root, data_root_portable, location="data_root")
    _reject_symlink_ancestors(data_root_path)
    if not data_root_path.is_dir() or data_root_path.is_symlink():
        raise FileNotFoundError(f"data_root must be a regular directory: {data_root_path}")
    split, loader_class, selection_paths, raw_rows = _dataset_layout(
        dataset, data_root_path
    )
    samples = []
    for index, sample_id, prompt_index, image_path, mask_path in raw_rows:
        _nonempty_string(sample_id, location=f"{dataset}.sample_id")
        samples.append(
            {
                "index": index,
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "image": _build_file_record(image_path, root),
                "mask": _build_file_record(mask_path, root),
            }
        )
    sample_ids = [row["sample_id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{dataset} eval loader contains duplicate sample IDs")
    payload = {
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "component_kind": "eval_dataset",
        "dataset": dataset,
        "data_root": data_root_portable,
        "split": split,
        "loader_class": loader_class,
        "sample_count": len(samples),
        "sample_id_sha256": _sample_id_sha256(sample_ids),
        "loader_order_sha256": logical_sha256(_loader_order_rows(samples)),
        "selection_files": [
            _build_file_record(path, root) for path in selection_paths
        ],
        "samples": samples,
    }
    _validate_dataset_component(payload)
    return _atomic_write_new(output_path, payload, repo_root=root)


def _validate_dataset_component(component):
    _exact_fields(component, _DATASET_FIELDS, location="dataset component")
    if component["component_schema_version"] != COMPONENT_SCHEMA_VERSION:
        raise ValueError("unsupported dataset component schema")
    if component["component_kind"] != "eval_dataset":
        raise ValueError("component is not an eval dataset")
    dataset = component["dataset"]
    if dataset not in EVAL_DATASETS:
        raise ValueError("dataset component names an unsupported dataset")
    expected_split, expected_loader = DATASET_PROTOCOL[dataset]
    if component["split"] != expected_split or component["loader_class"] != expected_loader:
        raise ValueError(f"dataset component protocol differs for {dataset}")
    portable_path(component["data_root"], location="dataset component.data_root")
    count = _nonnegative_int(
        component["sample_count"], location="dataset component.sample_count"
    )
    _digest(component["sample_id_sha256"], location="sample_id_sha256")
    _digest(component["loader_order_sha256"], location="loader_order_sha256")
    if not isinstance(component["selection_files"], list):
        raise ValueError("dataset component.selection_files must be a list")
    expected_selection_count = 1 if dataset == "pet" else 0
    if len(component["selection_files"]) != expected_selection_count:
        raise ValueError(f"{dataset} has the wrong selection-file count")
    for index, record in enumerate(component["selection_files"]):
        _validate_file_record(record, location=f"selection_files[{index}]")
    if not isinstance(component["samples"], list) or len(component["samples"]) != count:
        raise ValueError("dataset sample count does not match its rows")
    ids = []
    for index, row in enumerate(component["samples"]):
        _exact_fields(
            row,
            frozenset({"index", "sample_id", "prompt_index", "image", "mask"}),
            location=f"samples[{index}]",
        )
        if row["index"] != index:
            raise ValueError("dataset sample indices must be contiguous loader order")
        ids.append(_nonempty_string(row["sample_id"], location=f"samples[{index}].sample_id"))
        if isinstance(row["prompt_index"], bool) or not isinstance(
            row["prompt_index"], int
        ):
            raise ValueError(f"samples[{index}].prompt_index must be an integer")
        _validate_file_record(row["image"], location=f"samples[{index}].image")
        _validate_file_record(row["mask"], location=f"samples[{index}].mask")
    if len(ids) != len(set(ids)):
        raise ValueError("dataset component contains duplicate sample IDs")
    if _sample_id_sha256(ids) != component["sample_id_sha256"]:
        raise ValueError("dataset component sample-ID digest is inconsistent")
    if logical_sha256(_loader_order_rows(component["samples"])) != component[
        "loader_order_sha256"
    ]:
        raise ValueError("dataset component loader-order digest is inconsistent")
    return component


def _verify_dataset_component(component, root: Path, *, mode):
    _validate_dataset_component(component)
    data_root = _repo_path(root, component["data_root"], location="data_root")
    split, loader_class, selection_paths, raw_rows = _dataset_layout(
        component["dataset"], data_root
    )
    if split != component["split"] or loader_class != component["loader_class"]:
        raise ValueError("current eval loader protocol differs from dataset component")
    current_order = [
        {
            "index": index,
            "sample_id": sample_id,
            "prompt_index": prompt_index,
            "image_path": _portable_from_path(image, root, location="loader image"),
            "mask_path": _portable_from_path(mask, root, location="loader mask"),
        }
        for index, sample_id, prompt_index, image, mask in raw_rows
    ]
    if logical_sha256(current_order) != component["loader_order_sha256"]:
        raise ValueError("current eval loader order differs from dataset component")
    expected_selection = [record["path"] for record in component["selection_files"]]
    current_selection = [
        _portable_from_path(path, root, location="selection file")
        for path in selection_paths
    ]
    if current_selection != expected_selection:
        raise ValueError("current selection-file paths differ from dataset component")
    for index, record in enumerate(component["selection_files"]):
        _verify_file_record(
            record, root, mode=mode, location=f"selection_files[{index}]"
        )
    for index, row in enumerate(component["samples"]):
        _verify_file_record(
            row["image"], root, mode=mode, location=f"samples[{index}].image"
        )
        _verify_file_record(
            row["mask"], root, mode=mode, location=f"samples[{index}].mask"
        )
    return component["sample_count"]


def _paths_sorted_unique(paths, root, *, location):
    result = sorted(
        {
            _portable_from_path(
                _repo_path(root, str(path), location=location)
                if not Path(path).is_absolute()
                else Path(path),
                root,
                location=location,
            )
            for path in paths
        }
    )
    if not result:
        raise ValueError(f"{location} cannot be empty")
    return result


def build_source_component(paths, *, output_path, repo_root="."):
    root = _absolute_root(repo_root)
    normalized = _paths_sorted_unique(paths, root, location="source path")
    payload = {
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "component_kind": "source",
        "files": [
            _build_file_record(_repo_path(root, path, location="source path"), root)
            for path in normalized
        ],
    }
    _validate_source_component(payload)
    return _atomic_write_new(output_path, payload, repo_root=root)


def _validate_source_component(component):
    _exact_fields(component, _SOURCE_FIELDS, location="source component")
    if component["component_schema_version"] != COMPONENT_SCHEMA_VERSION:
        raise ValueError("unsupported source component schema")
    if component["component_kind"] != "source":
        raise ValueError("component is not source")
    if not isinstance(component["files"], list) or not component["files"]:
        raise ValueError("source component files cannot be empty")
    for index, record in enumerate(component["files"]):
        _validate_file_record(record, location=f"source.files[{index}]")
    paths = [record["path"] for record in component["files"]]
    if paths != sorted(set(paths)):
        raise ValueError("source component files must be sorted and unique")
    return component


def build_base_model_component(entries, *, output_path, repo_root="."):
    """Build model-file bindings; CLIPSeg snapshot leaf symlinks are supported."""

    root = _absolute_root(repo_root)
    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {"model", "path"}:
            raise ValueError(f"base model entry {index} must contain model and path")
        model = _nonempty_string(entry["model"], location=f"entries[{index}].model")
        path = str(entry["path"])
        source = (
            _repo_path(root, path, location=f"entries[{index}].path")
            if not Path(path).is_absolute()
            else _within_root(Path(path), root, location=f"entries[{index}].path")
        )
        normalized.append(
            {"model": model, "file": _build_file_record(source, root, allow_leaf_symlink=True)}
        )
    normalized.sort(key=lambda item: (item["model"], item["file"]["path"]))
    payload = {
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "component_kind": "base_models",
        "entries": normalized,
    }
    _validate_base_model_component(payload)
    return _atomic_write_new(output_path, payload, repo_root=root)


def base_model_entries_from_seed_extension_lock(path, *, repo_root="."):
    """Reuse the eight base-model paths already sealed by the seed extension."""

    lock = load_strict_json(path, repo_root=repo_root)
    entries = lock.get("base_model_files") if isinstance(lock, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("seed-extension lock has no base_model_files")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"base_model_files[{index}] is malformed")
        _digest(entry["sha256"], location=f"base_model_files[{index}].sha256")
        path_value = portable_path(
            entry["path"], location=f"base_model_files[{index}].path"
        ).as_posix()
        if "models--CIDAS--clipseg" in path_value:
            model = "clipseg"
        elif "deeplabv3_resnet50" in path_value:
            model = "deeplabv3"
        else:
            raise ValueError(f"cannot assign base model file to a model: {path_value}")
        result.append({"model": model, "path": path_value})
    return result


def _validate_base_model_component(component):
    _exact_fields(component, _BASE_MODEL_FIELDS, location="base-model component")
    if component["component_schema_version"] != COMPONENT_SCHEMA_VERSION:
        raise ValueError("unsupported base-model component schema")
    if component["component_kind"] != "base_models":
        raise ValueError("component is not base_models")
    if not isinstance(component["entries"], list) or not component["entries"]:
        raise ValueError("base-model entries cannot be empty")
    keys = []
    for index, entry in enumerate(component["entries"]):
        _exact_fields(entry, frozenset({"model", "file"}), location=f"entries[{index}]")
        model = _nonempty_string(entry["model"], location=f"entries[{index}].model")
        _validate_file_record(
            entry["file"], location=f"entries[{index}].file", allow_symlink=True
        )
        keys.append((model, entry["file"]["path"]))
    if keys != sorted(set(keys)):
        raise ValueError("base-model entries must be sorted and unique")
    return component


def build_checkpoint_component(config, *, output_path, repo_root="."):
    root = _absolute_root(repo_root)
    if not isinstance(config, Mapping):
        config = load_strict_json(config, repo_root=root)
    projection = science_projection(config)["config"]
    entries = []
    cache = {}
    for index, condition in enumerate(projection["conditions"]):
        if not isinstance(condition, dict):
            raise ValueError(f"conditions[{index}] must be an object")
        required = {"dataset", "model", "condition", "checkpoint"}
        if not required <= set(condition):
            raise ValueError(f"conditions[{index}] lacks checkpoint identity fields")
        row = {
            "dataset": _nonempty_string(condition["dataset"], location="dataset"),
            "model": _nonempty_string(condition["model"], location="model"),
            "condition": _nonempty_string(condition["condition"], location="condition"),
            "checkpoint": None,
        }
        checkpoint = condition["checkpoint"]
        if checkpoint is not None:
            portable = portable_path(checkpoint, location="checkpoint").as_posix()
            if portable not in cache:
                cache[portable] = _build_file_record(
                    _repo_path(root, portable, location="checkpoint"), root
                )
            row["checkpoint"] = copy.deepcopy(cache[portable])
        entries.append(row)
    entries.sort(key=lambda row: (row["dataset"], row["model"], row["condition"]))
    payload = {
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "component_kind": "checkpoints",
        "entries": entries,
    }
    _validate_checkpoint_component(payload)
    return _atomic_write_new(output_path, payload, repo_root=root)


def _validate_checkpoint_component(component):
    _exact_fields(component, _CHECKPOINT_FIELDS, location="checkpoint component")
    if component["component_schema_version"] != COMPONENT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint component schema")
    if component["component_kind"] != "checkpoints":
        raise ValueError("component is not checkpoints")
    if not isinstance(component["entries"], list) or not component["entries"]:
        raise ValueError("checkpoint entries cannot be empty")
    keys = []
    for index, entry in enumerate(component["entries"]):
        _exact_fields(
            entry,
            frozenset({"dataset", "model", "condition", "checkpoint"}),
            location=f"checkpoint.entries[{index}]",
        )
        key = tuple(
            _nonempty_string(entry[field], location=f"entries[{index}].{field}")
            for field in ("dataset", "model", "condition")
        )
        keys.append(key)
        if entry["checkpoint"] is not None:
            _validate_file_record(
                entry["checkpoint"], location=f"entries[{index}].checkpoint"
            )
    if keys != sorted(set(keys)):
        raise ValueError("checkpoint entries must be sorted and uniquely keyed")
    return component


def collect_environment(packages: Sequence[str] = DEFAULT_ENVIRONMENT_PACKAGES):
    names = tuple(packages)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("environment package names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("environment package names must be unique")
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_executable_sha256": _hash_open_regular(
            Path(sys.executable).resolve()
        )[0],
    }
    for name in names:
        versions[name] = importlib.metadata.version(name)
    if "torch" in names:
        import torch

        versions["cuda_runtime"] = torch.version.cuda
    return versions


def build_environment_component(
    *,
    output_path,
    repo_root=".",
    packages: Sequence[str] = DEFAULT_ENVIRONMENT_PACKAGES,
    environment=None,
):
    package_names = list(packages)
    values = collect_environment(package_names) if environment is None else copy.deepcopy(environment)
    payload = {
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "component_kind": "environment",
        "packages": package_names,
        "environment": values,
    }
    _validate_environment_component(payload)
    return _atomic_write_new(output_path, payload, repo_root=repo_root)


def _validate_environment_component(component):
    _exact_fields(component, _ENVIRONMENT_FIELDS, location="environment component")
    if component["component_schema_version"] != COMPONENT_SCHEMA_VERSION:
        raise ValueError("unsupported environment component schema")
    if component["component_kind"] != "environment":
        raise ValueError("component is not environment")
    packages = component["packages"]
    if (
        not isinstance(packages, list)
        or not packages
        or any(not isinstance(item, str) or not item for item in packages)
        or len(packages) != len(set(packages))
    ):
        raise ValueError("environment packages must be a non-empty unique list")
    environment = component["environment"]
    if not isinstance(environment, dict) or not environment:
        raise ValueError("environment values must be a non-empty object")
    _json_value(environment, location="environment")
    if set(packages) - set(environment):
        raise ValueError("environment values omit a locked package")
    return component


def _component_binding(path, root: Path, *, expected_kind):
    source = _within_root(Path(path), root, location=f"{expected_kind} component")
    component = load_strict_json(source, repo_root=root)
    validators = {
        "source": _validate_source_component,
        "base_models": _validate_base_model_component,
        "checkpoints": _validate_checkpoint_component,
        "environment": _validate_environment_component,
        "eval_dataset": _validate_dataset_component,
    }
    validators[expected_kind](component)
    return {
        "path": _portable_from_path(source, root, location=f"{expected_kind} component"),
        "sha256": _hash_open_regular(source)[0],
    }, component


def _validate_binding(binding, *, location):
    _exact_fields(binding, _BINDING_FIELDS, location=location)
    portable_path(binding["path"], location=f"{location}.path")
    _digest(binding["sha256"], location=f"{location}.sha256")


def _condition_key(condition):
    return condition["dataset"], condition["model"], condition["condition"]


def _config_condition_map(projected_config):
    result = {}
    for index, condition in enumerate(projected_config.get("conditions", [])):
        if not isinstance(condition, dict):
            raise ValueError(f"conditions[{index}] must be an object")
        required = {"dataset", "model", "condition", "checkpoint", "expected_num_samples"}
        if not required <= set(condition):
            raise ValueError(f"conditions[{index}] lacks required scientific fields")
        key = _condition_key(condition)
        if key in result:
            raise ValueError(f"duplicate campaign condition {key}")
        result[key] = condition
    if not result:
        raise ValueError("campaign has no conditions")
    return result


def _check_component_consistency(projected, datasets, base_models, checkpoints):
    conditions = _config_condition_map(projected)
    dataset_counts = {name: component["sample_count"] for name, component in datasets.items()}
    if set(dataset_counts) != set(EVAL_DATASETS):
        raise ValueError("root lock requires exactly the five binary eval datasets")
    checkpoint_rows = {_condition_key(row): row for row in checkpoints["entries"]}
    if set(checkpoint_rows) != set(conditions):
        raise ValueError("checkpoint component does not cover the exact condition grid")
    available_models = {row["model"] for row in base_models["entries"]}
    for key, condition in conditions.items():
        dataset, model, _ = key
        if dataset not in dataset_counts:
            raise ValueError(f"condition refers to an unlocked dataset: {dataset}")
        if model not in available_models:
            raise ValueError(f"condition refers to an unlocked base model: {model}")
        _validated_condition_counts(condition, dataset_counts[dataset], key=key)
        expected_path = condition["checkpoint"]
        record = checkpoint_rows[key]["checkpoint"]
        actual_path = None if record is None else record["path"]
        if expected_path != actual_path:
            raise ValueError(f"condition {key} checkpoint differs from checkpoint component")


def _validated_condition_counts(condition, dataset_sample_count, *, key):
    """Validate full-cohort and consumed-artifact counts for one condition."""

    expected = condition["expected_num_samples"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError(f"condition {key} has invalid expected_num_samples")
    expected_dataset = condition.get("expected_dataset_samples", expected)
    if (
        isinstance(expected_dataset, bool)
        or not isinstance(expected_dataset, int)
        or expected_dataset < 1
    ):
        raise ValueError(f"condition {key} has invalid expected_dataset_samples")
    if expected_dataset != dataset_sample_count:
        raise ValueError(f"condition {key} sample count differs from dataset component")
    freeze_limit = condition.get("freeze_limit")
    if freeze_limit is None:
        if expected != expected_dataset:
            raise ValueError(
                f"condition {key} needs freeze_limit for a development subset"
            )
    elif (
        isinstance(freeze_limit, bool)
        or not isinstance(freeze_limit, int)
        or freeze_limit < 1
        or freeze_limit != expected
        or freeze_limit > dataset_sample_count
    ):
        raise ValueError(f"condition {key} has inconsistent freeze_limit")
    return {
        "dataset_sample_count": dataset_sample_count,
        "expected_dataset_samples": expected_dataset,
        "sample_count": expected,
        "expected_num_samples": expected,
        "freeze_limit": freeze_limit,
    }


def build_root_lock(
    config_path,
    *,
    dataset_components: Mapping[str, str | Path],
    source_component,
    base_model_component,
    checkpoint_component,
    environment_component,
    output_path,
    repo_root=".",
):
    """Seal the config projection and small bindings to all component manifests."""

    root = _absolute_root(repo_root)
    config_source = _within_root(Path(config_path), root, location="campaign config")
    config = load_strict_json(config_source, repo_root=root)
    projection = science_projection(config)
    if set(dataset_components) != set(EVAL_DATASETS):
        raise ValueError("dataset_components must name exactly the five eval datasets")
    dataset_bindings = []
    dataset_payloads = {}
    for dataset in EVAL_DATASETS:
        binding, component = _component_binding(
            dataset_components[dataset], root, expected_kind="eval_dataset"
        )
        if component["dataset"] != dataset:
            raise ValueError(f"dataset component binding {dataset} names another dataset")
        dataset_bindings.append({"dataset": dataset, **binding})
        dataset_payloads[dataset] = component
    source_binding, _ = _component_binding(source_component, root, expected_kind="source")
    base_binding, base_payload = _component_binding(
        base_model_component, root, expected_kind="base_models"
    )
    checkpoint_binding, checkpoint_payload = _component_binding(
        checkpoint_component, root, expected_kind="checkpoints"
    )
    environment_binding, _ = _component_binding(
        environment_component, root, expected_kind="environment"
    )
    projected_config = projection["config"]
    _check_component_consistency(
        projected_config, dataset_payloads, base_payload, checkpoint_payload
    )
    payload = {
        "scientific_input_lock_schema_version": ROOT_LOCK_SCHEMA_VERSION,
        "campaign_id": projected_config["campaign_id"],
        "science_config": {
            "path": _portable_from_path(config_source, root, location="campaign config"),
            "projection_schema_version": SCIENCE_PROJECTION_SCHEMA_VERSION,
            "projection_sha256": logical_sha256(projection),
        },
        "components": {
            "datasets": dataset_bindings,
            "source": source_binding,
            "base_models": base_binding,
            "checkpoints": checkpoint_binding,
            "environment": environment_binding,
        },
    }
    _validate_root_lock(payload)
    return _atomic_write_new(output_path, payload, repo_root=root)


def _validate_root_lock(lock):
    _exact_fields(lock, _ROOT_FIELDS, location="scientific-input root lock")
    if lock["scientific_input_lock_schema_version"] != ROOT_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported scientific-input root-lock schema")
    _nonempty_string(lock["campaign_id"], location="root lock.campaign_id")
    science = lock["science_config"]
    _exact_fields(
        science,
        frozenset({"path", "projection_schema_version", "projection_sha256"}),
        location="root lock.science_config",
    )
    portable_path(science["path"], location="science_config.path")
    if science["projection_schema_version"] != SCIENCE_PROJECTION_SCHEMA_VERSION:
        raise ValueError("unsupported science projection schema")
    _digest(science["projection_sha256"], location="science_config.projection_sha256")
    components = lock["components"]
    _exact_fields(
        components,
        frozenset({"datasets", "source", "base_models", "checkpoints", "environment"}),
        location="root lock.components",
    )
    datasets = components["datasets"]
    if not isinstance(datasets, list) or len(datasets) != len(EVAL_DATASETS):
        raise ValueError("root lock must bind five dataset components")
    names = []
    for index, binding in enumerate(datasets):
        _exact_fields(
            binding,
            frozenset({"dataset", "path", "sha256"}),
            location=f"components.datasets[{index}]",
        )
        names.append(binding["dataset"])
        _validate_binding(
            {"path": binding["path"], "sha256": binding["sha256"]},
            location=f"components.datasets[{index}]",
        )
    if tuple(names) != EVAL_DATASETS:
        raise ValueError("dataset component bindings must use canonical dataset order")
    for name in ("source", "base_models", "checkpoints", "environment"):
        _validate_binding(components[name], location=f"components.{name}")
    return lock


def _load_bound_component(binding, root: Path, *, expected_kind):
    _validate_binding(binding, location=f"{expected_kind} binding")
    path = _repo_path(root, binding["path"], location=f"{expected_kind} binding.path")
    component = load_strict_json(path, repo_root=root)
    actual = _hash_open_regular(path)[0]
    if actual != binding["sha256"]:
        raise ValueError(f"{expected_kind} component manifest hash mismatch")
    validators = {
        "source": _validate_source_component,
        "base_models": _validate_base_model_component,
        "checkpoints": _validate_checkpoint_component,
        "environment": _validate_environment_component,
        "eval_dataset": _validate_dataset_component,
    }
    validators[expected_kind](component)
    return component


def load_root_lock(
    path,
    *,
    repo_root=".",
    expected_sha256=None,
    verify_component_manifests=True,
):
    root = _absolute_root(repo_root)
    source = _within_root(Path(path), root, location="root lock")
    lock = load_strict_json(source, repo_root=root)
    actual = _hash_open_regular(source)[0]
    if expected_sha256 is not None and actual != _digest(
        expected_sha256, location="expected root-lock SHA-256"
    ):
        raise ValueError("scientific-input root-lock hash mismatch")
    _validate_root_lock(lock)
    config_path = _repo_path(root, lock["science_config"]["path"], location="science config")
    config = load_strict_json(config_path, repo_root=root)
    projection = science_projection(config)
    if logical_sha256(projection) != lock["science_config"]["projection_sha256"]:
        raise ValueError("campaign science projection differs from root lock")
    if projection["config"]["campaign_id"] != lock["campaign_id"]:
        raise ValueError("campaign ID differs between config and root lock")
    result = {
        "path": source,
        "sha256": actual,
        "lock": lock,
        "config": config,
        "projection": projection,
        "components": {},
    }
    if verify_component_manifests:
        datasets = {}
        for binding in lock["components"]["datasets"]:
            datasets[binding["dataset"]] = _load_bound_component(
                {"path": binding["path"], "sha256": binding["sha256"]},
                root,
                expected_kind="eval_dataset",
            )
        result["components"]["datasets"] = datasets
        for key, kind in (
            ("source", "source"),
            ("base_models", "base_models"),
            ("checkpoints", "checkpoints"),
            ("environment", "environment"),
        ):
            result["components"][key] = _load_bound_component(
                lock["components"][key], root, expected_kind=kind
            )
        _check_component_consistency(
            projection["config"],
            datasets,
            result["components"]["base_models"],
            result["components"]["checkpoints"],
        )
    return result


def _verify_source(component, root, *, mode):
    for index, record in enumerate(component["files"]):
        _verify_file_record(record, root, mode=mode, location=f"source.files[{index}]")


def _verify_base_models(component, root, *, mode, model=None):
    selected = [row for row in component["entries"] if model is None or row["model"] == model]
    if model is not None and not selected:
        raise ValueError(f"no locked base-model files for {model}")
    for index, row in enumerate(selected):
        _verify_file_record(
            row["file"],
            root,
            mode=mode,
            location=f"base_models[{index}].file",
            allow_symlink=True,
        )


def _verify_checkpoints(component, root, *, mode, key=None):
    selected = [
        row for row in component["entries"] if key is None or _condition_key(row) == key
    ]
    if key is not None and len(selected) != 1:
        raise ValueError(f"missing unique checkpoint condition {key}")
    for index, row in enumerate(selected):
        if row["checkpoint"] is not None:
            _verify_file_record(
                row["checkpoint"],
                root,
                mode=mode,
                location=f"checkpoints[{index}]",
            )


def _verify_environment(component):
    actual = collect_environment(component["packages"])
    if actual != component["environment"]:
        raise ValueError(
            f"runtime environment differs from scientific lock: actual={actual}"
        )


def _normalized_mode(mode):
    if mode == "small":
        return "fast"
    if mode not in {"fast", "full"}:
        raise ValueError("verification mode must be 'small', 'fast', or 'full'")
    return mode


def _dataset_component_binding(lock, dataset):
    matches = [
        binding
        for binding in lock["components"]["datasets"]
        if binding["dataset"] == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"missing unique eval-dataset component for {dataset}")
    return matches[0]


def _verified_dataset_handle(root, lock, component, dataset, *, sample_limit=None):
    binding = _dataset_component_binding(lock, dataset)
    dataset_sample_count = component["sample_count"]
    if sample_limit is None:
        sample_limit = dataset_sample_count
    if (
        isinstance(sample_limit, bool)
        or not isinstance(sample_limit, int)
        or sample_limit < 1
        or sample_limit > dataset_sample_count
    ):
        raise ValueError("verified eval-dataset sample limit is inconsistent")
    selected_rows = component["samples"][:sample_limit]
    samples = {}
    for row in selected_rows:
        image = row["image"]
        mask = row["mask"]
        sample = LockedEvalSample(
            index=row["index"],
            sample_id=row["sample_id"],
            prompt_index=row["prompt_index"],
            image_path=image["path"],
            image_sha256=image["sha256"],
            image_size_bytes=image["size_bytes"],
            mask_path=mask["path"],
            mask_sha256=mask["sha256"],
            mask_size_bytes=mask["size_bytes"],
        )
        samples[sample.sample_id] = sample
    if len(samples) != sample_limit:
        raise ValueError("eval-dataset sample lookup is not one-to-one")
    return VerifiedEvalDataset(
        repo_root=root,
        dataset=dataset,
        component_sha256=binding["sha256"],
        dataset_sample_count=dataset_sample_count,
        sample_count=sample_limit,
        sample_id_sha256=_sample_id_sha256(
            [row["sample_id"] for row in selected_rows]
        ),
        loader_order_sha256=logical_sha256(_loader_order_rows(selected_rows)),
        # Keep the handle picklable for torch DataLoader worker processes.
        # Each row remains an immutable dataclass.
        samples_by_id=dict(samples),
    )


def _condition_input_hashes(binding, dataset):
    lock = binding["lock"]
    dataset_binding = _dataset_component_binding(lock, dataset)
    components = lock["components"]
    # Exactly seven roots are propagated into every frozen artifact.  Large
    # per-sample hashes stay in the bound dataset component, not this object.
    return {
        "root_lock_sha256": binding["sha256"],
        "science_projection_sha256": lock["science_config"][
            "projection_sha256"
        ],
        "eval_dataset_component_sha256": dataset_binding["sha256"],
        "source_component_sha256": components["source"]["sha256"],
        "base_model_component_sha256": components["base_models"]["sha256"],
        "checkpoint_component_sha256": components["checkpoints"]["sha256"],
        "environment_component_sha256": components["environment"]["sha256"],
    }


def condition_input_identity(binding, *, dataset, model, condition):
    """Derive one condition identity from an already loaded/verified root lock."""

    required = {"sha256", "lock", "projection", "components"}
    if not isinstance(binding, Mapping) or not required <= set(binding):
        raise TypeError("binding must come from load_root_lock")
    key = (dataset, model, condition)
    if key not in _config_condition_map(binding["projection"]["config"]):
        raise ValueError(f"condition is outside the locked campaign: {key}")
    if dataset not in binding["components"].get("datasets", {}):
        raise ValueError(f"condition dataset is not locked: {dataset}")
    hashes = _condition_input_hashes(binding, dataset)
    return {
        "scientific_input_hashes": hashes,
        "scientific_input_sha256": logical_sha256(hashes),
    }


def verified_eval_dataset(
    path,
    *,
    dataset,
    repo_root=".",
    expected_sha256=None,
    mode="small",
):
    """Return a verified sample-ID lookup for one locked eval cohort.

    ``small`` is the intended per-job guard and aliases metadata-based
    ``fast`` verification.  Use ``full`` for the authoritative byte audit.
    """

    verification_mode = _normalized_mode(mode)
    binding = load_root_lock(
        path, repo_root=repo_root, expected_sha256=expected_sha256
    )
    component = binding["components"]["datasets"].get(dataset)
    if component is None:
        raise ValueError(f"eval dataset is outside the locked campaign: {dataset}")
    root = _absolute_root(repo_root)
    _verify_dataset_component(component, root, mode=verification_mode)
    return _verified_dataset_handle(root, binding["lock"], component, dataset)


def verify_sample_bytes(
    verified_dataset: VerifiedEvalDataset,
    sample_id,
    image_path,
    mask_path,
    *,
    mode="full",
):
    """Verify paths and bytes for one loader item against its sample-ID lock."""

    if not isinstance(verified_dataset, VerifiedEvalDataset):
        raise TypeError("verified_dataset must come from verified_eval_dataset")
    verification_mode = _normalized_mode(mode)
    try:
        sample = verified_dataset.samples_by_id[sample_id]
    except KeyError as error:
        raise ValueError(
            f"sample_id {sample_id!r} is outside locked {verified_dataset.dataset}"
        ) from error
    root = verified_dataset.repo_root
    actual_image = _portable_from_path(image_path, root, location="sample image")
    actual_mask = _portable_from_path(mask_path, root, location="sample mask")
    if actual_image != sample.image_path or actual_mask != sample.mask_path:
        raise ValueError(f"loader paths differ from the lock for sample {sample_id!r}")
    image_record = {
        "kind": "regular",
        "path": sample.image_path,
        "size_bytes": sample.image_size_bytes,
        "sha256": sample.image_sha256,
    }
    mask_record = {
        "kind": "regular",
        "path": sample.mask_path,
        "size_bytes": sample.mask_size_bytes,
        "sha256": sample.mask_sha256,
    }
    _verify_file_record(
        image_record, root, mode=verification_mode, location=f"{sample_id}.image"
    )
    _verify_file_record(
        mask_record, root, mode=verification_mode, location=f"{sample_id}.mask"
    )
    return {
        "sample_id": sample_id,
        "image_path": sample.image_path,
        "image_sha256": sample.image_sha256,
        "mask_path": sample.mask_path,
        "mask_sha256": sample.mask_sha256,
    }


def read_verified_sample_file(
    verified_dataset: VerifiedEvalDataset,
    sample_id,
    role,
    path,
):
    """Read one image/mask once while verifying its path and locked bytes."""

    if not isinstance(verified_dataset, VerifiedEvalDataset):
        raise TypeError("verified_dataset must come from verified_eval_dataset")
    if role not in {"image", "mask"}:
        raise ValueError("verified sample role must be 'image' or 'mask'")
    try:
        sample = verified_dataset.samples_by_id[sample_id]
    except KeyError as error:
        raise ValueError(
            f"sample_id {sample_id!r} is outside locked {verified_dataset.dataset}"
        ) from error
    expected_path = getattr(sample, f"{role}_path")
    expected_sha256 = getattr(sample, f"{role}_sha256")
    expected_size = getattr(sample, f"{role}_size_bytes")
    actual_path = _portable_from_path(
        path, verified_dataset.repo_root, location=f"sample {role}"
    )
    if actual_path != expected_path:
        raise ValueError(
            f"loader {role} path differs from the lock for sample {sample_id!r}"
        )
    source = _repo_path(
        verified_dataset.repo_root,
        expected_path,
        location=f"sample {role}",
    )
    _reject_symlink_ancestors(source)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open locked {role}: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"locked {role} is not a regular file: {source}")
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"locked {role} changed while read: {source}")
        if before.st_size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"sample {sample_id!r} {role} content differs from its lock"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_root_lock(path, *, repo_root=".", expected_sha256=None, mode="full"):
    """Verify all five cohorts and every common scientific input."""

    verification_mode = _normalized_mode(mode)
    binding = load_root_lock(
        path, repo_root=repo_root, expected_sha256=expected_sha256
    )
    root = _absolute_root(repo_root)
    counts = {}
    for dataset, component in binding["components"]["datasets"].items():
        counts[dataset] = _verify_dataset_component(
            component, root, mode=verification_mode
        )
    _verify_source(binding["components"]["source"], root, mode=verification_mode)
    _verify_base_models(
        binding["components"]["base_models"], root, mode=verification_mode
    )
    _verify_checkpoints(
        binding["components"]["checkpoints"], root, mode=verification_mode
    )
    _verify_environment(binding["components"]["environment"])
    return {
        "root_lock_sha256": binding["sha256"],
        "mode": mode,
        "verification_mode": verification_mode,
        "campaign_id": binding["lock"]["campaign_id"],
        "dataset_sample_counts": counts,
    }


def verify_condition(
    path,
    *,
    dataset,
    model,
    condition,
    repo_root=".",
    expected_sha256=None,
    mode="fast",
):
    """Serializable summary of :func:`verify_condition_inputs`."""

    result = verify_condition_inputs(
        path,
        dataset=dataset,
        model=model,
        condition=condition,
        repo_root=repo_root,
        expected_sha256=expected_sha256,
        mode=mode,
    )
    return {
        key: value
        for key, value in result.items()
        if key != "eval_dataset"
    }


def verify_condition_inputs(
    path,
    *,
    dataset,
    model,
    condition,
    repo_root=".",
    expected_sha256=None,
    mode="small",
):
    """Verify and expose all scientific bindings for one freeze experiment."""

    consume = mode == "consume"
    verification_mode = "consume" if consume else _normalized_mode(mode)
    dataset_mode = "fast" if consume else verification_mode
    small_input_mode = "full" if consume else verification_mode
    binding = load_root_lock(
        path, repo_root=repo_root, expected_sha256=expected_sha256
    )
    root = _absolute_root(repo_root)
    key = (dataset, model, condition)
    conditions = _config_condition_map(binding["projection"]["config"])
    if key not in conditions:
        raise ValueError(f"condition is outside the locked campaign: {key}")
    dataset_component = binding["components"]["datasets"].get(dataset)
    if dataset_component is None:
        raise ValueError(f"condition dataset is not locked: {dataset}")
    count = _verify_dataset_component(dataset_component, root, mode=dataset_mode)
    if consume:
        # Selection metadata is small and controls cohort membership, so hash
        # it before inference. Image/mask bytes are verified on their one
        # unavoidable read through ``read_verified_sample_file``.
        for index, record in enumerate(dataset_component["selection_files"]):
            _verify_file_record(
                record,
                root,
                mode="full",
                location=f"selection_files[{index}]",
            )
    counts = _validated_condition_counts(conditions[key], count, key=key)
    _verify_source(
        binding["components"]["source"], root, mode=small_input_mode
    )
    _verify_base_models(
        binding["components"]["base_models"],
        root,
        mode=small_input_mode,
        model=model,
    )
    _verify_checkpoints(
        binding["components"]["checkpoints"],
        root,
        mode=small_input_mode,
        key=key,
    )
    checkpoint_rows = [
        row
        for row in binding["components"]["checkpoints"]["entries"]
        if _condition_key(row) == key
    ]
    if len(checkpoint_rows) != 1:
        raise ValueError(f"missing unique checkpoint condition {key}")
    _verify_environment(binding["components"]["environment"])
    input_hashes = _condition_input_hashes(binding, dataset)
    eval_dataset = _verified_dataset_handle(
        root,
        binding["lock"],
        dataset_component,
        dataset,
        sample_limit=counts["sample_count"],
    )
    return {
        "root_lock_sha256": binding["sha256"],
        "mode": mode,
        "verification_mode": verification_mode,
        "campaign_id": binding["lock"]["campaign_id"],
        "dataset": dataset,
        "model": model,
        "condition": condition,
        **counts,
        "checkpoint": copy.deepcopy(checkpoint_rows[0]["checkpoint"]),
        "scientific_input_hashes": input_hashes,
        "scientific_input_sha256": logical_sha256(input_hashes),
        "eval_dataset": eval_dataset,
    }


def _parse_assignment(value, *, location):
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError(f"{location} must use KEY=PATH syntax")
    return key, path


def _print_result(result):
    print(
        json.dumps(
            {
                key: value.as_posix() if isinstance(value, Path) else value
                for key, value in result.items()
                if key != "manifest"
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("build-dataset")
    dataset_parser.add_argument("--dataset", choices=EVAL_DATASETS, required=True)
    dataset_parser.add_argument("--data-root", default="data")
    dataset_parser.add_argument("--output", required=True)

    source_parser = subparsers.add_parser("build-source")
    source_parser.add_argument("--path", action="append", required=True)
    source_parser.add_argument("--output", required=True)

    base_parser = subparsers.add_parser("build-base-models")
    group = base_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entry", action="append")
    group.add_argument("--seed-extension-lock")
    base_parser.add_argument("--output", required=True)

    checkpoints_parser = subparsers.add_parser("build-checkpoints")
    checkpoints_parser.add_argument("--config", required=True)
    checkpoints_parser.add_argument("--output", required=True)

    environment_parser = subparsers.add_parser("build-environment")
    environment_parser.add_argument("--package", action="append")
    environment_parser.add_argument("--output", required=True)

    root_parser = subparsers.add_parser("build-root")
    root_parser.add_argument("--config", required=True)
    root_parser.add_argument("--dataset-component", action="append", required=True)
    root_parser.add_argument("--source-component", required=True)
    root_parser.add_argument("--base-model-component", required=True)
    root_parser.add_argument("--checkpoint-component", required=True)
    root_parser.add_argument("--environment-component", required=True)
    root_parser.add_argument("--output", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--lock", required=True)
    verify_parser.add_argument("--expected-sha256")
    verify_parser.add_argument(
        "--mode",
        choices=("small", "fast", "full", "consume"),
        default="full",
    )
    verify_parser.add_argument("--dataset")
    verify_parser.add_argument("--model")
    verify_parser.add_argument("--condition")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    root = args.repo_root
    if args.command == "build-dataset":
        result = build_dataset_component(
            args.dataset,
            data_root=args.data_root,
            output_path=args.output,
            repo_root=root,
        )
    elif args.command == "build-source":
        result = build_source_component(args.path, output_path=args.output, repo_root=root)
    elif args.command == "build-base-models":
        if args.seed_extension_lock:
            entries = base_model_entries_from_seed_extension_lock(
                args.seed_extension_lock, repo_root=root
            )
        else:
            entries = [
                {"model": model, "path": path}
                for model, path in (
                    _parse_assignment(value, location="--entry") for value in args.entry
                )
            ]
        result = build_base_model_component(entries, output_path=args.output, repo_root=root)
    elif args.command == "build-checkpoints":
        result = build_checkpoint_component(
            args.config, output_path=args.output, repo_root=root
        )
    elif args.command == "build-environment":
        result = build_environment_component(
            output_path=args.output,
            repo_root=root,
            packages=args.package or DEFAULT_ENVIRONMENT_PACKAGES,
        )
    elif args.command == "build-root":
        datasets = dict(
            _parse_assignment(value, location="--dataset-component")
            for value in args.dataset_component
        )
        result = build_root_lock(
            args.config,
            dataset_components=datasets,
            source_component=args.source_component,
            base_model_component=args.base_model_component,
            checkpoint_component=args.checkpoint_component,
            environment_component=args.environment_component,
            output_path=args.output,
            repo_root=root,
        )
    else:
        condition_values = (args.dataset, args.model, args.condition)
        if any(value is not None for value in condition_values):
            if any(value is None for value in condition_values):
                raise SystemExit(
                    "--dataset, --model, and --condition must be supplied together"
                )
            result = verify_condition(
                args.lock,
                dataset=args.dataset,
                model=args.model,
                condition=args.condition,
                repo_root=root,
                expected_sha256=args.expected_sha256,
                mode=args.mode,
            )
        else:
            result = verify_root_lock(
                args.lock,
                repo_root=root,
                expected_sha256=args.expected_sha256,
                mode=args.mode,
            )
    _print_result(result)
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
