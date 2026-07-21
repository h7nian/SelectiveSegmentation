"""Strict schema, integrity, and atomicity tests for frozen binary maps."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from selectseg.artifacts import (
    ARTIFACT_TYPE,
    load_binary_artifact,
    sample_id_sha256,
    sha256_file,
    write_binary_artifact,
)


def _arrays(index, *, dtype=np.float32):
    probability = np.array([[0.1 + index / 10, 0.9], [0.5, 0.0]], dtype=dtype)
    truth = np.array([[0, 1], [index % 2, 0]], dtype=np.uint8)
    return probability, truth


def _write(root, *, sample_ids=("case-a", "case-b"), samples=None, **overrides):
    if samples is None:
        samples = [
            (sample_id, *_arrays(index)) for index, sample_id in enumerate(sample_ids)
        ]
    arguments = {
        "dataset": "toy",
        "condition": "clipseg-target",
        "model": "clipseg",
        "split": "test",
        "class_index": 1,
        "class_name": "lesion",
        "checkpoint": {
            "path": "outputs/train/checkpoint.pt",
            "sha256": "a" * 64,
            "size_bytes": 123,
        },
        "base_model": {"name": "clipseg", "source": "vendor/model"},
        "source_sha256": "b" * 64,
        "environment": {
            "packages": {
                "python": "3.12",
                "numpy": "test",
                "torch": "test",
                "torchvision": "test",
                "transformers": "test",
            },
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        "preprocessing": {
            "model_input": "square resize with antialiasing",
            "probability_to_native_mask": (
                "bilinear interpolation with align_corners=False"
            ),
        },
        "cohort": "all held-out images",
        "sample_ids": list(sample_ids),
        "samples": samples,
        "command": ["python", "-m", "selectseg.pipeline.freeze"],
        "created_utc": "2026-07-19T12:00:00+00:00",
    }
    arguments.update(overrides)
    return write_binary_artifact(root, **arguments)


def _read_manifest(path):
    return json.loads(Path(path).read_text())


def _write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")


def _scientific_input(**overrides):
    value = {
        "root_lock_sha256": "1" * 64,
        "condition_input_sha256": "2" * 64,
        "science_projection_sha256": "3" * 64,
        "eval_dataset_component_sha256": "4" * 64,
        "source_component_sha256": "5" * 64,
        "base_model_component_sha256": "6" * 64,
        "checkpoint_component_sha256": "7" * 64,
        "environment_component_sha256": "8" * 64,
    }
    value.update(overrides)
    return value


def test_round_trip_preserves_order_native_arrays_and_provenance(tmp_path):
    manifest_path = _write(tmp_path)
    artifact = load_binary_artifact(manifest_path)
    manifest = artifact.manifest

    assert manifest_path == (
        tmp_path / "toy" / "clipseg-target" / manifest["artifact_id"] / "manifest.json"
    )
    assert manifest["artifact_type"] == ARTIFACT_TYPE
    assert manifest["schema_version"] == 2
    assert manifest["class_index"] == 1
    assert manifest["class_name"] == "lesion"
    assert manifest["num_samples"] == 2
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["sample_id_sha256"] == sample_id_sha256(["case-a", "case-b"])
    assert artifact.manifest_sha256 == sha256_file(manifest_path)

    samples = list(artifact.iter_samples())
    assert [sample.index for sample in samples] == [0, 1]
    assert [sample.sample_id for sample in samples] == ["case-a", "case-b"]
    for index, sample in enumerate(samples):
        expected_probability, expected_truth = _arrays(index)
        np.testing.assert_array_equal(
            sample.foreground_probability, expected_probability
        )
        np.testing.assert_array_equal(sample.truth, expected_truth)
        assert sample.foreground_probability.dtype == np.float32
        assert sample.truth.dtype == np.uint8
        assert not sample.foreground_probability.flags.writeable
        assert not sample.truth.flags.writeable


def test_schema_three_binds_scientific_input_roots_into_identity(tmp_path):
    first_path = _write(
        tmp_path / "first", scientific_input=_scientific_input()
    )
    first = load_binary_artifact(first_path, validate_payloads=False).manifest
    second_path = _write(
        tmp_path / "second",
        scientific_input=_scientific_input(condition_input_sha256="8" * 64),
    )
    second = load_binary_artifact(second_path, validate_payloads=False).manifest

    assert first["schema_version"] == 3
    assert first["scientific_input"] == _scientific_input()
    assert first["artifact_id"] != second["artifact_id"]


def test_schema_three_rejects_missing_extra_or_invalid_scientific_roots(tmp_path):
    with pytest.raises(ValueError, match="schema mismatch"):
        _write(
            tmp_path / "missing",
            scientific_input={
                key: value
                for key, value in _scientific_input().items()
                if key != "environment_component_sha256"
            },
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        _write(
            tmp_path / "extra",
            scientific_input={**_scientific_input(), "unreviewed": "9" * 64},
        )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        _write(
            tmp_path / "bad-hash",
            scientific_input=_scientific_input(root_lock_sha256="not-a-hash"),
        )

    manifest_path = _write(
        tmp_path / "tampered", scientific_input=_scientific_input()
    )
    manifest = _read_manifest(manifest_path)
    del manifest["scientific_input"]
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_binary_artifact(manifest_path, validate_payloads=False)


def test_writer_is_no_overwrite_and_leaves_no_staging_directory(tmp_path):
    first = _write(tmp_path)
    first_hash = sha256_file(first)
    with pytest.raises(FileExistsError, match="already exists"):
        _write(tmp_path)
    assert sha256_file(first) == first_hash
    parent = first.parent.parent
    assert not list(parent.glob(".*.tmp-*"))
    assert not list(parent.glob(".*.publish.lock"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dataset": "../escape"}, "safe path token"),
        ({"class_index": True}, "integer"),
        ({"sample_ids": ["case\nline"]}, "line breaks"),
        (
            {
                "checkpoint": {
                    "path": "../checkpoint.pt",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                }
            },
            "normalized relative path",
        ),
        ({"created_utc": "2026-07-19T12:00:00"}, "UTC offset"),
    ],
)
def test_writer_rejects_unsafe_or_ambiguous_metadata(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("probability", "truth", "match"),
    [
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.uint8),
            "dtype float32",
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=bool),
            "dtype uint8",
        ),
        (
            np.array([[0.0, 1.01]], dtype=np.float32),
            np.zeros((1, 2), dtype=np.uint8),
            r"lie in \[0, 1\]",
        ),
        (
            np.array([[0.0, np.nan]], dtype=np.float32),
            np.zeros((1, 2), dtype=np.uint8),
            "finite",
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.array([[0, 2], [0, 1]], dtype=np.uint8),
            "only 0 and 1",
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 3), dtype=np.uint8),
            "shapes must be equal",
        ),
    ],
)
def test_writer_strictly_rejects_bad_payloads(tmp_path, probability, truth, match):
    with pytest.raises(ValueError, match=match):
        _write(
            tmp_path,
            sample_ids=("only",),
            samples=[("only", probability, truth)],
        )
    condition_dir = tmp_path / "toy" / "clipseg-target"
    assert not list(condition_dir.glob(".*.tmp-*"))
    assert not [path for path in condition_dir.glob("*") if path.is_dir()]


def test_writer_rejects_short_extra_and_out_of_order_streams(tmp_path):
    with pytest.raises(ValueError, match="ended at index 1"):
        _write(
            tmp_path / "short",
            samples=[("case-a", *_arrays(0))],
        )
    with pytest.raises(ValueError, match="more than the declared"):
        _write(
            tmp_path / "extra",
            samples=[
                ("case-a", *_arrays(0)),
                ("case-b", *_arrays(1)),
                ("case-c", *_arrays(0)),
            ],
        )
    with pytest.raises(ValueError, match="sample order mismatch"):
        _write(
            tmp_path / "order",
            samples=[("case-b", *_arrays(1)), ("case-a", *_arrays(0))],
        )


def test_loader_rejects_payload_hash_tampering(tmp_path):
    manifest_path = _write(tmp_path)
    manifest = _read_manifest(manifest_path)
    payload = manifest_path.parent / manifest["samples"][0]["path"]
    with payload.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_binary_artifact(manifest_path)


def test_loader_rejects_path_traversal_before_access(tmp_path):
    manifest_path = _write(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["samples"][0]["path"] = "../outside.npz"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="path must equal"):
        load_binary_artifact(manifest_path, validate_payloads=False)


def test_loader_rejects_dtype_even_when_attacker_updates_payload_hash(tmp_path):
    manifest_path = _write(tmp_path)
    manifest = _read_manifest(manifest_path)
    payload = manifest_path.parent / manifest["samples"][0]["path"]
    probability, truth = _arrays(0, dtype=np.float64)[0], _arrays(0)[1]
    with payload.open("wb") as handle:
        np.savez_compressed(handle, foreground_probability=probability, truth=truth)
    manifest["samples"][0]["sha256"] = sha256_file(payload)
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="dtype float32"):
        load_binary_artifact(manifest_path)


def test_loader_rejects_reordered_sample_identity_and_extra_schema(tmp_path):
    reordered_path = _write(tmp_path / "reordered")
    manifest = _read_manifest(reordered_path)
    first = manifest["samples"][0]["sample_id"]
    second = manifest["samples"][1]["sample_id"]
    manifest["samples"][0]["sample_id"] = second
    manifest["samples"][1]["sample_id"] = first
    manifest["sample_id_sha256"] = sample_id_sha256([second, first])
    _write_manifest(reordered_path, manifest)
    with pytest.raises(ValueError, match="artifact_id does not match"):
        load_binary_artifact(reordered_path, validate_payloads=False)

    extra_path = _write(tmp_path / "extra-schema")
    extra_manifest = _read_manifest(extra_path)
    extra_manifest["unreviewed_field"] = 1
    _write_manifest(extra_path, extra_manifest)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_binary_artifact(extra_path, validate_payloads=False)


def test_loader_rejects_duplicate_json_keys_and_manifest_symlink(tmp_path):
    manifest_path = _write(tmp_path)
    content = manifest_path.read_text()
    manifest_path.write_text(content.replace("{", '{"schema_version": 1,', 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_binary_artifact(manifest_path, validate_payloads=False)

    real_path = _write(tmp_path / "symlink")
    link = tmp_path / "manifest.json"
    link.symlink_to(real_path)
    with pytest.raises(FileNotFoundError, match="non-symlink"):
        load_binary_artifact(link)


def test_sample_identifier_hash_is_order_sensitive():
    forward = sample_id_sha256(["a", "b"])
    reverse = sample_id_sha256(["b", "a"])
    assert forward != reverse
    assert forward == hashlib.sha256(b"a\nb").hexdigest()
