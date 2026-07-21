"""Integrity and numerical tests for locked probability ensembling."""

import json
from argparse import Namespace

import numpy as np
import pytest

from selectseg.artifacts import (
    load_binary_artifact,
    sha256_file,
    write_binary_artifact,
)
from selectseg.ensemble import (
    SOURCE_LOCK_ARTIFACT_TYPE,
    build_ensemble_artifact,
)


def _write_source(root, probability, truth, *, seed):
    return write_binary_artifact(
        root,
        dataset="toy",
        condition="clipseg-target",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="lesion",
        checkpoint={
            "path": f"outputs/seed-{seed}/checkpoint.pt",
            "sha256": str(seed + 1) * 64,
            "size_bytes": 1,
        },
        base_model={"name": "clipseg", "source": "vendor/model"},
        source_sha256="a" * 64,
        environment={
            "packages": {
                "python": "3.12",
                "numpy": "test",
                "torch": None,
                "torchvision": None,
                "transformers": None,
            },
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        preprocessing={
            "model_input": "square resize with antialiasing",
            "probability_to_native_mask": "bilinear interpolation",
        },
        cohort="one held-out image",
        sample_ids=["case-a"],
        samples=[("case-a", probability, truth)],
        command=["test", f"seed-{seed}"],
        created_utc=f"2026-07-2{seed}T00:00:00+00:00",
    )


def _source_record(path, seed):
    artifact = load_binary_artifact(path, validate_payloads=False)
    manifest = artifact.manifest
    return {
        "training_seed": seed,
        "manifest_path": str(path),
        "manifest_sha256": artifact.manifest_sha256,
        "artifact_id": manifest["artifact_id"],
        "dataset": manifest["dataset"],
        "condition": manifest["condition"],
        "model": manifest["model"],
        "split": manifest["split"],
        "num_samples": manifest["num_samples"],
        "sample_id_sha256": manifest["sample_id_sha256"],
    }


def _write_lock(tmp_path, probabilities, truths):
    sources = []
    for seed, (probability, truth) in enumerate(
        zip(probabilities, truths, strict=True)
    ):
        path = _write_source(tmp_path / f"seed-{seed}", probability, truth, seed=seed)
        sources.append(_source_record(path, seed))
    lock = {
        "schema_version": 1,
        "artifact_type": SOURCE_LOCK_ARTIFACT_TYPE,
        "cells": [
            {
                "dataset": "toy",
                "condition": "clipseg-target",
                "model": "clipseg",
                "expected_num_samples": 1,
                "sources": sources,
            }
        ],
    }
    path = tmp_path / "source.lock.json"
    path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
    return path


def _args(lock_path, output_root):
    return Namespace(
        source_lock=str(lock_path),
        expected_source_lock_sha256=sha256_file(lock_path),
        dataset="toy",
        condition="clipseg-target",
        output_root=str(output_root),
    )


def test_ensemble_uses_float64_mean_and_publishes_standard_artifact(tmp_path):
    probabilities = [
        np.array([[0.1, 0.4]], dtype=np.float32),
        np.array([[0.2, 0.5]], dtype=np.float32),
        np.array([[0.9, 0.6]], dtype=np.float32),
    ]
    truth = np.array([[0, 1]], dtype=np.uint8)
    lock_path = _write_lock(tmp_path, probabilities, [truth, truth, truth])

    manifest_path = build_ensemble_artifact(
        _args(lock_path, tmp_path / "ensemble"), argv=["test"]
    )
    artifact = load_binary_artifact(manifest_path)
    sample = next(artifact.iter_samples())
    expected = np.asarray(
        sum(probability.astype(np.float64) for probability in probabilities) / 3,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(sample.foreground_probability, expected)
    np.testing.assert_array_equal(sample.truth, truth)
    assert artifact.manifest["condition"] == "clipseg-target"
    assert artifact.manifest["checkpoint"]["sha256"] == sha256_file(lock_path)


def test_ensemble_rejects_truth_mismatch_before_publication(tmp_path):
    probability = np.array([[0.2, 0.8]], dtype=np.float32)
    truth = np.array([[0, 1]], dtype=np.uint8)
    mismatch = np.array([[1, 1]], dtype=np.uint8)
    lock_path = _write_lock(
        tmp_path,
        [probability, probability, probability],
        [truth, mismatch, truth],
    )
    output_root = tmp_path / "ensemble"

    with pytest.raises(ValueError, match="truth mismatch"):
        build_ensemble_artifact(_args(lock_path, output_root), argv=["test"])
    assert not list(output_root.rglob("manifest.json"))
