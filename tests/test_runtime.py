"""Tests for the isolated, locked confidence-runtime workflow."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze.main import EXPECTED_CONDITIONS
from scripts.analyze.runtime import (
    ANALYSIS_ARTIFACT_TYPE,
    ANALYSIS_SCHEMA_VERSION,
    build_report,
    load_runtime_condition,
    write_report,
)
from scripts.render.runtime import (
    OUTPUT_NAME,
    load_analysis,
    render_analysis,
    validate_analysis,
    write_output,
)
from scripts.submit.runtime import CPU_PARTITIONS, plan_runtime_jobs
from scripts.submit.runtime import plan_runtime_ladder_jobs
from scripts.submit.main import write_campaign_lock
from selectseg.studies.runtime import (
    LADDER_LOCK_SOURCE_PATHS,
    METHODS,
    METHODS_V2,
    RECORD_FIELDS,
    RUNTIME_ARTIFACT_TYPE,
    _measurement_order,
    _runtime_id,
    _selected_entries,
    load_runtime_ladder_lock,
    parse_args,
    runtime_source_fingerprint,
    run_benchmark,
)
from selectseg.artifacts import load_binary_artifact, write_binary_artifact
from selectseg.quadrature import sha256_file


ROOT = Path(__file__).resolve().parents[1]
MIDPOINT_SPEC = ROOT / "configs" / "estimators" / "midpoint-v1.json"
RUNTIME_SPEC = ROOT / "configs" / "auxiliary" / "binary_runtime-v1.json"
RUNTIME_LADDER_SPEC = ROOT / "configs" / "auxiliary" / "binary_runtime_ladder-v2.json"
RUNTIME_LADDER_LOCK = (
    ROOT / "configs" / "auxiliary" / "binary_runtime_ladder-v2.lock.json"
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _write_artifact(root, dataset, condition, *, num_samples):
    samples = []
    sample_ids = []
    for index in range(num_samples):
        height = 6 + index % 3
        width = 7 + index % 5
        probability = np.linspace(0.01, 0.99, height * width, dtype=np.float32).reshape(
            height, width
        )
        probability = np.roll(probability, index, axis=1)
        truth = (probability >= 0.55).astype(np.uint8)
        sample_id = f"{dataset}-{condition}-{index}"
        sample_ids.append(sample_id)
        samples.append((sample_id, probability, truth))
    model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
    return write_binary_artifact(
        root,
        dataset=dataset,
        condition=condition,
        model=model,
        split="test",
        class_index=1,
        class_name="foreground",
        checkpoint=None,
        base_model={"name": model, "source": "unit-test"},
        source_sha256="a" * 64,
        environment={
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
        preprocessing={
            "model_input": "none",
            "probability_to_native_mask": "none",
        },
        cohort="deterministic runtime unit-test panel",
        sample_ids=sample_ids,
        samples=samples,
        command=["pytest", "freeze"],
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _locked_campaign(tmp_path, *, num_samples=1):
    artifacts = []
    for dataset, condition in EXPECTED_CONDITIONS:
        artifacts.append(
            _write_artifact(
                tmp_path / "artifacts",
                dataset,
                condition,
                num_samples=num_samples,
            )
        )
    locked_artifacts = []
    for artifact in artifacts:
        manifest = json.loads(artifact.read_text())
        locked_artifacts.append(
            {
                "manifest_path": str(artifact),
                "manifest_sha256": sha256_file(artifact),
                "artifact_id": manifest["artifact_id"],
                "dataset": manifest["dataset"],
                "condition": manifest["condition"],
                "model": manifest["model"],
                "split": manifest["split"],
                "checkpoint_sha256": None,
                "source_sha256": manifest["source_sha256"],
                "sample_id_sha256": manifest["sample_id_sha256"],
                "num_samples": manifest["num_samples"],
            }
        )
    lock = {
        "lock_schema_version": 1,
        "campaign_id": "unit-runtime-campaign",
        "config": {"path": "unit-runtime-campaign.json", "sha256": "d" * 64},
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "estimator": {
            "spec_path": str(MIDPOINT_SPEC),
            "spec_sha256": sha256_file(MIDPOINT_SPEC),
            "estimator_id": "midpoint-v1",
            "target_measure": "uniform-threshold",
        },
        "paths": {
            "artifact_output_root": str(tmp_path / "unused-artifacts"),
            "common_output_root": str(tmp_path / "unused-common"),
            "simulation_output_root": str(tmp_path / "unused-simulations"),
            "assembly_output_root": str(tmp_path / "unused-assembled"),
        },
        "artifacts": locked_artifacts,
    }
    lock_path, lock_sha256 = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    return artifacts, lock_path, lock_sha256, lock


def _benchmark_arguments(
    artifact,
    lock_path,
    lock_sha256,
    *,
    benchmark_spec=RUNTIME_SPEC,
    benchmark_lock=None,
    benchmark_lock_sha256=None,
):
    arguments = [
        "--campaign-id",
        "unit-runtime-campaign",
        "--campaign-lock",
        str(lock_path),
        "--expected-campaign-lock-sha256",
        lock_sha256,
        "--artifact-manifest",
        str(artifact),
        "--expected-artifact-manifest-sha256",
        sha256_file(artifact),
        "--estimator-spec",
        str(MIDPOINT_SPEC),
        "--expected-estimator-spec-sha256",
        sha256_file(MIDPOINT_SPEC),
        "--benchmark-spec",
        str(benchmark_spec),
        "--expected-benchmark-spec-sha256",
        sha256_file(benchmark_spec),
    ]
    if benchmark_lock is not None:
        arguments.extend(
            [
                "--benchmark-lock",
                str(benchmark_lock),
                "--expected-benchmark-lock-sha256",
                benchmark_lock_sha256,
            ]
        )
    return arguments


def _write_runtime_ladder_lock(tmp_path, campaign_lock, campaign_sha256):
    lock = {
        "lock_schema_version": 1,
        "benchmark_id": "binary-confidence-runtime-ladder-v2",
        "spec": {
            "path": str(RUNTIME_LADDER_SPEC),
            "sha256": sha256_file(RUNTIME_LADDER_SPEC),
        },
        "canonical_campaign_lock": {
            "path": str(campaign_lock),
            "sha256": campaign_sha256,
            "campaign_id": "unit-runtime-campaign",
        },
        "estimator_spec": {
            "path": str(MIDPOINT_SPEC),
            "sha256": sha256_file(MIDPOINT_SPEC),
        },
        "source_files": [
            {
                "path": relative,
                "sha256": sha256_file(ROOT / relative),
            }
            for relative in LADDER_LOCK_SOURCE_PATHS
        ],
    }
    path = tmp_path / "runtime-ladder.lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n")
    return path, sha256_file(path)


def test_runtime_scorer_excludes_io_uses_fixed_grid_and_is_append_only(
    tmp_path, monkeypatch
):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=1)
    monkeypatch.chdir(tmp_path)
    for variable in THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    arguments = _benchmark_arguments(artifacts[0], lock_path, lock_sha256)
    records_path, manifest_path = run_benchmark(parse_args(arguments))
    manifest = json.loads(manifest_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]

    assert manifest["artifact_type"] == RUNTIME_ARTIFACT_TYPE
    assert manifest["num_selected_images"] == 1
    assert manifest["protocol"]["m"] == 32
    assert manifest["execution"]["benchmark_workers"] == 8
    assert "model inference" in manifest["benchmark_scope"]["excluded"]
    assert (
        "artifact manifest and payload I/O" in manifest["benchmark_scope"]["excluded"]
    )
    assert manifest["records_sha256"] == sha256_file(records_path)
    assert len(records) == 8
    assert {(row["method"], row["repetition"]) for row in records} == {
        (method, repetition) for method in METHODS for repetition in range(4)
    }
    assert all(set(row) == RECORD_FIELDS for row in records)
    assert all(row["wall_seconds"] > 0 for row in records)
    for method in METHODS:
        assert (
            len({row["result_sha256"] for row in records if row["method"] == method})
            == 1
        )
    with pytest.raises(FileExistsError, match="already exists"):
        run_benchmark(parse_args(arguments))


def test_runtime_scorer_requires_fixed_native_thread_environment(tmp_path, monkeypatch):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=1)
    monkeypatch.chdir(tmp_path)
    for variable in THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="thread environment is not fixed"):
        run_benchmark(
            parse_args(_benchmark_arguments(artifacts[0], lock_path, lock_sha256))
        )


def test_runtime_planner_is_exactly_one_cpu_job_per_locked_artifact(tmp_path):
    _, lock_path, _, _ = _locked_campaign(tmp_path, num_samples=1)
    jobs = plan_runtime_jobs(lock_path, RUNTIME_SPEC)
    assert len(jobs) == 16
    assert len({job.key for job in jobs}) == 16
    assert [job.command[job.command.index("--partition") + 1] for job in jobs] == [
        ",".join(
            CPU_PARTITIONS[index % len(CPU_PARTITIONS) :]
            + CPU_PARTITIONS[: index % len(CPU_PARTITIONS)]
        )
        for index in range(16)
    ]
    for job in jobs:
        command = job.command
        assert command[0] == "sbatch"
        assert "scripts/slurm/run.sbatch" in command
        assert "--gres" not in command
        assert "--array" not in command
        assert "--artifact-manifest" in command
        assert "--expected-artifact-manifest-sha256" in command
        assert "--expected-benchmark-spec-sha256" in command


def test_runtime_ladder_scores_four_methods_with_balanced_positions(
    tmp_path, monkeypatch
):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=1)
    runtime_lock, runtime_lock_sha = _write_runtime_ladder_lock(
        tmp_path, lock_path, lock_sha256
    )
    monkeypatch.chdir(tmp_path)
    for variable in THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    arguments = _benchmark_arguments(
        artifacts[0],
        lock_path,
        lock_sha256,
        benchmark_spec=RUNTIME_LADDER_SPEC,
        benchmark_lock=runtime_lock,
        benchmark_lock_sha256=runtime_lock_sha,
    )
    records_path, manifest_path = run_benchmark(parse_args(arguments))
    manifest = json.loads(manifest_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]

    assert manifest["protocol"]["m_values"] == [2, 8, 32]
    assert manifest["protocol"]["methods"] == list(METHODS_V2)
    assert manifest["protocol"]["measured_order"] == "williams_latin_v1"
    assert manifest["provenance"]["benchmark_lock_sha256"] == runtime_lock_sha
    assert len(records) == 16
    assert {(row["method"], row["repetition"]) for row in records} == {
        (method, repetition) for method in METHODS_V2 for repetition in range(4)
    }
    for method in METHODS_V2:
        assert {
            row["order_position"] for row in records if row["method"] == method
        } == {0, 1, 2, 3}
    ordered_pairs = set()
    for repetition in range(4):
        order = tuple(
            row["method"]
            for row in sorted(
                (row for row in records if row["repetition"] == repetition),
                key=lambda row: row["order_position"],
            )
        )
        ordered_pairs.update(zip(order, order[1:]))
    assert len(ordered_pairs) == 12


def test_runtime_ladder_lock_hash_changes_runtime_identity():
    identity = {
        "campaign_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "estimator_sha256": "3" * 64,
        "spec_sha256": "4" * 64,
        "source_sha256": "5" * 64,
    }
    assert _runtime_id(**identity, benchmark_lock_sha256="a" * 64) != _runtime_id(
        **identity, benchmark_lock_sha256="b" * 64
    )


def test_runtime_ladder_planner_is_locked_and_one_job_per_condition(tmp_path):
    _, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=1)
    runtime_lock, runtime_lock_sha = _write_runtime_ladder_lock(
        tmp_path, lock_path, lock_sha256
    )
    jobs = plan_runtime_ladder_jobs(
        runtime_lock,
        expected_runtime_lock_sha256=runtime_lock_sha,
    )
    assert len(jobs) == 16
    assert len({job.key for job in jobs}) == 16
    assert [job.command[job.command.index("--partition") + 1] for job in jobs] == [
        ",".join(
            CPU_PARTITIONS[index % len(CPU_PARTITIONS) :]
            + CPU_PARTITIONS[: index % len(CPU_PARTITIONS)]
        )
        for index in range(16)
    ]
    for job in jobs:
        assert job.phase == "binary_runtime_ladder_v2"
        assert "--array" not in job.command
        assert "--benchmark-lock" in job.command
        assert "--expected-benchmark-lock-sha256" in job.command
        assert "scripts/slurm/run.sbatch" in job.command

    lock = json.loads(runtime_lock.read_text())
    lock["source_files"][0]["sha256"] = "0" * 64
    runtime_lock.write_text(json.dumps(lock) + "\n")
    with pytest.raises(ValueError, match="locked source file"):
        load_runtime_ladder_lock(
            runtime_lock, expected_sha256=sha256_file(runtime_lock)
        )


def _write_synthetic_runtime_inputs(
    tmp_path,
    artifacts,
    lock_path,
    lock_sha256,
    *,
    runtime_spec=RUNTIME_SPEC,
    runtime_lock=None,
    runtime_lock_sha256=None,
):
    spec = json.loads(runtime_spec.read_text())
    methods = tuple(spec["protocol"]["methods"])
    inputs = []
    for condition_index, ((dataset, condition), artifact_path) in enumerate(
        zip(EXPECTED_CONDITIONS, artifacts, strict=True)
    ):
        frozen = load_binary_artifact(artifact_path, validate_payloads=False)
        selected = _selected_entries(frozen.manifest, 16)
        run_id = _runtime_id(
            campaign_sha256=lock_sha256,
            artifact_sha256=sha256_file(artifact_path),
            estimator_sha256=sha256_file(MIDPOINT_SPEC),
            spec_sha256=sha256_file(runtime_spec),
            source_sha256=runtime_source_fingerprint(),
            benchmark_lock_sha256=runtime_lock_sha256,
        )
        directory = tmp_path / "runtime" / dataset / condition / run_id
        directory.mkdir(parents=True)
        records = []
        method_digests = {
            method: hashlib.sha256(
                f"{dataset}/{condition}/{method}".encode()
            ).hexdigest()
            for method in methods
        }
        for repetition in range(4):
            order = _measurement_order(
                methods, repetition, spec["protocol"]["measured_order"]
            )
            for order_position, method in enumerate(order):
                base = {
                    "m2_joint": 0.006,
                    "m8_joint": 0.010,
                    "m32_joint": 0.020,
                    "dice_exact": 0.004,
                }[method]
                wall = base * (1 + 0.01 * condition_index + 0.02 * repetition)
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "dataset": dataset,
                        "condition": condition,
                        "method": method,
                        "repetition": repetition,
                        "order_position": order_position,
                        "num_images": 16,
                        "total_pixels": sum(row["pixels"] for row in selected),
                        "wall_seconds": wall,
                        "seconds_per_image": wall / 16,
                        "images_per_second": 16 / wall,
                        "result_sha256": method_digests[method],
                    }
                )
        records_path = directory / "records.jsonl"
        records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        model = "clipseg" if condition.startswith("clipseg") else "deeplabv3"
        provenance = {
            "campaign_id": "unit-runtime-campaign",
            "campaign_lock_path": str(lock_path),
            "campaign_lock_sha256": lock_sha256,
            "artifact_manifest_path": str(artifact_path),
            "artifact_manifest_sha256": sha256_file(artifact_path),
            "artifact_id": frozen.manifest["artifact_id"],
            "artifact_source_sha256": frozen.manifest["source_sha256"],
            "estimator_spec_path": str(MIDPOINT_SPEC),
            "estimator_spec_sha256": sha256_file(MIDPOINT_SPEC),
            "estimator_id": "midpoint-v1",
            "benchmark_spec_path": str(runtime_spec),
            "benchmark_spec_sha256": sha256_file(runtime_spec),
            "runtime_source_sha256": runtime_source_fingerprint(),
        }
        if runtime_lock is not None:
            provenance.update(
                {
                    "benchmark_lock_path": str(runtime_lock),
                    "benchmark_lock_sha256": runtime_lock_sha256,
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": RUNTIME_ARTIFACT_TYPE,
            "run_id": run_id,
            "runtime_id": run_id,
            "created_utc": "2026-07-19T00:00:00+00:00",
            "dataset": dataset,
            "condition": condition,
            "model": model,
            "split": "test",
            "benchmark_scope": {
                "timed": "confidence computation, task dispatch, and result collection",
                "excluded": ["model inference", "artifact manifest and payload I/O"],
                "interpretation": (
                    "descriptive hardware-dependent wall-clock throughput; not an "
                    "algorithmic-complexity result"
                ),
            },
            "protocol": spec["protocol"],
            "execution": spec["execution"],
            "selected_samples": selected,
            "num_selected_images": 16,
            "total_selected_pixels": sum(row["pixels"] for row in selected),
            "measurements_per_method": 4,
            "record_fields": sorted(RECORD_FIELDS),
            "environment": {
                "packages": {"python": "test", "numpy": "test", "scipy": "test"},
                "platform": "unit-test",
                "machine": "unit-test",
                "cpu_model": "Unit Test CPU",
                "logical_cpu_count": 8,
                "affinity_cpu_count": 8,
                "thread_environment": spec["execution"]["thread_environment"],
                "slurm": {
                    "job_id": "1",
                    "partition": CPU_PARTITIONS[condition_index % 3],
                    "node_list": "unit-node",
                    "cpus_per_task": "8",
                },
            },
            "memory": {
                "peak_rss_bytes": 512 * 2**20 + condition_index,
                "scope": "process high-water RSS",
                "method_attribution_available": False,
            },
            "provenance": provenance,
            "command": [],
            "num_records": len(methods) * 4,
            "records_sha256": sha256_file(records_path),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
        inputs.append(records_path)
    return inputs


def test_strict_runtime_analysis_and_compact_renderer(tmp_path):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=16)
    inputs = _write_synthetic_runtime_inputs(
        tmp_path, artifacts, lock_path, lock_sha256
    )
    report = build_report(lock_path, list(reversed(inputs)))
    assert report["artifact_type"] == ANALYSIS_ARTIFACT_TYPE
    assert report["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert len(report["conditions"]) == 16
    assert report["condition_sets"]["num_target_conditions"] == 10
    assert "hardware-dependent" in report["scope"]["timing_status"]
    assert "algorithmic complexity" in report["scope"]["complexity_status"]

    output = write_report(report, tmp_path / "analysis.json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(report, output)
    loaded, source_hash = load_analysis(output)
    assert loaded == report
    by_key = validate_analysis(loaded)
    assert len(by_key) == 16
    tex = render_analysis(loaded, source_hash=source_hash)
    assert tex.count(r"\begin{table*}[t]") == 1
    assert r"Process peak RSS, GiB $\downarrow$" not in tex
    assert "Process peak RSS, GiB" in tex
    assert r"\label{tab:binary-runtime}" in tex
    assert "M32 joint, ms/image" in tex
    assert "Dice-Exact, ms/image" in tex
    assert r"$\times$" in tex
    assert "\t" not in tex
    assert "CLIP-T / DL-T" in tex
    assert "hardware dependent" in tex
    assert source_hash in tex
    rendered = write_output(tex, tmp_path / "rendered")
    assert rendered.name == OUTPUT_NAME
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_output(tex, tmp_path / "rendered")

    corrupt = copy.deepcopy(report)
    corrupt["conditions"][0]["methods"]["m32_joint"]["milliseconds_per_image"] = 999
    with pytest.raises(ValueError, match="inconsistent"):
        validate_analysis(corrupt)


def test_runtime_ladder_strict_analysis_and_renderer(tmp_path):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=16)
    runtime_lock, runtime_lock_sha = _write_runtime_ladder_lock(
        tmp_path, lock_path, lock_sha256
    )
    inputs = _write_synthetic_runtime_inputs(
        tmp_path,
        artifacts,
        lock_path,
        lock_sha256,
        runtime_spec=RUNTIME_LADDER_SPEC,
        runtime_lock=runtime_lock,
        runtime_lock_sha256=runtime_lock_sha,
    )
    report = build_report(
        lock_path,
        list(reversed(inputs)),
        benchmark_lock=runtime_lock,
        expected_benchmark_lock_sha256=runtime_lock_sha,
    )
    canonical_report = build_report(
        lock_path,
        inputs,
        benchmark_lock=runtime_lock,
        expected_benchmark_lock_sha256=runtime_lock_sha,
    )
    assert canonical_report["analysis_id"] == report["analysis_id"]
    assert canonical_report["provenance"]["inputs"] == report["provenance"]["inputs"]
    assert report["provenance"]["runtime_protocol_version"] == 2
    assert report["provenance"]["benchmark_lock_sha256"] == runtime_lock_sha
    assert set(report["conditions"][0]["methods"]) == set(METHODS_V2)
    assert set(report["target_ranges"]) == {
        *(f"{method}_milliseconds_per_image" for method in METHODS_V2),
        "m32_joint_over_dice_exact_time_ratio",
    }
    tex = render_analysis(report, source_hash="d" * 64)
    for label in ("M2 joint", "M8 joint", "M32 joint", "Dice-Exact"):
        assert f"{label}, ms/image" in tex
        assert f"{label}, images/s" in tex
    assert "Williams-balanced-order" in tex
    assert "no confidence or boundary-distance result" in tex

    with pytest.raises(ValueError, match="requires its immutable v2 lock"):
        build_report(lock_path, inputs)

    for records_path in inputs:
        manifest_path = records_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["runtime_source_sha256"] = "c" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(
        ValueError,
        match="non-content-addressed runtime identity|different locked runtime source",
    ):
        build_report(
            lock_path,
            inputs,
            benchmark_lock=runtime_lock,
            expected_benchmark_lock_sha256=runtime_lock_sha,
        )


def test_runtime_loader_rejects_tampering(tmp_path):
    artifacts, lock_path, lock_sha256, _ = _locked_campaign(tmp_path, num_samples=16)
    inputs = _write_synthetic_runtime_inputs(
        tmp_path, artifacts, lock_path, lock_sha256
    )
    loaded = load_runtime_condition(inputs[0])
    assert len(loaded.records) == 8
    manifest_path = inputs[0].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_runtime_condition(inputs[0])
