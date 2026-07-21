"""Read-only closure tests for the seed diagnostic collector."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import collect_binary_seed_diagnostics as collector
from scripts.submit_binary_simulations import PlannedJob
from selectseg.binary_seed_extension import iter_experiments


@pytest.fixture
def complete_collection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    diagnostic_root = Path("outputs/diagnostic results")
    checkpoint_lock = Path("outputs/binary_seed_campaign/checkpoints.lock.json")
    receipt = checkpoint_lock.parent / "diagnose-submissions.jsonl"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("fixed receipt fixture\n", encoding="utf-8")

    datasets = [{"name": f"dataset-{index}"} for index in range(5)]
    models = [
        {"name": "clipseg", "condition": "clipseg-target"},
        {"name": "deeplabv3", "condition": "deeplabv3-target"},
    ]
    spec = {
        "datasets": datasets,
        "models": models,
        "protocol": {"training_seeds": [1, 2]},
        "gpu_profiles": [{"partition": "a"}, {"partition": "b"}],
        "paths": {
            "checkpoint_lock": checkpoint_lock.as_posix(),
            "diagnostic_root": diagnostic_root.as_posix(),
        },
    }
    binding = {"spec": spec}
    rows_by_seed = {1: [], 2: []}
    jobs = []
    loaded_by_path = {}
    expected_paths = []
    for index, experiment in enumerate(iter_experiments(spec)):
        dataset = experiment["dataset"]["name"]
        model = experiment["model"]["name"]
        condition = experiment["model"]["condition"]
        seed = experiment["training_seed"]
        artifact_id = f"{index + 1:016x}"
        manifest = Path("artifacts") / artifact_id / "manifest.json"
        manifest_sha = f"{index + 1:064x}"
        sample_sha = f"{index + 101:064x}"
        row = {
            "dataset": dataset,
            "model": model,
            "condition": condition,
            "training_seed": seed,
            "artifact_manifest_path": manifest.as_posix(),
            "artifact_manifest_sha256": manifest_sha,
            "artifact_id": artifact_id,
            "sample_id_sha256": sample_sha,
            "num_samples": index + 2,
        }
        rows_by_seed[seed].append(row)
        summary_path = (
            diagnostic_root
            / dataset
            / condition
            / artifact_id
            / f"diagnostic-{index:02d}"
            / "diagnostics.json"
        )
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text("{}\n", encoding="utf-8")
        loaded_by_path[summary_path.resolve()] = SimpleNamespace(
            summary_path=summary_path.resolve(),
            summary={
                "artifact": {
                    "artifact_id": artifact_id,
                    "manifest_path": manifest.as_posix(),
                    "manifest_sha256": manifest_sha,
                    "sample_id_sha256": sample_sha,
                    "num_samples": index + 2,
                    "dataset": dataset,
                    "condition": condition,
                    "model": model,
                },
                "descriptors": {"included": True, "num_rows": index + 2},
            },
        )
        expected_paths.append(summary_path.resolve())
        partition = ("agsmall", "amdsmall", "msismall")[index % 3]
        jobs.append(
            PlannedJob(
                phase="seed_diagnose",
                key=(seed, dataset, condition, partition),
                command=(
                    "sbatch",
                    "--parsable",
                    "--artifact-manifest",
                    manifest.as_posix(),
                    "--expected-artifact-manifest-sha256",
                    manifest_sha,
                    "--output-root",
                    diagnostic_root.as_posix(),
                    "--write-descriptors",
                ),
            )
        )

    downstream_binding = {
        "binding": binding,
        "lock": {
            "campaigns": [
                {
                    "training_seed": seed,
                    "freeze_records": list(reversed(rows_by_seed[seed])),
                }
                for seed in (2, 1)
            ]
        },
    }
    calls = {"receipt": 0, "loaded": []}

    def fake_load_downstream(path, *, expected_sha256):
        assert Path(path) == Path("downstream.lock.json")
        assert expected_sha256 == "a" * 64
        return downstream_binding

    def fake_validate_receipt(path, planned, *, phase):
        calls["receipt"] += 1
        assert Path(path).resolve() == receipt.resolve()
        assert tuple(planned) == tuple(reversed(jobs))
        assert phase == "diagnose"
        return {"count": 20, "job_ids": tuple(str(1000 + i) for i in range(20))}

    def fake_load_diagnostic(path, *, validate_descriptors):
        calls["loaded"].append((Path(path).resolve(), validate_descriptors))
        return loaded_by_path[Path(path).resolve()]

    monkeypatch.setattr(collector, "load_downstream_lock", fake_load_downstream)
    monkeypatch.setattr(
        collector,
        "plan_downstream_jobs",
        lambda downstream, phase: tuple(reversed(jobs)),
    )
    monkeypatch.setattr(collector, "_validate_receipt", fake_validate_receipt)
    monkeypatch.setattr(collector, "load_binary_diagnostics", fake_load_diagnostic)
    return SimpleNamespace(
        receipt=receipt,
        downstream_binding=downstream_binding,
        loaded_by_path=loaded_by_path,
        expected_paths=tuple(expected_paths),
        calls=calls,
    )


def _collect(fixture, **overrides):
    arguments = {
        "downstream_lock": "downstream.lock.json",
        "expected_downstream_lock_sha256": "a" * 64,
        "diagnose_receipt": fixture.receipt,
    }
    arguments.update(overrides)
    return collector.collect_diagnostic_summaries(**arguments)


def test_collector_returns_strict_locked_order_and_direct_export_argv(
    complete_collection,
):
    paths = _collect(complete_collection)
    assert paths == complete_collection.expected_paths
    assert complete_collection.calls["receipt"] == 1
    assert len(complete_collection.calls["loaded"]) == 20
    assert all(validate is True for _, validate in complete_collection.calls["loaded"])

    argv = collector.formatted_output(paths, "argv0").split(b"\0")
    assert argv[-1] == b""
    assert len(argv[:-1]) == 40
    assert argv[:-1:2] == [b"--diagnostic-summary"] * 20
    assert b"outputs/diagnostic results" in argv[1]

    lines = collector.formatted_output(paths, "arguments").decode().splitlines()
    assert len(lines) == 20
    assert all(line.startswith("--diagnostic-summary ") for line in lines)
    assert "'outputs/diagnostic results/" in lines[0]


def test_collector_rejects_nonfixed_receipt(complete_collection, tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text("not the fixed receipt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixed planner path"):
        _collect(complete_collection, diagnose_receipt=other)
    assert complete_collection.calls["receipt"] == 0


def test_collector_rejects_ambiguous_or_misbound_diagnostic(complete_collection):
    first = complete_collection.expected_paths[0]
    second = first.parent.parent / "another-diagnostic" / "diagnostics.json"
    second.parent.mkdir()
    second.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one completed diagnostic"):
        _collect(complete_collection)

    second.unlink()
    second.parent.rmdir()
    loaded = complete_collection.loaded_by_path[first]
    loaded.summary["artifact"]["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="artifact binding|locked seed-1 cell"):
        _collect(complete_collection)


def test_collector_rejects_duplicate_receipt_job_ids(complete_collection, monkeypatch):
    monkeypatch.setattr(
        collector,
        "_validate_receipt",
        lambda path, jobs, *, phase: {"count": 20, "job_ids": ("123",) * 20},
    )
    with pytest.raises(ValueError, match="20 unique submitted job IDs"):
        _collect(complete_collection)


def test_collector_rejects_missing_descriptor_payload(complete_collection):
    loaded = complete_collection.loaded_by_path[complete_collection.expected_paths[0]]
    loaded.summary = deepcopy(loaded.summary)
    loaded.summary["descriptors"]["included"] = False
    with pytest.raises(ValueError, match="include its descriptor"):
        _collect(complete_collection)
