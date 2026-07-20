"""Strict synthetic tests for the seed-extension public provenance exporter."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_binary_seed_extension as seed_analyzer
from scripts import export_binary_seed_provenance as exporter
from scripts.submit_binary_simulations import PlannedJob


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_receipt(path, jobs, *, changed_command=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    phase_name = Path(path).stem
    phase_offset = list(exporter.EXPECTED_PHASE_COUNTS).index(phase_name) * 1000
    for index, job in enumerate(jobs):
        command = list(job.command)
        if changed_command and index == 0:
            command.append("--changed")
        core = {
            "receipt_schema_version": 1,
            "created_utc": "2026-07-20T12:00:00+00:00",
            "phase": job.phase,
            "key": list(job.key),
            "command": command,
        }
        rows.append({**core, "status": "submitting", "job_id": None})
        rows.append(
            {
                **core,
                "status": "submitted",
                "job_id": str(900000 + phase_offset + index),
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _jobs(phase, count):
    internal = f"seed_{phase}"
    jobs = []
    for index in range(count):
        dataset = f"d{index // 4}"
        condition = f"c{index % 2}"
        training_seed = 1 + (index % 2)
        partition = f"p{index % 3}"
        if phase in {"train", "freeze"}:
            key = (f"d{index}", f"m{index}", training_seed, partition)
        elif phase == "common":
            key = (training_seed, dataset, f"{condition}-{index}", partition, 0.5)
        elif phase == "score":
            key = (
                training_seed,
                dataset,
                f"{condition}-{index}",
                partition,
                0.5,
                (2, 8, 32)[index % 3],
                0,
            )
        elif phase in {"assemble", "diagnose"}:
            key = (training_seed, dataset, f"{condition}-{index}", partition)
        elif phase == "analyze":
            key = ("all-seeds", "agsmall")
        elif phase == "render":
            key = ("seed-robustness-table", "agsmall")
        else:  # pragma: no cover - fixture contract
            raise AssertionError(phase)
        jobs.append(
            PlannedJob(
                phase=internal,
                key=key,
                command=("sbatch", "--parsable", f"wrapper-{phase}", str(index)),
            )
        )
    return tuple(jobs)


def _scheduler_event(spec_sha, receipt_sha, record_set_sha, private_sha):
    stats = {
        "minimum_seconds": 100,
        "maximum_seconds": 3000,
        "median_seconds": 800.0,
        "total_seconds": 20000,
    }
    limits = {
        "minimum_seconds": 7200,
        "maximum_seconds": 86400,
        "median_seconds": 46800.0,
        "total_seconds": 936000,
    }
    event = {
        "state_counts": [{"state": "COMPLETED", "count": 20}],
        "duration_seconds": stats,
        "timelimit_seconds": limits,
        "spec_lock_sha256": spec_sha,
        "train_submission_receipt_sha256": receipt_sha,
        "training_record_set_sha256": record_set_sha,
        "terminal_event_id": "e" * 64,
    }
    summary = {
        "summary_schema_version": 1,
        "artifact_type": "selectseg.public_seed_scheduler_summary",
        "auxiliary_id": exporter.EXPECTED_AUXILIARY_ID,
        "status": "complete",
        "expected_jobs": 20,
        "terminal_jobs": 20,
        "successful_jobs": 20,
        "failed_jobs": 0,
        "state_counts": event["state_counts"],
        "duration_seconds": stats,
        "timelimit_seconds": limits,
        "bindings": {
            "private_ledger_sha256": private_sha,
            "spec_lock_sha256": spec_sha,
            "train_submission_receipt_sha256": receipt_sha,
            "training_record_set_sha256": record_set_sha,
            "terminal_event_id": event["terminal_event_id"],
        },
    }
    return event, summary


@pytest.fixture
def complete_fixture(tmp_path, monkeypatch):
    datasets = []
    for dataset in exporter.EXPECTED_DATASETS:
        datasets.append(
            {
                "name": dataset,
                "train_split": "trainval" if dataset == "pet" else "train",
                "eval_split": "test",
                "train_count": 10,
                "train_sample_id_sha256": hashlib.sha256(
                    f"{dataset}-train".encode()
                ).hexdigest(),
                "eval_count": 3,
                "eval_sample_id_sha256": hashlib.sha256(
                    f"{dataset}-eval".encode()
                ).hexdigest(),
            }
        )
    models = [
        {"name": "clipseg", "condition": "clipseg-target"},
        {"name": "deeplabv3", "condition": "deeplabv3-target"},
    ]
    spec = {
        "datasets": datasets,
        "models": models,
        "protocol": {
            "training_seeds": [1, 2],
            "gamma": 0.5,
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "quadrature_seed": 0,
            "checkpoint_rule": "final_epoch_40",
        },
        "paths": {"analysis_root": (tmp_path / "private-analysis").as_posix()},
    }
    spec_sha = "1" * 64
    checkpoint_sha = "2" * 64
    downstream_sha = "3" * 64
    canonical_sha = "4" * 64
    analysis_source_sha = "5" * 64

    source_files = [
        {"path": "selectseg/train.py", "sha256": "6" * 64},
        {"path": "selectseg/data.py", "sha256": "7" * 64},
        {"path": "selectseg/models.py", "sha256": "8" * 64},
        {"path": "selectseg/freeze_binary_maps.py", "sha256": "9" * 64},
        {"path": "selectseg/binary_artifacts.py", "sha256": "a" * 64},
    ]
    clip_revision = "999e0328d9e10b484360c477313983f9afdd7050"
    base_model_files = [
        {
            "path": (
                "data/cache/huggingface/hub/models--CIDAS--clipseg-rd64-refined/"
                f"snapshots/{clip_revision}/config.json"
            ),
            "sha256": "8" * 64,
        },
        {
            "path": "data/cache/torch/hub/checkpoints/"
            "deeplabv3_resnet50_coco-cd0a2569.pth",
            "sha256": "9" * 64,
        },
    ]
    binding = {
        "path": tmp_path / "private" / "seed-spec.lock.json",
        "sha256": spec_sha,
        "spec": spec,
        "lock": {
            "spec": {"path": "configs/seed.json", "sha256": "a" * 64},
            "canonical_campaign_lock": {
                "path": "outputs/campaign.lock.json",
                "sha256": "b" * 64,
                "campaign_id": "binary-midpoint-main-v1",
            },
            "estimator_spec": {
                "path": "configs/estimators/midpoint-v1.json",
                "sha256": "c" * 64,
            },
            "source_files": source_files,
            "base_model_files": base_model_files,
        },
    }
    _write_json(binding["path"], {"fixture": True})

    checkpoints = []
    experiments = []
    for dataset_row in datasets:
        for model_row in models:
            for seed in (1, 2):
                identity = f"{dataset_row['name']}-{model_row['name']}-s{seed}"
                history = _write_json(
                    tmp_path / "history" / f"{identity}.json",
                    [{"epoch": epoch} for epoch in range(1, 41)],
                )
                train_record = _write_json(
                    tmp_path / "records" / f"{identity}.json", {"identity": identity}
                )
                checkpoint = {
                    "dataset": dataset_row["name"],
                    "model": model_row["name"],
                    "condition": model_row["condition"],
                    "training_seed": seed,
                    "checkpoint_path": f"private/{identity}/checkpoint.pt",
                    "checkpoint_sha256": hashlib.sha256(
                        f"checkpoint-{identity}".encode()
                    ).hexdigest(),
                    "checkpoint_size_bytes": 1234,
                    "train_config_path": f"private/{identity}/train.json",
                    "train_config_sha256": hashlib.sha256(
                        f"config-{identity}".encode()
                    ).hexdigest(),
                    "history_path": history.as_posix(),
                    "history_sha256": _sha(history),
                    "train_record_path": train_record.as_posix(),
                    "train_record_sha256": _sha(train_record),
                }
                checkpoints.append(checkpoint)
                experiments.append(
                    {
                        "dataset": dataset_row,
                        "model": model_row,
                        "training_seed": seed,
                    }
                )
    checkpoint_binding = {
        "path": tmp_path / "private" / "checkpoints.lock.json",
        "sha256": checkpoint_sha,
        "lock": {"checkpoints": checkpoints},
    }
    _write_json(checkpoint_binding["path"], {"fixture": True})

    analysis_cells = {}
    for dataset_row in datasets:
        for model_row in models:
            analysis_cells[(dataset_row["name"], model_row["condition"])] = {
                "dataset": dataset_row["name"],
                "condition": model_row["condition"],
                "model": model_row["name"],
                "num_images_per_seed": dataset_row["eval_count"],
                "sources": {},
                "summary": {"raw_aurc": {}, "contrasts": {}},
            }

    freeze_rows_by_seed = {1: [], 2: []}
    artifact_objects = {}
    condition_objects = {}
    diagnostic_objects = {}
    diagnostic_paths = []
    stage_sha = {
        "freeze": "d" * 64,
        "common": "e" * 64,
        "simulation": "f" * 64,
        "assembly": "0" * 64,
        "diagnostic": "1" * 64,
    }
    campaign_sha_by_seed = {1: "2" * 64, 2: "3" * 64}

    for dataset_row in datasets:
        for model_row in models:
            for seed in (0, 1, 2):
                run_id = hashlib.sha256(
                    f"run-{dataset_row['name']}-{model_row['name']}-{seed}".encode()
                ).hexdigest()[:16]
                records = (
                    tmp_path / "assemblies" / f"seed-{seed}" / run_id / "records.jsonl"
                )
                records.parent.mkdir(parents=True, exist_ok=True)
                records.write_text("{}\n", encoding="utf-8")
                records_sha = _sha(records)
                if seed == 0:
                    manifest = _write_json(
                        records.with_name("manifest.json"), {"run_id": run_id}
                    )
                    analysis_cells[(dataset_row["name"], model_row["condition"])][
                        "sources"
                    ]["0"] = {
                        "records": records.as_posix(),
                        "records_sha256": records_sha,
                        "manifest": manifest.as_posix(),
                        "manifest_sha256": _sha(manifest),
                    }
                    continue

                identity = f"{dataset_row['name']}-{model_row['name']}-s{seed}"
                checkpoint = next(
                    row
                    for row in checkpoints
                    if (row["dataset"], row["model"], row["training_seed"])
                    == (dataset_row["name"], model_row["name"], seed)
                )
                artifact_id = hashlib.sha256(
                    f"artifact-{identity}".encode()
                ).hexdigest()[:16]
                artifact_manifest_path = _write_json(
                    tmp_path / "artifacts" / artifact_id / "manifest.json",
                    {"artifact_id": artifact_id},
                )
                artifact_manifest_sha = _sha(artifact_manifest_path)
                artifact_manifest = {
                    "artifact_id": artifact_id,
                    "dataset": dataset_row["name"],
                    "condition": model_row["condition"],
                    "model": model_row["name"],
                    "num_samples": dataset_row["eval_count"],
                    "sample_id_sha256": dataset_row["eval_sample_id_sha256"],
                    "source_sha256": stage_sha["freeze"],
                }
                artifact_objects[artifact_manifest_path.resolve()] = SimpleNamespace(
                    manifest=artifact_manifest,
                    manifest_sha256=artifact_manifest_sha,
                    manifest_path=artifact_manifest_path,
                )
                freeze_rows_by_seed[seed].append(
                    {
                        "dataset": dataset_row["name"],
                        "model": model_row["name"],
                        "condition": model_row["condition"],
                        "training_seed": seed,
                        "artifact_manifest_path": artifact_manifest_path.as_posix(),
                        "artifact_manifest_sha256": artifact_manifest_sha,
                    }
                )

                common_jsonl_sha = hashlib.sha256(
                    f"common-{identity}".encode()
                ).hexdigest()
                common_manifest = _write_json(
                    tmp_path / "common" / identity / "manifest.json",
                    {
                        "jsonl_sha256": common_jsonl_sha,
                        "source_sha256": stage_sha["common"],
                    },
                )
                simulation = {}
                for count in (2, 8, 32):
                    simulation_jsonl_sha = hashlib.sha256(
                        f"simulation-{identity}-{count}".encode()
                    ).hexdigest()
                    simulation_manifest = _write_json(
                        tmp_path
                        / "simulation"
                        / identity
                        / str(count)
                        / "manifest.json",
                        {
                            "jsonl_sha256": simulation_jsonl_sha,
                            "source_sha256": stage_sha["simulation"],
                        },
                    )
                    simulation[str(count)] = {
                        "path": simulation_manifest.as_posix(),
                        "sha256": _sha(simulation_manifest),
                        "jsonl_sha256": simulation_jsonl_sha,
                    }
                assembly_manifest_value = {
                    "run_id": run_id,
                    "dataset": dataset_row["name"],
                    "condition": model_row["condition"],
                    "model": model_row["name"],
                    "num_images": dataset_row["eval_count"],
                    "sample_id_sha256": dataset_row["eval_sample_id_sha256"],
                    "checkpoint": {"sha256": checkpoint["checkpoint_sha256"]},
                    "source_sha256": stage_sha["common"],
                    "jsonl_sha256": records_sha,
                    "assembly": {
                        "campaign_lock_sha256": campaign_sha_by_seed[seed],
                        "artifact_manifest_sha256": artifact_manifest_sha,
                        "assembly_source_sha256": stage_sha["assembly"],
                        "common_manifest": {
                            "path": common_manifest.as_posix(),
                            "sha256": _sha(common_manifest),
                            "jsonl_sha256": common_jsonl_sha,
                        },
                        "common_manifest_sha256": _sha(common_manifest),
                        "common_jsonl_sha256": common_jsonl_sha,
                        "simulation_manifests": simulation,
                    },
                }
                assembly_manifest = _write_json(
                    records.with_name("manifest.json"), assembly_manifest_value
                )
                condition_objects[records] = SimpleNamespace(
                    manifest=assembly_manifest_value,
                    manifest_path=assembly_manifest,
                    jsonl_path=records,
                )
                analysis_cells[(dataset_row["name"], model_row["condition"])][
                    "sources"
                ][str(seed)] = {
                    "records": records.as_posix(),
                    "records_sha256": records_sha,
                    "manifest": assembly_manifest.as_posix(),
                    "manifest_sha256": _sha(assembly_manifest),
                }

                diagnostic_path = _write_json(
                    tmp_path / "diagnostics" / artifact_id / "diagnostics.json",
                    {"diagnostic_id": artifact_id},
                )
                diagnostic_paths.append(diagnostic_path)
                diagnostic_objects[diagnostic_path.resolve()] = SimpleNamespace(
                    summary_path=diagnostic_path,
                    summary_sha256=_sha(diagnostic_path),
                    summary={
                        "diagnostic_id": artifact_id,
                        "source_sha256": stage_sha["diagnostic"],
                        "artifact": {
                            "artifact_id": artifact_id,
                            "manifest_sha256": artifact_manifest_sha,
                            "sample_id_sha256": dataset_row["eval_sample_id_sha256"],
                            "num_samples": dataset_row["eval_count"],
                            "dataset": dataset_row["name"],
                            "condition": model_row["condition"],
                            "model": model_row["name"],
                        },
                        "descriptors": {
                            "included": True,
                            "sha256": hashlib.sha256(
                                f"descriptor-{identity}".encode()
                            ).hexdigest(),
                            "num_rows": dataset_row["eval_count"],
                        },
                    },
                )

    ordered_analysis_cells = [analysis_cells[key] for key in sorted(analysis_cells)]
    for cell_index, cell in enumerate(ordered_analysis_cells):
        raw_aurc = {}
        for risk_index, (risk, _) in enumerate(seed_analyzer.RISKS):
            raw_aurc[risk] = {}
            for method_index, (method, _) in enumerate(seed_analyzer.METHODS):
                raw_aurc[risk][method] = seed_analyzer._three_seed_summary(
                    {
                        seed: (
                            0.1
                            + 0.1 * risk_index
                            + 0.005 * method_index
                            + 0.001 * seed
                            + 0.00001 * cell_index
                        )
                        for seed in (0, 1, 2)
                    }
                )
        contrasts = {}
        for contrast in seed_analyzer.CONTRASTS:
            contrasts[contrast.name] = seed_analyzer._contrast_seed_summary(
                {
                    seed: (
                        raw_aurc[contrast.risk][contrast.left]["values"][str(seed)]
                        - raw_aurc[contrast.risk][contrast.right]["values"][str(seed)]
                    )
                    for seed in (0, 1, 2)
                }
            )
        cell["summary"] = {"raw_aurc": raw_aurc, "contrasts": contrasts}

    analysis = {
        "schema_version": 1,
        "analysis": {
            "estimand": (
                "descriptive target-model training-seed variation over seeds 0,1,2"
            ),
            "replication_unit": "one independently trained checkpoint",
            "inference": "none; no image pooling and no seed-level hypothesis test",
            "statistics": "three values, mean, range, and sample standard deviation",
            "aurc_scale": "raw [0,1]; renderers may display 100 x AURC",
            "contrast_definition": "AURC(left score) - AURC(right score)",
            "contrast_definitions": [
                asdict(contrast) for contrast in seed_analyzer.CONTRASTS
            ],
            "cohort_join_fields": list(seed_analyzer.COHORT_JOIN_FIELDS),
        },
        "provenance": {
            "downstream_lock": {
                "path": "private/downstream.json",
                "sha256": downstream_sha,
            },
            "canonical_seed0": {
                "path": "private/canonical.json",
                "sha256": canonical_sha,
                "campaign_lock_path": "private/campaign.json",
                "campaign_lock_sha256": "b" * 64,
            },
            "analysis_source_sha256": analysis_source_sha,
        },
        "cells": ordered_analysis_cells,
    }
    analysis["gate_c"] = seed_analyzer._gate_c(analysis["cells"])
    analysis_path = _write_json(
        tmp_path / "private-analysis" / "analysis.json", analysis
    )
    canonical_path = _write_json(tmp_path / "private" / "canonical.json", {"ok": True})
    # The fixed synthetic binding uses the exact canonical bytes below.
    canonical_sha = _sha(canonical_path)
    analysis["provenance"]["canonical_seed0"]["sha256"] = canonical_sha
    _write_json(analysis_path, analysis)
    table_path = tmp_path / "private-analysis" / "seed_robustness.tex"
    table_path.write_text("TABLE\n", encoding="utf-8")

    campaign_bindings = []
    lock_campaigns = []
    for seed in (1, 2):
        campaign = {
            "campaign_id": f"binary-seed{seed}-v1",
            "artifacts": [],
            "paths": {},
        }
        config = SimpleNamespace(
            sha256=hashlib.sha256(f"config-{seed}".encode()).hexdigest()
        )
        campaign_bindings.append(
            {
                "training_seed": seed,
                "config": config,
                "campaign_lock_path": tmp_path / f"campaign-seed-{seed}.json",
                "campaign_lock_sha256": campaign_sha_by_seed[seed],
                "campaign": campaign,
            }
        )
        lock_campaigns.append(
            {"training_seed": seed, "freeze_records": freeze_rows_by_seed[seed]}
        )
    downstream_binding = {
        "path": tmp_path / "private" / "downstream.lock.json",
        "sha256": downstream_sha,
        "binding": binding,
        "checkpoint_binding": checkpoint_binding,
        "lock": {"campaigns": lock_campaigns},
        "campaigns": tuple(campaign_bindings),
    }
    _write_json(downstream_binding["path"], {"fixture": True})

    plans = {
        phase: _jobs(phase, count)
        for phase, count in exporter.EXPECTED_PHASE_COUNTS.items()
    }
    train_receipt = _write_receipt(
        tmp_path / "receipts" / "train.jsonl", plans["train"]
    )
    downstream_receipts = {
        phase: _write_receipt(tmp_path / "receipts" / f"{phase}.jsonl", plans[phase])
        for phase in exporter.DOWNSTREAM_RECEIPT_NAMES
    }

    private_ledger = tmp_path / "private" / "scheduler.jsonl"
    private_ledger.write_text("{}\n", encoding="utf-8")
    scheduler_summary = tmp_path / "release" / "seed_scheduler_summary.json"

    monkeypatch.setattr(exporter, "load_spec_lock", lambda *a, **k: binding)
    monkeypatch.setattr(
        exporter, "load_checkpoint_lock", lambda *a, **k: checkpoint_binding
    )
    monkeypatch.setattr(
        exporter, "load_downstream_lock", lambda *a, **k: downstream_binding
    )
    monkeypatch.setattr(
        exporter, "validate_analysis_document", lambda value: analysis_cells
    )
    monkeypatch.setattr(exporter, "analyze_seed_extension", lambda *a, **k: analysis)
    monkeypatch.setattr(exporter, "render_table", lambda *a, **k: "TABLE\n")
    monkeypatch.setattr(exporter, "iter_experiments", lambda value: iter(experiments))
    monkeypatch.setattr(
        exporter,
        "load_binary_artifact",
        lambda path, **k: artifact_objects[Path(path).resolve()],
    )
    monkeypatch.setattr(
        exporter, "load_condition", lambda path: condition_objects[Path(path)]
    )
    monkeypatch.setattr(
        exporter,
        "load_binary_diagnostics",
        lambda path, **k: diagnostic_objects[Path(path).resolve()],
    )
    monkeypatch.setattr(exporter, "_expected_job_plans", lambda *a, **k: plans)
    monkeypatch.setattr(exporter, "TRAIN_RECEIPT", train_receipt)
    monkeypatch.setattr(
        exporter,
        "_expected_receipt_path",
        lambda binding_value, phase: downstream_receipts[phase],
    )

    def load_scheduler_closure(*args, **kwargs):
        del args
        expected_job_bindings = {
            (job.key[0], job.key[1], job.key[2]): str(900000 + index)
            for index, job in enumerate(plans["train"])
        }
        assert kwargs["expected_job_bindings"] == expected_job_bindings
        private_sha = _sha(private_ledger)
        event, summary = _scheduler_event(
            spec_sha,
            kwargs["expected_receipt_sha256"],
            kwargs["expected_record_set_sha256"],
            private_sha,
        )
        _write_json(scheduler_summary, summary)
        return {
            "private": {
                "event": event,
                "private_ledger_sha256": private_sha,
            },
            "public": summary,
        }

    monkeypatch.setattr(
        exporter, "load_scheduler_accounting_closure", load_scheduler_closure
    )
    monkeypatch.setattr(exporter, "_renderer_source_sha256", lambda: "4" * 64)

    arguments = {
        "spec_lock": binding["path"],
        "expected_spec_lock_sha256": spec_sha,
        "checkpoint_lock": checkpoint_binding["path"],
        "expected_checkpoint_lock_sha256": checkpoint_sha,
        "downstream_lock": downstream_binding["path"],
        "expected_downstream_lock_sha256": downstream_sha,
        "canonical_analysis": canonical_path,
        "expected_canonical_analysis_sha256": canonical_sha,
        "seed_analysis": analysis_path,
        "expected_seed_analysis_sha256": _sha(analysis_path),
        "table": table_path,
        "expected_table_sha256": _sha(table_path),
        "train_receipt": train_receipt,
        "downstream_receipts": downstream_receipts,
        "diagnostic_summaries": diagnostic_paths,
        "private_scheduler_ledger": private_ledger,
        "public_scheduler_summary": scheduler_summary,
        "public_analysis_output": (
            tmp_path / "release" / "seed_robustness_analysis.json"
        ),
        "public_provenance_output": tmp_path / "release" / "seed_provenance.json",
    }
    return SimpleNamespace(
        args=arguments,
        analysis=analysis,
        plans=plans,
        train_receipt=train_receipt,
        downstream_receipts=downstream_receipts,
    )


def test_complete_release_is_deterministic_path_free_and_accounts_for_162_jobs(
    complete_fixture,
):
    result = exporter.build_seed_publication(**complete_fixture.args)
    first_analysis = result["public_analysis"].read_bytes()
    first_provenance = result["public_provenance"].read_bytes()
    repeated = exporter.build_seed_publication(**complete_fixture.args)

    assert repeated["status"] == {"analysis": "unchanged", "guard": "unchanged"}
    assert result["public_analysis"].read_bytes() == first_analysis
    assert result["public_provenance"].read_bytes() == first_provenance

    public_analysis = json.loads(first_analysis)
    provenance = json.loads(first_provenance)
    assert public_analysis["artifact_type"] == exporter.PUBLIC_ANALYSIS_ARTIFACT_TYPE
    assert len(public_analysis["cells"]) == 10
    assert len(provenance["training"]) == 20
    assert len(provenance["cells"]) == 20
    assert [row["phase_id"] for row in provenance["phases"]] == list(
        exporter.PUBLIC_PHASES
    )
    assert sum(row["completed_jobs"] for row in provenance["phases"]) == 162
    assert provenance["scheduler"]["status"] == "complete"
    assert provenance["scheduler"]["successful_jobs"] == 20
    loaded = exporter.load_public_seed_release(
        result["public_analysis"],
        complete_fixture.args["public_scheduler_summary"],
        result["public_provenance"],
    )
    assert loaded["analysis"] == public_analysis
    assert loaded["scheduler"] == provenance["scheduler"]
    assert loaded["provenance"] == provenance

    text = (first_analysis + first_provenance).decode("utf-8")
    forbidden = (
        "/scratch/",
        "/home/",
        "saffo-a100",
        "apollo_agate",
        "ssafo",
        "sbatch",
        "SLURM_JOB_ID",
        "private-analysis",
        "checkpoint.pt",
    )
    assert not any(marker in text for marker in forbidden)
    assert all(
        set(source) == {"logical_id", "records_sha256", "manifest_sha256"}
        for cell in public_analysis["cells"]
        for source in cell["sources"].values()
    )


def test_missing_or_extra_downstream_receipt_fails_before_publication(complete_fixture):
    arguments = dict(complete_fixture.args)
    receipts = dict(arguments["downstream_receipts"])
    receipts.pop("diagnose")
    arguments["downstream_receipts"] = receipts
    with pytest.raises(ValueError, match="exactly freeze/common/score"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()
    assert not Path(arguments["public_provenance_output"]).exists()

    receipts["diagnose"] = complete_fixture.downstream_receipts["diagnose"]
    receipts["extra"] = complete_fixture.downstream_receipts["render"]
    with pytest.raises(ValueError, match="exactly freeze/common/score"):
        exporter.build_seed_publication(**arguments)


def test_receipt_identity_or_command_tamper_is_rejected(complete_fixture):
    arguments = dict(complete_fixture.args)
    tampered = Path(arguments["downstream_receipts"]["score"])
    _write_receipt(tampered, complete_fixture.plans["score"], changed_command=True)
    with pytest.raises(ValueError, match="submission command changed"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_provenance_output"]).exists()


def test_receipt_job_ids_must_be_unique_across_all_eight_phases(complete_fixture):
    arguments = dict(complete_fixture.args)
    train_rows = [
        json.loads(line)
        for line in Path(arguments["train_receipt"]).read_text().splitlines()
    ]
    reused_id = next(row["job_id"] for row in train_rows if row["job_id"] is not None)
    freeze_path = Path(arguments["downstream_receipts"]["freeze"])
    freeze_rows = [json.loads(line) for line in freeze_path.read_text().splitlines()]
    next(row for row in freeze_rows if row["job_id"] is not None)["job_id"] = reused_id
    freeze_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in freeze_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="162 globally unique jobs"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()
    assert not Path(arguments["public_provenance_output"]).exists()


def test_receipt_job_id_leading_zero_alias_is_rejected(complete_fixture):
    arguments = dict(complete_fixture.args)
    train_rows = [
        json.loads(line)
        for line in Path(arguments["train_receipt"]).read_text().splitlines()
    ]
    reused_id = next(row["job_id"] for row in train_rows if row["job_id"] is not None)
    freeze_path = Path(arguments["downstream_receipts"]["freeze"])
    freeze_rows = [json.loads(line) for line in freeze_path.read_text().splitlines()]
    next(row for row in freeze_rows if row["job_id"] is not None)["job_id"] = (
        f"0{reused_id}"
    )
    freeze_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in freeze_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical positive"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()
    assert not Path(arguments["public_provenance_output"]).exists()


def test_analysis_or_table_tamper_fails_closed(complete_fixture, monkeypatch):
    arguments = dict(complete_fixture.args)
    monkeypatch.setattr(
        exporter,
        "analyze_seed_extension",
        lambda *a, **k: {**complete_fixture.analysis, "gate_c": {"fired": True}},
    )
    with pytest.raises(ValueError, match="strict recomputation"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()

    monkeypatch.setattr(
        exporter,
        "analyze_seed_extension",
        lambda *a, **k: complete_fixture.analysis,
    )
    Path(arguments["table"]).write_text("TAMPERED\n", encoding="utf-8")
    arguments["expected_table_sha256"] = _sha(arguments["table"])
    with pytest.raises(ValueError, match="current renderer"):
        exporter.build_seed_publication(**arguments)


def test_exactly_twenty_explicit_diagnostics_are_required(complete_fixture):
    arguments = dict(complete_fixture.args)
    arguments["diagnostic_summaries"] = arguments["diagnostic_summaries"][:-1]
    with pytest.raises(ValueError, match="exactly 20 diagnostic"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()


def test_payload_guard_interruption_is_recoverable_and_never_overwrites(
    complete_fixture, monkeypatch
):
    arguments = dict(complete_fixture.args)
    original = exporter._atomic_new_or_identical
    calls = 0

    def interrupt(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(path, payload)

    monkeypatch.setattr(exporter, "_atomic_new_or_identical", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        exporter.build_seed_publication(**arguments)
    assert Path(arguments["public_analysis_output"]).is_file()
    assert not Path(arguments["public_provenance_output"]).exists()

    monkeypatch.setattr(exporter, "_atomic_new_or_identical", original)
    result = exporter.build_seed_publication(**arguments)
    assert result["status"] == {"analysis": "unchanged", "guard": "published"}

    Path(arguments["public_analysis_output"]).write_text("conflict\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicting"):
        exporter.build_seed_publication(**arguments)


def test_public_release_loader_rejects_payload_or_scheduler_drift(complete_fixture):
    result = exporter.build_seed_publication(**complete_fixture.args)
    analysis_path = result["public_analysis"]
    scheduler_path = Path(complete_fixture.args["public_scheduler_summary"])
    provenance_path = result["public_provenance"]

    analysis = json.loads(analysis_path.read_text())
    analysis["cells"][0]["sources"]["1"]["records_sha256"] = "f" * 64
    _write_json(analysis_path, analysis)
    with pytest.raises(ValueError, match="does not bind the analysis bytes"):
        exporter.load_public_seed_release(
            analysis_path, scheduler_path, provenance_path
        )

    provenance = json.loads(provenance_path.read_text())
    provenance["analysis"]["portable_analysis_sha256"] = _sha(analysis_path)
    _write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="differs from its provenance cell"):
        exporter.load_public_seed_release(
            analysis_path, scheduler_path, provenance_path
        )

    # Restore the actual public analysis from a deterministic rebuild.
    analysis_path.unlink()
    provenance_path.unlink()
    rebuilt = exporter.build_seed_publication(**complete_fixture.args)
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["duration_seconds"]["total_seconds"] += 1
    _write_json(scheduler_path, scheduler)
    with pytest.raises(ValueError, match="different scheduler summary"):
        exporter.load_public_seed_release(
            rebuilt["public_analysis"], scheduler_path, rebuilt["public_provenance"]
        )


def test_conflicting_guard_is_rejected_before_payload_publication(complete_fixture):
    arguments = dict(complete_fixture.args)
    guard = Path(arguments["public_provenance_output"])
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text("stale guard\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflicting"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()
    assert guard.read_text(encoding="utf-8") == "stale guard\n"


def test_public_schema_rejects_private_nested_fields(complete_fixture):
    result = exporter.build_seed_publication(**complete_fixture.args)
    value = json.loads(result["public_provenance"].read_text())
    value["campaign"]["private_path"] = "/scratch/private/run"
    with pytest.raises(ValueError, match="private infrastructure|forbidden keys"):
        exporter._validate_public_provenance(value)

    value = json.loads(result["public_provenance"].read_text())
    value["campaign"]["access_token"] = "redacted"
    with pytest.raises(ValueError, match="credential-bearing key"):
        exporter._validate_public_provenance(value)


def test_public_analysis_rejects_benign_unknown_nested_fields(complete_fixture):
    result = exporter.build_seed_publication(**complete_fixture.args)
    original = json.loads(result["public_analysis"].read_text())
    selectors = (
        lambda value: value["analysis"],
        lambda value: value["provenance"],
        lambda value: value["provenance"]["canonical_seed0"],
        lambda value: value["cells"][0],
        lambda value: value["cells"][0]["sources"]["0"],
        lambda value: value["cells"][0]["summary"],
        lambda value: value["cells"][0]["summary"]["raw_aurc"]["risk_dice"][
            "confidence_sdc"
        ],
        lambda value: value["cells"][0]["summary"]["contrasts"][
            "dice_vs_nhd_under_dice"
        ],
        lambda value: value["gate_c"],
    )
    for select in selectors:
        candidate = copy.deepcopy(original)
        select(candidate)["benign_extra"] = "ordinary-value"
        with pytest.raises(ValueError):
            exporter._validate_public_analysis(candidate)


def test_public_provenance_rejects_benign_unknown_nested_fields(complete_fixture):
    result = exporter.build_seed_publication(**complete_fixture.args)
    original = json.loads(result["public_provenance"].read_text())
    selectors = (
        lambda value: value["campaign"],
        lambda value: value["campaign"]["canonical_seed0"],
        lambda value: value["campaign"]["seed_campaigns"][0],
        lambda value: value["campaign"]["protocol"],
        lambda value: value["campaign"]["estimator"],
        lambda value: value["campaign"]["grid"],
        lambda value: value["datasets"][0],
        lambda value: value["base_models"][0],
        lambda value: value["base_models"][0]["files"][0],
        lambda value: value["code"],
        lambda value: value["code"]["locked_source_files"][0],
        lambda value: value["training"][0],
        lambda value: value["cells"][0],
        lambda value: value["cells"][0]["frozen"],
        lambda value: value["cells"][0]["assembly"],
        lambda value: value["cells"][0]["assembly"]["simulation_manifest_sha256"],
        lambda value: value["cells"][0]["diagnostics"],
        lambda value: value["phases"][0],
        lambda value: value["scheduler"],
        lambda value: value["scheduler"]["bindings"],
        lambda value: value["analysis"],
    )
    for select in selectors:
        candidate = copy.deepcopy(original)
        select(candidate)["benign_extra"] = "ordinary-value"
        with pytest.raises(ValueError):
            exporter._validate_public_provenance(candidate)


def test_public_output_names_and_directory_are_fixed(complete_fixture):
    arguments = dict(complete_fixture.args)
    arguments["public_analysis_output"] = Path(
        arguments["public_analysis_output"]
    ).with_name("analysis.json")
    with pytest.raises(ValueError, match="seed_robustness_analysis.json"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()

    arguments = dict(complete_fixture.args)
    arguments["public_provenance_output"] = (
        Path(arguments["public_provenance_output"]).parent
        / "different"
        / "seed_provenance.json"
    )
    with pytest.raises(ValueError, match="share one directory"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()
    assert not Path(arguments["public_provenance_output"]).exists()

    arguments = dict(complete_fixture.args)
    arguments["public_scheduler_summary"] = Path(
        arguments["public_scheduler_summary"]
    ).with_name("scheduler.json")
    with pytest.raises(ValueError, match="seed_scheduler_summary.json"):
        exporter.build_seed_publication(**arguments)
    assert not Path(arguments["public_analysis_output"]).exists()


def test_resolve_rejects_a_symlinked_input_ancestor(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        exporter._resolve(link / "manifest.json")
