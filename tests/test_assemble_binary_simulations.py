"""Focused real-artifact tests for common + M-shard strict assembly."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze.main import load_condition
from scripts.assemble import (
    FINAL_SCORE_FIELDS,
    assemble,
)
from scripts.submit.main import (
    build_campaign_lock,
    load_config,
    plan_common_jobs,
    plan_score_jobs,
    write_campaign_lock,
)
from selectseg.artifacts import write_binary_artifact
from selectseg.pipeline.common import (
    AUXILIARY_FIELDS,
    COMMON_SCORE_FIELDS,
    RISK_FIELDS,
    parse_args as parse_common_args,
    run_common,
)
from selectseg.pipeline.score import (
    parse_args as parse_simulation_args,
    run_simulation,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_config(tmp_path):
    estimator = tmp_path / "midpoint-v1.json"
    estimator.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimator_id": "midpoint-v1",
                "target_measure": "uniform-threshold",
                "rule": "midpoint",
                "randomized": False,
                "required_seed": 0,
            }
        )
        + "\n"
    )
    config = {
        "config_schema_version": 1,
        "campaign_id": "unit-main-v1",
        "protocol": {
            "gamma_values": [0.5],
            "m_values": [2, 8, 32],
            "quadrature_rule": "midpoint-v1",
            "seeds": [0],
        },
        "gpu_partitions": ["saffo-a100", "apollo_agate"],
        "estimator_spec": str(estimator),
        "paths": {
            "artifact_output_root": str(tmp_path / "artifacts"),
            "common_output_root": str(tmp_path / "common"),
            "simulation_output_root": str(tmp_path / "simulations"),
            "assembly_output_root": str(tmp_path / "assembled"),
        },
        "conditions": [
            {
                "dataset": "pet",
                "condition": "clipseg-general",
                "model": "clipseg",
                "checkpoint": None,
                "batch_size": 2,
                "expected_num_samples": 3,
            }
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return load_config(path)


def _write_artifact(tmp_path):
    probabilities = []
    truths = []
    for index in range(3):
        probability = np.linspace(
            0.02 + index / 100,
            0.98 - index / 100,
            12 * 12,
            dtype=np.float32,
        ).reshape(12, 12)
        truth = (np.roll(probability, index, axis=1) >= 0.53).astype(np.uint8)
        probabilities.append(probability)
        truths.append(truth)
    sample_ids = [f"image-{index}" for index in range(3)]
    return write_binary_artifact(
        tmp_path / "artifacts",
        dataset="pet",
        condition="clipseg-general",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="foreground",
        checkpoint=None,
        base_model={"name": "clipseg", "source": "synthetic-real-artifact"},
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
            "model_input": "synthetic",
            "probability_to_native_mask": "synthetic",
        },
        cohort="three canonical frozen samples",
        sample_ids=sample_ids,
        samples=list(zip(sample_ids, probabilities, truths, strict=True)),
        command=["pytest", "freeze"],
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _run_planned(job, wrapper, parser, runner):
    command = list(job.command)
    index = command.index(wrapper)
    return runner(parser(command[index + 4 :]))


def _campaign(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _write_config(tmp_path)
    artifact = _write_artifact(tmp_path)
    lock = build_campaign_lock(config, [artifact])
    lock_path, _ = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    _, common_manifest = _run_planned(
        plan_common_jobs(config, lock_path)[0],
        "scripts/slurm/run.sbatch",
        parse_common_args,
        run_common,
    )
    partials = []
    for job in plan_score_jobs(config, lock_path):
        _, manifest = _run_planned(
            job,
            "scripts/slurm/run.sbatch",
            parse_simulation_args,
            run_simulation,
        )
        partials.append(manifest)
    return lock_path, common_manifest, partials


def _rewrite_records(manifest_path, mutate):
    records = manifest_path.with_name("records.jsonl")
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    mutate(rows)
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(manifest_path.read_text())
    manifest["jsonl_sha256"] = _sha256(records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_real_artifact_common_three_scores_assemble_and_load_in_analyzer(tmp_path):
    lock_path, common, partials = _campaign(tmp_path)
    common_rows = [
        json.loads(line)
        for line in common.with_name("records.jsonl").read_text().splitlines()
    ]
    common_manifest = json.loads(common.read_text())
    assert common_manifest["risk_fields"] == list(RISK_FIELDS)
    assert common_manifest["auxiliary_fields"] == list(AUXILIARY_FIELDS)
    assert common_manifest["score_fields"] == list(COMMON_SCORE_FIELDS)
    assert all("risk_dice" in row for row in common_rows)
    for partial in partials:
        manifest = json.loads(partial.read_text())
        assert manifest["risk_fields"] == []
        assert manifest["auxiliary_fields"] == []
        assert len(manifest["score_fields"]) == 3
        rows = [
            json.loads(line)
            for line in partial.with_name("records.jsonl").read_text().splitlines()
        ]
        assert all(not (set(row) & set(RISK_FIELDS)) for row in rows)
        assert all(not (set(row) & set(COMMON_SCORE_FIELDS)) for row in rows)

    target = assemble(
        campaign_lock=lock_path,
        common=common,
        inputs=list(reversed(partials)),
        output_root=tmp_path / "assembled",
    )
    condition = load_condition(target / "records.jsonl")
    assert tuple(condition.manifest["score_fields"]) == FINAL_SCORE_FIELDS
    assert len(FINAL_SCORE_FIELDS) == 17
    assert set(condition.manifest["quadrature"]) == {"2", "8", "32"}
    assert condition.manifest["assembly"]["campaign_lock_sha256"] == _sha256(lock_path)
    assert len(condition.rows) == 3
    with pytest.raises(FileExistsError, match="already exists"):
        assemble(
            campaign_lock=lock_path,
            common=common,
            inputs=partials,
            output_root=tmp_path / "assembled",
        )


def test_assembler_checks_order_and_exact_non_float_identity_only(tmp_path):
    lock_path, common, partials = _campaign(tmp_path)
    _rewrite_records(
        partials[1], lambda rows: rows[1].__setitem__("height", rows[1]["height"] + 1)
    )
    with pytest.raises(ValueError, match="identity field 'height' differs"):
        assemble(
            campaign_lock=lock_path,
            common=common,
            inputs=partials,
            output_root=tmp_path / "assembled",
        )


def test_assembler_refuses_wrong_lock_incomplete_set_and_unknown_schema(tmp_path):
    lock_path, common, partials = _campaign(tmp_path)
    manifest = json.loads(partials[0].read_text())
    manifest["simulation"]["campaign_lock_sha256"] = "0" * 64
    partials[0].write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="different campaign-lock bytes"):
        assemble(
            campaign_lock=lock_path,
            common=common,
            inputs=partials,
            output_root=tmp_path / "assembled-lock",
        )

    with pytest.raises(ValueError, match="exactly 3 --input"):
        assemble(
            campaign_lock=lock_path,
            common=common,
            inputs=partials[:2],
            output_root=tmp_path / "assembled-short",
        )

    lock_path, common, partials = _campaign(tmp_path / "second")
    common_data = json.loads(common.read_text())
    common_data["unauthorized"] = True
    common.write_text(json.dumps(common_data) + "\n")
    with pytest.raises(ValueError, match="must contain exactly"):
        assemble(
            campaign_lock=lock_path,
            common=common,
            inputs=partials,
            output_root=tmp_path / "assembled-extra",
        )
