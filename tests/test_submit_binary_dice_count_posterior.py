import json
from pathlib import Path

import pytest

from scripts.analyze.counts import _copula_variant_id
from scripts.submit.counts import _commands_equivalent, _load_receipt, plan_commands
from selectseg.artifacts import sha256_file


def test_real_contract_plans_ten_independent_rotated_jobs():
    config_path = Path("configs/auxiliary/dice_coupling_analysis_v1.json")
    config = json.loads(config_path.read_text())
    planned = plan_commands(config, config_path)
    assert len(planned) == 10
    assert len({key for key, _ in planned}) == 10
    partition_requests = []
    for _, command in planned:
        assert command[0:2] == ["sbatch", "--parsable"]
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert not any(token.startswith("--array") for token in command)
        partition_requests.append(command[command.index("--partition") + 1])
    assert len(set(partition_requests)) == 4


def test_partition_contract_uses_same_submitter_for_eighty_jobs():
    config_path = Path("configs/auxiliary/dice_partition_ladder_v1.json")
    config = json.loads(config_path.read_text())
    planned = plan_commands(config, config_path)
    assert len(planned) == 80
    assert len({key for key, _ in planned}) == 80
    assert all(
        command.count("scripts/slurm/run.sbatch") == 1
        for _, command in planned
    )


def test_spatial_copula_contract_uses_one_gpu_job_per_repeat():
    config_path = Path("configs/auxiliary/spatial_copula_v1.json")
    config = json.loads(config_path.read_text())
    planned = plan_commands(config, config_path)
    assert len(planned) == 10 * 6 * 4
    assert len({key for key, _ in planned}) == len(planned)
    for key, command in planned:
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert command.count("--repeat-index") == 1
        assert command.count("--posterior-draws") == 1
        assert command[command.index("--gres") + 1] == "gpu:1"
        assert set(command[command.index("--partition") + 1].split(",")) == {
            "saffo-a100",
            "apollo_agate",
        }
        assert "repeat-" in key
        assert not any(token.startswith("--array") for token in command)


def test_spatial_copula_filters_enable_one_runtime_pilot_job():
    config_path = Path("configs/auxiliary/spatial_copula_v1.json")
    config = json.loads(config_path.read_text())
    planned = plan_commands(
        config,
        config_path,
        condition_filters=["fives/clipseg-target"],
        variant_filters=["g50-s50-k05"],
        repeat_filters=[0],
    )
    assert len(planned) == 1
    key, command = planned[0]
    assert key == "fives/clipseg-target/g50-s50-k05/repeat-0"
    assert command[command.index("--device") + 1] == "cuda"


def test_spatial_copula_analysis_contract_binds_scoring_contract_and_variants():
    score_path = Path("configs/auxiliary/spatial_copula_v1.json")
    analysis_path = Path("configs/auxiliary/spatial_copula_analysis_v1.json")
    score = json.loads(score_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    assert analysis["score_contract"] == {
        "path": score_path.as_posix(),
        "sha256": sha256_file(score_path),
    }
    variants = {variant["id"]: variant for variant in score["variants"]}
    primary = variants[analysis["primary_comparison"]["variant_id"]]
    manifest = {
        "spatial_copula": {
            "global_variance_weight": primary["global_variance_weight"],
            "spatial_variance_weight": primary["spatial_variance_weight"],
            "spatial_knot_spacing_diagonal": primary[
                "spatial_knot_spacing_diagonal"
            ],
        }
    }
    assert _copula_variant_id(manifest, variants) == primary["id"]


def test_submission_receipt_is_a_strict_resume_guard(tmp_path):
    contract_sha256 = "a" * 64
    receipt = tmp_path / "receipt.jsonl"
    event = {
        "created_utc": "2026-07-21T00:00:00+00:00",
        "key": "pet/clipseg-target/action_components",
        "analysis_contract_sha256": contract_sha256,
        "command": ["sbatch", "scripts/slurm/run.sbatch", "python"],
        "job_id": "12345",
    }
    receipt.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert _load_receipt(receipt, contract_sha256) == {event["key"]: event}

    with pytest.raises(ValueError, match="another contract"):
        _load_receipt(receipt, "b" * 64)

    receipt.write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="duplicate receipt key"):
        _load_receipt(receipt, contract_sha256)


def test_resume_equivalence_ignores_only_candidate_partition_order():
    left = [
        "sbatch",
        "--partition",
        "saffo-a100,apollo_agate",
        "runner",
        "--repeat-index",
        "0",
    ]
    reordered = [
        "sbatch",
        "--partition",
        "apollo_agate,saffo-a100",
        "runner",
        "--repeat-index",
        "0",
    ]
    changed_repeat = [*reordered[:-1], "1"]
    changed_partitions = list(reordered)
    changed_partitions[2] = "apollo_agate"
    assert _commands_equivalent(left, reordered)
    assert not _commands_equivalent(left, changed_repeat)
    assert not _commands_equivalent(left, changed_partitions)
