import json
from pathlib import Path

import pytest

from scripts.submit.counts import _load_receipt, plan_commands


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
