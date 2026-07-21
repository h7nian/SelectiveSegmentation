import json
from pathlib import Path

from scripts.submit.counts import plan_commands


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
