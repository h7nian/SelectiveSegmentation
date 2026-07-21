"""Planning contract for the independent probability-ensemble build wave."""

import json
from pathlib import Path

from scripts.submit_binary_probability_ensemble import plan_commands


def test_plan_has_one_non_array_job_per_condition_and_all_partition_candidates():
    config = json.loads(
        Path("configs/auxiliary/binary_probability_ensemble_v1.json").read_text()
    )
    commands = plan_commands(config, "a" * 64)

    assert len(commands) == 10
    assert len({key for key, _ in commands}) == 10
    expected_partitions = set(config["scheduler"]["cpu_partition_candidates"])
    for key, command in commands:
        assert key.count("/") == 1
        assert command[0:2] == ["sbatch", "--parsable"]
        partition_token = next(
            token for token in command if token.startswith("--partition=")
        )
        assert set(partition_token.split("=", 1)[1].split(",")) == expected_partitions
        assert not any(token.startswith("--array") for token in command)
        assert command.count("--dataset") == 1
        assert command.count("--condition") == 1
