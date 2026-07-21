from pathlib import Path

import pytest

from scripts import submit_scientific_input_components as submit


def test_plan_is_one_dataset_per_cpu_candidate_job(tmp_path):
    jobs = submit.plan_jobs(tmp_path / "components")
    assert len(jobs) == 5
    assert len({job.key for job in jobs}) == 5
    for job in jobs:
        command = list(job.command)
        assert command.count("scripts/slurm/build_scientific_dataset.sbatch") == 1
        assert command[command.index("--partition") + 1] == (
            "amdsmall,agsmall,msismall,saffo-2tb"
        )
        assert "--array" not in command
        assert not any(token.startswith("--array=") for token in command)


def test_plan_refuses_existing_component(tmp_path):
    output = tmp_path / "components"
    output.mkdir()
    (output / "pet.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        submit.plan_jobs(output)


def test_preflight_changes_only_parsable_flag(tmp_path, monkeypatch):
    calls = []

    def preflight(jobs):
        for job in jobs:
            calls.append(
                tuple(
                    "--test-only" if token == "--parsable" else token
                    for token in job.command
                )
            )
        return tuple(jobs)

    monkeypatch.setattr(submit, "preflight_plan", preflight)
    assert submit.main(
        [
            "--output-dir",
            str(tmp_path / "components"),
            "--scheduler-preflight-only",
        ]
    ) == ()
    assert len(calls) == 5
    assert all("--test-only" in command for command in calls)


def test_wrapper_has_no_hardcoded_partition_and_one_dataset_argument():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/build_scientific_dataset.sbatch"
    ).read_text()
    assert "#SBATCH --partition" not in wrapper
    assert "DATASET=$1" in wrapper
    assert "build-dataset" in wrapper
    assert "--dataset \"$DATASET\"" in wrapper
