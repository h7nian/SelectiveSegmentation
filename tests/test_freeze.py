"""CLI-level tests for decision-neutral frozen binary-map inference."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from selectseg.artifacts import load_binary_artifact, sha256_file
from selectseg.pipeline import freeze


class _FakeDataset:
    def __init__(self, spec, root, train, image_size, **kwargs):
        del spec, root
        assert train is False and image_size == 4
        self.verified_file_reader = kwargs.pop("verified_file_reader", None)
        assert not kwargs
        self.ids = ["sample-b", "sample-a"]
        self.images = [
            torch.full((3, 4, 4), 0.25, dtype=torch.float32),
            torch.full((3, 4, 4), 0.75, dtype=torch.float32),
        ]
        self.masks = [
            torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.long),
            torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.long),
        ]

    def __len__(self):
        return len(self.ids)

    def sample_id(self, index):
        return self.ids[index]

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


class _FakeModel(nn.Module):
    image_size = 4

    def __init__(self):
        super().__init__()
        self.predict_calls = 0

    def predict_probs(self, images):
        self.predict_calls += 1
        foreground = images[:, :1]
        return torch.cat([1 - foreground, foreground], dim=1)


def _patch_inference(monkeypatch, *, model=None, dataset_type=_FakeDataset):
    model = _FakeModel() if model is None else model
    monkeypatch.setattr(freeze, "SegDataset", dataset_type)
    monkeypatch.setattr(freeze, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return model


def _args(tmp_path, *extra):
    return [
        "--model",
        "deeplabv3",
        "--dataset",
        "kvasir",
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        *extra,
    ]


def test_cli_freezes_native_maps_in_declared_dataset_order(
    tmp_path, monkeypatch, capsys
):
    model = _patch_inference(monkeypatch)
    manifest_path = freeze.main(_args(tmp_path))
    artifact = load_binary_artifact(manifest_path)
    manifest = artifact.manifest

    assert model.predict_calls == 1
    assert manifest["dataset"] == "kvasir"
    assert manifest["condition"] == "deeplabv3-external"
    assert manifest["model"] == "deeplabv3"
    assert manifest["split"] == "test"
    assert manifest["class_index"] == 1
    assert manifest["class_name"] == "polyp"
    assert manifest["checkpoint"] is None
    assert manifest["num_samples"] == 2
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["environment"]["autocast_dtype"] == "disabled"
    assert "decision_rule" not in manifest
    assert "decision_threshold" not in manifest

    samples = list(artifact.iter_samples())
    assert [sample.sample_id for sample in samples] == ["sample-b", "sample-a"]
    assert [sample.foreground_probability.shape for sample in samples] == [
        (2, 3),
        (3, 2),
    ]
    assert samples[0].foreground_probability.tolist() == [
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
    ]
    assert set(samples[0].truth.ravel().tolist()) == {0, 1}

    output = capsys.readouterr().out.strip().splitlines()
    assert output[0] == f"saved {manifest_path}"
    assert output[1] == f"manifest_sha256={sha256_file(manifest_path)}"
    machine_line = json.loads(output[2])
    assert machine_line == {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }


def test_cli_limit_changes_the_frozen_cohort_without_reordering(tmp_path, monkeypatch):
    _patch_inference(monkeypatch)
    manifest_path = freeze.main(_args(tmp_path, "--limit", "1"))
    artifact = load_binary_artifact(manifest_path)
    samples = list(artifact.iter_samples())
    assert artifact.manifest["num_samples"] == 1
    assert artifact.manifest["cohort"] == (
        "first 1 images from a development-only subset"
    )
    assert [sample.sample_id for sample in samples] == ["sample-b"]


def test_cli_refuses_identical_rerun_before_second_inference(tmp_path, monkeypatch):
    model = _patch_inference(monkeypatch)
    first = freeze.main(_args(tmp_path))
    first_hash = sha256_file(first)
    assert model.predict_calls == 1
    with pytest.raises(FileExistsError, match="already exists"):
        freeze.main(_args(tmp_path))
    assert model.predict_calls == 1
    assert sha256_file(first) == first_hash


def test_cli_records_finetuned_checkpoint_identity(tmp_path, monkeypatch):
    _patch_inference(monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({}, checkpoint)
    manifest_path = freeze.main(_args(tmp_path, "--checkpoint", str(checkpoint)))
    manifest = load_binary_artifact(manifest_path).manifest
    assert manifest["condition"] == "deeplabv3-target"
    assert manifest["checkpoint"] == {
        "path": checkpoint.name,
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }


def test_scientific_freeze_binds_schema_three_before_inference(
    tmp_path, monkeypatch
):
    model = _patch_inference(monkeypatch)
    config = tmp_path / "campaign.json"
    config.write_text('{"campaign_id":"unit"}\n')
    lock = tmp_path / "scientific.lock.json"
    lock.write_text("{}\n")
    hashes = {
        "root_lock_sha256": "1" * 64,
        "science_projection_sha256": "2" * 64,
        "eval_dataset_component_sha256": "3" * 64,
        "source_component_sha256": "4" * 64,
        "base_model_component_sha256": "5" * 64,
        "checkpoint_component_sha256": "6" * 64,
        "environment_component_sha256": "7" * 64,
    }
    condition_sha = "8" * 64
    verify_calls = []

    def verify(path, **kwargs):
        verify_calls.append((path, kwargs))
        return {
            "checkpoint": None,
            "eval_dataset": object(),
            "scientific_input_hashes": hashes,
            "scientific_input_sha256": condition_sha,
        }

    monkeypatch.setattr(freeze, "verify_condition_inputs", verify)
    manifest_path = freeze.main(
        _args(
            tmp_path,
            "--campaign-config",
            str(config),
            "--expected-campaign-config-sha256",
            sha256_file(config),
            "--scientific-input-lock",
            str(lock),
            "--expected-scientific-input-lock-sha256",
            "9" * 64,
            "--expected-condition-input-sha256",
            condition_sha,
        )
    )
    manifest = load_binary_artifact(manifest_path, validate_payloads=False).manifest
    assert model.predict_calls == 1
    assert verify_calls[0][1]["mode"] == "consume"
    assert manifest["schema_version"] == 3
    assert manifest["scientific_input"] == {
        **hashes,
        "condition_input_sha256": condition_sha,
    }


def test_scientific_freeze_rejects_condition_hash_before_model_load(
    tmp_path, monkeypatch
):
    model = _patch_inference(monkeypatch)
    config = tmp_path / "campaign.json"
    config.write_text("{}\n")
    lock = tmp_path / "scientific.lock.json"
    lock.write_text("{}\n")
    monkeypatch.setattr(
        freeze,
        "verify_condition_inputs",
        lambda *args, **kwargs: {
            "checkpoint": None,
            "eval_dataset": object(),
            "scientific_input_hashes": {},
            "scientific_input_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="condition scientific-input"):
        freeze.main(
            _args(
                tmp_path,
                "--campaign-config",
                str(config),
                "--expected-campaign-config-sha256",
                sha256_file(config),
                "--scientific-input-lock",
                str(lock),
                "--expected-scientific-input-lock-sha256",
                "9" * 64,
                "--expected-condition-input-sha256",
                "8" * 64,
            )
        )
    assert model.predict_calls == 0


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("--batch-size", "0"), "batch-size"),
        (("--num-workers", "-1"), "num-workers"),
        (("--limit", "0"), "limit"),
        (("--expected-num-samples", "0"), "expected-num-samples"),
    ],
)
def test_cli_rejects_invalid_work_parameters(tmp_path, monkeypatch, extra, match):
    _patch_inference(monkeypatch)
    with pytest.raises(ValueError, match=match):
        freeze.main(_args(tmp_path, *extra))


def test_cli_rejects_wrong_predeclared_cohort_before_inference(tmp_path, monkeypatch):
    model = _patch_inference(monkeypatch)
    with pytest.raises(ValueError, match="expected 3"):
        freeze.main(_args(tmp_path, "--expected-num-samples", "3"))
    assert model.predict_calls == 0


def test_cli_require_cuda_fails_closed(tmp_path, monkeypatch):
    model = _patch_inference(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        freeze.main(_args(tmp_path, "--require-cuda"))
    assert model.predict_calls == 0


def test_cli_rejects_nonbinary_dataset_before_model_construction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model construction must not run")

    monkeypatch.setattr(freeze, "build_model", forbidden)
    with pytest.raises(ValueError, match="native binary"):
        freeze.main(
            [
                "--model",
                "deeplabv3",
                "--dataset",
                "voc",
                "--output-dir",
                str(tmp_path),
            ]
        )


class _WrongShapeModel(_FakeModel):
    def predict_probs(self, images):
        return images[:, :1]


def test_cli_rejects_wrong_model_output_and_does_not_publish(tmp_path, monkeypatch):
    _patch_inference(monkeypatch, model=_WrongShapeModel())
    with pytest.raises(ValueError, match=r"shape \(B, 2, H, W\)"):
        freeze.main(_args(tmp_path))
    condition_dir = tmp_path / "artifacts" / "kvasir" / "deeplabv3-external"
    assert not [path for path in condition_dir.glob("*") if path.is_dir()]
    assert not list(condition_dir.glob(".*.tmp-*"))


class _ChangingOrderDataset(_FakeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = [0, 0]

    def sample_id(self, index):
        self.calls[index] += 1
        if self.calls[index] > 1:
            return f"changed-{index}"
        return super().sample_id(index)


def test_cli_detects_dataset_order_mutation_during_inference(tmp_path, monkeypatch):
    _patch_inference(monkeypatch, dataset_type=_ChangingOrderDataset)
    with pytest.raises(RuntimeError, match="sample order changed"):
        freeze.main(_args(tmp_path))
    condition_dir = tmp_path / "artifacts" / "kvasir" / "deeplabv3-external"
    assert not [path for path in condition_dir.glob("*") if path.is_dir()]


def test_generic_slurm_runner_forwards_the_explicit_command():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "slurm"
        / "run.sbatch"
    ).read_text()
    assert "#SBATCH --partition" not in wrapper
    assert 'exec "$@"' in wrapper
