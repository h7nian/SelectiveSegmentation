"""CLI-level tests for decision-neutral frozen binary-map inference."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from selectseg.binary_artifacts import load_binary_artifact, sha256_file
from selectseg import freeze_binary_maps as freeze


class _FakeDataset:
    def __init__(self, spec, root, train, image_size):
        del spec, root
        assert train is False and image_size == 4
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


def test_slurm_wrapper_is_one_condition_and_forwards_extra_arguments():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "slurm"
        / "freeze_binary_maps.sbatch"
    ).read_text()
    assert "MODEL=$1" in wrapper
    assert "DATASET=$2" in wrapper
    assert "CHECKPOINT=$3" in wrapper
    assert "python -m selectseg.freeze_binary_maps" in wrapper
    assert "--require-cuda" in wrapper
    assert '"${ARGS[@]}" "$@"' in wrapper
