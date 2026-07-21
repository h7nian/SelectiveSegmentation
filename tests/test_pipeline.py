"""End-to-end smoke tests driving the real CLIs in subprocesses.

Each case fine-tunes for two tiny batches, then evaluates the saved
checkpoint on a handful of images, exactly as the SLURM jobs do.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

DATASET_MARKERS = {
    "pet": DATA_ROOT / "oxford-iiit-pet" / "annotations" / "trainval.txt",
    "voc": DATA_ROOT / "VOCdevkit" / "VOC2012" / "ImageSets" / "Segmentation" / "val.txt",
}


def run_cli(module, *args):
    subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        check=True,
        timeout=900,
    )


def skip_unless_downloaded(dataset):
    if not DATASET_MARKERS[dataset].exists():
        pytest.skip(f"{dataset} not downloaded; run scripts/download_assets.py")


@pytest.mark.parametrize(
    "name,dataset",
    [
        ("clipseg", "pet"),
        ("clipseg", "voc"),
        ("deeplabv3", "pet"),
        ("deeplabv3", "voc"),
    ],
)
def test_train_then_evaluate(tmp_path, name, dataset):
    skip_unless_downloaded(dataset)
    train_dir = tmp_path / "train"
    run_cli(
        "selectseg.train",
        "--model", name,
        "--dataset", dataset,
        "--output-dir", str(train_dir),
        "--epochs", "1",
        "--limit-batches", "2",
        "--batch-size", "2",
        # Multiprocess data loading can deadlock in restricted CI/sandbox
        # environments. Slurm exercises the production worker configuration;
        # this smoke test only needs the real train/evaluate code path.
        "--num-workers", "0",
    )
    checkpoint = train_dir / "checkpoint.pt"
    assert checkpoint.exists()

    eval_dir = tmp_path / "eval"
    run_cli(
        "selectseg.evaluate",
        "--model", name,
        "--dataset", dataset,
        "--checkpoint", str(checkpoint),
        "--output-dir", str(eval_dir),
        "--limit", "8",
        "--batch-size", "4",
        "--num-workers", "0",
    )
    result = json.loads((eval_dir / f"{name}-target_{dataset}.json").read_text())
    assert result["condition"] == f"{name}-target"
    assert result["num_images"] == 8
    assert 0.0 <= result["metrics"]["mean_iou"] <= 1.0
    assert 0.0 <= result["metrics"]["pixel_accuracy"] <= 1.0


def test_zero_shot_evaluate(tmp_path):
    skip_unless_downloaded("pet")
    run_cli(
        "selectseg.evaluate",
        "--model", "deeplabv3",
        "--dataset", "pet",
        "--output-dir", str(tmp_path),
        "--limit", "8",
        "--batch-size", "4",
        "--num-workers", "0",
    )
    result = json.loads((tmp_path / "deeplabv3-external_pet.json").read_text())
    assert result["condition"] == "deeplabv3-external"
    assert result["checkpoint"] is None
    assert "mean_hd95" in result["metrics"]
    assert "per_class_hd95" in result["metrics"]
    assert "fw_iou" in result["metrics"]
    assert "mean_class_accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["per_image_mean_iou"] <= 1.0
    # The COCO checkpoint knows cats and dogs, so zero-shot on Pet should
    # beat chance by a clear margin even on eight images.
    assert result["metrics"]["mean_iou"] > 0.3
