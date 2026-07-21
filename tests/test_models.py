"""Tests for the model wrappers on random tensors (no dataset needed).

Requires the pretrained checkpoints cached by
``scripts/download.py``.
"""

import gc
import os
from pathlib import Path

import pytest
import torch

from selectseg.data import IGNORE_INDEX, SPECS
from selectseg.models import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("SELECTSEG_RUN_MODEL_TESTS") != "1"
    or not (REPO_ROOT / "data" / "cache" / "huggingface").exists()
    or not (REPO_ROOT / "data" / "cache" / "torch").exists(),
    reason="set SELECTSEG_RUN_MODEL_TESTS=1 after caching models",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def release_model_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    ("name", "dataset"),
    [
        ("clipseg", "pet"),
        ("clipseg", "kvasir"),
        ("clipseg", "fives"),
        ("deeplabv3", "pet"),
    ],
)
def test_zero_shot_predict_probs(name, dataset):
    spec = SPECS[dataset]
    model = build_model(name, spec, finetuned=False).to(DEVICE).eval()
    size = model.image_size
    images = torch.rand(2, 3, size, size, device=DEVICE)
    with torch.inference_mode():
        probs = model.predict_probs(images)
    assert probs.shape == (2, spec.num_classes, size, size)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


@pytest.mark.parametrize("name", ["clipseg", "deeplabv3"])
def test_finetuned_loss_grads_and_checkpoint_roundtrip(name):
    dataset = "pet"
    spec = SPECS[dataset]
    torch.manual_seed(0)
    model = build_model(name, spec, finetuned=True).to(DEVICE).train()
    size = model.image_size
    images = torch.rand(2, 3, size, size, device=DEVICE)
    masks = torch.randint(0, spec.num_classes, (2, size, size), device=DEVICE)
    masks[0, :16] = IGNORE_INDEX
    prompts = torch.randint(0, len(spec.prompts), (2,), device=DEVICE)
    loss = model.compute_loss(images, masks, prompts)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.trainable_parameters()
    )
    # strict load into a freshly built architecture catches key mismatches
    fresh = build_model(name, spec, finetuned=True, init_weights=False)
    fresh.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})


def test_clipseg_finetunes_decoder_only():
    model = build_model("clipseg", SPECS["pet"], finetuned=True)
    assert all(not p.requires_grad for p in model.model.clip.parameters())
    assert all(p.requires_grad for p in model.model.decoder.parameters())


def test_clipseg_fully_ignored_mask_gives_zero_loss():
    model = build_model("clipseg", SPECS["pet"], finetuned=True).to(DEVICE)
    size = model.image_size
    images = torch.rand(1, 3, size, size, device=DEVICE)
    masks = torch.full((1, size, size), IGNORE_INDEX, dtype=torch.long, device=DEVICE)
    prompts = torch.zeros(1, dtype=torch.long, device=DEVICE)
    assert model.compute_loss(images, masks, prompts).item() == 0.0


def test_deeplab_external_class_maps():
    pet_model = build_model("deeplabv3", SPECS["pet"], finetuned=False)
    assert pet_model.class_map == [[8, 12]]  # cat, dog in the checkpoint vocabulary
