"""Offline checks for immutable binary-asset acquisition."""

import os
import sys
from types import ModuleType

from scripts import download as assets


def test_model_download_pins_hugging_face_components(monkeypatch, tmp_path):
    calls = []

    # ``download_models`` intentionally redirects the process-wide caches to
    # DATA_ROOT. Register their current values with monkeypatch first so the
    # fixture restores them even though the production function assigns via
    # ``os.environ`` directly. Otherwise this test leaks its temporary cache
    # into every later model/pipeline test in the same pytest process.
    monkeypatch.setenv("HF_HOME", os.environ["HF_HOME"])
    monkeypatch.setenv("TORCH_HOME", os.environ["TORCH_HOME"])

    class FakeCLIPSegModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))

    class FakeCLIPSegProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))

    class FakeSegFormerModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("segformer-model", model_id, kwargs))

    class FakeSegFormerProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("segformer-processor", model_id, kwargs))

    transformers = ModuleType("transformers")
    transformers.CLIPSegForImageSegmentation = FakeCLIPSegModel
    transformers.CLIPSegProcessor = FakeCLIPSegProcessor
    transformers.SegformerForSemanticSegmentation = FakeSegFormerModel
    transformers.SegformerImageProcessor = FakeSegFormerProcessor

    segmentation = ModuleType("torchvision.models.segmentation")

    class FakeWeights:
        COCO_WITH_VOC_LABELS_V1 = "locked-deeplabv3-weights"

    def fake_deeplabv3_resnet50(*, weights):
        calls.append(("deeplabv3", weights, {}))

    segmentation.DeepLabV3_ResNet50_Weights = FakeWeights
    segmentation.deeplabv3_resnet50 = fake_deeplabv3_resnet50
    models = ModuleType("torchvision.models")
    models.segmentation = segmentation
    torchvision = ModuleType("torchvision")
    torchvision.models = models

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.models.segmentation", segmentation)
    monkeypatch.setattr(assets, "DATA_ROOT", tmp_path / "data")

    assets.download_models()

    expected = {"revision": "999e0328d9e10b484360c477313983f9afdd7050"}
    assert calls[:2] == [
        ("model", "CIDAS/clipseg-rd64-refined", expected),
        ("processor", "CIDAS/clipseg-rd64-refined", expected),
    ]
    segformer = {"revision": "de01bae28967510f9ddd496c60a969357195400c"}
    assert calls[2:4] == [
        (
            "segformer-model",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            segformer,
        ),
        (
            "segformer-processor",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            segformer,
        ),
    ]
    assert calls[4] == ("deeplabv3", "locked-deeplabv3-weights", {})
    assert assets.CLIPSEG_REVISION == expected["revision"]
    assert assets.SEGFORMER_REVISION == segformer["revision"]
