"""Offline checks for immutable binary-asset acquisition."""

import sys
from types import ModuleType

from scripts import download_binary_assets as assets


def test_model_download_pins_both_clipseg_components(monkeypatch, tmp_path):
    calls = []

    class FakeCLIPSegModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))

    class FakeCLIPSegProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))

    transformers = ModuleType("transformers")
    transformers.CLIPSegForImageSegmentation = FakeCLIPSegModel
    transformers.CLIPSegProcessor = FakeCLIPSegProcessor

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
    assert calls[2] == ("deeplabv3", "locked-deeplabv3-weights", {})
    assert assets.CLIPSEG_REVISION == expected["revision"]
