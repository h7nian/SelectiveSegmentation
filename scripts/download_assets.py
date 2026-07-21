"""Download datasets and pretrained checkpoints (run once, on a login node).

Compute nodes may lack reliable outbound internet, so all remote assets are
fetched up front: Oxford-IIIT Pet, PASCAL VOC 2012, the CLIPSeg checkpoint,
and the DeepLabV3-ResNet50 COCO checkpoint.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Force the repo-local caches (overriding any personal HF_HOME/TORCH_HOME)
# so downloads land exactly where scripts/slurm/env.sh points jobs. Must be
# set before torch/transformers read them at import time.
os.environ["HF_HOME"] = str(REPO_ROOT / "data" / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(REPO_ROOT / "data" / "cache" / "torch")

from torchvision.datasets import OxfordIIITPet, VOCSegmentation  # noqa: E402
from torchvision.models.segmentation import (  # noqa: E402
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from selectseg.models import CLIPSEG_CHECKPOINT, CLIPSEG_REVISION  # noqa: E402

VOC_MIRROR_URL = "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"


def download_pet(data_root):
    print("[pet] downloading Oxford-IIIT Pet ...")
    OxfordIIITPet(data_root, split="trainval", download=True)
    print("[pet] done")


def download_voc(data_root):
    print("[voc] downloading PASCAL VOC 2012 ...")
    try:
        VOCSegmentation(data_root, year="2012", image_set="train", download=True)
    except Exception as error:  # host.robots.ox.ac.uk is frequently down
        print(f"[voc] primary host failed ({error}); trying mirror ...")
        tar_path = Path(data_root) / "VOCtrainval_11-May-2012.tar"
        urllib.request.urlretrieve(VOC_MIRROR_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_root)
        tar_path.unlink()
        VOCSegmentation(data_root, year="2012", image_set="train")
    print("[voc] done")


def download_checkpoints():
    print("[clipseg] downloading CLIPSeg checkpoint ...")
    CLIPSegForImageSegmentation.from_pretrained(
        CLIPSEG_CHECKPOINT, revision=CLIPSEG_REVISION
    )
    CLIPSegProcessor.from_pretrained(
        CLIPSEG_CHECKPOINT, revision=CLIPSEG_REVISION
    )
    print("[deeplabv3] downloading DeepLabV3-ResNet50 COCO checkpoint ...")
    deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    print("[checkpoints] done")


def main():
    data_root = REPO_ROOT / "data"
    data_root.mkdir(exist_ok=True)
    download_pet(data_root)
    download_voc(data_root)
    download_checkpoints()
    print("All assets ready.")


if __name__ == "__main__":
    main()
