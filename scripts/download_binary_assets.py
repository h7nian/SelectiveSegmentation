"""Download the focused binary benchmark assets with integrity checks.

Examples::

    python scripts/download_binary_assets.py --datasets pet kvasir fives
    python scripts/download_binary_assets.py --models-only

FIVES is distributed as a RAR archive, so extraction requires an ``unrar``
executable on ``PATH``. Archives are retained under ``data/`` for provenance.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_SHA256 = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
FIVES_URL = "https://ndownloader.figshare.com/files/34969398"
FIVES_SHA256 = "be72f9af286b107bcebcc08a9dae7fc55c3fb0959409b689e14c72f9fdc4ad8e"
FIVES_RELEASE_DIR = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        print(f"[cached] {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"[download] {url} -> {destination}")
    urllib.request.urlretrieve(url, partial)
    actual = _sha256(partial)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {partial}: expected {expected_sha256}, got {actual}"
        )
    partial.replace(destination)


def _safe_extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe path in {archive}: {member.filename!r}")
        source.extractall(destination)


def download_pet():
    from torchvision.datasets import OxfordIIITPet

    OxfordIIITPet(DATA_ROOT, split="trainval", download=True)
    OxfordIIITPet(DATA_ROOT, split="test", download=True)


def download_kvasir():
    if (DATA_ROOT / "Kvasir-SEG" / "images").is_dir():
        print("[cached] data/Kvasir-SEG")
        return
    archive = DATA_ROOT / "kvasir-seg.zip"
    _download(KVASIR_URL, archive, KVASIR_SHA256)
    _safe_extract_zip(archive, DATA_ROOT)


def download_fives():
    target = DATA_ROOT / "FIVES"
    if (target / "train" / "Original").is_dir():
        print("[cached] data/FIVES")
        return
    archive = DATA_ROOT / "fives.rar"
    _download(FIVES_URL, archive, FIVES_SHA256)
    unrar = shutil.which("unrar")
    if unrar is None:
        raise RuntimeError("FIVES extraction requires an 'unrar' executable on PATH")
    subprocess.run([unrar, "x", "-o+", str(archive), str(DATA_ROOT)], check=True)
    extracted = DATA_ROOT / FIVES_RELEASE_DIR
    if not extracted.is_dir():
        raise RuntimeError(f"FIVES archive did not create expected directory {extracted}")
    extracted.replace(target)


def download_models():
    # Set cache locations before importing either model library.
    os.environ["HF_HOME"] = str(DATA_ROOT / "cache" / "huggingface")
    os.environ["TORCH_HOME"] = str(DATA_ROOT / "cache" / "torch")
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    checkpoint = "CIDAS/clipseg-rd64-refined"
    CLIPSegForImageSegmentation.from_pretrained(checkpoint)
    CLIPSegProcessor.from_pretrained(checkpoint)
    deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("pet", "kvasir", "fives"),
        default=("pet", "kvasir", "fives"),
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="do not populate model caches"
    )
    parser.add_argument(
        "--models-only", action="store_true", help="skip all dataset downloads"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_ROOT.mkdir(exist_ok=True)
    if not args.models_only:
        downloaders = {
            "pet": download_pet,
            "kvasir": download_kvasir,
            "fives": download_fives,
        }
        for name in args.datasets:
            print(f"[{name}]")
            downloaders[name]()
    if not args.skip_models:
        print("[models]")
        download_models()
    print("Focused binary assets are ready.")


if __name__ == "__main__":
    main()
