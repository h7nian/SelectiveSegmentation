"""Datasets and transforms shared by all model conditions.

Every dataset yields images as float tensors in [0, 1] and masks as long
tensors of dataset class indices, with :data:`IGNORE_INDEX` marking pixels
excluded from losses and metrics.

Sample layout
    train: ``(image (3, S, S), mask (S, S), prompt_index)``
    eval:  ``(image (3, S, S), mask (H, W))`` with the mask at its original
           resolution, so metrics are computed at native resolution.

``prompt_index`` indexes :attr:`DatasetSpec.prompts` and is only consumed by
CLIPSeg fine-tuning; DeepLabV3 training ignores it.
"""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode, RandomCrop

IGNORE_INDEX = 255

# Random-scale range for training augmentation, relative to the crop size.
MIN_TRAIN_SCALE = 0.5
MAX_TRAIN_SCALE = 2.0

# Probability that CLIPSeg fine-tuning samples a prompt whose class is absent
# from the image (target all background), teaching the model to reject
# prompts that do not apply. Only used when the base dataset does not fix the
# prompt itself.
NEGATIVE_PROMPT_PROB = 0.25


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a dataset's class and prompt vocabulary.

    ``prompts[i]`` is a CLIPSeg text prompt referring to dataset class
    ``prompt_classes[i]``; several prompts may map to the same class (Pet
    maps both "cat" and "dog" onto the single foreground class "pet").
    Class index 0 is always background.
    """

    name: str
    class_names: tuple
    prompts: tuple
    prompt_classes: tuple
    train_split: str
    eval_split: str

    @property
    def num_classes(self):
        return len(self.class_names)


VOC_CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Natural-language variants of the VOC class names, used as CLIPSeg prompts.
VOC_PROMPTS = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
)

SPECS = {
    "pet": DatasetSpec(
        name="pet",
        class_names=("background", "pet"),
        prompts=("cat", "dog"),
        prompt_classes=(1, 1),
        train_split="trainval",
        eval_split="test",
    ),
    "kvasir": DatasetSpec(
        name="kvasir",
        class_names=("background", "polyp"),
        prompts=("polyp",),
        prompt_classes=(1,),
        train_split="train",
        eval_split="test",
    ),
    "fives": DatasetSpec(
        name="fives",
        class_names=("background", "retinal blood vessels"),
        prompts=("retinal blood vessels",),
        prompt_classes=(1,),
        train_split="train",
        eval_split="test",
    ),
    "isic": DatasetSpec(
        name="isic",
        class_names=("background", "skin lesion"),
        prompts=("skin lesion",),
        prompt_classes=(1,),
        train_split="train",
        eval_split="test",
    ),
    "tn3k": DatasetSpec(
        name="tn3k",
        class_names=("background", "thyroid nodule"),
        prompts=("thyroid nodule",),
        prompt_classes=(1,),
        train_split="train",
        eval_split="test",
    ),
    "voc": DatasetSpec(
        name="voc",
        class_names=VOC_CLASS_NAMES,
        prompts=VOC_PROMPTS,
        prompt_classes=tuple(range(1, len(VOC_CLASS_NAMES))),
        train_split="train",
        eval_split="val",
    ),
}


def _open_verified(path, *, sample_id, role, verified_file_reader):
    """Open a source image, optionally from content-verified in-memory bytes."""

    if verified_file_reader is None:
        return Image.open(path)
    payload = verified_file_reader(sample_id, role, Path(path))
    if not isinstance(payload, bytes):
        raise TypeError("verified_file_reader must return immutable bytes")
    return Image.open(io.BytesIO(payload))


class PetSegmentation(Dataset):
    """Oxford-IIIT Pet as binary segmentation (background/pet).

    Trimap values are 1 = pet, 2 = background, 3 = border.  The focused
    binary benchmark uses a predeclared border-inclusive target: values 1 and
    3 are foreground and value 2 is background.  This gives every image a
    total binary label domain, so the loss used by confidence is exactly the
    loss used by risk; no annotation-derived void mask enters confidence.
    The species column fixes the CLIPSeg prompt: 1 = cat, 2 = dog.
    """

    SPECIES_TO_PROMPT = {1: 0, 2: 1}

    def __init__(self, root, split, *, verified_file_reader=None):
        base = Path(root) / "oxford-iiit-pet"
        split_file = base / "annotations" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"{split_file} not found; run scripts/download_assets.py first"
            )
        self.images_dir = base / "images"
        self.trimaps_dir = base / "annotations" / "trimaps"
        self.verified_file_reader = verified_file_reader
        self.samples = []
        for line in split_file.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            image_id, _, species, _ = line.split()
            self.samples.append((image_id, int(species)))

    def __len__(self):
        return len(self.samples)

    def sample_id(self, index):
        """Stable dataset identifier used to join per-image experiment rows."""
        return self.samples[index][0]

    def __getitem__(self, index):
        image_id, species = self.samples[index]
        with _open_verified(
            self.images_dir / f"{image_id}.jpg",
            sample_id=image_id,
            role="image",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            image = source.convert("RGB")
        with _open_verified(
            self.trimaps_dir / f"{image_id}.png",
            sample_id=image_id,
            role="mask",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            trimap = TF.pil_to_tensor(source).long()
        mask = torch.zeros_like(trimap)
        mask[(trimap == 1) | (trimap == 3)] = 1
        mask[trimap == 2] = 0
        return image, mask, self.SPECIES_TO_PROMPT[species]


class VOCSegmentationBase(Dataset):
    """PASCAL VOC 2012 with masks as class-index tensors.

    The prompt index is left unspecified (-1); :class:`SegDataset` samples
    one per training image based on which classes are visible.
    """

    def __init__(self, root, split, *, verified_file_reader=None):
        self.base = VOCSegmentation(root, year="2012", image_set=split)
        self.verified_file_reader = verified_file_reader

    def __len__(self):
        return len(self.base)

    def sample_id(self, index):
        """Stable VOC image identifier (the JPEG stem)."""
        return Path(self.base.images[index]).stem

    def __getitem__(self, index):
        if self.verified_file_reader is None:
            image, target = self.base[index]
        else:
            sample_id = self.sample_id(index)
            with _open_verified(
                self.base.images[index],
                sample_id=sample_id,
                role="image",
                verified_file_reader=self.verified_file_reader,
            ) as source:
                image = source.convert("RGB")
            with _open_verified(
                self.base.masks[index],
                sample_id=sample_id,
                role="mask",
                verified_file_reader=self.verified_file_reader,
            ) as source:
                target = source.copy()
        mask = TF.pil_to_tensor(target).long()
        return image.convert("RGB"), mask, -1


class KvasirSegmentation(Dataset):
    """Kvasir-SEG as a deterministic binary polyp dataset.

    The official release has no train/test split. We pair files by stem, rank
    all stems by ``(SHA256(stem), stem)``, and assign the first
    ``floor(0.8 * N)`` ranked pairs to ``train`` and the remainder to ``test``.
    Membership therefore depends only on the complete set of stems, not on
    filesystem traversal order or process randomness, and the two splits are
    disjoint. Samples within each split are ordered by stem for stable indexing.

    Official masks are RGB JPEGs whose nominal values are 0 and 255. JPEG
    compression introduces small values near both modes, so foreground is
    defined by a fixed channel-maximum threshold of 128, not by nonzero tests.
    """

    _SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
    _SPLITS = frozenset({"train", "test"})

    def __init__(self, root, split, *, verified_file_reader=None):
        if split not in self._SPLITS:
            raise ValueError(
                f"Kvasir-SEG split must be one of {sorted(self._SPLITS)}, got {split!r}"
            )

        base = Path(root) / "Kvasir-SEG"
        images = self._files_by_stem(base / "images", "image")
        masks = self._files_by_stem(base / "masks", "mask")
        image_stems = set(images)
        mask_stems = set(masks)
        if image_stems != mask_stems:
            missing_masks = sorted(image_stems - mask_stems)
            missing_images = sorted(mask_stems - image_stems)
            raise ValueError(
                "Kvasir-SEG image/mask stems are not paired: "
                f"missing masks={missing_masks[:5]}, "
                f"missing images={missing_images[:5]}"
            )

        pairs = [(stem, images[stem], masks[stem]) for stem in image_stems]
        for stem, image_path, mask_path in pairs:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(
                        "Kvasir-SEG image/mask size mismatch for "
                        f"{stem!r}: image={image.size}, mask={mask.size}"
                    )

        pairs.sort(
            key=lambda sample: (
                hashlib.sha256(sample[0].encode("utf-8")).digest(),
                sample[0],
            )
        )
        train_count = 4 * len(pairs) // 5
        selected = pairs[:train_count] if split == "train" else pairs[train_count:]
        self.samples = sorted(selected, key=lambda sample: sample[0])
        self.verified_file_reader = verified_file_reader

    @classmethod
    def _files_by_stem(cls, directory, kind):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} not found; download Kvasir-SEG into "
                "data/Kvasir-SEG/{images,masks}"
            )
        files = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in cls._SUFFIXES:
                continue
            if path.stem in files:
                raise ValueError(
                    f"Kvasir-SEG has duplicate {kind} stem {path.stem!r}: "
                    f"{files[path.stem].name!r} and {path.name!r}"
                )
            files[path.stem] = path
        return files

    def __len__(self):
        return len(self.samples)

    def sample_id(self, index):
        """Stable Kvasir identifier (the paired image/mask stem)."""
        return self.samples[index][0]

    def __getitem__(self, index):
        _, image_path, mask_path = self.samples[index]
        with _open_verified(
            image_path,
            sample_id=self.samples[index][0],
            role="image",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            image = source.convert("RGB")
        with _open_verified(
            mask_path,
            sample_id=self.samples[index][0],
            role="mask",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            mask_channels = TF.pil_to_tensor(source)
        mask = (mask_channels.amax(dim=0, keepdim=True) >= 128).long()
        return image, mask, 0


class FivesSegmentation(Dataset):
    """FIVES retinal-vessel segmentation with its official train/test split.

    The extracted release is normalized to ``data/FIVES`` and contains
    ``{train,test}/{Original,Ground truth}``. Image and mask files are paired
    strictly by stem. Any nonzero mask value denotes a vessel pixel.
    """

    _SPLITS = frozenset({"train", "test"})
    _SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

    def __init__(self, root, split, *, verified_file_reader=None):
        if split not in self._SPLITS:
            raise ValueError(
                f"FIVES split must be one of {sorted(self._SPLITS)}, got {split!r}"
            )

        split_dir = Path(root) / "FIVES" / split
        images = self._files_by_stem(split_dir / "Original", "image")
        masks = self._files_by_stem(split_dir / "Ground truth", "mask")
        image_stems = set(images)
        mask_stems = set(masks)
        if image_stems != mask_stems:
            missing_masks = sorted(image_stems - mask_stems)
            missing_images = sorted(mask_stems - image_stems)
            raise ValueError(
                "FIVES image/mask stems are not paired: "
                f"missing masks={missing_masks[:5]}, "
                f"missing images={missing_images[:5]}"
            )

        self.samples = []
        for stem in sorted(image_stems):
            image_path, mask_path = images[stem], masks[stem]
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(
                        "FIVES image/mask size mismatch for "
                        f"{stem!r}: image={image.size}, mask={mask.size}"
                    )
            self.samples.append((stem, image_path, mask_path))
        self.verified_file_reader = verified_file_reader

    @classmethod
    def _files_by_stem(cls, directory, kind):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} not found; extract FIVES into "
                "data/FIVES/{train,test}/{Original,Ground truth}"
            )
        files = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in cls._SUFFIXES:
                continue
            if path.stem in files:
                raise ValueError(
                    f"FIVES has duplicate {kind} stem {path.stem!r}: "
                    f"{files[path.stem].name!r} and {path.name!r}"
                )
            files[path.stem] = path
        return files

    def __len__(self):
        return len(self.samples)

    def sample_id(self, index):
        """Stable FIVES identifier from the official filename stem."""
        return self.samples[index][0]

    def __getitem__(self, index):
        _, image_path, mask_path = self.samples[index]
        with _open_verified(
            image_path,
            sample_id=self.samples[index][0],
            role="image",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            image = source.convert("RGB")
        with _open_verified(
            mask_path,
            sample_id=self.samples[index][0],
            role="mask",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            mask_channels = TF.pil_to_tensor(source)
        mask = (mask_channels != 0).any(dim=0, keepdim=True).long()
        return image, mask, 0


class _StrictPairedBinarySegmentation(Dataset):
    """Shared loader for official train/test releases paired strictly by stem."""

    _SPLITS = frozenset({"train", "test"})
    _SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
    _DATASET_NAME = "binary dataset"
    _DOWNLOAD_HINT = "download the dataset first"
    _MASK_STEM_SUFFIX = ""

    def __init__(self, root, split, *, verified_file_reader=None):
        if split not in self._SPLITS:
            raise ValueError(
                f"{self._DATASET_NAME} split must be one of "
                f"{sorted(self._SPLITS)}, got {split!r}"
            )

        images_dir, masks_dir = self._split_directories(Path(root), split)
        images = self._files_by_stem(images_dir, "image")
        masks = self._files_by_stem(masks_dir, "mask", mask=True)
        image_stems = set(images)
        mask_stems = set(masks)
        if image_stems != mask_stems:
            missing_masks = sorted(image_stems - mask_stems)
            missing_images = sorted(mask_stems - image_stems)
            raise ValueError(
                f"{self._DATASET_NAME} image/mask stems are not paired: "
                f"missing masks={missing_masks[:5]}, "
                f"missing images={missing_images[:5]}"
            )

        self.samples = []
        for stem in sorted(image_stems):
            image_path, mask_path = images[stem], masks[stem]
            with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
                if image.size != mask_image.size:
                    raise ValueError(
                        f"{self._DATASET_NAME} image/mask size mismatch for "
                        f"{stem!r}: image={image.size}, mask={mask_image.size}"
                    )
            self.samples.append((stem, image_path, mask_path))
        self.verified_file_reader = verified_file_reader

    def _split_directories(self, root, split):
        raise NotImplementedError

    def _files_by_stem(self, directory, kind, *, mask=False):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} not found; {self._DOWNLOAD_HINT}"
            )

        files = {}
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in self._SUFFIXES:
                continue
            stem = path.stem
            if mask and self._MASK_STEM_SUFFIX:
                stem = stem.removesuffix(self._MASK_STEM_SUFFIX)
            if stem in files:
                raise ValueError(
                    f"{self._DATASET_NAME} has duplicate {kind} stem {stem!r}: "
                    f"{files[stem].name!r} and {path.name!r}"
                )
            files[stem] = path
        return files

    def __len__(self):
        return len(self.samples)

    def sample_id(self, index):
        """Return the stable paired image/mask stem."""
        return self.samples[index][0]

    def __getitem__(self, index):
        _, image_path, mask_path = self.samples[index]
        with _open_verified(
            image_path,
            sample_id=self.samples[index][0],
            role="image",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            image = source.convert("RGB")
        with _open_verified(
            mask_path,
            sample_id=self.samples[index][0],
            role="mask",
            verified_file_reader=self.verified_file_reader,
        ) as source:
            mask_channels = TF.pil_to_tensor(source)
        mask = (mask_channels.amax(dim=0, keepdim=True) >= 128).long()
        return image, mask, 0


class ISICSegmentation(_StrictPairedBinarySegmentation):
    """ISIC 2018 Task 1 lesion masks with the official train/test split."""

    _DATASET_NAME = "ISIC"
    _DOWNLOAD_HINT = "run scripts/download_binary_assets.py --datasets isic"
    _MASK_STEM_SUFFIX = "_segmentation"

    def _split_directories(self, root, split):
        release_split = "Training" if split == "train" else "Test"
        base = root / "ISIC2018"
        return (
            base / f"ISIC2018_Task1-2_{release_split}_Input",
            base / f"ISIC2018_Task1_{release_split}_GroundTruth",
        )


class TN3KSegmentation(_StrictPairedBinarySegmentation):
    """TN3K thyroid-nodule masks with the official trainval/test split."""

    _DATASET_NAME = "TN3K"
    _DOWNLOAD_HINT = "run scripts/download_binary_assets.py --datasets tn3k"

    def _split_directories(self, root, split):
        release_split = "trainval" if split == "train" else "test"
        extracted = root / "TN3K" / "extracted" / "Thyroid Dataset" / "tn3k"
        normalized = root / "TN3K" / "tn3k"
        base = extracted if extracted.is_dir() else normalized
        return base / f"{release_split}-image", base / f"{release_split}-mask"


def _base_dataset(spec, root, split, *, verified_file_reader=None):
    if spec.name == "pet":
        return PetSegmentation(
            root, split, verified_file_reader=verified_file_reader
        )
    if spec.name == "kvasir":
        return KvasirSegmentation(
            root, split, verified_file_reader=verified_file_reader
        )
    if spec.name == "fives":
        return FivesSegmentation(
            root, split, verified_file_reader=verified_file_reader
        )
    if spec.name == "isic":
        return ISICSegmentation(
            root, split, verified_file_reader=verified_file_reader
        )
    if spec.name == "tn3k":
        return TN3KSegmentation(
            root, split, verified_file_reader=verified_file_reader
        )
    if spec.name == "voc":
        return VOCSegmentationBase(
            root, split, verified_file_reader=verified_file_reader
        )
    raise ValueError(f"Unsupported dataset spec {spec.name!r}")


class SegDataset(Dataset):
    """Applies model-resolution transforms on top of a base dataset."""

    def __init__(
        self, spec, root, train, image_size, *, verified_file_reader=None
    ):
        self.spec = spec
        self.train = train
        self.image_size = image_size
        split = spec.train_split if train else spec.eval_split
        self.base = _base_dataset(
            spec,
            root,
            split,
            verified_file_reader=verified_file_reader,
        )

    def __len__(self):
        return len(self.base)

    def sample_id(self, index):
        """Forward the base dataset's stable identifier."""
        return self.base.sample_id(index)

    def __getitem__(self, index):
        image, mask, prompt_index = self.base[index]
        image = TF.pil_to_tensor(image)
        if not self.train:
            image = TF.resize(
                image, [self.image_size, self.image_size], antialias=True
            )
            return image.float() / 255, mask.squeeze(0)
        image, mask = _random_scale_crop_flip(image, mask, self.image_size)
        if prompt_index < 0:
            prompt_index = self._sample_prompt(mask)
        return image.float() / 255, mask, prompt_index

    def _sample_prompt(self, mask):
        """Pick a prompt, biased toward classes visible in the crop."""
        present = set(torch.unique(mask).tolist())
        positive = [
            i for i, c in enumerate(self.spec.prompt_classes) if c in present
        ]
        negative = [
            i for i in range(len(self.spec.prompts)) if i not in positive
        ]
        use_negative = negative and (
            not positive or torch.rand(()) < NEGATIVE_PROMPT_PROB
        )
        pool = negative if use_negative else positive
        return pool[int(torch.randint(len(pool), ()))]


def _random_scale_crop_flip(image, mask, size):
    """Jointly scale, pad, crop, and flip an image and its (1, H, W) mask."""
    scale = float(torch.empty(()).uniform_(MIN_TRAIN_SCALE, MAX_TRAIN_SCALE))
    short_side = max(1, int(size * scale))
    image = TF.resize(image, short_side, antialias=True)
    mask = TF.resize(mask, short_side, interpolation=InterpolationMode.NEAREST)
    pad_right = max(0, size - image.shape[-1])
    pad_bottom = max(0, size - image.shape[-2])
    if pad_right or pad_bottom:
        image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
        mask = TF.pad(mask, [0, 0, pad_right, pad_bottom], fill=IGNORE_INDEX)
    top, left, height, width = RandomCrop.get_params(image, (size, size))
    image = TF.crop(image, top, left, height, width)
    mask = TF.crop(mask, top, left, height, width)
    if torch.rand(()) < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    return image, mask.squeeze(0)


def eval_collate(batch):
    """Stack images; keep variable-size original masks as a list."""
    images = torch.stack([sample[0] for sample in batch])
    masks = [sample[1] for sample in batch]
    return images, masks
