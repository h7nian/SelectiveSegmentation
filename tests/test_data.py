"""Tests for the focused binary dataset wrappers and transforms.

Synthetic-input tests always run; tests touching downloaded releases skip
until ``scripts/download.py`` has been run.
"""

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from selectseg.data import (
    DutsSegmentation,
    FivesSegmentation,
    IGNORE_INDEX,
    ISICSegmentation,
    KvasirSegmentation,
    PetSegmentation,
    SPECS,
    SegDataset,
    TN3KSegmentation,
    _random_scale_crop_flip,
    eval_collate,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

requires_pet = pytest.mark.skipif(
    not (DATA_ROOT / "oxford-iiit-pet" / "annotations" / "trainval.txt").exists(),
    reason="Oxford-IIIT Pet not downloaded; run scripts/download.py",
)

requires_isic = pytest.mark.skipif(
    not (
        DATA_ROOT / "ISIC2018" / "ISIC2018_Task1-2_Training_Input"
    ).is_dir(),
    reason="ISIC 2018 not downloaded; run scripts/download.py",
)

requires_tn3k = pytest.mark.skipif(
    not (
        DATA_ROOT
        / "TN3K"
        / "extracted"
        / "Thyroid Dataset"
        / "tn3k"
        / "trainval-image"
    ).is_dir(),
    reason="TN3K not downloaded; run scripts/download.py",
)

requires_duts = pytest.mark.skipif(
    not (DATA_ROOT / "DUTS" / "DUTS-TR" / "DUTS-TR-Image").is_dir(),
    reason="DUTS not downloaded; run scripts/download.py",
)


def _write_kvasir_sample(root, stem, image_size=(9, 7), mask_size=None):
    """Create one paired RGB image and binary RGB mask under a fake release."""
    images = root / "Kvasir-SEG" / "images"
    masks = root / "Kvasir-SEG" / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 40, 60)).save(images / f"{stem}.jpg")
    mask = Image.new("RGB", mask_size or image_size, color=(0, 0, 0))
    mask.putpixel((1, 1), (0, 255, 0))
    mask.putpixel((2, 1), (0, 0, 1))
    mask.putpixel((3, 1), (128, 0, 0))
    mask.save(masks / f"{stem}.png")


def _write_fives_sample(root, split, stem, image_size=(9, 7), mask_size=None):
    """Create one paired image and binary mask under a fake FIVES release."""
    images = root / "FIVES" / split / "Original"
    masks = root / "FIVES" / split / "Ground truth"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 40, 60)).save(images / f"{stem}.png")
    mask = Image.new("L", mask_size or image_size, color=0)
    mask.putpixel((1, 1), 255)
    mask.save(masks / f"{stem}.png")


def _write_isic_sample(root, split, stem, image_size=(9, 7), mask_size=None):
    label = "Training" if split == "train" else "Test"
    images = root / "ISIC2018" / f"ISIC2018_Task1-2_{label}_Input"
    masks = root / "ISIC2018" / f"ISIC2018_Task1_{label}_GroundTruth"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 40, 60)).save(images / f"{stem}.jpg")
    mask = Image.new("L", mask_size or image_size, color=0)
    mask.putpixel((1, 1), 255)
    mask.putpixel((2, 1), 1)
    mask.putpixel((3, 1), 128)
    mask.save(masks / f"{stem}_segmentation.png")


def _write_tn3k_sample(
    root, split, stem, image_size=(9, 7), mask_size=None, *, extracted=False
):
    label = "trainval" if split == "train" else "test"
    base = root / "TN3K"
    if extracted:
        base = base / "extracted" / "Thyroid Dataset"
    base = base / "tn3k"
    images = base / f"{label}-image"
    masks = base / f"{label}-mask"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 40, 60)).save(images / f"{stem}.jpg")
    mask = Image.new("RGB", mask_size or image_size, color=(0, 0, 0))
    mask.putpixel((1, 1), (0, 255, 0))
    mask.putpixel((2, 1), (0, 0, 1))
    mask.putpixel((3, 1), (128, 0, 0))
    mask.save(masks / f"{stem}.png")


def _write_duts_sample(root, split, stem, image_size=(9, 7), mask_size=None):
    label = "DUTS-TR" if split == "train" else "DUTS-TE"
    base = root / "DUTS" / label
    images = base / f"{label}-Image"
    masks = base / f"{label}-Mask"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, color=(20, 40, 60)).save(images / f"{stem}.jpg")
    mask = Image.new("L", mask_size or image_size, color=0)
    mask.putpixel((1, 1), 255)
    mask.save(masks / f"{stem}.png")


def test_random_scale_crop_flip_shapes_and_values():
    torch.manual_seed(0)
    image = torch.randint(0, 256, (3, 200, 300), dtype=torch.uint8)
    mask = torch.zeros(1, 200, 300, dtype=torch.long)
    mask[:, 50:100, 50:100] = 1
    for _ in range(20):
        out_image, out_mask = _random_scale_crop_flip(image, mask, 160)
        assert out_image.shape == (3, 160, 160)
        assert out_mask.shape == (160, 160)
        assert set(out_mask.unique().tolist()) <= {0, 1, IGNORE_INDEX}


def test_eval_collate_keeps_native_masks():
    batch = [
        (torch.zeros(3, 8, 8), torch.zeros(5, 7, dtype=torch.long)),
        (torch.zeros(3, 8, 8), torch.zeros(9, 4, dtype=torch.long)),
    ]
    images, masks = eval_collate(batch)
    assert images.shape == (2, 3, 8, 8)
    assert [tuple(m.shape) for m in masks] == [(5, 7), (9, 4)]


def test_kvasir_spec_is_binary_polyp():
    spec = SPECS["kvasir"]
    assert spec.class_names == ("background", "polyp")
    assert spec.prompts == ("polyp",)
    assert spec.prompt_classes == (1,)
    assert (spec.train_split, spec.eval_split) == ("train", "test")


def test_kvasir_hash_split_is_exact_stable_and_disjoint(tmp_path):
    stems = [f"case_{index:02d}" for index in range(10)]
    # Reverse creation order to ensure filesystem iteration does not define the split.
    for stem in reversed(stems):
        _write_kvasir_sample(tmp_path, stem)

    train = KvasirSegmentation(tmp_path, "train")
    test = KvasirSegmentation(tmp_path, "test")
    train_again = KvasirSegmentation(tmp_path, "train")
    train_ids = [train.sample_id(i) for i in range(len(train))]
    test_ids = [test.sample_id(i) for i in range(len(test))]

    ranked = sorted(
        stems,
        key=lambda stem: (hashlib.sha256(stem.encode("utf-8")).digest(), stem),
    )
    expected_train = set(ranked[:8])
    assert len(train_ids) == 8 and len(test_ids) == 2
    assert set(train_ids) == expected_train
    assert set(test_ids) == set(stems) - expected_train
    assert set(train_ids).isdisjoint(test_ids)
    assert train_ids == [train_again.sample_id(i) for i in range(len(train_again))]
    assert train_ids == sorted(train_ids) and test_ids == sorted(test_ids)


def test_kvasir_thresholds_jpeg_mask_and_keeps_stable_sample_id(tmp_path):
    for index in range(5):
        _write_kvasir_sample(tmp_path, f"case_{index}", image_size=(9, 7))

    base = KvasirSegmentation(tmp_path, "test")
    image, mask, prompt = base[0]
    assert image.mode == "RGB" and image.size == (9, 7)
    assert mask.shape == (1, 7, 9) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) == {0, 1}
    assert mask[0, 1, 1] == 1
    assert mask[0, 1, 2] == 0  # JPEG residuals near zero remain background.
    assert mask[0, 1, 3] == 1  # The fixed inclusive threshold is 128.
    assert prompt == 0
    assert base.sample_id(0).startswith("case_")

    dataset = SegDataset(
        SPECS["kvasir"], tmp_path, train=False, image_size=16
    )
    resized_image, native_mask = dataset[0]
    assert resized_image.shape == (3, 16, 16)
    assert native_mask.shape == (7, 9)
    assert dataset.sample_id(0) == dataset.base.sample_id(0)


def test_eval_dataset_reads_image_and_mask_through_content_verifier(tmp_path):
    for index in range(5):
        _write_kvasir_sample(tmp_path, f"case_{index}")
    calls = []

    def verified_reader(sample_id, role, path):
        calls.append((sample_id, role, path.relative_to(tmp_path).as_posix()))
        return path.read_bytes()

    dataset = SegDataset(
        SPECS["kvasir"],
        tmp_path,
        train=False,
        image_size=16,
        verified_file_reader=verified_reader,
    )
    dataset[0]
    assert [(sample_id, role) for sample_id, role, _ in calls] == [
        (dataset.sample_id(0), "image"),
        (dataset.sample_id(0), "mask"),
    ]

    invalid = SegDataset(
        SPECS["kvasir"],
        tmp_path,
        train=False,
        image_size=16,
        verified_file_reader=lambda *_: bytearray(b"not immutable"),
    )
    with pytest.raises(TypeError, match="immutable bytes"):
        invalid[0]


def test_kvasir_rejects_unpaired_files(tmp_path):
    images = tmp_path / "Kvasir-SEG" / "images"
    masks = tmp_path / "Kvasir-SEG" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(images / "image_only.jpg")
    with pytest.raises(ValueError, match="not paired"):
        KvasirSegmentation(tmp_path, "train")


def test_kvasir_rejects_size_mismatch(tmp_path):
    _write_kvasir_sample(
        tmp_path, "bad_size", image_size=(8, 6), mask_size=(7, 6)
    )
    with pytest.raises(ValueError, match="size mismatch"):
        KvasirSegmentation(tmp_path, "test")


def test_fives_spec_is_binary_retinal_vessels():
    spec = SPECS["fives"]
    assert spec.class_names == ("background", "retinal blood vessels")
    assert spec.prompts == ("retinal blood vessels",)
    assert spec.prompt_classes == (1,)
    assert (spec.train_split, spec.eval_split) == ("train", "test")


def test_fives_uses_official_splits_and_stable_ids(tmp_path):
    for stem in ("2_G", "1_A"):
        _write_fives_sample(tmp_path, "train", stem)
    _write_fives_sample(tmp_path, "test", "3_D")

    train = FivesSegmentation(tmp_path, "train")
    test = FivesSegmentation(tmp_path, "test")
    assert [train.sample_id(i) for i in range(len(train))] == ["1_A", "2_G"]
    assert [test.sample_id(i) for i in range(len(test))] == ["3_D"]


def test_fives_loads_nonzero_mask_at_native_resolution(tmp_path):
    _write_fives_sample(tmp_path, "test", "3_D", image_size=(9, 7))
    base = FivesSegmentation(tmp_path, "test")
    image, mask, prompt = base[0]
    assert image.mode == "RGB" and image.size == (9, 7)
    assert mask.shape == (1, 7, 9) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) == {0, 1}
    assert mask[0, 1, 1] == 1 and prompt == 0

    dataset = SegDataset(SPECS["fives"], tmp_path, train=False, image_size=16)
    resized_image, native_mask = dataset[0]
    assert resized_image.shape == (3, 16, 16)
    assert native_mask.shape == (7, 9)
    assert dataset.sample_id(0) == "3_D"


def test_fives_rejects_unpaired_files_and_size_mismatch(tmp_path):
    images = tmp_path / "FIVES" / "train" / "Original"
    masks = tmp_path / "FIVES" / "train" / "Ground truth"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(images / "image_only.png")
    with pytest.raises(ValueError, match="not paired"):
        FivesSegmentation(tmp_path, "train")

    _write_fives_sample(
        tmp_path, "test", "bad_size", image_size=(8, 6), mask_size=(7, 6)
    )
    with pytest.raises(ValueError, match="size mismatch"):
        FivesSegmentation(tmp_path, "test")


@pytest.mark.parametrize(
    ("name", "class_name", "prompt"),
    [
        ("isic", "skin lesion", "skin lesion"),
        ("tn3k", "thyroid nodule", "thyroid nodule"),
    ],
)
def test_added_medical_dataset_specs_are_native_binary(name, class_name, prompt):
    spec = SPECS[name]
    assert spec.class_names == ("background", class_name)
    assert spec.prompts == (prompt,)
    assert spec.prompt_classes == (1,)
    assert (spec.train_split, spec.eval_split) == ("train", "test")


def test_isic_pairs_segmentation_suffix_and_preserves_official_split(tmp_path):
    _write_isic_sample(tmp_path, "train", "ISIC_0000002")
    _write_isic_sample(tmp_path, "train", "ISIC_0000001")
    _write_isic_sample(tmp_path, "test", "ISIC_0012169")

    train = ISICSegmentation(tmp_path, "train")
    test = ISICSegmentation(tmp_path, "test")
    assert [train.sample_id(i) for i in range(len(train))] == [
        "ISIC_0000001",
        "ISIC_0000002",
    ]
    assert test.sample_id(0) == "ISIC_0012169"
    image, mask, prompt = test[0]
    assert image.mode == "RGB" and image.size == (9, 7)
    assert mask.shape == (1, 7, 9) and mask.dtype == torch.long
    assert mask[0, 1, 1] == 1
    assert mask[0, 1, 2] == 0  # Low residuals are background.
    assert mask[0, 1, 3] == 1  # The threshold is inclusive.
    assert prompt == 0

    dataset = SegDataset(SPECS["isic"], tmp_path, train=False, image_size=16)
    resized_image, native_mask = dataset[0]
    assert resized_image.shape == (3, 16, 16)
    assert native_mask.shape == (7, 9)


def test_tn3k_uses_official_trainval_test_directories_and_threshold(tmp_path):
    _write_tn3k_sample(tmp_path, "train", "0002")
    _write_tn3k_sample(tmp_path, "train", "0001")
    _write_tn3k_sample(tmp_path, "test", "0000")

    train = TN3KSegmentation(tmp_path, "train")
    test = TN3KSegmentation(tmp_path, "test")
    assert [train.sample_id(i) for i in range(len(train))] == ["0001", "0002"]
    assert test.sample_id(0) == "0000"
    image, mask, prompt = test[0]
    assert image.mode == "RGB" and image.size == (9, 7)
    assert mask.shape == (1, 7, 9) and mask.dtype == torch.long
    assert mask[0, 1, 1] == 1  # Any channel can carry foreground.
    assert mask[0, 1, 2] == 0
    assert mask[0, 1, 3] == 1
    assert prompt == 0

    dataset = SegDataset(SPECS["tn3k"], tmp_path, train=False, image_size=16)
    resized_image, native_mask = dataset[0]
    assert resized_image.shape == (3, 16, 16)
    assert native_mask.shape == (7, 9)
    assert dataset.sample_id(0) == "0000"


def test_tn3k_accepts_downloaded_extraction_layout(tmp_path):
    _write_tn3k_sample(tmp_path, "test", "0000", extracted=True)
    dataset = TN3KSegmentation(tmp_path, "test")
    assert len(dataset) == 1 and dataset.sample_id(0) == "0000"


def test_duts_uses_official_splits_and_native_binary_masks(tmp_path):
    _write_duts_sample(tmp_path, "train", "train_b")
    _write_duts_sample(tmp_path, "train", "train_a")
    _write_duts_sample(tmp_path, "test", "test_a")

    train = DutsSegmentation(tmp_path, "train")
    test = DutsSegmentation(tmp_path, "test")
    assert [train.sample_id(index) for index in range(len(train))] == [
        "train_a",
        "train_b",
    ]
    assert test.sample_id(0) == "test_a"
    image, mask, prompt = test[0]
    assert image.mode == "RGB" and image.size == (9, 7)
    assert mask.shape == (1, 7, 9) and set(mask.unique().tolist()) == {0, 1}
    assert mask[0, 1, 1] == 1 and prompt == 0

    dataset = SegDataset(SPECS["duts"], tmp_path, train=False, image_size=16)
    resized_image, native_mask = dataset[0]
    assert resized_image.shape == (3, 16, 16)
    assert native_mask.shape == (7, 9)


def test_duts_rejects_unpaired_files_and_size_mismatch(tmp_path):
    _write_duts_sample(tmp_path, "train", "unpaired")
    next(tmp_path.rglob("unpaired.png")).unlink()
    with pytest.raises(ValueError, match="not paired"):
        DutsSegmentation(tmp_path, "train")

    _write_duts_sample(
        tmp_path, "test", "bad_size", image_size=(8, 6), mask_size=(7, 6)
    )
    with pytest.raises(ValueError, match="DUTS.*size mismatch"):
        DutsSegmentation(tmp_path, "test")


@pytest.mark.parametrize(
    ("writer", "dataset", "name"),
    [
        (_write_isic_sample, ISICSegmentation, "ISIC"),
        (_write_tn3k_sample, TN3KSegmentation, "TN3K"),
    ],
)
def test_added_medical_datasets_reject_size_mismatch(
    tmp_path, writer, dataset, name
):
    writer(tmp_path, "test", "bad", image_size=(8, 6), mask_size=(7, 6))
    with pytest.raises(ValueError, match=f"{name}.*size mismatch"):
        dataset(tmp_path, "test")


@pytest.mark.parametrize(
    ("writer", "dataset"),
    [
        (_write_isic_sample, ISICSegmentation),
        (_write_tn3k_sample, TN3KSegmentation),
    ],
)
def test_added_medical_datasets_reject_unpaired_and_duplicate_stems(
    tmp_path, writer, dataset
):
    writer(tmp_path, "test", "unpaired")
    next(tmp_path.rglob("unpaired*.png")).unlink()
    with pytest.raises(ValueError, match="not paired"):
        dataset(tmp_path, "test")

    writer(tmp_path, "train", "duplicate")
    image_path = next(tmp_path.rglob("duplicate.jpg"))
    Image.new("RGB", (9, 7), color=(20, 40, 60)).save(
        image_path.with_suffix(".png")
    )
    with pytest.raises(ValueError, match="duplicate image stem"):
        dataset(tmp_path, "train")


def test_isic_rejects_duplicate_stems_after_mask_suffix_removal(tmp_path):
    _write_isic_sample(tmp_path, "test", "ISIC_0000001")
    masks = (
        tmp_path / "ISIC2018" / "ISIC2018_Task1_Test_GroundTruth"
    )
    Image.new("L", (9, 7), color=0).save(
        masks / "ISIC_0000001_segmentation.jpg"
    )
    with pytest.raises(ValueError, match="duplicate mask stem"):
        ISICSegmentation(tmp_path, "test")


@pytest.mark.parametrize("dataset", [ISICSegmentation, TN3KSegmentation])
def test_added_medical_datasets_reject_nonofficial_split(tmp_path, dataset):
    with pytest.raises(ValueError, match="split must be one of"):
        dataset(tmp_path, "validation")


@requires_isic
def test_isic_actual_release_counts_pairing_and_sample():
    train = ISICSegmentation(DATA_ROOT, "train")
    test = ISICSegmentation(DATA_ROOT, "test")
    assert (len(train), len(test)) == (2594, 1000)
    train_ids = [train.sample_id(index) for index in range(len(train))]
    test_ids = [test.sample_id(index) for index in range(len(test))]
    assert train_ids == sorted(set(train_ids))
    assert test_ids == sorted(set(test_ids))
    assert set(train_ids).isdisjoint(test_ids)
    image, mask, prompt = test[0]
    assert image.size == (mask.shape[-1], mask.shape[-2])
    assert image.mode == "RGB" and mask.dtype == torch.long and prompt == 0
    assert set(mask.unique().tolist()) <= {0, 1}


@requires_tn3k
def test_tn3k_actual_release_counts_pairing_and_sample():
    train = TN3KSegmentation(DATA_ROOT, "train")
    test = TN3KSegmentation(DATA_ROOT, "test")
    assert (len(train), len(test)) == (2879, 614)
    train_ids = [train.sample_id(index) for index in range(len(train))]
    test_ids = [test.sample_id(index) for index in range(len(test))]
    assert train_ids == sorted(set(train_ids))
    assert test_ids == sorted(set(test_ids))
    # TN3K numbers each official split independently, so stems may overlap.
    image, mask, prompt = test[0]
    assert image.size == (mask.shape[-1], mask.shape[-2])
    assert image.mode == "RGB" and mask.dtype == torch.long and prompt == 0
    assert set(mask.unique().tolist()) <= {0, 1}


@requires_duts
def test_duts_actual_release_counts_pairing_and_sample():
    train = DutsSegmentation(DATA_ROOT, "train")
    test = DutsSegmentation(DATA_ROOT, "test")
    assert (len(train), len(test)) == (10553, 5019)
    train_ids = [train.sample_id(index) for index in range(len(train))]
    test_ids = [test.sample_id(index) for index in range(len(test))]
    assert train_ids == sorted(set(train_ids))
    assert test_ids == sorted(set(test_ids))
    assert set(train_ids).isdisjoint(test_ids)
    image, mask, prompt = test[0]
    assert image.size == (mask.shape[-1], mask.shape[-2])
    assert image.mode == "RGB" and mask.dtype == torch.long and prompt == 0
    assert set(mask.unique().tolist()) <= {0, 1}


@requires_pet
def test_pet_split_sizes():
    assert len(PetSegmentation(DATA_ROOT, "trainval")) == 3680
    assert len(PetSegmentation(DATA_ROOT, "test")) == 3669


@requires_pet
def test_pet_species_column_matches_naming_convention():
    # In Oxford-IIIT Pet, cat breeds are capitalized and dog breeds are
    # lowercase, which cross-checks our parse of the species column.
    base = PetSegmentation(DATA_ROOT, "trainval")
    for image_id, species in base.samples:
        assert species == (1 if image_id[0].isupper() else 2), image_id
    assert PetSegmentation.SPECIES_TO_PROMPT == {1: 0, 2: 1}


@requires_pet
def test_pet_train_sample():
    dataset = SegDataset(SPECS["pet"], DATA_ROOT, train=True, image_size=352)
    image, mask, prompt = dataset[0]
    assert image.shape == (3, 352, 352) and image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert mask.shape == (352, 352) and mask.dtype == torch.long
    # Padding introduced by spatial augmentation is excluded from training loss.
    assert set(mask.unique().tolist()) <= {0, 1, IGNORE_INDEX}
    assert prompt in (0, 1)


@requires_pet
def test_pet_eval_sample_keeps_native_mask():
    dataset = SegDataset(SPECS["pet"], DATA_ROOT, train=False, image_size=352)
    image, mask = dataset[0]
    assert image.shape == (3, 352, 352)
    assert mask.dim() == 2
    assert set(mask.unique().tolist()) <= {0, 1}


@requires_pet
def test_pet_uses_the_predeclared_border_inclusive_binary_target():
    base = PetSegmentation(DATA_ROOT, "test")
    _, mask, _ = base[0]
    assert IGNORE_INDEX not in mask.unique().tolist()
    assert set(mask.unique().tolist()) == {0, 1}
