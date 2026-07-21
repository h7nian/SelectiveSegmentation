import json
import os

import pytest
from PIL import Image

from selectseg import scientific_inputs as sci


def _png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGB" if isinstance(value, tuple) else "L"
    Image.new(mode, (4, 3), value).save(path)


def _paired(directory, stem):
    _png(directory / "images" / f"{stem}.png", (12, 34, 56))
    _png(directory / "masks" / f"{stem}.png", 255)


def _make_eval_data(root):
    data = root / "data"

    pet = data / "oxford-iiit-pet"
    (pet / "annotations").mkdir(parents=True)
    (pet / "annotations" / "test.txt").write_text(
        "pet_one 1 1 1\npet_two 1 2 1\n", encoding="utf-8"
    )
    _png(pet / "images" / "pet_one.jpg", (12, 34, 56))
    _png(pet / "annotations" / "trimaps" / "pet_one.png", 1)
    _png(pet / "images" / "pet_two.jpg", (65, 43, 21))
    _png(pet / "annotations" / "trimaps" / "pet_two.png", 1)

    kvasir = data / "Kvasir-SEG"
    for index in range(5):
        _paired(kvasir, f"case_{index}")

    fives = data / "FIVES" / "test"
    _png(fives / "Original" / "fives_one.png", (12, 34, 56))
    _png(fives / "Ground truth" / "fives_one.png", 255)

    isic = data / "ISIC2018"
    _png(
        isic / "ISIC2018_Task1-2_Test_Input" / "ISIC_1.jpg",
        (12, 34, 56),
    )
    _png(
        isic
        / "ISIC2018_Task1_Test_GroundTruth"
        / "ISIC_1_segmentation.png",
        255,
    )

    tn3k = data / "TN3K" / "tn3k"
    _png(tn3k / "test-image" / "tn_one.jpg", (12, 34, 56))
    _png(tn3k / "test-mask" / "tn_one.jpg", 255)
    return data


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_campaign(tmp_path, monkeypatch, *, pet_subset=False):
    _make_eval_data(tmp_path)
    dataset_results = {}
    for dataset in sci.EVAL_DATASETS:
        dataset_results[dataset] = sci.build_dataset_component(
            dataset,
            data_root="data",
            output_path=f"locks/datasets/{dataset}.json",
            repo_root=tmp_path,
        )

    source_path = tmp_path / "src" / "worker.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    source = sci.build_source_component(
        ["src/worker.py"],
        output_path="locks/source.json",
        repo_root=tmp_path,
    )

    blob = tmp_path / "cache" / "models" / "blobs" / "clip-bytes"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"locked clipseg bytes")
    snapshot = tmp_path / "cache" / "models" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to("../../blobs/clip-bytes")
    base_models = sci.build_base_model_component(
        [{"model": "clipseg", "path": "cache/models/snapshots/revision/model.bin"}],
        output_path="locks/base_models.json",
        repo_root=tmp_path,
    )

    checkpoint = tmp_path / "checkpoints" / "pet-clipseg.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"target checkpoint")

    counts = {
        dataset: result["manifest"]["sample_count"]
        for dataset, result in dataset_results.items()
    }
    conditions = [
        {
            "dataset": dataset,
            "model": "clipseg",
            "condition": "clipseg-general",
            "checkpoint": None,
            "batch_size": 1,
            "expected_num_samples": counts[dataset],
        }
        for dataset in sci.EVAL_DATASETS
    ]
    if pet_subset:
        conditions[0].update(
            expected_num_samples=1,
            expected_dataset_samples=counts["pet"],
            freeze_limit=1,
        )
    conditions.append(
        {
            "dataset": "pet",
            "model": "clipseg",
            "condition": "clipseg-target",
            "checkpoint": "checkpoints/pet-clipseg.pt",
            "batch_size": 1,
            "expected_num_samples": counts["pet"],
        }
    )
    if pet_subset:
        conditions[-1].update(
            expected_num_samples=1,
            expected_dataset_samples=counts["pet"],
            freeze_limit=1,
        )
    config = {
        "config_schema_version": 2,
        "campaign_id": "unit-campaign",
        "execution_policy": "locked-submit",
        "protocol": {"m_values": [2, 8, 32], "quadrature_rule": "midpoint-v1"},
        "gpu_partition_candidates": ["saffo-a100", "apollo_agate"],
        "cpu_partition_candidates": ["saffo-2tb"],
        "paths": {"output": "outputs/unit"},
        "estimator_spec": "configs/estimators/midpoint-v1.json",
        "conditions": conditions,
    }
    _write_json(tmp_path / "campaign.json", config)
    checkpoints = sci.build_checkpoint_component(
        "campaign.json",
        output_path="locks/checkpoints.json",
        repo_root=tmp_path,
    )
    environment_values = {"python": "unit", "unit-package": "1.0"}
    environment = sci.build_environment_component(
        output_path="locks/environment.json",
        repo_root=tmp_path,
        packages=["unit-package"],
        environment=environment_values,
    )
    monkeypatch.setattr(
        sci,
        "collect_environment",
        lambda packages=sci.DEFAULT_ENVIRONMENT_PACKAGES: environment_values.copy(),
    )

    root = sci.build_root_lock(
        "campaign.json",
        dataset_components={
            dataset: result["path"] for dataset, result in dataset_results.items()
        },
        source_component=source["path"],
        base_model_component=base_models["path"],
        checkpoint_component=checkpoints["path"],
        environment_component=environment["path"],
        output_path="locks/scientific-inputs.lock.json",
        repo_root=tmp_path,
    )
    return {
        "root": root,
        "datasets": dataset_results,
        "source": source,
        "base_models": base_models,
        "checkpoints": checkpoints,
        "environment": environment,
        "config": config,
    }


def test_science_projection_has_no_lock_cycle_and_binds_science():
    config = {
        "campaign_id": "campaign",
        "conditions": [{"dataset": "pet"}],
        "protocol": {"m": 32},
        "execution_policy": "preview",
        "paths": {"output": "one"},
        "scientific_input_lock": {"path": "lock.json", "sha256": "0" * 64},
    }
    original = sci.science_projection_sha256(config)
    changed_operations = dict(config)
    changed_operations["execution_policy"] = "submit"
    changed_operations["paths"] = {"output": "two"}
    changed_operations["scientific_input_lock"] = {
        "path": "new-lock.json",
        "sha256": "f" * 64,
    }
    assert sci.science_projection_sha256(changed_operations) == original

    changed_science = json.loads(json.dumps(config))
    changed_science["protocol"]["m"] = 64
    assert sci.science_projection_sha256(changed_science) != original


@pytest.mark.parametrize("value", ["/absolute", "../escape", "a/../b", "a\\b", "."])
def test_portable_path_rejects_unsafe_or_nonportable_values(value):
    with pytest.raises(ValueError):
        sci.portable_path(value)


def test_strict_json_rejects_duplicate_nonfinite_and_symlink(tmp_path):
    (tmp_path / "duplicate.json").write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        sci.load_strict_json("duplicate.json", repo_root=tmp_path)

    (tmp_path / "nan.json").write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        sci.load_strict_json("nan.json", repo_root=tmp_path)

    (tmp_path / "valid.json").write_text("{}", encoding="utf-8")
    (tmp_path / "link.json").symlink_to("valid.json")
    with pytest.raises(ValueError, match="symlink"):
        sci.load_strict_json("link.json", repo_root=tmp_path)


def test_all_dataset_components_bind_real_eval_order_and_pet_selection(tmp_path):
    _make_eval_data(tmp_path)
    observed = {}
    for dataset in sci.EVAL_DATASETS:
        result = sci.build_dataset_component(
            dataset,
            data_root="data",
            output_path=f"locks/{dataset}.json",
            repo_root=tmp_path,
        )
        manifest = result["manifest"]
        observed[dataset] = manifest["sample_count"]
        assert [row["index"] for row in manifest["samples"]] == list(
            range(manifest["sample_count"])
        )
        assert all(row["image"]["sha256"] for row in manifest["samples"])
        assert all(row["mask"]["sha256"] for row in manifest["samples"])
        assert manifest["loader_class"] == sci.DATASET_PROTOCOL[dataset][1]
        if dataset == "pet":
            assert len(manifest["selection_files"]) == 1
            assert manifest["samples"][0]["prompt_index"] == 0
        else:
            assert manifest["selection_files"] == []
    assert observed == {"pet": 2, "kvasir": 1, "fives": 1, "isic": 1, "tn3k": 1}


def test_relative_clipseg_snapshot_symlink_binds_target_and_resolved_bytes(tmp_path):
    blob = tmp_path / "cache" / "models" / "blobs" / "blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"model")
    snapshot = tmp_path / "cache" / "models" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    link = snapshot / "model.bin"
    link.symlink_to("../../blobs/blob")

    result = sci.build_base_model_component(
        [{"model": "clipseg", "path": "cache/models/snapshots/revision/model.bin"}],
        output_path="base.json",
        repo_root=tmp_path,
    )
    record = result["manifest"]["entries"][0]["file"]
    assert record["kind"] == "relative_symlink"
    assert record["symlink_target"] == "../../blobs/blob"
    assert record["resolved_path"] == "cache/models/blobs/blob"
    assert record["sha256"] == sci.sha256_file(blob, repo_root=tmp_path)

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"outside")
    escaping = tmp_path / "escape.bin"
    escaping.symlink_to(outside)
    with pytest.raises(ValueError, match="relative symlink target|repository root"):
        sci.build_base_model_component(
            [{"model": "clipseg", "path": "escape.bin"}],
            output_path="escape.json",
            repo_root=tmp_path,
        )


def test_atomic_components_are_deterministic_and_never_overwrite(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    first = sci.build_source_component(
        ["source.py"], output_path="one.json", repo_root=tmp_path
    )
    second = sci.build_source_component(
        ["source.py"], output_path="two.json", repo_root=tmp_path
    )
    assert first["sha256"] == second["sha256"]
    assert first["path"].read_bytes() == second["path"].read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sci.build_source_component(
            ["source.py"], output_path="one.json", repo_root=tmp_path
        )


def _check_count_condition(condition, *, dataset_count=5):
    key = ("pet", "clipseg", "clipseg-general")
    projected = {"conditions": [condition]}
    datasets = {
        dataset: {"sample_count": dataset_count if dataset == "pet" else 1}
        for dataset in sci.EVAL_DATASETS
    }
    base_models = {"entries": [{"model": "clipseg"}]}
    checkpoints = {
        "entries": [
            {
                "dataset": key[0],
                "model": key[1],
                "condition": key[2],
                "checkpoint": None,
            }
        ]
    }
    sci._check_component_consistency(
        projected, datasets, base_models, checkpoints
    )


def test_component_consistency_accepts_full_cohort_and_explicit_subset_counts():
    base = {
        "dataset": "pet",
        "model": "clipseg",
        "condition": "clipseg-general",
        "checkpoint": None,
    }
    _check_count_condition({**base, "expected_num_samples": 5})
    _check_count_condition(
        {
            **base,
            "expected_num_samples": 2,
            "expected_dataset_samples": 5,
            "freeze_limit": 2,
        }
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (
            {"expected_num_samples": 2, "freeze_limit": 2},
            "sample count differs",
        ),
        (
            {"expected_num_samples": 2, "expected_dataset_samples": 4, "freeze_limit": 2},
            "sample count differs",
        ),
        (
            {"expected_num_samples": 2, "expected_dataset_samples": 5},
            "needs freeze_limit",
        ),
        (
            {"expected_num_samples": 2, "expected_dataset_samples": 5, "freeze_limit": 3},
            "inconsistent freeze_limit",
        ),
        (
            {"expected_num_samples": 6, "expected_dataset_samples": 5, "freeze_limit": 6},
            "inconsistent freeze_limit",
        ),
        (
            {"expected_num_samples": 1, "expected_dataset_samples": 5, "freeze_limit": True},
            "inconsistent freeze_limit",
        ),
        (
            {"expected_num_samples": True, "expected_dataset_samples": 5, "freeze_limit": 1},
            "invalid expected_num_samples",
        ),
        (
            {"expected_num_samples": 1, "expected_dataset_samples": True, "freeze_limit": 1},
            "invalid expected_dataset_samples",
        ),
    ],
)
def test_component_consistency_rejects_unbound_or_inconsistent_subset_counts(
    counts, message
):
    condition = {
        "dataset": "pet",
        "model": "clipseg",
        "condition": "clipseg-general",
        "checkpoint": None,
        **counts,
    }
    with pytest.raises(ValueError, match=message):
        _check_count_condition(condition)


def test_root_full_fast_and_condition_verification(tmp_path, monkeypatch):
    campaign = _build_campaign(tmp_path, monkeypatch)
    root = campaign["root"]
    root_manifest_text = json.dumps(root["manifest"])
    assert "samples" not in root_manifest_text
    assert len(root["manifest"]["components"]["datasets"]) == 5

    full = sci.verify_root_lock(
        root["path"],
        repo_root=tmp_path,
        expected_sha256=root["sha256"],
        mode="full",
    )
    assert full["dataset_sample_counts"] == {
        "pet": 2,
        "kvasir": 1,
        "fives": 1,
        "isic": 1,
        "tn3k": 1,
    }

    condition = sci.verify_condition_inputs(
        root["path"],
        dataset="pet",
        model="clipseg",
        condition="clipseg-target",
        repo_root=tmp_path,
        expected_sha256=root["sha256"],
        mode="consume",
    )
    assert condition["verification_mode"] == "consume"
    assert len(condition["scientific_input_hashes"]) == 7
    assert condition["scientific_input_sha256"] == sci.logical_sha256(
        condition["scientific_input_hashes"]
    )
    locked_dataset = condition["eval_dataset"]
    locked = locked_dataset.samples_by_id["pet_one"]
    verified_sample = sci.verify_sample_bytes(
        locked_dataset,
        "pet_one",
        tmp_path / locked.image_path,
        tmp_path / locked.mask_path,
    )
    assert verified_sample["image_sha256"] == locked.image_sha256
    assert verified_sample["mask_sha256"] == locked.mask_sha256
    image_bytes = sci.read_verified_sample_file(
        locked_dataset,
        "pet_one",
        "image",
        tmp_path / locked.image_path,
    )
    assert image_bytes == (tmp_path / locked.image_path).read_bytes()

    serializable = sci.verify_condition(
        root["path"],
        dataset="pet",
        model="clipseg",
        condition="clipseg-general",
        repo_root=tmp_path,
        mode="fast",
    )
    json.dumps(serializable, allow_nan=False)
    assert "eval_dataset" not in serializable


def test_consume_mode_distinguishes_full_dataset_from_locked_artifact_prefix(
    tmp_path, monkeypatch
):
    campaign = _build_campaign(tmp_path, monkeypatch, pet_subset=True)
    condition = sci.verify_condition_inputs(
        campaign["root"]["path"],
        dataset="pet",
        model="clipseg",
        condition="clipseg-general",
        repo_root=tmp_path,
        expected_sha256=campaign["root"]["sha256"],
        mode="consume",
    )
    assert condition["dataset_sample_count"] == 2
    assert condition["expected_dataset_samples"] == 2
    assert condition["sample_count"] == 1
    assert condition["expected_num_samples"] == 1
    assert condition["freeze_limit"] == 1

    verified = condition["eval_dataset"]
    assert verified.dataset_sample_count == 2
    assert verified.sample_count == 1
    assert tuple(verified.samples_by_id) == ("pet_one",)
    with pytest.raises(ValueError, match="outside locked pet"):
        sci.read_verified_sample_file(
            verified,
            "pet_two",
            "image",
            tmp_path / "data/oxford-iiit-pet/images/pet_two.jpg",
        )


def test_fast_is_metadata_guard_while_full_detects_same_stat_byte_change(
    tmp_path, monkeypatch
):
    campaign = _build_campaign(tmp_path, monkeypatch)
    root = campaign["root"]
    pet_component = campaign["datasets"]["pet"]["manifest"]
    image_record = pet_component["samples"][0]["image"]
    image = tmp_path / image_record["path"]
    original = image.read_bytes()
    original_stat = image.stat()
    changed = bytearray(original)
    changed[-1] ^= 1
    image.write_bytes(changed)
    os.utime(image, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    sci.verify_condition_inputs(
        root["path"],
        dataset="pet",
        model="clipseg",
        condition="clipseg-general",
        repo_root=tmp_path,
        mode="small",
    )
    with pytest.raises(ValueError, match="content differs"):
        sci.verify_condition_inputs(
            root["path"],
            dataset="pet",
            model="clipseg",
            condition="clipseg-general",
            repo_root=tmp_path,
            mode="full",
        )


def test_loader_order_and_science_projection_are_rechecked(tmp_path, monkeypatch):
    campaign = _build_campaign(tmp_path, monkeypatch)
    root = campaign["root"]

    config = json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8"))
    config["execution_policy"] = "different-operation"
    config["scientific_input_lock"] = {
        "path": "locks/scientific-inputs.lock.json",
        "sha256": root["sha256"],
    }
    _write_json(tmp_path / "campaign.json", config)
    sci.load_root_lock(root["path"], repo_root=tmp_path)

    config["protocol"]["m_values"] = [64]
    _write_json(tmp_path / "campaign.json", config)
    with pytest.raises(ValueError, match="science projection"):
        sci.load_root_lock(root["path"], repo_root=tmp_path)

    _write_json(tmp_path / "campaign.json", campaign["config"])
    selection = tmp_path / "data/oxford-iiit-pet/annotations/test.txt"
    selection.write_text("pet_two 1 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loader order"):
        sci.verify_condition_inputs(
            root["path"],
            dataset="pet",
            model="clipseg",
            condition="clipseg-general",
            repo_root=tmp_path,
            mode="small",
        )


def test_seed_extension_base_model_paths_are_reusable():
    entries = sci.base_model_entries_from_seed_extension_lock(
        "configs/auxiliary/binary_seed_extension-v1.lock.json"
    )
    assert len(entries) == 8
    assert sum(entry["model"] == "clipseg" for entry in entries) == 7
    assert sum(entry["model"] == "deeplabv3" for entry in entries) == 1
