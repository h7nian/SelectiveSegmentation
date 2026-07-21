"""Tests for streamed, read-only frozen-artifact diagnostics."""

import json
from pathlib import Path

import numpy as np
import pytest

from selectseg.artifacts import sha256_file, write_binary_artifact
from selectseg.studies.diagnostics import (
    DESCRIPTORS_NAME,
    LADDER_NODES,
    load_binary_diagnostics,
    parse_args,
    run_diagnostics,
)


def _arrays():
    probability_a = np.array(
        [
            [0.00, 0.01, 0.02, 0.10],
            [0.20, 0.30, 0.40, 0.49],
            [0.50, 0.60, 0.70, 0.80],
            [0.90, 0.97, 0.99, 1.00],
        ],
        dtype=np.float32,
    )
    truth_a = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    probability_b = np.full((3, 5), 0.1, dtype=np.float32)
    truth_b = np.zeros((3, 5), dtype=np.uint8)
    return (
        ("case-a", probability_a, truth_a),
        ("case-b", probability_b, truth_b),
    )


def _artifact(tmp_path):
    samples = _arrays()
    return write_binary_artifact(
        tmp_path / "frozen",
        dataset="toy",
        condition="clipseg-general",
        model="clipseg",
        split="test",
        class_index=1,
        class_name="lesion",
        checkpoint=None,
        base_model={"name": "clipseg", "source": "unit-test"},
        source_sha256="a" * 64,
        environment={
            "packages": {
                "python": "3.12",
                "numpy": "unit-test",
                "torch": "unit-test",
                "torchvision": "unit-test",
                "transformers": "unit-test",
            },
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        preprocessing={
            "model_input": "none",
            "probability_to_native_mask": "none",
        },
        cohort="two synthetic held-out masks",
        sample_ids=[sample[0] for sample in samples],
        samples=samples,
        command=["pytest", "freeze"],
        created_utc="2026-07-19T00:00:00+00:00",
    )


def _args(
    tmp_path,
    manifest_path,
    *,
    output_name="diagnostics",
    chunk_size=4,
    descriptors=True,
):
    arguments = [
        "--artifact-manifest",
        str(manifest_path),
        "--expected-artifact-manifest-sha256",
        sha256_file(manifest_path),
        "--output-root",
        str(tmp_path / output_name),
        "--decision-threshold",
        "0.5",
        "--ece-bins",
        "5",
        "--pixel-chunk-size",
        str(chunk_size),
    ]
    if descriptors:
        arguments.append("--write-descriptors")
    args = parse_args(arguments)
    args.command_arguments = arguments
    return args


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _rehash_descriptors(summary_path, payload):
    descriptor_path = summary_path.parent / DESCRIPTORS_NAME
    descriptor_path.write_bytes(payload)
    summary = json.loads(summary_path.read_text())
    summary["descriptors"]["sha256"] = sha256_file(descriptor_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _brute_ece(probability, truth, num_bins):
    probability = probability.astype(np.float64).reshape(-1)
    truth = truth.reshape(-1)
    indices = np.minimum(np.floor(probability * num_bins).astype(int), num_bins - 1)
    result = 0.0
    for index in range(num_bins):
        selected = indices == index
        if np.any(selected):
            result += np.mean(selected) * abs(
                np.mean(probability[selected]) - np.mean(truth[selected])
            )
    return result


def test_streamed_metrics_match_brute_force_and_keep_labels_separate(tmp_path):
    manifest_path = _artifact(tmp_path)
    summary_path = run_diagnostics(
        _args(tmp_path, manifest_path),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    loaded = load_binary_diagnostics(summary_path)
    summary = loaded.summary
    samples = _arrays()
    probabilities = np.concatenate([sample[1].reshape(-1) for sample in samples])
    truths = np.concatenate([sample[2].reshape(-1) for sample in samples])

    expected_brier = np.mean(
        (probabilities.astype(np.float64) - truths) ** 2
    )
    assert summary["marginal_calibration"]["brier_score"] == pytest.approx(
        expected_brier
    )
    assert summary["marginal_calibration"]["ece"] == pytest.approx(
        _brute_ece(probabilities, truths, 5)
    )
    assert summary["counts"] == {
        "num_images": 2,
        "num_pixels": probabilities.size,
    }
    assert sum(
        entry["num_pixels"]
        for entry in summary["marginal_calibration"]["bins"]
    ) == probabilities.size
    hard = probabilities >= 0.5
    assert summary["hard_prediction"][
        "pixel_weighted_foreground_fraction"
    ] == pytest.approx(np.mean(hard))
    assert summary["hard_prediction"]["num_empty_masks"] == 1
    assert summary["hard_prediction"]["empty_mask_ratio"] == 0.5
    assert summary["truth"]["num_empty_masks"] == 1

    descriptor_path = summary_path.parent / DESCRIPTORS_NAME
    rows = _read_rows(descriptor_path)
    assert len(rows) == 2
    assert sha256_file(descriptor_path) == summary["descriptors"]["sha256"]
    assert set(rows[0]["prediction_only"]).isdisjoint(
        {"truth", "brier", "dice", "label"}
    )
    assert "confidence" not in json.dumps(rows, sort_keys=True)
    assert set(rows[0]) >= {"prediction_only", "label_only", "label_outcomes"}
    assert rows[1]["prediction_only"]["hard_empty_mask"] is True
    assert rows[1]["label_only"]["truth_empty_mask"] is True
    assert rows[1]["label_outcomes"]["hard_dice_loss"] == 0.0

    for row, (_, probability, truth) in zip(rows, samples, strict=True):
        masks = [probability >= threshold for threshold in LADDER_NODES]
        distinct = 1 + sum(
            not np.array_equal(left, right)
            for left, right in zip(masks[:-1], masks[1:], strict=True)
        )
        changed = [
            np.mean(left != right)
            for left, right in zip(masks[:-1], masks[1:], strict=True)
        ]
        foreground = [np.mean(mask) for mask in masks]
        prediction = row["prediction_only"]
        assert prediction["ladder_distinct_mask_count"] == distinct
        np.testing.assert_allclose(
            prediction["ladder_adjacent_changed_pixel_fractions"], changed
        )
        np.testing.assert_allclose(
            prediction["ladder_foreground_fractions"], foreground
        )
        assert row["label_only"]["truth_foreground_fraction"] == pytest.approx(
            np.mean(truth)
        )
        assert row["label_outcomes"]["marginal_brier_score"] == pytest.approx(
            np.mean((probability.astype(np.float64) - truth) ** 2)
        )

    limitation = summary["scope"]["marginal_calibration_limitation"]
    assert "marginal" in limitation
    assert "do not identify a joint mask posterior" in limitation
    assert "must not tune" in summary["scope"]["label_use_policy"]


def test_chunk_size_changes_identity_but_not_diagnostic_values(tmp_path):
    manifest_path = _artifact(tmp_path)
    first_path = run_diagnostics(
        _args(tmp_path, manifest_path, output_name="one", chunk_size=1),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    second_path = run_diagnostics(
        _args(tmp_path, manifest_path, output_name="many", chunk_size=1000),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    first = load_binary_diagnostics(first_path).summary
    second = load_binary_diagnostics(second_path).summary
    assert first["diagnostic_id"] != second["diagnostic_id"]
    for section in (
        "marginal_calibration",
        "hard_prediction",
        "truth",
        "shared_threshold_ladder",
    ):
        if section == "marginal_calibration":
            assert first[section]["brier_score"] == pytest.approx(
                second[section]["brier_score"]
            )
            assert first[section]["ece"] == pytest.approx(second[section]["ece"])
        else:
            assert first[section] == second[section]


def test_optional_descriptors_and_atomic_no_overwrite(tmp_path):
    manifest_path = _artifact(tmp_path)
    args = _args(tmp_path, manifest_path, descriptors=False)
    summary_path = run_diagnostics(
        args,
        created_utc="2026-07-19T01:00:00+00:00",
    )
    summary = load_binary_diagnostics(summary_path).summary
    assert summary["descriptors"] == {
        "included": False,
        "path": None,
        "sha256": None,
        "num_rows": 0,
        "schema_version": 1,
        "deterministic_given_artifact_and_specification": True,
    }
    assert not (summary_path.parent / DESCRIPTORS_NAME).exists()
    before = sha256_file(summary_path)
    with pytest.raises(FileExistsError, match="already exist"):
        run_diagnostics(
            args,
            created_utc="2026-07-19T02:00:00+00:00",
        )
    assert sha256_file(summary_path) == before
    assert not list(summary_path.parent.parent.glob(".*.tmp-*"))


def test_expected_manifest_digest_is_mandatory_and_checked_before_output(tmp_path):
    manifest_path = _artifact(tmp_path)
    args = _args(tmp_path, manifest_path)
    args.expected_artifact_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="artifact manifest SHA-256 mismatch"):
        run_diagnostics(args)
    assert not (tmp_path / "diagnostics").exists()

    args.expected_artifact_manifest_sha256 = "A" * 64
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        run_diagnostics(args)


def test_loader_rejects_descriptor_tampering_and_summary_schema_extension(tmp_path):
    manifest_path = _artifact(tmp_path)
    summary_path = run_diagnostics(
        _args(tmp_path, manifest_path),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    descriptor_path = summary_path.parent / DESCRIPTORS_NAME
    with descriptor_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        load_binary_diagnostics(summary_path)

    second_path = run_diagnostics(
        _args(tmp_path, manifest_path, output_name="second", descriptors=False),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    summary = json.loads(second_path.read_text())
    summary["unreviewed_extension"] = True
    second_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_binary_diagnostics(second_path)


def test_payload_validation_failure_leaves_no_partial_diagnostics(tmp_path):
    manifest_path = _artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    payload_path = manifest_path.parent / manifest["samples"][1]["path"]
    with payload_path.open("ab") as handle:
        handle.write(b"tamper")
    args = _args(tmp_path, manifest_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_diagnostics(args)
    output_parent = (
        tmp_path
        / "diagnostics"
        / "toy"
        / "clipseg-general"
        / manifest["artifact_id"]
    )
    assert output_parent.is_dir()
    assert not list(output_parent.iterdir())


@pytest.mark.parametrize("token", ["NaN", "1e999"])
def test_loader_rejects_duplicate_keys_and_all_nonfinite_json(tmp_path, token):
    manifest_path = _artifact(tmp_path)
    duplicate_path = run_diagnostics(
        _args(tmp_path, manifest_path, output_name="duplicate", descriptors=False),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    duplicate = duplicate_path.read_text().replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    duplicate_path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_binary_diagnostics(duplicate_path)

    nonfinite_path = run_diagnostics(
        _args(
            tmp_path,
            manifest_path,
            output_name=f"nonfinite-{token}",
            descriptors=False,
        ),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    summary = json.loads(nonfinite_path.read_text())
    value = summary["marginal_calibration"]["brier_score"]
    nonfinite_path.write_text(
        nonfinite_path.read_text().replace(
            f'"brier_score": {value}', f'"brier_score": {token}', 1
        )
    )
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_binary_diagnostics(nonfinite_path)


@pytest.mark.parametrize("corruption", ["duplicate", "nonfinite"])
def test_descriptor_validation_fails_closed_after_matching_rehash(
    tmp_path, corruption
):
    manifest_path = _artifact(tmp_path)
    summary_path = run_diagnostics(
        _args(tmp_path, manifest_path, output_name=corruption),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    descriptor_path = summary_path.parent / DESCRIPTORS_NAME
    lines = descriptor_path.read_bytes().splitlines(keepends=True)
    if corruption == "duplicate":
        lines[0] = lines[0].replace(
            b'"schema_version":1,',
            b'"schema_version":1,"schema_version":1,',
            1,
        )
        match = "duplicate JSON key"
    else:
        row = json.loads(lines[0])
        row["label_outcomes"]["fixed_bin_ece"] = float("nan")
        lines[0] = (json.dumps(row, allow_nan=True) + "\n").encode()
        match = "non-finite JSON"
    _rehash_descriptors(summary_path, b"".join(lines))
    with pytest.raises(ValueError, match=match):
        load_binary_diagnostics(summary_path)


def test_malformed_manifest_is_rejected_before_diagnostic_publication(tmp_path):
    manifest_path = _artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["unreviewed_extension"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n")
    args = _args(tmp_path, manifest_path)
    args.expected_artifact_manifest_sha256 = sha256_file(manifest_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        run_diagnostics(args)
    assert not (tmp_path / "diagnostics").exists()


def test_loader_rejects_finite_out_of_range_values(tmp_path):
    manifest_path = _artifact(tmp_path)
    summary_path = run_diagnostics(
        _args(tmp_path, manifest_path, descriptors=False),
        created_utc="2026-07-19T01:00:00+00:00",
    )
    summary = json.loads(summary_path.read_text())
    summary["marginal_calibration"]["brier_score"] = 1.01
    summary_path.write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        load_binary_diagnostics(summary_path)


def test_descriptor_payload_is_deterministic_for_fixed_artifact_and_spec(tmp_path):
    manifest_path = _artifact(tmp_path)
    paths = [
        run_diagnostics(
            _args(tmp_path, manifest_path, output_name=name),
            created_utc="2026-07-19T01:00:00+00:00",
        )
        for name in ("first", "second")
    ]
    summaries = [load_binary_diagnostics(path).summary for path in paths]
    assert summaries[0]["diagnostic_id"] == summaries[1]["diagnostic_id"]
    assert summaries[0]["descriptors"]["sha256"] == summaries[1]["descriptors"]["sha256"]
    assert (paths[0].parent / DESCRIPTORS_NAME).read_bytes() == (
        paths[1].parent / DESCRIPTORS_NAME
    ).read_bytes()
