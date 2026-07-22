"""Strict campaign-bound aggregation of binary artifact diagnostics."""

import json
from copy import deepcopy

import numpy as np
import pytest

import scripts.analyze.diagnostics as diagnostic_analysis
from scripts.analyze.main import EXPECTED_CONDITIONS
from scripts.analyze.diagnostics import (
    ANALYSIS_ARTIFACT_TYPE,
    DESIGNS,
    JSON_NAME,
    METRIC_KEYS,
    TEX_NAME,
    analyze,
    load_inputs,
    main,
    validate_analysis,
    write_outputs,
)
from scripts.submit.main import (
    Config,
    EXPECTED_PROTOCOL,
    build_campaign_lock,
    write_campaign_lock,
)
from selectseg.artifacts import sha256_file, write_binary_artifact
from selectseg.studies.diagnostics import parse_args, run_diagnostics


def _estimator(path):
    payload = {
        "schema_version": 1,
        "estimator_id": "midpoint-v1",
        "target_measure": "uniform-threshold",
        "rule": "midpoint",
        "randomized": False,
        "required_seed": 0,
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _artifact(root, dataset, condition, *, count=2, offset=0):
    model = condition.split("-", 1)[0]
    sample_ids = [f"{dataset}-{condition}-{index}" for index in range(count)]
    samples = []
    for index, sample_id in enumerate(sample_ids):
        probability = np.array(
            [
                [0.05 + 0.01 * offset, 0.25, 0.55],
                [0.75, 0.95 - 0.01 * offset, 0.40 + 0.02 * index],
            ],
            dtype=np.float32,
        )
        truth = np.array([[0, index % 2, 1], [1, 1, 0]], dtype=np.uint8)
        samples.append((sample_id, probability, truth))
    return write_binary_artifact(
        root,
        dataset=dataset,
        condition=condition,
        model=model,
        split="test",
        class_index=1,
        class_name="foreground",
        checkpoint=None,
        base_model={"name": model, "source": "synthetic"},
        source_sha256=(format(offset % 16, "x") * 64),
        environment={
            "packages": {
                "python": "3.12",
                "numpy": "test",
                "torch": "test",
                "torchvision": "test",
                "transformers": "test",
            },
            "device": "cpu",
            "cuda_runtime": None,
            "cuda_device": None,
            "autocast_dtype": "disabled",
        },
        preprocessing={
            "model_input": "synthetic",
            "probability_to_native_mask": "synthetic",
        },
        cohort="synthetic held-out smoke cohort",
        sample_ids=sample_ids,
        samples=samples,
        command=["pytest", "freeze"],
        created_utc="2026-07-19T00:00:00+00:00",
    )


def _diagnose(root, manifest, *, descriptors=False, ece_bins=15):
    arguments = [
        "--artifact-manifest",
        str(manifest),
        "--expected-artifact-manifest-sha256",
        sha256_file(manifest),
        "--output-root",
        str(root),
        "--decision-threshold",
        "0.5",
        "--ece-bins",
        str(ece_bins),
        "--pixel-chunk-size",
        "3",
    ]
    if descriptors:
        arguments.append("--write-descriptors")
    args = parse_args(arguments)
    args.command_arguments = arguments
    return run_diagnostics(args, created_utc="2026-07-19T01:00:00+00:00")


def _campaign(
    tmp_path,
    keys,
    *,
    count=2,
    descriptors=False,
    ece_bins=15,
    offset_base=0,
    campaign_id="binary-midpoint-main-v1",
):
    estimator = _estimator(tmp_path / "config" / "midpoint-v1.json")
    manifests = []
    conditions = []
    summaries = []
    for index, (dataset, condition) in enumerate(keys):
        manifest = _artifact(
            tmp_path / "artifacts",
            dataset,
            condition,
            count=count,
            offset=offset_base + index,
        )
        manifests.append(manifest)
        model = condition.split("-", 1)[0]
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "model": model,
                "checkpoint": None,
                "batch_size": 1,
                "expected_num_samples": count,
            }
        )
        summaries.append(
            _diagnose(
                tmp_path / "diagnostics",
                manifest,
                descriptors=descriptors,
                ece_bins=ece_bins,
            )
        )
    config = Config(
        path=tmp_path / "config" / "campaign.json",
        sha256="a" * 64,
        data={
            "campaign_id": campaign_id,
            "protocol": deepcopy(EXPECTED_PROTOCOL),
            "estimator_spec": str(estimator),
            "paths": {
                "artifact_output_root": str(tmp_path / "artifacts"),
                "common_output_root": str(tmp_path / "common"),
                "simulation_output_root": str(tmp_path / "simulations"),
                "assembly_output_root": str(tmp_path / "assembled"),
            },
            "conditions": conditions,
        },
    )
    lock = build_campaign_lock(config, manifests)
    lock_path, _ = write_campaign_lock(lock, tmp_path / "campaign.lock.json")
    return lock_path, summaries


def test_incomplete_smoke_analysis_is_lock_bound_deterministic_and_descriptive(
    tmp_path,
):
    keys = [("fives", "clipseg-general"), ("isic", "deeplabv3-target")]
    lock, summaries = _campaign(tmp_path, keys, descriptors=True)

    first = analyze(lock, list(reversed(summaries)), allow_incomplete=True)
    second = analyze(lock, summaries, allow_incomplete=True)
    assert first == second
    assert first["artifact_type"] == ANALYSIS_ARTIFACT_TYPE
    assert first["campaign"]["num_locked_conditions"] == 2
    assert first["campaign"]["num_analyzed_conditions"] == 2
    assert first["campaign"]["complete_predeclared_campaign"] is False
    assert [(row["dataset"], row["condition"]) for row in first["conditions"]] == [
        ("fives", "clipseg-general"),
        ("isic", "deeplabv3-target"),
    ]
    assert set(first["conditions"][0]["metrics"]) == set(METRIC_KEYS)
    assert (
        "do not identify a joint mask posterior"
        in first["scope"]["posterior_limitation"]
    )
    assert "never fit" in first["scope"]["label_use"]
    assert "no per-image failure case" in first["scope"]["descriptor_policy"]

    one = write_outputs(first, tmp_path / "one")
    two = write_outputs(second, tmp_path / "two")
    assert [path.name for path in one] == [JSON_NAME, TEX_NAME]
    assert one[0].read_bytes() == two[0].read_bytes()
    assert one[1].read_bytes() == two[1].read_bytes()
    table = one[1].read_text()
    assert "FIVES" in table and "ISIC 2018" in table
    assert "\\textsc{CLIP-G}" in table
    assert "\\textsc{DL-T}" in table
    assert "joint mask posterior" in table
    assert "% Generated from diagnostics_analysis.json SHA-256:" in table


def test_extension_diagnostics_use_the_same_analyzer_with_design_argument(tmp_path):
    design = DESIGNS["extension"]
    keys = [("pet", "segformer-target"), ("duts", "deeplabv3-target")]
    lock, summaries = _campaign(
        tmp_path,
        keys,
        campaign_id="architecture-domain-extension-v1",
    )

    result = analyze(
        lock,
        summaries,
        allow_incomplete=True,
        design=design,
    )
    outputs = write_outputs(result, tmp_path / "extension", design=design)

    assert outputs[1].name == "architecture_domain_diagnostics.tex"
    table = outputs[1].read_text()
    assert r"\label{tab:architecture-domain-diagnostics}" in table
    assert "Oxford Pet" in table and "DUTS" in table
    assert r"\textsc{SF-T}" in table and r"\textsc{DL-T}" in table


def test_canonical_mode_requires_exact_16_inputs_and_canonical_counts(tmp_path):
    lock, summaries = _campaign(
        tmp_path,
        [("fives", "clipseg-general")],
        count=1,
    )
    with pytest.raises(ValueError, match="exact 16-condition campaign lock"):
        analyze(lock, summaries)


def test_exact_16_condition_path_succeeds_when_canonical_counts_are_pinned(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        diagnostic_analysis,
        "EXPECTED_SAMPLE_COUNTS",
        {dataset: 1 for dataset, _ in EXPECTED_CONDITIONS},
    )
    lock, summaries = _campaign(tmp_path, EXPECTED_CONDITIONS, count=1)
    result = analyze(lock, list(reversed(summaries)))
    assert result["campaign"]["num_locked_conditions"] == 16
    assert result["campaign"]["num_analyzed_conditions"] == 16
    assert result["campaign"]["complete_predeclared_campaign"] is True
    assert len(result["conditions"]) == 16


@pytest.mark.parametrize(
    "campaign_id",
    ("binary-midpoint-main-v1", "binary-midpoint-main-v2"),
)
def test_canonical_mode_accepts_each_sealed_campaign_id(
    tmp_path, monkeypatch, campaign_id
):
    monkeypatch.setattr(
        diagnostic_analysis,
        "EXPECTED_SAMPLE_COUNTS",
        {dataset: 1 for dataset, _ in EXPECTED_CONDITIONS},
    )
    lock, summaries = _campaign(
        tmp_path,
        EXPECTED_CONDITIONS,
        count=1,
        campaign_id=campaign_id,
    )
    result = analyze(lock, summaries)
    assert result["campaign"]["campaign_id"] == campaign_id
    assert result["campaign"]["complete_predeclared_campaign"] is True


def test_canonical_mode_rejects_an_unsealed_campaign_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        diagnostic_analysis,
        "EXPECTED_SAMPLE_COUNTS",
        {dataset: 1 for dataset, _ in EXPECTED_CONDITIONS},
    )
    lock, summaries = _campaign(
        tmp_path,
        EXPECTED_CONDITIONS,
        count=1,
        campaign_id="binary-midpoint-main-v3-unsealed",
    )
    with pytest.raises(ValueError, match="sealed campaign IDs"):
        analyze(lock, summaries)


def test_explicit_inputs_reject_duplicates_unlocked_conditions_and_wrong_spec(tmp_path):
    lock, summaries = _campaign(
        tmp_path / "first",
        [("fives", "clipseg-general")],
    )
    with pytest.raises(ValueError, match="must be distinct"):
        load_inputs(lock, [summaries[0], summaries[0]], allow_incomplete=True)

    _, unrelated = _campaign(
        tmp_path / "second",
        [("isic", "clipseg-general")],
    )
    with pytest.raises(ValueError, match="absent from campaign lock"):
        load_inputs(lock, unrelated, allow_incomplete=True)

    wrong_ece_lock, wrong_ece = _campaign(
        tmp_path / "wrong-ece",
        [("fives", "clipseg-target")],
        ece_bins=7,
    )
    with pytest.raises(ValueError, match="15-bin ECE"):
        load_inputs(wrong_ece_lock, wrong_ece, allow_incomplete=True)

    changed_lock, _ = _campaign(
        tmp_path / "changed-source",
        [("fives", "clipseg-general")],
        offset_base=9,
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_inputs(changed_lock, summaries, allow_incomplete=True)


def test_descriptor_payload_and_source_artifact_tampering_fail_closed(tmp_path):
    lock, summaries = _campaign(
        tmp_path,
        [("fives", "clipseg-general")],
        descriptors=True,
    )
    descriptor = summaries[0].with_name("descriptors.jsonl")
    with descriptor.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="descriptor SHA-256 mismatch"):
        analyze(lock, summaries, allow_incomplete=True)


def test_analysis_schema_rejects_extra_fields_nonfinite_metrics_and_bad_order(tmp_path):
    lock, summaries = _campaign(
        tmp_path,
        [("fives", "clipseg-general"), ("isic", "deeplabv3-target")],
    )
    result = analyze(lock, summaries, allow_incomplete=True)

    extra = deepcopy(result)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_analysis(extra)

    nonfinite = deepcopy(result)
    nonfinite["conditions"][0]["metrics"]["ece"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_analysis(nonfinite)

    reordered = deepcopy(result)
    reordered["conditions"].reverse()
    with pytest.raises(ValueError, match="deterministically sorted"):
        validate_analysis(reordered)


def test_cli_has_no_discovery_fallback_and_writes_smoke_outputs(tmp_path):
    lock, summaries = _campaign(
        tmp_path,
        [("fives", "clipseg-general")],
    )
    with pytest.raises(SystemExit):
        main(["--campaign-lock", str(lock)])
    outputs = main(
        [
            "--campaign-lock",
            str(lock),
            "--inputs",
            str(summaries[0]),
            "--output-dir",
            str(tmp_path / "analysis"),
            "--paper-table",
            str(tmp_path / "paper" / TEX_NAME),
            "--allow-incomplete",
        ]
    )
    assert all(path.is_file() for path in outputs)
    assert len(outputs) == 3
    assert outputs[1].read_bytes() == outputs[2].read_bytes()
